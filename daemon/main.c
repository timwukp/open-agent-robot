/* open-agent-robot macOS daemon — the robot's eye/ear/mouth on the MacBook.
 *
 * Threads (plan D4, fork design):
 *   [miniaudio capture cb]  mic 16 kHz mono s16  -> mic_ring
 *   [miniaudio playback cb] spk 24 kHz mono s16  <- spk_ring (jitter buffer)
 *   [camera thread]         ffmpeg avfoundation 640x360 -> MJPEG -> newest-frame slot
 *   [lws service loop]      drains mic_ring first (audio priority), then frame slot;
 *                           received TTS PCM -> spk_ring; events -> stdout
 *
 * Wire: [1B type][8B capture-ts µs LE][payload]; connects ws://127.0.0.1:8765/stream
 * (through the SSH tunnel — no TLS in-daemon by design).
 *
 * Build: make   (deps: brew install ffmpeg libwebsockets; miniaudio auto-fetched)
 * Run:   ./oar-daemon [--check-perms] [--no-video]
 */
#define MA_NO_RUNTIME_LINKING
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libwebsockets.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "ring.h"

#define MIC_RATE 16000
#define SPK_RATE 24000
#define CHUNK_MS 20
#define MIC_CHUNK_BYTES (MIC_RATE / 1000 * CHUNK_MS * 2)   /* 640 */
#define SPK_CHUNK_BYTES (SPK_RATE / 1000 * CHUNK_MS * 2)   /* 960 */
#define PREBUF_BYTES (SPK_RATE * 2 * 200 / 1000)           /* 200 ms prebuffer */
#define CAM_W 640
#define CAM_H 480
#define CAM_FPS 30

enum { T_MIC = 0x01, T_JPEG = 0x02, T_TTS = 0x03, T_FLUSH = 0x04,
       T_EVT = 0x10, T_SYNC = 0x11 };

static uint64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ull + tv.tv_usec;
}

/* ---------------- shared state ---------------- */
/* spk ring holds a WHOLE utterance (TTS bursts the full clip; 1 MiB ≈ 21 s
 * @24 kHz s16) so playback never depends on live network timing */
static uint8_t mic_store[1 << 17], spk_store[1 << 20];
static ring_t mic_ring, spk_ring;

static pthread_mutex_t frame_mtx = PTHREAD_MUTEX_INITIALIZER;  /* newest-frame slot */
static uint8_t *frame_buf = NULL;
static size_t frame_len = 0;
static uint64_t frame_ts = 0;
static _Atomic int frame_fresh = 0;

static _Atomic int running = 1;
static _Atomic int spk_playing = 0;      /* half-duplex hint sent upstream */
static struct lws *ws_conn = NULL;
static struct lws_context *lws_ctx = NULL;   /* for cross-thread service wake */
static int no_video = 0;
static int no_audio = 0;                 /* --no-audio: camera-only operation */
static FILE *dump_rx = NULL;             /* --dump-audio: tee received TTS PCM */
static _Atomic long underruns = 0;
static _Atomic int spk_flush = 0;        /* T_FLUSH: drop buffered playback */
static _Atomic int barge_pending = 0;    /* loud voice over playback detected */
static _Atomic int barge_sent = 0;

/* ---------------- audio ---------------- */
#define BARGE_RMS 2500          /* speak louder than the speaker echo to barge in */
#define BARGE_SAMPLES (16000 * 3 / 10)   /* 0.3 s of qualifying audio */

static void capture_cb(ma_device *dev, void *out, const void *in, ma_uint32 frames) {
    (void)dev; (void)out;
    ring_write(&mic_ring, in, (size_t)frames * 2);  /* drop-whole if full */
    /* barge-in: sustained loud voice while the robot is speaking */
    if (atomic_load(&spk_playing) && !atomic_load(&barge_sent)) {
        const int16_t *s = in;
        double acc = 0;
        for (ma_uint32 i = 0; i < frames; i++) acc += (double)s[i] * s[i];
        static long loud_run = 0;
        if (acc / frames > (double)BARGE_RMS * BARGE_RMS) {
            loud_run += frames;
            if (loud_run >= BARGE_SAMPLES) {
                atomic_store(&barge_pending, 1);
                atomic_store(&barge_sent, 1);
                loud_run = 0;
            }
        } else if (loud_run > 0) {
            loud_run -= frames;             /* decay, don't hard-reset */
        }
    } else if (!atomic_load(&spk_playing)) {
        atomic_store(&barge_sent, 0);
    }
    /* wake the lws event loop — it blocks until an event, and without this
     * the only wakes are the server's 2s pings (observed: video starved to
     * 0.3 fps because sends happened once per ping) */
    if (lws_ctx && ring_used(&mic_ring) >= MIC_CHUNK_BYTES)
        lws_cancel_service(lws_ctx);
}

/* Playback with prebuffer hysteresis: partial reads at the buffer edge would
 * splice silence into the middle of the waveform every callback (continuous
 * crackle). Instead: stay silent until 150 ms is buffered, play only full
 * callbacks, and drop back to buffering on underrun (a clean pause). */
static _Atomic int spk_filling = 1;
static int16_t spk_last = 0;                /* for click-free ramps */

#define RAMP 96                             /* 4 ms fade at 24 kHz */

static void playback_cb(ma_device *dev, void *out, const void *in, ma_uint32 frames) {
    (void)dev; (void)in;
    size_t want = (size_t)frames * 2;
    int16_t *o = out;
    if (atomic_exchange(&spk_flush, 0)) {       /* consumer owns tail: safe */
        uint8_t scratch[4096];
        while (ring_read(&spk_ring, scratch, sizeof scratch) > 0) {}
        atomic_store(&spk_filling, 1);
        atomic_store(&spk_playing, 0);
        memset(out, 0, want);
        spk_last = 0;
        return;
    }
    size_t avail = ring_used(&spk_ring);
    if (atomic_load(&spk_filling)) {
        if (avail < PREBUF_BYTES) { memset(out, 0, want); spk_last = 0; return; }
        atomic_store(&spk_filling, 0);
    }
    if (avail < want) {   /* underrun: decay-ramp to zero (no click), refill */
        memset(out, 0, want);
        for (ma_uint32 i = 0; i < RAMP && i < frames; i++)
            o[i] = (int16_t)(spk_last * (RAMP - 1 - (int)i) / RAMP);
        spk_last = 0;
        atomic_store(&spk_filling, 1);
        atomic_store(&spk_playing, 0);
        atomic_fetch_add(&underruns, 1);
        return;
    }
    ring_read(&spk_ring, out, want);
    if (spk_last == 0) {                  /* resuming from silence: ramp in */
        for (ma_uint32 i = 0; i < RAMP && i < frames; i++)
            o[i] = (int16_t)((int)o[i] * (int)i / RAMP);
    }
    spk_last = o[frames - 1];
    atomic_store(&spk_playing, 1);
}

/* ---------------- camera thread ---------------- */
static void *camera_main(void *arg) {
    (void)arg;
    avdevice_register_all();
    const AVInputFormat *ifmt = av_find_input_format("avfoundation");
    AVFormatContext *fmt = NULL;
    AVDictionary *opts = NULL;
    /* Request the device's native small mode (640x480@30 is in its supported
     * list) — cheaper capture + encode than 720p, and observed to deliver a
     * steadier rate. No pixel_format (forcing nv12 yielded black frames);
     * sws_scale converts whatever arrives. */
    av_dict_set(&opts, "framerate", "30", 0);
    av_dict_set(&opts, "video_size", "640x480", 0);
    if (avformat_open_input(&fmt, "MacBook Pro Camera:", ifmt, &opts) < 0) {
        fprintf(stderr, "[cam] cannot open camera (TCC denied? run --check-perms)\n");
        return NULL;
    }
    avformat_find_stream_info(fmt, NULL);
    int vidx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++)
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { vidx = (int)i; break; }
    const AVCodec *dec = avcodec_find_decoder(fmt->streams[vidx]->codecpar->codec_id);
    AVCodecContext *dctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(dctx, fmt->streams[vidx]->codecpar);
    avcodec_open2(dctx, dec, NULL);

    const AVCodec *enc = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    AVCodecContext *ectx = avcodec_alloc_context3(enc);
    ectx->width = CAM_W; ectx->height = CAM_H;
    ectx->pix_fmt = AV_PIX_FMT_YUVJ420P;
    ectx->time_base = (AVRational){1, CAM_FPS};
    ectx->flags |= AV_CODEC_FLAG_QSCALE;
    ectx->global_quality = FF_QP2LAMBDA * 6;   /* ~q70 */
    avcodec_open2(ectx, enc, NULL);

    struct SwsContext *sws = NULL;
    AVPacket *pkt = av_packet_alloc(), *jpkt = av_packet_alloc();
    AVFrame *raw = av_frame_alloc(), *scaled = av_frame_alloc();
    scaled->width = CAM_W; scaled->height = CAM_H; scaled->format = AV_PIX_FMT_YUVJ420P;
    av_frame_get_buffer(scaled, 32);

    int skip = 0, nread = 0;
    while (atomic_load(&running)) {
        int rr = av_read_frame(fmt, pkt);
        if (rr == AVERROR(EAGAIN)) { usleep(5000); continue; }  /* avfoundation is non-blocking */
        if (rr < 0) {
            char eb[64]; av_strerror(rr, eb, sizeof eb);
            fprintf(stderr, "[cam] av_read_frame: %s\n", eb);
            break;
        }
        if ((++nread % 60) == 1) fprintf(stderr, "[cam] read pkt %d\n", nread);
        if (pkt->stream_index != vidx) { av_packet_unref(pkt); continue; }
        (void)skip;  /* no frame halving: send every frame the device delivers */
        if (avcodec_send_packet(dctx, pkt) == 0 && avcodec_receive_frame(dctx, raw) == 0) {
            if (!sws)
                sws = sws_getContext(raw->width, raw->height, raw->format,
                                     CAM_W, CAM_H, AV_PIX_FMT_YUVJ420P,
                                     SWS_BILINEAR, NULL, NULL, NULL);
            sws_scale(sws, (const uint8_t *const *)raw->data, raw->linesize,
                      0, raw->height, scaled->data, scaled->linesize);
            scaled->pts = raw->pts;
            if (avcodec_send_frame(ectx, scaled) == 0 &&
                avcodec_receive_packet(ectx, jpkt) == 0) {
                pthread_mutex_lock(&frame_mtx);
                frame_buf = realloc(frame_buf, (size_t)jpkt->size);
                memcpy(frame_buf, jpkt->data, (size_t)jpkt->size);
                frame_len = (size_t)jpkt->size;
                frame_ts = now_us();
                pthread_mutex_unlock(&frame_mtx);
                atomic_store(&frame_fresh, 1);
                if (lws_ctx) lws_cancel_service(lws_ctx);  /* wake event loop */
                static int nf = 0;
                if ((++nf % 30) == 1)
                    fprintf(stderr, "[cam] frame %d (%zu B jpeg)\n", nf, frame_len);
                av_packet_unref(jpkt);
            }
        }
        av_packet_unref(pkt);
    }
    avformat_close_input(&fmt);
    return NULL;
}

/* ---------------- websocket client ---------------- */
static _Atomic int ws_connecting = 0;

static int ws_writable(struct lws *wsi) {
    static uint8_t out[LWS_PRE + 9 + 65536];
    uint8_t *p = out + LWS_PRE;
    if (wsi != ws_conn) return 0;   /* ignore writables on stale connections */

    /* Write in a burst until the pipe chokes (lws_send_pipe_choked guards
     * multiple lws_write per WRITEABLE callback). One-message-per-callback
     * capped throughput at the callback rate (~1-8/s observed) and starved
     * video down to 0.6 fps at the consumer. Video first when fresh, then
     * aggregated audio. */
    if (atomic_exchange(&barge_pending, 0)) {   /* tiny, jumps the queue */
        static const char bmsg[] = "{\"type\":\"barge\"}";
        p[0] = T_EVT;
        uint64_t ts = now_us(); memcpy(p + 1, &ts, 8);
        memcpy(p + 9, bmsg, sizeof bmsg - 1);
        lws_write(wsi, p, 9 + sizeof bmsg - 1, LWS_WRITE_BINARY);
        fprintf(stderr, "[audio] barge-in sent\n");
        lws_callback_on_writable(wsi);
        return 0;
    }
    for (int burst = 0; burst < 8 && !lws_send_pipe_choked(wsi); burst++) {
        if (!no_video && atomic_exchange(&frame_fresh, 0)) {
            pthread_mutex_lock(&frame_mtx);
            size_t n = frame_len > 65536 ? 65536 : frame_len;
            p[0] = T_JPEG;
            memcpy(p + 1, &frame_ts, 8);
            memcpy(p + 9, frame_buf, n);
            pthread_mutex_unlock(&frame_mtx);
            int w = lws_write(wsi, p, 9 + n, LWS_WRITE_BINARY);
            static int nv = 0;
            if ((++nv % 100) == 1)
                fprintf(stderr, "[ws] video frame %d sent (%zu B, rc=%d)\n", nv, n, w);
            continue;
        }
        size_t avail = ring_used(&mic_ring);
        if (avail >= MIC_CHUNK_BYTES) {
            size_t chunks = avail / MIC_CHUNK_BYTES;
            if (chunks > 8) chunks = 8;
            size_t n = chunks * MIC_CHUNK_BYTES;
            p[0] = T_MIC;
            uint64_t ts = now_us(); memcpy(p + 1, &ts, 8);
            ring_read(&mic_ring, p + 9, n);
            lws_write(wsi, p, 9 + n, LWS_WRITE_BINARY);
            continue;
        }
        break;  /* nothing pending */
    }
    if (ring_used(&mic_ring) >= MIC_CHUNK_BYTES || atomic_load(&frame_fresh))
        lws_callback_on_writable(wsi);
    return 0;
}

static int ws_cb(struct lws *wsi, enum lws_callback_reasons reason,
                 void *user, void *in, size_t len) {
    (void)user;
    switch (reason) {
    case LWS_CALLBACK_CLIENT_ESTABLISHED:
        fprintf(stderr, "[ws] connected\n");
        ws_conn = wsi;
        atomic_store(&ws_connecting, 0);
        lws_callback_on_writable(wsi);
        break;
    case LWS_CALLBACK_CLIENT_RECEIVE: {
        /* lws delivers arbitrary FRAGMENTS of a message, not whole messages —
         * reassemble until final fragment + zero remaining payload, or bursts
         * of small messages get shredded (observed: 33% audio loss + one-byte
         * sample misalignment = continuous crackle). */
        static uint8_t rxbuf[1 << 20];
        static size_t rxlen = 0;
        if (rxlen + len <= sizeof rxbuf) {
            memcpy(rxbuf + rxlen, in, len);
            rxlen += len;
        } else {
            fprintf(stderr, "[ws] rx overflow, dropping message\n");
            rxlen = 0;
            break;
        }
        if (!lws_is_final_fragment(wsi) || lws_remaining_packet_payload(wsi) > 0)
            break;                       /* more fragments coming */
        const uint8_t *d = rxbuf;
        size_t mlen = rxlen;
        rxlen = 0;
        if (mlen < 9) break;
        if (d[0] == T_TTS) {
            if (dump_rx) { fwrite(d + 9, 1, mlen - 9, dump_rx); fflush(dump_rx); }
            if ((mlen - 9) & 1)
                fprintf(stderr, "[audio] ODD payload %zu — sample misalignment!\n", mlen - 9);
            if (!no_audio && ring_write(&spk_ring, d + 9, mlen - 9) == 0)
                fprintf(stderr, "[audio] spk ring FULL — dropped %zu bytes\n", mlen - 9);
        } else if (d[0] == T_FLUSH) {
            atomic_store(&spk_flush, 1);
            fprintf(stderr, "[audio] playback flushed (interrupt)\n");
        } else if (d[0] == T_EVT) {
            /* hud-only refreshes are for the browser; keep the terminal for
             * substantive events */
            if (mlen > 16 && strncmp((const char *)d + 9, "{\"hud\"", 6) != 0) {
                printf("[event] %.*s\n", (int)(mlen - 9), d + 9);
                fflush(stdout);
            }
        }
        break;
    }
    case LWS_CALLBACK_CLIENT_WRITEABLE:
        return ws_writable(wsi);
    case LWS_CALLBACK_EVENT_WAIT_CANCELLED:
        /* woken by a producer thread (camera frame / mic chunk ready) */
        if (ws_conn &&
            (ring_used(&mic_ring) >= MIC_CHUNK_BYTES || atomic_load(&frame_fresh)))
            lws_callback_on_writable(ws_conn);
        break;
    case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
    case LWS_CALLBACK_CLIENT_CLOSED:
        if (wsi == ws_conn || atomic_load(&ws_connecting)) {
            fprintf(stderr, "[ws] disconnected\n");
            if (wsi == ws_conn) ws_conn = NULL;
            atomic_store(&ws_connecting, 0);
        }
        break;
    default: break;
    }
    return 0;
}

static const struct lws_protocols protocols[] = {
    { "oar", ws_cb, 0, 65536 + 16, 0, NULL, 0 },
    LWS_PROTOCOL_LIST_TERM
};

/* ---------------- main ---------------- */
int main(int argc, char **argv) {
    int check_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--check-perms")) check_only = 1;
        if (!strcmp(argv[i], "--no-video")) no_video = 1;
        if (!strcmp(argv[i], "--no-audio")) no_audio = 1;
        if (!strcmp(argv[i], "--dump-audio")) {
            dump_rx = fopen("/tmp/oar_rx.raw", "wb");
            fprintf(stderr, "[audio] dumping received PCM to /tmp/oar_rx.raw\n");
        }
    }
    ring_init(&mic_ring, mic_store, sizeof mic_store);
    ring_init(&spk_ring, spk_store, sizeof spk_store);

    /* audio devices (skipped entirely under --no-audio) */
    ma_device cap, spk;
    if (!no_audio) {
        ma_device_config cc = ma_device_config_init(ma_device_type_capture);
        cc.capture.format = ma_format_s16; cc.capture.channels = 1;
        cc.sampleRate = MIC_RATE; cc.dataCallback = capture_cb;
        if (ma_device_init(NULL, &cc, &cap) != MA_SUCCESS ||
            ma_device_start(&cap) != MA_SUCCESS) {
            fprintf(stderr, "FAIL mic: check System Settings > Privacy > Microphone for your terminal\n");
            return 1;
        }
        ma_device_config pc = ma_device_config_init(ma_device_type_playback);
        pc.playback.format = ma_format_s16; pc.playback.channels = 1;
        pc.sampleRate = SPK_RATE; pc.dataCallback = playback_cb;
        if (ma_device_init(NULL, &pc, &spk) != MA_SUCCESS || ma_device_start(&spk) != MA_SUCCESS) {
            fprintf(stderr, "FAIL speaker init\n");
            return 1;
        }
        fprintf(stderr, "[audio] mic %d Hz + speaker %d Hz up\n", MIC_RATE, SPK_RATE);
    } else {
        fprintf(stderr, "[audio] DISABLED (--no-audio): no mic capture, no speaker output\n");
    }

    pthread_t cam_thread;
    if (!no_video) pthread_create(&cam_thread, NULL, camera_main, NULL);

    if (check_only) {
        sleep(2);
        int cam_ok = atomic_load(&frame_fresh);
        fprintf(stderr, "[check] mic=%zu bytes captured, camera=%s\n",
                ring_used(&mic_ring), cam_ok ? "OK" : "NO FRAME");
        atomic_store(&running, 0);
        if (!no_audio) {
            ma_device_uninit(&cap);
            ma_device_uninit(&spk);
        }
        return (cam_ok || no_video) ? 0 : 1;
    }

    /* websocket loop with reconnect */
    struct lws_context_creation_info info = {0};
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = protocols;
    struct lws_context *ctx = lws_create_context(&info);
    lws_ctx = ctx;
    /* single-connection loop: never start a second attempt while one is
     * pending (the old break-on-!ws_conn pattern spawned 2-3 parallel
     * connections during the handshake and split the audio stream) */
    while (atomic_load(&running)) {
        if (!ws_conn && !atomic_load(&ws_connecting)) {
            struct lws_client_connect_info c = {0};
            c.context = ctx; c.address = "127.0.0.1"; c.port = 8765;
            c.path = "/stream"; c.host = c.address; c.origin = c.address;
            c.protocol = "oar";
            atomic_store(&ws_connecting, 1);
            if (!lws_client_connect_via_info(&c)) {
                atomic_store(&ws_connecting, 0);
                usleep(500 * 1000);
            }
        }
        lws_service(ctx, 5);
        if (ws_conn &&
            (ring_used(&mic_ring) >= MIC_CHUNK_BYTES || atomic_load(&frame_fresh)))
            lws_callback_on_writable(ws_conn);
        static long last_ur = 0;
        long ur = atomic_load(&underruns);
        if (ur != last_ur) {
            fprintf(stderr, "[audio] playback underruns so far: %ld\n", ur);
            last_ur = ur;
        }
    }
    lws_context_destroy(ctx);
    return 0;
}
