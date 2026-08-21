"""tts_node — Kokoro-82M streaming TTS (zh + en), sentence-streamed & interruptible.

Subscribes /speech/say        String: text or JSON {"text","cache","id"}
           /speech/interrupt  Empty: abandon remaining sentences immediately
Publishes  /robot/mouth       UInt8MultiArray 24 kHz mono s16 chunks (burst per sentence)
           /speech/tts_active Bool — half-duplex gate for asr_node
           /speech/tts_started String JSON {"id","dur_s","t_us"} (dur exact for cached clips)
           /speech/tts_done   String id

Design: a worker thread consumes a queue; the rclpy executor stays free so the
interrupt callback is never blocked by a long story. Kokoro's KPipeline yields
per-sentence audio — each sentence is burst as soon as it's synthesized, so the
first sentence is audible in <1.5 s even for long replies. The game chant
(id=="chant") is exempt from interruption (the beat is sacred).
"""
from __future__ import annotations
import json
import os
import queue
import re
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, UInt8MultiArray
from std_msgs.msg import Empty as EmptyMsg

RATE = 24000
CHUNK = RATE * 60 // 1000 * 2  # 60 ms of s16 mono
HAS_CJK = re.compile(r"[一-鿿]")
# Cantonese marker characters -> route to the CosyVoice yue engine
HAS_YUE = re.compile(r"[係唔嘅喇咩乜嘢啲嗰噉咗冇睇諗俾攞邊度緊]")
GATE_TAIL_S = 0.35


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts")
        from kokoro import KPipeline
        self.get_logger().info("loading Kokoro (zh + en pipelines)…")
        self.pipes = {"z": KPipeline(lang_code="z"), "a": KPipeline(lang_code="a")}
        self.voices = {"z": "zf_xiaoxiao", "a": "af_heart"}
        self._cosyvoice = None      # lazy: Cantonese engine, ~2 GB VRAM on first use
        self.clip_cache: dict[str, np.ndarray] = {}
        self._q: queue.Queue = queue.Queue()
        self._interrupt = threading.Event()
        self.pub_pcm = self.create_publisher(UInt8MultiArray, "/robot/mouth", 200)
        self.pub_active = self.create_publisher(Bool, "/speech/tts_active", 10)
        self.pub_done = self.create_publisher(String, "/speech/tts_done", 10)
        self.pub_started = self.create_publisher(String, "/speech/tts_started", 10)
        self.create_subscription(String, "/speech/say", self.on_say, 20)
        self.create_subscription(EmptyMsg, "/speech/interrupt", self.on_interrupt, 5)
        threading.Thread(target=self._worker, daemon=True).start()
        # heal a stuck half-duplex gate left by a previous tts killed mid-speech
        self.pub_active.publish(Bool(data=False))
        self.get_logger().info("tts ready (sentence-streaming, interruptible)")

    # --- callbacks (fast, never block) -------------------------------------
    def on_say(self, msg: String):
        try:
            req = json.loads(msg.data)
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            req = {"text": msg.data}
        if req.get("text"):
            self._q.put(req)

    def on_interrupt(self, _msg):
        self._interrupt.set()
        # drain queued utterances too — the user changed the subject
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self.get_logger().info("interrupted")

    # --- worker -------------------------------------------------------------
    def _worker(self):
        while True:
            req = self._q.get()
            self._interrupt.clear()
            try:
                self.speak(req)
            except Exception as e:
                self.get_logger().error(f"speak failed: {e}")
                self.pub_active.publish(Bool(data=False))

    def _sentences(self, text: str):
        """Yield per-sentence int16 pcm from the right engine for the text."""
        if HAS_YUE.search(text):
            yield from self._sentences_yue(text)
            return
        lang = "z" if HAS_CJK.search(text) else "a"
        for _, _, audio in self.pipes[lang](text, voice=self.voices[lang]):
            wav = np.asarray(audio, dtype=np.float32)  # kokoro yields torch Tensors
            yield (np.clip(wav, -1, 1) * 32767).astype(np.int16)

    def _sentences_yue(self, text: str):
        """True Cantonese via the yue_server microservice (official CosyVoice
        in its own venv — heavy deps stay out of this one). Sentence-split
        locally so streaming/interrupt behavior matches the Kokoro path."""
        import urllib.request
        for sent in re.split(r"(?<=[。!?!?;;])", text):
            if not sent.strip():
                continue
            req = urllib.request.Request(
                "http://127.0.0.1:8901/tts",
                data=json.dumps({"text": sent}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                yield np.frombuffer(r.read(), dtype=np.int16)

    def _burst(self, pcm: np.ndarray):
        raw = pcm.tobytes()
        if os.environ.get("OAR_DUMP"):
            with open("/tmp/oar_tx.raw", "ab") as f:
                f.write(raw)
        for i in range(0, len(raw), CHUNK):
            m = UInt8MultiArray()
            m.data = raw[i:i + CHUNK]
            self.pub_pcm.publish(m)
            time.sleep(0.002)  # yield so the gateway's asyncio queue drains

    def _sleep_until(self, deadline: float, interruptible: bool) -> bool:
        """Sleep in slices; returns False if interrupted."""
        while True:
            if interruptible and self._interrupt.is_set():
                return False
            rem = deadline - time.monotonic()
            if rem <= 0:
                return True
            time.sleep(min(rem, 0.05))

    def speak(self, req: dict):
        text = req["text"]
        uid = req.get("id", text[:24])
        interruptible = uid != "chant"      # the game beat is never preempted
        t0 = time.monotonic()
        sent_s = 0.0                        # audio seconds already burst

        self.pub_active.publish(Bool(data=True))

        if req.get("cache"):
            pcm = self.clip_cache.get(text)
            if pcm is None:
                pcm = np.concatenate(list(self._sentences(text)))
                self.clip_cache[text] = pcm
            dur = len(pcm) / RATE
            self.pub_started.publish(String(data=json.dumps(
                {"id": uid, "dur_s": dur, "t_us": time.time_ns() // 1000})))
            self._burst(pcm)
            sent_s = dur
        else:
            self.pub_started.publish(String(data=json.dumps(
                {"id": uid, "dur_s": 0.0, "t_us": time.time_ns() // 1000})))
            first = True
            for pcm in self._sentences(text):
                if interruptible and self._interrupt.is_set():
                    break
                self._burst(pcm)
                sent_s += len(pcm) / RATE
                if first:
                    self.get_logger().info(
                        f"say({uid!r}): first sentence in {time.monotonic()-t0:.2f}s")
                    first = False
                # stay up to ~4 s ahead of playback (bigger cushion = fewer
                # Mac-side underruns; still interruptible mid-story)
                if not self._sleep_until(t0 + sent_s - 4.0, interruptible):
                    break

        # hold the half-duplex gate until playback finishes (+ tail), unless
        # interrupted (daemon flushed its ring — playback already stopped)
        finished = self._sleep_until(t0 + sent_s + GATE_TAIL_S, interruptible)
        self.pub_active.publish(Bool(data=False))
        self.pub_done.publish(String(data=uid))
        if not finished:
            self.get_logger().info(f"say({uid!r}) preempted after {sent_s:.1f}s")
        else:
            self.get_logger().info(f"say({uid!r}): {sent_s:.2f}s audio done")


def main():
    rclpy.init()
    rclpy.spin(TtsNode())


if __name__ == "__main__":
    main()
