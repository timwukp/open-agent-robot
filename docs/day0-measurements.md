# Day-0 Measurement Gates — 2026-08-20 (all PASSED)

Instance: g4dn.xlarge (Tesla T4 16GB, 4 vCPU, 16GB RAM), ap-southeast-1, Ubuntu 24.04,
NVIDIA driver 595.71.05 (server-open), ROS 2 Jazzy, mujoco 3.12.0, ultralytics 8.4.124,
TensorRT 11.2.1.2, mediapipe 1.0.1.

| Gate | Result | Verdict |
|---|---|---|
| (a) YOLO26 on T4 | **YOLO26s TensorRT fp16: 6.2 ms/frame = 161 FPS** (640×640, batch 1; engine build 170 s one-time) | ✅ perception has ~10× real-time headroom at 15 fps camera |
| (b) Haiku 4.5 TTFT from EC2 SG | **p50 900 ms, min 645, max 1270** (n=10, ConverseStream, `global.` inference profile — the direct model ID is NOT on-demand-invocable in ap-southeast-1, and only the `global.` profile exists there) | ✅ for async banter/chat (plan D2); confirms beat path must never wait on LLM |
| (c) VRAM residency | 527 MiB / 15360 MiB with YOLO26 engine loaded | ✅ huge margin; Kokoro+paraformer will add ~2-3 GB |
| (d) MacBook↔EC2 link | **TCP RTT median 5.6 ms** (5.6 ms median, 4.5–11.2 range; ICMP blocked by design — SG allows only 22/tcp) | ✅ Singapore region choice validated; audio streaming budget trivial |

## Cantonese TTS — measured 2026-08-22 (T4, `CUDA_AVAIL=True`, fp32 unless noted)

Round 2's stack survey recorded CosyVoice2-0.5B as "streaming, ~150 ms first packet".
**Measured, that is off by 51×**, and the model is also the wrong choice for another
reason. First-packet latency — not rtf — is the metric that matters for a talking robot.

| Model | Mode | First packet | rtf | Audio produced | Verdict |
|---|---|---|---|---|---|
| CosyVoice-300M-SFT, speaker `粤语女` | streaming | **3.63 s** | 1.37 | 4.88 s for 28 chars | ✅ correct output, too slow to converse |
| CosyVoice-300M-SFT | one-shot | 6.75 s | 1.39 | — | — |
| + fp16 | streaming | 4.52 s | 1.72 | — | ❌ fp16 is *slower* on a T4 |
| CosyVoice2-0.5B, zero-shot | streaming | **7.64 s** | 1.64 | **16.80 s for 22 chars** | ❌ 2.1× slower *and* wrong |
| CosyVoice2-0.5B, zero-shot | one-shot | 18.37 s | 1.09 | 16.80 s | ❌ |
| Kokoro-82M Mandarin (in use) | streaming | ~0.15 s | ~0.06 | — | baseline: real Cantonese costs 24× |

**Why CosyVoice2 was tried and why it is rejected.** It ships no built-in Cantonese
speaker, so zero-shot with our own 300M-SFT Cantonese clip as the reference was the
only route. It produced 16.80 s of audio for a sentence the SFT model renders in
under 5 s. Two candidate explanations were tested and both **rejected**: envelope
autocorrelation shows no periodic repetition (peak r = 0.14, *below* the known-good
clip's 0.16), and it is not trailing silence — the clip is **89% active speech,
14.97 s of it**. So the model generated ~3× too much *content*. Scope this honestly:
this refutes **synthetic-prompt zero-shot**, not CosyVoice2's Cantonese in general —
a ~10 s *human* Cantonese reference clip is untested and is the cheapest next probe.

**Decision:** keep CosyVoice-300M-SFT for Cantonese and attack latency with a
**phrase cache** — pre-synthesise the fixed lines (greeting, game chant, stop
acknowledgement) so the common cases cost ~0, and only novel sentences pay 3.63 s.

Three dependency defects found on the way, all of which apply to production if the
Cantonese path ever moves to zero-shot (today's SFT path never loads a prompt wav,
which is the only reason `speech/yue_server.py` works):

- `torchaudio.load(..., backend='soundfile')` — **the `backend` argument is ignored**
  by torchaudio 2.11; it routes to `load_with_torchcodec` and raises `ImportError`
  unless `torchcodec` is installed. Fix is `pip install torchcodec` or pin
  `torchaudio<2.9`.
- `frontend_zero_shot(…, prompt_wav, …)` takes a **path**, not a tensor — it calls
  `load_wav()` on it three times internally.
- `transformers` now honours the checkpoint dtype, so the Qwen2 backbone loads as
  **BFloat16** while the frontend feeds Float32 → `mat1 and mat2 must have the same
  dtype`. Passing `fp16=False` does **not** prevent this; the modules need an
  explicit `.float()`.

## Bonus findings during Day 0
- **Phase 1 already demonstrated**: G1-with-hands renders at 1280×720 via EGL headless,
  MJPEG streams on :8080; frame inspected visually — full body + articulated fingers visible
  (wave gesture mid-motion). MuJoCo offscreen framebuffer needed an `offwidth/offheight`
  wrapper XML (defaults to 640×480).
- **Kinematic fixed-base needs no model surgery**: gesture scene never calls `mj_step`,
  so the pelvis freejoint qpos never integrates — base is naturally fixed.
- **boto3 pitfall**: funasr pins boto3 1.34 (no `converse_stream`); venv needs
  `pip install -U boto3 botocore` after the ML stack.
- **`ubuntu-drivers install --gpgpu` is headless-only**: nvidia-smi and EGL user-space
  libs need explicit `nvidia-utils-<v>-server` + `libnvidia-gl-<v>-server`.
- **pkill self-kill trap**: `ssh 'pkill -f body_node.py; …relaunch body_node.py…'` kills
  the remote shell itself (pattern matches its own cmdline). Use pidfiles.
- MacBook daemon (miniaudio + ffmpeg-avfoundation camera): `--check-perms` passes —
  mic full-rate (68160 B / 2 s), camera OK at 1280×720@30→15 fps (avfoundation requires
  an exact supported mode; 15 fps requested directly is rejected).
