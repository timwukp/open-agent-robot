# Open-Architecture Multimodal Robot Agent Stack — Fact-Checked Research Report

**Date:** 2026-08-20 · **Method:** two deep-research workflow rounds (135 subagents total): 5-angle web sweep → 24 primary sources fetched → 120 claims extracted → 37 claims adversarially verified (3-vote panels on high-risk claims) → 34 confirmed, 2 refuted, 1 corrected. Every load-bearing number below traces to a primary source (official docs, GitHub/HuggingFace, AWS pricing feed) checked on 2026-08-20. Estimates that could NOT be verified are explicitly marked **[estimate]** or **[unverified]**.

**Scope:** no physical robot yet. Simulation + agent on AWS EC2; MacBook provides browser rendering and camera/mic/speaker via a C daemon. Hard constraint: **no AGPL components**.

---

## 0. TL;DR — Recommended reference architecture

| Layer | Pick | License | Why |
|---|---|---|---|
| Simulator | **Gazebo Jetty** (LTS, supported Sep 2025 → May 2031) headless on EC2, paired with ROS 2 Lyrical | Apache-2.0 | Official Gazebo↔ROS 2 pairing; runs headless with `gz sim -s` |
| GPU instance | **g6.xlarge** (1× NVIDIA L4, 24 GB VRAM) — $0.8048/hr on-demand us-east-1 (verified via AWS Pricing API) | — | 24 GB fits sim + detector + TTS; NVENC ×2 if you later want Isaac Sim streaming |
| Browser viz | **Foxglove web app (Chrome) + foxglove_bridge (MIT)** over an **SSH tunnel** (`ssh -L 8765:localhost:8765`) | bridge MIT; Foxglove app free-for-personal (no longer open-source) | Zero certs: `ws://localhost:8765` is exempt from Chrome's mixed-content block |
| Vision detector | **D-FINE** (accuracy king) or **RT-DETR** (speed); RF-DETR N/S/M/L as alternative | all Apache-2.0 | YOLO26 exists but is AGPL-3.0 — disqualified |
| Perception schema | ROS 2 **vision_msgs** `Detection2DArray` + `VisionInfo` | Apache-2.0 | Canonical; note `class_id` is a **string** in ROS 2 |
| Streaming ASR (zh+en) | **FunASR paraformer-zh-streaming** (600 ms chunks) + SenseVoiceSmall as final-pass refiner; deploy via **sherpa-onnx (Apache-2.0, C API)** | Apache-2.0 / MIT+FunASR-license | Parakeet/Canary/Moshi/Kyutai all lack Chinese |
| Streaming TTS (zh+en) | **Kokoro-82M** (+ v1.1-zh variant), served by Kokoro-FastAPI (~300 ms first packet on GPU) | Apache-2.0 (weights + lib) | Only Apache/MIT-weights TTS with Chinese; F5/Fish/XTTS all license-blocked |
| Local omni model | **MiniCPM-o 4.5** (9B, true full-duplex video+audio in / speech+text out, zh+en) — quantized full-duplex runs on a 12 GB GPU or Apple M4 Max 24 GB | repo Apache-2.0 (verify weights card before commercial use) | The only small open model that truly does streaming vision+audio+speech |
| Cloud LLM | **Claude Haiku 4.5** on Bedrock — $1/$5 per 1M tokens; ID `anthropic.claude-haiku-4-5-20251001-v1:0`. **There is no "Haiku 5"** (verified against the live Bedrock pricing table) | proprietary | Vision (image+text in, text out; NO audio), streaming, tools, prompt caching |
| Closed-source speech demo | **Amazon Nova 2 Sonic** (`amazon.nova-2-sonic-v1:0`) — English-side demo only: **7 languages, Chinese NOT supported; no vision input** | proprietary | Bidirectional streaming API; v1 published latency 1.09 s |
| Agent framework | **ROSA (NASA JPL, Apache-2.0, LangChain-based)** to start; thin custom loop later | Apache-2.0 | v1.0.10 (2026-03) added Anthropic/Claude support |
| C usage | C for the **hardware/peripheral layer only** (MacBook daemon: miniaudio + ffmpeg-avfoundation); NOT the whole platform | pub-domain/MIT-0, LGPL-2.1+ | ROS 2 core maintains only rclcpp/rclpy; C path (rclc) is micro-controller oriented |

---

## 1. Simulation & browser rendering (all claims verified 3-0 unless noted)

### 1.1 Simulator choice
- **Gazebo Jetty is the current release** — an LTS supported Sep 2025 → May 2031 (next release "M" due Mar 2027). Official ROS 2 pairings: **Jetty↔Lyrical, Harmonic↔Jazzy, Fortress↔Humble**; other combinations are marked incompatible/use-with-caution. Your ROS 2 distro choice on EC2 therefore fixes the Gazebo version. Source: gazebosim.org/docs.
- Gazebo runs headless on a server with `gz sim -s` (server-only flag) — no GUI, no GPU strictly required for physics-only runs.
- **Isaac Sim** (heavier alternative): official headless streaming exists — a native WebRTC client (Win/macOS/Linux) and a **browser web viewer via Docker Compose** (Chromium; NVIDIA's recommended cloud path). Requirements verified:
  - **NVENC hardware encoder required**; explicitly unsupported on A100. AWS mapping: g5 (A10G, 1× NVENC), g6 (L4, 2×), g6e (L40S, 3×) work; p4d does not; CPU-only ruled out.
  - Ports 49100/TCP (signaling), 47998/UDP (media), 8210/TCP (web viewer) — **shipped with NO authentication or encryption**; put behind TLS/auth before exposing.
- **Humanoid whole-body/muscle control: [unverified this round]** — no claims on this survived either round. Leads to validate next: `ros2_control` humanoid examples for joint-level control in Gazebo; MuJoCo (Apache-2.0) + MyoSuite for muscle-level actuation; Isaac Lab for RL policy training. Treat these as leads, not conclusions.

### 1.2 EC2 GPU instance (prices verified via AWS Pricing API, us-east-1 on-demand, 2026-08-20)

| Instance | GPU | VRAM | NVENC | $/hr |
|---|---|---|---|---|
| g5.xlarge | A10G | 24 GB | 1 | $1.006 |
| **g6.xlarge** | **L4** | **24 GB** | **2** | **$0.8048** |
| g6e.xlarge | L40S | 48 GB | 3 | $1.861 |

g6 is cheaper than g5 with a newer GPU. AWS's "up to 2× better than G4dn" figure is first-party marketing (verified 2-1); no official g6-vs-g5 comparison exists.

### 1.3 Browser rendering paths, ranked for the demo
1. **Foxglove web app + foxglove_bridge (fastest/cheapest).** foxglove_bridge is Foxglove's officially recommended live ROS 2 connection (over rosbridge); the web app in Chrome needs no ROS install on the MacBook. **Caveat 1:** Foxglove itself is no longer open-source (2024 license change; free for personal use) — the bridge package is MIT. **Caveat 2 (refuted claim — do not repeat):** "just open port 8765" does NOT work: Chrome blocks `ws://` from an https page (mixed content, verified against the Mixed Content / Secure Contexts specs). Working options, verified:
   - **SSH tunnel (recommended for demo):** `ssh -L 8765:localhost:8765 ec2-host`, then connect the web app to `ws://localhost:8765`. Loopback is spec-exempt from mixed content (Chromium lists localhost as a secure origin for any scheme). Zero certs, zero extra cost.
   - **Caddy reverse proxy** → automatic TLS → `wss://` (needs a DNS name). foxglove_bridge can also terminate TLS itself (`tls/certfile/keyfile` params).
   - **Tailscale:** a bare tailnet IP does NOT fix mixed content; you must use `tailscale serve 8765` (auto TLS on a ts.net name). Adds a DERP-relay hop when direct WireGuard fails.
2. **Amazon DCV full desktop (zero software cost on EC2).** HTML5 client in any modern browser on macOS, no install; you pay only instance + data-transfer-out (demo traffic usually within the 100 GB/mo free tier). Default web client cap: two 1920×1080 screens. Use when you want the full Gazebo GUI rather than topic-level viz.
3. **Isaac Sim WebRTC web viewer** — best visual fidelity, highest setup cost (Docker Compose on Ubuntu host, NVENC, and you must add TLS/auth yourself).

---

## 2. Vision pipeline

### 2.1 YOLO26: exists, and is disqualified
- **YOLO26 is real**: official Ultralytics model, announced 2025-09-25 at YOLO Vision 2025, pretrained n–x weights, COCO 40.9–57.5 mAP, 1.7–11.8 ms T4 TensorRT; positioned by Ultralytics as "the recommended model for new projects".
- **License: AGPL-3.0 / paid Enterprise, dual.** Ultralytics' license page states verbatim that *all* Ultralytics YOLO trained models fall under AGPL-3.0 by default, naming "the complete Ultralytics YOLO source code portfolio, including YOLO26". Whether AGPL legally binds weights is debated, but Ultralytics asserts it does → compliance risk stands. **Disqualified** under the no-AGPL constraint. (A broader claim about AGPL derivative-work scope was refuted 0-3 — the exact legal reach of AGPL was NOT verified; only Ultralytics' own position was.)

### 2.2 Verified Apache-2.0 alternatives (all license checks via GitHub API SPDX)

| Detector | License | COCO AP | Speed (T4) | Notes |
|---|---|---|---|---|
| **D-FINE** (Peterande/D-FINE) | Apache-2.0 | L: 54.0 / X: 55.8 | 124 / 78 FPS | ICLR 2025 spotlight; strongest verified Apache detector; built on RT-DETR lineage (also Apache — no upstream copyleft) |
| **RT-DETR** (lyuwenyu/RT-DETR) | Apache-2.0 | R18: 46.5 / R50: 53.1 | 217 / 108 FPS (FP16) | CVPR 2024 "DETRs Beat YOLOs"; repo active 2026-08 |
| **RF-DETR** (roboflow/rf-detr) | **Split**: N/S/M/L Apache-2.0; **XL/2XL = PML 1.0 (avoid)** | — | — | Also instance seg + keypoints (preview); Seg-XL/2XL are Apache but detection XL/2XL are not |

FPS numbers are author-reported on a single T4 class — re-benchmark on the L4 before locking the latency budget. YOLOX/YOLO-NAS were not fetched this round (older models; the three above dominate 2026 Apache-licensed SOTA).

### 2.3 Situational-awareness metadata schema
- **ROS 2 `vision_msgs` (Apache-2.0, ros-perception, active ros2 branch)** is the canonical schema. YOLO-family bounding-box output maps fully onto `Detection2DArray`.
- **Correction found by verification (the only refuted claim of round 2):** in the ROS 2 branch (≥2.0.0, Humble/Jazzy/Kilted), `ObjectHypothesis` carries **`string class_id`**, not a numeric ID — the numeric-ID + `VisionInfo` database lookup description is a stale ROS 1 remnant (CHANGELOG: 2.0.0 "Convert id to string", 3.0.0 "Rename ObjectHypothesis.id → class_id"). Design your agent-facing JSON accordingly: put human-readable class names directly in `class_id`; `VisionInfo` is optional.
- Suggested agent-facing envelope (JSON mirror of `Detection2DArray`, one message per frame): `{header:{stamp,frame_id}, detections:[{class_id:string, score:float, bbox:{cx,cy,w,h}, track_id?:string}], scene_meta:{camera, resolution}}` — **[design proposal, not a standard]**.

---

## 3. Speech stack (zh + en, streaming)

### 3.1 The elimination result (why the shortlist is short)
Verified against model cards, 2026-08-20:
- NVIDIA **Parakeet-TDT-0.6B v2/v3**: CC-BY-4.0 (commercial OK) but **no Chinese** (v3 = 25 European languages; RTFx ~3,333 is offline batch throughput, not streaming latency).
- NVIDIA **Canary-1b-v2 / Canary-Qwen-2.5B**: CC-BY-4.0 but no Chinese (original Canary-1B is the CC-BY-NC one).
- **Kyutai STT**: superb 0.5 s streaming delay, CC-BY-4.0 — English/French only.
- **Moonshine**: MIT, genuinely low-latency streaming — English-only (the zh checkpoint is an empty placeholder card, no license).
- **faster-whisper**: MIT, ~4× openai/whisper (large-v2 FP16: 13 min audio in 63 s / 4.5 GB; int8: 59 s / 2.9 GB) but **no native streaming** — needs a wrapper. ufal/whisper_streaming (~3.3 s latency) is declared outdated by its own authors, superseded by **SimulStreaming** (MIT, ~5× faster, 99 languages incl. zh) — a viable Whisper-based fallback.

### 3.2 Recommended ASR
- **Primary: FunASR `paraformer-zh-streaming`** — Apache-2.0 (HF card), Chinese+English, true chunked streaming with 600 ms chunks (chunk_size [0,10,5]) → first partial ≈ 600 ms + compute; FunASR ships a production WebSocket runtime (toolkit code MIT).
- **Final-pass refiner: SenseVoiceSmall** (zh/yue/en/ja/ko) — non-autoregressive, >5× faster than Whisper-Small; officially **non-streaming**, so use it endpoint-triggered for punctuation/quality. Weights under the custom FunASR model license with maintainer-confirmed commercial use (attribution required).
- **Deployment path that fits the C philosophy: sherpa-onnx (k2-fsa, Apache-2.0)** — a C-API runtime that runs streaming zipformer zh-en models and SenseVoice on macOS/Linux/Raspberry Pi. This is how the MacBook-side C daemon can do local ASR with zero Python.

### 3.3 Recommended TTS
- **Primary: Kokoro-82M** — weights AND inference lib Apache-2.0; Mandarin via `lang_code='z'`; dedicated `Kokoro-82M-v1.1-zh` variant (Apache-2.0, 100 professional zh speakers, self-described early release — validate prosody yourself). Native chunk-by-chunk generator; **Kokoro-FastAPI** (Apache-2.0) serves OpenAI-compatible streaming with measured **~300 ms first packet on GPU, <1 s on M3 Pro CPU**, 35–100× realtime GPU throughput. At 82M params it coexists with sim + detector on one L4.
- **Runner-up: Chatterbox Multilingual** (Resemble AI) — MIT code+weights, 500M, zh among 23 languages; but the "sub-200 ms" figure is their hosted service, and the OSS repo documents no streaming API (Turbo variant is English-only) → streaming is DIY.
- **CosyVoice 2 / Fun-CosyVoice 3.0** (repo now redirects to QwenAudio/CosyVoice; Apache-2.0 code): bidirectional streaming with 150 ms first-packet (README claim, not in the paper). **Attribution correction from verification:** "9 languages + 18 Chinese dialects" belongs to **Fun-CosyVoice 3.0** (Dec 2025); CosyVoice 2 itself covers zh/en/ja/ko + ~5 dialects. Weights: HF card tagged apache-2.0 but the ModelScope listing declares no license — verify before commercial use. Strong option if you want voice cloning; heavier than Kokoro.
- **License-blocked (verified, avoid for anything commercial):** F5-TTS (weights CC-BY-NC), Fish-Speech/OpenAudio S1 (Fish Audio Research License / CC-BY-NC-SA), XTTS-v2 (Coqui CPML strictly non-commercial; original repo dead since 2024-08), Orpheus-zh (Apache tag but finetuned from Llama-3.2 → provenance questionable).

### 3.4 AWS Nova Sonic (closed-source demo option) — significant findings
- The current model is **Amazon Nova 2 Sonic** (`amazon.nova-2-sonic-v1:0`, GA 2025-12-02); original `amazon.nova-sonic-v1:0` remains available.
- **Chinese is NOT supported.** Exactly 7 languages: English (US/UK/India/Australia), Spanish, German, French, Italian, Portuguese, Hindi. AWS explicitly recommends against other languages. → Nova Sonic can only demo the **English** side of your bilingual robot.
- **No vision input** (audio+text in; speech+text out) — confirmed against the AI Service Card. Pair it with Haiku 4.5 for images if used.
- API: `InvokeModelWithBidirectionalStream` (HTTP/2, JSON events: audioInput/audioOutput/textOutput/toolUse), 16 kHz LPCM in / 24 kHz out, **8-minute connection limit** (renew + session-continuation pattern) — matters for a long-running robot loop.
- Pricing (us-east-1, verified via the Bedrock pricing feed): speech **$3.00/1M in, $12.00/1M out**; text $0.33/$2.75 per 1M. Latency: v1 press release published **1.09 s** average perceived (speech-end → first speech); Nova 2 has no absolute published figure (Mar 2026 refresh cut p50 by 150 ms).
- Regions: us-east-1, us-west-2, ap-northeast-1 (Tokyo), eu-north-1.

---

## 4. Unified omni models (one open model for vision+audio+speech), all 3-0 verified

### 4.1 MiniCPM-o 4.5 — the local-hardware winner
- OpenBMB, open-sourced **2026-02-03** (repo now lives at OpenBMB/MiniCPM-V). **9B** omni: SigLip2 (vision) + Whisper-medium (audio-in) + CosyVoice2 (speech-out) + Qwen3-8B (LLM). Claimed to match Gemini 2.5 Flash on vision+speech.
- **True non-blocking full-duplex**: simultaneous video+audio input streams and speech+text output streams; proactive interaction. **Bilingual zh+en speech.**
- Hardware (correction-refined by the verify panel — the 28 GB and 12 GB figures come from different backends):
  - PyTorch bf16 web demo (full-duplex omni): ~28 GB GPU.
  - **llama.cpp-omni quantized path: full-duplex omni on a 12 GB NVIDIA GPU or Apple M4 Max with 24 GB RAM**; half-duplex speech on M3/M4 16 GB or a 12 GB GPU. GGUF ≈10 GB, AWQ ≈11 GB; Ollama/CPU inference supported.
  - Known README limitation: speech synthesis can mispronounce characters in full-duplex mode.
- License: repo Apache-2.0; **check the HF weights card before commercial use** (earlier MiniCPM weights used a custom registration license).
- **Fit:** this is the "跑在本地小機器" answer — one model replacing ASR+LLM+TTS+vision on a 12 GB Jetson-class box or a higher-end MacBook.

### 4.2 Qwen3-Omni-30B-A3B — the EC2-class option
- Apache-2.0; MoE Thinker-Talker; text+image+audio+video in, streaming text+speech out; 119 text / 19 speech-in / 10 speech-out languages (zh+en included).
- **VRAM: 78.85–144.81 GB BF16** (Instruct, theoretical minimum, transformers+FlashAttention2; Thinking variant 68.74–131.65 GB) → not a MacBook/Jetson model at full precision.
- Quantized asymmetry (important, from the verify panel): GGUF Q4_K_M (18.6 GB) and MLX builds run on a 32–48 GB MacBook **but only for audio+vision INPUT — the Talker's streaming speech OUTPUT is not supported in llama.cpp/MLX**; native speech-out still requires the CUDA vLLM/transformers stack.
- **Qwen2.5-Omni-7B (Apache-2.0)** is the smallest Apache omni that keeps speech output (Qwen2.5-Omni-3B is non-commercial "qwen-research"; Qwen3-Omni-Flash is API-only).

### 4.3 Moshi / MoshiVis — disqualified for this project
- Kyutai Moshi: full-duplex speech, 160 ms theoretical / ~200 ms practical on an L4; MIT/Apache code, CC-BY-4.0 weights; official MLX backend runs on a MacBook M3 (bf16/int8/int4). MoshiVis adds a 206M vision adapter + PaliGemma2 encoder (license ambiguity: README says Gemma license in one section, Apache in another — assume Gemma).
- **Moshi is English-only** (model card; Kyutai's ecosystem adds only French). → best-in-class latency reference, but unusable for a zh+en robot without unproven finetuning.

---

## 5. Agent framework & the "write everything in C" question

### 5.1 Cloud LLM: Claude Haiku 4.5 (there is no Haiku 5)
- Verified against the live Bedrock pricing table (2026-08-20): the Anthropic lineup tops out at Opus 5 / Sonnet 5 / Fable 5; **the newest Haiku is 4.5** (launched 2025-10-15). Zero occurrences of "Haiku 5".
- $1.00 / $5.00 per 1M tokens (batch 50% off; cache read $0.10; 5m cache write $1.25). ID `anthropic.claude-haiku-4-5-20251001-v1:0` (+ us./global./eu./au./jp. inference profiles).
- **Image+text in, text out. NO audio input** → ASR/TTS must be separate components (consistent with the architecture above). 200K context, 64K max output, streaming, tool calling, prompt caching (Bedrock minimum checkpoint 4,096 tokens). Structured outputs work on bedrock-runtime but NOT on the bedrock-mantle Messages endpoint.

### 5.2 Agent frameworks (verified subset)
- **ROSA (NASA JPL)**: Apache-2.0, LangChain-based natural-language agent for ROS1/ROS2 (inspect/diagnose/operate); v1.0.10 (2026-03-17) added Anthropic/Claude providers. Demos are TurtleSim; Isaac Sim listed "coming soon". Adopting it means a LangChain dependency — fine for the demo, replaceable by a thin custom loop (Bedrock Converse + tools) later.
- **dora-rs**: Apache-2.0 Rust dataflow middleware; native APIs for Rust/Python/**C**/C++ (its own table grades C/C++ "Supported", Rust/Python "First-class"); Apache Arrow zero-copy; self-reported 10–17× faster than ROS 2 Python; **experimental** bidirectional ROS 2 bridge → can coexist with the Gazebo/ROS 2 sim rather than replace it. **AGPL trap found during verification:** the Node Hub's `dora-yolo` node declares MIT but depends on `ultralytics>=8.3.24` (AGPL) — do NOT use dora-yolo; wire D-FINE/RT-DETR into your own node.
- LangGraph / Strands Agents / OpenVLA / π0 / GR00T N1: **[not verified this round]** — no fetched claims survived. VLA models are a genuine alternative architecture (vision→action end-to-end, skipping the schema-passing design) worth a dedicated follow-up if you get real hardware.

### 5.3 Is "the whole platform in C" sound engineering? No — C belongs at the edges (verified)
- ROS 2 core maintains **only rclcpp (C++) and rclpy (Python)**. `rcl` is a shared C core *designed for language bindings to wrap* ("C is the easiest language to wrap"), not for application code. Feature-complete C requires community **rclc**, documented via micro-ROS and **oriented to microcontrollers**.
- ROS 2 explicitly endorses polyglot nodes over generated interfaces. Fighting this means owning your own C framework with no ecosystem.
- **Recommendation:** C where it buys you something — (a) the MacBook peripheral daemon, (b) future MCU firmware via micro-ROS/rclc, (c) optionally dora-rs C-API nodes for hot paths. Agent/orchestration in Python; performance-critical ROS nodes in C++.

### 5.4 The MacBook C daemon (mic/speaker/camera) — verified engineering answers
- **Audio: miniaudio** — single C source file, Core Audio backend on macOS, zero external deps, public-domain/MIT-0 dual license, actively maintained. Notarization gotcha from its own header: default runtime framework linking may fail notarization → compile with `-DMA_NO_RUNTIME_LINKING` and link `-framework CoreFoundation -framework CoreAudio -framework AudioToolbox`.
- **Camera: libuvc is impossible** for the built-in camera — live-checked on this MacBook: the camera appears under `SPCameraDataType` and **nowhere in `SPUSBDataType`** (not a USB/UVC device). Even external UVC cams can't be claimed via libusb without an Apple entitlement CLI apps can't hold. Real options:
  1. **ffmpeg `avfoundation` input device (libavdevice — a pure C API)** — the pragmatic choice. FFmpeg is LGPL-2.1+ (keep `--enable-gpl` off); no AGPL anywhere.
  2. Direct AVFoundation from C via `objc_msgSend` (libobjc is a plain C library, explicitly intended for bridging) or a ~100-line Objective-C shim compiled into the daemon — zero dependencies, more code.
- **TCC permissions (hard constraint):** the binary needs `NSCameraUsageDescription`/`NSMicrophoneUsageDescription`. A daemon launched from Terminal gets the grant attributed to Terminal (fine for the demo). **A true headless LaunchDaemon has NO path to camera access** (MDM PPPC is deny-only for camera/mic) → run it as a LaunchAgent/app bundle in the user session.

---

## 6. Deployment topology

```
MacBook (demo I/O + viewing)                    AWS EC2 g6.xlarge (us-east-1, $0.8048/hr)
┌─────────────────────────────┐                 ┌───────────────────────────────────────────┐
│ C daemon (LaunchAgent)      │                 │ ROS 2 Lyrical + Gazebo Jetty (gz sim -s)  │
│  ├ miniaudio  mic/speaker   │  Opus/PCM over  │  ├ humanoid URDF + ros2_control           │
│  ├ ffmpeg-avfoundation cam  │  WebSocket/     │  ├ /camera_sim, /joint_states topics      │
│  └ sherpa-onnx (opt. local  │  WebRTC ───────▶│ Perception node: D-FINE (Apache-2.0)      │
│     streaming ASR, C API)   │                 │  └ vision_msgs/Detection2DArray ──┐       │
│ Chrome:                     │                 │ Speech: paraformer-zh-streaming,  │       │
│  Foxglove web app ──────────┼── ssh -L 8765 ─▶│  SenseVoice refine, Kokoro-82M TTS│       │
│  (ws://localhost:8765)      │                 │ Agent (ROSA / custom loop) ◀──────┘       │
└─────────────────────────────┘                 │  ├ Bedrock: Claude Haiku 4.5 (vision+plan)│
        Path B (closed demo): Nova 2 Sonic      │  └ actions → /joint_trajectory + TTS out  │
        (EN only) via InvokeModelWith-          └───────────────────────────────────────────┘
        BidirectionalStream from the daemon      Local-model variant: swap the speech+agent
                                                 column for MiniCPM-o 4.5 (12 GB quantized)
```

## 7. Voice-loop latency budget (speech-end → first audio out)

| Stage | Number | Status |
|---|---|---|
| Mic capture + uplink (MacBook→EC2) | 50–150 ms | **[estimate — measure]** |
| Streaming ASR final partial (paraformer, 600 ms chunks) | ~600 ms + compute | verified chunk size; compute unmeasured |
| Agent LLM (Haiku 4.5, short reply, streaming) | first tokens few-hundred ms | **[unpublished — measure via Bedrock]** |
| TTS first packet (Kokoro-FastAPI, GPU) | ~300 ms | verified (project-measured) |
| Downlink + playback buffer | 50–100 ms | **[estimate]** |
| **Total (open-source path)** | **≈1.1–1.7 s** | comparable to Nova Sonic v1's published 1.09 s |
| Alternative: MiniCPM-o 4.5 full-duplex (collapses ASR+LLM+TTS) | sub-second class claimed | **[no published ms figure — benchmark]** |
| Reference ceiling: Moshi full-duplex | ~200 ms on L4 | verified, but English-only |

## 8. Demo cost estimate

| Item | Rate | 8-hr demo day |
|---|---|---|
| g6.xlarge on-demand | $0.8048/hr | **$6.44** |
| (option) g6e.xlarge if Isaac Sim + big models | $1.861/hr | $14.89 |
| Haiku 4.5 agent calls (~200 turns × ~3K in/300 out) | $1/$5 per 1M | ~$0.90 |
| Nova 2 Sonic English demo (~30 min speech ≈ 200K speech tokens each way) | $3/$12 per 1M | ~$3.00 |
| DCV / Foxglove-via-SSH | $0 | $0 |
| **Total** | | **≈ $10–19/day** [token volumes are estimates] |

Stop the instance when idle; EBS storage (~$8/mo for 100 GB gp3) persists the setup. **[EBS rate from general knowledge — not re-verified.]**

## 9. Consolidated license table

| Component | Code | Weights | zh | Verdict |
|---|---|---|---|---|
| Gazebo Jetty / ROS 2 / vision_msgs | Apache-2.0 | — | — | ✅ |
| foxglove_bridge | MIT | — | — | ✅ (Foxglove app: free personal, not OSS) |
| Isaac Sim | proprietary (free tier) | — | — | ⚠️ demo-ok |
| YOLO26 (Ultralytics) | AGPL-3.0/Enterprise | AGPL-3.0 (their claim) | — | ❌ banned |
| D-FINE / RT-DETR | Apache-2.0 | Apache-2.0 | — | ✅ |
| RF-DETR N/S/M/L | Apache-2.0 | Apache-2.0 (XL/2XL = PML 1.0 ❌) | — | ✅ base sizes |
| paraformer-zh-streaming | MIT (FunASR) | Apache-2.0 (HF) | ✅ | ✅ |
| SenseVoiceSmall | MIT | FunASR license (commercial OK w/ attribution) | ✅ | ✅⚠️ |
| sherpa-onnx | Apache-2.0 | per-model | ✅ | ✅ |
| faster-whisper / SimulStreaming | MIT | Whisper MIT | ✅ | ✅ fallback |
| Kokoro-82M (+v1.1-zh) | Apache-2.0 | Apache-2.0 | ✅ | ✅ **recommended TTS** |
| Chatterbox Multilingual | MIT | MIT | ✅ | ✅ (no OSS streaming API) |
| CosyVoice 2 / Fun-CosyVoice 3 | Apache-2.0 | HF apache tag; ModelScope unlicensed | ✅ | ⚠️ verify weights |
| F5-TTS / Fish-Speech / OpenAudio / XTTS-v2 | misc | CC-BY-NC / research / CPML | ✅ | ❌ non-commercial |
| MiniCPM-o 4.5 | Apache-2.0 | check HF card | ✅ | ✅⚠️ **local omni pick** |
| Qwen3-Omni-30B-A3B / Qwen2.5-Omni-7B | Apache-2.0 | Apache-2.0 | ✅ | ✅ (3B variant ❌ research-only) |
| Moshi / MoshiVis | MIT/Apache | CC-BY-4.0 (+Gemma encoder) | ❌ EN-only | ❌ for zh |
| ROSA / dora-rs | Apache-2.0 | — | — | ✅ (dora-yolo node ❌ pulls ultralytics) |
| miniaudio | pub-domain / MIT-0 | — | — | ✅ |
| FFmpeg (avfoundation) | LGPL-2.1+ (no GPL parts) | — | — | ✅ |
| Claude Haiku 4.5 / Nova 2 Sonic | proprietary | — | Haiku ✅ / Sonic ❌ | demo/cloud tier |

## 10. Refuted or corrected during verification (do not repeat these)
1. ❌ "Open port 8765 and the Foxglove web app connects to ws://EC2_IP:8765" — blocked by mixed content (0-3). Use SSH tunnel / wss.
2. ❌ Broad claim about AGPL forcing release of the entire derivative work incl. weights — unverifiable as stated (0-3). Stick to "Ultralytics asserts AGPL covers its models".
3. ✏️ vision_msgs ROS 2: `class_id` is a string, not a numeric ID (1-1 refuted, corrected in §2.3).
4. ✏️ "9 languages + 18 dialects" belongs to Fun-CosyVoice 3.0, not CosyVoice 2.
5. ✏️ MiniCPM-o "28 GB full-duplex" is the PyTorch demo only; quantized full-duplex needs 12 GB.

## 11. Open items (not covered by verified evidence)
- Humanoid whole-body/muscle-level control stack in simulation (ros2_control vs MuJoCo/MyoSuite vs Isaac Lab) — needs its own research pass.
- LangGraph / Strands / VLA models (OpenVLA, π0, GR00T N1) as the agent layer — unverified.
- Measured end-to-end latency on the actual g6 + MacBook link — all [estimate] rows in §7.
- MiniCPM-o 4.5 HF weights-license text — read the card before any commercial decision.
- vision_msgs 3.x exact field layout for the target ROS distro (Lyrical) — confirm at build time.

---
*Sources: 24 primary sources round 1 + ~40 primary sources round 2 (AWS docs/pricing feed, gazebosim.org, docs.isaacsim.omniverse.nvidia.com, docs.foxglove.dev, GitHub/HF model cards, W3C specs). Full per-claim evidence and vote records: workflow transcripts `wf_805f95a7-318` and `wf_ee7e2e3c-78d` in the session directory.*
