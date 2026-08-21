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
