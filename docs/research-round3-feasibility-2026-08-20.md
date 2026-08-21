# Round 3 — NVIDIA Stack, Humanoid Control & Hard Feasibility (Red-Team Edition)

**Date:** 2026-08-20 · **Method:** deep-research workflow round 3 (105 agents: 23 sources → 115 claims → 25 verified: 24 confirmed / 1 refuted) + **hands-on measurements on the actual target machines** (MacBook device tests, Bedrock TTFT) + critical-thinking red-team analysis. Builds on `research-multimodal-robot-agent-stack-2026-08-20.md` (rounds 1–2). Constraint change this round: **YOLO26 allowed (demo/non-commercial)**; Haiku 4.5 confirmed as the cloud LLM.

---

## 0. Executive verdict

**Feasible-with-effort, with one architecture correction:** the humanoid-locomotion portion should run in **MuJoCo with Unitree's shipped pre-trained G1 policy (zero training required)**, not Gazebo. Gazebo Jetty remains fine as a world/sensor simulator for non-locomotion scenes, but nothing in the 2026 practitioner ecosystem walks a humanoid in the gz-sim (Jetty) line. This contradicts the round-1/2 simulator decision for the locomotion demo — flagged for your sign-off (§6).

| Angle | Verdict |
|---|---|
| 1. NVIDIA open stack | feasible-with-effort — Isaac Lab genuinely BSD-3; **Isaac ROS NITROS proprietary → excluded**; GR00T weights downloadable but license terms UNVERIFIED |
| 2. Humanoid whole-body | **feasible-NOW** — MuJoCo + `unitree_rl_gym` pre-trained G1 policy, 50 Hz on CPU |
| 3. YOLO26 demo | feasible-NOW — NMS-free end-to-end by default; huge headroom on L4 |
| 4. VLA architecture | demonstrable-in-sim (GR00T LIBERO/SimplerEnv variants) but **premature as primary architecture** — keep schema-passing agent |
| 5. End-to-end integration | highest-risk unknown — literature produced ZERO surviving claims across 3 rounds; partially de-risked by this round's hands-on measurements (§4) |

---

## 1. NVIDIA open-source robotics stack — what is actually open (all 3-0 verified unless noted)

| Piece | License reality | Usable? |
|---|---|---|
| **Isaac Lab** (core) | **BSD-3-Clause** genuine (isaaclab_mimic ext = Apache-2.0; cuRobo dep proprietary) | ✅ for RL training — but full workflows (PhysX, RTX, URDF/MJCF importers) require **proprietary-binary Isaac Sim 6.0.x**; only the new Kit-less Newton workflows avoid it. Ships H1/G1 velocity-tracking envs (`Isaac-Velocity-{Flat,Rough}-{H1,G1}-v0`) as **configs only — no pre-trained checkpoints** → training compute still needed |
| **Isaac ROS / NITROS** | **NOT Apache-2.0** (falsified our assumption): current repo under proprietary "NVIDIA ISAAC ROS SOFTWARE LICENSE" — §4a locks developed apps to NVIDIA-GPU systems; §4f forbids any use that would subject it to an open-source license (**directly conflicts with shipping AGPL YOLO26 in the same application**); $10 liability cap. ≤v3.1 (pre-Dec-2024) was Apache-2.0; `isaac_ros_nitros_bridge` subpackage still Apache-2.0 | ❌ **excluded from the framework.** If ever needed: isolate as a separate process exchanging plain ROS 2 messages |
| **GR00T N1.7** (current, EA tag, 2026-02) | Weights downloadable on HF; **license-permissiveness claim REFUTED 1-2** — do NOT assume the NVIDIA Open Model License allows commercial use/redistribution; read the text before depending on it | ⚠️ optional stretch component only |
| **MuJoCo Playground** (DeepMind) | **Apache-2.0 throughout** — recursive tree scan: one LICENSE, sole exception a CC0 texture; Menagerie humanoid models all BSD-3/Apache/MIT | ✅ the fully-open non-NVIDIA training alternative (G1JoystickFlatTerrain, H1, Berkeley Humanoid envs; MJX needs a GPU to train) |

GR00T N1.7 technicals (3-0): 3B dual-system VLA — Cosmos-Reason2-2B VLM backbone + flow-matching DiT action head (SigLip2 vision, T5 text, MLP proprio). Official inference: Jetson Orin 2.9 Hz → L20 23.3 Hz TRT → L40 26.0 Hz → H100 35.9 Hz. **L4 is not a benchmarked row** — plausible (between Orin and L20) but unproven. Four post-trained sim variants ship (SimplerEnv Bridge/Fractal, DROID, LIBERO); evaluated on 57 sim manipulation tasks — so a **sim-only VLA demo is an officially supported NVIDIA path**.

## 2. Humanoid whole-body control — the round-1/2 gap, now resolved (3-0 ×5)

The 2026 field consensus, verified against primary repos:
- **Unitree's official pipeline**: Train (Isaac Gym/Lab) → Sim2Sim validate in **MuJoCo** → Sim2Real. **Gazebo appears nowhere**; Unitree's only Gazebo package (`unitree_ros`) explicitly "cannot perform high-level control, namely walking".
- The strongest Gazebo counter-example, **rl_sar** (Apache-2.0, active, G1 walking with libtorch/onnxruntime policy inference), targets **Gazebo CLASSIC + ROS 2 Foxy/Humble only** — repo-wide search: 0 hits for ros_gz/gz_ros2_control/Jetty. Nothing drops into Gazebo Jetty + Ubuntu 24.04 without a port. Only two 0-star hobby repos attempt gz-sim humanoid locomotion.

**Recommended demo path (feasible-now, zero training):**
- **`unitree_rl_gym` (BSD-3-Clause)** ships pre-trained policies at `deploy/pre_train/{g1,h1,h1_2}/motion.pt` (G1 = ~146 KB, 12-DOF legs-only); `deploy_mujoco` runs G1 walking forward at **50 Hz on CPU MuJoCo** (config `cmd_init [0.5,0,0]`). All three checkpoint URLs verified live.
- Agent bridge = a thin MuJoCo↔ROS 2 node publishing velocity commands (vx, vy, yaw-rate) from the agent.
- **Caveats (honest):** repo quiescent since 2025-07-25 (active dev moved to `unitree_rl_lab`); policy is legs-only — "turn head + gesture" means layering direct upper-body joint commands, which **no source demonstrated on top of this policy**; whether the legs policy tolerates upper-body perturbation without falling is an open experiment (§5 A6). Fallback: rl_sar's whole-body-tracking checkpoints, or train in MuJoCo Playground.

## 3. YOLO26 integration (license waived) — feasible-now (3-0 ×3)

- **Natively end-to-end NMS-free by default** (one-to-one head outputs `(N, 300, 6)` directly; `end2end: True` in shipped yaml) → simpler TensorRT/ROS 2 pipeline, no NMS plugin.
- Vendor table (verbatim-matched 2026-08-20, **self-published — treat as upper bounds**): n 40.9 mAP / 38.9 ms CPU ONNX / 1.7 ms T4 TRT · s 48.6/87.2/2.5 · m 53.1/220.0/4.7 · l 55.0/286.2/6.2 · x 57.5/525.8/11.8. Even YOLO26**x** ≈ 85 FPS on T4-class → L4 has comfortable real-time headroom alongside the rest of the stack. The "43% faster CPU" figure is real but narrow: 26n vs 11n, ONNX, one Xeon SKU, "up to".
- Tracking/CoreML/ROS-node specifics and YOLO-World/YOLOE open-vocabulary detection: extracted but **not among the verified 25** — standard ultralytics features, validate at build time.

## 4. Hands-on measurements (this machine, today — the evidence literature couldn't provide)

### 4.1 MacBook device access (user-authorized test)
| Device | Result |
|---|---|
| Camera | ✅ ffmpeg avfoundation captured a real 1280×720 frame (devices: MacBook Pro Camera, Desk View, screen-capture) |
| Speaker | ✅ `say` Chinese TTS played (rc=0) |
| Mic (device layer) | ✅ AVAudioEngine tap: full 3.00 s @ 48 kHz, live signal |
| Mic (via ffmpeg) | ❌ **ffmpeg 8.1.2 avfoundation drops ~90% of audio samples** (0.3 s of samples per 3 s wall-clock; reproduced sandboxed + unsandboxed, both device syntaxes) |

**Consequence (revises round-2 §5.4):** the C daemon must use **miniaudio/CoreAudio for audio** (as round 2 recommended) — ffmpeg avfoundation is confirmed camera-only. TCC camera+mic permissions are already granted to the terminal host; no blocking prompts.

### 4.2 Bedrock Haiku 4.5 streaming TTFT (3 trials, ConverseStream, us-east-1, from the MacBook)
- Measured: **894 / 1211 / 1890 ms → avg 1332 ms** time-to-first-token (~40-token replies arrive in 4 coarse chunks).
- Red-team impact: the round-2 latency budget guessed "few hundred ms" — **optimistic**. In the real topology the agent runs ON EC2 us-east-1 (removes ~trans-Pacific RTT from each call), but ≳0.5–1 s TTFT must be planned for. Mitigations: short cached system prompts (Bedrock prompt caching, 4,096-token min checkpoint), speculative TTS on first sentence, or the local MiniCPM-o path for the tight loop.
- Revised voice-loop estimate (cloud path): ASR 600 ms + Haiku TTFT ~0.5–1 s (EC2-local, **[to re-measure from EC2]**) + Kokoro ~300 ms → **≈1.5–2.5 s** speech-end→first-audio. Nova 2 Sonic (EN demo) remains ~1.09 s-class; MiniCPM-o full-duplex is the low-latency zh path.

### 4.3 VRAM budget on one L4 24 GB **[arithmetic from verified numbers, not load-tested]**
| Resident | Est. |
|---|---|
| MiniCPM-o 4.5 AWQ (verified 11 GB wt) / bf16 (19 GB) | 11–19 GB |
| YOLO26 TensorRT engine + buffers | ~1–2 GB [estimate] |
| Kokoro-82M | <1 GB [estimate] |
| MuJoCo | CPU (0 GB GPU) |
| **Total** | **~13–22 GB** |

Quantized MiniCPM-o → fits with margin; bf16 → tight, no headroom for GR00T. Dropping Gazebo's GPU use (MuJoCo is CPU) is what makes this work. GR00T N1.7 coexistence on the same L4: **not budgetable — separate instance or skip**.

---

## 5. Red-team analysis (critical-thinking skill: assumption map → attacks → cascades → hardening)

### 5.1 Assumption map
| ID | Assumption | Type | Criticality | Evidence | Status |
|---|---|---|---|---|---|
| A1 | A humanoid can walk in our sim without training | Factual | High | Strong — pre-trained motion.pt verified, 50 Hz CPU | ✅ but repo quiescent |
| A2 | Everything fits one L4 24 GB | Factual | High | Medium — arithmetic only (§4.3), never load-tested | ⚠️ TEST FIRST |
| A3 | Voice loop feels "conversational" (<2 s) | Causal | High | Weak — TTFT measured 0.9–1.9 s from laptop; EC2-local unmeasured | ⚠️ measure on EC2 |
| A4 | MiniCPM-o 4.5 full-duplex works outside demo videos | Factual | High | **None — zero field reports survived 3 rounds** | 🔴 most uncertain |
| A5 | MacBook peripherals reachable from our code | Factual | High | Strong — measured today (§4.1) | ✅ closed |
| A6 | Legs-only policy tolerates head/arm gesture overlay | Causal | Med | None — no source demonstrated | ⚠️ 1-day experiment |
| A7 | AGPL YOLO26 is safe because "demo only" | Normative | Med | Your call — but AGPL §13 triggers on network interaction; if the repo is public open-source anyway, obligations are largely met; NITROS §4f conflict is REAL and resolved by excluding NITROS | ✅ accepted risk |
| A8 | MacBook→EC2 audio transport is a solved problem | Factual | Med | None verified; WebSocket-PCM is trivially buildable [design] | ⚠️ prototype |
| A9 | Gazebo Jetty needed at all for the demo | Methodological | Low | Refuted-by-ecosystem for locomotion; still fine for sensor/world scenes | decision §6 |

### 5.2 Sharpest attacks & cascade failures
1. **A4 → cascade:** if MiniCPM-o full-duplex is demo-ware, the local zh voice loop falls back to the pipeline path (paraformer→agent→Kokoro), which A3 says is 1.5–2.5 s — demo feels laggy → mitigate with half-duplex push-to-talk UX or Nova Sonic EN-side.
2. **A2 → cascade:** bf16 MiniCPM-o (19 GB) + TensorRT engine burst → OOM kills the omni model mid-demo → run MiniCPM-o AWQ from day one; keep bf16 as an A/B.
3. **A1 quiescence:** unitree_rl_gym stops working against new MuJoCo → pin MuJoCo version; fallback = MuJoCo Playground G1JoystickFlatTerrain (Apache-2.0, pushed 2026-08-18).
4. **Single point of failure:** the ONE g6.xlarge hosts sim+vision+speech+agent. Acceptable for demo; not for anything more.

### 5.3 Hardening plan (ordered, cheapest de-risk first)
1. **Day 0 ($1):** boot g6.xlarge → load MiniCPM-o AWQ + YOLO26 TRT + Kokoro together → `nvidia-smi` (closes A2). Re-measure Haiku TTFT from EC2 (closes A3).
2. **Day 0 (local, $0):** MiniCPM-o 4.5 half-duplex on the MacBook (M-series) to sanity-check the model quality claims (partially closes A4).
3. **Day 1:** MuJoCo + G1 motion.pt walking + ROS 2 velocity-command bridge (proves A1 end-to-end); then perturb with head-joint sinusoid (closes A6).
4. **Day 2:** C daemon skeleton — miniaudio capture/playback + ffmpeg camera frame grab + WebSocket PCM uplink (closes A8, uses today's verified device access).

**Residual risk after hardening:** MiniCPM-o full-duplex quality (A4) remains the fattest tail; everything else has a tested fallback.

---

## 6. Decision needed from you
1. **Locomotion simulator: switch to MuJoCo** (recommended — zero-training walking now) and keep Gazebo Jetty only if/when you want rich sensor worlds, **or** stay Gazebo-pure and accept porting rl_sar to gz-sim (weeks, uncharted)?
2. GR00T N1.7: keep as documented stretch goal pending a read of its actual license text, or drop?

## 7. Refuted / unresolved this round
- ❌ (1-2) "GR00T weights are under NVIDIA Open Model License permitting commercial+non-commercial use" — unverified; read the license text.
- Zero literature claims survived on: ROS 2+LLM+voice prior art, MiniCPM-o field reports, Bedrock TTFT (→ now measured §4.2), Gazebo Lyrical package maturity, VRAM budgets (→ arithmetic §4.3). These are measurement problems, not search problems.

*Sources: 23 primary (HF model cards nvidia/GR00T-N1.7-3B & Cosmos-Reason2-2B, github isaac-sim/IsaacLab, NVIDIA-ISAAC-ROS/isaac_ros_nitros LICENSE, unitreerobotics/unitree_rl_gym & unitree_ros, fan-ziqi/rl_sar, google-deepmind/mujoco_playground, docs.ultralytics.com/models/yolo26). Vote records in workflow `wf_9a210b0c-01d`. Hands-on: this session, 2026-08-20.*
