# open-agent-robot

An open, hackable multimodal humanoid-robot agent. The robot's **brain and body**
run as ROS 2 nodes on one GPU instance in the cloud; your **laptop is its eye,
ear and mouth**; your **browser is its face**. Everything rides a single SSH
tunnel, and the reference use case — a voice-triggered game of
rock-paper-scissors with beat-synchronised throws — exercises the whole loop:
hearing, seeing, reasoning, speaking and moving, all under a hard real-time beat.

The software is meant to move to a physical robot later, and Round 6 pins down
what that actually takes. The sim/real seam is a **DDS topic contract, not a node
swap**: `unitree_mujoco` publishes `rt/lowstate` and subscribes `rt/lowcmd`
byte-identically to a real Unitree G1, so one `hardware_interface::SystemInterface`
serves both and the switch collapses to two xacro parameters. The body layer does
not sit at that seam yet — it publishes `/joint_states` at the render tick, which
is both the wrong *direction* (`/joint_states` is by convention the driver's
feedback topic, so it would collide with a real G1 driver) and 4–43× too slow.
See [`docs/research-round6-simreal-joint-mapping-2026-08-22.md`](docs/research-round6-simreal-joint-mapping-2026-08-22.md).

> Status: working end-to-end demo (Mandarin / English / Cantonese speech,
> open-vocabulary scene description, RPS game, storytelling, interruptible
> speech). Research-grade, not a product. See `docs/` for the design record.

## Why it looks like this

Three constraints shaped the architecture, and they are worth stating because
they explain choices that otherwise look odd.

**1. A game beat cannot wait for a language model.** Measured first-token
latency from the instance to Bedrock is ~0.9 s at p50 — fine for banter, fatal
for "rock, paper, scissors — throw!". So the system is explicitly split:
a *reflex layer* (System 1) owns beat timing, stop words and barge-in, and it
**never** calls the LLM; a *deliberation layer* (System 2) owns conversation,
scene understanding and gesture choice. When a user shouts "stop", a
deterministic keyword match in `game_node` stops the game — not a model that
was asked nicely.

**2. Laptop hardware is the honest interface.** A robot that can't hear a real
room isn't interesting. A small C daemon owns the microphone, speaker and
camera directly (miniaudio for audio, libavdevice for the camera), because the
audio path is where latency and glitches actually live. It streams 20 ms PCM
chunks and MJPEG frames over one WebSocket with an 9-byte header
(`[1B type][8B capture timestamp µs]`).

**3. Speech must be interruptible.** TTS is synthesised and shipped
**sentence by sentence**, so the first sentence is audible in ~0.6 s even for a
long story, and a user talking over the robot triggers a flush that stops the
audio mid-sentence. That same kill switch is what makes an asynchronous output
safety check affordable later (see `docs/design-critique-memory-safety-*.md`).

## Architecture

```
MacBook                                  │  EC2 g4dn.xlarge (T4)  — one SSH tunnel
─────────────────────────────────────────┼──────────────────────────────────────────
daemon/  (C)                             │  gateway/  ws_gateway.py   (WS ⇄ ROS 2)
  mic 16 kHz ──┐                         │      │
  camera ──────┼── WebSocket /stream ────┼──────┤  speech/   asr_node   (FunASR streaming)
  speaker ◀────┘                         │      ├─          tts_node   (Kokoro zh/en +
                                         │      │                       CosyVoice yue)
web/robot-view/  (browser)               │      ├─ perception/ hand_node   (MediaPipe)
  MJPEG body view + event HUD ◀──────────┼──────┤             detect_node (YOLO26 TRT)
                                         │      ├─ agent/    agent_node (Bedrock Converse)
                                         │      ├─ game/     game_node  (reflex layer)
                                         │      └─ body/     body_node  (MuJoCo + MJPEG)
```

| Layer | Component | What it is |
|---|---|---|
| Hearing | `speech/asr_node.py` | FunASR `paraformer-zh-streaming`, 600 ms chunks, utterance accumulation, half-duplex gate so the robot never transcribes itself |
| Speech | `speech/tts_node.py` | Kokoro-82M (Mandarin + English) sentence-streamed; Cantonese routed to a CosyVoice microservice (`speech/yue_server.py`) by marker characters |
| Sight | `perception/detect_node.py` | YOLO26s compiled to TensorRT (161 FPS measured on the T4), rate-gated to 5 fps, aggregated over a 1 s window |
| Hands | `perception/hand_node.py` | MediaPipe Tasks HandLandmarker; the RPS classifier deliberately ignores the thumb (people hold it differently) |
| Reasoning | `agent/agent_node.py` | Bedrock Converse, Claude Sonnet 5 global inference profile, one downscaled camera frame attached per turn for open-vocabulary scene questions |
| Skills | `skills/loader.py` | Three-level loading: a name + one sentence resident per skill (96 measured input tokens), the body attached to the turn when a **frozen phrase table** — never the model — matches, deeper files pulled through a `read_skill_file` tool confined to `skills/` |
| Reflexes | `game/game_node.py`, `safety/guard.py` | Robot-led chant, resolve window `[T−0.1 s, T+0.6 s]`, stop words honoured in **any** state — and that is a property of the signature rather than of a test matrix: `guard.stop_requested(text)` takes the utterance and nothing else, so it *cannot* consult the state, and the node is checked structurally for dispatching it before it reads one. Per-game cooldown. The model is consulted **once, when the arm is raised**; the beat itself only plays a move that has already been decided, and the suite proves that structurally by deriving the call graph from the source (the beat path cannot reach the model at all) |
| Opponent model | `game/opponent.py` | Order-2..4 conditional transition tables over the **user's own** (move, outcome) history — never over the robot's moves, which would hand a counter-learner a lever on the context it is predicted from. Exploited only while an **anytime-valid e-value gate** (Ville, δ = 0.05 for the whole lifetime of one opponent) says this person has demonstrably left uniform; otherwise a fair 1/3 coin. Wired into `game_node`: each resolved round is published fire-and-forget to `/game/rps_round` so that `memory_node` stays the only process that writes to disk, and the accumulated table comes back on a **latched** topic at startup — a malformed restore is rejected whole rather than merged. `memory_node` is the next stage and does not exist yet, so today nothing publishes that topic: the table starts empty every boot, which means the first ~120 rounds fall back to the 1/3 coin (81 order-2 contexts cannot be filled by a 20-round session — that is arithmetic, not a tuning choice) |
| Constitution | `safety/constitution.md`, `safety/guard.py` | Ten numbered clauses that ship as **runtime data, not documentation**: the loader refuses to start if any registered clause id is missing from the file, because a deleted boundary is not a documentation problem. The system prompt is assembled by two functions and concatenated once — instructions (base + constitution) first, then fenced data blocks — and `instruction_region()` **rejects** any argument carrying a fence marker or the data preamble, so recalled text cannot be smuggled into the region the model reads as rules. Caps: 600 spoken characters, 60 turns, 3600 seconds, gestures from the one whitelist in `body/gestures.py` |
| Memory writer | `memory/store.py` | The single-writer append-only log behind `memory_node`. One writer thread serves both the `events` and `audit` streams — same retention period, same cascade-delete path, so they cannot disagree. Records are numbered inside the lock and published as a **high-water mark** (a batch can be enqueued out of order), a record that fails to serialise is rejected *without* consuming a sequence number, snapshots are replaced by `os.replace` so a reader holding the old file keeps reading intact bytes, and `flush()` raises rather than waiting forever when the writer thread has died |
| Body | `body/body_node.py` | MuJoCo kinematic puppet of a humanoid with three-fingered Dex3-1 hands (thumb 3 DOF, index 2, middle 2 — 7 per hand), streamed to the browser as MJPEG |
| Transport | `daemon/main.c`, `gateway/ws_gateway.py` | One WebSocket, binary framed; audio prioritised over video, both flow-controlled |

## Quick start

Prerequisites: an AWS account with Bedrock model access, the AWS CLI logged in,
macOS with Homebrew, and `brew install ffmpeg libwebsockets`.

```bash
# 1. Provision a locked-down GPU instance (SSH-only, your IP only, IMDSv2,
#    encrypted root, instance role scoped to Bedrock invoke). Idempotent.
infra/provision.sh            # writes infra/ssh_config + infra/local.env (git-ignored)

# 2. Install the stack on the instance. Three stages, run on the instance itself:
#      ./setup-ec2.sh driver   NVIDIA driver, then reboot
#      ./setup-ec2.sh stack    ROS 2 Jazzy + Python env + the G1 body model
#      ./setup-ec2.sh yue      Cantonese TTS venv (skip if you don't need Cantonese)
infra/setup-ec2.sh            # see the script header

# 3. Start the robot's brain and body (every node in the table, `yue` included)
ssh -F infra/ssh_config robot 'open-agent-robot/infra/run_ec2.sh start-all'

# 4. Open the tunnel, build and run the daemon, open the face
ssh -F infra/ssh_config -N -f robot
cd daemon && make && ./oar-daemon
open web/robot-view/index.html
```

Then say「我們玩石頭剪刀布」or "let's play rock paper scissors".
Full runbook with troubleshooting: [`docs/DEMO.md`](docs/DEMO.md).

## Repository map

| Path | Contents |
|---|---|
| `daemon/` | macOS C daemon (audio, camera, WebSocket client). `make` fetches `miniaudio.h`. |
| `speech/`, `perception/`, `agent/`, `game/`, `body/`, `gateway/` | ROS 2 nodes, one concern each |
| `web/robot-view/` | Single-page face: MJPEG body view + event HUD |
| `skills/` | Markdown skills plus the loader that gates them. See [`skills/README.md`](skills/README.md) |
| `safety/`, `memory/` | The constitution and the single-writer store — pure modules, zero ROS, so their tests run on a laptop |
| `tests/` | Offline suite, 566 cases: no ROS, no GPU, no network, no Bedrock. `pytest` runs it on a laptop |
| `infra/` | Provisioning, instance setup, process supervision (`run_ec2.sh` uses pid files, never `pkill` patterns) |
| `docs/` | Design record: seven research rounds (raw evidence under `docs/evidence/`), the day-0 latency measurements, the demo runbook, the memory/safety design critique, and the [test contract](docs/test-design.md) |

## Tests

```bash
pytest -m "not mutation"   # per-save loop
pytest                     # per-PR: adds the named mutants
pytest -m mutation         # the mutants alone (107 cases on 2026-08-23)
infra/check_deploy.sh      # is every node in the repo actually started by the runner?
```

Two properties are worth knowing before reading the suite:

**Nothing here uses a model to judge a test.** Judge-gated pipelines move the
*measured* pass rate 0.72 → 0.94 while real correctness stays flat at 0.20, and a
strict three-judge ensemble still accepts 55% of wrong answers. So anything needing
semantic judgement is either reduced to a machine-checkable proxy or moved out of CI
into human acceptance testing.

**Every assertion is paired with a named mutant it must kill** — a mutant that
produces byte-identical output means the assertion measures the wrong quantity, not
that the code is safe. Each mutation round also runs a no-mutant control that must
pass, because a broken harness reports success too. The word *every* in that
sentence is itself checked: a meta test derives what a mutant applier is (it rewrites
source text **and** turns the result back into a program) and fails if any test that
calls one is missing the `mutation` marker. It found two modules whose mutants
`pytest -m mutation` had never been running. `infra/check_deploy.sh`
distinguishes three exit codes for the same reason: 0 clean, 1 findings, **3 the
guard could not run** — a guard that cannot run must never report clean.

The full contract, including the items still unimplemented, is
[`docs/test-design.md`](docs/test-design.md).

**Three of that contract's own pass criteria were wrong, and the suite says so.** The
opponent model was supposed to fail its uniform-opponent case once the confidence
gate was removed — but rock-paper-scissors is zero-sum and symmetric, so against a
uniform opponent *every* policy scores exactly 0 in expectation. That criterion was
unachievable in principle, not unimplemented; the discriminating assertion is the
Ville bound instead (the gate ever opens on 3/100 seeds gated, 100/100 ungated). The
second criterion predicted the gate would stay shut against an adaptive
counter-learner. It is open on 71% of rounds and the model *wins* — because the loop
is closed: the robot's moves are a function of the user's own history, so an opponent
reacting to the robot is thereby reacting to that history. The third asked for the
game beat's p95 jitter to stay within 5 ms of its current value, which is a timing
assertion, and a test that goes red because a laptop was busy earns itself a `skip`
within two days. It was replaced by a **structural** oracle that gives the same answer
under any load: derive the method-level call graph from `game/game_node.py`, take the
closure reachable from the beat entry point, intersect it with the methods that touch
the model, and assert the intersection is empty. The jitter is still measured, just
never asserted — `docs/evidence/track-b/arm-cost.json` records p50 0.0082 ms / p95
0.0113 ms / max 0.1137 ms over 2000 calls on a table warmed by 300 real rounds, i.e.
0.57% of one 20 ms tick, on a path the tick never takes. All three corrections, with
the measurements that forced them, are in §2.H of the contract.

That structural gate is also the clearest example of why every assertion needs a
mutant. Its first version intersected the call-graph closure with a **written-down**
list of the methods allowed to touch the model — and the mutant that puts
`self.opponent.predict()` *inline* in the beat callback walked straight through it,
because inlining creates no new *named* toucher. Both sides of the intersection have
to be derived. Nothing about that was visible in review; the mutant found it.

## Design record

The `docs/` directory is the interesting part of this repository. It keeps the
research and the argument, not just the conclusion — including the findings that
went **against** intuition:

- [`docs/day0-measurements.md`](docs/day0-measurements.md) — the latency numbers the architecture is built on
- [`docs/research-multimodal-robot-agent-stack-2026-08-20.md`](docs/research-multimodal-robot-agent-stack-2026-08-20.md) — stack survey (ASR/TTS/vision/sim), with licences
- [`docs/research-round3-feasibility-2026-08-20.md`](docs/research-round3-feasibility-2026-08-20.md) — feasibility and hardware budget
- [`docs/research-round4-agent-brain-2026-08-21.md`](docs/research-round4-agent-brain-2026-08-21.md) — memory / skills / MCP-OAuth / critical thinking. Notable: **self-critique without external feedback measurably lowers accuracy**, so the critical-thinking skill is trigger-gated and only ever fed real external evidence.
- [`docs/design-critique-memory-safety-2026-08-22.md`](docs/design-critique-memory-safety-2026-08-22.md) — a red-team review of our own memory design, an episodic-memory schema, a forgetting policy, and the layered moral-boundary safety architecture. Core design law: **the safety of an LLM memory system comes from restricting the schema, not from filtering text.**
- [`docs/research-round5-memory-safety-2026-08-22.md`](docs/research-round5-memory-safety-2026-08-22.md) — the feasibility round that answered the critique's open questions with verified numbers: consolidation costs, retrieval latency budgets, and the published evidence for keeping hard boundaries outside the model. Two decisions were reversed by evidence — a local guard model is ruled out on **language coverage** rather than VRAM, and the salience formula drops voice-prosody signals because inferring emotion from biometric data is high-risk under the EU AI Act.
- [`docs/research-round6-simreal-joint-mapping-2026-08-22.md`](docs/research-round6-simreal-joint-mapping-2026-08-22.md) — can the MuJoCo puppet actually drive a physical humanoid one-to-one? 88 adversarially verified verdicts across 7 claim families. Notable: **the sim/real seam is a DDS topic contract, not a plugin**; muscle/tendon actuation is a closed door (zero purchasable units with an open interface, and the one team that tried to commercialise it shipped geared joint-space actuators instead); and **planning ability is decoupled from safety** — the best planner fails to plan on 0.4% of tasks but produces *dangerous* plans on 28.3%, so ordinary success monitoring is blind to the dominant failure mode.
- [`docs/research-round7-rl-and-two-tier-2026-08-22.md`](docs/research-round7-rl-and-two-tier-2026-08-22.md) — should this agent use reinforcement learning, and should a small local model share the work with the remote one? Two parallel research workflows, 213 agent calls, 245 claims recovered (raw evidence in [`docs/evidence/round7/`](docs/evidence/round7/)). Both answers are **no, for derived reasons rather than enumerated ones**: RL is admissible exactly where **the reward is measured by a non-LLM sensor**, which rules out every intervention point we had in mind; and the two-tier idea survives **only as latency masking**, not as routing — a router needs a mechanism cheaper than the big model that reliably knows "I don't know", and that mechanism is refuted along the whole chain (a routing plateau of +2.13 pp for 10× the data, an entropy router that escalated on 0.0% of TextWorld episodes, zero certifiable model pairs at a 10% risk bound). Notable and actionable **today**, independent of RL: asking a model "are you sure?" drops accuracy by up to **27%** and makes it wrongly admit a mistake 42–98% of the time, with correct→incorrect flips outnumbering the reverse; and hard schema-constrained decoding can be **100% schema-valid while executable accuracy falls 91.5% → 48.0%**, which structural validators catch 0% of — so reason free, constrain late. This round also corrects two of our own earlier claims (§8).

## Security posture

- Only port 22 is open on the instance, restricted to the operator's current IP; everything else is tunnelled. IMDSv2 required, encrypted root volume, instance role limited to `bedrock:InvokeModel`.
- Account-specific values (instance id, public IP, key path) are generated into git-ignored files by `provision.sh`. Nothing account-specific is tracked.
- Raw microphone audio and camera frames are never written to disk by default. One downscaled frame per conversational turn is sent to Bedrock — a third-party transfer that any deployment must disclose to its users.

## Licence

Our code is MIT (see [`LICENSE`](LICENSE)).

**One dependency is not**: the YOLO26 detector is used through Ultralytics,
which is **AGPL-3.0**. This repository is a research demo. Note that
`infra/run_ec2.sh start-all` *does* start the detection node — the licence
argument is isolation, not opt-in: `perception/detect_node.py` publishes a plain
`/detections` JSON topic and touches nothing else, so it can be replaced or
dropped on its own. If you intend anything commercial, either obtain an
Ultralytics licence or swap that one node for a permissively licensed detector.

Model weights carry their own licences (Kokoro, CosyVoice, FunASR, MediaPipe):
check each before redistributing. Vendored skills keep their own licence file —
`skills/critical-thinking/` is MIT.

繁體中文版:[README.zh-TW.md](README.zh-TW.md)
