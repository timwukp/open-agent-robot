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

The same picture drawn to scale, with the packets moving:
[`docs/architecture.svg`](docs/architecture.svg) — every node, topic and gap in it is
generated from `infra/ros_census.py`, which reads the launcher rather than a list, so a
node that stops being started stops being drawn. The animation uses CSS `offset-path`
(Chrome, Safari 16+, Firefox 122+); an older renderer degrades to the same diagram
without moving packets, which loses nothing but the motion. For the *sequences* — one
turn ear to mouth, stopping a voice already in the air, what `forget what I said` costs —
see [`docs/workflow.md`](docs/workflow.md), whose ten Mermaid charts are checked against
that same census in both directions.

| Layer | Component | What it is |
|---|---|---|
| Hearing | `speech/asr_node.py` | FunASR `paraformer-zh-streaming`, 600 ms chunks, utterance accumulation, half-duplex gate so the robot never transcribes itself |
| Speech | `speech/tts_node.py` | Kokoro-82M (Mandarin + English) sentence-streamed; Cantonese routed to a CosyVoice microservice (`speech/yue_server.py`) by marker characters |
| Sight | `perception/detect_node.py` | YOLO26s compiled to TensorRT (161 FPS measured on the T4), rate-gated to 5 fps, aggregated over a 1 s window |
| Hands | `perception/hand_node.py` | MediaPipe Tasks HandLandmarker; the RPS classifier deliberately ignores the thumb (people hold it differently) |
| Reasoning | `agent/agent_node.py` | Bedrock Converse, Claude Sonnet 5 global inference profile, one downscaled camera frame attached per turn for open-vocabulary scene questions |
| Skills | `skills/loader.py` | Three-level loading: a name + one sentence resident per skill (96 measured input tokens), the body attached to the turn when a **frozen phrase table** — never the model — matches, deeper files pulled through the `read_skill_file` tool, whose executor and path fence live here and whose registration lives in `agent/tools.py` — the split matters, because while this module owned the tool spec, "which tools can the model reach" was answered by the module that knows about skills, and a second tool would have been appended next to the first one |
| Tool boundary | `agent/tools.py` | The closed registry of everything the model may invoke, written before there was a second tool to cross it. Today it declares exactly one — `read_skill_file`, local and read-only — and that is the honest state of the system: the robot can read its own skill files and do nothing else. The registry exists now rather than alongside the first MCP client because an MCP client answers "which tools can the model reach" by asking a server at runtime, and at that point the offline suite structurally loses the ability to assert the boundary. So discovery may **narrow** the registry and never widen it: a tool a server offers and this repo does not declare is a finding, and `invoke` returns `unknown_tool` for it even when it is in the live offer set. Both sides of the dispatch gate are derived — executors are module functions named `_exec_<tool>` — because a second dict mapping names to functions is a reader that goes stale, and the stale reader here is a function the model can call that nothing declared. One invocation has **six** outcomes, and exactly one of them (`refused`) licenses the sentence "the boundary held": the previous single failure string described a name the model invented, a server that no longer offers the tool, a transport that failed and a policy that said no with the same line, which is how a network outage gets recorded as protection. Effect class is declared per tool and `MAX_EFFECT_HOPS` is **0**, separately from the three read hops, because three file reads and three irreversible actions are the same integer describing two different exposures |
| Reflexes | `game/game_node.py`, `safety/guard.py` | Robot-led chant, resolve window `[T−0.1 s, T+0.6 s]`, stop words honoured in **any** state — and that is a property of the signature rather than of a test matrix: `guard.stop_requested(text)` takes the utterance and nothing else, so it *cannot* consult the state, and the node is checked structurally for dispatching it before it reads one. Per-game cooldown. The model is consulted **once, when the arm is raised**; the beat itself only plays a move that has already been decided, and the suite proves that structurally by deriving the call graph from the source (the beat path cannot reach the model at all) |
| Opponent model | `game/opponent.py` | Order-2..4 conditional transition tables over the **user's own** (move, outcome) history — never over the robot's moves, which would hand a counter-learner a lever on the context it is predicted from. Exploited only while an **anytime-valid e-value gate** (Ville, δ = 0.05 for the whole lifetime of one opponent) says this person has demonstrably left uniform; otherwise a fair 1/3 coin. Wired into `game_node`: each resolved round is published fire-and-forget to `/game/rps_round` so that `memory_node` stays the only process that writes to disk, and the accumulated table comes back on a **latched** topic at startup — a malformed restore is rejected whole rather than merged. `memory_node` exists as of S3 but only publishes the profile card — it does not subscribe to `/game/rps_round` yet, so today nothing restores that table: it starts empty every boot, which means the first ~120 rounds fall back to the 1/3 coin (81 order-2 contexts cannot be filled by a 20-round session — that is arithmetic, not a tuning choice) |
| Constitution | `safety/constitution.md`, `safety/guard.py` | Ten numbered clauses that ship as **runtime data, not documentation**: the loader refuses to start if any registered clause id is missing from the file, because a deleted boundary is not a documentation problem. The system prompt is assembled by two functions and concatenated once — instructions (base + constitution) first, then fenced data blocks — and `instruction_region()` **rejects** any argument carrying a fence marker or the data preamble, so recalled text cannot be smuggled into the region the model reads as rules. Caps: 600 spoken characters, 60 turns, 3600 seconds, gestures from the one whitelist in `body/gestures.py` |
| System prompt | `agent/context.py` | The one place a prompt is assembled, and it exists because the seam above was correct and **not deployed**: an AST census of `safety/guard.py`'s fourteen public names against every production file found **thirteen with no caller outside the tests**, and `agent_node` did not import the module at all — its prompt was a 725-character literal plus the skills index, with the constitution nowhere in it. A boundary with no caller is not a boundary, and a green suite does not mention that; a caller census derived from the production files does. The node now loads the constitution **fail-closed and deliberately outside any `try`** (a malformed skill directory costs a capability, so it degrades; a constitution that will not load means every boundary clause is absent) and takes its entire prompt from here. Which region a value lands in is decided by the **parameter it arrives through**: the constitution and the skills index are instruction material, the profile card and the retrieval result are data, and a caller holding a recalled hit has no argument that reaches the rules. Block names come from a closed tuple, and `data_region` now **refuses** two names that sanitise to one fence name rather than renaming one, because two blocks with one name leave the reader unable to say where either ends. The card block is the resident card **unioned** with what the card path recalled — they disagree in normal operation, since one arrives on a latched topic and the other is read off `profile.json`, so taking either side alone drops a fact with nothing saying one was withheld. A degraded retrieval path is reported to the audit log, never described to the model. Zero ROS, zero boto3, zero clock, zero filesystem: the prompt the robot sends is a thing a laptop can rebuild and compare byte for byte |
| Output guard | `safety/output_guard.py`, `safety/output_guard_harness.py` | The sentence-level check on what the robot is about to **say**, and it is asynchronous by construction rather than by convention: `offer()` is annotated `-> None` and a test asserts its body contains no value-returning `return`, so a caller has nowhere to wait and a serial guard is unrepresentable in the API. A verdict arrives later through `verdict(uid, index, …)`, keyed by the span the *speaker* named — never a sentence the guard split, because sentence boundaries already have two owners (Kokoro's pipeline and the Cantonese marker split) and a third rule would interrupt the wrong sentence. What a verdict may claim depends on how much was heard, so audibility is four values rather than a boolean: the verdict beat the first chunk, audio started and the interrupt cut it (the normal case, given a measured 921 ms worst-case classifier), playback finished so the guard blocked nothing, or the speaker dropped the sentence itself — and that last one must be *told* to the guard, since from inside it "no audio yet" and "no audio ever" are the same absence with opposite meanings. An interrupt is emitted only when something is left to interrupt, and the audit record says `blocked` only when one actually went out; folding a fully-played sentence into "blocked" is what makes a report show full marks while the guard does nothing. A classifier failure is `errored`, not allowed. The 2.0 s deadline is 2.2× our own measured worst case rather than a chosen round number, and a test derives it from the evidence JSON |
| Red-team set | `tests/fixtures/redteam.jsonl` | 50 frozen candidate **outputs** the robot must not say — 38 Mandarin, 12 Cantonese, seven categories — graded offline by three deliberately lying classifiers so every exit code is driven without a network. Two things it refuses to do: pool the categories, because `identity_denial` and `boundary_override` have **no output-side enforcer at all** (the content filters have no such category and `PROMPT_ATTACK` is input-side only) and averaging them into five enforced ones hides a gap with other categories' marks; and compute a Cantonese block rate, because the vendor does not list the language, so a blocked Cantonese line is one observation about one string. The harness has four exit codes and **0 is unreachable today** — a test asserts that against the real frozen set, because 0 is the number a reader quotes. Per rule 3, zero escapes over a frozen set is a lower bound on exposure and never a guarantee; the set only grows |
| Profile card | `memory/profile.py`, `memory/memory_node.py` | ~17 hand-written facts under a **closed predicate set**, published once on `/memory/profile` with `transient_local` durability so a node that starts later still gets it without a round trip. It is a **read-only seed until S5**, and the lock lives in the single writer (`store.PROTECTED_SNAPSHOTS`) rather than in this module: the write API is `Store.write_snapshot`, so a reader that merely declines to call it constrains only the callers that went through it. `unlock_for_s5()` is the one named way past. The reason for the lock is provenance — a semantic fact must name the episodes it came from so that deleting an episode deletes what it produced, and there are no episodes until S4; a card written now would hold facts no deletion could reach. The 350-token resident budget is enforced at boot on the real card, not just asserted against the shipped example, and an over-budget card fails the node instead of warning: it is paid once per turn for the life of the process |
| Salience | `memory/salience.py` | How much a turn is worth remembering: `0.4·stated evaluation + 0.3·novelty + 0.2·asked-to-remember + 0.1·length`, all four computed from **text and intent only**. Pitch, loudness and speaking rate are behavioural characteristics under EU AI Act Art 3(34), and Art 3(40)'s narrow gate covers categorisation rather than emotion recognition, so a prosody term would make every remembered turn the output of biometric inference. The term exists but ships **off** — a term no code path can reach is a term no test can prove is excluded, so the switch-*on* run is the positive control that proves the acoustic fields were readable in the first place. The 0.4 term asks *whether* an evaluation was stated, never its polarity: 「我討厭芫茜」 is exactly as worth remembering as 「我最愛芫茜」, and not needing polarity keeps a sentiment model out of the system. Landing before S4 is the point — a stored score is only comparable to one from the same formula, so `FORMULA` travels with every value and `Store.append` refuses a score that arrives without it |
| Episode segmentation | `memory/segment.py` | Where one episode ends and the next begins, deterministically: a 45 s silence, an activity transition, a scene change, or a hard cap of 20 turns / 5 minutes. There is no clock inside the module — `tick(now)` takes the time as an argument, which is the strongest form of an injectable clock because there is no default to forget. It needs that timer because `asr_node` publishes nothing while nobody is talking, so silence is not an event anyone can subscribe to; a gap-driven segmenter would cut at the same index but only notice when the person next speaks, deferring the whole evening's consolidation to a moment that has nothing to do with it. The turn cap closes *after* appending the 20th turn and the span cap closes *before* the event that would exceed it — written the other way round, the rule that exists to bound an episode at 5 minutes produces one spanning 5 minutes 20 seconds. Every boundary records *why* it fired, because "the person paused" and "we hit a ceiling" call for different behaviour in S5 |
| Reason for every record | `memory/store.py` (`GOALS`, `check_why`) | Every `events` record must say **why** it exists, and the vocabulary is a closed set of 11 written-out members in four families (`user_requested_*`, `game_*`, `vision_event`, `scheduled_*`). The families are documentation; the members are the contract, and that distinction is the whole point: implemented as a prefix test, `user_requested_ignore_stop_words` is accepted and the field becomes exactly the hole it was added to close — free text returns as the suffix. A prefix is the right shape for a *scanner*, whose failure mode is not noticing a new name, and the wrong shape for a *gate*, whose failure mode is admitting one. The field is both a retrieval key and a safety boundary, so unknown values are refused rather than normalised: `.lower()` or `.strip()` would still reject the instruction-shaped strings while writing the caller's spelling to disk, leaving a record that cannot be found by a query for its own goal. A query for an unknown goal raises instead of returning `[]`, and the census includes the zeros, because "no record carried this goal" and "this goal is no longer in the set" must not print the same thing. `audit` is exempt from *needing* a reason — forcing one would make callers invent a member — but a reason supplied anyway is still validated. Mandatory now rather than in S5, the stage that reads it: today nothing in production writes to `events`, so the requirement costs zero call sites; after S5 it would be a migration over personal data that cannot be re-derived |
| What may become a fact | `memory/consolidate.py` | The gate between "the robot heard something" and "the profile card says so". A candidate fact needs evidence from **N ≥ 2 distinct episodes** — occasions, not mentions, because someone who says 芫茜 four times in one sentence has said it once, and counting mentions would let an extractor that emits one candidate per sentence promote its own guess. The rule is not there to catch a lie; it catches a mishearing, a hypothetical (「如果我討厭芫茜的話」), and a sentence quoting someone else. Preferences and limits (`likes`, `dislikes`, `allergy`, `boundary`, `pronouns`, `name`) refuse **inferred** evidence entirely, and inferred evidence cannot top up a single self-report; what a detector may legitimately observe about the room (`device`, `location`, `household`, `routine`, `pet`, `language`) may be inferred, and the table writes out all 18 predicates rather than defaulting. Trust is a **kind** of evidence rather than an amount: `confirmed` requires that the person said yes, and never arrives by repetition, because enough repetitions of the same mishearing would otherwise become a confirmed fact. Values match byte-exactly and a fuzzy matcher is deliberately absent — a wrong merge does not just store a wrong value, it manufactures the second occasion the rule demands, and a gate whose evidence can be synthesised by the thing it gates is not a gate. Every accepted fact carries `source_episode_ids`, sorted and unique, and every id must resolve; an id is derived from the episode's **first store sequence number**, never its position, because after an earlier episode is deleted a position-derived id resolves *successfully* to the wrong episode while a sequence-derived one dangles — and only a dangling reference can be found and acted on by the deletion path S5 builds next |
| Retrieval | `memory/recall.py` | Five paths over the memory on disk — the profile card, a structured `why.goal` filter, BM25 over the event log, cosine over the episode embeddings, and the RPS table — under a **150 ms deadline**. Each path returns a **status** in `(ran, no_artifact, no_encoder, deadline)`, because an empty result has four different causes and only one of them supports the sentence "nothing matched": a missing encoder, a missing artifact and an exhausted deadline are all *paths that never ran*, and they are exactly the mutant the cascade-delete test exists to kill. So "zero hits for the deleted episode" is only assertable together with `ran`. The path set is **derived, not listed**: a path can only exist as a `_run_<name>` method, and a test compares that set against the registry in both directions, so a new path with no registration and a registration with no method both go red; each path declares the files it reads and that declaration is joined against the artifacts actually on disk, because the write API accepts any plain file name. A hit that names no episode is split in two — a hand-written **seed** fact is outside the cascade by construction, an **orphan** should have named an episode and did not, and merging them into one `None` would hide the cheapest cascade cheat: deleting an episode's entry while leaving its events makes every reference to it point at nothing, so an attribution check reads zero hits while the text stays retrievable. A card fact reports **one hit per source episode**, not just the first, or deleting the second source would be told the card is clean while the row still cites it. Scores are **never fused across paths**: BM25 is unbounded and cosine is ≤ 1.0, so a single list ordered by score is ordered by a quantity that does not exist and whose calibration has never been measured — ranking is within a path, and `k` applies per path. There is no in-process cache, because an index that survives a deletion *is* the mutant |
| Ranking ablation | `memory/rank.py`, `memory/rank_harness.py` | Three ordering policies over a retrieval result — **decay**, **importance pruning**, and **reorder-without-deleting** — so a preference between them can be argued from a comparison rather than asserted. An arm is a policy *over* a `Result` and never an edit to `Hit.score`: scores are not comparable across paths, and `Hit` is immutable, so the type system enforces it. The load-bearing rule is that **absent data is a tier, not a default**. An episode with no `end_ts` defaulted to *now* becomes the freshest thing in the store; defaulted to zero it falls past the decay floor and is deleted from the answer. So nothing is defaulted: six freshness statuses (`dated`, `undatable`, `future_ts`, `no_episode_by_design`, `lost_attribution`, `no_entry`) and five importance statuses (`scored`, `no_salience`, `other_formula`, `unscorable`, `no_record`), each split into a comparable tier that may be reordered or dropped and a non-comparable tier left in the retriever's own order. `other_formula` is deliberately not `no_salience`: one needs a re-score, the other needs a decision about which formula is canonical, and comparing two formulas' numbers is a thing the ranker has to *notice* rather than average. What the suite cannot show is which arm is better — that needs the 30 frozen queries of T-F1, one day of real logs and one human annotation pass, and neither may be fabricated. So `quality` is the status `NO_LABELS` rather than a number, asking for a labelled score raises, and the harness has **four exit codes** instead of a boolean: 0 concluded (unreachable today and it must stay that way), 1 malformed, 2 well-formed and still unable to conclude — **the expected state** — and 3 could not run at all. Three arms agreeing perfectly is what an empty run looks like from outside, so zero queries and zero hits are both counted and both refuse. Decay is a function of elapsed time, so the clock is injected and recorded in the JSON: the same store replays byte-for-byte |
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
| `safety/`, `memory/`, `agent/context.py`, `agent/tools.py` | The constitution, the single-writer store, the salience formula, the episode segmenter, the nightly consolidation, the five-path retriever, the deletion path, the prompt assembler, the asynchronous output guard and the tool registry — pure modules, zero ROS, so their tests run on a laptop |
| `tests/` | Offline suite, 1586 cases: no ROS, no GPU, no network, no Bedrock. `pytest` runs it on a laptop |
| `infra/` | Provisioning, instance setup, process supervision (`run_ec2.sh` uses pid files, never `pkill` patterns) |
| `docs/` | Design record: seven research rounds (raw evidence under `docs/evidence/`), the day-0 latency measurements, the demo runbook, the memory/safety design critique, and the [test contract](docs/test-design.md) |

## Tests

```bash
pytest -m "not mutation"   # per-save loop
pytest                     # per-PR: adds the named mutants
pytest -m mutation         # the mutants alone (333 cases on 2026-08-26)
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

**Start here, and only here, for "what is the design":** [`docs/design.md`](docs/design.md)
is the single authoritative statement of it — the five promises and which layer each one
actually reaches, the System 1 / System 2 split and the measurement that forced it, the
closed list of invariants with the test that goes red for each, and what this project
deliberately does not build. That last section separates **refuted**, **deferred with named
preconditions** and **not evaluated**, because a document that writes all three as "we don't
do X" tells the next reader the evidence is settled when it is not. The schedule is
[`docs/plan.md`](docs/plan.md); the test contract is
[`docs/test-design.md`](docs/test-design.md). Which document supersedes or partly overturns
which is recorded in [`docs/INDEX.md`](docs/INDEX.md) — one row per document under `docs/`,
with a status from a closed set and a named reason, checked against the directory in both
directions by `infra/doc_status.py` so nothing can join `docs/` unlisted.

The rest of `docs/` is the **evidence**, kept verbatim rather than summarised into the
conclusion — including the findings that went **against** intuition. Read a round for how a
number was obtained, not for what the design currently is; some of these recommendations
have since been overturned, and `docs/design.md` §4.4 names the citations that must no
longer be quoted at all:

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
