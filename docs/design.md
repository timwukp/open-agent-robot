# open-agent-robot — the current design

This file is the **single authoritative statement of what this system is meant to be
today**. Before it existed, the answer lived in `~/.claude/plans/` on one laptop plus
six research documents and one design critique inside `docs/` — **191 KiB together** —
ordered by date. Measured on 2026-08-25, **four of those six research files contain no
supersession marker of any kind** — and one of the four is round 4, whose §0 build order
`docs/design-critique-memory-safety-2026-08-22.md` §4 replaces outright and whose memory
pick `docs/research-round5-memory-safety-2026-08-22.md` re-answers in its own header.
An outside contributor asking "what is the design?" had to read
those 191 KiB and work out for themselves which conclusions had expired. That is a real gap
for something calling itself open and hackable, so this file exists and the research
rounds become what they always were: **evidence, kept verbatim, not a design record
you are expected to diff by hand.**

Reading order: this file for the design, [`docs/plan.md`](plan.md) for what is being
built next, [`docs/test-design.md`](test-design.md) for the test contract. Which document
supersedes or partly overturns which is recorded in [`docs/INDEX.md`](INDEX.md), and §4.4
below carries the three retractions that have to be read even by someone who never opens
the index.

Two rules govern this file, because a design document is the easiest file in a
repository to let rot:

1. **Every claim about the running system names its evidence** — a file and line, a
   test that fails if the claim stops holding, or a measurement with a date. A claim
   with no evidence is marked as a claim.
2. **"Not doing X" is three different states, never one.** Refuted, deferred with
   named preconditions, and not yet evaluated are not the same sentence, and a
   document that collapses them tells the next reader the evidence is closed when it
   is not. §4 keeps them apart on purpose.

繁體中文版:[`docs/design.zh-TW.md`](design.zh-TW.md)

---

## 1. The five promises, and which layer each one actually reaches

The interesting number in this repository is not what it can do. It is the gap
between what it promises and what a running process reaches — and that gap is
**derived by a gate, not remembered**: `infra/wiring_census.py` reads every python
file and reports every seam that only `pytest` arrives at, checking the result
against `infra/dormant_seams.tsv` in both directions.

R1.1b added a fourth granularity to that gate, and the reason is worth stating here
rather than in the test design alone: the first three all answer *is this seam dead?*,
and none of them can answer *can the process the census describes even start?*
`aliases()`'s own docstring said it resolved `from <pkg> import <mod>` "**after a
`sys.path` bootstrap**" and never checked that a bootstrap was there. `memory_node` was
in exactly that state — see the correction to promise 2 below — so the census reported
its seams as reached while the node died before `rclpy.init()` on every boot. A false
*dormant* costs a reader an afternoon; a false *reached* is the one direction this tool
must make impossible, so `bootstrap_findings` derives the requirement from the runner's
own launch table instead of assuming it.

| # | Promise | Reaches | Evidence |
|---|---|---|---|
| 1 | **It hears, sees, speaks, moves — and discloses that it is a machine** | **Running end to end.** | The demo runbook [`docs/DEMO.md`](DEMO.md); Art 50(1) is spoken by `gateway/ws_gateway.py:105-119` through `safety/disclosure.py` on `/speech/say` with no LLM in the path, asserted by `tests/test_disclosure.py` (T-E1) |
| 2 | **It remembers you** | **The card is wired; the episodes are not.** *Corrected in R1.1b:* for the whole of R1.1 this row was true of the source and false of the running robot. `memory/memory_node.py` — the publisher of the card and the single writer — was launched as a file by `infra/run_ec2.sh`, so `sys.path[0]` was `memory/`, and there is no `memory` package inside `memory/`: `from memory import profile` raised `ModuleNotFoundError` from every working directory, before `rclpy.init()`, on every boot. The four other launched nodes each carry the `sys.path.insert` line and this one did not. So nothing was published, nothing was appended to `audit.jsonl`, and `run_ec2.sh status` showed a node that looked exactly like one nobody started. R1.1b adds the line and the gate that derives it. R1.1 subscribed `agent_node` to the latched `/memory/profile` and passes `card=` into `assemble()`, so a hand-written profile reaches the prompt of the running robot and a turn that beats the card leaves an audit record. Everything downstream of *writing* memory is still dormant: `memory/recall.py`, `memory/consolidate.py` and `memory/salience.py` execute no line at runtime and `memory/segment.py` is loaded by no node, so the robot remembers only what a human typed into `profile.json`. | For the wired half: `tests/test_profile_wiring.py`, and `/memory/profile` no longer has a row in `infra/graph_gaps.tsv`. For the dormant half: `assemble(recall=)` and `memory/segment.py` in `infra/dormant_seams.tsv`. Wired by R1.3 and R1.4 |
| 3 | **You can tell it to forget** | **Recognised, not acted on.** `safety/guard.py:403 forget_request()` already recognises a deletion request in Chinese and English, and has no caller — so the robot can be asked to forget and will not. `memory/erase.py` is loaded by no node. | Two rows in `infra/dormant_seams.tsv`; `tests/test_guard.py:125` and `:543` cover the recogniser itself. Wired by R1.5 |
| 4 | **It cannot be talked past its boundaries** | **Half wired, and the dormant half is the output side.** The input side is live: the constitution is loaded on every turn and the L0 stop reflex is deterministic. `safety/output_guard.py` — the side that inspects what the robot is about to *say* — is loaded by no node, and two functions in the live `safety/guard.py` have a second, weaker implementation inline in `agent_node` (see §3, defect D1). | `tests/test_guard.py` (which cites T-D1..T-D4) for the live half. For the dormant half the repo says so in its own data: `agent/tools.py`'s `EGRESS[SAY]` row carries `enforcement=NOT_WIRED`, and two `defect` rows sit in `infra/dormant_seams.tsv`. Wired by R1.6 |
| 5 | **It actually plays against you** | **Wired but amnesiac.** `game/opponent.py` runs, and `game_node` publishes each resolved round to `/game/rps_round` — which no node subscribes to. The transition table therefore starts empty every boot, and the ledger's own arithmetic says the first ~120 rounds fall back to a 1/3 coin (81 order-2 contexts). | The paired rows `/game/rps_round` + `/memory/rps_table` in `infra/graph_gaps.tsv`, which the gate requires to be wired at both ends or neither. Wired by R1.2 |

**The honest summary is two sentences now, and that is the point of R1.1:** promise 1
works, promise 2 reaches the prompt for the first time but only for facts a human typed,
and promises 3–5 are built, tested offline, and not connected. The six dormant subsystems
are **3,492 production lines plus 5,488 test lines = 8,980**, against **37,062** python
lines in the tree on 2026-08-26, so **24.2%**.

**The walk behind that denominator is written down here because the last two rounds' were
not.** It is `find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' |
xargs wc -l`, which includes `tests/`, `.push/` and `web/`. Earlier rounds quoted 26.6%,
25.8% and 25.3% without recording how the tree was counted, so this round could not
reproduce 35,507 to within 1,249 lines and **the movement from 25.3% to 24.2% must not be
read as progress** — part of it is a different walk and part of it is 600 new lines of
test. A number whose derivation is not written down cannot be compared to itself, which
is §D item 4's rule applied one level deeper than it was written.

That percentage is the weakest number on this page and it is worth saying why. Its
denominator is the whole tree, so **writing more tests anywhere makes the dormancy
figure fall without connecting anything** — and it did so twice in the two rounds before
this one, neither of which wired a single subsystem. R0b-1's census moved it from 26.6% to 25.8% by
adding 1,042 lines (`infra/wiring_census.py` 454 + `tests/test_wiring_census.py` 588);
then R0b-2 moved it to 25.3% by adding 713 more (`infra/doc_status.py`,
`tests/test_doc_status.py` and the widened gates) — a PR that was *about* documentation
status. A number that improves when you write a document about documents is not measuring
what its name says, and R1.1 shows the other half of the same problem: it wired a real
subsystem and 600 of its 700-odd new lines are tests, so the figure would have fallen by
about as much had it wired nothing. The count of *seams* is the number that behaves:
`infra/wiring_census.py` reports **41** today, `infra/dormant_seams.tsv` carries all 41
with a named reason, and every R1 PR is required to delete at least one row — R1.1 deleted
two and re-classified a third, which is the first time this figure moved for the reason it
exists. Quote the seam count; treat the percentage as scenery.

**What "reaches" means here, stated because it is narrower than it sounds.**
Reachability is computed on the static call graph from the processes
`infra/run_ec2.sh` launches. A module being imported and called does not mean a
message flows through it on real hardware — that is L4 in `docs/test-design.md` §3,
and L4 is blocked (§6). In the other direction, a file reported unreached has no
static call path *in this tree*; `importlib` or string reflection would walk past the
census, and this tree has no such call site today.

---

## 2. Two systems, and why the split is a latency measurement rather than a taste

The architecture's central choice is that a **reflex layer never calls the language
model**. This is not a philosophical position about System 1 and System 2; it is
arithmetic from `docs/day0-measurements.md`.

Measured from the instance to Bedrock, Haiku 4.5's time to first token is **p50 900 ms
(min 645, max 1270, n=10, `ConverseStream`, the `global.` inference profile — the
direct model ID is not on-demand-invocable in ap-southeast-1)**. A rock-paper-scissors
beat is roughly 500 ms. So any design in which the beat waits on a model is already
broken before the first user, and no amount of prompt engineering recovers it. That
single number is what forces the split:

| Layer | Owns | Never does | Enforced by |
|---|---|---|---|
| **System 1 — reflex** | Beat timing, stop words, barge-in, gesture whitelist lookup | Call an LLM. Not "avoids"; **cannot** | `tests/test_guard.py::test_the_stop_reflex_needs_no_boto3_even_lazily` runs the stop path with `boto3` replaced by an object that raises on attribute access — a static import scan is not enough, because a lazy `import boto3` inside a function would pass one |
| **System 2 — deliberation** | Conversation, scene description, gesture *choice*, storytelling | Own any deadline shorter than itself | The interrupt path: `/speech/interrupt` → `T_FLUSH` is what makes an asynchronous safety check affordable at all |

Two consequences of that split are worth stating because they look like accidents:

**The output guard has to be asynchronous — and that premise was borrowed, then
measured.** The design originally rested on a third party's client-side figure for
Bedrock Guardrails: p50 749 / p90 907 / max 1837 ms at **n=15**. An entire safety
architecture following from someone else's n=15 is worth checking, so
`safety/guardrail_latency_harness.py` (T-K3) measured `apply_guardrail` in our own
region, source=OUTPUT, n ≥ 30 per language group, run twice. Result: **p50 415.8–435.4
ms, p95 498.7–531.6 ms** (pooled n=60 per group) — about **44% lower** than the borrowed
number.

**The design survives, and it is worth being clear about why.** All three language
groups still exceed S7's 400 ms budget, and the async design's premise is only "this
call is too large to sit in series with speech" — not any particular value. So R1.6
routes the verdict through the *existing* `/speech/interrupt` → `T_FLUSH` kill switch
rather than gating speech on it: the robot starts talking and gets cut off mid-sentence
if the verdict says so. The kill switch was built for barge-in; safety rides it for
free.

Two things that measurement taught, both of which would have been invisible in a
one-run design:

- **No language is slower.** The between-group p50 gap is 19.6 ms; the *same group's*
  run-to-run 95% upper bound is 40.6 ms. The gap is inside the noise, so this data
  supports no claim that Cantonese or Mandarin costs more. The comparison's ruler has
  to be the retest interval's widest bound, never its point estimate — point-to-point
  on pure noise makes almost any difference look real.
- **The guardrail tier is read from the control plane, not written down.** The
  intuitive way to create a guardrail yields the Classic tier, which supports only
  en/fr/es — a tier this robot speaks *none* of. The original assertion,
  `evidence["tier"] == "STANDARD"` against a literal `"tier": "STANDARD"` the harness
  itself wrote, was green forever while measuring the wrong tier.

One more latency result that changes what to expect: `effort:"low"` **does not buy a
faster first token** (+48.3 ms, 95% CI [−6.5, +95.8], crossing zero). The parameter is
honoured — it shortens *generation*. For sentence-level streaming TTS that means it buys
shorter turns and lower output cost, not an earlier first word.

**Speech is shipped sentence by sentence**, which is what makes the interrupt
meaningful at all — you cannot flush an utterance that was synthesised as one blob. The
first-audio arithmetic is asymmetric between languages and the asymmetry is measured:
Mandarin through Kokoro-82M is **~0.15 s** to first packet, so first audio lands at
roughly 900 ms + 150 ms ≈ **1.05 s**; Cantonese through CosyVoice-300M-SFT is
**3.63 s** to first packet, 24× Kokoro. That gap is why Cantonese's fixed lines
(greeting, game chant, stop acknowledgement) are pre-synthesised into a phrase cache
and only novel sentences pay the 3.63 s. Nothing in this repo claims Cantonese is
*protected*, only that it is spoken — `tests/test_guard.py` and T-G4 both pin that
distinction, because a fast call on a language Guardrails does not list is the easiest
way to accidentally imply coverage.

---

## 3. Invariants — the closed list

These are the properties the system is not allowed to lose. Each one names the test
that fails if it is lost. The list is closed: something not on it is a design
preference, not an invariant, and may be traded away without a discussion here.

| # | Invariant | Where it is created | What fails if it is lost |
|---|---|---|---|
| I1 | **One writer.** *(R1.1b: this invariant was never at risk and that is the wrong kind of comfort — until R1.1b the writer did not run at all, so nothing was written by anyone. An invariant of the form "only X may do this" is satisfied vacuously by an X that cannot start, and no test in this repo could see that, because pytest puts the repo root on `sys.path` before importing anything.)* Only `memory_node` writes to disk, and only through `memory/store.py`. Every guarantee about what may be written lives in the single writer, never in the module that produces the record | `memory/store.py:1`; the principle is stated at `:265-267` | `tests/test_store.py` (which cites T-A1..T-A4), `tests/test_erase.py`. The reason is written down in the source: *"A guarantee enforced in the reader is a comment; enforced in the one writer, it holds for code that has never heard of the profile module"* — any of the nine nodes can reach the store directly, so a producer that merely declines to write constrains only its own callers |
| I2 | **Anything that moves the body is a text tag parsed after the fact — never a tool.** There *is* a real tool loop (I8), and `[gesture:*]` / `[game:start]` are deliberately kept out of it: the model writes them inside prose, and they are dispatched and stripped only after the reply exists. So no schema of ours ever advertises the actuators, and the `\w+` capture cannot express a colon, which is what puts `chant`'s parameter form out of the model's reach | `agent/agent_node.py:264-269`; the claim is declared as data in `agent/tools.py`'s `EGRESS[GESTURE]` row | `tests/test_guard.py::test_the_whitelist_has_exactly_one_creator` and `::test_a_thousand_random_tags_yield_only_whitelisted_names` (T-D3); `tests/test_tools.py` asserts the egress table's rows are internally consistent. **Caveat, stated in D1 below:** both guard tests bind `safety/guard.py`, which is the *dormant* copy of the tag parser |
| I3 | **Memory enters the prompt only inside the data fence.** Recalled text goes into a named block that cannot claim to be instruction; the region a value lands in is decided by the parameter it arrives through, so there is no argument a caller can use to put recalled text in the instruction region | `safety/guard.py:191 data_region` *refuses* a fence-name collision rather than resolving it; the names come from `agent/context.py:114 BLOCKS`, not from any caller | `tests/test_context.py` (T-C5 injection half): a hostile sentence recalled from the store must appear only inside the fence, and the instruction region must be byte-identical to the no-memory baseline |
| I4 | **L0 never goes through the language model.** The stop word, the beat and the whitelist are deterministic | `game/game_node.py`, `body/gestures.py:86 GESTURES` | T-D2, both halves: statically the reflex modules must not import `boto3`; dynamically `tests/test_guard.py:776` runs the stop path with a `boto3` double that raises on attribute access and requires it to still stop |
| I5 | **Disclosure travels a deterministic path.** EU AI Act Art 50(1) is spoken by the gateway from a fixed string, so it still happens when Bedrock is down or slow | `gateway/ws_gateway.py:14-18` (why the gateway owns it), `:105-119` (the latched utterance), `safety/disclosure.py` | `tests/test_disclosure.py` (T-E1), including that the position of the disclosure relative to the frame loop is asserted in the AST, that neither sentence routes to the out-of-process engine, and that the Chinese sentence is spoken first |
| I6 | **No model judges a test.** Anything needing semantic judgement is reduced to a machine-checkable proxy or moved to human acceptance testing | `docs/test-design.md` §0 rule 1 | Structural: there is no judge to break. The evidence for the rule is in the README — judge-gated pipelines move the *measured* pass rate 0.72 → 0.94 while real correctness stays at 0.20 |
| I7 | **Every assertion is paired with a named mutant it must kill**, and a mutant producing byte-identical output means the assertion measures the wrong quantity | `docs/test-design.md` §0 rule 2 | `tests/test_meta.py` (T-J7) — the word "every" is what T-J7 guards |
| I8 | **The tool surface is closed and declarative, and discovery may only narrow it.** `agent/tools.py:179 REGISTRY` declares exactly one tool today — `read_skill_file`, local, read-only. `MAX_EFFECT_HOPS = 0`, so a tool declaring `MUTATES` **cannot be invoked at all**; raising that number is an edit to that file in a PR, not a config change. `reconcile()` treats a tool a server offers but the repo does not declare as a *finding*, never as a tool | `agent/tools.py`, written before any MCP client exists — the same reason `store.py`'s `why` was made mandatory before anything wrote to `events`: a boundary is free to draw exactly once | `tests/test_tools.py` (62 collected cases from 38 test functions — the count `infra/doc_counts.py` reconciles is the collected one): `set(REGISTRY) == {"read_skill_file"}`, no `MUTATES` spec is declared, `toolConfig`'s tool order is derived from `REGISTRY`, and the `EGRESS` rows are checked for internal consistency. `toolConfig` is attached only on turns where a skill fired, so most turns carry no tool schema at all |

### D1 — the one invariant with a known defect, named rather than smoothed over

I2 has **two implementations of one vocabulary, and the live one produces no
findings.** `safety/guard.py:318 gesture_findings()` splits a reply's tags into
accepted plus *a finding per rejected tag*, and `:331 spoken_text()` returns the reply
with tags removed. Neither has a caller. `agent/agent_node.py:264-269` re-implements
both with inline regexes and **performs no whitelist check** — every `\w+` match is
published to `/body/gesture`.

**Be precise about what is and is not contained.** Nothing unsafe reaches an actuator:
`body/body_node.py:123` does `GESTURES.get(name)` and drops an unknown label, and that
containment is *declared*, not accidental — `agent/tools.py`'s `EGRESS[GESTURE]` row
records `enforcement=ENFORCED` with `enforcer="body.gestures.GESTURES"` and a written
reason, and T-D3 defends the whitelist's single creator. So the safety property holds.

What is actually lost is **the audit trail**. The written-for-this-purpose function
turns a rejected tag into a finding; the live path turns it into silence. Nothing
anywhere records that the model asked for a gesture the robot does not have, which is
precisely the signal that would tell you the model is drifting off the whitelist. The
enforcement moved one hop downstream, and the observability was dropped on the way.

It is recorded as two `defect` rows in `infra/dormant_seams.tsv`, each naming R1.6 as
the stage that fixes it, and `tests/test_wiring_census.py` pins those rows to the
actual regexes and to that `.get(name)`, so the duplication cannot be resolved silently
in one direction only. It is stated here rather than only in a commit message because a
defect that only a gate knows about is a defect the next reader will re-discover.

### Every way out, and what actually enforces it

`agent/tools.py:394 EGRESS` exists because **speech is not the only way out, and today
it is the only one anybody thought to guard.** It enumerates every channel the model's
output can reach, together with the enforcer that runs on it *in fact*. The column that
earns the table is the last one: it distinguishes `NOT_WIRED` (the enforcer is in this
repo and nothing calls it) from `NO_ENFORCER_TODAY` (there is nothing to call). Those
are the same absence with very different repair costs, and folding them together would
have let this document describe S7 as protection.

| Channel | Carries | Model can reach it | Enforcement today |
|---|---|---|---|
| `SAY` | words spoken aloud | yes | **`NOT_WIRED`** — `safety/output_guard.py` exists and an AST census finds zero production callers. This channel is unguarded *in fact*, and the row's own test goes red when R1.6 lands |
| `GESTURE` | whole-body motion | yes | `ENFORCED` by `body.gestures.GESTURES` (see D1 for what that costs in audit) |
| `GAME_CMD` | game state transitions | yes | `ENFORCED` — `game_node` owns the cooldown and stop reflex, so `[game:start]` is a *request* the reflex layer may decline |
| `SKILL_TOOL_CALL` | a filesystem read under `skills/` | yes | `ENFORCED` — containment checked on the resolved path *after* symlinks, and a refusal returns `REFUSED` rather than an error (T-I3) |
| `INTERRUPT` | silencing the robot | **no** | `NO_ENFORCER_TODAY`, and correctly so: published by `on_asr` on barge-in, never from model text; it carries no content and its worst case is silence |
| `MCP_TOOL_CALL` | whatever a remote server does | **no** | `NO_ENFORCER_TODAY` — no MCP client exists and `REGISTRY` declares no remote tool. `invoke()` *reads this row*, so the day a remote tool is declared it is refused until something is named here |

That last row is the pattern worth copying: the gate and the census are the same
object, so they cannot drift apart. And note the honest shape of `SAY` — the single
most important channel in the system is the one with an enforcer sitting in the repo
that nothing calls. R1.6 is that row.

**One vocabulary decision behind all of this.** `invoke()` returns one of **six**
outcomes, and exactly one of them means the boundary acted. `REFUSED` is a claim about
policy; `UNREACHABLE` is a claim about the network. A report that counts the second as
the first says the robot was protected by an outage.

---

## 4. What is deliberately not built, and in which of three states

A document that writes "we are not doing X" for three different epistemic situations
is worse than one that says nothing, because it tells the next reader the evidence is
closed. The three states are kept apart here, and the previous internal wording is
corrected where it collapsed them.

### 4.1 Refuted by measurement

| Not doing | Why, with the number |
|---|---|
| **A small model as router / escalation judge** | Round 7 §1: the routing plateau is **+2.13 pp** for 10× the data, 3× the encoder and fine-tuning — and that figure is the *upper bound* against a label oracle. An entropy router escalated on **0.0%** of TextWorld episodes. At a 10% risk budget, **zero** model pairs were certifiable. HybridLLM ranked worst of 21 methods; R2V was strictly worse than the pure LLM on 3/3 settings |
| **Reinforcement learning anywhere in this agent** | Round 7: RL is admissible exactly where **the reward is measured by a non-LLM sensor**, and that condition rules out every intervention point this system has. Engagement-as-reward is worse than useless — it optimises directly against telling the truth. There is also a prior, structural reason: **this robot runs no physics.** `body/body_node.py:98,146` calls `mujoco.mj_forward` and never `mj_step` — joint positions are set directly, so it cannot fall, cannot lose balance, and has no contact forces. An environment that produces no consequences has nothing to reward |
| **Training a locomotion policy on the T4** | Round 6: even a policy that trains transfers badly without system identification — the sim's `armature = 0` means gains tuned in simulation do not carry to hardware. Deploying someone else's pre-trained `motion.pt` is *using* RL, not doing it, and costs about three orders of magnitude less. Note what would have to be paid first: moving from `mj_forward` to `mj_step` is an unpriced prerequisite, not a flag |
| **Prosody-derived emotion in the salience formula** | Round 5: inferring emotion from biometric signals is high-risk under the EU AI Act. `memory/salience.py` drops voice-prosody signals for that reason, and T-C6 asserts both halves — the feature off, and a positive control with it on |
| **Ebbinghaus-style decay as the right-to-be-forgotten mechanism** | A decay curve is not deletion. Erasure must be an act with a receipt (tombstone, cascade, confirmation), which is what `memory/erase.py` and T-C4 implement |
| **A model judging a test** | I6 above |

### 4.2 Deferred, with named preconditions — **not** refuted

This subsection exists because an earlier internal draft wrote "not doing DDTSR",
which turned Round 7's *deferral* into a *veto*. Round 7 §1.2's own verdict column
reads "re-listed as a latency-masking candidate, after axis A".

| Deferred | Measured upside | Preconditions before it can be reconsidered |
|---|---|---|
| **A small local model as latency masking** (not as a decision-maker) | Round 7 §1.2, measured on an almost identical stack: Qwen3-0.6B emitting knowledge-free connectives + sherpa-onnx + CosyVoice2 on one card with the big model remote, **1003 → 548 ms (−45.4%)**, quality unchanged. The mechanism that survives is precisely that **the small model decides nothing** | (a) SFT + curriculum + an entropy gate; (b) connective coverage — the gain falls to **17.5%** when coverage drops 94% → 38%. That statistic is **T-K2**, which is doubly blocked: it needs real dialogue logs (R1.3's output) *and* it runs on the instance, so it is behind the `/32` too (§6); (c) **no Chinese or Cantonese evidence at all**; (d) a hard placement rule from Round 7 §1.5, below |
| **A local QLoRA domain expert** | Not evaluated, but two constraints already follow it: 4-bit VRAM arithmetic (GB ≈ parameter count → **1.5B–4B with short context** inside the 7.9 GiB budget, after YOLO26's 527 MiB and Kokoro + paraformer's 2–3 GB), and Round 7's tool-calling measurement — Qwen2.5-14B multilingual BFCL picks the **right tool 91–95%** of the time but emits a **fully correct call only 22–32%** | Same as above plus a decision about what "domain" means here |

**A hard placement rule that any local first-tier proposal must answer first.**
Round 7 §1.5 measured that a two-tier architecture *introduces* an attack surface: a
cheap model upstream of a cloud model creates a relay channel, and at the **memory
write node** the cheap model relayed injection **~53%** of the time while the strong
model was **0/164**. Cloud-side alignment cannot rescue content that was already
rewritten upstream. R1.3 is exactly the PR that connects `/speech/asr` to
`store.append`, so: **no local first tier may sit upstream of the memory write path.**

### 4.3 Not yet evaluated — no evidence either way

| Not evaluated | Why the existing rounds do not answer it |
|---|---|
| **Offline / disconnected availability** | Every occurrence of "offline" in Round 7 means *offline preference data* or *offline simulation*. Not one sentence addresses working without a network. Round 7 measured with two rulers, latency and cost; availability is a third ruler nobody has picked up |
| **Whether the demo's promises survive a network partition** | Cannot be enumerated before there is episode data to enumerate over — i.e. not before R1.3 |

### 4.4 Three claims from the research rounds that may not be reused

The rounds are kept verbatim as evidence (§7), which means the retractions inside them
have to be surfaced rather than left for a reader to find on page 12. Round 7's Appendix A
(證據來源清單) records its own source audit, and it voided things:

- **`arXiv 2607.24440` is a vision-language image-degradation study** that the research
  pipeline filed as a text-LLM result. **The three quantitative / confidence-calibration
  claims that cited it are void.** Nothing in this file rests on them.
- **`37.6% → 56.9%` ("scale does not save safety") is not in SafeAgentBench**
  (`arXiv 2412.13178`), which states as much explicitly. The conclusion may still be
  true; the citation is not. This file does not use the figure.
- **Two pipeline records swapped dates once** (2026-07-30 / 2026-07-07 for entries #21
  and #22), so which paper a citation resolves to must be re-checked before reuse.

Of six arXiv IDs cited in one workflow, five were verified to exist with matching
titles. That is the ratio to keep in mind when quoting a round: **1 in 6 needed
correction, and the correction was only found by going to the sources.**

### 4.5 Deliberately unchanged, with the reason

- **`memory/store.py` is not being split.** Its 1045 lines do hold three concerns
  (schema vocabulary, data-root policy, the writer), but splitting re-anchors all
  **15 mutation cases** in `tests/test_store.py` — 18 named mutants across them, because
  the position check applies four in one test — and adds an import edge. Earlier drafts of
  this line said "15 named mutants", which is the collected *case* count wearing the
  mutant count's label; the two are now separated because `infra/doc_counts.py` gates the
  first and cannot derive the second. The chosen
  fix is to make the schema surface explicit with block headers and an exported list,
  in place.
- **`infra/arch_render.py`'s 1118 lines are not being refactored.** The
  code-to-drawing ratio is arguable — `arch_render.py` plus
  `tests/test_architecture.py` is 1,748 lines to draw and verify one picture, more than
  the production code of perception + speech + gateway + body combined (1,428). But it
  is green and byte-for-byte rebuilt by its test; refactoring it now buys risk and no
  promise. The ratio is recorded as a proportion problem, not a defect.

---

## 5. Where the tests are, versus where the risk is

This is the second-largest honest gap after §1, and it is stated here because a test
count is not a safety argument.

**This section corrects the analysis it replaces.** The plan this round works from
claimed "6 modules, 932 lines, zero tests". Re-deriving it: the six line counts sum to
**1026**, not 932, and three of the six *are* read by a test. The claim was assembled
by hand, which is the failure mode this whole repository is built to avoid, so here is
the derived version. The rule used — a module counts as read if any file under `tests/`
imports it or names its filename, excluding `tests/test_wiring_census.py`, which by
construction names every path in the tree.

**Read by no test at all — 3 files, 449 lines,** and every one of them is where a
promise terminates:

| File | Lines | Why it matters |
|---|---|---|
| `body/body_node.py` | 213 | The gesture *consumer*. T-D3 proves the sending side emits only whitelisted names; nothing tests what happens at the actuator — and D1 above shows the containment for the live tag path currently rests here, in this file, untested |
| `perception/detect_node.py` | 140 | Vision. The whole of promise 1's "sees" |
| `speech/asr_node.py` | 96 | T-B2's premise, *"publishes nothing during silence"*, is written down only in another file's comment (`tests/test_segment.py:331` cites `asr_node.py:79-87` in prose). No assertion reads this file, so the premise of the segmentation tests is unenforced |

**Read only in part.** These are covered, but not where the risk is:

| File | Lines | What is asserted, and what is not |
|---|---|---|
| `gateway/ws_gateway.py` | 219 | AST-asserted in two files: `tests/test_disclosure.py` pins the disclosure's position relative to the frame loop, and `tests/test_voices.py` pins the marking path. **Nothing asserts `:72`** — the one line that maps `/speech/interrupt` to the `T_FLUSH` byte. That is the line R1.6 intends to hang the output guard on (§2), so it is the highest-value single assertion R4 can add |
| `perception/hand_node.py` | 137 | `tests/test_game_wiring.py:476` derives the game's accepted vocabulary from `hand_node.classify` rather than writing it down. The sampling window and the reward decision — the **only non-LLM reward signal in the system**, and therefore the whole legitimacy of the game layer — are untested |
| `memory/rank_harness.py` | 221 | Imported by `tests/test_rank.py`, which asserts its JSON output. The plan lists it as a deletion candidate for being untested; that premise is false and R2 should drop it |
| `body/gestures.py` | 175 | The `GESTURES` keys *are* bound: `safety/guard.py:315 allowed_gestures()` returns `frozenset(_gestures.GESTURES)` and `tests/test_guard.py:434` asserts the exact set, so the whitelist cannot drift silently. The keyframe geometry and the player are untested |
| `infra/day0_measure.py` | 89 | Named by no test, and it is also the one measurement script with no verdict function (§1.5 of the plan) — so it can neither fail nor be checked |

R4 addresses these with the idiom `tests/test_game_wiring.py` already proves works:
structural AST assertions on nodes that import `rclpy`, with pure logic extracted into
ROS-free functions and unit-tested. Ordering follows the table: `ws_gateway.py:72`
first, because R1.6 depends on it.

---

## 6. The four things blocked on a human, and exactly what each blocks

None of R0–R4 needs any of them. Stating what each one blocks is the point: a blocker
with no named consequence turns into a permanent excuse.

| Blocked on | Blocks |
|---|---|
| **Security group `/32`** (adding this laptop's egress IP to the instance's inbound rule — a permission *widening*, so a person has to authorise it; SSM is not a fallback, because the instance profile is registered and the agent is not) | The on-machine measurements: **T-K1** (whether 50 Hz `mj_step` closes on 4 vCPU), **T-K2** (the connective-coverage statistic), **T-K4** (`gte-multilingual-base` short-record latency, which decides whether R1.4's 150 ms deadline is even feasible), plus T-F2's p95, T-G1's adjudication→silence 400 ms, T-G2's first-sentence 50 ms delta, and T-G4's real run. It blocks **measurement, not wiring** — every R1 PR is laptop-verifiable |
| **Real dialogue logs plus one human annotation pass** | T-F1/T-F2 retrieval quality, and (with T-K2) the connective-coverage number that decides whether latency masking is worth 45.4% or 17.5% (§4.2). **This cannot be fabricated** — a synthetic annotation pass produces a number with no referent |
| **Manual UAT of the SVG animation in GitHub's own viewer** | The L5 half of T-M5/T-M6. GitHub renders the SVG as `<img src=raw…svg>` on `viewscreen.githubusercontent.com` — a *different origin* — so nothing in the page can be inspected programmatically. A person has to open it and look |
| **The `RETENTION_DAYS = 90` policy decision** (`memory/erase.py:170`) | Nothing in code: the sweep is implemented and T-C4 tests the mechanism. It blocks only the *claim* that 90 days is a policy rather than a default someone typed — which is why `docs/test-design.md` §7 states outright that the retention tests cover the mechanism and not the threshold |

**Not on this list, because it is done.** An earlier draft of this section listed
"real Guardrails latency" under `/32`. It was measured — T-K3 shipped
`guardrail-latency.json` with p50 415.8–435.4 ms (§2). Leaving a completed measurement
in a blocked list is how a blocker becomes a permanent excuse, which is the failure this
section is shaped to avoid.

---

## 7. How this file is kept from rotting

Four mechanisms, all four in force as of this PR:

1. **[`docs/INDEX.md`](INDEX.md) plus `infra/doc_status.tsv`** give every document under
   `docs/` a status from a closed set and, where applicable, a named successor — including
   the form *overturned in part*, because §4.4's case (measurements stand, the citation
   does not) is not the same as superseded and a two-value column would have flattened it.
   Round 4 is why the relation column is `;`-joined rather than single-valued: it has two
   independent overturners, and one slot would have forced dropping one of them while the
   row read as complete. A new document cannot join silently, a stale row is a finding, and
   `infra/doc_status.py` checks the index against the table against the directory in all
   three directions. It also reads the attribution sentence in §0 of this file, because an
   earlier draft of that sentence credited round 7 with overturning round 4 and round 7
   never names round 4 except to cite it approvingly — prose about which document overturns
   which is exactly as unverified as any other prose.
2. **§1's dormancy claims are reconciled against the tree, in both directions**, by
   `infra/wiring_census.py`, so an R1 PR that wires a subsystem without deleting its
   row goes red. **And the numbers in this file are now gated too, which they were not
   when this section was first written.** `infra/doc_counts.py` reconciles case, mutant
   and module counts against `pytest --collect-only`, and its `DOCS` tuple was
   `docs/test-design.md`, `README.md`, `README.zh-TW.md` — so the collected-case count
   quoted above (62, in I8) sat hand-checked inside a document whose whole argument is that
   hand-checked numbers rot. R0b-2 widened the tuple to this file, its Chinese edition,
   `docs/plan.md` — and `pytest.ini`, which is not a document at all. The `mutation`
   marker's help text carries the whole-layer wall-clock figures so they sit beside the
   marker they describe, and they were two rounds stale: prose explaining why a duration
   must be re-measured every round, quoting a 1512-case suite with 304 mutants. Membership
   in that tuple is therefore *a file that quotes one of these numbers*, never *a file under
   `docs/`*. Two further details of the widening are worth keeping, because both were
   mistakes first: adding a path only makes a file *readable*, so nothing is checked until a
   pattern quotes a number out of it — every mutant therefore changes a number in a
   document rather than shrinking the tuple. And I8's sentence quotes **two** quantities,
   62 cases from 38 test functions, which needed two derivations; one derivation checking
   both would have left the gap between them free to drift.

   The widening also found where the checks were *narrow* rather than absent. A §0 row in
   `docs/test-design.md` states two numbers — how many cases the module collects and how many
   of those the mutation layer selects — and only the first had ever been read. Comparing the
   second found three wrong rows on its first run: one saying 13 where the layer collected 12,
   and two stating a breakdown of named mutants (17 + 1, 8 + 2) with no total at all, so
   collection had no number to disagree with. **A breakdown is not a total**: 17 + 1 is 18
   only to a reader who adds it up, and a gate does no arithmetic. The same three-line pattern
   holds for prose: a module's count appears in the table *and* in the paragraphs discussing
   it, and the table check reads only the table — which is how the original 25-vs-23 pair
   survived four lines apart. Those sites are now found by scanning the document for the
   phrasing rather than by listing them, so a paragraph written next month is checked the day
   it is written; zero sites is a finding, because zero means the phrasing moved.
3. **Every "not doing" in §4 keeps its state label.** When evidence arrives that moves
   an item between §4.1, §4.2 and §4.3, the move *is* the edit — not a rewrite that
   erases which state it used to be in.
4. **Numbers whose derivation is not automated carry their date and their
   denominator.** §1's 24.2%, §5's line counts and §2's latency figures each say what
   they were measured against and when, because the alternative — a bare percentage — is
   the form in which a stale number survives longest. Where this file corrects an
   earlier claim (§1's 26.6%, then its 25.8%, then its 25.3%, §5's "932 lines", §6's
   Guardrails entry), the correction says what the old number was — and R1.1 found the
   sharper version of this rule: three of those four readings recorded no **walk**, so
   the fourth could not be compared to them at all, and §1 now writes the `find`
   invocation down. §1's percentage has now been
   corrected three times in three rounds, which is the argument for quoting the seam count
   instead: the count moves when a seam moves, and the percentage moves when anyone
   adds a test anywhere. A correction that hides the thing it replaced asks the
   next reader to trust it, and this file's whole argument is that trust is not the
   mechanism. A timing number carries two more things — its **n** and the machine's
   **load** — because a narrow interval is not evidence of stability: the first two
   whole-suite passes this round read 239.4 s and 239.4 s, identical to the tenth of a
   second, and six further passes of the same command on the same tree spanned **137 s**.
   That width then dissolved three conclusions the two-pass version had already stated
   (that the machine was slower under load, that the parts summed to less than the whole,
   and that this round was slower than the last), which is why
   [`docs/test-design.md`](test-design.md) §3 now writes an n beside every interval and
   records where it stopped adding passes.
