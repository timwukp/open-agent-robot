# open-agent-robot — build plan

What is done, what is next, and in what order. The design argument is not repeated
here — it lives in [`docs/design.md`](design.md), and the test contract lives in
[`docs/test-design.md`](test-design.md). This file is the schedule.

**Why it is in the tree.** Until this PR the plan lived in `~/.claude/plans/` on one
laptop. That is a governance defect and not a filing preference: `infra/dormant_seams.tsv`
requires every `pending_stage` and `defect` row to name an `R<n>.<m>` stage, and the gate
`tests/test_wiring_census.py` checks that the stage id is *well-formed* — it cannot check
that the stage *exists*, because until now the document defining stages was not in the
repository. 15 rows currently name a stage. One of them (`agent/tools.py::reconcile()`)
names **R6.1**, a stage no earlier plan had defined at all; §9 below defines it, which is
the immediate reason this file had to be written rather than merely moved.

From now on: the plan is in the tree, and a stage id in a manifest is a link into §3–§9.

---

## 1. Where the three tracks stand

Three tracks, and the numbering they each use, because the repository has **two** stage
vocabularies and a reader who does not know that will misread both:

- **`S<n>` — feature stages** (S0 test skeleton → S8 tool boundary). Track A's spine, used
  throughout `docs/test-design.md` and in the source. `S` stages ship *capability*.
- **`R<n>.<m>` — this round's stages** (R0 converge → R6.1 tool loop on-robot). Used by
  `infra/dormant_seams.tsv` and by this file. `R` stages ship *connection, convergence and
  coverage*, mostly of things `S` stages already built.

| Track | What it is | Status | Evidence |
|---|---|---|---|
| **A** | Memory, disclosure and the moral boundary: S0 → S8 | **Offline half complete, S0 through S8.** Every module exists, is tested offline, and the S-stage that built it is recorded per T-ID in `docs/test-design.md` §2. **The on-robot half is not connected** — see the seam count below | `docs/test-design.md` §2.A–§2.L; the dormancy is `infra/dormant_seams.tsv` (42 rows) |
| **B** | The rock-paper-scissors layer: the only non-LLM reward signal in the system | **Wired and playing, and amnesic every boot.** `game/opponent.py` runs; `game_node` publishes each resolved round on `/game/rps_round`; nothing subscribes, so the transition table is empty at every start and the ledger's own arithmetic gives **~120 rounds of a 1/3 coin** (81 order-2 contexts) | the `/game/rps_round` and `/memory/rps_table` rows of `infra/graph_gaps.tsv`, counted as one gap from both ends |
| **C** | On-machine measurement | **2 of 5 done.** K3 (Guardrails latency) and K5 shipped; **K1, K2, K4 are blocked** — all three must run on the instance, and the security group is `/32`-restricted pending a human | `docs/test-design.md` §2.K and the L4 row of §3; blockers in [`docs/design.md`](design.md) §6 |

**The one number to quote is the seam count, not a percentage.** `infra/wiring_census.py`
reports **42** seams reachable only under `pytest`, and `infra/dormant_seams.tsv` carries 42
rows with named reasons, checked in both directions. The ratio of dormant lines to tree
lines (25.8% on 2026-08-25) falls whenever tests are added anywhere, including tests that
wire nothing — [`docs/design.md`](design.md) §1 derives why.

**What R1 has to close, derived from the manifests rather than remembered:**

| Manifest | Rows naming a stage | Which stage |
|---|---|---|
| `infra/dormant_seams.tsv` | 3 | R1.1 (`assemble(card=)`, `audit_payload()`, `unlock_for_s5()`) |
| | 4 | R1.3 (`memory/segment.py`, `salience.stamp()`, `consolidate()`, `known_ids()`) |
| | 1 | R1.4 (`assemble(recall=)`) |
| | 3 | R1.5 (`memory/erase.py`, `recall.deletion_findings()`, `guard.forget_request()`) |
| | 4 | R1.6 (`safety/output_guard.py`, `guard.clamp_reply()`, and the two `defect` rows) |
| | 1 | R6.1 (`agent/tools.py::reconcile()`) — §9 |
| `infra/graph_gaps.tsv` | 3 `pending` | R1.1 (`/memory/profile`), R1.2 (`/game/rps_round` + `/memory/rps_table`) |
| | 1 `pending` | **none** — `/game/events`, see R1.7 |

R1.2 is the one wiring PR that deletes **no** `dormant_seams.tsv` row: its work is entirely
node-half topic wiring, so its rows are in `graph_gaps.tsv`. A reader checking R1.2 against
the wrong manifest would conclude it does nothing.

---

## 2. Preserved deliberately

Convergence is not rewriting. These are the things this repository does that are both
correct and uncommon, and no stage below is permitted to break them:

1. **The manifest idiom** — closed class sets, checks in both directions, a named reason per
   row, and a stale row that is itself a finding. `infra/graph_gaps.tsv`'s header writes down
   `unused` as a class that *should stay empty*. Every new gate in this round copies this shape.
2. **Guarantees enforced in the single writer**, never in the reader (`memory/store.py:265-267`).
3. **One assertion, one named mutant, plus a no-mutant control.**
4. **Honest boundaries written in the source** — Cantonese is not claimed to be protected,
   0.80 is a regression line and not a validated standard, a ceiling that does not look
   inside containers says so.
5. **No LLM as judge**, anywhere in the test system.

---

## 3. R0 — converge (in progress)

| Stage | Contents | Status |
|---|---|---|
| **R0a** | `infra/wiring_census.py` + `infra/dormant_seams.tsv` + `tests/test_wiring_census.py`; the `T-L*` → `T-M*` docstring fix; the L1 time budget restored to `docs/test-design.md` §3 with the measured overrun | **Shipped — PR #28** |
| **R0b-1** | `docs/design.md` + `docs/design.zh-TW.md` + this file; both READMEs' Design-record section pointing at `docs/design.md` as the single authority | **This PR** |
| **R0b-2** | `docs/INDEX.md` + `infra/doc_status.tsv` + `infra/doc_status.py` + `tests/test_doc_status.py`; plus the rows that a new by-hand tool forces: `infra/standalone_entrypoints.tsv`, `infra/dormant_seams.tsv`, and a `MODULE_PROSE` row in `infra/arch_render.py` with an SVG regeneration. Plus widening `infra/doc_counts.py`'s `DOCS` tuple to `docs/design.md` and this file | Next |

**R0b-2's design is already fixed, because two of its decisions are not obvious.**
Columns are `path <TAB> status <TAB> relation <TAB> reason`. `status` is the closed set
`current` / `evidence` / `superseded`. `relation` takes `-`, `superseded_by:<path>`, or
**`overturned_in_part_by:<path>`** — that third form exists because
[`docs/design.md`](design.md) §4.4 is exactly that case: round 7's measurements stand while
three of its citations do not, and a two-valued column would flatten a partial retraction
into either "still current" or "dead". The scan walks `docs/**/*.md` **recursively** — the
measured inventory as of this PR is **16 files**, 14 at the top level plus
`docs/evidence/round7/README.md` and `docs/evidence/t-e3-browser-disclosure.md`, and a
non-recursive glob would silently exempt the evidence directory. R0b-2 adds
`docs/INDEX.md`, so that count moves to 17 in the same PR that starts gating it — which is
the point: a document cannot join `docs/` without a row and a number both changing. It carries a `ran` flag so
that a scan finding zero documents exits non-zero, and it mirrors the three-exit-code
vocabulary of `redact_scan.py` / `doc_counts.py` / `wiring_census.py`.

**And one row that R0b-1 could not close about itself.** `infra/doc_counts.py`'s `DOCS`
tuple is `docs/test-design.md`, `README.md`, `README.zh-TW.md` — so
[`docs/design.md`](design.md) and this file quote numbers that **no gate reads**. R0b-2
widens `DOCS` to both — and the mutant has to be aimed at the right thing. Adding a path to
`DOCS` only makes the file *readable*; nothing checks it until a pattern in `audit()` quotes
a number out of it, so a widened tuple with no new pattern is coverage in name only. The
mutant is therefore "change the number in the document and require a finding", not "shrink
the tuple". [`docs/design.md`](design.md) §7 states this gap in place rather than implying
coverage it does not have.

---

## 4. R1 — wiring (six PRs, plus one disposition)

Each PR connects one dormant subsystem and **deletes at least one manifest row**. All of it
is verifiable on the laptop; `/32` blocks none of it. The idiom is the one
`tests/test_game_wiring.py` already proved: structural / AST assertions on nodes that import
`rclpy`, with pure logic extracted into ROS-free functions and unit-tested.

**The honest boundary, to be repeated in every one of these PR messages:** a structural
assertion proves the wire was *declared*. It does not prove a message flows. That is L4, and
L4 needs the machine.

| # | PR | What changes | Rows deleted | The real risk |
|---|---|---|---|---|
| **R1.1** | `/memory/profile` → `agent_node` | Subscribe latched; pass `card` into `assemble`; call `audit_payload()` → `guard.audit_record`; turn `unlock_for_s5()` | 3 seams + 1 graph gap | **Not wiring — a design decision.** `assemble` runs once in `__init__` while the card arrives asynchronously. R1.1 must decide what a turn that beats the card does, and emit a finding for it rather than silently composing a card-less prompt |
| **R1.2** | The RPS pair | `memory_node` subscribes `/game/rps_round`; publishes `/memory/rps_table` latched | 2 graph gaps | The pair must be wired at both ends or neither — the ledger counts it as one gap from both sides. Table round-trip is already covered by T-H5; the node half is new |
| **R1.3** | Episode streaming | `memory_node` subscribes `/speech/asr`, `/detections`, `/game/events_out`; runs `segment` → `salience.stamp` → `consolidate` → `store.append` | 4 seams | The single-writer invariant: only `memory_node` may persist. The 45 s silence must be timed *by `memory_node`*, because `asr_node` publishes nothing during silence — there is no message whose absence a subscriber could notice |
| **R1.4** | `/memory/recall` service | `memory_node` serves it; `agent_node` calls it from a worker thread with a **150 ms** hard deadline, falling back to the file card | 1 seam (which alone holds 1,757 production lines dormant) | **Must not** use `spin_until_future_complete` — that spins a second executor against the single spin at `agent_node.py:203`. Whether 150 ms is even achievable is **T-K4**, which is blocked (§1, Track C) |
| **R1.5** | The deletion path | `guard.forget_request()` → `/memory/forget` → `erase.delete` → `recall.deletion_findings()` → confirmation utterance | 3 seams | The confirmation must be published **after** deletion returns and after the findings verify it, or the robot promises something that has not happened |
| **R1.6** | The output guard | `OutputGuard` into the say path; verdict routes through the existing `/speech/interrupt` → `T_FLUSH`; `clamp_reply()`, `gesture_findings()` and `spoken_text()` replace the inline regexes at `agent_node.py:264-269` | 2 seams + **the 2 `defect` rows** | Last, because it is the only one that can be structurally correct and still fail on real latency. It is also the only one that removes a **duplicate implementation** rather than adding a call — see [`docs/design.md`](design.md) §3, D1. It depends on `ws_gateway.py:72`, the one line mapping `/speech/interrupt` to the `T_FLUSH` byte, which **no test asserts today** (§7) |
| **R1.7** | `/game/events` — **decide, don't defer** | Either wire it (`game_node` accepting browser events, which today it does not: it takes commands only on `/game/cmd`) or reclassify it | 1 graph gap | This row is `pending` and **names no stage**. It became visible only by deriving R1's coverage from the manifests instead of from the plan, and it is the one pending gap the six PRs above do not close. Its own reason says "voice is the wired path", which is an argument for reclassifying to `unused` with a written reason — but leaving it `pending` forever is the failure mode the class set exists to prevent |

**Order, and why it is a dependency order rather than a preference.** R1.1 is smallest and
carries the heaviest promise. R1.2 must wire both ends. R1.3 must precede R1.4 (recall has
nothing to retrieve until episodes are written) and precedes R5 entirely (§8). R1.6 is last.

---

## 5. R2 — deletion candidates (list only, nothing removed without a per-item yes)

`docs/deletion-candidates.md`, one entry each with: the evidence, who is affected, and which
gate would prove nothing broke.

1. **The `/speech/tts_done` publisher** (`speech/tts_node.py`). `infra/graph_gaps.tsv`
   classes it `unused` and writes "a published signal looking for a reason to exist", and
   that class is defined as one that should stay empty. The half-duplex gate uses
   `/speech/tts_active`; the Art 50(2) marking rides `/speech/tts_started`.
2. **The duplicated harness scaffolding** across five harnesses, once R3's shared contract
   lands — estimated 300–400 lines. Not before.
3. **Research-round content: nothing is deleted.** The rounds are measured evidence and stay
   verbatim; R0b-2 adds status markers only.

**Removed from this list.** An earlier draft listed `memory/rank_harness.py` (221 lines) as a
deletion candidate on the grounds of "zero tests". That premise is false: `tests/test_rank.py`
imports it and asserts its JSON output. See [`docs/design.md`](design.md) §5.

---

## 6. R3 — converge the two duplicated mechanisms

1. **One harness contract.** A module holding the closed exit vocabulary
   (`0 ok / 1 defect / 2 no-claim / 3 cannot-run`), `emit(report, path)`, and the `ran`-flag
   rule; five harnesses migrate onto it. The gate derives each harness's exit-code set **from
   the module** rather than reading five hard-coded copies; a harness with no `ran` flag is a
   finding. Today the vocabulary is written down in **1 of 5** harnesses,
   `safety/measure_stats.py` is shared by 2 of 5, and `infra/day0_measure.py` has **no verdict
   function at all** — so it can neither fail nor be checked.
2. **A T-ID namespace gate** (`tests/test_design_ids.py`), three directions: every ID cited
   under `tests/` must be a matrix row; every ID §0 assigns to a file must be cited **in that
   file**; every defined ID must be either assigned or listed in a named `NOT_IMPLEMENTED`
   table with a reason. Range notation (`T-A1..T-A4`) needs a positive control, and section
   tracking must be **load-bearing** — `T-F5` appears once in §2.F and once in the L4 evidence
   table, so removing section tracking must go red.
3. **Layer markers.** `pytest.ini` gains `l1` / `l2` / `l4` / `l5`; the gate requires exactly
   one layer marker per test — **an unmarked test is red, not implicitly L1** — and
   `docs/test-design.md` §3's per-layer membership is derived from markers instead of written
   by hand. The honest cost: every existing case must be marked once, and that cost is
   already sunk — a field is free to mandate exactly once, before the first writer exists,
   and the first test was written in S0.

---

## 7. R4 — test the hardware seams

Five files, in this order, because the order is derived from what depends on them:

1. **`gateway/ws_gateway.py:72`** — the `/speech/interrupt` → `T_FLUSH` mapping. **Highest-value
   single assertion in the repository right now**, because R1.6 routes the output guard's
   verdict through it and nothing asserts it today.
2. `speech/asr_node.py` — T-B2's premise ("publishes nothing during silence") is currently
   stated only in a *comment in another file* (`tests/test_segment.py:331` cites
   `asr_node.py:79-87` in prose). No assertion has ever read that file.
3. `perception/hand_node.py` — the sampling window and reward adjudication: the only non-LLM
   reward signal in the system, hence the legitimacy of the whole game layer.
4. `body/body_node.py` — the actuator end of the gesture path, which is where D1's containment
   currently lives.
5. `perception/detect_node.py` — vision.

---

## 8. R5 — a local first tier (deferred, and the state of each sub-question is not "no")

Not started, and deliberately not decided here: the three-state breakdown is in
[`docs/design.md`](design.md) §4.1–§4.3. In summary — routing is **refuted**, latency masking
is **deferred with named preconditions**, offline availability is **unevaluated**.

**Its first step is not choosing a model.** Both live sub-questions need real conversation
logs to measure: discourse-connective coverage decides whether the masking gain is 45.4% or
17.5% (T-K2), and "which promises survive a network partition" cannot even be enumerated
before there is episode data. So R5 begins with a pure analysis script run **after R1.3** —
zero cost, zero new components.

**One hard placement rule that any local-first-tier proposal must answer first.** Round 7 §1.5
measured that a cheap model upstream of a cloud model creates a relay channel: at the
**memory-writing node**, the cheap model relayed injections at ~**53%** against **0/164** for
the strong model. Since R1.3 is precisely the PR that connects `/speech/asr` to
`store.append`: **no local first tier may sit upstream of the memory write path.**

---

## 9. R6.1 — the tool loop's on-robot half (definition)

This stage exists because a manifest row demands it:
`infra/dormant_seams.tsv` classes `agent/tools.py::reconcile()` as `pending_stage` naming
**R6.1**, and no stage of R0–R5 touches it. Defining it is what makes that row checkable
rather than decorative.

**What it is.** S8's in-machine side. S8's offline half shipped: `agent/tools.py` declares a
closed registry, `tests/test_tools.py` holds it closed (62 cases per `docs/test-design.md`
§0), and `agent_node.py` already runs a bounded tool loop — `toolConfig` is attached only on
turns where a skill fired, `MAX_TOOL_HOPS = 3`, `MAX_EFFECT_HOPS = 0`. What does not exist is
anything that **connects to a server**: no MCP client, no OAuth flow, no Strands integration.

**Scope, and equally what is out of scope:**

| In R6.1 | Not in R6.1 |
|---|---|
| An MCP client that can reach one server and hand its offer set to `reconcile()` | Any second tool in `REGISTRY`. Widening the surface is a separate, per-tool decision |
| The OAuth device / authorization-code flow and where the token lives (**not** in the tree; the repo's rule is that account-specific values are git-ignored) | Raising `MAX_EFFECT_HOPS` above 0. A `MUTATES` tool is unreachable by construction, and that number is changed by editing `agent/tools.py` in a PR, never by configuration |
| Wiring `EGRESS[MCP_TOOL_CALL]` from `NO_ENFORCER_TODAY` to a real enforcer — `invoke()` already reads that row, so the first remote tool is refused until something is written there | Anything on the `SAY` channel. That is R1.6 |
| A test that a server offering **undeclared** names produces findings and **zero** callable tools | The privateEndpoint / private-IdP work; unrelated |

**Entry condition.** R6.1 cannot start before R1.6, for a reason that is about ordering and
not tidiness: `EGRESS` is a single table, and R1.6 changes the `SAY` row from `NOT_WIRED` to
enforced. Adding a remote-tool enforcer to the same table first would mean the repository's
most important egress channel is unguarded while its newest one is guarded — and the table's
own tests read every row, so the two stages would collide in the same assertions.

**Exit criteria.** `reconcile()` has a production caller; the `MCP_TOOL_CALL` row names a real
enforcer; a server offering names outside `REGISTRY` yields findings and no callable tool; and
the `invoke()` result vocabulary still distinguishes `REFUSED` (a policy claim) from
`UNREACHABLE` (a network claim), because a report that counts the second as the first says the
robot was protected by an outage.

---

## 10. Verification

Every PR, before pushing, on the pinned 3.12 interpreter (every gate here shells out to
`python3`, and a homebrew relink to a Python without pytest reports "No module named pytest"
rather than a red test):

```
pytest                       all cases, rc 0
pytest -m mutation           the mutation layer, rc 0
infra/check_deploy.sh        rc 0, 0 findings
infra/doc_counts.py          cases / mutants / modules reconciled three ways, rc 0
infra/wiring_census.py       dormancy census, both directions, rc 0
infra/redact_scan.py         against .push/files.txt, rc 0
tests/test_doc_status.py     (from R0b-2) document status table, both directions
```

Counts are not repeated here — they live in `docs/test-design.md` §3 and are re-measured per
round, because a duplicated count is a second thing to forget. `pytest -m "not mutation"` is
re-timed every round for the same reason: R3's markers change each layer's membership, and a
budget that was not re-measured is prose.

**Push discipline:** never `git push`. GitHub Git Data API only, every blob verified against
`git hash-object` and read back byte-for-byte, redaction scan before the push, and
`README.md` / `README.zh-TW.md` in the same PR.

---

## 11. What is blocked on a human

Four items, none of which block R0–R4. Each one's specific consequence — which tests it
holds up, and why that matters — is in [`docs/design.md`](design.md) §6: the security-group
`/32`, real logs plus one human annotation pass, the manual UAT of the SVG animation in
GitHub's own viewer, and the `RETENTION_DAYS = 90` policy decision.
