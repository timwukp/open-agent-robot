# Document index — what is current, what is evidence, what has been overturned

This file answers one question: **which of these documents may I cite as the design, and
which are dated measurements that a later round has partly retired?**

It exists because of a measurement, not a principle. On 2026-08-25 `docs/` held six
research rounds and one design critique — **191 KiB together** — and **four of the six
carried no supersession marker of any kind**. One of the four is round 4, whose §0 build
order the critique replaces outright and whose memory pick round 5 re-answers in its own
header. Someone who opened round 4 first had no way to learn either fact. The rounds are
kept verbatim on purpose, because they are measured evidence and the measurements are
still good; so the retraction has to live outside them, in a place a machine checks.

**Read this way:**

- **current** — cite it as the design. Start at [`docs/design.md`](design.md).
- **evidence** — cite it for *how a number was obtained*, never for what the design is
  today. The measurement is still good; a conclusion drawn from it may not be.
- **overturned_in_part_by** — some conclusion here is dead and the rest still stands.
  Follow the link before quoting anything from that document.

Three retractions are important enough that they are written out in full in
[`docs/design.md` §4.4](design.md) rather than left for a reader to reach on page 12: a
misfiled arXiv id, a percentage pair that its cited source explicitly does not contain,
and two transposed dates in the pipeline record.

## The table

`infra/doc_status.tsv` is the machine-readable form of this table, and
`infra/doc_status.py` checks the two against each other in both directions — a row here
with no row there is a finding, and so is the reverse. Two readers of one set of facts
will otherwise each stay green while the other side moves.

| document | status | relation | why it is kept |
|---|---|---|---|
| [`docs/INDEX.md`](INDEX.md) | current | `-` | this file; a reader lands on the status list before opening any round |
| [`docs/design.md`](design.md) | current | `-` | the single authority for what this system is trying to be |
| [`docs/design.zh-TW.md`](design.zh-TW.md) | current | `-` | the Traditional Chinese counterpart, shipped in the same PR by convention |
| [`docs/plan.md`](plan.md) | current | `-` | the schedule, and what each stage must delete from a manifest |
| [`docs/test-design.md`](test-design.md) | current | `-` | the test contract: the §0 case table, the mutant rule, the five layers |
| [`docs/DEMO.md`](DEMO.md) | current | `-` | the runbook for the one promise that runs end to end today |
| [`docs/workflow.md`](workflow.md) | current | `-` | the sequence charts, counterpart to `docs/architecture.svg`'s standing shape |
| [`docs/day0-measurements.md`](day0-measurements.md) | evidence | `-` | the dated day-0 gates on the instance, including the 900 ms Bedrock TTFT |
| [`docs/design-critique-memory-safety-2026-08-22.md`](design-critique-memory-safety-2026-08-22.md) | evidence | `-` | the round 4 critique — an overturner, not something overturned |
| [`docs/research-multimodal-robot-agent-stack-2026-08-20.md`](research-multimodal-robot-agent-stack-2026-08-20.md) | evidence | `overturned_in_part_by:docs/research-round3-feasibility-2026-08-20.md`;`overturned_in_part_by:docs/research-round6-simreal-joint-mapping-2026-08-22.md` | rounds 1–2, the stack survey; its hard no-AGPL constraint became a demo-only waiver, and its xacro verdict was flipped |
| [`docs/research-round3-feasibility-2026-08-20.md`](research-round3-feasibility-2026-08-20.md) | evidence | `-` | the feasibility red-team; its YOLO26 waiver and TTFT measurement are both still operative |
| [`docs/research-round4-agent-brain-2026-08-21.md`](research-round4-agent-brain-2026-08-21.md) | evidence | `overturned_in_part_by:docs/design-critique-memory-safety-2026-08-22.md`;`overturned_in_part_by:docs/research-round5-memory-safety-2026-08-22.md` | the agent-brain round; its §0 build order was replaced and its memory pick re-answered |
| [`docs/research-round5-memory-safety-2026-08-22.md`](research-round5-memory-safety-2026-08-22.md) | evidence | `-` | the memory and moral-boundary round; round 6 independently corroborates its guard finding |
| [`docs/research-round6-simreal-joint-mapping-2026-08-22.md`](research-round6-simreal-joint-mapping-2026-08-22.md) | evidence | `overturned_in_part_by:docs/research-round7-rl-and-two-tier-2026-08-22.md` | the sim-to-real round; it already carries two in-file round 7 corrections |
| [`docs/research-round7-rl-and-two-tier-2026-08-22.md`](research-round7-rl-and-two-tier-2026-08-22.md) | evidence | `-` | the newest round; nothing overturns it, but its own appendix A retracts three citations |
| [`docs/evidence/round7/README.md`](evidence/round7/README.md) | evidence | `-` | the index of round 7's raw machine output, recovered from `journal.jsonl` |
| [`docs/evidence/t-e3-browser-disclosure.md`](evidence/t-e3-browser-disclosure.md) | evidence | `-` | the dated L5 browser walk-through a human had to run by eye |

**17 documents**, and that number is checked rather than typed: `infra/doc_status.py`
counts `docs/**/*.md` recursively and reports a finding if this sentence disagrees. The
recursion is the point — `docs/evidence/` holds two of the seventeen, and a
non-recursive glob would have exempted that directory while reporting clean.

## What `superseded` is for, and why it is empty

`status` has a third value this table does not use. Nothing here has been *wholly*
replaced, because evidence does not expire — only conclusions drawn from it do. The class
exists so that the day a document really is finished, deleting it becomes a reviewable
question with a named replacement rather than a habit. A closed set with an empty class
is a prediction the guard will keep honest; a class invented on the day it is first
needed is a decision made under pressure.
