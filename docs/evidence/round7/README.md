# Round 7 raw research evidence

Machine output behind [`docs/research-round7-rl-and-two-tier-2026-08-22.md`](../../research-round7-rl-and-two-tier-2026-08-22.md).
Two `deep-research` workflows ran in parallel on 2026-08-22 (213 agent calls total): workflow 1 =
two-tier / small-model architecture, workflow 2 = reinforcement learning at five intervention points.

| File | Size | What it is |
|---|---|---|
| `workflow1-recovered-claims.txt` | 82 KB | **Start here for workflow 1.** All 125 extracted claims, each with source URL, source quality, publish date, importance and a verbatim quote (≤800 chars) |
| `workflow2-recovered-claims.txt` | 77 KB | **Start here for workflow 2.** Same format, 120 claims |
| `workflow1-two-tier.output.json` | 229 KB | Workflow 1 final tool output (logs + stats + synthesis) |
| `workflow2-rl.output.json` | 239 KB | Workflow 2 final tool output |
| `workflow1-journal.jsonl` | 563 KB | Workflow 1 per-agent journal, 214 records — the ground truth the synthesis summarized |
| `workflow2-journal.jsonl` | 525 KB | Workflow 2 per-agent journal, 213 records |

## Why the raw journals are kept

**Both syntheses under-reported their own coverage.** Workflow 1 reported that three of five search
angles "produced ZERO evidence"; workflow 2 reported that intervention points 3/4/5 "produced ZERO
surviving claims". Both statements are false. The run statistics contradict them:

| | workflow 1 | workflow 2 |
|---|---|---|
| search angles | 5 | 5 |
| sources fetched | 25 | 24 |
| claims **extracted** | **125** | **120** |
| claims **verified** | 25 | 25 |
| confirmed / refuted | 1 / 24 | 5 / 20 |
| dropped by verification budget | 3 | 2 |
| agent calls | 107 | 106 |

The gap is between *extracted* and *verified*, i.e. the claims were **located but unadjudicated** —
not missing. Recovering them is what the two `*-recovered-claims.txt` files are.

## How the claims were recovered

```python
import json, re
started, recs = {}, []
for line in open("workflow2-journal.jsonl"):
    d = json.loads(line)
    if d.get("type") == "started":
        started[d.get("agentId")] = d
    elif d.get("type") == "result":
        recs.append(d)

def urlof(d):                      # claims records carry no URL of their own
    blob = json.dumps(started.get(d.get("agentId"), {}), ensure_ascii=False)
    m = re.findall(r'https?://[^\s"\\]+', blob)
    return m[0] if m else "?"

for d in recs:
    for c in d["result"].get("claims", []):
        print(urlof(d), c["importance"], c["claim"], c["quote"])
```

Take `type == "result"` records whose `result` object has a `claims` key; each claim carries `claim`,
`importance` and `quote`. Source URLs are **not** in the result record — reverse-look them up from the
`started` record with the same `agentId`.

## Reading the refutation votes

**Treat source-level checks as signal and vote counts as noise.** 24 of 25 (workflow 1) and 20 of 25
(workflow 2) claims were voted down, yet 5 of 6 cited arXiv IDs exist with matching titles and two had
their numbers re-verified verbatim — several 0–3 results are fetch/CAPTCHA failures, not refutations.
One self-contradiction proves it: the same 1-shot-RLVR paper's "one example suffices" was 0–3 refuted
while its own sub-claim "most of the gain is format correction" was 2–1 confirmed, and the latter
presupposes the former.

Two refutations *were* correct and load-bearing, and both are recorded in §8 of the report: the
Marlin-on-Turing claim came from vLLM v0.6.1 (2024) documentation, and arXiv 2607.24440 is a
vision-language image-degradation paper that the pipeline mis-scoped as text-LLM results, voiding
three quantization/confidence claims drawn from it.

## Redaction

Absolute home paths were rewritten (`/Users/<user>` → `~`, session-slug `-Users-<user>-` →
`-Users-home-`); 12 occurrences across four files. Scanned clean for instance IDs, 12-digit account
IDs, ARNs, `.pem` filenames, EC2 public DNS names, access keys and IPv4 addresses. The only
IPv4-shaped string that remains is `v0.10.1.1`, a vLLM version.
