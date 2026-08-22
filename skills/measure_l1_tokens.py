#!/usr/bin/env python3
"""Measure the REAL input-token cost of the resident L1 skill index (test T-I1).

Why this exists as a separate script instead of a test: the CI assertion uses a
character proxy (skills/loader.approx_tokens), and a proxy nobody ever compared
against a real tokenizer is a number that only looks measured. So the real figure
is taken once, by hand, against the same model agent_node uses, and pasted into
tests/test_skills_loader.py next to the assertion it justifies.

Method: two Converse calls, identical except that one carries the L1 index in the
system block. The difference in `usage.inputTokens` is the L1 cost. Measuring the
difference rather than the absolute count subtracts the fixed per-request
overhead, which is not ours and would inflate the figure.

    python skills/measure_l1_tokens.py

Costs two ~200-token requests (well under a cent). Needs Bedrock credentials and
is never run by the test suite or by any deploy stage — the runner does not
launch it, which is why it is declared in infra/standalone_entrypoints.tsv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills import loader  # noqa: E402

MODEL_ID = "global.anthropic.claude-sonnet-5"   # must match agent/agent_node.py
REGION = "ap-southeast-1"
PROBE = "hi"


def _input_tokens(client, system_text: str) -> int:
    kwargs = {
        "modelId": MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": PROBE}]}],
        "inferenceConfig": {"maxTokens": 1},
    }
    if system_text:
        kwargs["system"] = [{"text": system_text}]
    resp = client.converse(**kwargs)
    return int(resp["usage"]["inputTokens"])


def main() -> int:
    import boto3

    skills = loader.discover()
    if not skills:
        print("no skills discovered — nothing to measure", file=sys.stderr)
        return 3
    index = loader.l1_index(skills)

    client = boto3.client("bedrock-runtime", region_name=REGION)
    baseline = _input_tokens(client, "")
    withl1 = _input_tokens(client, index)

    measured = withl1 - baseline
    proxy = loader.approx_tokens(index)
    per_skill = {s.name: loader.approx_tokens(s.l1_line) for s in skills}
    print(json.dumps({
        "model": MODEL_ID,
        "region": REGION,
        "skills": [s.name for s in skills],
        "index_chars": len(index),
        "baseline_input_tokens": baseline,
        "with_l1_input_tokens": withl1,
        "measured_l1_tokens": measured,
        "proxy_l1_tokens": proxy,
        "proxy_over_measured": round(proxy / measured, 3) if measured else None,
        "proxy_tokens_per_skill_line": per_skill,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
