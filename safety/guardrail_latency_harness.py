#!/usr/bin/env python3
"""T-K3: what Bedrock Guardrails actually costs us in latency, in our own region.

`docs/test-design.md` §5 item 3 and the S7 row both rest on ONE number that is not
ours: a third party's client-side p50 749 / p90 907 / max 1837 ms at n=15. The whole
S7 design — asynchronous guard plus the existing `/speech/interrupt` kill-voice path
rather than a serial pre-check — follows from that number being large. A design
premise taken from someone else's n=15 is worth measuring before more is built on it.

What this measures, and what it does not:

  * `apply_guardrail` wall time per call, source=OUTPUT (S7 guards what the robot is
    about to SAY, not what it hears), n >= 30 per language group per run, reported
    p50/p95/max with n per §6.0 rule 4.
  * The whole thing runs TWICE by default and both runs are kept. One run cannot tell
    you whether a ~30 ms gap between two language groups is the language or the day:
    the between-group difference is only interpretable against the run-to-run
    difference of the SAME group, so that comparison is computed and stored rather
    than left for a reader to eyeball. Budget comparisons use the pooled samples.
  * Warm-up calls are discarded from the statistics but RECORDED. A discarded sample
    justified by a number that lives only in a comment is a number nobody can check
    (§6.0 rule 6); `warmup_samples_ms` puts the justification in the evidence file.
  * Three groups: Mandarin, English, Cantonese. Cantonese is measured for LATENCY
    only. Guardrails does not list Cantonese support, so a latency number for it says
    nothing about whether anything would be blocked — `docs/test-design.md` T-G4 is
    explicit that no report may claim Cantonese is protected, and a fast call on an
    unsupported language is the easiest way to accidentally imply it is.
  * Cost per call comes from the response's own `usage` text-unit counts, never from
    the latency or from a token estimate (T-F4: each metric derived from its own
    source). The USD figure is left to the caller: this writes units, and the rate.

Timeouts are counted separately and never scored as errors (§6.0 rule 4).

Run by hand against a guardrail that already exists; the identifier is passed in and
NEVER written to the evidence file — a guardrail id and its ARN carry the account id,
and `infra/redact_scan.py` would reject the push.

Standard tier is the tier S7 specifies, because Classic covers en/fr/es only — nothing
this robot speaks. The tier is READ from `bedrock:GetGuardrail` rather than asserted
here: Standard cannot be enabled without a cross-Region guardrail profile, so the
guardrail you get by creating one the obvious way is Classic, and a hard-coded
"Standard" in the output would be a latency number for the wrong tier that no test
could catch. See `read_tier`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "safety"))
from measure_stats import budget_verdict, median_difference_ci, percentiles  # noqa: E402

REGION = "ap-southeast-1"
READ_TIMEOUT = 30.0
# S7's budget: the guard runs asynchronously and cuts the voice through T_FLUSH, so
# the number that matters is verdict-to-silence <= 400 ms, of which this call is the
# part we do not control. Recorded as the comparison, not as a gate.
S7_BUDGET_MS = 400.0

# One typical robot utterance per group. Deliberately similar in length so that a
# latency difference between groups is about the language, not about the text size.
UTTERANCES = {
    "mandarin": "我現在要停下來,不再繼續動作,等你告訴我下一步。",
    "english": "I am stopping now and will not move again until you tell me to.",
    "cantonese": "我而家停低,唔會再動,等你話我知下一步做乜。",
}


def _client():
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(read_timeout=READ_TIMEOUT,
                                      retries={"max_attempts": 1}))


def _control():
    return boto3.client("bedrock", region_name=REGION)


def read_tier(ctl, guardrail_id: str, version: str) -> dict:
    """The tier, from the control plane rather than from a string in this file.

    This matters more than it looks. Standard tier is the only tier that covers the
    languages this robot speaks (Classic is en/fr/es), and it cannot be enabled without
    cross-Region inference — so a guardrail created the obvious way is CLASSIC, and a
    harness that simply prints "STANDARD" produces a latency number for a tier the
    system cannot use while every test that checks the tier still passes.

    The profile id is recorded; its ARN is NOT — the ARN carries the account id.
    """
    described = ctl.get_guardrail(guardrailIdentifier=guardrail_id,
                                  guardrailVersion=version)
    tier = described.get("contentPolicy", {}).get("tier", {}).get("tierName")
    profile = described.get("crossRegionDetails", {}).get("guardrailProfileId")
    return {"tier": tier, "cross_region_profile": profile,
            "tier_read_from": "bedrock:GetGuardrail contentPolicy.tier.tierName"}


def apply_seconds(rt, guardrail_id: str, version: str, text: str, now=time.monotonic):
    """(seconds, action, text_units) for one OUTPUT check; seconds None on timeout.

    `now` is injectable for the reason `docs/test-design.md` §1 gives: a test that has to
    produce a 40 ms latency by really sleeping for 40 ms is asserting on the scheduler.
    That is not hypothetical here — the language-gap mutant test was written with a
    sleeping double and a `30 ms ± 15` assertion, and it read 46.0 ms on a loaded laptop
    while passing in isolation. In production `now` is the real monotonic clock; that is
    the whole point of the harness. In a test the double advances a fake one.
    """
    start = now()
    try:
        resp = rt.apply_guardrail(
            guardrailIdentifier=guardrail_id, guardrailVersion=version,
            source="OUTPUT", content=[{"text": {"text": text}}])
    except Exception as exc:                       # noqa: BLE001 — a timeout is data
        if "ReadTimeout" in type(exc).__name__ or "timeout" in str(exc).lower():
            return None, "TIMEOUT", 0
        raise
    elapsed = now() - start
    units = sum(v for k, v in resp.get("usage", {}).items() if k.endswith("Units"))
    return elapsed, resp.get("action", "UNKNOWN"), units


CAVEATS = {
    "cantonese": "latency only — Guardrails does not list Cantonese, so no claim is "
                 "made here that Cantonese output is protected (T-G4)",
}

BUDGET_WHAT = "apply_guardrail alone vs the whole S7 verdict-to-silence budget"


def measure(rt, guardrail_id: str, version: str, n: int, warmup: int = 2,
            now=time.monotonic) -> dict:
    """Per-language latency for ONE run. Warm-up calls are made, kept out of the
    statistics, and recorded: the first `apply_guardrail` of a session is slower than
    the steady state, and `warmup_samples_ms` is what lets a reader check that claim
    instead of taking a comment's word for it."""
    out = {"region": REGION, "source": "OUTPUT",
           "guardrail_version": version, "requested_n_per_group": n,
           "warmup_calls_discarded": warmup, "read_timeout_s": READ_TIMEOUT,
           "note": "guardrail identifier deliberately absent — it carries the account id",
           "groups": {}}
    for group, text in UTTERANCES.items():
        warmups = []
        for _ in range(warmup):
            seconds, _action, _unit = apply_seconds(rt, guardrail_id, version, text,
                                                    now=now)
            warmups.append(None if seconds is None else round(seconds * 1000, 1))
        samples, timeouts, actions, units = [], 0, set(), 0
        for _ in range(n):
            seconds, action, unit = apply_seconds(rt, guardrail_id, version, text,
                                                  now=now)
            actions.add(action)
            units += unit
            if seconds is None:
                timeouts += 1
            else:
                samples.append(seconds)
        summary = percentiles(samples)
        out["groups"][group] = {
            **summary,
            "chars": len(text),
            "warmup_samples_ms": warmups,
            "timeouts_not_scored_as_errors": timeouts,
            "actions_seen": sorted(actions),
            "text_units_billed": units,
            "budget": budget_verdict(summary, S7_BUDGET_MS, BUDGET_WHAT),
        }
        if group in CAVEATS:
            out["groups"][group]["coverage_caveat"] = CAVEATS[group]
    return out


def _seconds(group: dict) -> list[float]:
    """The stored millisecond samples back as seconds, for pooling across runs."""
    return [ms / 1000.0 for ms in group["samples_ms"]]


def measure_runs(rt, guardrail_id: str, version: str, n: int, runs: int = 2,
                 warmup: int = 2, tier_info: dict | None = None,
                 now=time.monotonic) -> dict:
    """Repeat the whole measurement and keep every run.

    The reason this exists rather than one longer run: the interesting comparison in
    the result is BETWEEN language groups, and a single run gives no scale to judge it
    on. Re-measuring the same group is that scale. Pooled samples carry the budget
    comparison, because a p95 over 2n calls has a tail two runs wide rather than one.
    """
    assert runs >= 1
    blocks = [{"run": i + 1,
               "groups": measure(rt, guardrail_id, version, n, warmup,
                                 now=now)["groups"]}
              for i in range(runs)]
    out = {"region": REGION, "source": "OUTPUT",
           **(tier_info or {"tier": None, "cross_region_profile": None,
                            "tier_read_from": "not read — no control-plane client"}),
           "guardrail_version": version, "requested_n_per_group": n,
           "runs_requested": runs, "warmup_calls_discarded": warmup,
           "read_timeout_s": READ_TIMEOUT,
           "note": "guardrail identifier deliberately absent — it carries the account id",
           "runs": blocks, "pooled": {"groups": {}}, "run_to_run": {}}
    for group in UTTERANCES:
        pooled_seconds = [s for b in blocks for s in _seconds(b["groups"][group])]
        summary = percentiles(pooled_seconds)
        entry = {
            **summary,
            "chars": len(UTTERANCES[group]),
            "pooled_over_runs": runs,
            "timeouts_not_scored_as_errors": sum(
                b["groups"][group]["timeouts_not_scored_as_errors"] for b in blocks),
            "actions_seen": sorted({a for b in blocks
                                    for a in b["groups"][group]["actions_seen"]}),
            "text_units_billed": sum(b["groups"][group]["text_units_billed"]
                                     for b in blocks),
            "budget": budget_verdict(summary, S7_BUDGET_MS, BUDGET_WHAT),
        }
        if group in CAVEATS:
            entry["coverage_caveat"] = CAVEATS[group]
        out["pooled"]["groups"][group] = entry
        if runs >= 2:
            first = _seconds(blocks[0]["groups"][group])
            out["run_to_run"][group] = {
                "later_minus_first_median": [
                    median_difference_ci(_seconds(b["groups"][group]), first)
                    for b in blocks[1:]],
                "why": "the scale a between-group difference has to be read against: "
                       "the same group re-measured. A between-group gap smaller than "
                       "this is a gap about the day, not about the language.",
            }
    medians = [g["p50_ms"] for g in out["pooled"]["groups"].values() if g["p50_ms"]]
    # The yardstick is the run-to-run INTERVAL, not its point estimate. A re-measurement
    # whose point shift is 13.9 ms but whose 95% CI reaches 47.7 ms can produce a 40 ms
    # gap out of nothing, so comparing one point against another would call a 27 ms
    # language gap real while the noise it is being compared to is twice that.
    bounds = [abs(b) for e in out["run_to_run"].values()
              for d in e["later_minus_first_median"] for b in d["ci95_ms"]]
    shifts = [abs(d["point_ms"]) for e in out["run_to_run"].values()
              for d in e["later_minus_first_median"]]
    spread = round(max(medians) - min(medians), 1) if medians else None
    ceiling = round(max(bounds), 1) if bounds else None
    out["interpretation"] = {
        "between_group_p50_spread_ms": spread,
        "run_to_run_noise_ceiling_ms": ceiling,
        "largest_run_to_run_p50_point_shift_ms": round(max(shifts), 1) if shifts else None,
        "language_gap_exceeds_replication_noise":
            None if spread is None or ceiling is None else spread > ceiling,
        "note": "when false, the per-language latency differences in this file are NOT "
                "attributable to the language — a repeat of the same group can move the "
                "median that far by itself. The ceiling is the widest 95% bound of a "
                "same-group repeat, not its point estimate, because the point estimate "
                "of pure noise is near zero and would make any gap look real. Honest "
                "limit: the repeats are within one session minutes apart, so this bounds "
                "short-term noise only — it says nothing about day-to-day drift. Either "
                "way this is a measurement against a yardstick, never a correctness "
                "claim (§6.0 rule 4).",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrail-id", required=True,
                    help="not recorded in the output; it carries the account id")
    ap.add_argument("--guardrail-version", default="DRAFT")
    ap.add_argument("--n", type=int, default=30, help="calls per language group per run")
    ap.add_argument("--runs", type=int, default=2,
                    help="independent repeats; 2 is the minimum that gives a "
                         "between-group difference anything to be compared against")
    ap.add_argument("--out", default="docs/evidence/track-c")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tier_info = read_tier(_control(), args.guardrail_id, args.guardrail_version)
    result = measure_runs(_client(), args.guardrail_id, args.guardrail_version,
                          args.n, args.runs, tier_info=tier_info)
    (out_dir / "guardrail-latency.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
