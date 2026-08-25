#!/usr/bin/env python3
"""Track C: what a resident prompt block actually costs, in real `usage.inputTokens`.

Two measurements, one on-machine run each, JSON out (§6.0 rule 6 — a number typed
into prose is unverified):

  M1  the char-to-token ratio for OUR text.  `tests/test_guard.py` asserts the
      constitution's size in CHARACTERS, because CI must not call Bedrock. A
      character cap is only a token budget if the ratio is measured. One such
      calibration already exists — `skills/measure_l1_tokens.py` measured the L1
      index on 2026-08-22 and found `skills.loader.approx_tokens` under-counts by
      21% — so this measurement is also a REPLICATION on a second, unrelated text:
      a calibration taken once on one string is a coincidence until it holds twice.
      Method: one baseline call with no system block, then one call per sample; the
      delta in `usage.inputTokens` is that sample's cost, reported next to the
      proxy's prediction for the same string. Tokenisation is not exactly additive,
      so a delta carries the baseline's boundary effects — at these sizes that is a
      per-sample error of a token or two, which is reported rather than hidden.

  M2  first-token latency under `output_config.effort` (§0-8 / T-K5). The plan
      claims low effort buys latency with zero new components. Measured with
      `converse_stream`, timing the first delta that carries text, n >= 30 per
      condition. Per §6.0 rule 4: p50/p95/max are reported WITH n, and a call that
      times out is counted separately and never scored as an error.

  M3  whether the parameter M2 varies is HONOURED at all. Without this, M2's null
      result is worthless: "effort makes no difference to latency" and "the field was
      accepted and silently ignored" produce identical evidence, and this project has
      already been bitten once by a request field that was a no-op. Two independent
      channels, neither of them latency: (a) send a garbage value and an unknown key —
      a parameter that accepts nonsense is not being read; (b) count
      `usage.outputTokens` per effort level, which is a different quantity from a
      first-token clock and therefore cannot be confounded with it. `unset` is measured
      too — and it is the comparison that decides the verdict, because the deployment
      choice is `effort: low` versus sending nothing, not low versus an explicit high.
      Differences are bootstrap intervals, and an interval straddling 0 is recorded as
      UNRESOLVED rather than as equality. Truncation is read from `stopReason`, not
      inferred: a condition that stopped at `max_tokens` is longer than it looks, which
      biases the measured saving DOWNWARD, so the file records the bias direction.

All three are level M: they compare against a budget, they do not prove correctness.
Run from the repo root. Costs cents; needs Bedrock access in the node's region.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "skills"))
from loader import approx_tokens  # noqa: E402  the repo's single character proxy
from measure_stats import (median, median_difference_bootstrap,  # noqa: E402
                           median_difference_ci, percentiles)

MODEL_ID = "global.anthropic.claude-sonnet-5"   # agent/agent_node.py:37
REGION = "ap-southeast-1"                       # agent/agent_node.py:38
PROBE = "用一句話說你好。"
READ_TIMEOUT = 30.0                             # a timeout is data, not an error

# M2's prompt. The "三句話" is harmless for a first-token clock — the instruction cannot
# change when the first token arrives — and it keeps the calls cheap.
EFFORT_PROMPT = "用三句話說明為什麼機器人應該在被喊停時立刻停下。"

# M3 needs a DIFFERENT prompt, and this is the measured reason rather than a preference:
# run against EFFORT_PROMPT, every condition landed at 112-263 output tokens with fully
# overlapping ranges, because a prompt that says "three sentences" fixes the length and
# leaves `effort` nothing to express itself in. A no-op field and an honoured field look
# identical under that prompt — the same confound M3 exists to remove. So M3 asks an
# open question with no length instruction, and `no_length_instruction` below is a test,
# not a comment.
EFFORT_LENGTH_PROMPT = ("如果機器人在移動途中同時聽到有人喊「停」,又看到旁邊有人跌倒,"
                        "它應該先做哪一件事?請說明你的判斷。")
EFFORT_LENGTH_MAX_TOKENS = 900
# Closed list of the length-dictating phrases our prompts actually use. A prompt for M3
# containing any of these would cap the quantity M3 measures.
LENGTH_INSTRUCTIONS = ("一句", "兩句", "二句", "三句", "簡短", "字以內", "briefly",
                       "one sentence", "in a sentence")
# Not a plausible-looking near-miss: a value the API cannot possibly mean.
GARBAGE_EFFORT = "definitely-not-an-effort-level"


def _client():
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(read_timeout=READ_TIMEOUT, retries={"max_attempts": 2}))


# --------------------------------------------------------------- M1 token ratio
def _samples() -> list[tuple[str, str]]:
    """(name, text) — the file as shipped, and the same file with the machinery
    stripped, so the ratio is attributed to the prose rather than to markdown."""
    raw = (REPO_ROOT / "safety" / "constitution.md").read_text()
    stripped = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    stripped = re.sub(r"^[#\-*\s]+", "", stripped, flags=re.M).strip()
    return [("constitution_as_shipped", raw), ("constitution_prose_only", stripped)]


def measure_token_ratio(brt) -> dict:
    def call(system: str | None) -> int:
        kw = {"modelId": MODEL_ID,
              "messages": [{"role": "user", "content": [{"text": PROBE}]}],
              "inferenceConfig": {"maxTokens": 16}}
        if system is not None:
            kw["system"] = [{"text": system}]
        return brt.converse(**kw)["usage"]["inputTokens"]

    baseline = call(None)
    out = {"model": MODEL_ID, "region": REGION, "baseline_input_tokens": baseline,
           "method": "inputTokens(system=sample) - inputTokens(no system block)",
           "proxy": "skills.loader.approx_tokens — the ONE proxy in this repo, so "
                    "that a second budget is not a second calibration",
           "samples": []}
    for name, text in _samples():
        tokens = call(text) - baseline
        proxy = approx_tokens(text)
        out["samples"].append({
            "name": name, "chars": len(text), "tokens": tokens,
            "tokens_per_char": round(tokens / len(text), 4),
            "proxy_tokens": proxy,
            # The number tests/test_skills_loader.py already carries as PROXY_RATIO
            # from a different string. Equal-ish here = a replicated calibration;
            # far apart = the proxy is text-dependent and neither budget is real.
            "proxy_over_real": round(proxy / tokens, 4) if tokens else None,
        })
    return out


# ------------------------------------------------------ M2 first-token latency
def first_token_seconds(brt, effort: str | None) -> float | None:
    """Seconds until the first delta carrying text; None on a read timeout."""
    kw = {"modelId": MODEL_ID,
          "messages": [{"role": "user", "content": [{"text": EFFORT_PROMPT}]}],
          "inferenceConfig": {"maxTokens": 256}}
    if effort is not None:
        kw["additionalModelRequestFields"] = {"output_config": {"effort": effort}}
    start = time.monotonic()
    try:
        for event in brt.converse_stream(**kw)["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if delta.get("text"):
                return time.monotonic() - start
    except Exception as exc:                      # noqa: BLE001 — a timeout is data
        if "ReadTimeout" in type(exc).__name__ or "timeout" in str(exc).lower():
            return None
        raise
    return None


def measure_effort_latency(brt, n: int) -> dict:
    out = {"model": MODEL_ID, "region": REGION, "requested_n_per_condition": n,
           "read_timeout_s": READ_TIMEOUT,
           "prompt_note": "one fixed Chinese prompt; no memory, no skills, maxTokens 256",
           "conditions": {}}
    raw: dict[str, list[float]] = {}
    for label, effort in (("effort_high_default", "high"), ("effort_low", "low")):
        samples, timeouts = [], 0
        for _ in range(n):
            got = first_token_seconds(brt, effort)
            if got is None:
                timeouts += 1
            else:
                samples.append(got)
        raw[label] = samples
        out["conditions"][label] = {**percentiles(samples),
                                    "timeouts_not_scored_as_errors": timeouts}
    high, low = raw["effort_high_default"], raw["effort_low"]
    if high and low:
        out["high_minus_low_median"] = median_difference_ci(high, low)
    out["verdict"] = effort_verdict(out)
    return out


# --------------------------------------------- M3 is the parameter honoured at all?
def rejects(brt, fields: dict) -> bool:
    """True if Bedrock refuses the request outright.

    This is the cheap half of M3 and it answers a question latency cannot: a field
    that accepts nonsense is a field nobody reads. Only `ValidationException` counts
    as a rejection — a throttle or an auth failure says nothing about the field, so it
    is re-raised rather than silently scored as "validated".
    """
    try:
        brt.converse(modelId=MODEL_ID,
                     messages=[{"role": "user", "content": [{"text": PROBE}]}],
                     inferenceConfig={"maxTokens": 8},
                     additionalModelRequestFields=fields)
    except Exception as exc:                       # noqa: BLE001 — a refusal is data
        if "ValidationException" in f"{type(exc).__name__} {exc}":
            return True
        raise
    return False


def output_length(brt, effort: str | None) -> tuple[int, str | None]:
    """(usage.outputTokens, stopReason) for one non-streaming call.

    `stopReason` is returned rather than inferred from the token count: a condition
    that stopped at `max_tokens` has a length that is a FLOOR, and the only honest way
    to know that is to read the field that says so.
    """
    kw = {"modelId": MODEL_ID,
          "messages": [{"role": "user", "content": [{"text": EFFORT_LENGTH_PROMPT}]}],
          "inferenceConfig": {"maxTokens": EFFORT_LENGTH_MAX_TOKENS}}
    if effort is not None:
        kw["additionalModelRequestFields"] = {"output_config": {"effort": effort}}
    resp = brt.converse(**kw)
    return resp["usage"]["outputTokens"], resp.get("stopReason")


def measure_effort_effect(brt, n: int) -> dict:
    """Whether `output_config.effort` is honoured, on two channels that are not clocks.

    `effort_unset` is measured because "the default is high" is itself a claim in
    §0-8, and it is the condition that would expose the field being ignored: if unset,
    low and high all produce the same lengths, the parameter does nothing.
    """
    out = {"why": "a null latency result and a silently ignored field produce identical "
                  "evidence; neither channel here is a latency measurement",
           "requested_n_per_condition": n,
           "max_tokens": EFFORT_LENGTH_MAX_TOKENS,
           "prompt_note": "an open question, NOT the latency prompt: measured against "
                          "the latency prompt every condition landed at 112-263 output "
                          "tokens with overlapping ranges, because '用三句話' fixes the "
                          "length and leaves effort nothing to change",
           "no_length_instruction": not any(p in EFFORT_LENGTH_PROMPT
                                            for p in LENGTH_INSTRUCTIONS),
           "channel_a_validation": {
               "garbage_value_rejected": rejects(
                   brt, {"output_config": {"effort": GARBAGE_EFFORT}}),
               "unknown_output_config_key_rejected": rejects(
                   brt, {"output_config": {"effort": "low", "not_a_real_key": 1}}),
               "known_good_value_accepted": not rejects(
                   brt, {"output_config": {"effort": "low"}}),
               "garbage_value_sent": GARBAGE_EFFORT,
           },
           "channel_b_output_tokens": {}}
    samples: dict[str, list[int]] = {}
    for label, effort in (("effort_low", "low"), ("effort_high", "high"),
                          ("effort_unset", None)):
        tokens, stops = [], []
        for _ in range(n):
            count, stop = output_length(brt, effort)
            tokens.append(count)
            stops.append(stop)
        samples[label] = tokens
        capped = sum(1 for s in stops if s == "max_tokens")
        out["channel_b_output_tokens"][label] = {
            "n": len(tokens),
            "median_output_tokens": median(tokens),
            "min_output_tokens": min(tokens) if tokens else None,
            "max_output_tokens": max(tokens) if tokens else None,
            # kept, not just summarised: a summary whose raw calls are gone cannot be
            # re-derived, and every derived flag below is checkable from these
            "output_tokens_samples": tokens,
            "stopped_at_max_tokens": capped,
            "length_is_a_floor_not_a_length": capped > 0,
            "stop_reasons_seen": sorted({s for s in stops if s is not None}),
        }
    out.update(_effect_derivation(out["channel_a_validation"],
                                 out["channel_b_output_tokens"], samples))
    return out


def _effect_derivation(validation: dict, conditions: dict,
                       samples: dict[str, list[int]]) -> dict:
    """The conclusions, derived from the two channels' raw numbers.

    Three decisions worth stating, because each replaced something weaker:

    1. The behaviour test is a bootstrap CI on the median difference, the same yardstick
       the latency and guardrail harnesses use. The first version demanded DISJOINT
       ranges (`max(low) < min(high)`) to avoid statistics — but 100% separation is a
       threshold with no interpretation, it fails on one overlapping sample, and its
       strictness moves with n. An interval says how big the effect might be; a
       separation test only says whether ten samples happened not to touch.
    2. The comparison that decides `honored` is **unset vs low**, not high vs low. That
       is the actual deployment choice — send `effort: low`, or send nothing — and
       comparing two values we would never both ship answers a question nobody asked.
    3. A CI that straddles 0 is recorded as UNRESOLVED, never as "no difference". At
       n=10 `high - low` straddles 0; that is a statement about this n, and calling it
       equality is the same error as calling `[0, 0]` decisive.
    """
    low, high, unset = (samples.get(k) or [] for k in
                        ("effort_low", "effort_high", "effort_unset"))
    comparisons = {}
    for name, (a, b) in (("unset_minus_low", (unset, low)),
                         ("high_minus_low", (high, low)),
                         ("unset_minus_high", (unset, high))):
        if not (a and b):
            continue
        raw = median_difference_bootstrap(a, b)
        comparisons[name] = {
            "point_tokens": round(raw["point"], 1),
            "ci95_tokens": [round(v, 1) for v in raw["ci95"]],
            "ci_excludes_zero": raw["ci_excludes_zero"],
            "resolved_at_this_n": raw["ci_excludes_zero"],
            "seed": raw["seed"], "rounds": raw["rounds"],
        }
    primary = comparisons.get("unset_minus_low", {})
    read = bool(validation.get("garbage_value_rejected")
                and validation.get("known_good_value_accepted"))
    shorter = bool(primary.get("ci_excludes_zero")
                   and (primary.get("point_tokens") or 0) > 0)
    ratio = (round(median(low) / median(unset), 3)
             if low and unset and median(unset) else None)
    capped = {k: bool(v.get("stopped_at_max_tokens")) for k, v in conditions.items()}
    default_vs_high = comparisons.get("unset_minus_high", {})
    return {
        "primary_comparison": "unset_minus_low — the deployment choice is send "
                              "effort:low or send nothing, not low versus high",
        "median_difference_tokens": comparisons,
        "parameter_is_read": read,
        "low_produces_shorter_output": shorter,
        "low_over_unset_median": ratio,
        # bias DIRECTION, from stopReason rather than from "the median equals the cap":
        # a truncated longer condition looks shorter than it is, so the saving is
        # understated, never overstated
        "truncation_understates_the_saving": any(
            capped.get(k) for k in ("effort_unset", "effort_high")),
        "conditions_that_hit_the_cap": sorted(k for k, v in capped.items() if v),
        "default_equals_high": (
            None if not default_vs_high else
            (False if default_vs_high["ci_excludes_zero"] else "unresolved at this n")),
        "default_equals_high_note": "§0-8 states the default is `high`. A CI straddling "
                                    "0 is NOT evidence that two conditions are equal, so "
                                    "this can report `False` or `unresolved`, never True.",
        "honored": read and shorter,
        "honored_note": "two independent channels: the API rejects values it does not "
                        "understand, AND the generated length changes by more than the "
                        "interval. Either alone is weak — validation could be a schema "
                        "with no consumer, and a length gap alone could be sampling "
                        "noise — so `honored` requires both. Level M: it says the field "
                        "does something, never that the something is good.",
    }


def rederive(evidence: dict) -> dict:
    """Recompute every DERIVED field from the raw calls the file already kept.

    This is the point of §6.0 rule 6 made executable: a summary that cannot be rebuilt
    from its own raw samples is a number nobody can check. When the arithmetic changes —
    as it did when the behaviour test moved from disjoint ranges to a bootstrap interval
    — the honest action is to re-derive, not to re-bill Bedrock for calls that would
    return different samples and make the two versions incomparable.

    It is deliberately NOT a repair tool that runs on its own. `tests/` asserts that the
    shipped file already equals its own re-derivation, so a stale derived block is a red
    test rather than something this function quietly papers over.
    """
    honored = evidence.get("parameter_honored")
    if honored:
        samples = {k: v["output_tokens_samples"]
                   for k, v in honored["channel_b_output_tokens"].items()}
        honored.update(_effect_derivation(honored["channel_a_validation"],
                                          honored["channel_b_output_tokens"], samples))
    evidence["verdict"] = effort_verdict(evidence)
    return evidence


def effort_verdict(evidence: dict) -> str:
    """The one sentence in the file, derived from the numbers beside it (§6.0 rule 6).

    Three outcomes, not two. Before M3 existed this could only say "unsupported",
    which conflated "effort does not move first-token latency" with "effort is a
    no-op" — and those call for opposite decisions: the first means aim the parameter
    at a different metric, the second means stop sending it.
    """
    diff = evidence.get("high_minus_low_median") or {}
    honored = evidence.get("parameter_honored") or {}
    if diff.get("ci_excludes_zero") and (diff.get("point_ms") or 0) > 0:
        return ("effort:low measurably reduces first-token latency — §0-8's latency "
                "claim holds for this workload")
    if honored.get("honored"):
        return ("effort IS honoured, but §0-8 pointed it at the wrong metric: "
                "first-token latency is unchanged (that CI straddles 0) while output "
                f"length falls to {honored.get('low_over_unset_median')}x of the default "
                "— and that saving is understated where the longer condition hit the "
                "cap. With sentence-level streaming TTS the win is turn length and "
                "output cost, not time-to-speech")
    if honored:
        return ("no first-token difference AND no evidence the field is read — this "
                "measurement cannot distinguish a null effect from a silently ignored "
                "parameter, so it supports no conclusion in either direction")
    return ("no measurable first-token latency difference at this n — the CI straddles "
            "0. Whether that is a null effect or a field Bedrock ignores is UNTESTED "
            "here; run measure_effort_effect before drawing either conclusion")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/track-c")
    ap.add_argument("--n", type=int, default=30, help="calls per effort condition")
    ap.add_argument("--effort-n", type=int, default=10,
                    help="calls per condition for M3; these run to maxTokens "
                         f"{EFFORT_LENGTH_MAX_TOKENS}, so they cost more than M2's")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-effort-check", action="store_true",
                    help="leaves the verdict unable to tell a null effect from a no-op")
    ap.add_argument("--rederive", action="store_true",
                    help="recompute the derived fields from the raw calls already in "
                         "effort-latency.json; makes no Bedrock calls")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rederive:
        path = out_dir / "effort-latency.json"
        result = rederive(json.loads(path.read_text()))
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0

    brt = _client()

    ratio = measure_token_ratio(brt)
    (out_dir / "constitution-tokens.json").write_text(
        json.dumps(ratio, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(ratio, ensure_ascii=False))

    if not args.skip_latency:
        latency = measure_effort_latency(brt, args.n)
        if not args.skip_effort_check:
            latency["parameter_honored"] = measure_effort_effect(brt, args.effort_n)
            # recomputed, because the verdict now has a third outcome available to it
            latency["verdict"] = effort_verdict(latency)
        (out_dir / "effort-latency.json").write_text(
            json.dumps(latency, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(latency, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
