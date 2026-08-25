"""Statistics shared by the on-machine measurement harnesses (§6.0 rules 4 and 6).

One home, because two copies of a statistic is the WINS/BEATS defect: nothing forces
them to agree and the one that drifts is the one nobody reads. Imported by
`safety/token_cost_harness.py` (T-K5/T-F5) and `safety/guardrail_latency_harness.py`
(T-K3). No `__main__`, no network, no Bedrock — pure functions over lists of seconds.

Two rules from §6.0 are encoded here rather than remembered:

  rule 4  a p95 below n=20 is the max wearing a percentile's name, so the summary
          says which of the two it is instead of leaving the reader to check n.
  rule 6  the raw calls are kept alongside the summary, because a difference of
          medians cannot be given a confidence interval from p50/p95 alone.
"""
from __future__ import annotations

import random
import statistics

BOOTSTRAP_SEED = 20260823
BOOTSTRAP_ROUNDS = 10000
SMALL_SAMPLE = 20


def median(values: list[float]) -> float | None:
    """The median, or None for an empty sample.

    Here rather than `statistics.median` at each call site for the reason this whole
    module exists: `tests/test_measure_stats.py` forbids the harnesses from importing
    `statistics` at all, so that "compute it yourself, just this once, it's only a
    median" has nowhere to start. None rather than 0.0 on empty — a zero median reads
    as a measurement that came back fast.
    """
    return statistics.median(values) if values else None


def percentiles(values: list[float]) -> dict:
    """Summarise seconds as milliseconds, keeping the samples it was derived from."""
    ordered = sorted(values)
    n = len(ordered)
    p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))] if n else None
    return {
        "n": n,
        "p50_ms": round(statistics.median(ordered) * 1000, 1) if n else None,
        "p95_ms": round(p95 * 1000, 1) if n else None,
        "p95_is_really_max": n < SMALL_SAMPLE,
        "max_ms": round(ordered[-1] * 1000, 1) if n else None,
        "samples_ms": [round(v * 1000, 1) for v in values],
    }


def median_difference_bootstrap(a: list[float], b: list[float]) -> dict:
    """Percentile-bootstrap 95% CI for median(a) - median(b), in whatever unit came in.

    Seeded so the number is reproducible (§6.1: an unseeded random source produces an
    unreproducible red). A CI straddling 0 is the answer, not a failure to find one.

    Unit-free on purpose: latency comes in as seconds and wants milliseconds out, while
    output-token counts are already the unit they are reported in. One bootstrap, two
    presentations — a second copy of this loop is the WINS/BEATS defect.

    `ci_excludes_zero` is `lo > 0 or hi < 0`, NOT `(lo > 0) == (hi > 0)`. The latter is
    what this function used to say and it is wrong on the boundary: for two identical
    samples the CI is [0, 0] and `(False) == (False)` is True, so a difference of exactly
    nothing was reported as significantly non-zero. Same bug for any CI whose upper bound
    lands on 0. A test pins both boundaries.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = []
    for _ in range(BOOTSTRAP_ROUNDS):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    lo = diffs[int(0.025 * (BOOTSTRAP_ROUNDS - 1))]
    hi = diffs[int(0.975 * (BOOTSTRAP_ROUNDS - 1))]
    return {
        "point": statistics.median(a) - statistics.median(b),
        "ci95": [lo, hi],
        "ci_excludes_zero": lo > 0 or hi < 0,
        "seed": BOOTSTRAP_SEED, "rounds": BOOTSTRAP_ROUNDS,
    }


def median_difference_ci(a: list[float], b: list[float]) -> dict:
    """`median_difference_bootstrap` for callers whose samples are seconds.

    Same numbers, scaled and rounded for reporting; the units are in the key names so a
    millisecond figure cannot be read as a second one.
    """
    raw = median_difference_bootstrap(a, b)
    return {
        "point_ms": round(raw["point"] * 1000, 1),
        "ci95_ms": [round(v * 1000, 1) for v in raw["ci95"]],
        "ci_excludes_zero": raw["ci_excludes_zero"],
        "seed": raw["seed"], "rounds": raw["rounds"],
    }


def budget_verdict(summary: dict, budget_ms: float, what: str) -> dict:
    """Compare a summary against a budget — the only claim a level-M number supports.

    Deliberately not a pass/fail gate: `docs/test-design.md` §6.0 rule 4 says a
    measurement is compared to a budget, and `over_budget` is a statement about this
    run on this day, not a property of the system.
    """
    p95 = summary["p95_ms"]
    return {
        "what": what, "budget_ms": budget_ms, "p95_ms": p95,
        "over_budget": p95 is not None and p95 > budget_ms,
        "n": summary["n"],
        "caveat": ("p95 is the max at this n — the tail is unmeasured, not small"
                   if summary["p95_is_really_max"] else None),
    }
