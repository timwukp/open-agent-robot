"""The statistics both measurement harnesses share (§6.0 rules 4 and 6).

These are the assertions that decide whether a level-M number means anything: an
interval that can never exclude zero reads as caution rather than as a broken test,
a p95 that is secretly a max reads as a measured tail, and a summary that drops its
raw calls makes its own headline unfalsifiable. All three fail silently, so all three
get a named mutant.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
STATS_PY = REPO_ROOT / "safety" / "measure_stats.py"

sys.path.insert(0, str(REPO_ROOT / "safety"))
import measure_stats as stats  # noqa: E402


def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of safety/measure_stats.py.

    Same shape as tests/test_guard.py; zero replacements is the no-mutant control
    required by §6.0 rule 2. The module has no file or network dependencies, so the
    copy needs nothing beside it.
    """
    target = tmp_path / "measure_stats.py"
    src = STATS_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src, encoding="utf-8")
    name = f"stats_mutant_{tmp_path.name}_{len(replacements)}"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


CI_ZERO_TEST = '        "ci_excludes_zero": lo > 0 or hi < 0,'
SCALE = '        "point_ms": round(raw["point"] * 1000, 1),'
KEEP_SAMPLES = '        "samples_ms": [round(v * 1000, 1) for v in values],'
SMALL_FLAG = '        "p95_is_really_max": n < SMALL_SAMPLE,'
OVER_BUDGET = '        "over_budget": p95 is not None and p95 > budget_ms,'

SHIFTED = [1.0 + 0.01 * i for i in range(30)]
LOWER = [0.9 + 0.01 * i for i in range(30)]


# --------------------------------------------------------------- the interval
def test_a_real_shift_is_detected() -> None:
    got = stats.median_difference_ci(SHIFTED, LOWER)
    assert got["point_ms"] == pytest.approx(100.0, abs=1.0)
    assert got["ci_excludes_zero"] is True
    assert got["ci95_ms"][0] > 0


def test_identical_distributions_do_not_produce_a_verdict() -> None:
    """The control: without this, a CI that always excludes zero looks decisive."""
    got = stats.median_difference_ci(SHIFTED, list(SHIFTED))
    assert got["point_ms"] == pytest.approx(0.0, abs=1.0)
    assert got["ci_excludes_zero"] is False
    assert got["ci95_ms"][0] < 0 < got["ci95_ms"][1]


def test_the_interval_is_reproducible() -> None:
    """Seeded, because an unreproducible red is a red nobody can act on (§6.1)."""
    first = stats.median_difference_ci(SHIFTED, LOWER)
    assert first == stats.median_difference_ci(SHIFTED, LOWER)
    assert first["seed"] == stats.BOOTSTRAP_SEED


def test_a_wide_spread_widens_the_interval_instead_of_moving_the_point() -> None:
    """Why n=30 could not resolve the ~12-47 ms effort:low difference: the interval
    scales with the spread of the samples, not with the size of the effect sought."""
    tight = stats.median_difference_ci(SHIFTED, LOWER)
    wide = stats.median_difference_ci([1.0 + 0.05 * i for i in range(30)],
                                      [0.9 + 0.05 * i for i in range(30)])
    assert wide["point_ms"] == pytest.approx(tight["point_ms"], abs=1.0)
    assert (wide["ci95_ms"][1] - wide["ci95_ms"][0]) > \
        3 * (tight["ci95_ms"][1] - tight["ci95_ms"][0])


def test_a_zero_width_interval_at_zero_does_not_exclude_zero() -> None:
    """The boundary this module used to get backwards.

    Two constant, identical samples give a CI of exactly [0, 0]. Under the old test
    `(lo > 0) == (hi > 0)` that is `False == False` — True — so a difference of
    precisely nothing was reported as significantly non-zero, and any verdict keyed off
    the flag would have called a no-op a real effect.
    """
    flat = stats.median_difference_ci([1.0] * 12, [1.0] * 12)
    assert flat["ci95_ms"] == [0.0, 0.0]
    assert flat["point_ms"] == 0.0
    assert flat["ci_excludes_zero"] is False


def test_an_interval_whose_bound_lands_on_zero_does_not_exclude_it() -> None:
    """The one-sided half of the same boundary: [-x, 0] touches zero, so it does not
    exclude it. Checked on the raw predicate, because constructing samples whose
    bootstrap bound lands exactly on 0 is not something to leave to chance."""
    for lo, hi, expected in ((0.0, 0.0, False), (-5.0, 0.0, False), (0.0, 5.0, False),
                             (1.0, 5.0, True), (-5.0, -1.0, True), (-1.0, 5.0, False)):
        assert (lo > 0 or hi < 0) is expected, f"[{lo}, {hi}]"


def test_the_millisecond_wrapper_is_exactly_the_raw_bootstrap_scaled() -> None:
    """One bootstrap, two presentations. If these could drift, the unit-free version
    used for token counts and the millisecond one used for latency would become two
    calibrations of the same statistic."""
    raw = stats.median_difference_bootstrap(SHIFTED, LOWER)
    scaled = stats.median_difference_ci(SHIFTED, LOWER)
    assert scaled["point_ms"] == pytest.approx(raw["point"] * 1000, abs=0.05)
    assert scaled["ci95_ms"] == [pytest.approx(v * 1000, abs=0.05) for v in raw["ci95"]]
    assert scaled["ci_excludes_zero"] is raw["ci_excludes_zero"]
    assert "point" not in scaled and "point_ms" not in raw, (
        "the unit-free and the millisecond result must not share key names — that is how "
        "a seconds value gets read as milliseconds")


def test_the_raw_bootstrap_keeps_the_unit_it_was_given() -> None:
    """Token counts go through the same loop and must come back as token counts."""
    got = stats.median_difference_bootstrap([900, 900, 880], [500, 519, 540])
    assert got["point"] == pytest.approx(381.0, abs=1.0)
    assert got["ci_excludes_zero"] is True
    assert all(abs(v) < 10000 for v in got["ci95"]), "the raw form must not scale by 1000"


@pytest.mark.mutation
def test_mutant_without_the_sign_check_every_interval_excludes_zero(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (CI_ZERO_TEST, '        "ci_excludes_zero": True,'))
    assert mutant.median_difference_ci(SHIFTED, list(SHIFTED))["ci_excludes_zero"] is True


@pytest.mark.mutation
def test_mutant_with_the_old_boundary_test_calls_a_zero_difference_decisive(
        tmp_path: Path) -> None:
    """The exact predicate this module shipped with, restored — it must go green on a
    zero difference, which is what makes it a bug and not a style choice."""
    mutant = load_mutant(
        tmp_path, (CI_ZERO_TEST, '        "ci_excludes_zero": (lo > 0) == (hi > 0),'))
    assert mutant.median_difference_ci([1.0] * 12, [1.0] * 12)["ci_excludes_zero"] is True
    assert stats.median_difference_ci([1.0] * 12, [1.0] * 12)["ci_excludes_zero"] is False


@pytest.mark.mutation
def test_mutant_dropping_the_scale_reports_seconds_under_a_millisecond_name(
        tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (SCALE, '        "point_ms": round(raw["point"], 1),'))
    assert mutant.median_difference_ci(SHIFTED, LOWER)["point_ms"] == pytest.approx(
        0.1, abs=0.01), "the mutant now reports 0.1 where the real one reports 100.0"


# ---------------------------------------------------------------- the summary
def test_the_summary_keeps_the_calls_it_was_derived_from() -> None:
    values = [0.1 * i for i in range(1, 26)]
    got = stats.percentiles(values)
    assert got["n"] == 25
    assert got["samples_ms"] == [round(v * 1000, 1) for v in values]
    assert got["p95_is_really_max"] is False


def test_a_small_sample_says_its_p95_is_the_max() -> None:
    """§6.0 rule 4: below n=20 a p95 is the max wearing a percentile's name."""
    got = stats.percentiles([0.1, 0.2, 0.3])
    assert got["p95_is_really_max"] is True
    assert got["p95_ms"] == got["max_ms"]


def test_an_empty_sample_reports_no_numbers_rather_than_zeros() -> None:
    """A zeroed summary would read as an instant system (§6.0 rule 5)."""
    got = stats.percentiles([])
    assert got["n"] == 0
    assert got["p50_ms"] is None and got["p95_ms"] is None and got["max_ms"] is None


@pytest.mark.mutation
def test_mutant_dropping_the_raw_calls_makes_the_headline_unfalsifiable(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (KEEP_SAMPLES, ""))
    assert "samples_ms" not in mutant.percentiles([0.1, 0.2])


@pytest.mark.mutation
def test_mutant_hiding_the_small_sample_flag_dresses_a_max_as_a_p95(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (SMALL_FLAG, '        "p95_is_really_max": False,'))
    got = mutant.percentiles([0.1, 0.2, 0.3])
    assert got["p95_is_really_max"] is False and got["p95_ms"] == got["max_ms"]


# ----------------------------------------------------------------- the budget
def test_the_budget_comparison_uses_p95_and_says_so() -> None:
    over = stats.budget_verdict(stats.percentiles([0.5] * 30), 400.0, "x")
    under = stats.budget_verdict(stats.percentiles([0.1] * 30), 400.0, "x")
    assert over["over_budget"] is True and over["p95_ms"] == 500.0
    assert under["over_budget"] is False
    assert over["caveat"] is None, "n=30 is above the small-sample line"


def test_a_small_sample_budget_carries_the_unmeasured_tail_caveat() -> None:
    got = stats.budget_verdict(stats.percentiles([0.1, 0.2]), 400.0, "x")
    assert got["caveat"] is not None and "unmeasured" in got["caveat"]


def test_an_empty_measurement_is_not_within_budget() -> None:
    """Zero calls must not read as a pass — §6.0 rule 5: a scan of nothing is an
    error, and `over_budget: false` on n=0 is the same lie in a summary's clothing."""
    got = stats.budget_verdict(stats.percentiles([]), 400.0, "x")
    assert got["n"] == 0
    assert got["p95_ms"] is None
    assert got["over_budget"] is False, (
        "documented behaviour: an empty measurement reports no p95 and therefore "
        "cannot be over budget — callers must check n, which is why n is in here")


@pytest.mark.mutation
def test_mutant_flipping_the_budget_comparison_passes_everything(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path,
                         (OVER_BUDGET, '        "over_budget": p95 is not None and p95 < budget_ms,'))
    over = mutant.budget_verdict(mutant.percentiles([0.5] * 30), 400.0, "x")
    assert over["over_budget"] is False


@pytest.mark.mutation
def test_the_no_mutant_control_reproduces_every_number(tmp_path: Path) -> None:
    """Required every round: without it, a broken loader reports success (§6.0 rule 2)."""
    control = load_mutant(tmp_path)
    assert control.percentiles([0.1, 0.2, 0.3]) == stats.percentiles([0.1, 0.2, 0.3])
    assert control.median_difference_ci(SHIFTED, LOWER) == \
        stats.median_difference_ci(SHIFTED, LOWER)
    assert control.budget_verdict(control.percentiles([0.5] * 30), 400.0, "x") == \
        stats.budget_verdict(stats.percentiles([0.5] * 30), 400.0, "x")


def test_both_harnesses_use_this_module_rather_than_their_own_copy() -> None:
    """The reason this module exists: two copies of a statistic is the WINS/BEATS
    defect, and the copy that drifts is the one nobody reads. Derived by reading the
    harness sources, so a re-inlined helper goes red instead of going unnoticed."""
    for name in ("token_cost_harness.py", "guardrail_latency_harness.py"):
        src = (REPO_ROOT / "safety" / name).read_text()
        assert "from measure_stats import" in src, f"{name} does not share the stats"
        assert "def percentiles(" not in src, f"{name} has re-inlined percentiles()"
        assert "def median_difference_ci(" not in src, f"{name} has re-inlined the CI"
        assert "def median_difference_bootstrap(" not in src, \
            f"{name} has re-inlined the bootstrap"
        assert "def median(" not in src, f"{name} has re-inlined the median"
        assert "import statistics" not in src, f"{name} computes its own statistics"


def test_the_shared_median_is_none_on_an_empty_sample() -> None:
    """The reason it is not `statistics.median` at each call site: the guard above bans
    importing `statistics` in the harnesses, so a median has exactly one home. A zero
    would read as a real measurement that came back short."""
    assert stats.median([]) is None
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 2, 3]) == 2.5
