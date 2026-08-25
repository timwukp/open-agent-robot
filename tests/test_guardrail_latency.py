"""T-K3: the guardrail latency harness, and the evidence it left behind.

No network. What is checked is everything that could make the measurement lie: a
warm-up call folded into the samples (the first `apply_guardrail` of a session ran
671 ms against a ~450 ms steady state), a timeout scored as a fast call, a cost
figure inferred from latency instead of read from the response, an account
identifier written into a file that gets pushed, and a Cantonese latency number
presented without the caveat that Guardrails does not cover Cantonese.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
HARNESS_PY = REPO_ROOT / "safety" / "guardrail_latency_harness.py"
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "track-c" / "guardrail-latency.json"

sys.path.insert(0, str(REPO_ROOT / "safety"))
import guardrail_latency_harness as harness  # noqa: E402

from doubles import FakeClock  # noqa: E402


def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of safety/guardrail_latency_harness.py.

    Zero replacements is the no-mutant control required by §6.0 rule 2. `measure_stats`
    resolves to the repository's real module (already imported), so a mutant of this
    file is measured against the one shared statistics implementation — which is the
    point of the shared module.
    """
    target = tmp_path / "guardrail_latency_harness.py"
    src = HARNESS_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src, encoding="utf-8")
    name = f"gr_mutant_{tmp_path.name}_{len(replacements)}"
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


WARMUP = ("        for _ in range(warmup):\n"
          "            seconds, _action, _unit = apply_seconds(rt, guardrail_id, version, text,\n"
          "                                                    now=now)\n"
          "            warmups.append(None if seconds is None else round(seconds * 1000, 1))")
RECORD = "            warmups.append(None if seconds is None else round(seconds * 1000, 1))"
UNITS = '    units = sum(v for k, v in resp.get("usage", {}).items() if k.endswith("Units"))'
POOLED = '        pooled_seconds = [s for b in blocks for s in _seconds(b["groups"][group])]'
TIER = '    tier = described.get("contentPolicy", {}).get("tier", {}).get("tierName")'
CEILING = ('    bounds = [abs(b) for e in out["run_to_run"].values()\n'
           '              for d in e["later_minus_first_median"] for b in d["ci95_ms"]]')


class FakeReadTimeoutError(Exception):
    """Named so the harness's timeout branch recognises it the way botocore's is."""


class SlowFirstCall:
    """A guardrail whose first call is slow, like the real one. Every later call is
    fast. A harness that keeps the warm-up reports a cold connection as the cost."""

    def __init__(self, slow_calls: int = 2, units: int = 1) -> None:
        self.calls = 0
        self.slow_calls = slow_calls
        self.units = units
        self.seen: list[dict] = []

    def apply_guardrail(self, **kw):
        self.seen.append(kw)
        self.calls += 1
        return {"action": "NONE",
                "usage": {"contentPolicyUnits": self.units, "topicPolicyUnits": 0}}


def test_the_call_reports_seconds_action_and_units_from_the_response() -> None:
    rt = SlowFirstCall(units=3)
    seconds, action, units = harness.apply_seconds(rt, "gid", "DRAFT", "hello")
    assert seconds is not None and seconds >= 0.0
    assert action == "NONE"
    assert units == 3, "the billed units must come from the response, not an estimate"


def test_the_output_side_is_what_gets_checked() -> None:
    """S7 guards what the robot is about to say; an INPUT check measures the wrong
    thing and would still produce a plausible latency number."""
    rt = SlowFirstCall()
    harness.apply_seconds(rt, "gid", "DRAFT", "hello")
    assert rt.seen[0]["source"] == "OUTPUT"


def test_a_read_timeout_is_a_timeout_not_a_fast_call() -> None:
    class Timing:
        def apply_guardrail(self, **kw):
            raise FakeReadTimeoutError("read timeout")
    seconds, action, units = harness.apply_seconds(Timing(), "gid", "DRAFT", "x")
    assert seconds is None and action == "TIMEOUT" and units == 0


def test_a_real_failure_is_not_swallowed_as_a_timeout() -> None:
    class Broken:
        def apply_guardrail(self, **kw):
            raise ValueError("AccessDeniedException")
    with pytest.raises(ValueError):
        harness.apply_seconds(Broken(), "gid", "DRAFT", "x")


def test_the_warmup_calls_are_made_and_then_discarded() -> None:
    rt = SlowFirstCall()
    result = harness.measure(rt, "gid", "DRAFT", n=5, warmup=2)
    groups = len(harness.UTTERANCES)
    assert rt.calls == groups * (5 + 2), "warm-up calls must actually be made"
    for group in result["groups"].values():
        assert group["n"] == 5, "and must not appear in the samples"
        assert len(group["samples_ms"]) == 5


class Scheduled(SlowFirstCall):
    """A guardrail on a fixed timetable: three groups whose medians sit 15 ms apart,
    and a within-group spread wide enough that re-measuring the SAME group can move its
    median much further than the gap between groups. That is the case where comparing a
    gap against a point estimate and against an interval give opposite answers.

    It advances an injected clock instead of sleeping, and that is a correction rather
    than a refinement. The sleeping version passed on its own and read **46.0 ms** for a
    30 ms nominal gap inside a full-suite run, because a median that straddles two sleep
    levels picks up the scheduler's overshoot unevenly. A measurement harness's own test
    must not measure the machine it runs on; `docs/test-design.md` §1 already required an
    injectable clock and this file was the one place still without one.
    """

    BASE_MS = {"mandarin": 10, "english": 25, "cantonese": 40}
    # Same multiset in both runs, so the point shift is small, while the median of any
    # resample is either the low or the high half — which is what widens the interval.
    OFFSETS_MS = [0, 0, 0, 120, 120, 120]

    def __init__(self) -> None:
        super().__init__()
        self.taken: dict[str, int] = {}
        self.clock = FakeClock()

    def apply_guardrail(self, **kw):
        text = kw["content"][0]["text"]["text"]
        group = next(g for g, t in harness.UTTERANCES.items() if t == text)
        i = self.taken.get(group, 0)
        self.taken[group] = i + 1
        self.clock.advance(
            (self.BASE_MS[group] + self.OFFSETS_MS[i % len(self.OFFSETS_MS)]) / 1000)
        return super().apply_guardrail(**kw)


# The real `guardrailProfileArn` carries the account id, so the double cannot contain a
# realistic one — infra/redact_scan.py rejects any ARN in a pushed file, and evading it
# in test code would teach the pattern it exists to stop. A sentinel tests the property
# that actually matters (that this FIELD is never copied) and tests it more strongly:
# it holds for any ARN format, not just the one string a grep would have looked for.
MUST_NOT_BE_COPIED = "guardrail-profile-arn-sentinel-account-bearing-field"


class LyingControlPlane:
    """Reports a tier the harness did not ask for, and hands back the account-bearing
    field it must not keep. A double that echoed "STANDARD" would let a hard-coded tier
    pass, which is exactly how the previous run of this harness went unnoticed."""

    def __init__(self, tier: str = "CLASSIC") -> None:
        self.tier = tier

    def get_guardrail(self, **kw):
        return {"contentPolicy": {"tier": {"tierName": self.tier}},
                "crossRegionDetails": {"guardrailProfileId": "apac.guardrail.v1:0",
                                       "guardrailProfileArn": MUST_NOT_BE_COPIED}}


def test_the_tier_is_read_from_the_api_not_declared_by_the_harness() -> None:
    """Classic tier covers en/fr/es only. A latency number labelled Standard while the
    guardrail was Classic is a number for a tier this system cannot use — and Standard
    needs cross-Region inference, so Classic is what you get by default."""
    info = harness.read_tier(LyingControlPlane("CLASSIC"), "gid", "DRAFT")
    assert info["tier"] == "CLASSIC"
    assert info["cross_region_profile"] == "apac.guardrail.v1:0"
    assert "GetGuardrail" in info["tier_read_from"]


def test_the_tier_reaches_the_output_verbatim() -> None:
    result = harness.measure_runs(
        SlowFirstCall(), "gid", "DRAFT", n=2, runs=2, warmup=0,
        tier_info=harness.read_tier(LyingControlPlane("CLASSIC"), "gid", "DRAFT"))
    assert result["tier"] == "CLASSIC", "the harness overrode what the API said"


def test_the_profile_arn_field_is_not_carried_into_the_output() -> None:
    """`crossRegionDetails` has two fields and only one of them is safe: the profile id
    is a public constant, its ARN carries the account id."""
    info = harness.read_tier(LyingControlPlane(), "gid", "DRAFT")
    assert MUST_NOT_BE_COPIED not in json.dumps(info)
    result = harness.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=1, runs=1,
                                  warmup=0, tier_info=info)
    assert MUST_NOT_BE_COPIED not in json.dumps(result)
    assert result["cross_region_profile"] == "apac.guardrail.v1:0", (
        "the safe half must survive — dropping both would pass a bare absence check")


def test_an_unread_tier_is_null_rather_than_assumed() -> None:
    """No control-plane client means the tier is unknown, and unknown must not render
    as the tier we hoped for."""
    result = harness.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=1, runs=1, warmup=0)
    assert result["tier"] is None
    assert "not read" in result["tier_read_from"]


def test_the_discarded_warmups_are_still_written_down() -> None:
    """The harness discards warm-ups because the first call of a session is slow. That
    justification is a claim; recording the discarded samples is what makes it one a
    reader can check instead of one they have to believe (§6.0 rule 6)."""
    result = harness.measure(SlowFirstCall(), "gid", "DRAFT", n=4, warmup=3)
    for group in result["groups"].values():
        assert len(group["warmup_samples_ms"]) == 3
        assert group["n"] == 4, "and they are still out of the statistics"


def test_two_runs_are_kept_separately_as_well_as_pooled() -> None:
    rt = SlowFirstCall()
    result = harness.measure_runs(rt, "gid", "DRAFT", n=5, runs=2, warmup=1)
    assert [b["run"] for b in result["runs"]] == [1, 2]
    groups = len(harness.UTTERANCES)
    assert rt.calls == 2 * groups * (5 + 1)
    for group in harness.UTTERANCES:
        assert all(b["groups"][group]["n"] == 5 for b in result["runs"])
        pooled = result["pooled"]["groups"][group]
        assert pooled["n"] == 10, "the pooled summary must span both runs"
        assert len(pooled["samples_ms"]) == 10
        assert pooled["text_units_billed"] == sum(
            b["groups"][group]["text_units_billed"] for b in result["runs"])


def test_the_run_to_run_difference_is_computed_because_it_is_the_yardstick() -> None:
    """A ~30 ms gap between two language groups means nothing until you know what the
    same group does when simply measured again. One run cannot supply that."""
    two = harness.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=5, runs=2, warmup=0)
    for group in harness.UTTERANCES:
        entry = two["run_to_run"][group]
        assert len(entry["later_minus_first_median"]) == 1
        assert "ci95_ms" in entry["later_minus_first_median"][0]
    one = harness.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=5, runs=1, warmup=0)
    assert one["run_to_run"] == {}, (
        "a single run must not present a replication it does not have")


def test_the_caveat_survives_pooling() -> None:
    result = harness.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=2, runs=2, warmup=0)
    assert "coverage_caveat" in result["pooled"]["groups"]["cantonese"]
    assert "coverage_caveat" not in result["pooled"]["groups"]["mandarin"]


def test_every_language_group_is_measured_and_none_is_silently_dropped() -> None:
    result = harness.measure(SlowFirstCall(), "gid", "DRAFT", n=3, warmup=0)
    assert set(result["groups"]) == set(harness.UTTERANCES)
    assert set(result["groups"]) >= {"mandarin", "english", "cantonese"}


def test_the_cantonese_group_carries_the_coverage_caveat() -> None:
    """T-G4: no report may imply Cantonese is protected. A fast latency number on an
    unsupported language is the easiest way to imply it by accident."""
    result = harness.measure(SlowFirstCall(), "gid", "DRAFT", n=2, warmup=0)
    caveat = result["groups"]["cantonese"]["coverage_caveat"]
    assert "no claim" in caveat and "Cantonese" in caveat
    assert "coverage_caveat" not in result["groups"]["mandarin"]


def test_the_guardrail_identifier_never_reaches_the_output() -> None:
    """It carries the account id, and infra/redact_scan.py would reject the push."""
    result = harness.measure(SlowFirstCall(), "secret-guardrail-id", "DRAFT", n=2, warmup=0)
    assert "secret-guardrail-id" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.mutation
def test_mutant_keeping_the_warmup_folds_a_cold_connection_into_the_median(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (WARMUP, "        pass"))
    rt = SlowFirstCall()
    result = mutant.measure(rt, "gid", "DRAFT", n=5, warmup=2)
    assert rt.calls == len(harness.UTTERANCES) * 5, "no warm-up calls were made"
    assert all(g["n"] == 5 for g in result["groups"].values())


@pytest.mark.mutation
def test_mutant_judging_the_language_gap_against_a_point_instead_of_an_interval(
        tmp_path: Path) -> None:
    """The defect this block was written with and then fixed. The point estimate of a
    pure-noise repeat sits near zero, so comparing a gap against it declares almost any
    gap real — here a 30 ms language gap clears a 5 ms point shift while the interval
    that produced that point reaches 60 ms."""
    mutant = load_mutant(tmp_path, (CEILING,
                                    '    bounds = [abs(d["point_ms"]) for e in out["run_to_run"].values()\n'
                                    '              for d in e["later_minus_first_median"]]'))

    rt = Scheduled()
    real = harness.measure_runs(rt, "gid", "DRAFT", n=6, runs=2, warmup=0,
                                now=rt.clock.monotonic)["interpretation"]
    assert real["between_group_p50_spread_ms"] == 30.0, (
        "exact, not approximate: with the clock injected the schedule is the measurement")
    assert real["largest_run_to_run_p50_point_shift_ms"] < 60
    assert real["run_to_run_noise_ceiling_ms"] > 90, "the interval is the wide one"
    assert real["language_gap_exceeds_replication_noise"] is False

    bad_rt = Scheduled()
    bad = mutant.measure_runs(bad_rt, "gid", "DRAFT", n=6, runs=2, warmup=0,
                              now=bad_rt.clock.monotonic)["interpretation"]
    assert bad["run_to_run_noise_ceiling_ms"] == \
        bad["largest_run_to_run_p50_point_shift_ms"], "the mutant compares point to point"
    assert bad["language_gap_exceeds_replication_noise"] is True, (
        f"a {bad['between_group_p50_spread_ms']} ms gap cleared a "
        f"{bad['run_to_run_noise_ceiling_ms']} ms bar that the real interval puts at "
        f"{real['run_to_run_noise_ceiling_ms']} ms")


@pytest.mark.mutation
def test_mutant_declaring_the_tier_instead_of_reading_it(tmp_path: Path) -> None:
    """The defect this whole seam exists for: the previous run of this harness wrote
    `"tier": "STANDARD"` as a literal, so the test asserting Standard tier passed
    without anything having checked the guardrail."""
    mutant = load_mutant(tmp_path, (TIER, '    tier = "STANDARD"'))
    info = mutant.read_tier(LyingControlPlane("CLASSIC"), "gid", "DRAFT")
    assert info["tier"] == "STANDARD", "the mutant reports the tier it wants"


@pytest.mark.mutation
def test_mutant_discarding_the_warmups_without_recording_them(tmp_path: Path) -> None:
    """The subtler of the two: the calls are still made and still excluded, so every
    latency number is right — but the reason for excluding them is back to being an
    unverifiable sentence in a docstring."""
    mutant = load_mutant(tmp_path, (RECORD, "            pass"))
    result = mutant.measure(SlowFirstCall(), "gid", "DRAFT", n=4, warmup=3)
    assert result["groups"]["mandarin"]["warmup_samples_ms"] == []
    assert result["groups"]["mandarin"]["n"] == 4, "the samples themselves are intact"


@pytest.mark.mutation
def test_mutant_pooling_only_the_first_run_hides_the_second(tmp_path: Path) -> None:
    """A pooled summary that silently spans one run reports half the tail while the
    file still says two runs were requested."""
    mutant = load_mutant(tmp_path, (POOLED,
                                    '        pooled_seconds = _seconds(blocks[0]["groups"][group])'))
    result = mutant.measure_runs(SlowFirstCall(), "gid", "DRAFT", n=5, runs=2, warmup=0)
    assert result["pooled"]["groups"]["mandarin"]["n"] == 5
    assert result["pooled"]["groups"]["mandarin"]["pooled_over_runs"] == 2, (
        "which is exactly why n is asserted against runs * requested_n_per_group")


@pytest.mark.mutation
def test_mutant_estimating_the_cost_instead_of_reading_it(tmp_path: Path) -> None:
    """T-F4: each metric from its own source. A unit count derived from anything but
    the response's `usage` is a cost figure that cannot be reconciled with a bill."""
    mutant = load_mutant(tmp_path, (UNITS, "    units = 1"))
    _, _, units = mutant.apply_seconds(SlowFirstCall(units=7), "gid", "DRAFT", "x")
    assert units == 1


@pytest.mark.mutation
def test_the_no_mutant_control_measures_the_same_groups(tmp_path: Path) -> None:
    control = load_mutant(tmp_path)
    result = control.measure(SlowFirstCall(), "gid", "DRAFT", n=3, warmup=1)
    assert set(result["groups"]) == set(harness.UTTERANCES)
    assert all(g["n"] == 3 for g in result["groups"].values())


# ---------------------------------------------- the evidence, and what it decides
def test_the_evidence_records_our_region_and_the_tier_s7_specifies() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["region"] == harness.REGION
    assert evidence["tier"] == "STANDARD", (
        "Classic tier covers en/fr/es only — a latency number measured on Classic "
        "would be a number for a tier this system cannot use")
    assert "GetGuardrail" in evidence["tier_read_from"], (
        "the tier above has to have been read from the guardrail, not typed here")
    assert evidence["cross_region_profile"], (
        "Standard tier cannot be enabled without a cross-Region guardrail profile, so "
        "a Standard tier with no profile recorded did not come from a real guardrail")
    assert evidence["source"] == "OUTPUT"


def test_the_evidence_carries_no_account_identifier() -> None:
    text = EVIDENCE.read_text()
    assert "arn:aws" not in text
    assert not any(part.isdigit() and len(part) == 12
                   for part in text.replace(":", " ").replace('"', " ").split())


def test_every_group_reached_the_planned_sample_size_in_every_run() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    runs = evidence["runs"]
    assert len(runs) == evidence["runs_requested"] >= 2, (
        "a run that did not happen cannot be the yardstick for the other one")
    for block in runs:
        for group, data in block["groups"].items():
            assert data["n"] >= 30, f"run {block['run']} {group} ran {data['n']} calls"
            assert data["n"] == len(data["samples_ms"]), f"{group} lost its raw calls"
    for group, data in evidence["pooled"]["groups"].items():
        assert data["n"] == evidence["runs_requested"] * evidence["requested_n_per_group"]
        assert data["p95_is_really_max"] is False


def test_the_discarded_warmups_are_in_the_evidence_and_justify_discarding_them() -> None:
    """If a re-measurement ever shows the warm-up calls were NOT slower, the harness's
    reason for throwing them away is wrong and the docstring has to change — which is
    the point of keeping them in the file rather than in a comment."""
    evidence = json.loads(EVIDENCE.read_text())
    slowest_warmup = -1.0
    for block in evidence["runs"]:
        for group, data in block["groups"].items():
            assert data["warmup_samples_ms"], f"run {block['run']} {group} recorded none"
            assert len(data["warmup_samples_ms"]) == evidence["warmup_calls_discarded"]
            timed = [v for v in data["warmup_samples_ms"] if v is not None]
            assert timed, f"run {block['run']} {group}: every warm-up timed out"
            slowest_warmup = max([slowest_warmup] + timed)
    worst_p50 = max(d["p50_ms"] for d in evidence["pooled"]["groups"].values())
    assert slowest_warmup > worst_p50, (
        f"slowest warm-up {slowest_warmup} ms is not above the worst pooled median "
        f"{worst_p50} ms — discarding warm-ups is no longer justified by the data")


def test_the_between_group_gap_is_reported_against_the_run_to_run_gap() -> None:
    """§6.0 rule 6 applied to a comparison rather than a number: the file must contain
    the yardstick, not just the thing being measured. This asserts the yardstick is
    there and is a real interval; whether the language gap clears it is reported, not
    gated, because n=30 per run cannot settle a ~30 ms difference either way."""
    evidence = json.loads(EVIDENCE.read_text())
    assert set(evidence["run_to_run"]) == set(evidence["pooled"]["groups"]), (
        "a group without a re-measurement has no scale to read its gap against")
    medians = [d["p50_ms"] for d in evidence["pooled"]["groups"].values()]
    between = round(max(medians) - min(medians), 1)
    widest, ceiling = 0.0, 0.0
    for group, entry in evidence["run_to_run"].items():
        assert entry["later_minus_first_median"], f"{group} has no replication"
        for diff in entry["later_minus_first_median"]:
            lo, hi = diff["ci95_ms"]
            assert lo <= diff["point_ms"] <= hi, f"{group}: point outside its own CI"
            widest = max(widest, abs(diff["point_ms"]))
            ceiling = max(ceiling, abs(lo), abs(hi))
    interp = evidence["interpretation"]
    assert interp["between_group_p50_spread_ms"] == between
    assert interp["largest_run_to_run_p50_point_shift_ms"] == round(widest, 1)
    assert interp["run_to_run_noise_ceiling_ms"] == round(ceiling, 1)
    assert interp["run_to_run_noise_ceiling_ms"] >= \
        interp["largest_run_to_run_p50_point_shift_ms"], (
        "the noise ceiling is a CI bound; it cannot be below the point it brackets")
    assert interp["language_gap_exceeds_replication_noise"] == (between > ceiling), (
        "the file's own interpretation does not follow from the numbers beside it")


def test_the_measurement_supports_s7s_asynchronous_design() -> None:
    """The load-bearing conclusion. S7 chose an asynchronous guard plus kill-voice
    over a serial pre-check because the guard is slow relative to the 400 ms
    verdict-to-silence budget. The plan's evidence for that was somebody else's
    n=15. This asserts OUR numbers still say it — and it is written to go red if
    they ever stop saying it, because then the design should be revisited rather
    than the number quietly updated."""
    evidence = json.loads(EVIDENCE.read_text())
    for group, data in evidence["pooled"]["groups"].items():
        assert data["budget"]["over_budget"] is True, (
            f"{group}: apply_guardrail p95 {data['p95_ms']} ms now fits inside the "
            f"{data['budget']['budget_ms']} ms budget — a serial guard may have "
            f"become viable, which is a design decision, not a test update")


def test_the_cost_per_call_is_recorded_in_billable_units() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    for group, data in evidence["pooled"]["groups"].items():
        assert data["text_units_billed"] >= data["n"], (
            f"{group}: {data['text_units_billed']} units for {data['n']} calls — "
            "fewer units than calls means the usage field was not read")


def test_the_harness_does_not_run_on_import() -> None:
    src = HARNESS_PY.read_text()
    assert 'if __name__ == "__main__":' in src
    assert "\nmain()" not in src
