"""T-K5 / T-F5 harness: the statistics and the Bedrock seam, with no network.

The harness itself is a level-M measurement, so nothing here checks a latency
number. What is checked is the part that can be wrong silently: a confidence
interval that never excludes zero (a permanently inconclusive verdict reads like
caution), a token-cost delta that quietly reports the baseline instead of the
sample, and a summary that drops the raw calls it was derived from.

The Bedrock client is a **lying** double, not an echo (§6.1): it returns token
counts unrelated to the text it was handed, and streams whose opening events carry
no text at all, so a code path that ignores the response cannot pass.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
HARNESS_PY = REPO_ROOT / "safety" / "token_cost_harness.py"

sys.path.insert(0, str(REPO_ROOT / "safety"))
import token_cost_harness as harness  # noqa: E402


def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of safety/token_cost_harness.py.

    Same shape as tests/test_guard.py; zero replacements is the no-mutant control
    required by §6.0 rule 2.

    The copy goes into `tmp_path/safety/` and `constitution.md` is copied beside it,
    because the harness resolves `REPO_ROOT` from its own `__file__` — a copy placed
    anywhere else reads the sample texts out of a directory that does not exist.
    """
    dst = tmp_path / "safety"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "safety" / "constitution.md", dst / "constitution.md")
    target = dst / "token_cost_harness.py"
    src = HARNESS_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src, encoding="utf-8")
    name = f"harness_mutant_{tmp_path.name}_{len(replacements)}"
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


DELTA = "        tokens = call(text) - baseline"
FIRST_TEXT = '            if delta.get("text"):'
INTERVAL = ('    shorter = bool(primary.get("ci_excludes_zero")\n'
            '                   and (primary.get("point_tokens") or 0) > 0)')
PRIMARY = '    primary = comparisons.get("unset_minus_low", {})'
READ = ('    read = bool(validation.get("garbage_value_rejected")\n'
        '                and validation.get("known_good_value_accepted"))')
HONORED_BRANCH = '    if honored.get("honored"):'
UNRESOLVED = ('            (False if default_vs_high["ci_excludes_zero"] '
              'else "unresolved at this n")),')


# ------------------------------------------------------- the token-cost delta
class LyingBedrock:
    """Returns token counts unrelated to the text it was given, so a harness that
    reports the baseline, or the raw call, instead of the delta cannot pass."""

    def __init__(self, per_call: list[int]) -> None:
        self.per_call = list(per_call)
        self.systems: list[str | None] = []

    def converse(self, **kw):
        self.systems.append(kw["system"][0]["text"] if "system" in kw else None)
        return {"usage": {"inputTokens": self.per_call.pop(0), "outputTokens": 1,
                          "totalTokens": 2, "cacheReadInputTokens": 0}}


def test_the_token_cost_is_the_delta_not_the_raw_call() -> None:
    brt = LyingBedrock([7, 1007, 507])            # baseline, then two samples
    got = harness.measure_token_ratio(brt)
    assert got["baseline_input_tokens"] == 7
    assert [s["tokens"] for s in got["samples"]] == [1000, 500]
    assert brt.systems[0] is None, "the baseline must be measured with no system block"
    assert all(s is not None for s in brt.systems[1:])


def test_every_sample_reports_its_own_ratio() -> None:
    brt = LyingBedrock([0, 100, 200])
    got = harness.measure_token_ratio(brt)
    for sample, real in zip(got["samples"], harness._samples()):
        assert sample["name"] == real[0]
        assert sample["chars"] == len(real[1])
        assert sample["tokens_per_char"] == pytest.approx(
            sample["tokens"] / sample["chars"], abs=1e-4)


@pytest.mark.mutation
def test_mutant_forgetting_the_baseline_inflates_every_cost(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (DELTA, "        tokens = call(text)"))
    brt = LyingBedrock([7, 1007, 507])
    got = mutant.measure_token_ratio(brt)
    assert [s["tokens"] for s in got["samples"]] == [1007, 507]


@pytest.mark.mutation
def test_the_no_mutant_control_measures_the_same_numbers(tmp_path: Path) -> None:
    """Required every round: without it, a broken loader reports success (§6.0 rule 2)."""
    control = load_mutant(tmp_path)
    brt = LyingBedrock([7, 1007, 507])
    assert [s["tokens"] for s in control.measure_token_ratio(brt)["samples"]] == [1000, 500]
    assert control.first_token_seconds(StreamingBedrock([None, "hi"]), "low") is not None


# -------------------------------------------------- the first-token measurement
class FakeReadTimeoutError(Exception):
    """Named so the harness's timeout branch recognises it the way botocore's is."""


class StreamingBedrock:
    """A stream whose first events carry no text. A harness that times the first
    event instead of the first *token* reports a latency that never happened."""

    def __init__(self, texts: list[str | None], raise_timeout: bool = False) -> None:
        self.texts = texts
        self.raise_timeout = raise_timeout

    def converse_stream(self, **kw):
        if self.raise_timeout:
            raise FakeReadTimeoutError("read timeout")
        events = [{"messageStart": {"role": "assistant"}}]
        for text in self.texts:
            events.append({"contentBlockDelta": {"delta": {} if text is None
                                                 else {"text": text}}})
        return {"stream": iter(events)}


def test_the_clock_stops_at_the_first_delta_that_carries_text() -> None:
    got = harness.first_token_seconds(StreamingBedrock([None, None, "你好"]), "low")
    assert got is not None and got >= 0.0


def test_a_stream_that_never_emits_text_is_not_a_latency_sample() -> None:
    """None, not 0.0 — a zero would silently pull the median down."""
    assert harness.first_token_seconds(StreamingBedrock([None, None]), "low") is None


def test_a_read_timeout_is_reported_as_a_timeout_not_an_error() -> None:
    """§6.0 rule 4: botocore's default turned 4 unfinished calls into 4 wrong ones."""
    assert harness.first_token_seconds(
        StreamingBedrock([], raise_timeout=True), "low") is None


def test_a_real_failure_is_not_swallowed_as_a_timeout() -> None:
    class Broken:
        def converse_stream(self, **kw):
            raise ValueError("AccessDeniedException")
    with pytest.raises(ValueError):
        harness.first_token_seconds(Broken(), "low")


def test_the_effort_field_is_only_sent_when_asked_for() -> None:
    seen = {}

    class Recording(StreamingBedrock):
        def converse_stream(self, **kw):
            seen.update(kw)
            return super().converse_stream(**kw)

    harness.first_token_seconds(Recording(["hi"]), None)
    assert "additionalModelRequestFields" not in seen
    harness.first_token_seconds(Recording(["hi"]), "low")
    assert seen["additionalModelRequestFields"] == {"output_config": {"effort": "low"}}


@pytest.mark.mutation
def test_mutant_timing_the_first_event_understates_latency(tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (FIRST_TEXT, "            if event:"))
    # the messageStart event now stops the clock, so a text-less stream reports a
    # sample where the real harness reports None
    assert mutant.first_token_seconds(StreamingBedrock([None, None]), "low") is not None


# ------------------------------------- M3: is the parameter honoured, or ignored?
class FakeValidationException(Exception):
    """Named so the harness's rejection branch recognises it as botocore's would."""


class FakeThrottlingException(Exception):
    """A refusal that says nothing about the field. Must NOT count as validation."""


class EffortBedrock:
    """Validates `output_config` the way Bedrock does, and its OUTPUT LENGTH depends on
    the effort it was handed.

    Lying, not echoing (§6.1): the counts have nothing to do with the prompt, and
    `stopReason` is supplied INDEPENDENTLY of the count — so a harness that infers
    truncation from "the count equals the cap" and one that reads `stopReason` give
    different answers against this double, which is the only way to tell them apart.

    The tiny `maxTokens` validation probes are answered without consuming a length,
    so each condition's pool lands in the evidence exactly as written here.
    """

    PROBE_MAX_TOKENS = 100

    def __init__(self, lengths: dict, stop_reasons: dict | None = None,
                 validates: bool = True) -> None:
        self.lengths = lengths
        self.stop_reasons = stop_reasons or {}
        self.validates = validates
        self.calls: list[str | None] = []

    def converse(self, **kw):
        cfg = (kw.get("additionalModelRequestFields") or {}).get("output_config", {})
        effort = cfg.get("effort")
        if self.validates:
            if effort not in (None, "low", "high"):
                raise FakeValidationException("ValidationException: unsupported effort")
            if set(cfg) - {"effort"}:
                raise FakeValidationException("ValidationException: unknown key")
        if kw["inferenceConfig"]["maxTokens"] < self.PROBE_MAX_TOKENS:
            return {"usage": {"inputTokens": 5, "outputTokens": 3}, "stopReason": "end_turn"}
        index = self.calls.count(effort)
        self.calls.append(effort)
        pool = self.lengths[effort]
        return {"usage": {"inputTokens": 5, "outputTokens": pool[index % len(pool)]},
                "stopReason": self.stop_reasons.get(effort, "end_turn")}


HONOURED = {"low": [500, 519, 540], "high": [900, 900, 900], None: [900, 900, 900]}
CAPPED = {"high": "max_tokens", None: "max_tokens"}
# overlapping ranges: the median is lower but the interval straddles 0
OVERLAPPING = {"low": [400, 500, 600], "high": [500, 600, 700], None: [500, 600, 700]}
# shaped like the real measurement: unset-vs-low resolves, high-vs-low does not
PARTIAL = {"low": [100, 140, 180], "high": [90, 150, 210], None: [300, 300, 300]}


def test_a_field_that_accepts_nonsense_is_a_field_nobody_reads() -> None:
    """The whole point of channel A. Same lengths either way; only validation differs."""
    honest = harness.measure_effort_effect(EffortBedrock(HONOURED, CAPPED), 3)
    assert honest["channel_a_validation"]["garbage_value_rejected"] is True
    assert honest["channel_a_validation"]["unknown_output_config_key_rejected"] is True
    assert honest["parameter_is_read"] is True

    deaf = harness.measure_effort_effect(
        EffortBedrock(HONOURED, CAPPED, validates=False), 3)
    assert deaf["channel_a_validation"]["garbage_value_rejected"] is False
    assert deaf["parameter_is_read"] is False
    assert deaf["honored"] is False, (
        "an API that swallows a garbage effort value has not been shown to read the "
        "field, no matter what the lengths did")


def test_a_throttle_is_not_evidence_about_the_field() -> None:
    class Throttled:
        def converse(self, **kw):
            raise FakeThrottlingException("ThrottlingException: slow down")
    with pytest.raises(FakeThrottlingException):
        harness.rejects(Throttled(), {"output_config": {"effort": "low"}})


def test_the_length_channel_needs_an_interval_not_a_lower_median() -> None:
    """A lower median with an interval that straddles 0 is not evidence of a shorter
    answer. This is the case a bare point comparison gets wrong."""
    got = harness.measure_effort_effect(EffortBedrock(OVERLAPPING), 3)
    assert got["parameter_is_read"] is True          # channel A still fires
    assert got["median_difference_tokens"]["unset_minus_low"]["point_tokens"] > 0
    assert got["median_difference_tokens"]["unset_minus_low"][
        "ci_excludes_zero"] is False
    assert got["low_produces_shorter_output"] is False
    assert got["honored"] is False


def test_the_deciding_comparison_is_the_deployment_choice() -> None:
    """unset vs low, not high vs low: shipping means sending `effort: low` or sending
    nothing. Shaped like the real run, where high-vs-low does not resolve at n=10 but
    the question that matters does."""
    got = harness.measure_effort_effect(EffortBedrock(PARTIAL), 3)
    comps = got["median_difference_tokens"]
    assert comps["unset_minus_low"]["ci_excludes_zero"] is True
    assert comps["high_minus_low"]["ci_excludes_zero"] is False
    assert got["primary_comparison"].startswith("unset_minus_low")
    assert got["low_produces_shorter_output"] is True
    assert got["honored"] is True


def test_a_no_op_field_is_reported_as_a_no_op() -> None:
    flat = {"low": [700, 700, 700], "high": [700, 700, 700], None: [700, 700, 700]}
    got = harness.measure_effort_effect(EffortBedrock(flat), 3)
    assert got["median_difference_tokens"]["unset_minus_low"]["ci95_tokens"] == [0.0, 0.0]
    assert got["low_produces_shorter_output"] is False
    assert got["honored"] is False
    # this double IS equal across conditions, and the harness still must not say so:
    # a CI of [0, 0] is a measurement that found nothing, not a proof of equality
    assert got["default_equals_high"] == "unresolved at this n"


def test_every_call_survives_in_the_evidence() -> None:
    got = harness.measure_effort_effect(EffortBedrock(HONOURED, CAPPED), 3)
    for label, cond in got["channel_b_output_tokens"].items():
        assert cond["n"] == 3, label
        assert len(cond["output_tokens_samples"]) == 3, f"{label} lost its raw calls"
        assert cond["median_output_tokens"] == pytest.approx(
            statistics.median(cond["output_tokens_samples"]))
    assert got["channel_b_output_tokens"]["effort_low"]["output_tokens_samples"] == [
        500, 519, 540]


def test_truncation_is_read_from_stop_reason_not_inferred_from_the_count() -> None:
    """A condition whose counts sit exactly on the cap but which ended on its own is
    NOT a floor. Inferring truncation from `count == maxTokens` gets this backwards."""
    at_cap = {"low": [500, 500, 500],
              "high": [harness.EFFORT_LENGTH_MAX_TOKENS] * 3,
              None: [harness.EFFORT_LENGTH_MAX_TOKENS] * 3}
    ended = harness.measure_effort_effect(EffortBedrock(at_cap), 3)   # all end_turn
    assert ended["channel_b_output_tokens"]["effort_high"][
        "length_is_a_floor_not_a_length"] is False
    assert ended["truncation_understates_the_saving"] is False
    assert ended["conditions_that_hit_the_cap"] == []

    cut = harness.measure_effort_effect(EffortBedrock(at_cap, CAPPED), 3)
    assert cut["channel_b_output_tokens"]["effort_high"][
        "length_is_a_floor_not_a_length"] is True
    assert cut["truncation_understates_the_saving"] is True
    assert cut["conditions_that_hit_the_cap"] == ["effort_high", "effort_unset"]


def test_the_verdict_has_three_outcomes_not_two() -> None:
    """"effort does nothing to latency" and "effort is ignored" call for opposite
    decisions — aim it elsewhere, or stop sending it."""
    faster = {"high_minus_low_median": {"ci_excludes_zero": True, "point_ms": 40.0}}
    assert "measurably reduces" in harness.effort_verdict(faster)

    null = {"high_minus_low_median": {"ci_excludes_zero": False, "point_ms": 4.0}}
    assert "UNTESTED" in harness.effort_verdict(null), (
        "with no honoured-check attached, the verdict must admit it cannot tell")

    wrong_metric = {**null, "parameter_honored": {"honored": True,
                                                 "low_over_unset_median": 0.875}}
    sentence = harness.effort_verdict(wrong_metric)
    assert "wrong metric" in sentence and "0.875" in sentence

    no_op = {**null, "parameter_honored": {"honored": False}}
    assert "cannot distinguish" in harness.effort_verdict(no_op)


def test_the_length_measurement_uses_a_prompt_that_does_not_dictate_the_length() -> None:
    """This is not a style preference, it is the measured confound.

    Run against the latency prompt (`用三句話說明…`), all three conditions produced
    112-263 output tokens with overlapping ranges — the instruction fixed the length, so
    an honoured field and an ignored one looked identical, which is precisely the
    ambiguity M3 exists to remove. The prompt sent by `output_length` must therefore be
    free of length instructions, and the harness must be the thing that says so.
    """
    seen = []

    class Recording(EffortBedrock):
        def converse(self, **kw):
            seen.append(kw["messages"][0]["content"][0]["text"])
            return super().converse(**kw)

    harness.output_length(Recording(HONOURED), "low")
    assert seen == [harness.EFFORT_LENGTH_PROMPT]
    assert seen[0] != harness.EFFORT_PROMPT
    for phrase in harness.LENGTH_INSTRUCTIONS:
        assert phrase not in harness.EFFORT_LENGTH_PROMPT, (
            f"M3's prompt dictates its own length via {phrase!r} — the quantity it "
            "measures is then set by the prompt, not by effort")
    # and the guard is not vacuous: the list does catch the prompt it was written for
    assert any(p in harness.EFFORT_PROMPT for p in harness.LENGTH_INSTRUCTIONS)
    assert harness.measure_effort_effect(
        EffortBedrock(HONOURED, CAPPED), 1)["no_length_instruction"] is True


@pytest.mark.mutation
def test_mutant_using_the_point_estimate_calls_a_straddling_interval_decisive(
        tmp_path: Path) -> None:
    """Drop the interval and keep the point: the same defect as comparing a between-group
    spread against a point shift, which is what made the guardrail harness call a 19.6 ms
    language gap real against 40.6 ms of replication noise."""
    mutant = load_mutant(tmp_path, (
        INTERVAL, '    shorter = bool((primary.get("point_tokens") or 0) > 0)'))
    got = mutant.measure_effort_effect(EffortBedrock(OVERLAPPING), 3)
    assert got["low_produces_shorter_output"] is True
    assert got["honored"] is True
    assert harness.measure_effort_effect(
        EffortBedrock(OVERLAPPING), 3)["honored"] is False


@pytest.mark.mutation
def test_mutant_deciding_on_high_versus_low_answers_a_question_nobody_asked(
        tmp_path: Path) -> None:
    """At the measured n, high-vs-low does not resolve while unset-vs-low does. Keying
    the conclusion off the wrong pair throws away a resolved answer."""
    mutant = load_mutant(
        tmp_path, (PRIMARY, '    primary = comparisons.get("high_minus_low", {})'))
    assert mutant.measure_effort_effect(EffortBedrock(PARTIAL), 3)["honored"] is False
    assert harness.measure_effort_effect(EffortBedrock(PARTIAL), 3)["honored"] is True


@pytest.mark.mutation
def test_mutant_reading_a_straddling_interval_as_equality(tmp_path: Path) -> None:
    """§0-8 says the default is `high`. A CI that straddles 0 cannot establish that, and
    a harness that reports True there manufactures a confirmation of the plan."""
    mutant = load_mutant(tmp_path, (UNRESOLVED, "            True),"))
    flat = {"low": [700, 700, 700], "high": [700, 700, 700], None: [700, 700, 700]}
    assert mutant.measure_effort_effect(EffortBedrock(flat), 3)["default_equals_high"] \
        is True
    assert harness.measure_effort_effect(EffortBedrock(flat), 3)["default_equals_high"] \
        == "unresolved at this n"


@pytest.mark.mutation
def test_mutant_dropping_the_validation_channel_believes_a_deaf_api(
        tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (READ, "    read = True"))
    got = mutant.measure_effort_effect(
        EffortBedrock(HONOURED, CAPPED, validates=False), 3)
    assert got["parameter_is_read"] is True
    assert got["honored"] is True, (
        "without channel A the harness cannot tell a read field from an ignored one")


@pytest.mark.mutation
def test_mutant_truthy_block_reports_an_ignored_field_as_honoured(
        tmp_path: Path) -> None:
    """`if honored:` instead of `if honored.get("honored"):` — a non-empty dict is
    truthy whatever it says, so the no-op case gets the honoured sentence."""
    mutant = load_mutant(tmp_path, (HONORED_BRANCH, "    if honored:"))
    no_op = {"high_minus_low_median": {"ci_excludes_zero": False, "point_ms": 4.0},
             "parameter_honored": {"honored": False}}
    assert "wrong metric" in mutant.effort_verdict(no_op)
    assert "cannot distinguish" in harness.effort_verdict(no_op)


def test_the_stripped_sample_is_smaller_but_not_empty() -> None:
    """The prose-only sample exists to attribute the ratio to Chinese text rather
    than to markdown; a regex that ate the whole file would silently report a
    ratio for nothing."""
    shipped, prose = harness._samples()
    assert shipped[0] == "constitution_as_shipped"
    assert len(prose[1]) < len(shipped[1])
    assert len(prose[1]) > 0.5 * len(shipped[1]), "the strip regex ate the document"
    assert "<!--" not in prose[1]


# ------------------------------------------- the evidence the harness left behind
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "track-c"
TOKENS_JSON = EVIDENCE / "constitution-tokens.json"
LATENCY_JSON = EVIDENCE / "effort-latency.json"


def _skills_proxy_ratio() -> float:
    """PROXY_RATIO out of tests/test_skills_loader.py by AST, not by copying it.

    Two copies of a calibration is the WINS/BEATS defect: nothing forces them to
    agree, and the one that drifts is the one nobody reads.
    """
    import ast
    tree = ast.parse((TESTS_DIR / "test_skills_loader.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PROXY_RATIO" for t in node.targets):
            return float(ast.literal_eval(node.value))
    raise AssertionError("PROXY_RATIO is gone from tests/test_skills_loader.py — the "
                         "proxy calibration this file cross-checks against has moved")


def test_the_measurement_used_the_model_the_agent_actually_calls() -> None:
    """A budget measured on another model is not this system's budget."""
    evidence = json.loads(TOKENS_JSON.read_text())
    agent_src = (REPO_ROOT / "agent" / "agent_node.py").read_text()
    assert f'MODEL_ID = "{evidence["model"]}"' in agent_src
    assert f'REGION = "{evidence["region"]}"' in agent_src


def test_the_character_proxy_replicates_on_a_second_text() -> None:
    """The skills L1 index gave proxy/real = 0.79 on 2026-08-22. If the constitution
    lands near it, one calibration covers both budgets; if not, the proxy is
    text-dependent and every character cap in the repo is decoration."""
    ratio = _skills_proxy_ratio()
    evidence = json.loads(TOKENS_JSON.read_text())
    for sample in evidence["samples"]:
        assert sample["proxy_over_real"] == pytest.approx(ratio, abs=0.05), (
            f"{sample['name']}: proxy/real {sample['proxy_over_real']} vs the skills "
            f"calibration {ratio} — the two budgets no longer share one calibration")


def test_the_recorded_constitution_cost_matches_the_shipped_file() -> None:
    """Evidence that describes a different file than the one on disk is stale."""
    evidence = json.loads(TOKENS_JSON.read_text())
    shipped = (REPO_ROOT / "safety" / "constitution.md").read_text()
    as_shipped = next(s for s in evidence["samples"]
                      if s["name"] == "constitution_as_shipped")
    assert as_shipped["chars"] == len(shipped), (
        "constitution.md changed since the token measurement — re-run "
        "safety/token_cost_harness.py --skip-latency")


def test_the_latency_verdict_is_derived_not_asserted_in_prose() -> None:
    """§6.0 rule 6: the sentence in the JSON must follow from the numbers in it, and it
    must be the sentence the shipped function produces from them — re-derived here
    rather than pattern-matched, so editing the prose without re-running goes red."""
    evidence = json.loads(LATENCY_JSON.read_text())
    diff = evidence["high_minus_low_median"]
    lo, hi = diff["ci95_ms"]
    assert diff["ci_excludes_zero"] == ((lo > 0) == (hi > 0))
    decisive = diff["ci_excludes_zero"] and diff["point_ms"] > 0
    assert decisive == ("measurably reduces" in evidence["verdict"])
    assert evidence["verdict"] == harness.effort_verdict(evidence)


def test_the_recorded_run_checked_that_effort_is_honoured_at_all() -> None:
    """Without this block the latency null result is uninterpretable: it would be
    equally consistent with Bedrock ignoring the field."""
    evidence = json.loads(LATENCY_JSON.read_text())
    honored = evidence["parameter_honored"]
    assert honored["channel_a_validation"]["garbage_value_rejected"] is True
    assert honored["channel_a_validation"]["known_good_value_accepted"] is True
    assert honored["no_length_instruction"] is True, (
        "this run measured length under a prompt that fixes the length")
    low = honored["channel_b_output_tokens"]["effort_low"]
    unset = honored["channel_b_output_tokens"]["effort_unset"]
    for label, cond in honored["channel_b_output_tokens"].items():
        assert cond["n"] >= 10, f"{label} ran {cond['n']} calls"
        assert len(cond["output_tokens_samples"]) == cond["n"], f"{label} lost its calls"
    # every derived flag re-derived from the raw samples in the same file
    primary = honored["median_difference_tokens"]["unset_minus_low"]
    lo, hi = primary["ci95_tokens"]
    assert primary["ci_excludes_zero"] == (lo > 0 or hi < 0)
    assert honored["low_produces_shorter_output"] == (
        primary["ci_excludes_zero"] and primary["point_tokens"] > 0)
    assert honored["honored"] == (honored["parameter_is_read"]
                                 and honored["low_produces_shorter_output"])
    assert honored["low_over_unset_median"] == pytest.approx(
        statistics.median(low["output_tokens_samples"])
        / statistics.median(unset["output_tokens_samples"]), abs=0.001)
    assert honored["default_equals_high"] is not True, (
        "a straddling interval cannot confirm §0-8's claim that the default is high")


def test_the_recorded_summary_is_rebuildable_from_its_own_raw_calls() -> None:
    """§6.0 rule 6, made executable: a derived block that cannot be reproduced from the
    samples beside it is a number nobody can check. Read-only — this asserts the shipped
    file already agrees with the shipped arithmetic, so a stale block goes red instead of
    being quietly repaired by `--rederive`."""
    evidence = json.loads(LATENCY_JSON.read_text())
    honored = evidence["parameter_honored"]
    samples = {k: v["output_tokens_samples"]
               for k, v in honored["channel_b_output_tokens"].items()}
    fresh = harness._effect_derivation(honored["channel_a_validation"],
                                      honored["channel_b_output_tokens"], samples)
    for key, value in fresh.items():
        assert honored[key] == value, (
            f"{key} in the evidence does not follow from the raw calls in the same file "
            "— re-run safety/token_cost_harness.py --rederive")
    assert evidence["verdict"] == harness.effort_verdict(evidence)


def test_a_capped_condition_is_never_reported_as_a_finished_length() -> None:
    """The measured `high` runs hit maxTokens, so the ratio is an upper bound and the
    file has to say so — a truncated length quoted as a length understates the saving."""
    honored = json.loads(LATENCY_JSON.read_text())["parameter_honored"]
    for label, cond in honored["channel_b_output_tokens"].items():
        floor = cond["stopped_at_max_tokens"] > 0
        assert cond["length_is_a_floor_not_a_length"] == floor, label
        if floor:
            assert "max_tokens" in cond["stop_reasons_seen"], label


def test_both_effort_conditions_reached_the_planned_sample_size() -> None:
    evidence = json.loads(LATENCY_JSON.read_text())
    for label, cond in evidence["conditions"].items():
        assert cond["n"] >= 30, f"{label} ran {cond['n']} calls"
        assert cond["n"] == len(cond["samples_ms"]), f"{label} lost its raw calls"
        assert cond["p95_is_really_max"] is False, f"{label} n is below 20"
        assert "timeouts_not_scored_as_errors" in cond


def test_the_harness_does_not_run_on_import() -> None:
    """Importing it in CI must not call Bedrock; only __main__ measures."""
    src = HARNESS_PY.read_text()
    assert 'if __name__ == "__main__":' in src
    assert "\nmain()" not in src
