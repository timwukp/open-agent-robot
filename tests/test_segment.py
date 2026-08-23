"""Offline tests for S4's deterministic episode segmentation — T-B1, T-B2, T-B4.

**T-B1 asks for something stronger than "each rule works".**
The table's own wording is 「逐條停用 → 只有對應那一案紅」: the point is not that five
rules fire, it is that the five assertions do not mask one another. A suite where the
silence stream also happens to trip the activity rule stays green when the silence
rule is deleted, and nothing in "each rule produces a boundary" notices. So the shape
here is a 5×5 identity: disabling rule *r* must change the segmentation of stream *r*
and leave the other four **bit-identical**. That is one assertion that cannot be
satisfied by an overlapping fixture, and the streams are indexed by rule name so a
sixth rule with no stream reds the derivation guard rather than being skipped.

**T-B2's stated oracle cannot kill the mutant it names.**
The table says: 60 s with no messages at all, and 「邊界仍產生」. A purely
event-driven segmenter produces that boundary too — it notices the gap when the next
event finally arrives, at the same index, with the same reason. Two of the tests below
demonstrate exactly that: on the gap stream the timer-less mutant is byte-identical to
the real module. What distinguishes them is *when* the episode closes, so the real
assertion is that it closes with **no later event in the stream at all** — because
closing is what triggers S5's summarisation and embedding, and a robot left alone for
an evening should not have that work deferred until somebody next speaks.

**T-B4's stated oracle cannot kill the ordering mutant either, for a subtler reason.**
「同輸入跑兩次比 hash」 is run inside one process, and `set` iteration order is fixed
for the life of a process. A `set(self.rules)` mutant is therefore perfectly
reproducible within a run and still order-dependent between runs — green under the
literal test, wrong on the next machine. Killing it needs two processes with different
`PYTHONHASHSEED`, and it needs an event that trips two rules whose *relative* order
differs between those two seeds (`activity` before `scene` under one, after it under
the other — measured, not assumed; the test reds if a future Python stops
discriminating rather than passing vacuously). A `sorted(self.rules)` mutant is
deterministic across processes and needs the opposite treatment: a single event
tripping `silence` and `activity`, where sorted order and specification order
disagree. Two mutants, two assertions, neither redundant.

**The hard-cap asymmetry is a finding, not a preference.**
`memory/segment.py` closes on the turn cap *after* appending and on the span cap
*before*. The mutant that makes the span cap inclusive — the shape it would naturally
have if written to match the turn cap — produces an episode spanning 320 s under a
300 s cap, i.e. the rule that exists to bound the span becomes the reason it is
exceeded. `cap_findings()` reports it, which is the same invariant T-B3 will check
against a day of real logs; that half of T-B3 needs human annotation and is not
claimed here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_PY = REPO_ROOT / "memory" / "segment.py"

from memory import segment as sg  # noqa: E402


# --------------------------------------------------------------------- fixtures
def ev(index: int, ts: float, *, kind: str = sg.TURN_KIND,
       activity: str = "conversation") -> dict:
    """One stored event. `seq` deliberately differs from the index.

    A store sequence number and a position in a fixture are two different claims, and
    if they were equal here an assertion on `indices` would pass while reading `seqs`.
    """
    return {"seq": 100 + index, "ts": float(ts), "kind": kind, "activity": activity}


# One stream per rule, keyed by the rule it is meant to trip — and *only* that rule.
def _silence_stream() -> list[dict]:
    # gaps of 10 s, then one of 80 s. Nothing else moves: same activity, same kind,
    # 5 turns, 110 s total.
    return [ev(i, t) for i, t in enumerate((0, 10, 20, 100, 110))]


def _activity_stream() -> list[dict]:
    return [ev(0, 0), ev(1, 10), ev(2, 20),
            ev(3, 30, activity="game"), ev(4, 40, activity="game")]


def _scene_stream() -> list[dict]:
    return [ev(0, 0), ev(1, 10), ev(2, 20),
            ev(3, 30, kind=sg.SCENE_KIND), ev(4, 40)]


def _max_turns_stream() -> list[dict]:
    # 21 turns 5 s apart: 105 s span, no gap over 45 s, one activity, no scene change.
    return [ev(i, i * 5) for i in range(21)]


def _max_span_stream() -> list[dict]:
    # 40 s apart, which is under the 45 s silence threshold on purpose: this is the
    # only stream on which the span cap can be reached without the silence rule
    # firing first, and it is the stream that exposes the inclusive-cap bug.
    return [ev(i, i * 40) for i in range(9)]


STREAMS = {
    "silence": _silence_stream,
    "activity": _activity_stream,
    "scene": _scene_stream,
    "max_turns": _max_turns_stream,
    "max_span": _max_span_stream,
}


def shape(episodes) -> list[tuple]:
    """The comparable form of a segmentation. Rounded, so ulp noise is not a boundary."""
    return [(ep.indices, ep.seqs, ep.reason, ep.turns, round(ep.span, 6), ep.activity)
            for ep in episodes]


def digest(episodes) -> str:
    return hashlib.sha256(
        json.dumps([ep.as_dict() for ep in episodes], sort_keys=True,
                   ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/segment.py. Zero replacements = control.

    Same idiom as tests/test_salience.py: the copy keeps importing the *real*
    `memory.store`, and each anchor is asserted present and unique so a refactor turns
    a stale mutant into a red test rather than a silently vacuous one.
    """
    dst = tmp_path / f"mut_{len(replacements)}_{abs(hash(replacements)) % 10**8}"
    dst.mkdir(parents=True, exist_ok=True)
    src = SEGMENT_PY.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in segment.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    out = dst / "segment.py"
    out.write_text(src)
    name = f"segment_mutant_{dst.name}"
    spec = importlib.util.spec_from_file_location(name, out)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


TICK_GUARD = "        if not self._open:\n            return []\n"
RULE_LOOP = "        for rule in self.rules:\n"
FIRST_TS_RESET = "        self._first_ts = None\n"
TURN_COUNT = '        if event["kind"] == TURN_KIND:\n'
MAX_TURNS_CHECK = '        if "max_turns" in self.rules and self._turns >= self.max_turns:\n'
PRE_RULES_LINE = 'PRE_RULES = ("silence", "activity", "scene", "max_span")\n'
KIND_CHECK = "    if kind not in KINDS:\n"
ORDER_GUARD = "        if self._last_ts is not None and ts < self._last_ts:\n"
SETATTR_GUARD = "    def __setattr__(self, name, value):\n"


# ------------------------------------------------------------ derivation guards
def test_every_rule_has_a_stream_that_isolates_it():
    """§0 rule 5: a hand-written stream table cannot notice a sixth rule."""
    assert set(STREAMS) == set(sg.RULES), (
        f"streams cover {sorted(STREAMS)} but the rule set is {sorted(sg.RULES)}; a "
        "rule with no isolating stream is a rule T-B1 does not test")
    for name, build in STREAMS.items():
        events = build()
        assert len(events) >= 5, f"stream {name!r} collapsed to {len(events)} events"


def test_the_plan_numbers_are_the_shipped_defaults():
    assert (sg.SILENCE_S, sg.MAX_TURNS, sg.MAX_SPAN_S) == (45.0, 20, 300.0)
    seg = sg.Segmenter()
    assert (seg.silence_s, seg.max_turns, seg.max_span_s) == (45.0, 20, 300.0)


def test_pre_rules_is_a_subset_of_rules_and_excludes_only_the_turn_cap():
    assert set(sg.PRE_RULES) < set(sg.RULES)
    assert set(sg.RULES) - set(sg.PRE_RULES) == {"max_turns"}, (
        "the turn cap is the only rule that closes after appending; anything else "
        "closing late can produce an episode past its own cap")


def test_reasons_extends_rules_with_end_of_stream_only():
    assert sg.REASONS[:len(sg.RULES)] == sg.RULES
    assert set(sg.REASONS) - set(sg.RULES) == {"end_of_stream"}


# ------------------------------------------------------- T-B1: exact boundaries
def test_silence_stream_cuts_at_the_eighty_second_gap():
    eps = sg.replay(_silence_stream())
    assert shape(eps) == [
        ((0, 1, 2), (100, 101, 102), "silence", 3, 20.0, "conversation"),
        ((3, 4), (103, 104), "end_of_stream", 2, 10.0, "conversation"),
    ]


def test_activity_stream_cuts_where_the_activity_changes():
    eps = sg.replay(_activity_stream())
    assert shape(eps) == [
        ((0, 1, 2), (100, 101, 102), "activity", 3, 20.0, "conversation"),
        ((3, 4), (103, 104), "end_of_stream", 2, 10.0, "game"),
    ]


def test_scene_stream_cuts_before_the_scene_change_event():
    eps = sg.replay(_scene_stream())
    assert shape(eps) == [
        ((0, 1, 2), (100, 101, 102), "scene", 3, 20.0, "conversation"),
        # the scene_change itself is not a turn: two events, one turn.
        ((3, 4), (103, 104), "end_of_stream", 1, 10.0, "conversation"),
    ]


def test_max_turns_closes_inclusively_at_exactly_twenty():
    eps = sg.replay(_max_turns_stream())
    assert [ep.reason for ep in eps] == ["max_turns", "end_of_stream"]
    assert eps[0].turns == 20 and eps[0].indices == tuple(range(20))
    assert eps[1].indices == (20,)


def test_max_span_closes_exclusively_so_the_cap_is_never_exceeded():
    eps = sg.replay(_max_span_stream())
    assert [ep.reason for ep in eps] == ["max_span", "end_of_stream"]
    assert eps[0].indices == tuple(range(8))
    assert eps[0].span == 280.0 <= sg.MAX_SPAN_S
    assert eps[1].indices == (8,)


def test_no_stream_trips_a_rule_other_than_its_own():
    """The premise the 5×5 identity rests on. Without it the matrix is decorative."""
    for name, build in STREAMS.items():
        reasons = {ep.reason for ep in sg.replay(build())} - {"end_of_stream"}
        assert reasons == {name}, f"stream {name!r} also tripped {reasons - {name}}"


@pytest.mark.mutation
@pytest.mark.parametrize("disabled", sg.RULES)
def test_mutant_one_rule_disabled_changes_only_its_own_stream(disabled):
    """T-B1's named mutant, as an identity rather than five separate checks.

    Disabling a rule is a supported argument rather than a text patch precisely so
    this can be stated as "changed if and only if it is this rule's stream" — a
    patched copy could show that a rule matters, never that the five assertions are
    independent.
    """
    kept = [r for r in sg.RULES if r != disabled]
    for name, build in STREAMS.items():
        events = build()
        before = shape(sg.replay(events))
        after = shape(sg.replay(events, rules=kept))
        assert (before != after) == (name == disabled), (
            f"disabling {disabled!r} changed stream {name!r}: {before != after}; "
            f"expected change only on {disabled!r}")


@pytest.mark.mutation
def test_mutant_disabling_the_turn_cap_breaks_the_hard_cap_invariant():
    """T-B3's 「放寬硬上限 → 不變量那條紅」, on synthetic data.

    The F1-against-annotation half of T-B3 needs a day of real logs annotated by hand;
    the invariant half does not, and it is the half that is machine-checkable.
    """
    events = _max_turns_stream()
    assert sg.cap_findings(sg.replay(events)) == []
    loose = sg.replay(events, rules=[r for r in sg.RULES if r != "max_turns"])
    findings = sg.cap_findings(loose)
    assert len(findings) == 1 and "21 turns" in findings[0], findings


def test_cap_findings_is_empty_on_every_stream():
    for name, build in STREAMS.items():
        assert sg.cap_findings(sg.replay(build())) == [], name


@pytest.mark.mutation
def test_mutant_an_inclusive_span_cap_exceeds_its_own_cap(tmp_path):
    """The asymmetry in the module docstring, made executable.

    Written to match the turn cap — close after appending, on `>=` — the span rule
    admits the event that pushes the episode past 300 s and then closes, so the
    episode it produces spans 320 s. The real module closes first and produces 280 s.
    """
    m = load_mutant(
        tmp_path,
        (PRE_RULES_LINE, 'PRE_RULES = ("silence", "activity", "scene")\n'),
        (MAX_TURNS_CHECK,
         '        if "max_span" in self.rules and self._first_ts is not None \\\n'
         '                and self._last_ts - self._first_ts >= self.max_span_s:\n'
         '            out.append(self._close("max_span"))\n'
         + MAX_TURNS_CHECK),
    )
    events = _max_span_stream()
    real = sg.replay(events)
    bad = m.replay(events)
    assert real[0].span == 280.0
    assert bad[0].span == 320.0
    assert sg.cap_findings(real) == []
    findings = m.cap_findings(bad)
    assert len(findings) == 1 and "320.0s" in findings[0], findings


@pytest.mark.mutation
def test_mutant_every_kind_counts_against_the_turn_cap(tmp_path):
    """A scene change is a boundary signal, not a turn."""
    m = load_mutant(tmp_path, (TURN_COUNT, "        if True:\n"))
    # 19 turns and one non-turn event: 20 events, 19 turns. Interleaved so the
    # non-turn event is not the one that would close anything.
    events = [ev(i, i * 5, kind="gesture" if i == 5 else sg.TURN_KIND)
              for i in range(20)]
    assert [e["kind"] for e in events].count("gesture") == 1
    real = sg.replay(events)
    assert [ep.reason for ep in real] == ["end_of_stream"] and real[0].turns == 19
    bad = m.replay(events)
    assert [ep.reason for ep in bad] == ["max_turns"] and bad[0].turns == 20


# ------------------------------------------------------------- T-B2: the timer
GAP_EVENTS = [ev(0, 0), ev(1, 10), ev(2, 20)]
LATE_EVENT = ev(3, 100)


def test_a_tick_closes_the_episode_with_no_later_event_in_the_stream():
    """T-B2, in the form that actually distinguishes the two designs.

    `asr_node.py:79-87` publishes nothing during silence, so the 60 s of quiet in the
    middle of this stream is 60 s of no input at all. The stream ends there — nothing
    arrives afterwards — and the episode must still be closed. The threshold is
    measured from the last event, so the 60 s of quiet the table describes is a tick
    at 80 s after an event at 20 s.
    """
    eps = sg.replay(GAP_EVENTS, ticks=[30.0, 50.0, 80.0])
    assert shape(eps) == [((0, 1, 2), (100, 101, 102), "silence", 3, 20.0,
                           "conversation")]
    # No trailing end_of_stream: `flush()` found nothing open, which is the whole
    # claim. An episode still open here is one S5 would never be handed.


def test_a_tick_before_the_threshold_does_not_close_anything():
    """The positive control's mirror: if any tick closed an episode, the assertion
    above would hold for a segmenter that cuts on every timer fire."""
    eps = sg.replay(GAP_EVENTS, ticks=[30.0, 44.9])
    assert [ep.reason for ep in eps] == ["end_of_stream"]
    assert eps[0].indices == (0, 1, 2)


def test_the_tick_threshold_is_inclusive_at_exactly_forty_five_seconds():
    assert [ep.reason for ep in sg.replay(GAP_EVENTS, ticks=[65.0])] == ["silence"]
    assert [ep.reason for ep in sg.replay(GAP_EVENTS, ticks=[64.999])] == \
        ["end_of_stream"]


def test_a_tick_can_close_on_the_span_cap_with_no_event_arriving():
    """The span cap has the same problem as the silence rule: a conversation that
    stops mid-episode 299 s in must not stay open for hours."""
    events = [ev(i, i * 40) for i in range(5)]     # span 160 s, gaps under 45 s
    eps = sg.replay(events, ticks=[160.1, 200.0, 320.0])
    # the silence rule reaches its threshold first (160 + 45 = 205 < 320)
    assert [ep.reason for ep in eps] == ["silence"]
    # with silence off, the span cap is what closes it — from a tick, not an event
    eps = sg.replay(events, ticks=[320.0], rules=[r for r in sg.RULES if r != "silence"])
    assert [ep.reason for ep in eps] == ["max_span"]


@pytest.mark.mutation
def test_mutant_no_timer_is_indistinguishable_on_the_gap_stream(tmp_path):
    """Why 「邊界仍產生」 is the wrong oracle, stated as an assertion.

    On a stream where somebody eventually speaks again, the timer-less segmenter is
    **byte-identical** to the real one: same index, same reason, same everything. Any
    test that only looks at the boundary passes this mutant.
    """
    m = load_mutant(tmp_path, (TICK_GUARD, "        if True:\n            return []\n"))
    events = GAP_EVENTS + [LATE_EVENT]
    assert digest(m.replay(events)) == digest(sg.replay(events))
    assert [ep.reason for ep in sg.replay(events)] == ["silence", "end_of_stream"]


@pytest.mark.mutation
def test_mutant_no_timer_leaves_the_episode_open_when_nothing_follows(tmp_path):
    """The same mutant, killed. The discriminator is *when*, not *where*."""
    m = load_mutant(tmp_path, (TICK_GUARD, "        if True:\n            return []\n"))
    assert [ep.reason for ep in sg.replay(GAP_EVENTS, ticks=[80.0])] == ["silence"]
    assert [ep.reason for ep in m.replay(GAP_EVENTS, ticks=[80.0])] == ["end_of_stream"]


def test_a_tie_puts_the_tick_before_the_event():
    """Stated in `replay`, so it has to be asserted rather than inherited from sort().

    A tick at exactly the instant the silence ends means "45 s have passed with
    nothing new", which was true until this event. Ordering the event first would let
    the utterance that breaks a silence cancel the silence it broke.
    """
    events = GAP_EVENTS + [ev(3, 65.0)]
    assert [ep.reason for ep in sg.replay(events, ticks=[65.0])] == \
        ["silence", "end_of_stream"]
    # and the boundary is the same one the gap alone would have produced
    assert sg.replay(events)[0].indices == sg.replay(events, ticks=[65.0])[0].indices


# --------------------------------------------------- T-B4: determinism and order
MIXED = (_silence_stream() + [ev(5, 200, activity="game"), ev(6, 210, activity="game"),
                              ev(7, 220, kind=sg.SCENE_KIND, activity="game")])


def test_the_same_input_twice_gives_the_same_digest():
    """T-B4 as the table states it. Necessary, and not sufficient — see below."""
    assert digest(sg.replay(MIXED, ticks=[300.0])) == \
        digest(sg.replay(MIXED, ticks=[300.0]))


def test_two_rules_at_once_resolve_in_specification_order():
    """An event that is both a silence gap and an activity change reports `silence`."""
    events = [ev(0, 0), ev(1, 10), ev(2, 100, activity="game")]
    assert [ep.reason for ep in sg.replay(events)] == ["silence", "end_of_stream"]
    events = [ev(0, 0), ev(1, 10), ev(2, 20, kind=sg.SCENE_KIND, activity="game")]
    assert [ep.reason for ep in sg.replay(events)] == ["activity", "end_of_stream"]


def test_the_rule_order_is_not_reorderable_by_an_argument():
    seg = sg.Segmenter(rules=["max_turns", "silence", "activity", "scene", "max_span"])
    assert seg.rules == sg.RULES


@pytest.mark.mutation
def test_mutant_sorted_rule_order_reports_the_wrong_reason(tmp_path):
    """Deterministic ordering mutant: alphabetical puts `activity` before `silence`."""
    m = load_mutant(tmp_path, (RULE_LOOP, "        for rule in sorted(self.rules):\n"))
    events = [ev(0, 0), ev(1, 10), ev(2, 100, activity="game")]
    assert [ep.reason for ep in sg.replay(events)] == ["silence", "end_of_stream"]
    assert [ep.reason for ep in m.replay(events)] == ["activity", "end_of_stream"]


HASH_SEEDS = ("0", "1")
DRIVER = """\
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import segment as sg
events = json.load(open(sys.argv[3]))
eps = sg.replay(events)
print(hashlib.sha256(json.dumps([e.as_dict() for e in eps], sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest())
"""


def run_in_subprocess(tmp_path: Path, module_dir: Path, events: list[dict],
                      seed: str) -> str:
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    payload = tmp_path / f"events_{seed}.json"
    payload.write_text(json.dumps(events))
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(
        [sys.executable, str(driver), str(module_dir), str(REPO_ROOT), str(payload)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120)
    assert out.returncode == 0, f"driver failed under seed {seed}: {out.stderr}"
    return out.stdout.strip()


# An event that is both a scene change and an activity change. Chosen because the two
# rules' *relative* order differs between the two hash seeds under a set(), which is
# what makes the cross-seed assertion able to discriminate at all.
TWO_RULE_EVENTS = [ev(0, 0), ev(1, 10),
                   ev(2, 20, kind=sg.SCENE_KIND, activity="game"), ev(3, 30,
                                                                     activity="game")]


@pytest.mark.mutation
def test_control_no_mutant_is_deterministic_across_hash_seeds(tmp_path):
    """§0 rule 2's no-mutant control. If this reds, the harness is broken, not the code."""
    control = load_mutant(tmp_path)          # zero replacements
    assert control.RULES == sg.RULES
    module_dir = Path(control.__file__).parent
    digests = {s: run_in_subprocess(tmp_path, module_dir, TWO_RULE_EVENTS, s)
               for s in HASH_SEEDS}
    assert len(set(digests.values())) == 1, digests
    assert digests[HASH_SEEDS[0]] == digest(sg.replay(TWO_RULE_EVENTS))


@pytest.mark.mutation
def test_mutant_set_rule_order_survives_one_process_and_dies_across_two(tmp_path):
    """The mutant T-B4's literal form cannot kill, and the form that kills it.

    Within a process `set` iteration is fixed, so "run it twice and compare" is green.
    Across two processes with different `PYTHONHASHSEED` the reported reason changes.
    The second assertion also guards the first: if a future Python stopped
    discriminating between these seeds, this test reds instead of passing vacuously.
    """
    m = load_mutant(tmp_path, (RULE_LOOP, "        for rule in set(self.rules):\n"))
    assert digest(m.replay(TWO_RULE_EVENTS)) == digest(m.replay(TWO_RULE_EVENTS)), \
        "the set mutant is reproducible in-process — that is the point"
    module_dir = Path(m.__file__).parent
    digests = {s: run_in_subprocess(tmp_path, module_dir, TWO_RULE_EVENTS, s)
               for s in HASH_SEEDS}
    assert len(set(digests.values())) == 2, (
        f"seeds {HASH_SEEDS} no longer discriminate set iteration order for "
        f"{sg.RULES}; pick a pair that does, or this mutant is untested: {digests}")


# ------------------------------------------------------- refusals, not defaults
def test_every_declared_kind_is_accepted():
    """Positive control for the refusal below: a checker that rejects everything
    would satisfy "unknown kinds are refused" while breaking the system."""
    for kind in sg.KINDS:
        sg.check_event(ev(0, 0.0, kind=kind))
    for activity in sg.ACTIVITIES:
        sg.check_event(ev(0, 0.0, activity=activity))


@pytest.mark.parametrize("bad", [
    {"kind": "typo", "ts": 0.0, "activity": "conversation"},
    {"kind": None, "ts": 0.0, "activity": "conversation"},
    {"ts": 0.0, "activity": "conversation"},
])
def test_an_unknown_kind_is_refused(bad):
    with pytest.raises(sg.SegmentError, match="unknown event kind"):
        sg.check_event(bad)


@pytest.mark.parametrize("ts", [None, "0.0", True, [0.0]])
def test_a_missing_or_non_numeric_ts_is_refused(ts):
    with pytest.raises(sg.SegmentError, match="did not come from it"):
        sg.check_event({"kind": sg.TURN_KIND, "ts": ts, "activity": "conversation"})


def test_an_unknown_activity_is_refused():
    with pytest.raises(sg.SegmentError, match="unknown activity"):
        sg.check_event({"kind": sg.TURN_KIND, "ts": 0.0, "activity": "chatting"})


def test_an_out_of_order_event_is_refused():
    seg = sg.Segmenter()
    seg.feed(ev(0, 10.0), 0)
    with pytest.raises(sg.SegmentError, match="precedes the previous"):
        seg.feed(ev(1, 9.0), 1)


def test_an_out_of_order_event_is_refused_across_a_boundary():
    """The seam the surviving `_last_ts` exists for.

    A boundary reached by `feed` immediately reopens with that event, so `_last_ts` is
    reassigned on the way out and the reset would be invisible. The seam is a boundary
    reached by `tick`: nothing is open, nothing has been appended, and the only thing
    standing between the store and an event timestamped before the episode that just
    closed is the timestamp this segmenter kept.
    """
    seg = sg.Segmenter()
    seg.feed(ev(0, 100.0), 0)
    assert [e.reason for e in seg.tick(200.0)] == ["silence"]
    assert not seg.is_open()
    with pytest.raises(sg.SegmentError, match="precedes the previous"):
        seg.feed(ev(1, 50.0), 1)


@pytest.mark.mutation
def test_mutant_resetting_last_ts_at_a_boundary_accepts_time_going_backwards(tmp_path):
    m = load_mutant(tmp_path,
                    (FIRST_TS_RESET, FIRST_TS_RESET + "        self._last_ts = None\n"))
    seg = m.Segmenter()
    seg.feed(ev(0, 100.0), 0)
    assert [e.reason for e in seg.tick(200.0)] == ["silence"]
    seg.feed(ev(1, 50.0), 1)          # the real module raises here
    assert seg.is_open() and seg.flush()[0].start_ts == 50.0


@pytest.mark.mutation
def test_mutant_accepting_an_unknown_kind_switches_off_the_turn_cap(tmp_path):
    """The docstring's claim: an unrecognised kind that "contributes nothing" is how
    the turn cap stops existing while every event keeps flowing."""
    m = load_mutant(tmp_path, (KIND_CHECK, "    if False:\n"))
    events = [{"seq": 100 + i, "ts": i * 5.0, "kind": "trun",
               "activity": "conversation"} for i in range(25)]
    with pytest.raises(sg.SegmentError, match="unknown event kind"):
        sg.replay(events)
    eps = m.replay(events)
    assert [ep.reason for ep in eps] == ["end_of_stream"]
    assert eps[0].turns == 0 and len(eps[0].indices) == 25


@pytest.mark.mutation
def test_mutant_removing_the_order_guard_invents_a_boundary(tmp_path):
    m = load_mutant(tmp_path, (ORDER_GUARD, "        if False:\n"))
    events = [ev(0, 0.0), ev(1, 100.0), ev(2, 50.0)]
    with pytest.raises(sg.SegmentError):
        sg.replay(events)
    # The mutant segments it happily: the gap to the backwards event is negative, so
    # no rule fires and it joins the open episode, whose end_ts now precedes its
    # start_ts. A negative span is not a boundary error — it is an episode that
    # claims to have ended before it began, and every downstream cap reads that span.
    eps = m.replay(events)
    assert [ep.reason for ep in eps] == ["silence", "end_of_stream"]
    assert eps[1].span == -50.0
    assert m.cap_findings(eps) == []


@pytest.mark.parametrize("kw", [
    {"silence_s": 0}, {"silence_s": -1}, {"max_turns": 0}, {"max_span_s": 0},
])
def test_non_positive_thresholds_are_refused(kw):
    with pytest.raises(sg.SegmentError, match="must be positive"):
        sg.Segmenter(**kw)


def test_an_unknown_rule_name_is_refused():
    with pytest.raises(sg.SegmentError, match="unknown rule"):
        sg.Segmenter(rules=["silence", "sillence"])


def test_an_unknown_boundary_reason_is_refused():
    with pytest.raises(sg.SegmentError, match="unknown boundary reason"):
        sg.Episode("timeout", (1,), (0,), 0.0, 1.0, 1, "conversation")


# ------------------------------------------------------------------ the episode
def test_a_closed_episode_is_immutable():
    ep = sg.replay(_silence_stream())[0]
    with pytest.raises(sg.SegmentError, match="immutable"):
        ep.reason = "activity"
    with pytest.raises(sg.SegmentError, match="immutable"):
        ep.turns = 99


@pytest.mark.mutation
def test_mutant_a_mutable_episode_can_be_relabelled_after_the_fact(tmp_path):
    """A boundary that can be edited after it is recorded is not a record of one —
    and `cap_findings` reads exactly the fields that would be edited."""
    m = load_mutant(tmp_path, (SETATTR_GUARD, "    def _disabled(self, name, value):\n"))
    ep = m.replay(_max_turns_stream())[0]
    ep.turns = 3
    assert m.cap_findings([ep]) == []


def test_an_episode_serialises_to_json_without_the_fixture_indices():
    """`indices` are a fixture concept; what a store round-trips is `seqs`."""
    ep = sg.replay(_silence_stream())[0]
    d = ep.as_dict()
    assert "indices" not in d
    assert d["seqs"] == [100, 101, 102]
    assert json.loads(json.dumps(d)) == d


def test_an_empty_stream_produces_no_episodes():
    assert sg.replay([]) == []
    seg = sg.Segmenter()
    assert seg.flush() == [] and seg.tick(10_000.0) == [] and not seg.is_open()


def test_open_turns_counts_only_turns():
    seg = sg.Segmenter()
    seg.feed(ev(0, 0.0, kind=sg.SCENE_KIND), 0)
    assert seg.open_turns == 0 and seg.is_open()
    seg.feed(ev(1, 1.0), 1)
    assert seg.open_turns == 1


# ------------------------------------------------- the rules= exemption is visible
def scan_for_rules_argument(paths) -> tuple[int, list[str]]:
    """Return (files read, offending "path:line") for a `rules=` keyword outside tests/.

    The needle is assembled rather than written, because this file passes the argument
    many times and a scanner searching for its own literal would find itself. Spaces
    are stripped before matching, so `rules = [...]` cannot evade it; `self.rules=`
    and `==` are excluded by looking at the neighbouring character rather than by
    listing the phrasings I happen to remember.
    """
    kw = "rules" + "="
    read, hits = 0, []
    for path in sorted(paths):
        read += 1
        for n, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.replace(" ", "")
            i = line.find(kw)
            while i >= 0:
                before = line[i - 1] if i else ""
                after = line[i + len(kw):i + len(kw) + 1]
                if before not in (".", "_") and after != "=":
                    label = (path.relative_to(REPO_ROOT)
                             if path.is_relative_to(REPO_ROOT) else path.name)
                    hits.append(f"{label}:{n}")
                i = line.find(kw, i + len(kw))
    return read, hits


def production_python_files() -> list[Path]:
    skip = {"tests", ".push", ".git", "docs", "web"}
    return [p for p in REPO_ROOT.rglob("*.py")
            if not (set(p.relative_to(REPO_ROOT).parts) & skip)]


def test_nothing_outside_tests_narrows_the_rule_set():
    files = production_python_files()
    read, hits = scan_for_rules_argument(files)
    assert read >= 10, f"the scan read {read} files; §0 rule 5 makes that an error"
    assert hits == [], (
        f"{hits} passes rules= outside tests/: dropping a boundary rule in production "
        "is a behaviour change that must be reviewable, not an argument")


def test_the_scanner_finds_a_planted_call(tmp_path):
    """Positive control. A scanner that cannot find one is not evidence of none."""
    planted = tmp_path / "planted.py"
    planted.write_text("seg = Segmenter(rules=['silence'])\n")
    read, hits = scan_for_rules_argument([planted])
    assert read == 1 and len(hits) == 1
