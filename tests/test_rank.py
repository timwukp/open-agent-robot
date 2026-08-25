"""Offline tests for S6's three ablation arms — T-F6, and the honesty it turns on.

**Why almost every assertion here is about a tier rather than a number.** The plan asks
for three arms — ①衰減 ②重要性剪枝 ③不刪除只改排序 — and the interesting defects are all
in what an arm does with a hit it *cannot* judge. A missing `end_ts` defaulted to `now`
makes the undated episode the freshest thing in the store and sorts it first; defaulted
to 0 it makes it the oldest and, under decay, deletes it from the answer. An absent
salience scored as 0.0 prunes every unscored hit, and scored as 1.0 prunes nothing while
still appearing to read a threshold. Each of those is a named mutant below, and each one
produces a plausible-looking ranking — which is why the assertions are on tiers and
findings and not on "the order changed".

**Why the two `no_salience` mutants are asserted differently.** The `absent → 1.0`
mutant keeps exactly the hits the real module keeps: its order and its drops are
identical. A test comparing hit sets would call it dead and be wrong about which
quantity it measured, so the assertion is that the real module *reports* what it could
not judge and the mutant reports nothing. The other direction really does delete, so
that one is a set comparison.

**Why the ordering fixtures look over-specified.** The key is `weight / (1 + position)`
and the reciprocal-rank term dominates: an arm that reads the age correctly still leaves
a slightly-older hit on top when that hit is already first. So the two-event stores below
are built with the factors *far enough apart to disagree* (14 days against 0 at a 7-day
half-life), because a fixture where every arm agrees cannot tell a working arm from one
that ignores its own signal.

**Why the arm-collision check has a control.** Three arms with identical output are one
arm reported three times, and on a one-hit store that is the *expected* state — which
makes it useless as a gate unless something distinguishes "collided here" from "collided
everywhere". `verdict()` aggregates, so the tests carry both a store where the arms
differ and a store where they cannot, and assert two different exit codes.

**Why no test here asserts which arm is better.** That is T-F1's 30 frozen queries; they
need one day of real logs plus one human annotation pass, and neither may be fabricated.
`quality` is a status and the harness exits 2, and both of those are asserted — the
blocked state is a tested property rather than a note in a document.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory import consolidate as co  # noqa: E402
from memory import rank as rk  # noqa: E402
from memory import rank_harness as rh  # noqa: E402
from memory import recall as rc  # noqa: E402
from memory import salience as sa  # noqa: E402

RANK_PY = REPO_ROOT / "memory" / "rank.py"
HARNESS_PY = REPO_ROOT / "memory" / "rank_harness.py"

DAY = 24 * 60 * 60.0
NOW = 1000 * DAY          # an absolute clock reading, injected everywhere

# Episodes named for the state each one exists to produce, and minted through the one
# creator of that format so a renamed id scheme moves these with it.
EP_UNDATED = co.episode_id(1)    # no end_ts at all
EP_RECENT = co.episode_id(2)     # 2 days old
EP_OLD = co.episode_id(4)        # 400 days old, past the decay floor
EP_FUTURE = co.episode_id(6)     # end_ts after the injected clock
EP_GONE = co.episode_id(9)       # cited by the card, absent from the warm layer

# `salience.bigrams` is character bigrams with whitespace removed, so a query only
# reaches the card and the BM25 path if it shares a two-character run with them.
QTEXT = "花生小明"
ANSWER = "user_requested_answer"
OLD_FORMULA = "s3.4/text-3term"   # a formula id that is not the current one


# ------------------------------------------------------------------- fixtures
def _event(seq: int, text: str, *, salience: float | None = None,
           formula: str | None = None) -> dict:
    rec = {"seq": seq, "kind": "turn", "ts": float(seq), "text": text,
           "why": {"goal": ANSWER}}
    if salience is not None:
        rec["salience"] = salience
        rec["salience_formula"] = sa.FORMULA if formula is None else formula
    return rec


def episode(eid: str, seqs, summary: str, end_ts: float | None = None) -> dict:
    entry = {"id": eid, "seqs": list(seqs), "summary": summary}
    if end_ts is not None:
        entry["end_ts"] = end_ts
    return entry


def plant(root: Path, *, events, episodes, card=None, rps=None) -> Path:
    """Write a store. `card`/`rps` left as `None` means the artifact is absent, which is
    a different state from an empty one — the path reports `no_artifact`, not `ran`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / rc.EVENTS_ARTIFACT).write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
        encoding="utf-8")
    (root / rc.EPISODES_ARTIFACT).write_text(
        json.dumps({"version": 1, "episodes": episodes}, ensure_ascii=False),
        encoding="utf-8")
    if card is not None:
        (root / rc.CARD_ARTIFACT).write_text(
            json.dumps({"version": 1, "facts": card}, ensure_ascii=False),
            encoding="utf-8")
    if rps is not None:
        (root / rc.RPS_ARTIFACT).write_text(
            json.dumps({"version": 1, "rounds": rps}, ensure_ascii=False),
            encoding="utf-8")
    return root


def full(root: Path) -> Path:
    """One store reaching every freshness and importance status the disk can produce.

    Deliberately not minimal, and the seq order matters. The unjudgeable event is
    **seq 1**, first in the retriever's own order, because that is the only arrangement
    in which "treat a missing end_ts as now" is observable: the ordering key divides by
    position, so an unjudgeable hit that is already last stays last under either policy
    and the mutant would be output-identical.

    `test_the_fixture_reaches_every_status_the_disk_can_produce` asserts the coverage —
    a fixture that quietly stopped producing `other_formula` would leave that branch's
    assertions green and vacuous.
    """
    return plant(
        root,
        events=[
            _event(1, "我叫小明"),                                  # undated ep, no salience
            _event(2, "我對花生過敏", salience=0.70),               # dated, scored high
            _event(3, "今天有點累", salience=0.10),                 # below PRUNE_MIN
            _event(4, "我很久以前說過的話", salience=0.80),          # past the decay floor
            _event(5, "這句用舊公式算過", salience=0.90,
                   formula=OLD_FORMULA),                            # other_formula
            _event(6, "未來的一句", salience=0.60),                 # episode ends ahead
            _event(7, "沒有人認領這一句", salience=0.55),           # in no episode: orphan
        ],
        episodes=[episode(EP_UNDATED, [1], "自我介紹"),
                  episode(EP_RECENT, [2, 3], "過敏與疲累", NOW - 2 * DAY),
                  episode(EP_OLD, [4, 5], "舊事", NOW - 400 * DAY),
                  episode(EP_FUTURE, [6], "未來", NOW + 10 * DAY)],
        card=[{"kind": "name", "value": "小明", "trust": "seed",
               "source_episode_ids": []},
              {"kind": "allergy", "value": "花生", "trust": "observed",
               "source_episode_ids": [EP_GONE]}],
        rps=[{"episode": EP_RECENT, "user": "rock", "outcome": "win"}],
    )


def two_events(root: Path, *, first_salience: float, second_salience: float) -> Path:
    """Two dated hits whose recency and importance can be made to disagree.

    14 days is two half-lives (weight 0.25) and the second hit is brand new (1.0). The
    second is *second* in baseline order so its key is halved — 0.5 against 0.25, far
    enough apart that a recency-reading arm swaps them and a recency-ignoring arm does
    not. Both stay above `DECAY_FLOOR`, so nothing here is dropped.
    """
    return plant(root,
                 events=[_event(1, "舊的一句", salience=first_salience),
                         _event(2, "新的一句", salience=second_salience)],
                 episodes=[episode(EP_UNDATED, [1], "舊", NOW - 14 * DAY),
                           episode(EP_RECENT, [2], "新", NOW)])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return full(tmp_path / "oar")


def query(root: Path, text: str = QTEXT):
    """A `Result` over the store. `structured` returns every event for any query."""
    return rc.Retriever(root).query(text)


def ev_of(root: Path, text: str = QTEXT, *, now: float = NOW):
    result = query(root, text)
    return result, rk.evidence(result, root, now=now)


def structured(root: Path, arm: str) -> tuple[str, ...]:
    result, ev = ev_of(root)
    return rk.rank(arm, result, ev).order["structured"]


def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/rank.py. Zero replacements = the control.

    Keyed by label for the reason the other mutation tests give: two mutants sharing a
    directory let the first import win, which presents as a mutant that mysteriously
    survived.
    """
    dst = tmp_path / f"mut_{label}" / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    src = RANK_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in rank.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "rank.py"
    target.write_text(src, encoding="utf-8")
    name = f"rank_mutant_{tmp_path.name}_{label}"
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


def mutant_order(module, root: Path, arm: str, path: str = "structured"):
    result = query(root)
    return module.rank(arm, result, module.evidence(result, root, now=NOW)).order[path]


# ------------------------------------------------ mutant anchors, one per constraint
UNDATABLE_RETURN = "        return Freshness(UNDATABLE)\n"
PRUNE_GUARD = "            if imp.status != SCORED:\n"
JUDGED = "            judged += 1\n"
REORDER_RETURN = "    return Ranking(REORDER, order, (), findings)\n"
PER_PATH = "        out[path] = tuple(h.ref for h in result.hits(k=None) if h.path == path)\n"
OTHER_FORMULA_BRANCH = "    if stamped != baseline:\n"
QUALITY_FIELD = '        "quality": NO_LABELS,\n'
TIERED_TAIL = "    return tuple(scored) + tuple(r for r in refs if r not in keys)\n"
DECAY_KEY = "            keys[ref] = weight / (1.0 + i)\n"


# ============================================== the registry, derived on both sides
def test_arm_functions_and_the_registry_are_the_same_set() -> None:
    """T-F6's failure is 「A/B 只有 A」, and a three-name list cannot notice it.

    Both directions, because they catch different mistakes: an `_arm_` function with no
    registry entry is an arm the ablation never runs, and a registry entry with no
    function is a lookup error on the first comparison.
    """
    implemented = {n[len("_arm_"):] for n in vars(rk) if n.startswith("_arm_")}
    assert implemented == set(rk.ARMS), (
        f"registry {sorted(rk.ARMS)} vs implementations {sorted(implemented)}")


def test_the_registry_is_three_arms_because_round_five_asked_for_three() -> None:
    assert len(rk.ARMS) == 3 and rk.arms() == rk.ARMS
    assert len(set(rk.ARMS)) == 3, "no arm may appear twice"
    assert rk.policy_findings() == []


def test_an_unregistered_arm_cannot_be_run(root: Path) -> None:
    result, ev = ev_of(root)
    with pytest.raises(rk.RankError, match="unknown arm"):
        rk.rank("recency_squared", result, ev)
    with pytest.raises(rk.RankError, match="unknown arm"):
        rk.Ranking("recency_squared", {p: () for p in rc.paths()}, (), ())


def test_a_ranking_must_carry_every_path(root: Path) -> None:
    """A path missing from a ranking is one the comparison silently does not examine."""
    order = {p: () for p in rc.paths()}
    order.pop("rps")
    with pytest.raises(rk.RankError, match="every path exactly once"):
        rk.Ranking(rk.DECAY, order, (), ())


def test_the_scorable_paths_are_derived_from_what_each_path_reads() -> None:
    """Importance lives on event records, so a path can only have one if it reads them.

    Derived rather than listed: a sixth path reading `events.jsonl` has to be inside
    this set on the day it is written, and a written-down pair of names could notice
    neither that nor `structured` dropping events.
    """
    assert set(rk.scorable_paths()) == {
        n for n in rc.paths() if rc.EVENTS_ARTIFACT in rc.PATHS[n].reads}
    assert rk.scorable_paths() == ("structured", "lexical")


# ================================================================ policy numbers
def test_the_policy_numbers_are_labelled_policy_choices() -> None:
    """§6.0 rule 6, for numbers that cannot be verified at all.

    Nothing here measured a half-life and no external result names one, so what is
    checked is that the code says which *kind* of number it is — the same assertion
    `erase.RETENTION_DAYS` carries.
    """
    src = RANK_PY.read_text(encoding="utf-8")
    head = src.split("HALF_LIFE_DAYS")[0].rsplit("# ------", 1)[-1]
    assert "policy default" in head, (
        "the policy-numbers block lost the comment saying these are policy choices")
    assert rk.HALF_LIFE_S == rk.HALF_LIFE_DAYS * 24 * 60 * 60


def test_one_half_life_halves_the_weight_and_recency_is_monotone() -> None:
    assert rk.recency(0.0) == 1.0
    assert rk.recency(rk.HALF_LIFE_S) == pytest.approx(0.5)
    assert rk.recency(2 * rk.HALF_LIFE_S) == pytest.approx(0.25)
    weights = [rk.recency(i * DAY) for i in range(0, 30)]
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 < w <= 1.0 for w in weights)


def test_recency_refuses_a_negative_age_and_a_non_positive_half_life() -> None:
    """Both refusals, because both produce an arm that looks like it ran.

    A negative age scores above every real hit; a non-positive half-life makes every hit
    equally recent, so the decay arm becomes the baseline while still reporting itself.
    """
    with pytest.raises(rk.RankError, match="negative"):
        rk.recency(-1.0)
    with pytest.raises(rk.RankError, match="half_life_s must be positive"):
        rk.recency(1.0, half_life_s=0.0)


# ===================================================================== evidence
def test_the_fixture_reaches_every_status_the_disk_can_produce(root: Path) -> None:
    """Non-vacuity for everything below. A branch no fixture reaches is untested.

    `no_record` is deliberately absent here: it needs a hit whose ref names an event
    that is not on disk, which no `Retriever` can produce, so it is tested against a
    hand-built `Result` instead of being faked into the fixture.
    """
    _, ev = ev_of(root)
    census = ev.census()
    reached = {s for s, n in census["freshness"].items() if n} | {
        s for s, n in census["importance"].items() if n}
    expected = {rk.DATED, rk.UNDATABLE, rk.FUTURE_TS, rk.NO_EPISODE_BY_DESIGN,
                rk.LOST_ATTRIBUTION, rk.NO_ENTRY,
                rk.SCORED, rk.NO_SALIENCE, rk.OTHER_FORMULA, rk.UNSCORABLE}
    assert expected <= reached, f"the fixture no longer reaches {sorted(expected - reached)}"
    assert census["importance"][rk.NO_RECORD] == 0


def test_the_census_carries_every_status_including_the_zeros(root: Path) -> None:
    _, ev = ev_of(root)
    assert sorted(ev.census()["freshness"]) == sorted(rk.FRESHNESS_STATUSES)
    assert sorted(ev.census()["importance"]) == sorted(rk.IMPORTANCE_STATUSES)


@pytest.mark.parametrize("now", [None, "2026-08-24", True])
def test_evidence_refuses_an_absent_or_non_numeric_clock(root: Path, now) -> None:
    """§6.1: the module never reads a clock, so a caller that forgot one must not get
    numbers. `True` is in the grid because `isinstance(True, int)` is the check that
    would otherwise let a boolean through as a timestamp of 1."""
    result = query(root)
    with pytest.raises(rk.RankError, match="not a clock reading"):
        rk.evidence(result, root, now=now)


def test_rank_py_reads_no_clock_and_the_harness_does() -> None:
    """Both halves, because the one-sided version passes on a module that reads nothing.

    §6.1 requires the clock to be injected into the pure module. The harness is where a
    wall-clock reading is legitimate, and asserting that it *has* one is what makes the
    first assertion a statement about placement rather than about absence.
    """
    tree = ast.parse(RANK_PY.read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.Import) for a in n.names} | {
        n.module.split(".")[0] for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module}
    assert "time" not in imported, "memory/rank.py acquired a clock of its own"
    assert not {"rclpy", "boto3", "torch"} & imported, sorted(imported)
    assert "time.time()" in HARNESS_PY.read_text(encoding="utf-8"), (
        "the harness stopped reading a clock, so --now has nowhere to default from "
        "and the injection has no source")


def test_a_dated_hit_carries_its_age_and_an_undatable_one_carries_no_number(
        root: Path) -> None:
    _, ev = ev_of(root)
    fresh = ev.freshness_of(rc.event_ref(2))
    assert (fresh.status, fresh.age_s) == (rk.DATED, pytest.approx(2 * DAY))
    undated = ev.freshness_of(rc.event_ref(1))
    assert (undated.status, undated.age_s) == (rk.UNDATABLE, None)
    assert any("carries no end_ts" in f for f in ev.findings), ev.findings


def test_the_two_no_episode_cases_stay_apart(root: Path) -> None:
    """`SEED` and `ORPHAN` are two markers in `recall` for a reason, and merging them
    here would undo it: one is outside the cascade by construction, the other is a lost
    attribution, which is a defect."""
    _, ev = ev_of(root)
    assert ev.freshness_of(rc.card_ref(0)).status == rk.NO_EPISODE_BY_DESIGN
    assert ev.freshness_of(rc.event_ref(7)).status == rk.LOST_ATTRIBUTION
    assert any(rc.ORPHAN in f for f in ev.findings), ev.findings


def test_a_card_fact_citing_a_missing_episode_is_no_entry_not_undatable(
        root: Path) -> None:
    _, ev = ev_of(root)
    assert ev.freshness_of(rc.card_ref(1, EP_GONE)).status == rk.NO_ENTRY


def test_a_future_end_ts_is_its_own_status_rather_than_a_negative_age(
        root: Path) -> None:
    """A clock disagreement after a restart is a real state, and a negative age would
    score above every genuine hit instead of being reported."""
    _, ev = ev_of(root)
    assert ev.freshness_of(rc.event_ref(6)).status == rk.FUTURE_TS
    assert any("after the injected clock" in f for f in ev.findings), ev.findings


def test_an_unscored_event_and_an_unscorable_path_are_different_statuses(
        root: Path) -> None:
    _, ev = ev_of(root)
    assert ev.importance_of(rc.event_ref(1)).status == rk.NO_SALIENCE
    assert ev.importance_of(rc.card_ref(0)).status == rk.UNSCORABLE
    assert ev.importance_of(rc.event_ref(2)).value == pytest.approx(0.70)


def test_a_score_from_another_formula_is_not_a_smaller_number(root: Path) -> None:
    """`memory/salience.py` states the obligation this inherits: two formula ids are two
    quantities, and comparing them is a thing S6 has to *notice*."""
    _, ev = ev_of(root)
    imp = ev.importance_of(rc.event_ref(5))
    assert imp.status == rk.OTHER_FORMULA
    assert imp.value is None, "an incomparable score must not be handed to an arm"
    assert imp.formula == OLD_FORMULA
    assert any(OLD_FORMULA in f for f in ev.findings), ev.findings


def test_a_hit_with_no_event_record_is_reported(root: Path) -> None:
    """Unreachable from a `Retriever`, so it is built by hand rather than faked on disk."""
    hits = [rc.Hit("lexical", EP_RECENT, "沒有這筆", 1.0, rc.event_ref(999))]
    results = [rc.PathResult(n, rc.RAN, hits if n == "lexical" else ())
               for n in rc.paths()]
    ev = rk.evidence(rc.Result(QTEXT, results), root, now=NOW)
    assert ev.importance_of(rc.event_ref(999)).status == rk.NO_RECORD
    assert any("no event record on disk" in f for f in ev.findings), ev.findings


def test_a_freshness_or_importance_cannot_carry_a_value_it_denies_having() -> None:
    """The type refuses the two shapes that would let an absence be compared."""
    with pytest.raises(rk.RankError, match="the two states this type exists"):
        rk.Freshness(rk.UNDATABLE, 0.0)
    with pytest.raises(rk.RankError, match="the two states this type exists"):
        rk.Freshness(rk.DATED)
    with pytest.raises(rk.RankError, match="an absent measurement is not"):
        rk.Importance(rk.NO_SALIENCE, 0.0)
    with pytest.raises(rk.RankError, match="an absent measurement is not"):
        rk.Importance(rk.SCORED)
    with pytest.raises(rk.RankError, match="outside"):
        rk.Importance(rk.SCORED, 1.5)
    with pytest.raises(rk.RankError, match="unknown freshness status"):
        rk.Freshness("quite_old")
    with pytest.raises(rk.RankError, match="unknown importance status"):
        rk.Importance("quite_important")


def test_an_arm_cannot_rank_a_hit_the_evidence_was_not_built_from(root: Path) -> None:
    """Two objects describing different results is the failure this refuses; a lookup
    error deep in an arm would present as a crash in the ranking arithmetic."""
    _, ev = ev_of(root)
    with pytest.raises(rk.RankError, match="was not built from"):
        ev.freshness_of(rc.event_ref(4242))


# ========================================================================= arms
def test_decay_suppresses_what_is_past_the_floor_and_keeps_what_is_not(
        root: Path) -> None:
    result, ev = ev_of(root)
    ranking = rk.rank(rk.DECAY, result, ev)
    assert rc.event_ref(4) in ranking.dropped, "a 400-day-old hit survived the floor"
    assert rc.event_ref(2) not in ranking.dropped
    assert rk.kept_findings(result, ranking) == []


def test_decay_leaves_the_undatable_hit_in_the_tail_and_says_so(root: Path) -> None:
    """The headline constraint: a missing `end_ts` must not become a timestamp.

    Asserted from both ends. It is not first (which is what `now` would make it) and it
    is not dropped (which is what 0 would make it under the floor), and the arm reports
    how many hits it could not judge and why.
    """
    result, ev = ev_of(root)
    ranking = rk.rank(rk.DECAY, result, ev)
    order = ranking.order["structured"]
    undated = rc.event_ref(1)
    assert undated in order and undated not in ranking.dropped
    assert order.index(undated) > 0, "the undated episode ranked first"
    dated = [r for r in order if ev.freshness_of(r).status == rk.DATED]
    assert dated, "the fixture has no dated hit, so the tail assertion is vacuous"
    assert order.index(undated) > max(order.index(r) for r in dated), \
        "an unjudgeable hit outranked every hit the arm could judge"
    assert any(f"{rk.UNDATABLE}=1" in f for f in ranking.findings), ranking.findings


def test_decay_ranks_a_recent_hit_above_an_older_one(tmp_path: Path) -> None:
    """The arm has to actually use the age, and a two-hit store is where that shows.

    Both hits are `structured`, whose scores are all 1.0, so the baseline order is the
    event order and the recency factor is the only thing that can reorder them.
    """
    root = two_events(tmp_path / "oar", first_salience=0.5, second_salience=0.5)
    result, _ = ev_of(root)
    baseline = tuple(h.ref for h in result.hits(k=None) if h.path == "structured")
    assert baseline == (rc.event_ref(1), rc.event_ref(2)), baseline
    assert structured(root, rk.DECAY) == (rc.event_ref(2), rc.event_ref(1))


def test_prune_removes_the_low_salience_hit_and_keeps_the_unjudgeable_ones(
        root: Path) -> None:
    result, ev = ev_of(root)
    ranking = rk.rank(rk.PRUNE, result, ev)
    assert rc.event_ref(3) in ranking.dropped, "salience 0.10 survived PRUNE_MIN 0.30"
    assert rc.event_ref(2) not in ranking.dropped
    for ref in (rc.event_ref(1), rc.event_ref(5), rc.card_ref(0)):
        assert ref not in ranking.dropped, f"{ref} was pruned on an absent measurement"
    assert any("kept unjudged" in f for f in ranking.findings), ranking.findings
    assert rk.kept_findings(result, ranking) == []


def test_prune_leaves_the_order_alone(root: Path) -> None:
    """So that a difference against `reorder` is attributable to the removal.

    An arm that pruned *and* reordered would make the ablation unable to say which of
    the two changes produced the difference, which is the question ③ exists to answer.
    """
    result, ev = ev_of(root)
    ranking = rk.rank(rk.PRUNE, result, ev)
    for path in rc.paths():
        baseline = [h.ref for h in result.hits(k=None) if h.path == path]
        assert list(ranking.order[path]) == [
            r for r in baseline if r not in ranking.dropped]


def test_prune_reports_the_paths_it_cannot_reach_at_all(root: Path) -> None:
    """The honest limit, stated by the arm rather than left for a reader to infer.

    Importance lives on event records, so `card` and `rps` are untouched by this arm no
    matter what the threshold is. An unchanged answer on those paths would otherwise
    read as "nothing there was unimportant".
    """
    result, ev = ev_of(root)
    ranking = rk.rank(rk.PRUNE, result, ev)
    hit_paths = {h.path for h in result.hits(k=None)}
    unreachable = hit_paths - set(rk.scorable_paths())
    assert unreachable, "no unscorable path has a hit, so this assertion is vacuous"
    for path in sorted(unreachable):
        assert any(f"{rk.PRUNE}/{path}" in f and "0 of" in f
                   for f in ranking.findings), (path, ranking.findings)


def test_reorder_drops_nothing_and_is_a_permutation_of_the_result(root: Path) -> None:
    result, ev = ev_of(root)
    ranking = rk.rank(rk.REORDER, result, ev)
    assert ranking.dropped == ()
    assert rk.kept_findings(result, ranking) == []
    for path in rc.paths():
        baseline = [h.ref for h in result.hits(k=None) if h.path == path]
        assert sorted(ranking.order[path]) == sorted(baseline)


def test_reorder_uses_both_recency_and_importance(tmp_path: Path) -> None:
    """③ carries the combined signal, so a store where the two factors disagree is what
    shows it reads both. Recency alone swaps these two; recency × importance does not,
    and the second assertion is what makes the first about *both* factors."""
    root = two_events(tmp_path / "oar", first_salience=0.9, second_salience=0.4)
    assert structured(root, rk.REORDER) == (rc.event_ref(1), rc.event_ref(2))
    assert structured(root, rk.DECAY) == (rc.event_ref(2), rc.event_ref(1))


def test_no_arm_fuses_the_paths(root: Path) -> None:
    """BM25 is unbounded and cosine is at most 1.0, so one list ordered by score is
    ordered by a quantity that does not exist. Every arm keeps the paths apart."""
    result, ev = ev_of(root)
    for name in rk.arms():
        ranking = rk.rank(name, result, ev)
        assert sorted(ranking.order) == sorted(rc.paths())
        for path, refs in ranking.order.items():
            from_here = {h.ref for h in result.hits(k=None) if h.path == path}
            assert set(refs) <= from_here, f"{name}/{path} holds another path's hits"


def test_an_arm_never_edits_the_result_it_was_given(root: Path) -> None:
    """`Hit` is immutable, so this is enforced; the assertion is that no arm tries."""
    result, ev = ev_of(root)
    before = json.dumps(result.as_dict(), sort_keys=True, ensure_ascii=False)
    for name in rk.arms():
        rk.rank(name, result, ev)
    assert json.dumps(result.as_dict(), sort_keys=True, ensure_ascii=False) == before


def test_the_arms_are_deterministic(root: Path) -> None:
    result, ev = ev_of(root)
    for name in rk.arms():
        first = json.dumps(rk.rank(name, result, ev).as_dict(), sort_keys=True)
        second = json.dumps(rk.rank(name, result, ev).as_dict(), sort_keys=True)
        assert first == second


def test_kept_findings_catches_a_hit_that_vanished_without_being_dropped(
        root: Path) -> None:
    """The positive control for the invariant `kept_findings` exists to check.

    A hand-built `Ranking` that lost a ref, because the real arms do not lose one — and
    a check that only ever sees clean input is not evidence that it works.
    """
    result, _ = ev_of(root)
    order = {p: tuple(h.ref for h in result.hits(k=None) if h.path == p)
             for p in rc.paths()}
    order["structured"] = order["structured"][1:]
    findings = rk.kept_findings(result, rk.Ranking(rk.DECAY, order, (), ()))
    assert any("vanished without being dropped" in f for f in findings), findings
    order["structured"] = order["structured"] + ("events.jsonl#4242",)
    findings = rk.kept_findings(result, rk.Ranking(rk.DECAY, order, (), ()))
    assert any("is not in the result" in f for f in findings), findings
    stray = rk.kept_findings(result, rk.Ranking(
        rk.DECAY, order, ("events.jsonl#7777",), ()))
    assert any("which was never a hit" in f for f in stray), stray


# =================================================================== the report
def test_the_report_carries_all_three_arms_and_no_quality_number(root: Path) -> None:
    result, ev = ev_of(root)
    report = rk.ablation(result, ev)
    assert sorted(report["arms"]) == sorted(rk.arms())
    assert report["quality"] == rk.NO_LABELS
    assert isinstance(report["quality"], str), "a quality number appeared"
    json.dumps(report, ensure_ascii=False)      # the evidence file is JSON (§6.0 rule 6)


def test_labelled_scoring_refuses_rather_than_estimating(root: Path) -> None:
    """The one place a fabricated number would be easiest and most damaging."""
    result, ev = ev_of(root)
    with pytest.raises(rk.RankError, match="must not be faked"):
        rk.ablation(result, ev, labels={"q01": [EP_RECENT]})


def test_a_missing_arm_is_a_finding(root: Path) -> None:
    result, ev = ev_of(root)
    report = rk.ablation(result, ev)
    assert rk.ablation_findings(report) == [], rk.ablation_findings(report)
    report["arms"].pop(rk.PRUNE)
    findings = rk.ablation_findings(report)
    assert any("did not run" in f and rk.PRUNE in f for f in findings), findings


def test_an_unregistered_arm_in_a_report_is_a_finding(root: Path) -> None:
    result, ev = ev_of(root)
    report = rk.ablation(result, ev)
    report["arms"]["recency_squared"] = report["arms"][rk.DECAY]
    findings = rk.ablation_findings(report)
    assert any("not in the registry" in f for f in findings), findings


def test_a_quality_number_in_a_report_is_a_finding(root: Path) -> None:
    result, ev = ev_of(root)
    report = rk.ablation(result, ev)
    report["quality"] = 0.62
    findings = rk.ablation_findings(report)
    assert any("the only honest value" in f for f in findings), findings


def test_reorder_dropping_something_is_a_finding(root: Path) -> None:
    result, ev = ev_of(root)
    report = rk.ablation(result, ev)
    report["arms"][rk.REORDER]["dropped"] = [rc.event_ref(2)]
    findings = rk.ablation_findings(report)
    assert any("cannot remove anything" in f for f in findings), findings


def test_identical_arms_are_data_and_the_control_is_a_store_where_they_differ(
        tmp_path: Path) -> None:
    """Both directions, because the collision check is only useful if it can be absent.

    A one-hit store cannot distinguish three policies — one hit, dated, scored above the
    threshold, so every arm returns it unchanged. The fixture store can. Asserting only
    the first would leave a detector that reports a collision unconditionally looking
    correct.
    """
    lonely = plant(tmp_path / "one",
                   events=[_event(1, "只有一句", salience=0.9)],
                   episodes=[episode(EP_UNDATED, [1], "一", NOW - 1 * DAY)])
    result, ev = ev_of(lonely, "只有")
    report = rk.ablation(result, ev)
    assert report["identical_arms"] == [sorted(rk.arms())], report["identical_arms"]
    assert any("identical orders" in f for f in rk.ablation_findings(report))
    assert rk.ablation_findings(report, include_identical=False) == []

    result, ev = ev_of(full(tmp_path / "many"))
    assert rk.ablation(result, ev)["identical_arms"] == [], "the arms stopped differing"


def test_the_conclusion_is_blocked_and_says_which_data_is_missing(root: Path) -> None:
    result, ev = ev_of(root)
    blockers = rk.conclusion_blockers(rk.ablation(result, ev))
    assert len(blockers) == 1
    assert "T-F1" in blockers[0] and "annotation" in blockers[0]
    assert "fabricated" in blockers[0]


# ====================================================================== harness
def test_the_query_format_needs_an_id_and_refuses_a_duplicate() -> None:
    pairs, findings = rh.parse_queries(
        "# 標題\n\n> 註解\n\n- `q01` 我對什麼過敏\n* `q02` 我叫什麼\nplain prose\n")
    assert pairs == [("q01", "我對什麼過敏"), ("q02", "我叫什麼")]
    assert findings == []
    _, dupes = rh.parse_queries("- `q01` 甲\n- `q01` 乙\n")
    assert any("duplicate query id" in f for f in dupes), dupes


def _run_harness(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HARNESS_PY)] + args,
                          cwd=str(REPO_ROOT), capture_output=True, text=True)


def _queries(path: Path, *lines: str) -> Path:
    path.write_text("".join(f"- `q{n:02d}` {t}\n" for n, t in enumerate(lines, 1)),
                    encoding="utf-8")
    return path


def test_the_harness_cannot_run_without_a_store_or_a_query_file(tmp_path: Path) -> None:
    """Exit 3, the code this repo reserves for a guard that could not run.

    The missing query file is the loud record that S6's conclusion is blocked on data
    collection: a default query set written today would measure its own author.
    """
    root = full(tmp_path / "oar")
    missing = _run_harness(["--root", str(tmp_path / "nope")])
    assert missing.returncode == 3 and "CANNOT RUN" in missing.stderr
    no_queries = _run_harness(["--root", str(root),
                               "--queries", str(tmp_path / "none.md")])
    assert no_queries.returncode == 3
    # The refusal has to name the test the data is owed to and say why a default set is
    # not a substitute; a bare "file not found" reads as a misconfiguration.
    assert "T-F1" in no_queries.stderr
    assert "human annotation pass" in no_queries.stderr
    assert "measure its own author" in no_queries.stderr


def test_the_harness_refuses_a_query_file_that_lists_nothing(tmp_path: Path) -> None:
    """§6.0 rule 5: a run over zero queries must not report three agreeing arms."""
    root = full(tmp_path / "oar")
    empty = tmp_path / "empty.md"
    empty.write_text("# 還沒有凍結任何查詢\n", encoding="utf-8")
    proc = _run_harness(["--root", str(root), "--queries", str(empty)])
    assert proc.returncode == 3 and "0 queries" in proc.stderr


def test_the_harness_refuses_a_run_that_returned_no_hits(tmp_path: Path) -> None:
    """Three arms over an empty answer are identical for a reason that is not policy."""
    root = plant(tmp_path / "oar", events=[], episodes=[])
    proc = _run_harness(["--root", str(root),
                         "--queries", str(_queries(tmp_path / "q.md", "完全不會命中")),
                         "--now", str(NOW), "--out", str(tmp_path / "o.json")])
    assert proc.returncode == 3 and "0 hits" in proc.stderr


def test_the_harness_exits_two_with_three_arms_and_no_labels(tmp_path: Path) -> None:
    """The expected state today, and it is a tested property rather than a note.

    Exit 2 means the comparison is well-formed and still cannot conclude anything. The
    JSON is written in that state on purpose: the evidence of a blocked comparison is
    itself evidence.
    """
    root = full(tmp_path / "oar")
    out = tmp_path / "rank-ablation.json"
    proc = _run_harness(["--root", str(root),
                         "--queries", str(_queries(tmp_path / "q.md", QTEXT, "花生")),
                         "--now", str(NOW), "--out", str(out)])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "blocked" in proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["queries"] == 2 and data["arms"] == list(rk.arms())
    assert data["now"] == NOW, "the clock the run used is not recorded"
    assert data["root_hits"] > 0
    for report in data["reports"].values():
        assert report["structural_findings"] == []
        assert report["quality"] == rk.NO_LABELS
        assert report["blockers"], "a run with no labels must carry its blocker"
        for arm in report["arms"].values():
            assert arm["kept_findings"] == []


def test_the_harness_exits_one_when_the_arms_cannot_be_distinguished(
        tmp_path: Path) -> None:
    """A collision on *every* query is one policy under three names — T-F6's 「A/B 只有
    A」 wearing a full report. On one query out of several it is a note, and the two
    stores here are what make that distinction tested rather than claimed."""
    lonely = plant(tmp_path / "one",
                   events=[_event(1, "只有一句", salience=0.9)],
                   episodes=[episode(EP_UNDATED, [1], "一", NOW - 1 * DAY)])
    proc = _run_harness(["--root", str(lonely),
                         "--queries", str(_queries(tmp_path / "q.md", "只有")),
                         "--now", str(NOW), "--out", str(tmp_path / "o.json")])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "same policy under two names" in proc.stdout


def test_the_harness_is_reproducible_from_its_recorded_clock(tmp_path: Path) -> None:
    """Decay is a function of elapsed time, so a run with no recorded clock is not
    replayable — the same store gives a different order tomorrow."""
    root = full(tmp_path / "oar")
    queries = _queries(tmp_path / "q.md", QTEXT)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        proc = _run_harness(["--root", str(root), "--queries", str(queries),
                             "--now", str(NOW), "--out", str(out)])
        assert proc.returncode == 2, proc.stdout + proc.stderr
    assert first.read_bytes() == second.read_bytes()


# ======================================================================= mutants
@pytest.mark.mutation
def test_control_an_unmutated_copy_ranks_exactly_as_the_real_module_does(
        tmp_path: Path) -> None:
    """§6.0 rule 2's other half: a harness that reports success on a broken control."""
    module = load_mutant(tmp_path, "control")
    root = full(tmp_path / "oar")
    result = query(root)
    mine = module.ablation(result, module.evidence(result, root, now=NOW))
    real = rk.ablation(result, rk.evidence(result, root, now=NOW))
    assert json.dumps(mine, sort_keys=True, ensure_ascii=False) == \
        json.dumps(real, sort_keys=True, ensure_ascii=False)


@pytest.mark.mutation
def test_mutant_a_missing_end_ts_treated_as_now_ranks_the_undated_episode_first(
        tmp_path: Path) -> None:
    """The headline mutant. An absent timestamp defaulted to the present moment makes
    the one episode nobody dated the freshest thing in the store."""
    module = load_mutant(tmp_path, "undated_is_now",
                         (UNDATABLE_RETURN, "        return Freshness(DATED, 0.0)\n"))
    root = full(tmp_path / "oar")
    undated = rc.event_ref(1)
    assert mutant_order(module, root, module.DECAY).index(undated) == 0, (
        "the mutant did not rank the undated episode first, so this test is not "
        "measuring the default it exists to refuse")
    assert structured(root, rk.DECAY).index(undated) > 0


@pytest.mark.mutation
def test_mutant_an_absent_salience_scored_as_zero_prunes_everything_unscored(
        tmp_path: Path) -> None:
    module = load_mutant(
        tmp_path, "absent_is_zero",
        (PRUNE_GUARD, "            if False:\n"),
        (JUDGED, JUDGED + "            imp = imp if imp.value is not None else "
                          "Importance(SCORED, 0.0)\n"))
    root = full(tmp_path / "oar")
    result = query(root)
    dropped = module.rank(module.PRUNE, result,
                          module.evidence(result, root, now=NOW)).dropped
    assert rc.event_ref(1) in dropped, "the mutant kept the unscored hit"
    real = rk.rank(rk.PRUNE, result, rk.evidence(result, root, now=NOW))
    assert rc.event_ref(1) not in real.dropped


@pytest.mark.mutation
def test_mutant_an_absent_salience_scored_as_maximal_is_output_identical_and_silent(
        tmp_path: Path) -> None:
    """A mutant whose order and drops match the real module's exactly, so the assertion
    has to be on the report. Treating an absent measurement as 1.0 prunes nothing — the
    same hits, with nothing anywhere saying a threshold was never applied."""
    module = load_mutant(
        tmp_path, "absent_is_one",
        (PRUNE_GUARD, "            if False and imp.status:\n"),
        (JUDGED, JUDGED + "            imp = imp if imp.value is not None else "
                          "Importance(SCORED, 1.0)\n"))
    root = full(tmp_path / "oar")
    result = query(root)
    mutant = module.rank(module.PRUNE, result,
                         module.evidence(result, root, now=NOW))
    real = rk.rank(rk.PRUNE, result, rk.evidence(result, root, now=NOW))
    assert mutant.order == real.order and mutant.dropped == real.dropped, (
        "this mutant is supposed to be output-identical; if it is not, the assertion "
        "below is testing something easier than it claims")
    assert not any("kept unjudged" in f for f in mutant.findings)
    assert any("kept unjudged" in f for f in real.findings), real.findings


@pytest.mark.mutation
def test_mutant_reorder_that_drops_is_caught(tmp_path: Path) -> None:
    """③'s whole job is to isolate ordering from removal, so an arm that removes anything
    makes the ablation unable to answer the question it was built for."""
    module = load_mutant(
        tmp_path, "reorder_drops",
        (REORDER_RETURN, "    return Ranking(REORDER, order, tuple(\n"
                         "        [r for refs in order.values() for r in refs][:1]),\n"
                         "        findings)\n"))
    root = full(tmp_path / "oar")
    result = query(root)
    ev = module.evidence(result, root, now=NOW)
    assert module.rank(module.REORDER, result, ev).dropped, "the mutant dropped nothing"
    report = module.ablation(result, ev)
    assert any("cannot remove anything" in f
               for f in module.ablation_findings(report))


@pytest.mark.mutation
def test_mutant_a_fused_baseline_returns_another_paths_hits(tmp_path: Path) -> None:
    """Every path's hits in one bucket: the ordering-by-name-dressed-as-relevance that
    `Result.hits(k)` applies `k` per path to prevent."""
    module = load_mutant(
        tmp_path, "fused",
        (PER_PATH, "        out[path] = tuple(h.ref for h in result.hits(k=None))\n"))
    root = full(tmp_path / "oar")
    result = query(root)
    from_card = {h.ref for h in result.hits(k=None) if h.path == "card"}
    assert from_card, "the fixture produced no card hits, so this mutant is unobservable"
    assert from_card <= set(mutant_order(module, root, module.REORDER)), (
        "the mutant did not fuse the paths")
    assert not from_card & set(structured(root, rk.REORDER))


@pytest.mark.mutation
def test_mutant_comparing_across_formula_ids_is_caught(tmp_path: Path) -> None:
    """`memory/salience.py`'s obligation, as a mutant: a score from another formula
    silently compared as though the two numbers were the same quantity."""
    module = load_mutant(tmp_path, "one_formula",
                         (OTHER_FORMULA_BRANCH, "    if False:\n"))
    root = full(tmp_path / "oar")
    result = query(root)
    mutant_ev = module.evidence(result, root, now=NOW)
    assert mutant_ev.importance_of(rc.event_ref(5)).status == module.SCORED, (
        "the mutant did not compare across formulas")
    assert not any(OLD_FORMULA in f for f in mutant_ev.findings)
    real_ev = rk.evidence(result, root, now=NOW)
    assert real_ev.importance_of(rc.event_ref(5)).status == rk.OTHER_FORMULA


@pytest.mark.mutation
def test_mutant_a_quality_number_without_labels_is_caught(tmp_path: Path) -> None:
    """The one number that would be easiest to print and hardest to notice: a value
    computed with no labels behind it, which reads as 「三組一樣好」."""
    module = load_mutant(tmp_path, "quality_number",
                         (QUALITY_FIELD, '        "quality": 0.5,\n'))
    root = full(tmp_path / "oar")
    result = query(root)
    report = module.ablation(result, module.evidence(result, root, now=NOW))
    assert report["quality"] == 0.5
    assert any("the only honest value" in f
               for f in module.ablation_findings(report))
    assert module.conclusion_blockers(report) == [], (
        "the mutant's report claims a quality, so nothing blocks its conclusion — "
        "which is exactly why the finding above has to exist")


@pytest.mark.mutation
def test_mutant_an_unjudgeable_hit_promoted_to_the_front_is_caught(
        tmp_path: Path) -> None:
    """The tail is the whole mechanism: put the unjudgeable tier first and every hit
    nobody could measure outranks every hit somebody did."""
    module = load_mutant(
        tmp_path, "tier_first",
        (TIERED_TAIL, "    return tuple(r for r in refs if r not in keys) + "
                      "tuple(scored)\n"))
    root = full(tmp_path / "oar")
    undated = rc.event_ref(1)
    assert mutant_order(module, root, module.DECAY).index(undated) == 0, (
        "the mutant did not promote the tail")
    assert structured(root, rk.DECAY).index(undated) > 0


@pytest.mark.mutation
def test_mutant_a_decay_arm_that_ignores_the_age_is_caught(tmp_path: Path) -> None:
    """An arm that keeps the baseline order while reporting itself as an arm. This is
    「A/B 只有 A」 in the shape a missing-arm check cannot see, because the arm ran."""
    module = load_mutant(tmp_path, "no_decay",
                         (DECAY_KEY, "            keys[ref] = 1.0 / (1.0 + i)\n"))
    root = two_events(tmp_path / "oar", first_salience=0.5, second_salience=0.5)
    assert mutant_order(module, root, module.DECAY) == (
        rc.event_ref(1), rc.event_ref(2)), "the mutant still reordered"
    assert structured(root, rk.DECAY) == (rc.event_ref(2), rc.event_ref(1))
