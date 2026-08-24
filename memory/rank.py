"""memory/rank.py — S6's three ablation arms, as policies over a finished `Result`.

Zero ROS, zero boto3, zero network, zero LLM, no clock of its own and no disk beyond
the artifacts `memory.recall` already reads. Every arm is a pure function of a
`Result` plus an `Evidence` bundle, so the whole of T-F6 runs under pytest on a laptop.

**Why S6 is not "A/B the two scorers we already have".**
The plan reads 「扁平+時間評分 A/B Graphiti」, which sounds like turning on a temporal
term that exists. It does not exist: `grep -n "decay\\|recency\\|temporal\\|half_life"`
over `memory/recall.py` matches nothing, and the only match in the whole tree is one
line of `docs/test-design.md` recording that no published work has done the
decay-vs-no-decay ablation at all. So the three arms 第五輪 asked for — ①衰減
②重要性剪枝 ③不刪除只改排序 — are three things that have to be *written* before
anything can be compared, and this module is them.

**Why an arm is a policy over a `Result` and never an edit to `Hit.score`.**
`Result.hits(k)` applies `k` per path on purpose, because BM25 is unbounded and cosine
is at most 1.0: a single list ordered by score is ordered by a quantity that does not
exist. An arm that rewrote `Hit.score` would fuse those two scales while looking like
a ranking change, and every consumer downstream would inherit the fusion silently. So
an arm returns a `Ranking` — refs, grouped by path, in the arm's order — and the
`Result` it was given is untouched. `Hit` is immutable, so this is enforced rather
than intended.

**Why the ordering key is the reciprocal *rank* and not the score.**
Within one path, a hit's position is already a scale-free relevance statement: it is
the only comparable quantity the retriever produces. `1 / (1 + position)` therefore
carries the relevance half of every arm's key, and the temporal and importance factors
multiply into it. This is the one construction that lets a temporal term exist without
first measuring a BM25-to-cosine calibration nobody here has measured.

**Why absent data is a tier and never a default.**
`parse_episodes` requires only `id`, `seqs` and `summary`, so `end_ts` may legitimately
be missing (`erase.datable` exists because of exactly that). Defaulting a missing
timestamp to `now` makes the undated episode the *freshest* one and sorts it first;
defaulting it to 0 makes it the oldest and, under `decay`, deletes it from the answer.
Importance has the same shape from the other side: salience lives on events
(`store.RECORD_FIELD_DEPS`), so a card, episode or rps hit has no salience at all —
scoring that absence as 0 prunes four fifths of the answer, and scoring it as 1.0
prunes nothing while still appearing to read a threshold.

So an arm splits each path into a **comparable tier** it may reorder or drop and a
**non-comparable tier** it leaves in baseline order and reports. No arm ever invents a
number for a missing measurement, and `Ranking.findings` says how many hits each arm
could not judge and why. The honest cost is stated rather than implied: importance can
only reach the two event paths, so `prune`'s effect size is bounded by construction.

**Why a formula id mismatch is its own status.**
`salience.py` states the obligation this module inherits: "a formula changed in S6
silently makes every earlier record incomparable with every later one while every gate
stays green… Comparing across formula ids is then a thing S6 has to *notice*." A
record stamped by a different formula is not a smaller number, it is a different
quantity, so `OTHER_FORMULA` is separate from `NO_SALIENCE` — one needs a re-score and
the other needs a decision about which formula is canonical.

**Why the quality field is a status and not a number.**
T-F6's criterion is 「三組數字齊備才允許下結論」, and the numbers that would decide
which arm is better are T-F1's 30 frozen queries with human relevance labels. They do
not exist yet: they need one day of real logs and one annotation pass, and fabricating
either would be the one failure this whole design is built to refuse. So `ablation()`
reports `quality: "no_labels"` and `ablation_findings()` refuses the conclusion.
Printing a number computed without labels would state "the two are equally good" in the
one format nobody re-reads.

Honest boundaries, so they are not implied:
* The half-life and both thresholds are **policy defaults with no measurement behind
  them**. Nothing here justifies them and T-F1 is what would.
* Decay is retrieval suppression, never erasure (plan §1). A hit `decay` drops is
  still on disk and `memory.erase` is the only thing that removes it.
* This module compares *mechanical* properties of three orderings. It cannot say which
  ordering a person would prefer, and it does not claim to.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memory import recall
from memory.erase import datable
from memory.salience import formula_id
from memory.store import StoreError


class RankError(StoreError):
    """An arm that does not exist, an absent clock, or a ranking that lost a hit."""


# ------------------------------------------------------------------------- arms
#
# The three arms 第五輪 required. Declared here and asserted set-equal to the
# `_arm_<name>` functions in this module, in both directions, for the reason
# `recall.PATHS` gives: an arm implemented but unregistered is one the ablation never
# runs, and an arm registered but unimplemented is an `AttributeError` on the first
# comparison. T-F6's failure mode is 「A/B 只有 A」, and a three-name list is precisely
# what cannot notice that only two of them ran.
DECAY = "decay"
PRUNE = "prune"
REORDER = "reorder"
ARMS = (DECAY, PRUNE, REORDER)

# ----------------------------------------------------------------- policy numbers
#
# Every number below is a **policy default**: no measurement in this repo produced it,
# no external result names it, and T-F1's labelled queries are what would justify one.
# They are named constants so that changing one is a diff a reviewer can see, and
# `tests/test_rank.py` asserts this comment survives — a number that stops saying which
# kind of number it is has quietly become a finding.
#
# 7 days: long enough that a week-old preference still ranks, short enough that the
# arm does something measurable inside a single evaluation window.
HALF_LIFE_DAYS = 7.0
HALF_LIFE_S = HALF_LIFE_DAYS * 24 * 60 * 60

# Three half-lives. `decay` drops a hit whose recency weight falls below this; at the
# default half-life that is 21 days.
DECAY_FLOOR = 0.125

# A first, unique utterance scores 0.3 from the novelty term alone (`salience.WEIGHTS`),
# so this threshold keeps anything novel and prunes restated filler. That is a
# description of what the number does, not evidence that it is the right one.
PRUNE_MIN = 0.30

# ---------------------------------------------------------------- freshness status
#
# Six statuses rather than "has a timestamp or not", because the five non-`DATED` cases
# have five different fixes and merging any two of them hides one behind the other.
DATED = "dated"                              # end_ts present and not in the future
UNDATABLE = "undatable"                      # the warm-layer entry carries no end_ts
FUTURE_TS = "future_ts"                      # end_ts is after `now`: clock disagreement
NO_EPISODE_BY_DESIGN = "no_episode_by_design"  # a seed fact; outside the cascade
LOST_ATTRIBUTION = "lost_attribution"        # recall.ORPHAN: this one *is* a defect
NO_ENTRY = "no_entry"                        # names an episode the warm layer lacks
FRESHNESS_STATUSES = (DATED, UNDATABLE, FUTURE_TS, NO_EPISODE_BY_DESIGN,
                      LOST_ATTRIBUTION, NO_ENTRY)

# --------------------------------------------------------------- importance status
SCORED = "scored"                # an event record carrying salience and its formula
NO_SALIENCE = "no_salience"      # an event record written without one
OTHER_FORMULA = "other_formula"  # scored by a different formula: a different quantity
UNSCORABLE = "unscorable"        # this path does not read events, so it has no salience
NO_RECORD = "no_record"          # the ref names an event that is not on disk
IMPORTANCE_STATUSES = (SCORED, NO_SALIENCE, OTHER_FORMULA, UNSCORABLE, NO_RECORD)

# The quality field's only honest value until T-F1's labels exist.
NO_LABELS = "no_labels"


def arms() -> tuple[str, ...]:
    """The ablation arms, in declaration order. The registry `ablation()` iterates."""
    return ARMS


def scorable_paths() -> tuple[str, ...]:
    """The paths whose hits can carry an importance, derived from what each path reads.

    Derived rather than listed: salience lives on event records, so a path can only
    have an importance if it reads `events.jsonl`. A written-down pair of names could
    not notice a sixth path that reads events, and could not notice `structured`
    dropping events either.
    """
    return tuple(name for name in recall.paths()
                 if recall.EVENTS_ARTIFACT in recall.PATHS[name].reads)


# --------------------------------------------------------------------- evidence
class Freshness:
    """How old one hit is, or which of five reasons says it cannot be aged."""

    __slots__ = ("status", "age_s")

    def __init__(self, status: str, age_s: float | None = None) -> None:
        if status not in FRESHNESS_STATUSES:
            raise RankError(f"unknown freshness status {status!r}; the set is "
                            f"{FRESHNESS_STATUSES}")
        if (status == DATED) != (age_s is not None):
            raise RankError(
                f"{status!r} with age_s={age_s!r}: an age and a reason there is no age "
                "are the two states this type exists to keep apart")
        self.status = status
        self.age_s = None if age_s is None else float(age_s)

    def as_dict(self) -> dict:
        return {"status": self.status, "age_s": self.age_s}


class Importance:
    """How much one hit's source turn was worth remembering, or why that is unknown."""

    __slots__ = ("status", "value", "formula")

    def __init__(self, status: str, value: float | None = None,
                 formula: str | None = None) -> None:
        if status not in IMPORTANCE_STATUSES:
            raise RankError(f"unknown importance status {status!r}; the set is "
                            f"{IMPORTANCE_STATUSES}")
        if (status == SCORED) != (value is not None):
            raise RankError(
                f"{status!r} with value={value!r}: an absent measurement is not a "
                "measurement, and a value without SCORED is one an arm would compare")
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise RankError(
                f"salience {value!r} is outside [0, 1]; memory.salience.score clamps, "
                "so a value out here came from somewhere that is not the formula")
        self.status = status
        self.value = None if value is None else float(value)
        self.formula = formula

    def as_dict(self) -> dict:
        return {"status": self.status, "value": self.value, "formula": self.formula}


class Evidence:
    """Everything the arms may know about the hits of one `Result`, keyed by `Hit.ref`.

    Built once and shared by all three arms, so the arms differ in their *policy* and
    not in what they were told. An arm that read the disk itself could disagree with
    its siblings about the fixture, and the comparison would be reporting that
    disagreement as an effect of the policy.
    """

    __slots__ = ("freshness", "importance", "findings", "formula")

    def __init__(self, freshness: Mapping[str, Freshness],
                 importance: Mapping[str, Importance], findings: Sequence[str],
                 formula: str) -> None:
        if sorted(freshness) != sorted(importance):
            raise RankError(
                "every hit needs both a freshness and an importance entry; "
                f"{sorted(set(freshness) ^ set(importance))} has only one")
        self.freshness = dict(freshness)
        self.importance = dict(importance)
        self.findings = list(findings)
        self.formula = formula

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self.freshness))

    def freshness_of(self, ref: str) -> Freshness:
        try:
            return self.freshness[ref]
        except KeyError:
            raise RankError(
                f"no freshness for {ref!r}: an arm is ranking a hit this evidence was "
                "not built from, so the two are describing different results") from None

    def importance_of(self, ref: str) -> Importance:
        try:
            return self.importance[ref]
        except KeyError:
            raise RankError(f"no importance for {ref!r}") from None

    def census(self) -> dict[str, dict[str, int]]:
        """Counts per status for both axes, **zeros included**.

        Every member of both status tuples, for `Result.census`'s reason: omitting the
        zeros makes "nothing was unrankable" and "the unrankable counter was removed"
        the same output.
        """
        fresh = {s: 0 for s in FRESHNESS_STATUSES}
        imp = {s: 0 for s in IMPORTANCE_STATUSES}
        for f in self.freshness.values():
            fresh[f.status] += 1
        for i in self.importance.values():
            imp[i.status] += 1
        return {"freshness": fresh, "importance": imp}

    def as_dict(self) -> dict:
        return {"formula": self.formula, "census": self.census(),
                "findings": list(self.findings)}


def evidence(result: recall.Result, root: Path, *, now: float,
             formula: str | None = None) -> Evidence:
    """Build the `Evidence` for `result` from the artifacts in `root` at time `now`.

    `now` is keyword-only and has no default, for §6.1's reason: a module that called
    `time.time()` itself would make every decay assertion a test of the wall clock, and
    a default of "now" would let a caller forget the clock and still get numbers.
    """
    if now is None or isinstance(now, bool) or not isinstance(now, (int, float)):
        raise RankError(
            f"now={now!r} is not a clock reading: decay is a function of elapsed time "
            "and an absent clock is not the present moment")
    baseline = formula_id() if formula is None else str(formula)
    root = Path(root)

    entries = {}
    if (root / recall.EPISODES_ARTIFACT).exists():
        entries = {e["id"]: e
                   for e in recall.load_episodes(root / recall.EPISODES_ARTIFACT)}
    dated, _undatable = datable(entries.values())
    dated_ids = {e["id"]: float(e["end_ts"]) for e in dated}

    salient: dict[str, tuple[float | None, str | None]] = {}
    if (root / recall.EVENTS_ARTIFACT).exists():
        for event in recall.load_events(root):
            value = event.get("salience")
            salient[recall.event_ref(event["seq"])] = (
                None if value is None else float(value),
                event.get("salience_formula"))

    scorable = set(scorable_paths())
    freshness: dict[str, Freshness] = {}
    importance: dict[str, Importance] = {}
    findings: list[str] = []
    for hit in result.hits(k=None):
        freshness[hit.ref] = _freshness_of(hit, entries, dated_ids, now, findings)
        importance[hit.ref] = _importance_of(hit, salient, scorable, baseline, findings)
    return Evidence(freshness, importance, findings, baseline)


def _freshness_of(hit: recall.Hit, entries: Mapping[str, Mapping],
                  dated_ids: Mapping[str, float], now: float,
                  findings: list[str]) -> Freshness:
    if hit.episode == recall.SEED:
        return Freshness(NO_EPISODE_BY_DESIGN)
    if hit.episode == recall.ORPHAN:
        findings.append(
            f"{hit.ref}: {recall.ORPHAN} — no episode to age it against. The missing "
            "thing is the attribution, not the timestamp")
        return Freshness(LOST_ATTRIBUTION)
    if hit.episode not in entries:
        findings.append(
            f"{hit.ref}: names episode {hit.episode} which the warm layer does not "
            f"hold, so no arm can rank it by time")
        return Freshness(NO_ENTRY)
    if hit.episode not in dated_ids:
        findings.append(
            f"{hit.ref}: episode {hit.episode} carries no end_ts. It is ranked in the "
            "non-comparable tier rather than treated as brand new or as ancient")
        return Freshness(UNDATABLE)
    age = float(now) - dated_ids[hit.episode]
    if age < 0.0:
        findings.append(
            f"{hit.ref}: episode {hit.episode} ends {-age:.3f}s after the injected "
            "clock. A negative age would score above every real hit")
        return Freshness(FUTURE_TS)
    return Freshness(DATED, age)


def _importance_of(hit: recall.Hit,
                   salient: Mapping[str, tuple[float | None, str | None]],
                   scorable: Iterable[str], baseline: str,
                   findings: list[str]) -> Importance:
    if hit.path not in scorable:
        return Importance(UNSCORABLE)
    if hit.ref not in salient:
        findings.append(
            f"{hit.ref}: no event record on disk behind a hit from the {hit.path} "
            "path, so its importance cannot be looked up")
        return Importance(NO_RECORD)
    value, stamped = salient[hit.ref]
    if value is None:
        return Importance(NO_SALIENCE)
    if stamped != baseline:
        findings.append(
            f"{hit.ref}: salience {value!r} was stamped by {stamped!r}, not "
            f"{baseline!r}. Two formulas are two quantities; comparing them would be "
            "the silent incomparability memory/salience.py names")
        return Importance(OTHER_FORMULA, None, stamped)
    return Importance(SCORED, value, stamped)


# --------------------------------------------------------------------- rankings
class Ranking:
    """One arm's answer: an order per path, what it dropped, and what it could not judge.

    `order` is a mapping keyed by path and never a single fused list, for the reason
    `Result.hits` gives. `dropped` is flat, because dropping is not an ordering
    question.
    """

    __slots__ = ("arm", "order", "dropped", "findings")

    def __init__(self, arm: str, order: Mapping[str, Sequence[str]],
                 dropped: Sequence[str], findings: Sequence[str]) -> None:
        if arm not in ARMS:
            raise RankError(f"unknown arm {arm!r}; the registry is {arms()}")
        if sorted(order) != sorted(recall.paths()):
            raise RankError(
                f"a ranking must carry every path exactly once; got {sorted(order)} "
                f"against {sorted(recall.paths())} — a missing path is one the "
                "comparison would silently not look at")
        self.arm = arm
        self.order = {name: tuple(order[name]) for name in recall.paths()}
        self.dropped = tuple(dropped)
        self.findings = list(findings)

    def refs(self) -> tuple[str, ...]:
        """Every kept ref, paths in run order and each path in the arm's order."""
        return tuple(r for name in recall.paths() for r in self.order[name])

    def as_dict(self) -> dict:
        return {"arm": self.arm,
                "order": {k: list(v) for k, v in self.order.items()},
                "dropped": list(self.dropped), "findings": list(self.findings),
                "kept": len(self.refs())}


def _baseline(result: recall.Result) -> dict[str, tuple[str, ...]]:
    """Each path's hits in the retriever's own order. The order every arm starts from."""
    out: dict[str, tuple[str, ...]] = {}
    for path in recall.paths():
        out[path] = tuple(h.ref for h in result.hits(k=None) if h.path == path)
    return out


def recency(age_s: float, *, half_life_s: float = HALF_LIFE_S) -> float:
    """0.5 ** (age / half-life). In (0, 1], and 1.0 for something that just happened."""
    if half_life_s <= 0:
        raise RankError(
            f"half_life_s must be positive, got {half_life_s!r}: a non-positive "
            "half-life makes every hit equally recent, which reads as a decay arm "
            "that ran")
    if age_s < 0:
        raise RankError(f"age_s={age_s!r} is negative; that is FUTURE_TS, not an age")
    return float(0.5 ** (float(age_s) / float(half_life_s)))


def _tiered(refs: Sequence[str], keys: Mapping[str, float]) -> tuple[str, ...]:
    """`refs` with the keyed ones first, descending, then the rest in baseline order.

    Two tiers rather than one list with a substituted key. Any substitute would be a
    number invented for a missing measurement, and both available substitutes are
    wrong in a way that looks like a working arm: the maximum promotes every unmeasured
    hit and the minimum buries it, which under `decay` is a deletion.
    """
    scored = sorted((r for r in refs if r in keys),
                    key=lambda r: (-keys[r], refs.index(r)))
    return tuple(scored) + tuple(r for r in refs if r not in keys)


def _tier_findings(arm: str, path: str, refs: Sequence[str],
                   keys: Mapping[str, float], reasons: Mapping[str, str]) -> list[str]:
    unranked = [r for r in refs if r not in keys]
    if not unranked:
        return []
    census: dict[str, int] = {}
    for ref in unranked:
        census[reasons[ref]] = census.get(reasons[ref], 0) + 1
    return [f"{arm}/{path}: {len(unranked)} of {len(refs)} hit(s) left in baseline "
            f"order because they cannot be judged: "
            f"{', '.join(f'{k}={v}' for k, v in sorted(census.items()))}"]


def _arm_decay(result: recall.Result, ev: Evidence, *,
               half_life_s: float = HALF_LIFE_S,
               floor: float = DECAY_FLOOR) -> Ranking:
    """①衰減. Reciprocal rank × recency, and a hit too old to matter is suppressed.

    Suppressed, not deleted: plan §1 is explicit that decay is probabilistic retrieval
    suppression and that 「忘記我說的 X」 must go through `memory.erase`'s deterministic
    transaction. A dropped hit is still on disk and `dropped` names it.
    """
    order, dropped, findings = {}, [], []
    for path, refs in _baseline(result).items():
        keys, reasons, kept = {}, {}, []
        for i, ref in enumerate(refs):
            fresh = ev.freshness_of(ref)
            if fresh.status != DATED:
                reasons[ref] = fresh.status
                kept.append(ref)
                continue
            weight = recency(fresh.age_s, half_life_s=half_life_s)
            if weight < floor:
                dropped.append(ref)
                continue
            keys[ref] = weight / (1.0 + i)
            kept.append(ref)
        order[path] = _tiered(kept, keys)
        findings += _tier_findings(DECAY, path, kept, keys, reasons)
    return Ranking(DECAY, order, dropped, findings)


def _arm_prune(result: recall.Result, ev: Evidence, *,
               minimum: float = PRUNE_MIN) -> Ranking:
    """②重要性剪枝. Baseline order, and a hit below the threshold is removed.

    The order is deliberately untouched, so that a difference between this arm and
    `reorder` is attributable to the *removal* and not to a second change made at the
    same time. Its honest limit: importance lives on event records, so this arm can
    only reach the paths `scorable_paths()` derives — a finding says so per path
    rather than leaving the reader to infer it from an unchanged answer.
    """
    order, dropped, findings = {}, [], []
    for path, refs in _baseline(result).items():
        kept, judged = [], 0
        reasons: dict[str, str] = {}
        for ref in refs:
            imp = ev.importance_of(ref)
            if imp.status != SCORED:
                reasons[ref] = imp.status
                kept.append(ref)
                continue
            judged += 1
            if imp.value < minimum:
                dropped.append(ref)
            else:
                kept.append(ref)
        order[path] = tuple(kept)
        if refs and judged == 0:
            findings.append(
                f"{PRUNE}/{path}: 0 of {len(refs)} hit(s) could be judged, so this "
                f"path is untouched by the importance arm. Statuses: "
                f"{sorted(set(reasons.values()))}")
        elif reasons:
            findings.append(
                f"{PRUNE}/{path}: {len(reasons)} of {len(refs)} hit(s) kept unjudged "
                f"rather than pruned on an absent measurement: "
                f"{sorted(set(reasons.values()))}")
    return Ranking(PRUNE, order, dropped, findings)


def _arm_reorder(result: recall.Result, ev: Evidence, *,
                 half_life_s: float = HALF_LIFE_S) -> Ranking:
    """③不刪除只改排序. Reciprocal rank × recency × importance, and nothing is dropped.

    This is the control the whole ablation turns on. Comparing 衰減 against a baseline
    conflates two changes — a different order and a smaller answer — so an arm that
    carries the same ordering signal with an empty `dropped` is what makes the question
    「刪除本身有幫助嗎」 answerable at all. `dropped` being empty is an invariant of this
    arm, asserted rather than merely true.
    """
    order, findings = {}, []
    for path, refs in _baseline(result).items():
        keys, reasons = {}, {}
        for i, ref in enumerate(refs):
            fresh, imp = ev.freshness_of(ref), ev.importance_of(ref)
            if fresh.status != DATED or imp.status != SCORED:
                reasons[ref] = (fresh.status if fresh.status != DATED
                                else imp.status)
                continue
            keys[ref] = (recency(fresh.age_s, half_life_s=half_life_s)
                         * imp.value / (1.0 + i))
        order[path] = _tiered(refs, keys)
        findings += _tier_findings(REORDER, path, refs, keys, reasons)
    return Ranking(REORDER, order, (), findings)


def rank(arm: str, result: recall.Result, ev: Evidence, **kwargs: Any) -> Ranking:
    """Run one arm. Dispatched through the registry, so an unregistered arm cannot run."""
    if arm not in ARMS:
        raise RankError(f"unknown arm {arm!r}; the registry is {arms()}")
    return globals()[f"_arm_{arm}"](result, ev, **kwargs)


# -------------------------------------------------------------------- comparison
def ablation(result: recall.Result, ev: Evidence, *,
             labels: Mapping[str, Iterable[str]] | None = None) -> dict:
    """Run every registered arm over one result and return a JSON-ready comparison.

    §6.0 rule 6: the numbers come out of here as data and the documents cite the data.
    `quality` is a status because the labels that would make it a number are T-F1's,
    and T-F1 needs real logs and one human annotation pass.
    """
    if labels is not None:
        raise RankError(
            "labelled scoring is not implemented, and it must not be faked: T-F1's 30 "
            "frozen queries need one day of real logs and one annotation pass. A "
            "quality number computed without them would state 'the arms are equally "
            "good' in the only format nobody re-reads")
    rankings = {name: rank(name, result, ev) for name in arms()}
    baseline = _baseline(result)
    return {
        "query": result.query,
        "formula": ev.formula,
        "half_life_s": HALF_LIFE_S,
        "decay_floor": DECAY_FLOOR,
        "prune_min": PRUNE_MIN,
        "baseline": {k: list(v) for k, v in baseline.items()},
        "evidence": ev.as_dict(),
        "arms": {name: r.as_dict() for name, r in rankings.items()},
        "identical_arms": identical_arms(rankings),
        "quality": NO_LABELS,
    }


def identical_arms(rankings: Mapping[str, Ranking]) -> list[list[str]]:
    """Groups of arms whose order **and** drops coincide, as data rather than as prose.

    Two arms with identical output are not two agreeing measurements, they are one
    measurement reported twice — §6.0 rule 2 seen from the other side, where the
    byte-identical output is the whole comparison rather than one mutant. Returned as
    data so a caller can aggregate it across queries: one collision on one query is
    noise, and a collision on *every* query means the two policies are the same policy.
    """
    shape: dict[tuple, list[str]] = {}
    for name in sorted(rankings):
        r = rankings[name]
        key = (tuple((p, tuple(r.order[p])) for p in recall.paths()),
               tuple(sorted(r.dropped)))
        shape.setdefault(key, []).append(name)
    return [sorted(names) for names in shape.values() if len(names) > 1]


def ablation_findings(report: Mapping, *, include_identical: bool = True) -> list[str]:
    """What is structurally wrong with `report`. Empty means "well-formed", not "good".

    T-F6's criterion is 「三組數字齊備才允許下結論;缺任一組 → harness 非零退出」, so a
    missing arm is a finding here and a distinct exit code in `memory/rank_harness.py`.
    Deliberately **not** the same list as `conclusion_blockers`: "the comparison is
    malformed" and "the comparison is fine and still cannot conclude anything" are two
    claims, and a single findings list would let a fixed structural defect look like
    progress towards a conclusion it is not progress towards.

    `include_identical=False` leaves the arm-collision check to the caller. A multi-query
    harness has to aggregate that one — a collision on one query is noise and a collision
    on every query is one policy under two names — and it aggregates from
    `report["identical_arms"]`, which is data. Re-deriving it by matching the strings
    this function returns would make the severity rule depend on the wording of a
    message.
    """
    findings: list[str] = []
    present = set(report.get("arms", {}))
    missing = [a for a in arms() if a not in present]
    if missing:
        findings.append(
            f"arm(s) {missing} did not run: a comparison of {sorted(present)} is "
            "the 「A/B 只有 A」 T-F6 exists to refuse")
    extra = sorted(present - set(arms()))
    if extra:
        findings.append(f"{extra} is in the report but not in the registry {arms()}")

    for names in (report.get("identical_arms", ()) if include_identical else ()):
        findings.append(
            f"arms {sorted(names)} produced identical orders and identical drops on "
            "this result, so the comparison distinguishes fewer policies than it "
            "reports")

    reorder = report.get("arms", {}).get(REORDER)
    if reorder is not None and reorder["dropped"]:
        findings.append(
            f"{REORDER} dropped {reorder['dropped']}: the arm that isolates ordering "
            "from removal cannot remove anything")

    if report.get("quality") != NO_LABELS:
        findings.append(
            f"quality is {report.get('quality')!r}: the only honest value before "
            f"T-F1's labels exist is {NO_LABELS!r}")
    return findings


def conclusion_blockers(report: Mapping) -> list[str]:
    """Why `report` may not be read as saying which arm is better. Never empty today.

    Separate from `ablation_findings` because it is a separate claim, and unconditional
    because the blocker is data that does not exist: T-F1's 30 frozen queries need one
    day of real logs and one human annotation pass. A comparison that printed a number
    here would state 「三組一樣好」 in the one format nobody re-reads.
    """
    blockers = []
    if report.get("quality") == NO_LABELS:
        blockers.append(
            "no relevance labels exist, so this run compares mechanical properties of "
            "three orderings and cannot rank them by quality. T-F1's 30 frozen queries "
            "are the blocker: they need one day of real logs and one annotation pass, "
            "and neither may be fabricated")
    return blockers


def kept_findings(result: recall.Result, ranking: Ranking) -> list[str]:
    """Whether `ranking` is a permutation of `result` minus exactly what it dropped.

    An arm that loses a hit without reporting it as dropped is the defect this catches,
    and it is the one that would be invisible in the report: the answer just gets
    quietly shorter, which is indistinguishable from a policy working well.
    """
    findings = []
    baseline = _baseline(result)
    for path in recall.paths():
        before, after = set(baseline[path]), set(ranking.order[path])
        if len(ranking.order[path]) != len(after):
            findings.append(f"{ranking.arm}/{path}: a ref appears twice in the order")
        lost = sorted(before - after - set(ranking.dropped))
        if lost:
            findings.append(
                f"{ranking.arm}/{path}: {lost} vanished without being dropped")
        gained = sorted(after - before)
        if gained:
            findings.append(f"{ranking.arm}/{path}: {gained} is not in the result")
    stray = sorted(set(ranking.dropped) - {r for v in baseline.values() for r in v})
    if stray:
        findings.append(f"{ranking.arm}: dropped {stray}, which was never a hit")
    return findings


def policy_findings() -> list[str]:
    """Everything wrong with the policy numbers. A pure function, not an import-time
    assert: `python -O` removes an assert and a node may want to refuse to boot."""
    findings = []
    if HALF_LIFE_S != HALF_LIFE_DAYS * 24 * 60 * 60:
        findings.append("HALF_LIFE_S is not HALF_LIFE_DAYS converted")
    if HALF_LIFE_S <= 0:
        findings.append(f"HALF_LIFE_S is {HALF_LIFE_S!r}, which is not a half-life")
    if not 0.0 < DECAY_FLOOR < 1.0:
        findings.append(
            f"DECAY_FLOOR is {DECAY_FLOOR!r}: at 0 the decay arm never suppresses "
            "anything and at 1 it suppresses everything, and both look like an arm "
            "that ran")
    if not 0.0 < PRUNE_MIN <= 1.0:
        findings.append(f"PRUNE_MIN is {PRUNE_MIN!r}, which is not a salience")
    if math.isclose(recency(HALF_LIFE_S), 0.5) is False:
        findings.append("one half-life does not halve the weight")
    if set(ARMS) != {DECAY, PRUNE, REORDER} or len(ARMS) != 3:
        findings.append(f"ARMS is {ARMS}: 第五輪 asked for exactly three arms")
    return findings
