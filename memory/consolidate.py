"""Semantic consolidation — the gate a profile fact has to pass (plan §S5).

Zero ROS, zero network, zero LLM: this module decides *whether* a candidate fact
may be written and what its provenance is. Extraction — turning a conversation into
candidate `(kind, value, channel)` triples — is the part that calls a model, and it
lives in the node. The split is the point: the expensive, non-deterministic half
proposes, and this deterministic half is the only thing that can accept. A design
where the extractor also decides has no place to put a test.

**Two rules, and the second one is the one that is easy to get wrong.**

T-C2: a fact needs evidence from **N ≥ 2 distinct episodes**. The failure mode it
exists for is not a lie — it is a mishearing, or a hypothetical ("如果我討厭芫茜的話"),
or a sentence the person said once while quoting someone else. One episode is one
occasion, and the cheapest thing that distinguishes a durable preference from a
transcription artefact is whether it comes back on a different occasion.

The rule is therefore about **occasions, not mentions**. Someone who says 芫茜 four
times inside one sentence has said it once as far as this gate is concerned, and
counting mentions would make an enthusiastic sentence indistinguishable from a
repeated one — while also handing an extractor that emits duplicates the power to
promote its own guess. `plan §0-6` is explicit that the closed predicate set stops
fabricated *facts* and does nothing about *preferences*; this counting rule is what
guards those, so it is the rule an implementation may not weaken quietly.

T-C3: every accepted fact carries `source_episode_ids`, non-empty, and every id in
it resolves to an episode that exists. `memory.profile.Fact` already refuses an
empty tuple for any trust above `seed` — that refusal is the last line, not this
one — but it cannot check *resolvability*, because it has no episode set. An id that
does not resolve is worse than a missing one: it is a fact that survives the
deletion of everything behind it (plan §0-5), which is the defect the whole
build-order correction was about.

**Why an episode id is derived from a store sequence number and never a position.**
`segment.Episode` carries both `seqs` (store sequence numbers) and `indices`
(positions in the replayed stream), and only one of them can be an identity. Take
the position: delete an earlier episode, re-segment, and every later episode's index
shifts down by one — so a fact whose source was episode 7 now names episode 6 and
**resolves successfully to the wrong episode**. Nothing raises, nothing is missing,
and the fact has silently acquired a source that never produced it. Take the first
sequence number: the same deletion leaves that id resolving to nothing, which is a
dangling reference the cascade can find and act on. Both designs are wrong after a
deletion; only one of them is *detectably* wrong, and detectable is the entire
requirement for a deletion path (T-C4). §S4.5 already bought the other half of
this — a record refused for its content consumes no sequence number — so the ids
this module mints have no gaps that a deletion could be confused with.

The id is zero-padded to a fixed width so that lexicographic order is numeric
order: `source_episode_ids` is stored sorted, and a mixed-width scheme would sort
`ep-1000000` before `ep-999999`, making two facts with the same sources serialise
differently. A sequence number past the width is **refused** rather than widened,
because widening is the version where the bug appears once, in production, on the
millionth event, in a file nobody re-reads.

**Value matching is byte-exact, and that is a decision rather than a limitation.**
Two spellings of the same preference count as two facts here, and each therefore
needs its own two episodes. The alternative — a fuzzy matcher upstream of the
counter — is a component that can merge `allergy: 花生` with `allergy: 花生醬`, and
a wrong merge does not just store a wrong value: it *manufactures* the second
occasion that the N ≥ 2 rule demands. A gate whose evidence can be synthesised by
the thing it is gating is not a gate. So any merging is the extractor's explicit,
logged decision, and this module counts what it is handed.

The same reasoning refuses a value that is not already stripped, rather than
stripping it. `Fact` strips on construction, so a gate that grouped on the raw
string would count `"芫茜"` and `"芫茜 "` as two facts and then store one value
twice — the grouping key and the stored value have to be the same bytes or the
count is about something the file does not contain. This is §S4.5's refuse-don't-
normalise rule for a different reason: there the risk was one retrieval bucket
splitting into two spellings, here it is two counts collapsing into one row.

**Trust is a kind of evidence, not an amount of it.** `confirmed` requires that the
person said yes — a `user_confirmed` channel — and never arrives by repetition,
however many episodes agree. Promotion on a count means that enough repetitions of
the same mishearing become a confirmed fact, which is the one outcome the N ≥ 2
rule was chosen to prevent; the extra episodes are more of the same evidence, not a
better kind. `seed` is never minted here at all: it is the level whose exemption
from `source_episode_ids` exists only for a hand-written file.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from memory.profile import (PREDICATES, TRUST_LEVELS, TRUST_SEED, Fact,
                            ProfileError)
from memory.segment import Episode
from memory.store import StoreError

# The N in "N ≥ 2 independent episodes" (plan §0-6, design item T-C2). Named, and
# read from here by the tests, so that a change to it is one diff line in one file
# rather than a number that has to be found in three.
MIN_EPISODES = 2

# Episode identity. See the module docstring for why this is a sequence number and
# not a position, and why an overflow is refused rather than widened.
EPISODE_ID_PREFIX = "ep-"
EPISODE_ID_DIGITS = 6
MAX_EPISODE_SEQ = 10 ** EPISODE_ID_DIGITS - 1

# How a candidate came to be. Closed, for the reason every other enum in this tree
# is: the channel decides whether the evidence may be counted at all, so a
# free-text channel is a way to count evidence that no rule was written for.
SELF_REPORTED = "self_reported"    # the person said it about themselves
USER_CONFIRMED = "user_confirmed"  # the person said yes when asked
INFERRED = "inferred"              # the robot concluded it from something else
CHANNELS = (SELF_REPORTED, USER_CONFIRMED, INFERRED)

# Channels that raise trust to `confirmed`. A tuple rather than a single constant
# because the set may grow (a signed profile edit would belong here) and must not
# grow by accident: `trust_for` reads this and nothing else.
CONFIRMING_CHANNELS = (USER_CONFIRMED,)

# Which predicates may count *inferred* evidence, written out for every member of
# `PREDICATES` rather than looked up with a default.
#
# `INFERENCE_ALLOWED.get(kind, True)` would let the next predicate default to
# inferable, and `.get(kind, False)` would let it default to blocked — either way
# the decision nobody made becomes indistinguishable from a decision, which is the
# rule `store.WHY_REQUIRED` was written out for. `test_consolidate.py` asserts the
# key set equals `set(PREDICATES)`, so adding a predicate without ruling on it is a
# red test rather than a silent inheritance.
#
# The False rows are the ones plan §0-6 names plus the two that are more dangerous
# for the same reason: `likes`/`dislikes` are the open door in a closed predicate
# set, and `allergy`/`boundary` are the facts whose being wrong is not an
# inconvenience. Everything a detector can legitimately observe about the room is
# True; everything that is a claim about the person's inner state or limits is not.
INFERENCE_ALLOWED = {
    "name": False,            # a heard name is a transcription, not an observation
    "call_me": False,
    "language": True,         # which language they spoke in is observable
    "pronouns": False,        # inferring these from anything is the failure mode
    "likes": False,           # plan §0-6: the open door in the closed set
    "dislikes": False,
    "allergy": False,         # wrong here is not an inconvenience
    "occupation": False,
    "location": True,         # a scene classifier may legitimately say "kitchen"
    "birthday": False,
    "household": True,        # "someone else is in the room" is a detection
    "pet": True,
    "routine": True,          # timestamps are exactly what may be aggregated
    "goal": False,            # a stated goal; guessing it is putting words in
    "skill_interest": False,
    "device": True,           # the robot can see a device
    "boundary": False,        # a user-stated limit, by definition user-stated
    "note": False,            # free-form value under a closed key: never inferred
}

# Every outcome a candidate group can have. The refusals are enumerated because the
# report has to include them: a consolidation pass that reports only what it wrote
# makes "nothing was proposed" and "everything was refused" the same observation,
# and the second one is the one worth waking up to. Same rule as `goal_census`
# including its zeros (§S4.5).
ACCEPTED = "accepted"
DECISIONS = (
    ACCEPTED,
    "one_episode",              # evidence exists but from fewer than MIN_EPISODES
    "inference_not_allowed",    # all evidence was inferred for a predicate that forbids it
)


class ConsolidateError(StoreError):
    """A malformed candidate, an unresolvable episode id, or an id that overflows.

    Subclasses `StoreError` for the reason `ProfileError` and `SegmentError` do:
    one `except` at the node boundary must be able to catch every refusal the
    memory layer can raise, and a separate hierarchy is how one of them ends up
    escaping into the ROS executor and killing the writer.
    """


def episode_id(seq: int) -> str:
    """The stable id of the episode whose first stored record has sequence `seq`."""
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise ConsolidateError(
            f"an episode id is derived from a store sequence number, got "
            f"{seq!r} ({type(seq).__name__})")
    if seq < 1:
        raise ConsolidateError(
            f"sequence numbers start at 1, got {seq} — a 0 here means the caller "
            "passed a list position, which is the identity this module refuses "
            "(see the module docstring: a position resolves to the wrong episode "
            "after a deletion instead of dangling)")
    if seq > MAX_EPISODE_SEQ:
        raise ConsolidateError(
            f"sequence {seq} exceeds the {EPISODE_ID_DIGITS}-digit id width "
            f"(max {MAX_EPISODE_SEQ}); widening it now would make already-stored "
            "ids sort out of order, so this is a refusal and not a resize")
    return f"{EPISODE_ID_PREFIX}{seq:0{EPISODE_ID_DIGITS}d}"


def episode_id_of(episode: Episode) -> str:
    """`episode_id` of an actual `Episode`, which is where its first seq lives."""
    if not isinstance(episode, Episode):
        raise ConsolidateError(f"expected an Episode, got {type(episode).__name__}")
    if not episode.seqs:
        raise ConsolidateError(
            "an episode with no stored records has no identity; it cannot be a "
            "source for a fact and it cannot be deleted, so it must not be one")
    return episode_id(episode.seqs[0])


def known_ids(episodes: Iterable[Episode]) -> tuple[str, ...]:
    """The resolvable id set, derived from episodes rather than written down.

    Derived because a hand-written id list is the artefact that cannot notice a new
    episode — the same reason T-D1 builds its state set from `game_node`'s source.
    """
    return tuple(episode_id_of(e) for e in episodes)


class Observation:
    """One piece of evidence for one candidate fact. Immutable, like `Fact`.

    Immutable for a narrower reason than `Fact`'s: the report published at the end
    of a consolidation pass holds these, and a consumer that renumbered an
    observation's episode would be editing provenance after the decision that
    relied on it.
    """

    __slots__ = ("kind", "value", "channel", "episode")

    def __init__(self, kind: str, value: str, channel: str, episode: str) -> None:
        if kind not in PREDICATES:
            raise ConsolidateError(
                f"unknown predicate {kind!r}; the set is closed and lives in "
                f"memory.profile.PREDICATES ({len(PREDICATES)} entries)")
        if not isinstance(value, str) or not value.strip():
            raise ConsolidateError(f"{kind}: value must be a non-empty string, "
                                   f"got {value!r}")
        if value != value.strip():
            raise ConsolidateError(
                f"{kind}: value {value!r} has surrounding whitespace. Fact strips on "
                "construction, so this gate would count the stripped and unstripped "
                "spellings as two occasions and then store one row — the grouping "
                "key has to be the bytes that get stored")
        if channel not in CHANNELS:
            raise ConsolidateError(
                f"{kind}: unknown channel {channel!r}, expected one of {CHANNELS}")
        if not isinstance(episode, str) or not episode.startswith(EPISODE_ID_PREFIX):
            raise ConsolidateError(
                f"{kind}: {episode!r} is not an episode id; ids are minted by "
                "consolidate.episode_id and look like ep-000042")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "episode", episode)

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConsolidateError(
            f"an Observation is immutable ({name}={value!r}): the consolidation "
            "report holds it as the provenance of a decision already made")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "channel": self.channel,
                "episode": self.episode}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Observation({self.kind!r}, {self.value!r}, {self.channel!r}, "
                f"{self.episode!r})")


def observation(record: Mapping) -> Observation:
    """Build an `Observation` from a stored/extracted mapping, refusing extras.

    Extra keys are refused rather than ignored: an extractor that emits `confidence`
    or `mentions` is proposing a rule this gate does not implement, and silently
    dropping the field makes that proposal look accepted.
    """
    if not isinstance(record, Mapping):
        raise ConsolidateError(f"expected a mapping, got {type(record).__name__}")
    expected = {"kind", "value", "channel", "episode"}
    extra = sorted(set(record) - expected)
    if extra:
        raise ConsolidateError(
            f"unexpected field(s) {extra} on a candidate: this gate counts "
            f"episodes and channels only, so a field it does not read is a rule "
            "nobody implemented")
    missing = sorted(expected - set(record))
    if missing:
        raise ConsolidateError(f"candidate is missing {missing}")
    return Observation(record["kind"], record["value"], record["channel"],
                       record["episode"])


def trust_for(channels: Collection[str]) -> str:
    """The trust level `channels` earns. Never `seed`, never a function of count."""
    if not channels:
        raise ConsolidateError("no channels: a trust level with no evidence behind "
                               "it is the thing source_episode_ids exists to stop")
    unknown = sorted(set(channels) - set(CHANNELS))
    if unknown:
        raise ConsolidateError(f"unknown channel(s) {unknown}; the set is {CHANNELS}")
    level = "confirmed" if any(c in CONFIRMING_CHANNELS for c in channels) else "observed"
    # Asserted rather than assumed: `TRUST_LEVELS` lives in another module, and a
    # rename there would otherwise turn every fact this module mints into one that
    # `Fact` refuses, at the last possible moment and for a reason two files away.
    if level not in TRUST_LEVELS or level == TRUST_SEED:
        raise ConsolidateError(
            f"{level!r} is not a mintable trust level; memory.profile.TRUST_LEVELS "
            f"is {TRUST_LEVELS} and {TRUST_SEED!r} is reserved for the hand-written "
            "card, which is the only file whose facts may have no source")
    return level


class Decision:
    """What happened to one candidate `(kind, value)` group, accepted or not."""

    __slots__ = ("kind", "value", "decision", "episodes", "dropped_inferred", "trust")

    def __init__(self, kind: str, value: str, decision: str,
                 episodes: tuple[str, ...], dropped_inferred: int,
                 trust: str | None) -> None:
        if decision not in DECISIONS:
            raise ConsolidateError(f"unknown decision {decision!r}; the set is "
                                   f"{DECISIONS}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "dropped_inferred", dropped_inferred)
        object.__setattr__(self, "trust", trust)

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConsolidateError(f"a Decision is a record of a past pass ({name})")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "decision": self.decision,
                "episodes": list(self.episodes),
                "dropped_inferred": self.dropped_inferred, "trust": self.trust}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Decision({self.kind!r}, {self.value!r}, {self.decision})"


class Consolidation:
    """The result of one pass: the facts, and a decision for every candidate.

    Both, because the facts alone cannot answer "did anything get refused" — and a
    pass that refuses everything looks exactly like a quiet night if only the
    accepted rows are reported.
    """

    __slots__ = ("facts", "decisions")

    def __init__(self, facts: Sequence[Fact], decisions: Sequence[Decision]) -> None:
        object.__setattr__(self, "facts", tuple(facts))
        object.__setattr__(self, "decisions", tuple(decisions))

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConsolidateError(f"a Consolidation is a record of a past pass ({name})")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def census(self) -> dict[str, int]:
        """Counts for **every** member of `DECISIONS`, including the zeros."""
        counts = {d: 0 for d in DECISIONS}
        for d in self.decisions:
            counts[d.decision] += 1
        return counts

    def as_dict(self) -> dict:
        return {"facts": [f.as_dict() for f in self.facts],
                "decisions": [d.as_dict() for d in self.decisions],
                "census": self.census()}


def consolidate(candidates: Iterable[Mapping | Observation], *,
                episode_ids: Collection[str],
                min_episodes: int = MIN_EPISODES) -> Consolidation:
    """Decide which candidates become facts. Deterministic, no I/O, no model.

    `episode_ids` is the resolvable set — pass `known_ids(episodes)`. It is required
    rather than defaulted to "anything shaped like an id", because T-C3's assertion
    is that every stored id *resolves*, and a gate with no episode set can only
    check the shape.
    """
    if min_episodes < 1:
        raise ConsolidateError(
            f"min_episodes={min_episodes} accepts a fact with no evidence at all; "
            f"the design item is N >= {MIN_EPISODES} and 1 is already the value "
            "T-C2 names as the mutant")
    known = set(episode_ids)
    if not known:
        raise ConsolidateError(
            "the resolvable episode set is empty, so no candidate can be accepted "
            "and every refusal would be attributed to the wrong rule. If a pass "
            "legitimately has no episodes it has no candidates either")
    unresolvable = sorted(i for i in known if not _looks_like_id(i))
    if unresolvable:
        raise ConsolidateError(
            f"{unresolvable} are not episode ids; the resolvable set has to come "
            "from consolidate.known_ids(episodes), not from a hand-written list")

    # Grouping preserves first-appearance order rather than sorting: the report is
    # read next to the log it came from, and `dict` has kept insertion order since
    # 3.7 — so this is deterministic without a sort key that would have to be
    # chosen between `kind` and `value` (T-B4's determinism requirement, one layer
    # up: the same candidates must produce the same facts in the same order).
    groups: dict[tuple[str, str], list[Observation]] = {}
    for candidate in candidates:
        obs = candidate if isinstance(candidate, Observation) else observation(candidate)
        if obs.episode not in known:
            raise ConsolidateError(
                f"{obs.kind}: episode {obs.episode!r} does not resolve — a fact "
                "sourced to an episode that does not exist survives the deletion "
                "of everything behind it, which is plan §0-5's defect")
        groups.setdefault((obs.kind, obs.value), []).append(obs)

    facts: list[Fact] = []
    decisions: list[Decision] = []
    for (kind, value), obs_list in groups.items():
        counted = obs_list
        dropped = 0
        if not INFERENCE_ALLOWED[kind]:
            counted = [o for o in obs_list if o.channel != INFERRED]
            dropped = len(obs_list) - len(counted)
        # Distinct episodes, sorted — the count is over occasions, and the stored
        # order is the id order so that two passes over the same evidence produce
        # byte-identical provenance.
        episodes = tuple(sorted({o.episode for o in counted}))
        if not counted:
            decisions.append(Decision(kind, value, "inference_not_allowed", (),
                                      dropped, None))
            continue
        if len(episodes) < min_episodes:
            decisions.append(Decision(kind, value, "one_episode", episodes, dropped,
                                      None))
            continue
        trust = trust_for([o.channel for o in counted])
        try:
            fact = Fact(kind, value, trust, episodes)
        except ProfileError as exc:  # pragma: no cover - defence in depth
            raise ConsolidateError(
                f"{kind}: this gate accepted a candidate that Fact refuses "
                f"({exc}); the two rule sets have diverged") from exc
        facts.append(fact)
        decisions.append(Decision(kind, value, ACCEPTED, episodes, dropped, trust))
    return Consolidation(facts, decisions)


def _looks_like_id(value: object) -> bool:
    """Shape check for a minted id. Deliberately not a prefix test on its own.

    A bare `startswith(EPISODE_ID_PREFIX)` accepts `ep-` and `ep-abc`, which is
    §S4.5's lesson about a family prefix standing in for a closed set: the check
    that matters is that the id round-trips through the minter, so that is the
    check.
    """
    if not isinstance(value, str) or not value.startswith(EPISODE_ID_PREFIX):
        return False
    digits = value[len(EPISODE_ID_PREFIX):]
    if len(digits) != EPISODE_ID_DIGITS or not digits.isdigit():
        return False
    # `isdigit` is True for non-ASCII digits ('٤', '４'), and `int()` accepts them,
    # so an id built from those would round-trip through `int` and still not be the
    # bytes any other reader compares against. The round-trip is the assertion.
    return episode_id(int(digits)) == value


def provenance_findings(facts: Iterable[Fact], episode_ids: Collection[str]) -> list[str]:
    """Everything wrong with `facts` as provenance (design item T-C3). [] = clean.

    A findings list rather than a raise, matching `profile.budget_findings` and
    `segment.cap_findings`: this is run over a whole card, and the first bad fact
    must not hide the rest.
    """
    known = set(episode_ids)
    findings: list[str] = []
    for n, fact in enumerate(facts):
        where = f"fact {n} ({fact.kind}: {fact.value!r})"
        if fact.trust == TRUST_SEED:
            # The one exemption, and it is stated here rather than skipped silently:
            # a seed fact with sources would be a hand-written file claiming a
            # provenance nothing produced, so that is a finding too.
            if fact.source_episode_ids:
                findings.append(f"{where}: trust {TRUST_SEED!r} with "
                                f"{len(fact.source_episode_ids)} source(s); the seed "
                                "card is hand-written and cites no episode")
            continue
        if not fact.source_episode_ids:
            findings.append(f"{where}: trust {fact.trust!r} with no source episode")
            continue
        for eid in fact.source_episode_ids:
            if not _looks_like_id(eid):
                findings.append(f"{where}: {eid!r} is not a minted episode id")
            elif eid not in known:
                findings.append(f"{where}: source {eid} does not resolve")
        if list(fact.source_episode_ids) != sorted(set(fact.source_episode_ids)):
            findings.append(
                f"{where}: sources {list(fact.source_episode_ids)} are not sorted "
                "and unique, so the same evidence can serialise two ways and a "
                "duplicate id can inflate the episode count")
    return findings
