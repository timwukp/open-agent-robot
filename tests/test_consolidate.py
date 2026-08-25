"""Offline tests for the S5 consolidation gate — T-C2 (N ≥ 2) and T-C3 (provenance).

**Why the positive control is not optional here.** T-C2's pass criterion is that a
preference mentioned once, and a preference mentioned twice inside one episode, do
**not** land. A module that accepts nothing at all satisfies both, and so does a
misspelled predicate, a channel name the extractor never emits, or a group key that
never collides. So every refusal case below is paired with the same evidence spread
across two episodes, which must land — that is what establishes there was something
to refuse rather than nothing to accept.

**Why the episode-identity test deletes an episode.** The plan's T-C3 criterion is
"`source_episode_ids` non-empty and every id resolves". Both halves are satisfied
forever by a scheme that derives the id from a *position*, and satisfied for the
wrong reason: after an earlier episode is deleted, every later position shifts down,
so a fact whose source was episode 7 resolves cleanly to episode 6. Nothing is
missing and nothing raises — the fact has silently acquired a source that never
produced it. The only test that can see the difference is one that deletes and
re-segments, and it is the one that decides whether the deletion path T-C4 builds
next has anything to stand on: a dangling id can be found and acted on, a
re-pointed id cannot.

**Why the mention/occasion mutant is behavioural rather than a count check.** The
"count mentions" mutant produces the same number of facts as the real module for
every input where mentions and episodes agree, which is most of them. What separates
them is one shape — the same value twice inside one episode — and that shape is also
the one an extractor produces by accident when it emits a candidate per sentence. So
the mutant is not killed by "fewer facts appear", it is killed by that specific
fixture, and the fixture is asserted to be that shape rather than assumed.

The mutants live in text-patched copies of `memory/consolidate.py`, keyed by label
for the reason `tests/test_why.py` gives: two mutants sharing a directory would let
the first import win, which presents as a mutant that mysteriously survives.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory import consolidate as co  # noqa: E402
from memory import segment as sg  # noqa: E402
from memory.profile import PREDICATES, TRUST_LEVELS, TRUST_SEED, Fact  # noqa: E402

CONSOLIDATE_PY = REPO_ROOT / "memory" / "consolidate.py"

# Two ids that exist for every test that does not care where they came from. Minted
# rather than written as literals: a hand-typed "ep-000001" would keep passing if
# the width or the prefix changed, and the id format is a stored-data contract.
E1 = co.episode_id(1)
E2 = co.episode_id(2)
E3 = co.episode_id(3)
KNOWN = (E1, E2, E3)


def obs(kind: str, value: str, channel: str, episode: str) -> dict:
    return {"kind": kind, "value": value, "channel": channel, "episode": episode}


def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/consolidate.py.

    The copy uses absolute imports (`from memory.profile import …`), so it binds to
    the *real* profile and segment modules — the mutation is scoped to this file,
    which is what makes a surviving mutant a statement about this file's assertions.
    Zero replacements gives the no-mutant control (§0 rule 2).
    """
    dst = tmp_path / f"mut_{label}" / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    src = CONSOLIDATE_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in consolidate.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "consolidate.py"
    target.write_text(src, encoding="utf-8")
    name = f"consolidate_mutant_{tmp_path.name}_{label}"
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


# ----------------------------------------------------------------- mutant anchors
MIN_N = "MIN_EPISODES = 2"
DISTINCT = "        episodes = tuple(sorted({o.episode for o in counted}))\n"
INFERENCE_READ = "        if not INFERENCE_ALLOWED[kind]:\n"
TRUST_RULE = ('    level = "confirmed" if any(c in CONFIRMING_CHANNELS for c in channels) '
              'else "observed"\n')
RESOLVE_CHECK = "            elif eid not in known:\n"
ID_FROM_SEQ = "    return episode_id(episode.seqs[0])\n"
ID_WIDTH_GUARD = "    if seq > MAX_EPISODE_SEQ:\n"
LOOKS_LIKE_WIDTH = ("    if len(digits) != EPISODE_ID_DIGITS or not digits.isdigit():\n"
                    "        return False\n")
LOOKS_LIKE_ROUNDTRIP = "    return episode_id(int(digits)) == value\n"
UNSTRIPPED_REFUSAL = "        if value != value.strip():\n"
CENSUS_ZEROS = "        counts = {d: 0 for d in DECISIONS}\n"
CENSUS_COUNT = "            counts[d.decision] += 1\n"
EXTRA_FIELD = "    extra = sorted(set(record) - expected)\n"


# ===================================================== T-C2: N >= 2 episodes (D)
def test_one_mention_does_not_land_and_two_episodes_do() -> None:
    """The plan's ① and ③ in one assertion, because ③ is ①'s positive control."""
    once = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                          episode_ids=KNOWN)
    assert once.facts == ()
    assert [d.decision for d in once.decisions] == ["one_episode"]

    twice = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1),
                            obs("likes", "芫茜", co.SELF_REPORTED, E2)],
                           episode_ids=KNOWN)
    assert len(twice.facts) == 1
    assert twice.facts[0].kind == "likes" and twice.facts[0].value == "芫茜"
    assert twice.facts[0].source_episode_ids == (E1, E2)


def test_two_mentions_inside_one_episode_do_not_land() -> None:
    """The plan's ②. The rule counts occasions; this is the shape that separates them."""
    result = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1),
                             obs("likes", "芫茜", co.SELF_REPORTED, E1),
                             obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                            episode_ids=KNOWN)
    assert result.facts == ()
    assert result.decisions[0].decision == "one_episode"
    assert result.decisions[0].episodes == (E1,), "three mentions, one occasion"


def test_min_episodes_is_the_named_constant_and_two_is_what_it_says() -> None:
    assert co.MIN_EPISODES == 2


def test_min_episodes_below_one_is_refused_not_clamped() -> None:
    with pytest.raises(co.ConsolidateError, match="no evidence at all"):
        co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                       episode_ids=KNOWN, min_episodes=0)


def test_a_stricter_threshold_is_allowed_because_only_weakening_is_the_defect() -> None:
    two = [obs("likes", "芫茜", co.SELF_REPORTED, E1),
           obs("likes", "芫茜", co.SELF_REPORTED, E2)]
    assert co.consolidate(two, episode_ids=KNOWN, min_episodes=3).facts == ()
    assert len(co.consolidate(two + [obs("likes", "芫茜", co.SELF_REPORTED, E3)],
                              episode_ids=KNOWN, min_episodes=3).facts) == 1


def test_two_different_values_are_two_candidates_not_two_occasions() -> None:
    """Byte-exact matching, stated as a test so a fuzzy matcher cannot arrive quietly.

    A merge upstream of the counter would not merely store a wrong value: it would
    manufacture the second occasion this gate demands, which is the one input a gate
    must not accept from the thing it gates.
    """
    result = co.consolidate([obs("allergy", "花生", co.SELF_REPORTED, E1),
                             obs("allergy", "花生醬", co.SELF_REPORTED, E2)],
                            episode_ids=KNOWN)
    assert result.facts == ()
    assert result.census() == {"accepted": 0, "one_episode": 2,
                              "inference_not_allowed": 0}


# ============================================== the self-report rule (plan §0-6)
def test_inference_allowed_rules_on_every_predicate_with_no_default() -> None:
    """A `.get(kind, …)` default makes the decision nobody made look like a decision."""
    assert set(co.INFERENCE_ALLOWED) == set(PREDICATES), (
        sorted(set(PREDICATES) - set(co.INFERENCE_ALLOWED)),
        sorted(set(co.INFERENCE_ALLOWED) - set(PREDICATES)))
    assert all(isinstance(v, bool) for v in co.INFERENCE_ALLOWED.values())
    # The rows plan §0-6 names, asserted individually: a table this file only
    # counted could have every value flipped and stay green.
    for kind in ("likes", "dislikes", "allergy", "boundary", "pronouns", "name"):
        assert co.INFERENCE_ALLOWED[kind] is False, kind
    # And it is not a table of all-False, which would make the rule vacuous and the
    # `inference_not_allowed` decision the only one reachable.
    assert any(co.INFERENCE_ALLOWED.values())


def test_inferred_evidence_for_a_preference_is_dropped_not_counted() -> None:
    result = co.consolidate([obs("likes", "芫茜", co.INFERRED, E1),
                             obs("likes", "芫茜", co.INFERRED, E2)],
                            episode_ids=KNOWN)
    assert result.facts == ()
    assert result.decisions[0].decision == "inference_not_allowed"
    assert result.decisions[0].dropped_inferred == 2
    assert result.decisions[0].episodes == ()


def test_inferred_evidence_cannot_top_up_a_single_self_report() -> None:
    """The composition case, which is where a "drop it" rule usually leaks."""
    result = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1),
                             obs("likes", "芫茜", co.INFERRED, E2),
                             obs("likes", "芫茜", co.INFERRED, E3)],
                            episode_ids=KNOWN)
    assert result.facts == ()
    assert result.decisions[0].decision == "one_episode"
    assert result.decisions[0].episodes == (E1,)
    assert result.decisions[0].dropped_inferred == 2


def test_inferred_evidence_counts_for_a_predicate_that_allows_it() -> None:
    """The positive control for the drop rule: the table has both kinds of row."""
    result = co.consolidate([obs("device", "客廳的電風扇", co.INFERRED, E1),
                             obs("device", "客廳的電風扇", co.INFERRED, E2)],
                            episode_ids=KNOWN)
    assert len(result.facts) == 1
    assert result.facts[0].trust == "observed"
    assert result.decisions[0].dropped_inferred == 0


# ================================================== trust is a kind, not an amount
def test_trust_is_observed_until_the_person_says_yes() -> None:
    assert co.trust_for([co.SELF_REPORTED, co.SELF_REPORTED]) == "observed"
    assert co.trust_for([co.SELF_REPORTED] * 9) == "observed", "repetition is not consent"
    assert co.trust_for([co.SELF_REPORTED, co.USER_CONFIRMED]) == "confirmed"
    assert co.trust_for([co.INFERRED]) == "observed"


def test_trust_never_mints_the_seed_level() -> None:
    for channels in ([co.SELF_REPORTED], [co.USER_CONFIRMED], [co.INFERRED],
                     list(co.CHANNELS)):
        assert co.trust_for(channels) != TRUST_SEED
        assert co.trust_for(channels) in TRUST_LEVELS


def test_trust_refuses_no_evidence_and_an_unknown_channel() -> None:
    with pytest.raises(co.ConsolidateError, match="no channels"):
        co.trust_for([])
    with pytest.raises(co.ConsolidateError, match="unknown channel"):
        co.trust_for(["overheard"])


def test_confirming_channels_is_a_subset_of_channels() -> None:
    assert set(co.CONFIRMING_CHANNELS) <= set(co.CHANNELS)
    assert co.INFERRED not in co.CONFIRMING_CHANNELS


# ========================================================== candidate validation
def test_an_unstripped_value_is_refused_rather_than_stripped() -> None:
    """Fact strips, so grouping on the raw string counts one occasion twice."""
    with pytest.raises(co.ConsolidateError, match="surrounding whitespace"):
        co.Observation("likes", "芫茜 ", co.SELF_REPORTED, E1)
    # The stripped spelling is accepted, so this is a refusal and not a rejection of
    # the value itself.
    assert co.Observation("likes", "芫茜", co.SELF_REPORTED, E1).value == "芫茜"


@pytest.mark.parametrize("kind,value,channel,episode,match", [
    ("capital_of", "台北", co.SELF_REPORTED, E1, "unknown predicate"),
    ("likes", "", co.SELF_REPORTED, E1, "non-empty string"),
    ("likes", "   ", co.SELF_REPORTED, E1, "non-empty string"),
    ("likes", 42, co.SELF_REPORTED, E1, "non-empty string"),
    ("likes", "芫茜", "overheard", E1, "unknown channel"),
    ("likes", "芫茜", co.SELF_REPORTED, "1", "not an episode id"),
    ("likes", "芫茜", co.SELF_REPORTED, 1, "not an episode id"),
])
def test_observation_refuses_a_malformed_candidate(kind, value, channel, episode,
                                                   match) -> None:
    with pytest.raises(co.ConsolidateError, match=match):
        co.Observation(kind, value, channel, episode)


def test_an_observation_is_immutable() -> None:
    o = co.Observation("likes", "芫茜", co.SELF_REPORTED, E1)
    with pytest.raises(co.ConsolidateError, match="immutable"):
        o.channel = co.USER_CONFIRMED
    with pytest.raises(co.ConsolidateError):
        del o.episode


def test_an_extra_field_on_a_candidate_is_refused_not_ignored() -> None:
    """An extractor emitting `mentions` is proposing a rule this gate does not have."""
    with pytest.raises(co.ConsolidateError, match="unexpected field"):
        co.observation({**obs("likes", "芫茜", co.SELF_REPORTED, E1), "mentions": 4})
    with pytest.raises(co.ConsolidateError, match="missing"):
        co.observation({"kind": "likes", "value": "芫茜"})
    with pytest.raises(co.ConsolidateError, match="mapping"):
        co.observation(["likes", "芫茜"])


def test_consolidate_accepts_observations_and_mappings_alike() -> None:
    mixed = [co.Observation("pet", "一隻叫小黑的貓", co.SELF_REPORTED, E1),
             obs("pet", "一隻叫小黑的貓", co.USER_CONFIRMED, E2)]
    result = co.consolidate(mixed, episode_ids=KNOWN)
    assert len(result.facts) == 1 and result.facts[0].trust == "confirmed"


# ================================================== T-C3: provenance (D + property)
def test_every_accepted_fact_carries_sorted_unique_resolvable_sources() -> None:
    result = co.consolidate(
        [obs("pet", "貓", co.SELF_REPORTED, e) for e in (E3, E1, E2, E1)],
        episode_ids=KNOWN)
    assert len(result.facts) == 1
    ids = result.facts[0].source_episode_ids
    assert ids == (E1, E2, E3), "sources must be sorted, unique and complete"
    assert co.provenance_findings(result.facts, KNOWN) == []


def test_provenance_findings_names_an_unresolvable_source() -> None:
    fact = Fact("pet", "貓", "observed", (E1, co.episode_id(999)))
    findings = co.provenance_findings([fact], KNOWN)
    assert len(findings) == 1 and "does not resolve" in findings[0]


def test_provenance_findings_reports_every_fact_not_just_the_first() -> None:
    """A findings list, so one bad fact cannot hide the rest of the card."""
    bad = [Fact("pet", "貓", "observed", (co.episode_id(998),)),
           Fact("likes", "芫茜", "confirmed", (co.episode_id(999),))]
    assert len(co.provenance_findings(bad, KNOWN)) == 2


def test_provenance_findings_rejects_unsorted_or_duplicated_sources() -> None:
    assert co.provenance_findings([Fact("pet", "貓", "observed", (E2, E1))],
                                  KNOWN)[0].endswith(
        "and a duplicate id can inflate the episode count")
    assert co.provenance_findings([Fact("pet", "貓", "observed", (E1, E1))], KNOWN)


def test_provenance_findings_rejects_an_id_that_was_never_minted() -> None:
    assert "not a minted episode id" in co.provenance_findings(
        [Fact("pet", "貓", "observed", ("ep-abc",))], KNOWN)[0]
    assert "not a minted episode id" in co.provenance_findings(
        [Fact("pet", "貓", "observed", ("42",))], KNOWN)[0]


def test_a_seed_fact_is_exempt_from_sources_and_refused_for_having_them() -> None:
    assert co.provenance_findings([Fact("name", "Tim", TRUST_SEED)], KNOWN) == []
    findings = co.provenance_findings([Fact("name", "Tim", TRUST_SEED, (E1,))], KNOWN)
    assert len(findings) == 1 and "hand-written" in findings[0]


def test_a_clean_card_of_seed_and_consolidated_facts_has_no_findings() -> None:
    result = co.consolidate([obs("language", "粵語", co.SELF_REPORTED, E1),
                             obs("language", "粵語", co.USER_CONFIRMED, E2)],
                            episode_ids=KNOWN)
    card = [Fact("name", "Tim", TRUST_SEED), *result.facts]
    assert co.provenance_findings(card, KNOWN) == []


# ======================================================= episode identity (T-C3)
def test_an_episode_id_is_derived_from_a_sequence_number() -> None:
    assert co.episode_id(1) == "ep-000001"
    assert co.episode_id(999999) == "ep-999999"
    assert co.episode_id_of(sg.Episode("silence", (42, 43), (0, 1), 0.0, 1.0, 2,
                                       "conversation")) == "ep-000042"


def test_an_id_wider_than_the_field_is_refused_not_widened() -> None:
    """Widening is the version where the bug appears once, on the millionth event."""
    with pytest.raises(co.ConsolidateError, match="sort out of order"):
        co.episode_id(co.MAX_EPISODE_SEQ + 1)
    assert co.episode_id(co.MAX_EPISODE_SEQ) == f"ep-{co.MAX_EPISODE_SEQ}"


def test_ids_sort_in_numeric_order_because_they_are_fixed_width() -> None:
    ids = [co.episode_id(n) for n in (2, 10, 999999, 1)]
    assert sorted(ids) == [co.episode_id(n) for n in (1, 2, 10, 999999)]


@pytest.mark.parametrize("seq,match", [
    (0, "sequence numbers start at 1"),
    (-1, "sequence numbers start at 1"),
    (True, "derived from a store sequence number"),
    ("1", "derived from a store sequence number"),
    (1.0, "derived from a store sequence number"),
])
def test_episode_id_refuses_a_non_sequence(seq, match) -> None:
    with pytest.raises(co.ConsolidateError, match=match):
        co.episode_id(seq)


def test_an_episode_with_no_records_has_no_identity() -> None:
    empty = sg.Episode("end_of_stream", (), (), 0.0, 0.0, 0, "conversation")
    with pytest.raises(co.ConsolidateError, match="cannot be deleted"):
        co.episode_id_of(empty)
    with pytest.raises(co.ConsolidateError, match="expected an Episode"):
        co.episode_id_of({"seqs": (1,)})


def test_known_ids_is_derived_from_the_episodes_not_written_down() -> None:
    episodes = sg.replay(_stream())
    assert co.known_ids(episodes) == tuple(co.episode_id(e.seqs[0]) for e in episodes)
    assert len(set(co.known_ids(episodes))) == len(episodes), "ids must be unique"


def test_consolidate_refuses_a_hand_written_resolvable_set() -> None:
    with pytest.raises(co.ConsolidateError, match="not episode ids"):
        co.consolidate([obs("pet", "貓", co.SELF_REPORTED, E1)],
                       episode_ids=["episode-1"])
    with pytest.raises(co.ConsolidateError, match="resolvable episode set is empty"):
        co.consolidate([], episode_ids=[])


def test_an_unresolvable_source_is_refused_at_the_gate() -> None:
    with pytest.raises(co.ConsolidateError, match="does not resolve"):
        co.consolidate([obs("pet", "貓", co.SELF_REPORTED, co.episode_id(77))],
                       episode_ids=KNOWN)


# --------------------------------------------------------------- the report (D)
def test_the_census_carries_every_decision_including_the_zeros() -> None:
    result = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                            episode_ids=KNOWN)
    counts = result.census()
    assert list(counts) == list(co.DECISIONS), "census key order is not DECISIONS order"
    assert counts["one_episode"] == 1
    assert counts["accepted"] == 0 and counts["inference_not_allowed"] == 0
    assert sum(counts.values()) == len(result.decisions)


def test_the_report_is_json_serialisable_and_stable() -> None:
    """§0 rule 6: the numbers a night's pass reports have to leave as data."""
    args = ([obs("likes", "芫茜", co.SELF_REPORTED, E1),
             obs("pet", "貓", co.SELF_REPORTED, E1),
             obs("pet", "貓", co.USER_CONFIRMED, E2)], )
    first = co.consolidate(*args, episode_ids=KNOWN).as_dict()
    second = co.consolidate(*args, episode_ids=KNOWN).as_dict()
    assert json.dumps(first, ensure_ascii=False) == json.dumps(second, ensure_ascii=False)
    assert first["census"]["accepted"] == 1
    assert [d["decision"] for d in first["decisions"]] == ["one_episode", "accepted"]


def test_a_decision_is_immutable_and_its_set_is_closed() -> None:
    d = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                       episode_ids=KNOWN).decisions[0]
    with pytest.raises(co.ConsolidateError, match="record of a past pass"):
        d.decision = "accepted"
    with pytest.raises(co.ConsolidateError, match="unknown decision"):
        co.Decision("likes", "芫茜", "probably", (), 0, None)
    assert co.ACCEPTED in co.DECISIONS


def test_a_consolidation_is_immutable() -> None:
    result = co.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1)],
                            episode_ids=KNOWN)
    with pytest.raises(co.ConsolidateError, match="record of a past pass"):
        result.facts = ()


def test_the_order_of_facts_follows_the_order_of_the_candidates() -> None:
    """Determinism one layer up from T-B4: same evidence, same facts, same order."""
    pairs = [("pet", "貓"), ("language", "粵語"), ("routine", "十九點後在家")]
    candidates = [obs(k, v, co.SELF_REPORTED, e) for k, v in pairs for e in (E1, E2)]
    result = co.consolidate(candidates, episode_ids=KNOWN)
    assert [(f.kind, f.value) for f in result.facts] == pairs


def test_consolidate_never_emits_a_fact_the_profile_module_would_refuse() -> None:
    """The two rule sets are in different files; this is the seam that must hold."""
    candidates = [obs(kind, f"value-{kind}", co.SELF_REPORTED, e)
                  for kind in PREDICATES for e in (E1, E2)]
    result = co.consolidate(candidates, episode_ids=KNOWN)
    assert len(result.facts) == len(PREDICATES)
    for fact in result.facts:
        assert Fact(fact.kind, fact.value, fact.trust,
                    fact.source_episode_ids).as_dict() == fact.as_dict()
    assert co.provenance_findings(result.facts, KNOWN) == []


# ========================================================== named mutants (L3)
def _stream() -> list[dict]:
    """Three episodes, split by activity changes, with store sequence numbers.

    Sequence numbers deliberately do not start at 1 and do not equal the list
    positions: an id scheme that read a position would otherwise produce the same
    strings as one that read a sequence, and every identity assertion below would be
    green under both.
    """
    return [
        {"seq": 11, "kind": "turn", "ts": 1.0, "activity": "conversation"},
        {"seq": 12, "kind": "turn", "ts": 2.0, "activity": "conversation"},
        {"seq": 13, "kind": "game_round", "ts": 3.0, "activity": "game"},
        {"seq": 14, "kind": "game_round", "ts": 4.0, "activity": "game"},
        {"seq": 15, "kind": "turn", "ts": 5.0, "activity": "conversation"},
        {"seq": 16, "kind": "turn", "ts": 6.0, "activity": "conversation"},
    ]


@pytest.mark.mutation
def test_no_mutant_control(tmp_path) -> None:
    """§0 rule 2: a harness that cannot import a clean copy kills every mutant."""
    clean = load_mutant(tmp_path, "control")
    assert clean.MIN_EPISODES == co.MIN_EPISODES
    assert clean.INFERENCE_ALLOWED == co.INFERENCE_ALLOWED
    assert clean.DECISIONS == co.DECISIONS
    assert clean.consolidate([obs("likes", "芫茜", clean.SELF_REPORTED, E1)],
                             episode_ids=KNOWN).facts == ()
    landed = clean.consolidate([obs("likes", "芫茜", clean.SELF_REPORTED, E1),
                                obs("likes", "芫茜", clean.SELF_REPORTED, E2)],
                               episode_ids=KNOWN)
    assert len(landed.facts) == 1


@pytest.mark.mutation
def test_mutant_n_equals_one_lands_a_single_mention(tmp_path) -> None:
    """T-C2's own mutant: N = 1, which is what a "just store it" version does."""
    mutant = load_mutant(tmp_path, "n1", (MIN_N, "MIN_EPISODES = 1"))
    once = [obs("likes", "芫茜", co.SELF_REPORTED, E1)]
    assert len(mutant.consolidate(once, episode_ids=KNOWN).facts) == 1
    assert co.consolidate(once, episode_ids=KNOWN).facts == ()


@pytest.mark.mutation
def test_mutant_counting_mentions_lands_one_episode_said_twice(tmp_path) -> None:
    """T-C2's second mutant. Killed by one fixture shape, not by a smaller total.

    The mutant agrees with the real module on every input where mentions and
    occasions coincide, so the assertion has to be about the shape that separates
    them — and that shape is asserted to be present rather than assumed.
    """
    mutant = load_mutant(tmp_path, "mentions", (
        DISTINCT, "        episodes = tuple(o.episode for o in counted)\n"))
    same = [obs("likes", "芫茜", co.SELF_REPORTED, E1),
            obs("likes", "芫茜", co.SELF_REPORTED, E1)]
    assert len({o["episode"] for o in same}) == 1, "the fixture must be one occasion"
    assert len(mutant.consolidate(same, episode_ids=KNOWN).facts) == 1
    assert co.consolidate(same, episode_ids=KNOWN).facts == ()
    # And it is not a mutant that accepts everything: one mention is still refused,
    # which is why "fewer facts appeared" could not have caught it.
    assert mutant.consolidate(same[:1], episode_ids=KNOWN).facts == ()
    # The stored provenance is the visible damage — a duplicated id in a field the
    # deletion path iterates.
    fact = mutant.consolidate(same, episode_ids=KNOWN).facts[0]
    assert fact.source_episode_ids == (E1, E1)
    assert co.provenance_findings([fact], KNOWN), "duplicated sources must be a finding"


@pytest.mark.mutation
def test_mutant_defaulting_the_inference_table_lets_a_new_predicate_in(tmp_path) -> None:
    """`INFERENCE_ALLOWED.get(kind, True)` — the default that decides for you."""
    mutant = load_mutant(tmp_path, "infer_default", (
        INFERENCE_READ, "        if not INFERENCE_ALLOWED.get(kind, True):\n"),
        ('    "likes": False,           # plan §0-6: the open door in the closed set\n',
         ""))
    inferred = [obs("likes", "芫茜", co.INFERRED, E1),
                obs("likes", "芫茜", co.INFERRED, E2)]
    assert len(mutant.consolidate(inferred, episode_ids=KNOWN).facts) == 1
    assert co.consolidate(inferred, episode_ids=KNOWN).facts == ()
    # The key-set assertion is the other half: with the row deleted the table no
    # longer rules on every predicate, and that is a red test rather than silence.
    assert set(mutant.INFERENCE_ALLOWED) != set(PREDICATES)


@pytest.mark.mutation
def test_mutant_promoting_trust_by_count_confirms_a_repeated_guess(tmp_path) -> None:
    """Trust is a kind of evidence. Three repetitions of a mishearing are not consent."""
    mutant = load_mutant(tmp_path, "trust_count", (
        TRUST_RULE,
        '    level = "confirmed" if (any(c in CONFIRMING_CHANNELS for c in channels) '
        'or len(channels) >= 3) else "observed"\n'))
    thrice = [obs("pet", "貓", co.SELF_REPORTED, e) for e in (E1, E2, E3)]
    assert mutant.consolidate(thrice, episode_ids=KNOWN).facts[0].trust == "confirmed"
    assert co.consolidate(thrice, episode_ids=KNOWN).facts[0].trust == "observed"
    # Still not a mutant that confirms everything: two episodes stay observed, so a
    # test that only checked "confirmed exists somewhere" would have missed it.
    assert co.consolidate(thrice[:2], episode_ids=KNOWN).facts[0].trust == "observed"
    assert mutant.consolidate(thrice[:2], episode_ids=KNOWN).facts[0].trust == "observed"


@pytest.mark.mutation
def test_mutant_position_ids_repoint_a_fact_at_the_wrong_episode(tmp_path) -> None:
    """The identity mutant, and the reason T-C3's two criteria are not enough.

    Delete the first episode and re-segment. The real scheme leaves the deleted
    episode's id resolving to nothing — a dangling reference the cascade can find.
    The position scheme renumbers, so the same id now resolves *successfully* to a
    different episode: nothing raises, nothing is missing, and the fact has acquired
    a source that never produced it. That is the failure a deletion path cannot
    detect, which is why the id is a sequence number.
    """
    mutant = load_mutant(tmp_path, "position", (
        ID_FROM_SEQ, "    return episode_id(episode.indices[0] + 1)\n"))
    events = _stream()
    before = sg.replay(events)
    assert len(before) == 3, [e.reason for e in before]

    # A fact sourced to the first two episodes, under each scheme.
    real_ids = co.known_ids(before)
    mutant_ids = mutant.known_ids(before)
    assert real_ids != mutant_ids, "the two schemes must differ on this fixture"

    surviving = events[2:]                     # episode 1's records are deleted
    after = sg.replay(surviving)
    assert len(after) == 2

    # Real scheme: the deleted episode's id resolves to nothing, and the surviving
    # episodes keep the ids the fact already cites.
    assert real_ids[0] not in co.known_ids(after)
    assert co.known_ids(after) == real_ids[1:]
    dangling = Fact("pet", "貓", "observed", tuple(sorted(real_ids[:2])))
    assert co.provenance_findings([dangling], co.known_ids(after)), \
        "a deleted episode must leave a finding"

    # Position scheme: the id set is unchanged by the deletion, so the fact's first
    # source silently now means the episode that used to be second.
    assert mutant.known_ids(after)[0] == mutant_ids[0]
    repointed = Fact("pet", "貓", "observed", tuple(sorted(mutant_ids[:2])))
    assert mutant.provenance_findings([repointed], mutant.known_ids(after)) == [], \
        "the mutant reports a clean card after deleting the source"


@pytest.mark.mutation
def test_mutant_widening_the_id_breaks_the_sort_that_provenance_depends_on(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "widen", (ID_WIDTH_GUARD, "    if False:\n"))
    big = mutant.episode_id(co.MAX_EPISODE_SEQ + 1)
    small = mutant.episode_id(co.MAX_EPISODE_SEQ)
    assert sorted([big, small]) == [big, small], "the mutant sorts 1000000 before 999999"
    with pytest.raises(co.ConsolidateError):
        co.episode_id(co.MAX_EPISODE_SEQ + 1)
    # And the widened id is not even recognised as one, so it would be stored and
    # then reported as never minted — two inconsistent readings of the same value.
    assert not co._looks_like_id(big)


@pytest.mark.mutation
def test_mutant_prefix_only_id_check_accepts_the_bare_family(tmp_path) -> None:
    """§S4.5's lesson, one module later: a prefix is not a closed set."""
    mutant = load_mutant(tmp_path, "prefix",
                         (LOOKS_LIKE_WIDTH, "    if False:\n        return False\n"),
                         (LOOKS_LIKE_ROUNDTRIP, "    return True\n"))
    for bogus in ("ep-", "ep-abcdef", "ep-00000٤", "ep-0000001"):
        assert mutant._looks_like_id(bogus), bogus
        assert not co._looks_like_id(bogus), bogus
    assert co.provenance_findings([Fact("pet", "貓", "observed", ("ep-abcdef",))],
                                 KNOWN)
    assert mutant.provenance_findings([Fact("pet", "貓", "observed", ("ep-abcdef",))],
                                      KNOWN)[0].endswith("does not resolve")


@pytest.mark.mutation
def test_mutant_dropping_the_resolve_check_accepts_a_source_that_is_gone(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "resolve", (RESOLVE_CHECK, "            elif False:\n"))
    fact = Fact("pet", "貓", "observed", (co.episode_id(999),))
    assert mutant.provenance_findings([fact], KNOWN) == []
    assert co.provenance_findings([fact], KNOWN)
    # Not a mutant that reports nothing at all: a missing source is still a finding,
    # so "the findings list is non-empty somewhere" could not have caught it.
    assert mutant.provenance_findings([Fact("name", "Tim", TRUST_SEED, (E1,))], KNOWN)


@pytest.mark.mutation
def test_mutant_normalising_the_value_stores_one_row_for_two_counts(tmp_path) -> None:
    """Refuse rather than strip: the grouping key has to be the stored bytes."""
    mutant = load_mutant(tmp_path, "strip", (UNSTRIPPED_REFUSAL, "        if False:\n"))
    mixed = [obs("likes", "芫茜", co.SELF_REPORTED, E1),
             obs("likes", "芫茜 ", co.SELF_REPORTED, E1)]
    landed = mutant.consolidate(mixed, episode_ids=KNOWN)
    assert landed.facts == (), "two groups of one, under the mutant"
    assert len(landed.decisions) == 2
    assert {d.value for d in landed.decisions} == {"芫茜", "芫茜 "}
    # …and the damage: the same evidence spread over two episodes becomes two facts
    # whose stored values are byte-identical, because Fact strips.
    two = mutant.consolidate([obs("likes", "芫茜", co.SELF_REPORTED, E1),
                              obs("likes", "芫茜 ", co.SELF_REPORTED, E1),
                              obs("likes", "芫茜", co.SELF_REPORTED, E2),
                              obs("likes", "芫茜 ", co.SELF_REPORTED, E2)],
                             episode_ids=KNOWN)
    assert len(two.facts) == 2
    assert two.facts[0].value == two.facts[1].value, "one row, stored twice"
    with pytest.raises(co.ConsolidateError, match="surrounding whitespace"):
        co.consolidate(mixed, episode_ids=KNOWN)


@pytest.mark.mutation
def test_mutant_census_without_the_zeros_hides_a_night_of_refusals(tmp_path) -> None:
    """A pass that reports only what it wrote makes "quiet" and "all refused" one word."""
    mutant = load_mutant(
        tmp_path, "census", (CENSUS_ZEROS, "        counts = {}\n"),
        (CENSUS_COUNT,
         "            counts[d.decision] = counts.get(d.decision, 0) + 1\n"))
    refused = [obs("likes", "芫茜", co.SELF_REPORTED, E1)]
    assert mutant.consolidate(refused, episode_ids=KNOWN).census() == {"one_episode": 1}
    assert co.consolidate(refused, episode_ids=KNOWN).census()["accepted"] == 0
    # A night with nothing proposed and a night with everything refused are the same
    # dict under the mutant once the refusal class is absent from both.
    assert mutant.consolidate([], episode_ids=KNOWN).census() == {}


@pytest.mark.mutation
def test_mutant_ignoring_an_extra_field_accepts_a_rule_nobody_wrote(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "extra", (EXTRA_FIELD, "    extra = []\n"))
    record = {**obs("likes", "芫茜", co.SELF_REPORTED, E1), "mentions": 7}
    assert mutant.observation(record).as_dict()["value"] == "芫茜"
    assert "mentions" not in mutant.observation(record).as_dict()
    with pytest.raises(co.ConsolidateError, match="unexpected field"):
        co.observation(record)
