"""Offline tests for memory/erase.py — T-C4's deletion action, plus retention.

Design item T-C4 is one row in docs/test-design.md and two halves in the code. The
evidence half shipped with memory/recall.py (`deletion_findings`, the five path
statuses); this file tests the half that removes things, and the design's own wording
is what shapes it:

> 五條路徑各 **0 命中**;tombstone 存在;只剩 E 的事實消失、有第二來源的事實保留並移除
> E。**fixture 必須保證兩個分支都存在(用計數斷言),否則測試會空綠**

Three things here are not what the row asked for, and the reasons are the point.

**The fixture proves the query would have hit.** "Five paths, zero hits" is satisfied
by asking a question the store never had an answer to, so every end-to-end test here
queries **before** the deletion and asserts a non-zero count per path first. The
queries come from `erase.verification_queries`, which is a function rather than a
convention because the text it needs stops existing the moment the deletion runs — a
receipt could not hand it back without becoming a copy of the deleted content.

**"Tombstone exists" is upgraded to a store-wide invariant.** A test that finds a
tombstone proves the writer wrote one. `erase.tombstone_findings` re-derives, for every
tombstone in the audit log, that the episode is still absent from all four artifacts —
which is what catches a crash between the two writes and a backup restored over a
deletion. The converse ("every gap has a tombstone") is *not* checked and not claimed:
sequence numbers are shared by both streams and consumed by snapshot and rewrite
operations that write no record, so a gap has four innocent causes.

**Retention gets a boundary test and no threshold test.** `RETENTION_DAYS = 90` is a
policy default with no measurement and no legal citation behind it, so asserting on 90
would be asserting on a guess. What is asserted is the mechanism: the boundary is exact
at one-second resolution from both sides, and an episode that cannot be dated is
reported rather than silently kept or silently swept.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ERASE_PY = REPO_ROOT / "memory" / "erase.py"

from memory import consolidate as cs  # noqa: E402
from memory import erase as er  # noqa: E402
from memory import profile as pf  # noqa: E402
from memory import recall as rc  # noqa: E402
from memory import store as st  # noqa: E402

E = cs.episode_id(1)          # the episode every test deletes
OTHER = cs.episode_id(20)     # the one that must survive it
ANSWER = "user_requested_answer"
WHY = {"goal": ANSWER}


# --------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/erase.py.

    Same shape as tests/test_store.py's, including the uniqueness assertion: an
    ambiguous anchor would patch one of two sites and the surviving one would make
    the mutant look killed. Zero replacements is the no-mutant control (§6.0 rule 2).
    """
    dst = tmp_path / "erase_mutant"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "erase.py"
    src = ERASE_PY.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer in erase.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src)
    name = f"erase_mutant_{tmp_path.name}_{len(replacements)}"
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


# -------------------------------------------------------------------- fixtures
#
# Written through the real `Store`, not planted as files. Sequence numbers are what
# `rewrite_stream` addresses and what `consolidate.episode_id` derives identity from,
# so a fixture that invents them would test a store whose counter disagrees with its
# own files — and the tombstone's event references would be fiction.
def build(tmp_path: Path, *, card=True, rps=True) -> "st.Store":
    """A store holding two episodes' worth of everything, E's and OTHER's.

    Both episodes appear in every artifact, so each assertion has something that must
    go and something that must stay. A fixture with only E cannot tell a deletion from
    a truncation.
    """
    store = st.Store(tmp_path / "data", _allow_ephemeral=True)
    mine, theirs = [], []
    for text in ("我叫 Tim 你記住", "我住新加坡", "我喜歡喝烏龍茶"):
        mine.append(store.append("events", {"kind": "turn", "text": text, "why": WHY}))
    for text in ("今天天氣很好", "我們玩猜拳吧"):
        theirs.append(store.append("events", {"kind": "turn", "text": text,
                                              "why": WHY}))
    # One guard record, so `audit.jsonl` exists on disk. Without it the file is
    # absent, `registry_findings` has nothing to classify and `coverage_findings`
    # nothing to report — both would be green for the same reason an empty scan is,
    # and the retained-stream assertions would be testing a file that never existed.
    store.append("audit", {"kind": "guard", "decision": "allow",
                           "rule": "length_cap", "ref": "events#1"})
    store.flush()
    episodes = [
        {"id": E, "seqs": mine, "summary": "使用者自我介紹:名字 Tim,住新加坡,愛烏龍茶",
         "embedding": [1.0, 0.0, 0.0], "start_ts": 100.0, "end_ts": 200.0},
        {"id": OTHER, "seqs": theirs, "summary": "閒聊天氣後開始玩猜拳",
         "embedding": [0.0, 1.0, 0.0], "start_ts": 300.0, "end_ts": 400.0},
    ]
    store.write_snapshot(rc.EPISODES_ARTIFACT, {"version": 1, "episodes": episodes})
    if card:
        # Four facts, one per branch the card eraser has to take, and
        # `test_the_card_fixture_contains_both_branches` counts them so this stays
        # true if somebody edits the list.
        st.unlock_for_s5(rc.CARD_ARTIFACT)
        store.write_snapshot(rc.CARD_ARTIFACT, {"version": 1, "facts": [
            {"kind": "name", "value": "Tim", "trust": "seed",
             "source_episode_ids": []},                                # seed: untouched
            {"kind": "location", "value": "新加坡", "trust": "observed",
             "source_episode_ids": [E]},                               # goes whole
            {"kind": "likes", "value": "烏龍茶", "trust": "confirmed",
             "source_episode_ids": [E, OTHER]},                        # loses one cite
            {"kind": "language", "value": "粵語", "trust": "observed",
             "source_episode_ids": [OTHER]},                           # untouched
        ]})
    if rps:
        store.write_snapshot(rc.RPS_ARTIFACT, {"version": 1, "rounds": [
            {"user": "rock", "outcome": "loss", "episode": E},
            {"user": "paper", "outcome": "win", "episode": E},
            {"user": "scissors", "outcome": "draw", "episode": OTHER},
            {"user": "rock", "outcome": "win"},                        # unattributed
        ]})
    store.flush()
    return store


def encoder(text: str):
    """A deterministic stand-in for gte-multilingual-base.

    One-hot on a keyword, so a cosine is exactly 1.0 or 0.0 and no assertion here is
    inside a float tolerance that a reversed comparison would still satisfy. It has to
    be able to *hit* — an encoder that returned a constant would make every semantic
    assertion pass before the deletion and after it.
    """
    if "烏龍" in text or "Tim" in text or "新加坡" in text:
        return [1.0, 0.0, 0.0]
    if "猜拳" in text or "天氣" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class Census:
    """What every retrieval path returns about one episode, counted per path.

    Per path rather than in total, because "the store still holds E" and "this
    particular path still holds E" are different claims and only the second one says
    which artifact was missed.

    `orphan` is counted separately from `named`, and that separation is what corrects
    the design row. T-C4 says "五條路徑各 0 命中", and I first wrote the criterion as
    `deletion_findings == []` — which is unreachable on a realistic store. An
    unattributed rps round is a real state of the real system: `game_node` sends each
    round to `memory_node` fire-and-forget, so a round played while no episode is open
    is recorded with no episode, permanently. `deletion_findings` reports every such
    item, correctly — an orphan is the cheapest way to fake a cascade — but a
    *pre-existing* orphan is not evidence about this deletion. Demanding an empty list
    would only be satisfiable by a fixture with no unattributed round, i.e. by deleting
    the case `_plan_rps` exists to handle.

    So the criterion is three separate claims: no hit names the episode, no path
    failed to run, and **the orphan count per path did not grow**. The third is the
    one that catches an erasure that removes the episode entry and leaves its events
    behind, which is the defect the row's own wording was reaching for.
    """

    __slots__ = ("named", "orphan", "findings")

    def __init__(self, root: Path, queries, episode: str = E) -> None:
        self.named = {name: 0 for name in rc.paths()}
        self.orphan = {name: 0 for name in rc.paths()}
        self.findings: list[str] = []
        ret = rc.Retriever(root, encoder=encoder, deadline_s=60.0)
        for query in queries:
            result = ret.query(query)
            self.findings.extend(rc.deletion_findings(result, episode))
            for hit in result.hits(k=None):
                if hit.episode == episode:
                    self.named[hit.path] += 1
                elif hit.episode == rc.ORPHAN:
                    self.orphan[hit.path] += 1

    @property
    def reached_every_path(self) -> bool:
        return all(self.named[name] > 0 for name in rc.paths())

    def assert_erased(self, before: "Census", episode: str = E) -> None:
        assert self.named == {name: 0 for name in rc.paths()}, self.named
        assert [f for f in self.findings if episode in f] == [], self.findings
        assert [f for f in self.findings if "did not run" in f] == [], self.findings
        assert self.orphan == before.orphan, (
            f"the deletion created orphans: {before.orphan} → {self.orphan}")


@pytest.fixture(autouse=True)
def _relock_the_card():
    """Undo `unlock_for_s5` after every test.

    The unlock is process-global module state, so a test that leaves it set would make
    `test_a_locked_card_refuses_the_deletion_unconditionally` pass or fail depending on
    ordering — the T-A5 guarantee would be disabled by a neighbour.
    """
    yield
    st._UNLOCKED.clear()


# ------------------------------------------------------- the registry (D)
def test_every_eraser_is_registered_and_every_registration_has_an_eraser():
    """Both directions, derived from the source rather than from a written list.

    A `_plan_*` function with no registration is unreachable code that looks like
    coverage; a registration with no function is an `AttributeError` on the first
    deletion. Same derivation `recall.py` uses for its `_run_<path>` methods.
    """
    planners = {name for name in vars(er) if name.startswith("_plan_")}
    registered = {fn.__name__ for fn in er.ERASERS.values()}
    assert planners == registered, (
        f"unregistered planners {sorted(planners - registered)}, "
        f"missing planners {sorted(registered - planners)}")
    assert len(set(er.ERASERS.values())) == len(er.ERASERS), \
        "two artifacts share a planner, so one of them is erased by accident"


def test_a_realistic_root_is_fully_classified(tmp_path):
    """No unclassified artifact on a root that holds everything a store writes.

    This is the check that could not come from `recall.PATHS`: `audit.jsonl` is on
    disk, no retrieval path reads it, and it must be a written decision rather than an
    omission. `coverage_findings` already reports it as uncovered *for retrieval* —
    the two functions disagreeing on the same file is the point.
    """
    store = build(tmp_path)
    try:
        assert er.registry_findings(store.root) == []
        assert any("audit.jsonl" in f for f in rc.coverage_findings(store.root)), (
            "audit.jsonl is no longer retrieval-uncovered, so the reason the erasure "
            "registry is derived from the disk needs re-deriving")
        assert "audit.jsonl" in er.RETAINED
    finally:
        store.close()


def test_a_new_artifact_forces_a_decision(tmp_path):
    """A file a future node writes must not be silently walked past."""
    store = build(tmp_path)
    try:
        (store.root / "skills_used.json").write_text("{}", encoding="utf-8")
        findings = er.registry_findings(store.root)
        assert any("skills_used.json" in f for f in findings), findings
        with pytest.raises(er.EraseError, match="unclassified"):
            er.delete(store, E, goal=er.FORGET_GOAL)
    finally:
        store.close()


def test_registry_findings_reports_a_readable_artifact_with_no_eraser(tmp_path,
                                                                     monkeypatch):
    """The other direction: reachable by a query and touched by no erasure.

    Worse than merely uncovered, because a retrieval path returns it. Simulated by
    removing an eraser rather than by inventing a path, so the assertion is about the
    join and not about a fixture.
    """
    store = build(tmp_path)
    try:
        monkeypatch.delitem(er.ERASERS, rc.CARD_ARTIFACT)
        findings = er.registry_findings(store.root)
        assert any(rc.CARD_ARTIFACT in f and "no eraser" in f for f in findings), \
            findings
    finally:
        store.close()


def test_the_attribution_artifact_is_erased_last():
    """Alphabetical order gets this exactly wrong, which is why it is a function."""
    order = er.erase_order()
    assert set(order) == set(er.ERASERS)
    assert order[-1] == er.ATTRIBUTION_ARTIFACT
    assert sorted(er.ERASERS)[0] == er.ATTRIBUTION_ARTIFACT, (
        "episodes.json no longer sorts first, so the plain-`sorted()` mutant is no "
        "longer the defect this ordering exists to avoid — re-derive it")


def test_a_deletion_goal_cannot_be_defined_without_being_a_store_goal():
    """The check is inside the constant's creator, so the creator is what gets tested.

    A goal this module names that `store.GOALS` does not contain makes every tombstone
    unappendable, and the first person to discover that would be somebody asking to be
    forgotten. Asserting only that today's two members are legal would leave the
    mechanism untested — the third goal somebody adds next year is the one at risk.
    """
    for goal in er.DELETION_GOALS:
        assert goal in st.GOALS
    with pytest.raises(er.EraseError, match="not a member of memory.store.GOALS"):
        er._deletion_goal("user_requested_forgett")
    # And it is an assignment, not a loop: `infra/entrypoints.py` counts a top-level
    # statement that is not a declaration as import-time work, which would classify
    # this library as a runnable entrypoint nothing launches (check 8). Asserted here
    # rather than left to the deploy guard, because the guard's message names the file
    # and not the reason.
    import ast
    body = ast.parse(ERASE_PY.read_text()).body
    work = [n.lineno for n in body
            if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                  ast.ClassDef, ast.Assign, ast.AnnAssign))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert work == [], f"import-time work at lines {work} makes this file an entrypoint"


# ------------------------------------------------------- the end-to-end T-C4 (D)
def test_deleting_an_episode_empties_all_five_paths_that_held_it(tmp_path):
    """T-C4 proper, with the "would have hit" half asserted first.

    Every path is checked for a non-zero count **before** the deletion, so a green
    result afterwards cannot come from asking the wrong question. The criterion after
    it is `Census.assert_erased`, which is three claims rather than one — see `Census`
    for why an empty `deletion_findings` is the wrong bar.
    """
    store = build(tmp_path)
    try:
        queries = er.verification_queries(store.root, E)
        before = Census(store.root, queries)
        assert before.reached_every_path, (
            f"the fixture does not reach every path, so a clean result after the "
            f"deletion would be vacuous: {before.named}")

        receipt = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()

        Census(store.root, queries).assert_erased(before)
        # OTHER survives every one of them. A deletion that emptied the store would
        # satisfy everything above.
        survivors = rc.Retriever(store.root, encoder=encoder,
                                 deadline_s=60.0).query("猜拳").hits(k=None)
        assert any(h.episode == OTHER for h in survivors), survivors
        # One finding, and it is the kept unattributed round — not an absence of
        # findings. Same correction as `Census`: the receipt for a *correct* deletion
        # on a realistic store is not empty, because one thing was deliberately not
        # removed and saying so is the whole design.
        assert len(receipt.findings) == 1, receipt.findings
        assert "name no episode" in receipt.findings[0]
        assert receipt.total == 8, receipt.as_dict()
    finally:
        store.close()


def test_the_receipt_counts_come_from_the_disk_not_from_the_plan(tmp_path):
    """Per artifact, and each count derived separately from the file it describes."""
    store = build(tmp_path)
    try:
        receipt = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        assert receipt.removed[rc.EVENTS_ARTIFACT] == (1, 2, 3)
        assert receipt.removed[rc.EPISODES_ARTIFACT] == (E,)
        assert receipt.count(rc.CARD_ARTIFACT) == 2
        assert receipt.count(rc.RPS_ARTIFACT) == 2
        # And the files agree, read independently of the receipt.
        assert [e["seq"] for e in rc.load_events(store.root)] == [4, 5]
        warm = rc.load_episodes(store.root / rc.EPISODES_ARTIFACT)
        assert [e["id"] for e in warm] == [OTHER]
        assert len(rc.load_rounds(store.root / rc.RPS_ARTIFACT)) == 2
    finally:
        store.close()


def test_the_tombstone_names_ids_and_counts_and_quotes_no_deleted_text(tmp_path):
    """The closed field set, checked against the file the erasure may never shorten.

    `audit.jsonl` is retained, so any user text quoted in a tombstone is the one copy
    no future deletion can reach — the erasure would be undone by its own receipt. The
    assertion is a substring search for the deleted sentences over the whole audit
    file, not a field-name check, because a field named anything at all would carry it.
    """
    store = build(tmp_path)
    try:
        texts = [e["text"] for e in rc.load_events(store.root)
                 if e["seq"] in (1, 2, 3)]
        assert texts, "fixture has no text to look for"
        receipt = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()

        audit = (store.root / "audit.jsonl").read_text(encoding="utf-8")
        for text in texts:
            assert text not in audit, f"the tombstone quotes deleted text: {text!r}"
        assert "使用者自我介紹" not in audit, "the tombstone quotes the summary"

        stones = er.tombstones(store.root)
        assert len(stones) == 1
        stone = stones[0]
        assert stone["episode"] == E
        assert stone["seq"] == receipt.tombstone_seq
        assert stone["why"] == {"goal": er.FORGET_GOAL}
        assert set(stone) <= set(er.TOMBSTONE_FIELDS)
        assert stone["removed"][rc.EVENTS_ARTIFACT] == [1, 2, 3]
    finally:
        store.close()


@pytest.mark.parametrize("record,needle", [
    ({"kind": "tombstone", "episode": E, "why": {"goal": er.FORGET_GOAL},
      "removed": {}, "seq": 1, "ts": 0.0, "text": "我喜歡喝烏龍茶"}, "unknown field"),
    ({"kind": "tombstone", "episode": "nope", "why": {}, "removed": {},
      "seq": 1, "ts": 0.0}, "not a minted id"),
    ({"kind": "tombstone", "episode": E, "why": {}, "removed": [], "seq": 1,
      "ts": 0.0}, "must be an object"),
    ({"kind": "tombstone", "episode": E, "why": {}, "seq": 1, "ts": 0.0,
      "removed": {"secrets.json": []}}, "no eraser"),
    ({"kind": "tombstone", "episode": E, "why": {}, "seq": 1, "ts": 0.0,
      "removed": {rc.EVENTS_ARTIFACT: ["1"]}}, "sequence numbers"),
])
def test_a_tombstone_is_validated_on_the_way_back_in(record, needle):
    """Read-side validation, so a future writer that adds a field reds here.

    A write-side-only check protects the writer that exists today. The excerpt field a
    later maintainer adds to make a receipt more helpful is exactly the change this has
    to refuse, and it will not go through this module's builder.
    """
    with pytest.raises(er.EraseError, match=needle):
        er.parse_tombstone(record, where="<test>")


def test_the_audit_stream_is_not_shortened_by_a_deletion(tmp_path):
    """`RETAINED`, observed rather than asserted from the constant.

    Two deletions, and the first tombstone still there after the second — which is the
    property that makes a tombstone worth writing at all.
    """
    store = build(tmp_path)
    try:
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        first = (store.root / "audit.jsonl").read_text(encoding="utf-8")
        er.delete(store, OTHER, goal=er.RETENTION_GOAL)
        store.flush()
        second = (store.root / "audit.jsonl").read_text(encoding="utf-8")
        assert second.startswith(first), "the audit log was rewritten, not appended to"
        assert len(er.tombstones(store.root)) == 2
    finally:
        store.close()


def test_every_receipt_states_what_the_erasure_does_not_reach(tmp_path):
    """`retained` on the receipt, not only in a docstring.

    An omission documented once is an omission the next reader of a green receipt does
    not see, and the specific omission here is load-bearing: nothing enforces that an
    audit record omits user text, so "deleted" does not yet mean "no trace".
    """
    store = build(tmp_path)
    try:
        receipt = er.delete(store, E, goal=er.FORGET_GOAL)
        assert "audit.jsonl" in receipt.retained
        assert receipt.as_dict()["retained"]["audit.jsonl"]
        assert any("audit.jsonl" in line for line in er.retained_findings())
    finally:
        store.close()


# --------------------------------------------------------- the card, both branches
def test_the_card_fixture_contains_both_branches(tmp_path):
    """The counting assertion the design row demands, on the fixture itself.

    Without it the card test is green whenever the fixture happens to hold only
    single-source facts — the whole test would be reporting on a branch that never ran.
    """
    store = build(tmp_path)
    try:
        card = pf.load(store.root / rc.CARD_ARTIFACT)
        only_e = [f for f in card.facts if list(f.source_episode_ids) == [E]]
        shared = [f for f in card.facts
                  if E in f.source_episode_ids and len(f.source_episode_ids) > 1]
        seeds = [f for f in card.facts if f.trust == pf.TRUST_SEED]
        untouched = [f for f in card.facts
                     if f.source_episode_ids and E not in f.source_episode_ids]
        assert len(only_e) == 1 and len(shared) == 1
        assert len(seeds) == 1 and len(untouched) == 1
    finally:
        store.close()


def test_a_fact_sourced_only_to_the_episode_goes_and_a_shared_one_keeps_its_source(
        tmp_path):
    """Both branches, and getting either wrong is a defect with no symptom.

    Dropping the shared fact deletes something nobody asked to have deleted; keeping
    the single-source one leaves an assertion about a person whose only evidence is
    gone. Neither shows up as an error anywhere.
    """
    store = build(tmp_path)
    try:
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        card = pf.load(store.root / rc.CARD_ARTIFACT)
        by_kind = {f.kind: f for f in card.facts}
        assert "location" not in by_kind, "a fact sourced only to E survived"
        assert "likes" in by_kind, "a fact with a second source was deleted"
        assert list(by_kind["likes"].source_episode_ids) == [OTHER]
        assert by_kind["name"].trust == pf.TRUST_SEED
        assert list(by_kind["language"].source_episode_ids) == [OTHER]
    finally:
        store.close()


@pytest.mark.mutation
def test_the_always_remove_the_whole_fact_mutant_deletes_a_second_episodes_evidence(
        tmp_path):
    """`if remaining:` → `if False:`. Over-deletion, and it writes successfully.

    Nothing refuses this card: it is a shorter card of legal facts. The only thing that
    notices is the assertion about the *other* episode's fact, which is why the branch
    assertion is written as "likes survives with OTHER" rather than as a fact count.
    """
    module = load_mutant(tmp_path, ("        if remaining:", "        if False:"))
    store = build(tmp_path)
    try:
        module.delete(store, E, goal=module.FORGET_GOAL)
        store.flush()
        kinds = {f.kind for f in pf.load(store.root / rc.CARD_ARTIFACT).facts}
        assert "likes" not in kinds, \
            "the mutant kept the shared fact, so the branch assertion proves nothing"
    finally:
        store.close()


@pytest.mark.mutation
def test_the_never_remove_a_whole_fact_mutant_cannot_even_write_the_card(tmp_path):
    """`if remaining:` → `if True:`. Killed by the reader, one layer down.

    Worth separating from its twin because the *kill mechanism* is different, and the
    difference is a property of `profile.py` rather than of this module: a fact whose
    `source_episode_ids` becomes empty while its trust stays `observed` is refused,
    because "only 'seed' facts may have none". So the mutant that keeps every fact
    produces a card its own reader rejects, and the pre-write re-parse turns that into
    a refusal before a byte is written.

    The two mutants together are what make the branch pair non-vacuous: one is caught
    by an assertion about another episode, the other by an invariant a different module
    creates. A single mutant would have left one of those mechanisms untested.
    """
    module = load_mutant(tmp_path, ("        if remaining:", "        if True:"))
    store = build(tmp_path)
    try:
        before = (store.root / rc.CARD_ARTIFACT).read_bytes()
        with pytest.raises(module.EraseError, match="its own reader refuses"):
            module.delete(store, E, goal=module.FORGET_GOAL)
        store.flush()
        assert (store.root / rc.CARD_ARTIFACT).read_bytes() == before
        assert module.tombstones(store.root) == ()
    finally:
        store.close()


@pytest.mark.mutation
def test_control_an_unmutated_copy_erases_exactly_as_the_real_module_does(tmp_path):
    """The no-mutant control §6.0 rule 2 requires, run every round with the mutants.

    Every mutant in this file is a text-patched copy imported under a fresh module
    name, so the copy itself is a variable: a harness that broke on the copy — a
    relative import that no longer resolves, a module-level check that fires twice —
    would fail every mutant test and read as "all mutants killed". This runs the
    identical path with zero replacements and asserts the deletion is the same one.
    """
    module = load_mutant(tmp_path)
    assert module.erase_order() == er.erase_order()
    store = build(tmp_path)
    try:
        queries = er.verification_queries(store.root, E)
        before = Census(store.root, queries)
        assert before.reached_every_path, before.named
        receipt = module.delete(store, E, goal=module.FORGET_GOAL)
        store.flush()
        Census(store.root, queries).assert_erased(before)
        assert receipt.total == 8, receipt.as_dict()
        assert len(module.tombstones(store.root)) == 1
        assert module.tombstone_findings(store.root) == []
    finally:
        store.close()


def test_a_deletion_that_would_empty_the_card_refuses_before_writing_anything(tmp_path):
    """`profile.parse` refuses an empty card, so this has to refuse first.

    The alternative is a store that will not boot, discovered hours later at start-up
    rather than at the write that caused it. Every artifact is checked unchanged,
    because a refusal after the first write is a half-done deletion.
    """
    store = build(tmp_path, card=False)
    try:
        st.unlock_for_s5(rc.CARD_ARTIFACT)
        store.write_snapshot(rc.CARD_ARTIFACT, {"version": 1, "facts": [
            {"kind": "location", "value": "新加坡", "trust": "observed",
             "source_episode_ids": [E]}]})
        store.flush()
        before = {name: (store.root / name).read_bytes()
                  for name in er.ERASERS if (store.root / name).exists()}
        with pytest.raises(er.EraseError, match="its own reader refuses"):
            er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        for name, body in before.items():
            assert (store.root / name).read_bytes() == body, \
                f"{name} was written before the refusal"
        assert er.tombstones(store.root) == (), \
            "a tombstone claims a deletion that was refused"
    finally:
        store.close()


def test_a_locked_card_refuses_the_deletion_unconditionally(tmp_path):
    """Refused even when no fact cites the episode, and that is the whole point.

    A precondition that fires only when there is something to delete is first tested
    by the deletion that cannot afford to fail. Here the card holds nothing of E's, so
    the deletion would "succeed" — and the next one, on an episode the card does cite,
    would silently leave a fact behind.
    """
    store = build(tmp_path, card=False)
    try:
        st.unlock_for_s5(rc.CARD_ARTIFACT)
        store.write_snapshot(rc.CARD_ARTIFACT, {"version": 1, "facts": [
            {"kind": "language", "value": "粵語", "trust": "observed",
             "source_episode_ids": [OTHER]}]})
        store.flush()
        st._UNLOCKED.clear()                     # back to the S3 seed state
        assert st.is_locked(rc.CARD_ARTIFACT)
        with pytest.raises(er.EraseError, match="read-only seed"):
            er.delete(store, E, goal=er.FORGET_GOAL)
        assert er.tombstones(store.root) == ()
        assert [e["seq"] for e in rc.load_events(store.root)] == [1, 2, 3, 4, 5]
    finally:
        store.close()


# ------------------------------------------------------------------ the rps table
def test_rounds_naming_the_episode_go_and_unattributed_rounds_are_kept_and_reported(
        tmp_path):
    """The one place the erasure keeps something and says why.

    An unattributed round cannot be shown to be E's and cannot be shown not to be.
    Deleting it on suspicion would destroy another episode's behavioural data to make
    this receipt read clean, so it stays and the receipt carries a finding — which is
    the difference between an incomplete deletion and a silent one.
    """
    store = build(tmp_path)
    try:
        receipt = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        rounds = rc.load_rounds(store.root / rc.RPS_ARTIFACT)
        assert [r.get("episode") for r in rounds] == [OTHER, None]
        assert any(rc.RPS_ARTIFACT in f and "name no episode" in f
                   for f in receipt.findings), receipt.findings
    finally:
        store.close()


@pytest.mark.mutation
def test_the_rps_mutant_that_deletes_unattributed_rounds_goes_red(tmp_path):
    """Removing what it cannot prove is E's. Over-deletion, the unrepairable side."""
    module = load_mutant(tmp_path, (
        "        if eid == episode:\n"
        "            refs.append(f\"{recall.RPS_ARTIFACT}#{i}\")\n"
        "            continue",
        "        if eid == episode or eid is None:\n"
        "            refs.append(f\"{recall.RPS_ARTIFACT}#{i}\")\n"
        "            continue"))
    store = build(tmp_path)
    try:
        module.delete(store, E, goal=module.FORGET_GOAL)
        store.flush()
        rounds = rc.load_rounds(store.root / rc.RPS_ARTIFACT)
        assert [r.get("episode") for r in rounds] == [OTHER], (
            "mutant kept the unattributed round, so the test above proves nothing")
    finally:
        store.close()


# ------------------------------------------------ idempotence and crash recovery
def test_a_second_deletion_removes_nothing_and_says_so(tmp_path):
    """Idempotent, not merely repeatable.

    "Already gone" and "never there" are the same observation; only one of them is
    expected, so the second run reports the difference instead of returning a receipt
    identical to the first one's.
    """
    store = build(tmp_path)
    try:
        first = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        second = er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        assert first.total == 8 and second.total == 0
        assert er.tombstone_findings(store.root) == []
        assert len(er.tombstones(store.root)) == 2, \
            "the second deletion left no record that it ran"
    finally:
        store.close()


def test_a_crash_after_the_tombstone_is_self_announcing_and_re_runnable(tmp_path):
    """The recoverable half of the ordering, and why the tombstone goes first.

    Interrupted between the claim and the removals, the audit log says a deletion
    happened that the disk has not performed — `tombstone_findings` reports it and
    re-running the same transaction completes it. The reverse order leaves records gone
    with nothing to say why, which is indistinguishable from corruption.
    """
    store = build(tmp_path)
    try:
        boom = RuntimeError("power cut")

        def crash(step: str) -> None:
            if step == "tombstone":
                raise boom

        with pytest.raises(RuntimeError):
            er.delete(store, E, goal=er.FORGET_GOAL, _after_step=crash)
        store.flush()
        findings = er.tombstone_findings(store.root)
        assert findings, "a tombstone with no removals behind it went unreported"
        assert any(rc.EVENTS_ARTIFACT in f for f in findings), findings

        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        assert er.tombstone_findings(store.root) == []
    finally:
        store.close()


def test_a_crash_before_the_attribution_map_leaves_a_finishable_state(tmp_path):
    """Why `episodes.json` is written last, tested as recoverability.

    Interrupted after the events are gone, the warm layer still names E and still maps
    its remaining references — so the second run finds the same episode and finishes.
    With the map erased first, the surviving events belong to nothing and no later run
    can tell which orphans were E's.
    """
    store = build(tmp_path)
    try:
        queries = er.verification_queries(store.root, E)
        before = Census(store.root, queries)
        assert before.reached_every_path, before.named

        def crash(step: str) -> None:
            if step == rc.RPS_ARTIFACT:
                raise RuntimeError("power cut")

        with pytest.raises(RuntimeError):
            er.delete(store, E, goal=er.FORGET_GOAL, _after_step=crash)
        store.flush()
        warm = rc.load_episodes(store.root / rc.EPISODES_ARTIFACT)
        assert E in {e["id"] for e in warm}, \
            "the attribution map went before the artifacts that depend on it"
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        assert er.tombstone_findings(store.root) == []
        Census(store.root, queries).assert_erased(before)
    finally:
        store.close()


@pytest.mark.mutation
def test_the_alphabetical_order_mutant_orphans_the_events(tmp_path):
    """`sorted(ERASERS)` puts episodes.json first. The defect is what survives.

    After the mutant crashes mid-transaction the events that were E's are attributed
    to nothing, so `deletion_findings` reports orphans — a state no re-run can resolve,
    because the map that named them is gone.
    """
    module = load_mutant(tmp_path, (
        "    rest = tuple(sorted(a for a in ERASERS if a != ATTRIBUTION_ARTIFACT))\n"
        "    return rest + (ATTRIBUTION_ARTIFACT,)",
        "    return tuple(sorted(ERASERS))"))
    assert module.erase_order()[0] == er.ATTRIBUTION_ARTIFACT
    store = build(tmp_path)
    try:
        queries = er.verification_queries(store.root, E)
        before = Census(store.root, queries)

        def crash(step: str) -> None:
            if step == rc.EPISODES_ARTIFACT:
                raise RuntimeError("power cut")

        with pytest.raises(RuntimeError):
            module.delete(store, E, goal=module.FORGET_GOAL, _after_step=crash)
        store.flush()
        warm = rc.load_episodes(store.root / rc.EPISODES_ARTIFACT)
        assert E not in {e["id"] for e in warm}
        after = Census(store.root, queries)
        assert sum(after.orphan.values()) > sum(before.orphan.values()), \
            f"the mutant orphaned nothing: {before.orphan} → {after.orphan}"
        # Derived from the registry, not from a written-down path name: a path added
        # later that reads the event stream is in this claim automatically.
        readers = [n for n, spec in rc.PATHS.items()
                   if rc.EVENTS_ARTIFACT in spec.reads]
        assert readers, "no path reads the event stream; this assertion is vacuous"
        assert any(after.orphan[n] > 0 for n in readers), after.orphan

        # The unrecoverable half, and the reason this is a defect and not a slower
        # route to the same place. Re-running the deletion — which repairs the
        # correct order (see `test_a_crash_before_the_attribution_map_...`) — cannot
        # repair this one: the map that said which events were E's is gone, so the
        # plan finds no sequence numbers, removes nothing, and reports success while
        # the text is still on the disk.
        again = module.delete(store, E, goal=module.FORGET_GOAL)
        store.flush()
        assert again.count(rc.EVENTS_ARTIFACT) == 0, \
            "the re-run found the events, so the map was not needed to identify them"
        final = Census(store.root, queries)
        assert any(final.orphan[n] > 0 for n in readers), (
            f"the orphans healed, so this ordering costs nothing: {final.orphan}")

        # And the one thing that still knows which records those are is the audit log.
        # This is not the outcome I expected to assert here — I first wrote it as "the
        # invariant cannot see it either", and that was wrong: the tombstone carries
        # the sequence numbers, so `tombstone_findings` names them exactly. The cost of
        # the wrong ordering is therefore narrower than "undetectable" and still real:
        # `delete` cannot repair it, because `delete` plans from the map, so the repair
        # needs a person reading the tombstone. Correct order → a re-run finishes;
        # mutant order → an audit finding and a manual fix.
        findings = module.tombstone_findings(store.root)
        assert any(rc.EVENTS_ARTIFACT in f and "still holds seq" in f
                   for f in findings), findings
        assert any(str(seq) in f for f in findings for seq in (1, 2, 3))
    finally:
        store.close()


def test_the_crash_hook_is_reachable_from_tests_only():
    """`_after_step` is an argument, so this is derivable rather than a promise."""
    callers = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "/.git/" in str(path) or path.name.startswith("."):
            continue
        if "_after_step" in path.read_text(encoding="utf-8"):
            callers.append(path.relative_to(REPO_ROOT).as_posix())
    assert callers, "the hook vanished; this test would pass vacuously"
    outside = [c for c in callers
               if not c.startswith("tests/") and c != "memory/erase.py"]
    assert outside == [], f"crash injection is reachable from production: {outside}"


# ---------------------------------------------------------- tombstone invariant
def test_a_restored_backup_over_a_deletion_is_reported(tmp_path):
    """What makes a tombstone more than a comment.

    A file restored from a backup taken before the deletion reintroduces exactly the
    records a person asked to have removed, and nothing in the retrieval path would
    say so — the store looks healthy. The audit log is the only thing that remembers,
    so the invariant is re-derived from it rather than checked once at deletion time.
    """
    store = build(tmp_path)
    try:
        backup = (store.root / rc.EVENTS_ARTIFACT).read_bytes()
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        assert er.tombstone_findings(store.root) == []
        (store.root / rc.EVENTS_ARTIFACT).write_bytes(backup)
        findings = er.tombstone_findings(store.root)
        assert any("still holds seq" in f for f in findings), findings
    finally:
        store.close()


def test_a_card_that_still_cites_a_tombstoned_episode_is_reported(tmp_path):
    """By content, not by the refs the tombstone listed.

    Positions shift when a fact is removed, so a positional re-check would need the
    pre-deletion file. Checking that no fact *cites* the episode is position-free and
    survives any later edit to the card.
    """
    store = build(tmp_path)
    try:
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        card = pf.load(store.root / rc.CARD_ARTIFACT).as_dict()
        card["facts"].append({"kind": "location", "value": "新加坡",
                              "trust": "observed", "source_episode_ids": [E]})
        st.unlock_for_s5(rc.CARD_ARTIFACT)
        store.write_snapshot(rc.CARD_ARTIFACT, card)
        store.flush()
        findings = er.tombstone_findings(store.root)
        assert any(rc.CARD_ARTIFACT in f and "still cite" in f for f in findings), \
            findings
    finally:
        store.close()


def test_tombstone_findings_is_empty_on_a_store_that_never_deleted_anything(tmp_path):
    """No tombstones means nothing to verify, and that is not a finding.

    Asserted because the opposite mistake is easy: reporting "no deletions have been
    verified" would make every fresh store look broken, and a check that is red by
    default gets switched off.
    """
    store = build(tmp_path)
    try:
        assert er.tombstones(store.root) == ()
        assert er.tombstone_findings(store.root) == []
    finally:
        store.close()


# ----------------------------------------------------------------- verification
def test_verification_queries_come_from_the_episodes_own_text(tmp_path):
    """The queries have to be able to hit, and they only exist before the deletion."""
    store = build(tmp_path)
    try:
        queries = er.verification_queries(store.root, E)
        assert any("烏龍茶" in q for q in queries)
        assert any("新加坡" in q for q in queries)
        assert not any("猜拳" in q for q in queries), \
            "a query drawn from another episode would report a false positive"
        assert queries == tuple(sorted(set(queries))), "queries are not deterministic"
    finally:
        store.close()


def test_verification_queries_refuse_an_episode_with_no_text(tmp_path):
    """An empty query set passes for the same reason a broken check does."""
    store = build(tmp_path)
    try:
        er.delete(store, E, goal=er.FORGET_GOAL)
        store.flush()
        with pytest.raises(er.EraseError, match="no text anywhere"):
            er.verification_queries(store.root, E)
        with pytest.raises(er.EraseError, match="not a minted"):
            er.verification_queries(store.root, "ep-not-an-id")
    finally:
        store.close()


# --------------------------------------------------------------------- retention
def test_the_retention_boundary_is_exact_from_both_sides():
    """One second either side, at one-second resolution.

    The period itself is a policy default with nothing measured behind it, so what is
    asserted is the mechanism. An off-by-one here is a day of somebody's data, in
    whichever direction the error goes.
    """
    episodes = [{"id": E, "seqs": [1], "summary": "s", "end_ts": 1_000.0}]
    period = 90 * 24 * 3600
    assert er.expired(episodes, 1_000.0 + period, retention_s=period) == (E,)
    assert er.expired(episodes, 1_000.0 + period - 1, retention_s=period) == ()
    assert er.expired(episodes, 1_000.0 + period + 1, retention_s=period) == (E,)


@pytest.mark.mutation
def test_the_strict_inequality_mutant_keeps_the_episode_one_tick_too_long():
    """`>=` → `>`: the period becomes its own length plus one tick of the clock."""
    src = ERASE_PY.read_text()
    anchor = "if float(now) - float(e[\"end_ts\"]) >= retention_s]"
    assert src.count(anchor) == 1
    ns: dict = {}
    exec(compile(src.replace(anchor, "if float(now) - float(e['end_ts']) > "
                             "retention_s]"), "<mutant>", "exec"), ns)
    episodes = [{"id": E, "seqs": [1], "summary": "s", "end_ts": 0.0}]
    assert ns["expired"](episodes, 100.0, retention_s=100.0) == ()


def test_expired_orders_oldest_first_and_a_non_positive_period_is_refused():
    """Oldest first is what makes an interrupted sweep converge.

    A sweep killed half way through has removed the most-expired episodes rather than
    an arbitrary subset, so re-running reaches the same state as one uninterrupted run.
    """
    episodes = [{"id": OTHER, "seqs": [1], "summary": "s", "end_ts": 500.0},
                {"id": E, "seqs": [2], "summary": "s", "end_ts": 100.0}]
    assert er.expired(episodes, 10_000.0, retention_s=10.0) == (E, OTHER)
    with pytest.raises(er.EraseError, match="non-positive"):
        er.expired(episodes, 10_000.0, retention_s=0)


def test_an_episode_with_no_end_ts_is_reported_rather_than_kept_or_swept():
    """Two return values because they are two claims.

    `end_ts` is optional in the warm-layer schema, so an undatable episode is a real
    state. It can never expire, which means it sits outside every retention period —
    and a filtered list would make "silently kept forever" look the same as "not due
    yet".
    """
    episodes = [{"id": E, "seqs": [1], "summary": "s", "end_ts": 0.0},
                {"id": OTHER, "seqs": [2], "summary": "s"}]
    dated, undatable = er.datable(episodes)
    assert [e["id"] for e in dated] == [E]
    assert undatable == (OTHER,)
    assert er.expired(episodes, 10_000.0, retention_s=10.0) == (E,)


def test_a_sweep_deletes_only_what_expired_and_carries_the_undatable_finding(tmp_path):
    """One receipt per episode, and the finding rides on it.

    A single aggregate receipt could not say which of forty episodes a finding belongs
    to, and a count of "3 deleted" with the undatable ones counted nowhere is how data
    outlives its retention period with nothing looking wrong.
    """
    store = build(tmp_path)
    try:
        warm = [dict(e) for e in rc.load_episodes(store.root / rc.EPISODES_ARTIFACT)]
        for entry in warm:
            if entry["id"] == OTHER:
                entry.pop("end_ts", None)        # undatable: can never expire
        store.write_snapshot(rc.EPISODES_ARTIFACT, {"version": 1, "episodes": warm})
        store.flush()

        receipts = er.sweep(store, now=200.0 + 10.0, retention_s=10.0)
        store.flush()
        assert [r.episode for r in receipts] == [E]
        assert receipts[0].goal == er.RETENTION_GOAL
        assert any("no end_ts" in f for f in receipts[0].findings), \
            receipts[0].findings
        assert {e["id"] for e in
                rc.load_episodes(store.root / rc.EPISODES_ARTIFACT)} == {OTHER}
    finally:
        store.close()


def test_a_sweep_with_nothing_due_writes_nothing(tmp_path):
    """A no-op sweep must not append a tombstone claiming a deletion."""
    store = build(tmp_path)
    try:
        assert er.sweep(store, now=201.0, retention_s=10_000.0) == ()
        store.flush()
        assert er.tombstones(store.root) == ()
    finally:
        store.close()


def test_the_sweep_uses_the_injected_clock(tmp_path):
    """No `time.time()` inside the module's decision path.

    A sweep whose clock cannot be injected is a retention test that has to wait ninety
    days, which means it is a test nobody runs (§6.1).
    """
    store = build(tmp_path)
    try:
        assert er.sweep(store, clock=lambda: 100.0, retention_s=10.0) == ()
        receipts = er.sweep(store, clock=lambda: 1_000_000.0, retention_s=10.0)
        store.flush()
        assert {r.episode for r in receipts} == {E, OTHER}
    finally:
        store.close()


def test_the_retention_default_is_labelled_a_policy_choice():
    """A number with no measurement behind it must say so where it is defined.

    §6.0 rule 6: a figure in prose is unverified. This one cannot be verified at all —
    no regulation names it and nothing here measured it — so what is checked is that
    the code says which kind of number it is.
    """
    assert er.RETENTION_S == er.RETENTION_DAYS * 24 * 60 * 60
    src = ERASE_PY.read_text()
    head = src.split("RETENTION_DAYS")[0]
    assert "policy default" in head.rsplit("# ------", 1)[-1], \
        "RETENTION_DAYS lost the comment saying it is a policy choice"


# ------------------------------------------------------------------- refusals
@pytest.mark.parametrize("episode,goal,needle", [
    (E, "user_requested_answer", "must be one of"),
    (E, "not_a_goal", "must be one of"),
    ("ep-nope", er.FORGET_GOAL, "not a minted episode id"),
    ("", er.FORGET_GOAL, "not a minted episode id"),
])
def test_a_deletion_refuses_a_bad_target_or_reason(tmp_path, episode, goal, needle):
    """Refused before anything is written, and the goal check is not cosmetic.

    A deletion that could carry `user_requested_answer` would be a removal of personal
    data recorded as an ordinary turn, and the retention sweep and a person's spoken
    request have different audit obligations.
    """
    store = build(tmp_path)
    try:
        with pytest.raises(er.EraseError, match=needle):
            er.delete(store, episode, goal=goal)
        assert er.tombstones(store.root) == ()
        assert [e["seq"] for e in rc.load_events(store.root)] == [1, 2, 3, 4, 5]
    finally:
        store.close()


def test_a_deletion_without_an_attribution_map_refuses_instead_of_reporting_clean(
        tmp_path):
    """No warm layer means no event can be shown to be E's.

    Removing nothing from the event stream and calling that success is the exact shape
    of a clean zero obtained by never looking.
    """
    store = st.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        store.append("events", {"kind": "turn", "text": "我叫 Tim", "why": WHY})
        # The card lock is checked first and would mask this refusal, which is the
        # right order — an unconditional precondition should fire before a conditional
        # one — so the lock is lifted to reach the check this test is about.
        st.unlock_for_s5(rc.CARD_ARTIFACT)
        store.flush()
        with pytest.raises(er.EraseError, match="attribution map"):
            er.delete(store, E, goal=er.FORGET_GOAL)
    finally:
        store.close()


def test_a_removal_cannot_claim_to_be_both_a_snapshot_and_a_rewrite():
    """Two mechanisms in one plan would apply the change twice."""
    with pytest.raises(er.EraseError, match="applied twice"):
        er.Removal("x.json", (), body={"a": 1}, drop_seqs=[1])


def test_the_erasure_writes_through_the_stores_own_root(tmp_path):
    """No root parameter, so a receipt cannot describe a directory nothing touched.

    Derived from the signature rather than asserted by behaviour: an argument that does
    not exist cannot be passed wrongly, and this is the check that notices it coming
    back.
    """
    import inspect
    params = set(inspect.signature(er.delete).parameters)
    assert params == {"store", "episode", "goal", "_after_step"}, params
