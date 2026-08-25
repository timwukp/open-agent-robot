"""Offline tests for the retrieval paths — T-C4's precondition, T-F3, T-F4's honesty.

**Why almost every assertion here is about a status rather than a hit count.** The
plan's T-C4 criterion is "delete episode E, query all five retrieval paths, 0 hits
each". Written that way it is satisfied, greenly and permanently, by five paths that
never ran: no encoder, no artifact on disk, no time left in the budget. And that is
the same defect T-C4's own named mutant describes — "remove the index rebuild → red"
— because an index that was never built returns nothing while the data behind it is
untouched. So the tests below assert on `(status, hits)` pairs, and the headline
mutants are the three that turn a skipped path into `ran`.

**Why the deletion check runs against a *deleted* fixture rather than an empty one.**
An empty store gives 0 hits from everything, which is the shape a broken check and a
working check agree on. Every deletion assertion here therefore has a before-and-after:
the fixture is asserted to hit *before* the removal, so "0 hits after" is a statement
about the removal instead of about the query.

**Why `SEED` and `ORPHAN` are two markers.** A hit with no episode means either a
hand-written seed fact (outside the cascade by construction) or an item that lost its
attribution. A single `None` merges the one that is fine with the one that is the bug,
and then the check either flags every seed forever or stays quiet about every orphan.
The orphan case is tested directly, because deleting an episode entry and leaving its
events behind is the cheapest way to pass a cascade test without deleting anything.

**Why the tokeniser and the id shape are tested here at all.** This module added
`salience.bigrams` and `consolidate.is_episode_id` so that it would not restate two
rules that already exist. The tests that pin those two are in this file because this
file is the reason they were added: `_bigrams` returning a *set* was silently wrong
for term frequency, and a second `startswith("ep-")` here would have been a second
definition of the stored id format.

Mutants live in text-patched copies of `memory/recall.py`, keyed by label, for the
reason `tests/test_why.py` gives: two mutants sharing a directory let the first import
win, which presents as a mutant that mysteriously survives.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory import consolidate as co  # noqa: E402
from memory import recall as rc  # noqa: E402
from memory import salience as sa  # noqa: E402
from memory.profile import TRUST_SEED  # noqa: E402
from memory.store import GOALS, StoreError  # noqa: E402

RECALL_PY = REPO_ROOT / "memory" / "recall.py"

E1 = co.episode_id(1)
E2 = co.episode_id(3)     # deliberately not consecutive: ids come from store seqs
E3 = co.episode_id(7)

ANSWER = "user_requested_answer"
REMEMBER = "user_requested_remember"


# ------------------------------------------------------------------- fixtures
def _event(seq: int, text: str, goal: str = ANSWER) -> dict:
    """One record shaped the way `Store.append` shapes it.

    `why` carries **only** `goal`: `store.check_why` refuses a companion key, so a
    fixture with a spare field would fail for a reason that has nothing to do with
    retrieval, and the failure would look like a retrieval bug.
    """
    return {"seq": seq, "kind": "turn", "ts": float(seq), "text": text,
            "why": {"goal": goal}}


def plant(root: Path, *, events=None, episodes=None, card=None, rps=None) -> Path:
    """Write the artifacts a test needs and nothing else.

    Selective on purpose: "this path had no artifact" is a state the tests have to be
    able to construct, so a helper that always writes all four would make the
    `no_artifact` status unreachable and its assertions vacuous.
    """
    root.mkdir(parents=True, exist_ok=True)
    if events is not None:
        (root / rc.EVENTS_ARTIFACT).write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
            encoding="utf-8")
    if episodes is not None:
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


def episode(eid: str, seqs, summary: str, embedding=None) -> dict:
    entry = {"id": eid, "seqs": list(seqs), "summary": summary}
    if embedding is not None:
        entry["embedding"] = list(embedding)
    return entry


def fact(kind: str, value: str, trust: str, sources) -> dict:
    return {"kind": kind, "value": value, "trust": trust,
            "source_episode_ids": list(sources)}


def unit(i: int, n: int = 3) -> list[float]:
    """A one-hot vector, so cosine values in the tests are 1.0 or 0.0 exactly.

    Exact rather than approximate because a threshold comparison against a
    float-tolerant score is a test that would still pass with the similarity
    reversed, as long as it stayed inside the tolerance.
    """
    return [1.0 if j == i else 0.0 for j in range(n)]


def full(root: Path) -> Path:
    """All four artifacts, three episodes, and one card fact citing two of them."""
    return plant(
        root,
        events=[_event(1, "我對花生過敏"), _event(2, "今天好累"),
                _event(3, "我叫小明", REMEMBER), _event(9, "再玩一局")],
        episodes=[episode(E1, [1, 2], "過敏與疲累", unit(0)),
                  episode(E2, [3], "自我介紹", unit(1)),
                  episode(E3, [9], "猜拳", unit(2))],
        card=[fact("name", "小明", TRUST_SEED, []),
              fact("allergy", "花生", "observed", [E1, E2])],
        rps=[{"episode": E1, "user": "rock", "outcome": "win"},
             {"episode": E3, "user": "paper", "outcome": "loss"}],
    )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return full(tmp_path / "oar")


def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/recall.py.

    Absolute imports inside the copy bind it to the *real* salience, profile,
    consolidate and store modules, so a mutation is scoped to this file and a
    surviving mutant is a statement about this file's assertions. Zero replacements
    gives the no-mutant control (§6.0 rule 2).
    """
    dst = tmp_path / f"mut_{label}" / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    src = RECALL_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in recall.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "recall.py"
    target.write_text(src, encoding="utf-8")
    name = f"recall_mutant_{tmp_path.name}_{label}"
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
NO_ARTIFACT_STATUS = "                return PathResult(spec.name, NO_ARTIFACT)\n"
NO_ENCODER_STATUS = "            return PathResult(spec.name, NO_ENCODER)\n"
DEADLINE_STATUS = "                results.append(PathResult(name, DEADLINE))\n"
CENSUS_ZEROS = "        counts = {status: 0 for status in STATUSES}\n"
CENSUS_COUNT = "            counts[r.status] += 1\n"
CARD_ALL_SOURCES = "            for eid in sources:\n"
STRUCTURED_ORPHAN = '            hits.append(Hit("structured", attrib.get(event["seq"], ORPHAN),\n'
LEXICAL_ORPHAN = '                hits.append(Hit("lexical", attrib.get(event["seq"], ORPHAN), doc,\n'
PACK = "    return bigrams(text)\n"
COSINE_DIMS = "    if len(a) != len(b):\n"
EMPTY_QUERY = "        if not isinstance(text, str) or not text.strip():\n"
ARTIFACT_CENSUS = ("    found = [p.name for p in root.iterdir()\n"
                   "             if p.is_file() and not p.name.startswith(\".\")\n"
                   "             and p.suffix in ARTIFACT_SUFFIXES]\n")
CARD_EXEMPT = '    "card": PathSpec("card", (CARD_ARTIFACT,), deadline_exempt=True),\n'
DELETION_ALL_HITS = "    for hit in result.hits(k=None):\n"
EPISODE_DUPES = "    dupes = sorted({i for i in ids if ids.count(i) > 1})\n"
ATTRIB_OVERLAP = '            if seq in out and out[seq] != entry["id"]:\n'
UNEMBEDDED = '        return tuple(e["id"] for e in load_episodes(path)\n'


# =========================================== the registry, derived on both sides
def test_run_methods_and_registry_are_the_same_set() -> None:
    """A path can only exist as a `_run_<name>`, and only run if it is registered.

    Both directions matter and they catch different mistakes. A `_run_` method with
    no registry entry is a retrieval path `query()` never calls and a deletion never
    examines. A registry entry with no method is an `AttributeError` on the first
    query. Neither is caught by a written-down list of five names, which is the whole
    reason this assertion is derived from `dir()` instead.
    """
    methods = {name[len("_run_"):] for name in dir(rc.Retriever)
               if name.startswith("_run_")}
    assert methods == set(rc.PATHS), (
        f"registry {sorted(rc.PATHS)} vs methods {sorted(methods)}")


def test_path_order_covers_the_registry_exactly() -> None:
    assert set(rc.PATH_ORDER) == set(rc.PATHS)
    assert len(rc.PATH_ORDER) == len(rc.PATHS), "no path may appear twice"
    assert rc.paths() == rc.PATH_ORDER


def test_the_registry_is_five_paths_because_the_plan_says_five() -> None:
    """A count assertion, so adding a sixth path is a decision and not a diff.

    Not a name list — the names are checked by derivation above. This pins the
    *number* the plan's T-C4 criterion is written in terms of, so a sixth path
    arrives together with an edit to the criterion instead of silently under it.
    """
    assert len(rc.PATHS) == 5, sorted(rc.PATHS)


# ================================================= the join key, one creator each
#
# `Hit.ref` is read by three consumers that never call the code that wrote it:
# `deletion_findings` reports it, `memory/erase.py` **rebuilds** card and rps refs to
# say which rows a deletion removed, and S6's ranking arms join a hit back to the
# event carrying its salience. Before `event_ref`/`episode_ref`/`card_ref`/`rps_ref`
# existed the format had nine producers across two production modules — the shape of
# the `.push/files.txt` divergence, where two writers of one format agreed only for as
# long as nobody edited one of them, and the reader that noticed was whichever one
# happened to compare.
#
# So the assertion below is not "the nine known sites call the creators": a list of
# nine sites cannot notice a tenth. It is a census over every production file for the
# *pattern* — an f-string joining an `*_ARTIFACT` name to a `#` — with the four
# creators as the only permitted matches. A new producer written inline is then a red
# test rather than a divergence waiting for a consumer to find it.
def _inline_ref_sites(source: str, label: str) -> list[str]:
    """Every f-string in `source` that joins an `*_ARTIFACT` name to a literal `#`.

    Reported as `label:lineno` strings so a failure names the file and the line. The
    detector deliberately looks for the *shape* rather than for the four artifact
    constants by name: a sixth artifact added by a future node would otherwise be
    outside the census on the day it was introduced.
    """
    sites = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.JoinedStr):
            continue
        names, text = [], ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text += part.value
            elif isinstance(part, ast.FormattedValue):
                inner = part.value
                if isinstance(inner, ast.Name):
                    names.append(inner.id)
                elif isinstance(inner, ast.Attribute):
                    names.append(inner.attr)
        if "#" in text and any(n.endswith("_ARTIFACT") for n in names):
            sites.append(f"{label}:{node.lineno}")
    return sites


def _production_python_files() -> list[Path]:
    skip = {"tests", ".push", ".git", "docs", "web"}
    return sorted(p for p in REPO_ROOT.rglob("*.py")
                  if not (set(p.relative_to(REPO_ROOT).parts) & skip))


def _creator_line_span() -> range:
    """The line range covering the four ref creators in `memory/recall.py`.

    Derived from the AST rather than written down, so moving or renaming a creator
    moves the exemption with it instead of silently widening it.
    """
    names = {"event_ref", "episode_ref", "card_ref", "rps_ref"}
    spans = [(n.lineno, n.end_lineno) for n in ast.parse(RECALL_PY.read_text(
        encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(spans) == len(names), f"expected {sorted(names)}, found {spans}"
    return range(min(s for s, _ in spans), max(e for _, e in spans) + 1)


def test_the_ref_detector_finds_the_creators_themselves() -> None:
    """Non-vacuity, first. A census that matches nothing proves nothing about none.

    The four creators are the only f-strings of this shape that are *supposed* to
    exist, so they are also the positive control: if the detector cannot see them it
    cannot see a tenth producer either, and the census below would be four lines of
    green that never looked at anything.
    """
    span = _creator_line_span()
    found = _inline_ref_sites(RECALL_PY.read_text(encoding="utf-8"), "recall.py")
    inside = [s for s in found if int(s.split(":")[1]) in span]
    assert len(inside) == 4, f"detector saw {found} in creator span {span}"


def test_no_production_file_builds_a_hit_ref_outside_a_creator() -> None:
    files = _production_python_files()
    assert len(files) >= 10, (
        f"the census read {len(files)} production files; §6.0 rule 5 makes a scan "
        "that read almost nothing an error, not a pass")
    span = _creator_line_span()
    offenders = []
    for path in files:
        label = str(path.relative_to(REPO_ROOT))
        for site in _inline_ref_sites(path.read_text(encoding="utf-8"), label):
            if label == "memory/recall.py" and int(site.split(":")[1]) in span:
                continue                      # the creators are the exemption
            offenders.append(site)
    assert offenders == [], (
        f"{offenders} builds a Hit.ref inline. Three consumers depend on that "
        "string's exact spelling and none of them call this code; use "
        "recall.event_ref / episode_ref / card_ref / rps_ref so a drift has to be "
        "a change to the four functions the tests point at")


@pytest.mark.mutation
def test_the_census_catches_a_tenth_producer_written_inline() -> None:
    """The mutant is the change this test exists to refuse: a new call site that
    formats the ref itself. Applied to `erase.py`, because that is the module that
    already rebuilds refs to compare against what `recall.py` emits — the divergence
    would be silent there in exactly the way T-J10's was."""
    src = (REPO_ROOT / "memory" / "erase.py").read_text(encoding="utf-8")
    anchor = "            refs.append(recall.rps_ref(i))\n"
    assert src.count(anchor) == 1, "mutant anchor moved in erase.py"
    assert _inline_ref_sites(src, "erase.py") == [], "erase.py is clean today"
    mutant = src.replace(
        anchor, '            refs.append(f"{recall.RPS_ARTIFACT}#{i}")\n')
    assert _inline_ref_sites(mutant, "erase.py"), (
        "the census cannot see a producer written inline in erase.py, so the test "
        "above would stay green while the two modules drifted apart")


def test_the_creators_address_events_by_seq_and_cards_by_source() -> None:
    """The two ref shapes whose spelling carries information a consumer acts on.

    An event ref naming a list index instead of a store seq would still parse and
    still join — against the wrong event, silently, and only for stores where a
    deletion had already shifted the indices. And a card fact citing two episodes
    produces one hit per source: without the episode in the ref, deleting one of the
    two reads as removing the whole row and deleting the other as removing nothing.
    """
    assert rc.event_ref(7) == f"{rc.EVENTS_ARTIFACT}#7"
    assert rc.episode_ref(E1) == f"{rc.EPISODES_ARTIFACT}#{E1}"
    assert rc.rps_ref(2) == f"{rc.RPS_ARTIFACT}#2"
    assert rc.card_ref(1) == f"{rc.CARD_ARTIFACT}#1"
    assert rc.card_ref(1, E1) == f"{rc.CARD_ARTIFACT}#1@{E1}"
    assert rc.card_ref(1, E1) != rc.card_ref(1, E2) != rc.card_ref(1)


def test_every_path_declares_at_least_one_artifact() -> None:
    for name, spec in rc.PATHS.items():
        assert spec.reads, name
        for artifact in spec.reads:
            assert artifact.endswith(rc.ARTIFACT_SUFFIXES), (name, artifact)


def test_a_path_that_reads_nothing_is_refused() -> None:
    with pytest.raises(rc.RecallError, match="reads nothing"):
        rc.PathSpec("ghost", ())


def test_only_the_card_path_is_exempt_from_the_deadline() -> None:
    """The exemption is the fallback's guarantee, so exactly one path may hold it."""
    exempt = {n for n, s in rc.PATHS.items() if s.deadline_exempt}
    assert exempt == {"card"}, exempt


# ==================================================== artifact coverage (T-C4)
def test_artifacts_are_read_off_the_disk(store: Path) -> None:
    assert rc.artifacts(store) == (rc.EPISODES_ARTIFACT, rc.EVENTS_ARTIFACT,
                                   rc.CARD_ARTIFACT, rc.RPS_ARTIFACT)


def test_a_missing_root_censuses_as_empty_rather_than_raising(tmp_path: Path) -> None:
    assert rc.artifacts(tmp_path / "nope") == ()
    assert rc.coverage_findings(tmp_path / "nope") == []


def test_temp_and_dot_files_are_not_artifacts(store: Path) -> None:
    """`Store` writes `.profile.json.<rand>.tmp` mid-replace; flagging that would
    report a finding for a healthy atomic write in progress."""
    (store / ".profile.json.abc123.tmp").write_text("{}", encoding="utf-8")
    (store / "notes.txt").write_text("x", encoding="utf-8")
    assert ".profile.json.abc123.tmp" not in rc.artifacts(store)
    assert "notes.txt" not in rc.artifacts(store)
    assert rc.coverage_findings(store) == []


def test_a_new_artifact_nobody_reads_is_a_finding(store: Path) -> None:
    """The sixth-file case. `Store.write_snapshot` accepts any plain name, so there
    is no enum anywhere to derive this from — the disk is the only producer."""
    assert rc.coverage_findings(store) == []
    (store / "skills_seen.json").write_text("{}", encoding="utf-8")
    findings = rc.coverage_findings(store)
    assert len(findings) == 1
    assert "skills_seen.json" in findings[0]
    assert "no retrieval path reads" in findings[0]


def test_coverage_findings_names_every_uncovered_artifact(store: Path) -> None:
    for name in ("a_new.json", "b_new.jsonl"):
        (store / name).write_text("{}", encoding="utf-8")
    findings = rc.coverage_findings(store)
    assert len(findings) == 2
    assert sum("a_new.json" in f for f in findings) == 1
    assert sum("b_new.jsonl" in f for f in findings) == 1


# ======================================================== Hit / PathResult / Result
def test_a_hit_is_immutable() -> None:
    hit = rc.Hit("card", E1, "allergy: 花生", 1.0, "profile.json#0")
    with pytest.raises(rc.RecallError, match="immutable"):
        hit.score = 2.0
    with pytest.raises(rc.RecallError, match="immutable"):
        del hit.episode


def test_a_hit_refuses_an_episode_that_is_neither_an_id_nor_a_marker() -> None:
    for bad in ("ep-", "ep-abcdef", "episode-000001", "", None, 7, "EP-000001"):
        with pytest.raises(rc.RecallError, match="neither a minted episode id"):
            rc.Hit("card", bad, "t", 1.0, "r")


def test_a_hit_accepts_both_markers_and_a_minted_id() -> None:
    for episode_ref in (E1, rc.SEED, rc.ORPHAN):
        assert rc.Hit("card", episode_ref, "t", 1.0, "r").episode == episode_ref


def test_a_hit_refuses_an_unregistered_path() -> None:
    with pytest.raises(rc.RecallError, match="unknown path"):
        rc.Hit("graph", E1, "t", 1.0, "r")


def test_the_two_markers_are_not_minted_ids() -> None:
    """Otherwise a marker could collide with a real episode and a deletion would
    match the wrong thing."""
    for marker in rc.NOT_AN_EPISODE:
        assert not co.is_episode_id(marker)


def test_a_path_that_did_not_run_cannot_carry_hits() -> None:
    hit = rc.Hit("card", E1, "t", 1.0, "r")
    for status in (rc.NO_ARTIFACT, rc.NO_ENCODER, rc.DEADLINE):
        with pytest.raises(rc.RecallError, match="did not run"):
            rc.PathResult("card", status, [hit])
    assert rc.PathResult("card", rc.RAN, [hit]).hits == (hit,)


def test_path_result_refuses_an_unknown_status() -> None:
    with pytest.raises(rc.RecallError, match="unknown status"):
        rc.PathResult("card", "slow")


def test_a_result_must_carry_every_registered_path_once() -> None:
    partial = [rc.PathResult(n, rc.RAN) for n in rc.paths()[:4]]
    with pytest.raises(rc.RecallError, match="every registered path"):
        rc.Result("q", partial)
    doubled = [rc.PathResult(n, rc.RAN) for n in rc.paths()] + \
              [rc.PathResult("card", rc.RAN)]
    with pytest.raises(rc.RecallError, match="every registered path"):
        rc.Result("q", doubled)


def test_the_census_carries_every_status_including_the_zeros() -> None:
    result = rc.Result("q", [rc.PathResult(n, rc.RAN) for n in rc.paths()])
    assert set(result.census()) == set(rc.STATUSES)
    assert result.census() == {rc.RAN: 5, rc.NO_ARTIFACT: 0, rc.NO_ENCODER: 0,
                               rc.DEADLINE: 0}


def test_status_of_refuses_an_unknown_path() -> None:
    result = rc.Result("q", [rc.PathResult(n, rc.RAN) for n in rc.paths()])
    with pytest.raises(rc.RecallError, match="unknown path"):
        result.status_of("graph")


# =============================================================== ranking (T-F4)
def test_hits_are_ranked_within_a_path_and_never_fused(store: Path) -> None:
    """Scores are compared inside a path and never across one.

    BM25 is unbounded and cosine is at most 1.0, so one list ordered by score is
    ordered by a quantity that does not exist. The observable consequence asserted
    here is that the output stays grouped by path in `PATH_ORDER`.
    """
    result = rc.Retriever(store, encoder=lambda t: unit(0)).query("花生")
    order = [h.path for h in result.hits(k=None)]
    assert order == sorted(order, key=rc.paths().index)
    for name in rc.paths():
        scores = [h.score for h in result.hits(k=None) if h.path == name]
        assert scores == sorted(scores, reverse=True), name


def test_ranking_is_deterministic_under_ties(store: Path) -> None:
    """Every structured hit scores 1.0, so the tie-break is the whole ordering."""
    first = rc.Retriever(store).query("我")
    second = rc.Retriever(store).query("我")
    assert [h.as_dict() for h in first.hits(k=None)] == \
           [h.as_dict() for h in second.hits(k=None)]
    refs = [h.ref for h in first.hits(k=None) if h.path == "structured"]
    assert refs == sorted(refs)


def test_k_is_applied_per_path_not_to_the_concatenation(store: Path) -> None:
    result = rc.Retriever(store, encoder=lambda t: unit(0)).query("我")
    per_path = {}
    for hit in result.hits(k=2):
        per_path[hit.path] = per_path.get(hit.path, 0) + 1
    assert max(per_path.values()) <= 2
    assert len(per_path) >= 3, "a global top-2 would return one path's two best only"


def test_k_none_returns_everything(store: Path) -> None:
    result = rc.Retriever(store).query("我")
    assert len(result.hits(k=None)) >= len(result.hits(k=1))


# ================================================================ BM25 / cosine
def test_bm25_returns_one_score_per_document_in_order() -> None:
    scores = rc.bm25("花生", ["我對花生過敏", "今天好累", "花生醬"])
    assert len(scores) == 3
    assert scores[0] > 0.0 and scores[2] > 0.0
    assert scores[1] == 0.0


def test_bm25_counts_term_frequency_not_presence() -> None:
    """The reason `salience.bigrams` returns a tuple. Equal lengths, so the only
    difference between the two documents is how often the term occurs."""
    repeated, once = "好累好累好累好累", "好累一二三四五六"
    assert len(repeated) == len(once)
    a, b = rc.bm25("好累", [repeated, once])
    assert a > b, (a, b)


def test_bm25_of_an_empty_corpus_is_empty() -> None:
    assert rc.bm25("花生", []) == ()


def test_bm25_never_scores_a_common_term_negative() -> None:
    """The `+0.5/+0.5` idf. The classic `log(N/df)` form goes negative for a term in
    every document, which would let a shared term push a document *below* one that
    does not contain it."""
    scores = rc.bm25("花生", ["花生一", "花生二", "花生三"])
    assert all(s > 0.0 for s in scores), scores


def test_bm25_is_deterministic() -> None:
    docs = ["我對花生過敏", "花生醬很好吃", "今天好累"]
    assert rc.bm25("花生 好累", docs) == rc.bm25("花生 好累", docs)


def test_cosine_of_known_vectors() -> None:
    assert rc.cosine(unit(0), unit(0)) == pytest.approx(1.0)
    assert rc.cosine(unit(0), unit(1)) == pytest.approx(0.0)
    assert rc.cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_refuses_a_dimension_mismatch() -> None:
    """Scoring a mismatch as 0.0 is a semantic path that runs, reports `ran`, and is
    blind — which is the combination the status field exists to make impossible."""
    with pytest.raises(rc.RecallError, match="dimension mismatch"):
        rc.cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_refuses_a_zero_vector_and_an_empty_one() -> None:
    with pytest.raises(rc.RecallError, match="no direction"):
        rc.cosine([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(rc.RecallError, match="empty embedding"):
        rc.cosine([], [])


# ========================================================== episodes.json parsing
def test_load_episodes_accepts_the_planted_warm_layer(store: Path) -> None:
    entries = rc.load_episodes(store / rc.EPISODES_ARTIFACT)
    assert [e["id"] for e in entries] == [E1, E2, E3]


@pytest.mark.parametrize("episodes, match", [
    ([{"id": E1, "seqs": [1], "summary": "s", "mentions": 2}], "unknown field"),
    ([{"id": "ep-1", "seqs": [1], "summary": "s"}], "not a minted episode"),
    ([{"id": E1, "seqs": [], "summary": "s"}], "non-empty list"),
    ([{"id": E1, "seqs": [True], "summary": "s"}], "non-empty list"),
    ([{"id": E1, "seqs": [1], "summary": "   "}], "no summary"),
    ([{"id": E1, "seqs": [1], "summary": "s", "embedding": []}], "non-empty list"),
    ([{"id": E1, "seqs": [1], "summary": "s", "embedding": "abc"}], "non-empty list"),
    ([{"id": E1, "seqs": [1], "summary": "s"},
      {"id": E1, "seqs": [2], "summary": "t"}], "duplicate episode id"),
    (["not an object"], "not an object"),
])
def test_load_episodes_refuses_a_malformed_warm_layer(tmp_path, episodes, match) -> None:
    root = plant(tmp_path / "oar", episodes=episodes)
    with pytest.raises(rc.RecallError, match=match):
        rc.load_episodes(root / rc.EPISODES_ARTIFACT)


def test_load_episodes_refuses_a_non_v1_object(tmp_path: Path) -> None:
    root = tmp_path / "oar"
    root.mkdir()
    (root / rc.EPISODES_ARTIFACT).write_text('{"version": 2, "episodes": []}',
                                             encoding="utf-8")
    with pytest.raises(rc.RecallError, match="v1 object"):
        rc.load_episodes(root / rc.EPISODES_ARTIFACT)


def test_load_episodes_refuses_unparsable_json(tmp_path: Path) -> None:
    root = tmp_path / "oar"
    root.mkdir()
    (root / rc.EPISODES_ARTIFACT).write_text('{"version": 1, "epi', encoding="utf-8")
    with pytest.raises(rc.RecallError, match="not valid JSON"):
        rc.load_episodes(root / rc.EPISODES_ARTIFACT)


def test_a_missing_embedding_is_allowed_and_reported(tmp_path: Path) -> None:
    """An episode summarised before the encoder existed is a real state. An index
    silently covering half the episodes answers every query and is wrong about all
    of them, so the gap is named rather than assumed away."""
    root = plant(tmp_path / "oar",
                 episodes=[episode(E1, [1], "有向量", unit(0)),
                           episode(E2, [2], "沒有向量")])
    assert rc.Retriever(root).unembedded() == (E2,)


def test_unembedded_on_a_missing_warm_layer_is_empty(tmp_path: Path) -> None:
    assert rc.Retriever(tmp_path).unembedded() == ()


def test_attribution_maps_sequences_to_episodes(store: Path) -> None:
    attrib = rc.attribution(rc.load_episodes(store / rc.EPISODES_ARTIFACT))
    assert attrib == {1: E1, 2: E1, 3: E2, 9: E3}


def test_attribution_refuses_two_episodes_claiming_one_sequence(tmp_path) -> None:
    """The segmenter partitions the stream, so an overlap means this file was not
    written by it — and a deletion driven by an ambiguous map removes one reference
    and leaves the other."""
    entries = [episode(E1, [1, 2], "a"), episode(E2, [2, 3], "b")]
    with pytest.raises(rc.RecallError, match="partition the stream"):
        rc.attribution(entries)


# ================================================================ events parsing
def test_events_require_a_store_stamped_sequence(tmp_path: Path) -> None:
    root = plant(tmp_path / "oar", events=[{"kind": "turn", "text": "x",
                                            "why": {"goal": ANSWER}}],
                 episodes=[])
    with pytest.raises(rc.RecallError, match="no 'seq'"):
        rc.Retriever(root).query("x")


def test_a_partial_event_line_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    """A truncated line is the failure the single-writer invariant exists to prevent;
    skipping it would make that failure unobservable."""
    root = plant(tmp_path / "oar", events=[_event(1, "ok")], episodes=[])
    with (root / rc.EVENTS_ARTIFACT).open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "kind": "tu\n')
    with pytest.raises(rc.RecallError, match="not valid JSON"):
        rc.Retriever(root).query("ok")


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    root = plant(tmp_path / "oar", events=[_event(1, "ok")],
                 episodes=[episode(E1, [1], "s")])
    with (root / rc.EVENTS_ARTIFACT).open("a", encoding="utf-8") as f:
        f.write("\n")
    assert rc.Retriever(root).query("ok").status_of("structured") == rc.RAN


# ================================================================ goal filtering
def test_goals_none_means_every_goal(store: Path) -> None:
    result = rc.Retriever(store).query("我")
    structured = [h for h in result.hits(k=None) if h.path == "structured"]
    assert len(structured) == 4


def test_goals_filter_narrows_the_structured_path(store: Path) -> None:
    result = rc.Retriever(store).query("我", goals=[REMEMBER])
    structured = [h for h in result.hits(k=None) if h.path == "structured"]
    assert [h.ref for h in structured] == [f"{rc.EVENTS_ARTIFACT}#3"]


def test_an_unknown_goal_in_a_query_raises(store: Path) -> None:
    """`store.filter_by_goal` refuses it, and this test is here to prove the refusal
    survives the trip through `Retriever` rather than being swallowed."""
    with pytest.raises(StoreError, match="unknown goal"):
        rc.Retriever(store).query("我", goals=["user_requested_everything"])


def test_every_goal_the_fixture_uses_is_a_real_goal() -> None:
    assert ANSWER in GOALS and REMEMBER in GOALS


# ==================================================================== statuses
def test_all_five_paths_run_on_a_complete_store(store: Path) -> None:
    result = rc.Retriever(store, encoder=lambda t: unit(0)).query("花生")
    assert result.census() == {rc.RAN: 5, rc.NO_ARTIFACT: 0, rc.NO_ENCODER: 0,
                               rc.DEADLINE: 0}


def test_no_encoder_is_a_status_not_an_empty_result(store: Path) -> None:
    result = rc.Retriever(store).query("花生")
    assert result.status_of("semantic") == rc.NO_ENCODER
    assert [h for h in result.hits(k=None) if h.path == "semantic"] == []
    # And the four paths that do not need one are unaffected.
    assert result.census()[rc.RAN] == 4


def test_a_missing_artifact_is_a_status(tmp_path: Path) -> None:
    root = plant(tmp_path / "oar", events=[_event(1, "花生")],
                 episodes=[episode(E1, [1], "s", unit(0))])
    result = rc.Retriever(root, encoder=lambda t: unit(0)).query("花生")
    assert result.status_of("card") == rc.NO_ARTIFACT
    assert result.status_of("rps") == rc.NO_ARTIFACT
    assert result.status_of("lexical") == rc.RAN


def test_an_empty_store_reports_four_missing_artifacts_and_no_encoder(tmp_path) -> None:
    """Every path is unavailable, and every one of them says so. The reading this
    replaces is five clean zeros."""
    result = rc.Retriever(tmp_path / "empty").query("花生")
    assert result.census() == {rc.RAN: 0, rc.NO_ARTIFACT: 4, rc.NO_ENCODER: 1,
                               rc.DEADLINE: 0}
    assert result.hits(k=None) == ()


def test_status_precedence_is_encoder_before_artifact(tmp_path: Path) -> None:
    """Fixed by code order on purpose: a path missing both would otherwise report a
    different status depending on which check an edit happened to move first, and the
    census is compared across runs."""
    result = rc.Retriever(tmp_path / "empty").query("花生")
    assert result.status_of("semantic") == rc.NO_ENCODER


# ================================================================ deadline (T-F3)
def test_the_deadline_skips_the_expensive_paths_and_keeps_the_card(store: Path) -> None:
    """Injected clock, so the schedule is exact rather than a race against the
    machine — a sleeping double reads whatever the suite's load makes it read."""
    ticks = iter([0.0] + [1.0] * 20)
    result = rc.Retriever(store, encoder=lambda t: unit(0),
                          clock=lambda: next(ticks)).query("花生")
    assert result.status_of("card") == rc.RAN
    assert result.census() == {rc.RAN: 1, rc.NO_ARTIFACT: 0, rc.NO_ENCODER: 0,
                               rc.DEADLINE: 4}
    assert [h.path for h in result.hits(k=None)] == ["card", "card"]


def test_a_budget_that_is_exactly_spent_still_skips(store: Path) -> None:
    """`>=`, not `>`: a path started with zero budget left cannot finish inside it."""
    ticks = iter([0.0] + [rc.DEADLINE_S] * 20)
    result = rc.Retriever(store, clock=lambda: next(ticks)).query("花生")
    assert result.census()[rc.DEADLINE] == 4


def test_a_generous_budget_skips_nothing(store: Path) -> None:
    ticks = iter([0.0] + [0.001 * i for i in range(1, 30)])
    result = rc.Retriever(store, encoder=lambda t: unit(0),
                          clock=lambda: next(ticks)).query("花生")
    assert result.census()[rc.DEADLINE] == 0


def test_a_non_positive_deadline_is_refused(store: Path) -> None:
    """It would skip every non-exempt path on every turn, and the census would read
    as a machine that is merely slow."""
    for bad in (0.0, -0.150):
        with pytest.raises(rc.RecallError, match="must be positive"):
            rc.Retriever(store, deadline_s=bad)


def test_the_default_deadline_is_the_plans_150_ms() -> None:
    assert rc.DEADLINE_S == 0.150


# ================================================================= empty query
@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 7])
def test_an_empty_query_is_refused(store: Path, bad) -> None:
    """It returns nothing from every path *while reporting that every path ran*,
    which is indistinguishable from a store that holds nothing — and a retrieval
    test written against it passes for the same reason a broken one does."""
    with pytest.raises(rc.RecallError, match="empty query"):
        rc.Retriever(store).query(bad)


# ============================================== attribution: SEED vs ORPHAN
def test_a_seed_fact_is_marked_seed_not_orphaned(store: Path) -> None:
    result = rc.Retriever(store).query("小明")
    card = [h for h in result.hits(k=None) if h.path == "card"]
    assert any(h.episode == rc.SEED for h in card), [h.as_dict() for h in card]


def test_a_card_fact_emits_one_hit_per_source_episode(store: Path) -> None:
    """A fact citing two episodes that reported only the first would tell a deletion
    of the second that the card is clean while the row still cites it."""
    result = rc.Retriever(store).query("花生")
    card = [h for h in result.hits(k=None) if h.path == "card"]
    assert {h.episode for h in card} == {E1, E2}
    assert {h.ref for h in card} == {f"{rc.CARD_ARTIFACT}#1@{E1}",
                                     f"{rc.CARD_ARTIFACT}#1@{E2}"}


def test_an_event_in_no_episode_is_marked_orphan(tmp_path: Path) -> None:
    root = plant(tmp_path / "oar",
                 events=[_event(1, "在 episode 裡"), _event(2, "還沒關閉")],
                 episodes=[episode(E1, [1], "s")])
    result = rc.Retriever(root).query("episode 還沒")
    by_ref = {h.ref: h.episode for h in result.hits(k=None) if h.path == "structured"}
    assert by_ref[f"{rc.EVENTS_ARTIFACT}#1"] == E1
    assert by_ref[f"{rc.EVENTS_ARTIFACT}#2"] == rc.ORPHAN


def test_an_rps_round_with_no_episode_is_an_orphan(tmp_path: Path) -> None:
    """A table of bare counts cannot say which episode a count came from, so no
    deletion could ever prove it removed one."""
    root = plant(tmp_path / "oar",
                 rps=[{"user": "rock", "outcome": "win"}])
    result = rc.Retriever(root).query("rock")
    assert [h.episode for h in result.hits(k=None) if h.path == "rps"] == [rc.ORPHAN]


def test_an_rps_table_of_bare_counts_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "oar"
    root.mkdir()
    (root / rc.RPS_ARTIFACT).write_text('{"rock": 4, "paper": 2}', encoding="utf-8")
    with pytest.raises(rc.RecallError, match="'rounds' list"):
        rc.Retriever(root).query("rock")


def test_an_rps_round_with_a_bogus_episode_is_refused(tmp_path: Path) -> None:
    root = plant(tmp_path / "oar", rps=[{"episode": "ep-7", "user": "rock"}])
    with pytest.raises(rc.RecallError, match="not a minted episode id"):
        rc.Retriever(root).query("rock")


# ========================================================= deletion_findings (T-C4)
def test_deletion_findings_refuses_a_non_id() -> None:
    result = rc.Result("q", [rc.PathResult(n, rc.RAN) for n in rc.paths()])
    with pytest.raises(rc.RecallError, match="not a minted episode id"):
        rc.deletion_findings(result, "ep-7")


def test_a_path_that_did_not_run_is_a_finding_not_a_clean_zero(tmp_path: Path) -> None:
    """The headline. Five empty paths, every one of them unavailable, and the
    plan's literal criterion — "0 hits each" — is satisfied."""
    result = rc.Retriever(tmp_path / "empty").query("花生")
    assert result.hits(k=None) == (), "the literal criterion is met"
    findings = rc.deletion_findings(result, E1)
    assert len(findings) == 5
    assert all("did not run" in f for f in findings)
    for name in rc.paths():
        assert any(f.startswith(name + " did not run") for f in findings), name


def test_a_surviving_reference_is_a_finding(store: Path) -> None:
    result = rc.Retriever(store, encoder=lambda t: unit(1)).query("自我介紹 小明")
    findings = rc.deletion_findings(result, E2)
    assert any(E2 in f and "still returns" in f for f in findings), findings


def test_a_clean_deletion_reports_nothing_and_the_fixture_proves_it_could(tmp_path) -> None:
    """Before-and-after, because an empty store gives 0 hits from everything and a
    broken check agrees with a working one on that input.

    Before: every path runs and something cites E2. After E2's episode entry, its
    events, its card sources and its rounds are all removed: still five `ran`
    statuses, and nothing cites E2 — no findings.
    """
    before = full(tmp_path / "before")
    result = rc.Retriever(before, encoder=lambda t: unit(1)).query("自我介紹 小明")
    assert result.census()[rc.RAN] == 5
    assert any(h.episode == E2 for h in result.hits(k=None)), "must hit before"

    after = plant(
        tmp_path / "after",
        events=[_event(1, "我對花生過敏"), _event(2, "今天好累"), _event(9, "再玩一局")],
        episodes=[episode(E1, [1, 2], "過敏與疲累", unit(0)),
                  episode(E3, [9], "猜拳", unit(2))],
        card=[fact("name", "小明", TRUST_SEED, []),
              fact("allergy", "花生", "observed", [E1])],
        rps=[{"episode": E1, "user": "rock", "outcome": "win"},
             {"episode": E3, "user": "paper", "outcome": "loss"}],
    )
    clean = rc.Retriever(after, encoder=lambda t: unit(1)).query("自我介紹 小明")
    assert clean.census()[rc.RAN] == 5
    assert rc.deletion_findings(clean, E2) == []


def test_orphaning_the_events_does_not_pass_the_deletion_check(tmp_path: Path) -> None:
    """Delete the *episode entry* and leave its events. Every reference to E2 becomes
    a reference to nothing, and an attribution-only check reads 0 hits for E2 while
    the text is still retrievable. This is the cheapest way to pass a cascade test
    without deleting anything, so it has to be a finding."""
    root = plant(
        tmp_path / "oar",
        events=[_event(1, "我對花生過敏"), _event(3, "我叫小明", REMEMBER)],
        episodes=[episode(E1, [1], "過敏", unit(0))],   # E2's entry is gone
        card=[fact("allergy", "花生", "observed", [E1])],
        rps=[{"episode": E1, "user": "rock", "outcome": "win"}],
    )
    result = rc.Retriever(root, encoder=lambda t: unit(0)).query("小明 花生")
    assert result.census()[rc.RAN] == 5
    assert not any(h.episode == E2 for h in result.hits(k=None)), \
        "nothing names E2 any more — that is exactly the trap"
    findings = rc.deletion_findings(result, E2)
    assert findings and all("orphan" in f for f in findings), findings


def test_a_seed_fact_alone_is_not_a_deletion_finding(tmp_path: Path) -> None:
    """The other side of the two-marker decision: a hand-written fact is outside the
    cascade, so it must not make every deletion check permanently dirty."""
    root = plant(tmp_path / "oar", events=[], episodes=[],
                 card=[fact("name", "小明", TRUST_SEED, [])],
                 rps=[])
    result = rc.Retriever(root).query("小明")
    assert any(h.episode == rc.SEED for h in result.hits(k=None))
    findings = rc.deletion_findings(result, E2)
    assert [f for f in findings if "orphan" in f] == []
    assert all("semantic did not run" in f for f in findings), findings


# =========================================== the two one-creator companion edits
def test_bigrams_is_ordered_and_keeps_repeats() -> None:
    assert sa.bigrams("我好累") == ("我好", "好累")
    assert sa.bigrams("好累好累") == ("好累", "累好", "好累")
    assert sa.bigrams("我 好 累") == sa.bigrams("我好累")
    assert sa.bigrams("我") == ("我",)
    assert sa.bigrams("   ") == ()


def test_the_bigram_set_is_still_a_set_and_novelty_is_unchanged() -> None:
    """The refactor's regression check: `novelty` takes a Jaccard, so it needs the
    set — the tuple exists because `bm25` needs the frequencies."""
    assert sa._bigrams("好累好累") == {"好累", "累好"}
    assert sa.novelty("我好累") == 1.0
    assert sa.novelty("我好累", ["我好累"]) == 0.0
    assert 0.0 < sa.novelty("我好累", ["我好餓"]) < 1.0


def test_recall_uses_the_salience_tokeniser() -> None:
    """One creator. A second tokeniser here would split the retrieval bucket by
    which module produced the term."""
    assert rc._pack("我好累") == sa.bigrams("我好累")


def test_is_episode_id_delegates_and_recall_never_restates_the_format() -> None:
    assert co.is_episode_id(E1)
    assert not co.is_episode_id("ep-")
    assert not co.is_episode_id("ep-abcdef")
    assert not co.is_episode_id("ep-００００４２")     # isdigit() is True for these
    source = RECALL_PY.read_text(encoding="utf-8")
    # Described rather than reproduced: a literal `"ep-"` in this assertion would be
    # found by the scan reading its own test file if the two ever shared a tree.
    prefix = co.EPISODE_ID_PREFIX
    code = "\n".join(ln for ln in source.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert f'"{prefix}"' not in code, "recall.py must not restate the id prefix"
    assert "EPISODE_ID_DIGITS" not in code, "nor the width"


# ================================================================== as_dict
def test_as_dict_round_trips_through_json(store: Path) -> None:
    result = rc.Retriever(store, encoder=lambda t: unit(0)).query("花生")
    blob = json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)
    back = json.loads(blob)
    assert back["query"] == "花生"
    assert set(back["census"]) == set(rc.STATUSES)
    assert [r["name"] for r in back["results"]] == list(rc.paths())


# ===================================================================== mutants
@pytest.mark.mutation
def test_no_mutant_control(tmp_path) -> None:
    """§6.0 rule 2: a harness that cannot import a clean copy kills every mutant."""
    clean = load_mutant(tmp_path, "control")
    assert clean.PATH_ORDER == rc.PATH_ORDER
    assert clean.STATUSES == rc.STATUSES
    assert clean.NOT_AN_EPISODE == rc.NOT_AN_EPISODE
    root = full(tmp_path / "store")
    result = clean.Retriever(root, encoder=lambda t: unit(1)).query("自我介紹")
    assert result.census()[clean.RAN] == 5
    assert clean.deletion_findings(result, E2), "E2 is still cited before deletion"
    assert clean.deletion_findings(
        clean.Retriever(tmp_path / "empty").query("花生"), E1)


@pytest.mark.mutation
def test_mutant_missing_artifact_reports_ran(tmp_path) -> None:
    """T-C4's criterion, satisfied by a path that never opened a file."""
    mutant = load_mutant(tmp_path, "artifact_ran",
                         (NO_ARTIFACT_STATUS,
                          "                return PathResult(spec.name, RAN)\n"))
    empty = tmp_path / "empty"
    m_result = mutant.Retriever(empty, encoder=lambda t: unit(0)).query("花生")
    assert m_result.hits(k=None) == ()
    assert mutant.deletion_findings(m_result, E1) == [], "the mutant reports clean"
    r_result = rc.Retriever(empty, encoder=lambda t: unit(0)).query("花生")
    # All five: an encoder *was* injected here, so `semantic` clears that check and
    # then fails on its artifact like the other four. One finding per path.
    findings = rc.deletion_findings(r_result, E1)
    assert len(findings) == 5
    assert all(f"({rc.NO_ARTIFACT})" in f for f in findings), findings
    # Not a mutant that breaks everything: a complete store still works.
    good = mutant.Retriever(full(tmp_path / "s"), encoder=lambda t: unit(0)).query("花生")
    assert good.census()[mutant.RAN] == 5


@pytest.mark.mutation
def test_mutant_no_encoder_reports_ran(tmp_path) -> None:
    """"Remove the index rebuild → red": an unbuilt index that reports it ran."""
    mutant = load_mutant(tmp_path, "encoder_ran",
                         (NO_ENCODER_STATUS,
                          "            return PathResult(spec.name, RAN)\n"))
    root = full(tmp_path / "store")
    m_result = mutant.Retriever(root).query("自我介紹")     # no encoder injected
    assert m_result.status_of("semantic") == mutant.RAN
    assert [h for h in m_result.hits(k=None) if h.path == "semantic"] == []
    assert not any("semantic did not run" in f
                   for f in mutant.deletion_findings(m_result, E3))
    r_result = rc.Retriever(root).query("自我介紹")
    assert r_result.status_of("semantic") == rc.NO_ENCODER
    assert any("semantic did not run" in f
               for f in rc.deletion_findings(r_result, E3))


@pytest.mark.mutation
def test_mutant_deadline_reports_ran(tmp_path) -> None:
    """T-F3's degradation, laundered into four clean zeros."""
    mutant = load_mutant(tmp_path, "deadline_ran",
                         (DEADLINE_STATUS,
                          "                results.append(PathResult(name, RAN))\n"))
    root = full(tmp_path / "store")
    ticks = iter([0.0] + [1.0] * 20)
    m_result = mutant.Retriever(root, clock=lambda: next(ticks)).query("自我介紹")
    assert m_result.census()[mutant.RAN] == 5
    assert not any("did not run" in f
                   for f in mutant.deletion_findings(m_result, E3))
    ticks2 = iter([0.0] + [1.0] * 20)
    r_result = rc.Retriever(root, clock=lambda: next(ticks2)).query("自我介紹")
    assert r_result.census()[rc.DEADLINE] == 4
    assert len([f for f in rc.deletion_findings(r_result, E3)
                if "did not run" in f]) == 4


@pytest.mark.mutation
def test_mutant_card_not_exempt_leaves_the_degraded_turn_with_nothing(tmp_path) -> None:
    """The deadline is allowed to skip the fallback it degrades *to*."""
    mutant = load_mutant(tmp_path, "card_not_exempt",
                         (CARD_EXEMPT,
                          '    "card": PathSpec("card", (CARD_ARTIFACT,)),\n'))
    root = full(tmp_path / "store")
    ticks = iter([0.0] + [1.0] * 20)
    m_result = mutant.Retriever(root, clock=lambda: next(ticks)).query("小明")
    assert m_result.census()[mutant.DEADLINE] == 5
    assert m_result.hits(k=None) == (), "nothing left to answer the turn with"
    ticks2 = iter([0.0] + [1.0] * 20)
    r_result = rc.Retriever(root, clock=lambda: next(ticks2)).query("小明")
    assert r_result.status_of("card") == rc.RAN
    assert r_result.hits(k=None), "the fallback answered"


@pytest.mark.mutation
def test_mutant_census_without_zeros_hides_a_degraded_turn(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "census_sparse",
                         (CENSUS_ZEROS, "        counts = {}\n"),
                         (CENSUS_COUNT,
                          "            counts[r.status] = counts.get(r.status, 0) + 1\n"))
    root = full(tmp_path / "store")
    m_result = mutant.Retriever(root, encoder=lambda t: unit(0)).query("花生")
    assert set(m_result.census()) != set(mutant.STATUSES)
    assert mutant.DEADLINE not in m_result.census(), \
        "'no path was skipped' and 'the counter was removed' print the same"
    r_result = rc.Retriever(root, encoder=lambda t: unit(0)).query("花生")
    assert set(r_result.census()) == set(rc.STATUSES)


@pytest.mark.mutation
def test_mutant_card_reports_only_its_first_source(tmp_path) -> None:
    """A fact citing two episodes, checked against a deletion of the second."""
    mutant = load_mutant(tmp_path, "card_first_source",
                         (CARD_ALL_SOURCES, "            for eid in sources[:1]:\n"))
    root = full(tmp_path / "store")
    m_result = mutant.Retriever(root).query("花生")
    m_card = [h for h in m_result.hits(k=None) if h.path == "card"]
    assert {h.episode for h in m_card} == {E1}
    assert not any("card still returns" in f
                   for f in mutant.deletion_findings(m_result, E2))
    r_result = rc.Retriever(root).query("花生")
    assert {h.episode for h in r_result.hits(k=None) if h.path == "card"} == {E1, E2}
    assert any("card still returns" in f
               for f in rc.deletion_findings(r_result, E2))


@pytest.mark.mutation
def test_mutant_orphans_marked_seed_pass_the_cascade_check(tmp_path) -> None:
    """One marker instead of two: the case that is fine absorbs the case that is
    the bug.

    Both event-reading paths are mutated together because they are one decision — a
    mutant applied to `structured` alone would be killed by `lexical` still reporting
    the orphan, which would say nothing about whether the two markers are needed.
    """
    mutant = load_mutant(tmp_path, "orphan_as_seed",
                         (STRUCTURED_ORPHAN,
                          '            hits.append(Hit("structured", '
                          'attrib.get(event["seq"], SEED),\n'),
                         (LEXICAL_ORPHAN,
                          '                hits.append(Hit("lexical", '
                          'attrib.get(event["seq"], SEED), doc,\n'))
    root = plant(tmp_path / "oar",
                 events=[_event(1, "我對花生過敏"), _event(3, "我叫小明")],
                 episodes=[episode(E1, [1], "過敏", unit(0))],
                 card=[fact("allergy", "花生", "observed", [E1])],
                 rps=[{"episode": E1, "user": "rock", "outcome": "win"}])
    m_result = mutant.Retriever(root, encoder=lambda t: unit(0)).query("小明 花生")
    assert mutant.deletion_findings(m_result, E2) == [], "the mutant reports clean"
    r_result = rc.Retriever(root, encoder=lambda t: unit(0)).query("小明 花生")
    assert any("orphan" in f for f in rc.deletion_findings(r_result, E2))


@pytest.mark.mutation
def test_mutant_deletion_check_truncated_to_top_k_stops_looking(tmp_path) -> None:
    """`hits()` instead of `hits(k=None)`: the sixth hit is still a reference."""
    mutant = load_mutant(tmp_path, "deletion_topk",
                         (DELETION_ALL_HITS, "    for hit in result.hits():\n"))
    # Six events, all scoring 1.0 on the structured path, so `(-score, ref)` orders
    # them by ref — and the deleted episode's event has the last ref.
    seqs = list(range(1, 7))
    root = plant(tmp_path / "oar",
                 events=[_event(s, "小明") for s in seqs],
                 episodes=[episode(E1, seqs[:5], "前五", unit(0)),
                           episode(E2, seqs[5:], "第六", unit(1))])
    r_result = rc.Retriever(root, encoder=lambda t: unit(1)).query("小明")
    structured = [h for h in r_result.hits(k=None) if h.path == "structured"]
    assert len(structured) == 6 and structured[-1].episode == E2, \
        "the fixture must put E2's hit past the top five"
    assert any("structured still returns" in f
               for f in rc.deletion_findings(r_result, E2))
    m_result = mutant.Retriever(root, encoder=lambda t: unit(1)).query("小明")
    assert not any("structured still returns" in f
                   for f in mutant.deletion_findings(m_result, E2))


@pytest.mark.mutation
def test_mutant_bm25_over_a_set_loses_term_frequency(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "pack_set",
                         (PACK, "    return tuple(sorted(set(bigrams(text))))\n"))
    # The fixture has to isolate term frequency from *document length*, which BM25
    # also normalises by. These two have the same distinct-bigram set — so under the
    # mutant both documents have identical length and identical tf, and the scores
    # coincide exactly — while the real tokeniser sees three terms against two, with
    # 好累 twice against once.
    twice, once = "好累好累", "好累好"
    assert set(sa.bigrams(twice)) == set(sa.bigrams(once))
    assert sa.bigrams(twice).count("好累") == 2
    assert sa.bigrams(once).count("好累") == 1
    m_a, m_b = mutant.bm25("好累", [twice, once])
    assert m_a == pytest.approx(m_b), "the mutant cannot tell them apart"
    r_a, r_b = rc.bm25("好累", [twice, once])
    assert r_a > r_b, (r_a, r_b)
    # Still a working scorer otherwise, so "scores got worse" could not have caught it.
    assert mutant.bm25("好累", [twice, "無關文字"])[0] > 0.0


@pytest.mark.mutation
def test_mutant_cosine_scores_a_dimension_mismatch_as_a_number(tmp_path) -> None:
    """An encoder swapped without rebuilding the index: a path that runs and is
    blind, which is the combination the status field exists to prevent."""
    mutant = load_mutant(tmp_path, "cosine_dims",
                         (COSINE_DIMS, "    if False and len(a) != len(b):\n"))
    assert isinstance(mutant.cosine([1.0, 0.0], [1.0, 0.0, 0.0]), float)
    with pytest.raises(rc.RecallError, match="dimension mismatch"):
        rc.cosine([1.0, 0.0], [1.0, 0.0, 0.0])
    root = full(tmp_path / "store")
    m_result = mutant.Retriever(root, encoder=lambda t: [1.0, 0.0]).query("自我介紹")
    assert m_result.status_of("semantic") == mutant.RAN
    with pytest.raises(rc.RecallError, match="dimension mismatch"):
        rc.Retriever(root, encoder=lambda t: [1.0, 0.0]).query("自我介紹")


@pytest.mark.mutation
def test_mutant_empty_query_is_accepted(tmp_path) -> None:
    """The two **scored** paths go silently empty while reporting that they ran.

    Not all five: `structured` is a filter and returns the same records for any query,
    so it still finds E2. The claim is narrower than "everything returns nothing" and
    the test says which paths, because the wide version would have been satisfied by
    the fixture rather than by the refusal.
    """
    mutant = load_mutant(tmp_path, "empty_query",
                         (EMPTY_QUERY, "        if not isinstance(text, str):\n"))
    root = full(tmp_path / "store")
    m_result = mutant.Retriever(root, encoder=lambda t: unit(1)).query("   ")
    assert m_result.census()[mutant.RAN] == 5
    for silent in ("card", "lexical"):
        assert m_result.status_of(silent) == mutant.RAN
        assert [h for h in m_result.hits(k=None) if h.path == silent] == [], silent
    m_findings = mutant.deletion_findings(m_result, E2)
    assert not any(f.startswith(("card still", "lexical still")) for f in m_findings)

    # The real module refuses the query outright; asked properly, both paths cite E2.
    with pytest.raises(rc.RecallError, match="empty query"):
        rc.Retriever(root, encoder=lambda t: unit(1)).query("   ")
    r_findings = rc.deletion_findings(
        rc.Retriever(root, encoder=lambda t: unit(1)).query("花生 小明"), E2)
    assert any(f.startswith("card still") for f in r_findings), r_findings
    assert any(f.startswith("lexical still") for f in r_findings), r_findings


@pytest.mark.mutation
def test_mutant_declared_artifact_census_cannot_see_a_sixth_file(tmp_path) -> None:
    """The five-name-list defect, one level down: a written-down artifact list."""
    mutant = load_mutant(tmp_path, "declared_artifacts", (
        ARTIFACT_CENSUS,
        "    found = [EVENTS_ARTIFACT, EPISODES_ARTIFACT, CARD_ARTIFACT, "
        "RPS_ARTIFACT]\n"))
    root = full(tmp_path / "store")
    assert mutant.coverage_findings(root) == [] == rc.coverage_findings(root)
    (root / "skills_seen.json").write_text("{}", encoding="utf-8")
    assert mutant.coverage_findings(root) == [], "the mutant cannot see the new file"
    assert len(rc.coverage_findings(root)) == 1


@pytest.mark.mutation
def test_mutant_duplicate_episode_ids_are_accepted(tmp_path) -> None:
    """Two entries under one id: a deletion removes one and leaves one."""
    mutant = load_mutant(tmp_path, "dupe_ids",
                         (EPISODE_DUPES, "    dupes = []\n"))
    root = plant(tmp_path / "oar",
                 episodes=[episode(E1, [1], "第一份", unit(0)),
                           episode(E1, [2], "第二份", unit(0))])
    assert len(mutant.load_episodes(root / rc.EPISODES_ARTIFACT)) == 2
    with pytest.raises(rc.RecallError, match="duplicate episode id"):
        rc.load_episodes(root / rc.EPISODES_ARTIFACT)


@pytest.mark.mutation
def test_mutant_overlapping_attribution_silently_takes_the_last(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "attrib_overlap",
                         (ATTRIB_OVERLAP, "            if False:\n"))
    entries = [episode(E1, [1, 2], "a"), episode(E2, [2, 3], "b")]
    assert mutant.attribution(entries)[2] == E2, "one reference silently lost"
    with pytest.raises(rc.RecallError, match="partition the stream"):
        rc.attribution(entries)


@pytest.mark.mutation
def test_mutant_unembedded_always_reports_nothing(tmp_path) -> None:
    """An index covering half the episodes answers every query and is wrong about
    all of them; the regression line would move with nothing to say why."""
    mutant = load_mutant(tmp_path, "unembedded_empty",
                         (UNEMBEDDED, "        return ()\n"),
                         ('                     if e.get("embedding") is None)\n', ""))
    root = plant(tmp_path / "oar",
                 episodes=[episode(E1, [1], "有向量", unit(0)),
                           episode(E2, [2], "沒有向量")])
    assert mutant.Retriever(root).unembedded() == ()
    assert rc.Retriever(root).unembedded() == (E2,)
