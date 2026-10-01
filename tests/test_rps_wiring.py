"""R1.2 — the RPS pair: rounds reach the disk, the accumulated model reaches the game.

`game/game_node.py` has published `/game/rps_round` and subscribed latched to
`/memory/rps_table` since S2. Nothing was on the other end of either. So the transition
table lived in RAM, was discarded on every restart, and `game/opponent.py`'s own gate kept
the robot on the 1/3 coin for the first ~120 rounds of every session — 81 order-2 contexts
at `min_context_count = 3`, which is arithmetic and not a tuning choice. "It really plays
against you" was true of one process's lifetime and false of the robot.

`infra/graph_gaps.tsv` counted those two topics as **one** gap from both sides, and its
reason column said why: wiring one end alone would look like progress and change nothing,
because a table nobody persists is a table nobody can restore. Both rows are deleted in
this PR, together, which is the registry's stated rule.

**The honest boundary, the same sentence as every R1 PR.** Every assertion here about a
node file is made against its **source**, because both node files import `rclpy` and §6.1
requires this suite to run on a laptop with no ROS. These tests prove the *shape* of the
wiring — that the subscription is declared, that the QoS matches the subscriber's, that
the snapshot is handed to the writer before the model counts the round — and they do
**not** prove a message flows. That is L4, and it needs the security group.

**What is checked behaviourally, and it is more than the shape here.** `memory/rps_state.py`
imports no ROS, so the whole disk-and-wire asymmetry is exercised for real: rounds are
appended, replayed, published, and the replayed model is compared against the live one. The
one claim that could not be tested any other way is `Opponent`'s: that a model built by N
`observe()` calls and the same model rebuilt from those N rounds agree **to the last bit**.
`memory/rps_state.py`'s docstring rests its whole safety argument on that (the wire form is
derived and never stored), so it is asserted rather than cited.

**The claim this file deliberately does NOT make.** `store.write_snapshot` enqueues and
returns; the `os.replace` happens on the writer thread. So "the round is durable before the
model counts it" is false, and the parked draft of `rps_state.py` said it. What is pinned
below is the weaker true statement: the drift is bounded to **one round**, and the round
that discovers an asynchronous failure is the one refused. A test that asserted the
stronger claim would pass — nothing here can see the writer thread — which is exactly why
it would be the more dangerous test to write.

Every derivation is a plain function of a parsed tree, imported from
`tests/test_profile_wiring.py` rather than re-implemented: two readers of one format drift
when each only checks itself, and these are the same node files.
"""
from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from game import opponent as opp  # noqa: E402
from memory import recall as rc  # noqa: E402
from memory import rps_state as rps  # noqa: E402
from memory import store as st  # noqa: E402
# The derivations, not copies of them. `parse`, `constants`, `topic_edges` and
# `qos_keywords` already read these two files for R1.1; a second implementation here
# would be a second reader of one format, free to agree with itself while disagreeing
# with the gate that ships.
from test_profile_wiring import (  # noqa: E402
    MEMORY_SRC, MEMORY_TREE, constants, parse, qos_keywords, stmt_index, topic_edges)

GAME_PY = REPO_ROOT / "game" / "game_node.py"
GAME_SRC = GAME_PY.read_text(encoding="utf-8")
GAME_TREE, GAME = parse(GAME_SRC, "GameNode")
MEMORY = parse(MEMORY_SRC, "MemoryNode")[1]

# The audit kind for a boot-time table restore, read off the authority rather than typed.
RESTORED_KIND = next(k for k in st.AUDIT_FIELDS if k == "rps_restored")

MOVES = ("rock", "paper", "scissors")


# ------------------------------------------------------------------ derivations
def topic_literals(tree: ast.Module, methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Every topic string a node's `__init__` names, resolved through its constants.

    Both sides of a pair have to agree on a name that appears twice in the tree, and a
    check that read only one file could not notice a rename.
    """
    subs, pubs = topic_edges(tree, methods)
    return set(subs) | set(pubs)


def block_position(fn: ast.FunctionDef, predicate) -> tuple[int, int] | None:
    """`(block identity, index in that block)` for the first statement matching `predicate`.

    `test_profile_wiring.stmt_index` indexes the *top level* of a function body, which is
    the right granularity there and the wrong one here: both calls this file cares about
    live inside one `try:` in `on_rps_round`, so a top-level index collapses them to the
    same number and the order check silently becomes an existence check — it reported the
    healthy tree as broken, which is the harmless direction of a real blind spot.

    So each statement's nested blocks are searched before the statement itself, and the
    answer carries the identity of the containing list. Two matches are only comparable
    when that identity is the same; positions across different blocks are not an order.
    """
    def search(body: list[ast.stmt]) -> tuple[int, int] | None:
        for i, stmt in enumerate(body):
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    got = search(inner)
                    if got is not None:
                        return got
            if any(predicate(n) for n in ast.walk(stmt)):
                return (id(body), i)
        return None
    return search(fn.body)


def calls(attr: str):
    return lambda n: (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == attr)


def snapshot_before_commit(methods: dict[str, ast.FunctionDef]) -> bool:
    """True when `write_snapshot` is called before `commit` inside `on_rps_round`.

    Statement order, not merely presence. Both calls exist in either arrangement, and the
    wrong arrangement is the one that persists a refused round as though it had happened
    — a defect no field on disk would record.
    """
    fn = methods["on_rps_round"]
    snap = block_position(fn, calls("write_snapshot"))
    commit = block_position(fn, calls("commit"))
    if snap is None or commit is None:
        return False
    assert snap[0] == commit[0], (
        "write_snapshot and commit are no longer in the same block, so their order is "
        "conditional and this derivation cannot decide it — read the method")
    return snap[1] < commit[1]


def publish_is_guarded_by_rounds(methods: dict[str, ast.FunctionDef]) -> bool:
    """True when the boot publish of the table sits under a test of `self.table.rounds`.

    An unconditional boot publish is the mutant that matters most and looks most correct:
    an `Opponent` with zero observations is exactly the model `game_node` already built,
    so the latched empty sample carries no information and *does* make `on_rps_table` log
    a restore that did not happen — to every subscriber that joins later, too.
    """
    for node in ast.walk(methods["__init__"]):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "self.table.rounds" not in test:
            continue
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "publish"
               and "pub_table" in ast.unparse(c.func.value)
               for c in ast.walk(node)):
            return True
    return False


def restore_record_before_subscriptions(methods: dict[str, ast.FunctionDef]) -> bool:
    """True when the `rps_restored` append precedes every `create_subscription`."""
    fn = methods["__init__"]
    append = stmt_index(fn, lambda n: isinstance(n, ast.Dict) and any(
        isinstance(k, ast.Constant) and k.value == "kind"
        and isinstance(v, ast.Constant) and v.value == RESTORED_KIND
        for k, v in zip(n.keys, n.values)))
    first_sub = stmt_index(fn, lambda n: isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute)
                           and n.func.attr == "create_subscription")
    return append >= 0 and first_sub >= 0 and append < first_sub


def loaded_in_main(tree: ast.Module) -> bool:
    """True when `rps_state.load` is called in `main`, not in the constructor.

    `load` can refuse a present-but-unreadable table, and a refusal raised inside a
    constructed `Node` is a refusal that has already created publishers.
    """
    main = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main is not None, "no main() in memory_node.py"
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "load" and "rps" in ast.unparse(n.func.value)
               for n in ast.walk(main))


def played(rounds: int, *, seed: int = 7) -> list[dict]:
    """A deterministic sequence of legal rounds, long enough to open the gate."""
    rng = random.Random(seed)
    return [rps.round_record(rng.choice(MOVES), rng.choice(MOVES))
            for _ in range(rounds)]


# ============================================================ the pair, both ends
def test_the_two_topic_names_are_the_same_strings_in_both_files() -> None:
    """A rename on one side is the failure this pair is most exposed to.

    Both names are constants in both files, so a typo produces two nodes that start
    cleanly, log cleanly, and never exchange a message. `infra/ros_census.py` would report
    the two halves as separate one-ended topics — which is the state this PR just left.
    """
    memory_topics = topic_literals(MEMORY_TREE, MEMORY)
    game_topics = topic_literals(GAME_TREE, GAME)
    for name in ("/game/rps_round", "/memory/rps_table"):
        assert name in memory_topics, f"memory_node no longer touches {name}"
        assert name in game_topics, f"game_node no longer touches {name}"
    consts = constants(MEMORY_TREE)
    assert consts["RPS_ROUND_TOPIC"] == "/game/rps_round"
    assert consts["RPS_TABLE_TOPIC"] == "/memory/rps_table"


def test_the_ends_are_opposite_ways_round() -> None:
    """memory_node subscribes to rounds and publishes the table; game_node the reverse.

    Asserted rather than assumed because both nodes create both kinds of endpoint, and a
    pair wired the same way round on both sides is two publishers and no subscriber.
    """
    m_subs, m_pubs = topic_edges(MEMORY_TREE, MEMORY)
    g_subs, g_pubs = topic_edges(GAME_TREE, GAME)
    assert "/game/rps_round" in m_subs and "/game/rps_round" in g_pubs
    assert "/memory/rps_table" in m_pubs and "/memory/rps_table" in g_subs
    assert "/memory/rps_table" not in m_subs, "memory_node subscribes to its own table"
    assert "/game/rps_round" not in m_pubs, "memory_node publishes the rounds it consumes"


def test_the_table_is_published_with_the_durability_the_game_subscribes_with() -> None:
    """`TABLE_QOS` against `game_node`'s `LATCHED`, keyword by keyword where they overlap.

    The asymmetry is deliberate and so is the comparison's shape: `game_node.LATCHED`
    declares `depth` and `durability` only, so the two settings memory_node adds
    (`history`, `reliability`) have no counterpart to disagree with. What must match is
    every keyword the subscriber actually states — a volatile subscriber against a latched
    publisher is a `game_node` that plays the coin for its whole session while the log on
    the other side says the table was published.
    """
    mine = qos_keywords(MEMORY_TREE, "TABLE_QOS")
    theirs = qos_keywords(GAME_TREE, "LATCHED")
    assert theirs, "game_node's LATCHED is gone — this comparison measures nothing"
    for key, value in theirs.items():
        assert mine.get(key) == value, (
            f"QoS {key}: publisher says {mine.get(key)!r}, subscriber says {value!r}")
    m_subs, _ = topic_edges(GAME_TREE, GAME)
    assert m_subs["/memory/rps_table"] == "LATCHED"
    _, m_pubs = topic_edges(MEMORY_TREE, MEMORY)
    assert m_pubs["/memory/rps_table"] == "TABLE_QOS"


def test_the_two_qos_objects_are_not_one_object() -> None:
    """`TABLE_QOS` is its own assignment, not an alias of `PROFILE_QOS`.

    They hold the same four settings today and they are two decisions: the card is
    published once per boot, the table after every round, so `depth` is a genuinely
    separate choice. An alias would let one be re-tuned while silently re-tuning the other.
    """
    assert qos_keywords(MEMORY_TREE, "TABLE_QOS") == \
        qos_keywords(MEMORY_TREE, "PROFILE_QOS"), \
        "the two agree today; if that changed on purpose, this assertion is the record"
    names = [t.id for stmt in MEMORY_TREE.body if isinstance(stmt, ast.Assign)
             and isinstance(stmt.value, ast.Call)
             and ast.unparse(stmt.value.func) == "QoSProfile"
             for t in stmt.targets if isinstance(t, ast.Name)]
    assert names.count("TABLE_QOS") == 1, (
        f"TABLE_QOS is not its own QoSProfile(...) call: {names}")


# ============================================== the order the node does things in
def test_the_snapshot_is_handed_over_before_the_model_counts_the_round() -> None:
    assert snapshot_before_commit(MEMORY), (
        "on_rps_round lets the model count a round before the writer has the file")


def test_the_boot_publish_is_guarded_by_having_rounds() -> None:
    assert publish_is_guarded_by_rounds(MEMORY)


def test_the_restore_record_cannot_be_overtaken_by_an_arriving_round() -> None:
    """`rps_restored` is appended before either subscription exists.

    Same construction as `profile_loaded`, and the same reason: the record saying what the
    robot knew at boot must not be able to land behind a record from a node that started
    first. Here the risk is concrete — the first round could arrive during `__init__`.
    """
    assert restore_record_before_subscriptions(MEMORY)


def test_the_table_is_loaded_where_a_refusal_can_still_stop_the_node() -> None:
    assert loaded_in_main(MEMORY_TREE)
    assert "RpsStateError" in MEMORY_SRC, \
        "memory_node no longer catches the one type rps_state.load raises"


def test_the_restore_record_is_one_the_single_writer_accepts(tmp_path: Path) -> None:
    """Built here and handed to the real `check_audit`, not compared to a copy of the row.

    This is the check that would have caught R1.1's finding — a producer whose record the
    declaration refuses — before the node ran anywhere.
    """
    record = {"kind": RESTORED_KIND, "existed": True, "rounds": 3, "orphans": 3}
    st.check_audit(record)
    surplus = {**record, "moves": ["rock"]}
    with pytest.raises(st.StoreError, match="undeclared field"):
        st.check_audit(surplus)


def test_the_record_the_node_writes_declares_no_field_the_row_omits() -> None:
    """The literal in `memory_node.py`, read by AST — `profile_loaded`'s own control."""
    found = []
    for node in ast.walk(MEMORY["__init__"]):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {k.value: v for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        kind = pairs.get("kind")
        if isinstance(kind, ast.Constant) and kind.value == RESTORED_KIND:
            found.append(sorted(pairs))
    assert found, (f"no {RESTORED_KIND!r} literal in memory_node.__init__ — either the "
                   "producer moved, in which case this control measures nothing, or the "
                   "kind is unproduced and its AUDIT_FIELDS row is dead")
    for keys in found:
        surplus = sorted(set(keys) - set(st.AUDIT_IMPLICIT_FIELDS)
                         - set(st.AUDIT_FIELDS[RESTORED_KIND]))
        assert not surplus, f"memory_node writes undeclared field(s) {surplus}"
        missing = sorted(set(st.AUDIT_FIELDS[RESTORED_KIND]) - set(keys))
        assert not missing, (
            f"the row declares {missing} and the producer never writes it — a declared "
            "field no record carries is a field a reader may not rely on")


# ========================================== the disk/wire asymmetry, for real
def test_the_two_formats_are_genuinely_incompatible_in_both_directions() -> None:
    """The premise of the whole module: neither end can hold the other's shape.

    Asserted because it is the reason `rps_state.py` exists, and a premise nothing checks
    is a premise that gets "simplified" away. Write the model's shape to disk and
    `recall.load_rounds` raises; write recall's shape to the wire and `from_dict` raises.
    """
    model = rps.replay(played(5))
    with pytest.raises(rc.RecallError):
        rc.parse_rounds(model.to_dict(), where="the model's own shape")
    with pytest.raises(Exception):
        opp.Opponent.from_dict({"version": rps.VERSION, "rounds": played(5)},
                               rng=random.Random(0))


def test_the_wire_form_is_derived_from_the_rounds_to_the_last_bit() -> None:
    """A model fed N rounds and the same model rebuilt from those rounds are equal.

    This is the claim that makes "derive the wire form, store only the rounds" safe, and
    it is only true because `Opponent.__post_init__` sums with `math.fsum` rather than a
    running `+=`. Asserted on a replayed model rather than cited from a comment.
    """
    rounds = played(40)
    live = opp.Opponent(rng=random.Random(0))
    for entry in rounds:
        live.observe(entry["user"], entry["robot"])
    replayed = rps.replay(rounds)
    assert replayed.to_dict() == live.to_dict()
    assert replayed.evidence_matches_recomputation()
    round_tripped = opp.Opponent.from_dict(json.loads(
        rps.RpsState(model=replayed, rounds=rounds, existed=True).payload()),
        rng=random.Random(0))
    assert round_tripped.to_dict() == live.to_dict()
    assert live.gate_open() == round_tripped.gate_open()


def test_enough_replayed_rounds_open_the_gate_that_a_fresh_model_keeps_shut() -> None:
    """The behavioural point of the whole PR, stated as a difference.

    Restoring is only worth doing if the restored model plays differently, and `Opponent`'s
    own gate is what decides that. A fresh model is shut; a model replayed from a long
    enough table is open. Without this, every assertion above could hold while the robot
    still played the 1/3 coin.
    """
    assert not opp.Opponent(rng=random.Random(0)).gate_open()
    rounds = [rps.round_record("rock", "scissors") for _ in range(400)]
    assert rps.replay(rounds).gate_open(), (
        "400 identical rounds do not open the gate — either min_context_count moved or "
        "replay is not feeding observe()")


def test_a_round_this_module_accepts_cannot_be_refused_by_the_model_afterwards() -> None:
    """`round_record` and `Opponent.observe` must agree on every pair, legal and not.

    This agreement is what allows the snapshot to be handed over before `commit`: a round
    the file already contains that the model then rejects is the one arrangement neither
    side could recover from.
    """
    for user in MOVES:
        for robot in MOVES:
            record = rps.round_record(user, robot)
            opp.Opponent(rng=random.Random(0)).observe(user, robot)
            assert record["outcome"] == opp.outcome_for_user(user, robot)
    for user, robot in (("", "rock"), ("ROCK", "rock"), ("rock", "lizard"),
                        ("rock", ""), ("石頭", "rock"), ("rock ", "rock")):
        with pytest.raises(opp.OpponentError):
            rps.round_record(user, robot)
        with pytest.raises(opp.OpponentError):
            opp.Opponent(rng=random.Random(0)).observe(user, robot)


def test_the_writer_refuses_a_file_the_deletion_reader_would_reject() -> None:
    """`snapshot` validates through `recall.parse_rounds`, the reader a deletion depends on.

    Two readers of one format, so assert the writer consults the other one rather than
    that each is self-consistent. Both directions `parse_rounds` actually closes are
    exercised: a round that is not an object, and an `episode` that is not a minted id —
    the second being a value `round_record` itself will accept, so the cross-check is the
    only thing standing between a hand-passed episode id and the disk.
    """
    good = rps.snapshot(played(3))
    assert set(good) == {"version", "rounds"}
    with pytest.raises(rps.RpsStateError, match="a deletion depends on"):
        rps.snapshot(["not a round"])
    hand_stamped = rps.round_record("rock", "paper", episode="ep-not-minted")
    assert hand_stamped["episode"] == "ep-not-minted", \
        "round_record now validates the episode id itself, so this is the wrong writer"
    with pytest.raises(rps.RpsStateError, match="not a minted episode id"):
        rps.snapshot([hand_stamped])


def test_the_writer_cross_check_does_not_promise_the_round_is_replayable() -> None:
    """The boundary of the check above, asserted so it is not read as more than it is.

    `parse_rounds` leaves the *round entry's* fields open on purpose (an unknown field on
    a removed round goes with the round). So a round missing `robot` passes `snapshot` and
    is refused later by `replay` — at the next boot, not at the write. Nothing produces
    such a round today, because `candidate` goes through `round_record`, and that is the
    guarantee: the shape check is not what makes rounds replayable, `round_record` is.
    """
    half = {"user": "rock", "outcome": "win", "episode": None}
    rps.snapshot([half])                                    # accepted by the writer
    with pytest.raises(rps.RpsStateError, match="cannot be replayed"):
        rps.replay([half])                                  # refused at the next boot


def test_every_round_written_now_is_unattributed_and_says_so() -> None:
    """`episode` is present and null until R1.3, and `orphans` counts exactly those.

    The field is written now because a field set is free to mandate only before the first
    writer exists, and R1.2 is that writer. The consequence is designed rather than
    discovered: `recall._run_rps` maps a null episode to ORPHAN.
    """
    state = rps.RpsState(model=opp.Opponent(rng=random.Random(0)))
    for user, robot in (("rock", "paper"), ("paper", "paper")):
        record, whole = state.candidate(user, robot)
        assert "episode" in record and record["episode"] is None
        state.commit(record)
        assert whole["rounds"][-1] == record
    assert state.orphans() == len(state.rounds) == 2
    assert rc.ORPHAN == "orphan", "the label recall maps a null episode to has moved"


def test_a_missing_table_is_not_an_error_and_an_unreadable_one_is(tmp_path: Path) -> None:
    """The two states a boot can find, and they are not the same state.

    A missing file is a first-ever run. A present-and-unreadable file stops the node,
    because the alternative is the first round atomically replacing something a human
    could still have repaired.
    """
    empty = rps.load(tmp_path)
    assert empty.rounds == [] and empty.existed is False
    assert not empty.model.gate_open()

    (tmp_path / rps.ARTIFACT).write_text('{"version": 1, "rounds": "nope"}',
                                         encoding="utf-8")
    with pytest.raises(rps.RpsStateError, match="present and unreadable"):
        rps.load(tmp_path)


def test_a_table_read_and_empty_is_distinguishable_from_no_table(tmp_path: Path) -> None:
    """`existed` is the flag that separates two causes of `rounds == 0`.

    R1.1b's finding one layer up: a zero with two causes needs the flag, and both produce
    the same publish (none), so only the record can tell them apart afterwards.
    """
    (tmp_path / rps.ARTIFACT).write_text('{"version": 1, "rounds": []}', encoding="utf-8")
    read_and_empty = rps.load(tmp_path)
    assert read_and_empty.rounds == [] and read_and_empty.existed is True
    assert rps.load(tmp_path / "nowhere").existed is False
    assert "existed" in st.AUDIT_FIELDS[RESTORED_KIND], (
        "the flag is not on the record, so audit.jsonl cannot tell the two causes apart")


def test_a_round_that_cannot_be_replayed_aborts_the_whole_load() -> None:
    """Not skipped. A partially restored table is a model trained on a subset nobody chose.

    Skipping is also the more expensive failure: the resulting counts look exactly like a
    person who played differently, and nothing would ever say otherwise.
    """
    rounds = played(4)
    rounds[2] = {"user": "rock", "robot": "lizard", "outcome": "win", "episode": None}
    with pytest.raises(rps.RpsStateError, match="round 2 cannot be replayed"):
        rps.replay(rounds)
    with pytest.raises(rps.RpsStateError, match="round 1 is not an object"):
        rps.replay([played(1)[0], "not a dict"])


def test_the_drift_between_model_and_disk_is_bounded_to_one_round(tmp_path: Path) -> None:
    """The weaker true claim, pinned so the stronger false one cannot come back.

    `write_snapshot` enqueues; the `os.replace` is the writer thread's. So an asynchronous
    failure is discovered by `Store._raise_if_failed` on entry to the NEXT call — which
    means the model can be one round ahead of the disk and no further, because the round
    that discovers the failure raises before its own `commit`.

    Simulated by failing the store, which is what a full disk looks like from here: the
    next `write_snapshot` raises, and the assertion is that `candidate`/`commit` are split
    such that the raise lands before the model has counted anything.
    """
    # `_allow_ephemeral=True` is the visible exemption tests/test_store.py uses:
    # tmp_path is under the system temp dir, which the location gate refuses for
    # real data and which is exactly right for a round trip nobody keeps.
    store = st.Store(root=tmp_path, _allow_ephemeral=True)
    try:
        state = rps.load(tmp_path)
        record, whole = state.candidate("rock", "paper")
        store.write_snapshot(rps.ARTIFACT, whole)
        state.commit(record)
        assert len(state.rounds) == 1

        store._failure = OSError("no space left on device")     # the async failure
        second, whole2 = state.candidate("paper", "scissors")
        with pytest.raises(st.StoreError, match="writer thread died"):
            store.write_snapshot(rps.ARTIFACT, whole2)
        # The raise happened before `commit`, so the model is still where the disk is.
        assert len(state.rounds) == 1, (
            "the model counted a round whose snapshot was refused — the split between "
            "candidate() and commit() is not doing the one thing it exists for")
        assert second["user"] == "paper"
    finally:
        store._failure = None
        store.close()


def test_the_round_trip_through_a_real_store_is_the_table_the_game_would_restore(
        tmp_path: Path) -> None:
    """End to end without ROS: rounds -> Store -> disk -> load -> replay -> wire payload.

    The one test here that touches every piece the node touches, in the node's order, so a
    change to any of the four formats involved shows up as one failure rather than as a
    passing test per layer.
    """
    # `_allow_ephemeral=True` is the visible exemption tests/test_store.py uses:
    # tmp_path is under the system temp dir, which the location gate refuses for
    # real data and which is exactly right for a round trip nobody keeps.
    store = st.Store(root=tmp_path, _allow_ephemeral=True)
    try:
        state = rps.load(tmp_path)
        for user, robot in [("rock", "paper"), ("paper", "paper"), ("scissors", "rock"),
                            ("rock", "rock"), ("paper", "scissors")]:
            record, whole = state.candidate(user, robot)
            store.write_snapshot(rps.ARTIFACT, whole)
            state.commit(record)
        store.close()

        restored = rps.load(tmp_path)
        assert restored.existed is True
        assert len(restored.rounds) == 5
        assert restored.orphans() == 5
        assert restored.model.to_dict() == state.model.to_dict()
        adopted = opp.Opponent.from_dict(json.loads(restored.payload()),
                                         rng=random.Random(0))
        assert adopted.to_dict() == state.model.to_dict()
        assert restored.observations() == {k: sum(sum(c) for c in adopted.tables[k].values())
                                           for k in sorted(adopted.orders)}
    finally:
        store.close()


def test_the_artifact_name_has_one_creator() -> None:
    """`rps_state.ARTIFACT` is `recall`'s constant, not a second spelling of the name.

    `memory/erase.py` rebuilds this file by name, so a second spelling here would be a
    second thing to keep in step with the deletion path.
    """
    assert rps.ARTIFACT is rc.RPS_ARTIFACT
    src = (REPO_ROOT / "memory" / "rps_state.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value == rc.RPS_ARTIFACT]
    assert not literals, (
        f"rps_state.py writes {rc.RPS_ARTIFACT!r} as a literal, so the name now has two "
        "creators and erase's rebuild can be kept in step with only one of them")


def test_the_snapshot_name_is_one_the_store_accepts() -> None:
    """A plain file name, and not a protected one — both would refuse at runtime."""
    assert "/" not in rps.ARTIFACT and not rps.ARTIFACT.startswith(".")
    assert not st.is_locked(rps.ARTIFACT), (
        f"{rps.ARTIFACT} is a protected snapshot, so every round would be refused until "
        "unlock_for_s5 — which only accepts the profile card")


# ================================================================ layer L3: mutants
# Each mutant patches the source of a node or of `memory/rps_state.py` and requires the
# derivation above to change its answer. The anchor assertion is separate from the
# outcome, so a stale mutant reports "anchor gone" rather than "mutant survived".

def mutate_memory(old: str, new: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    assert old in MEMORY_SRC, f"mutant anchor no longer in memory_node.py: {old!r}"
    return parse(MEMORY_SRC.replace(old, new, 1), "MemoryNode")


def mutate_game(old: str, new: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    assert old in GAME_SRC, f"mutant anchor no longer in game_node.py: {old!r}"
    return parse(GAME_SRC.replace(old, new, 1), "GameNode")


@pytest.mark.mutation
def test_no_mutant_control() -> None:
    """§6.0 rule 2: with no substitution applied every derivation reports the healthy
    value. A harness that failed for all inputs would "kill" every mutant below while
    proving nothing about any of them."""
    tree, methods = mutate_memory("class MemoryNode(Node):", "class MemoryNode(Node):")
    gtree, gmethods = mutate_game("class GameNode(Node):", "class GameNode(Node):")
    m_subs, m_pubs = topic_edges(tree, methods)
    g_subs, g_pubs = topic_edges(gtree, gmethods)
    assert m_subs["/game/rps_round"] == "10"
    assert m_pubs["/memory/rps_table"] == "TABLE_QOS"
    assert g_subs["/memory/rps_table"] == "LATCHED"
    assert "/game/rps_round" in g_pubs
    assert qos_keywords(tree, "TABLE_QOS")["durability"] == \
        qos_keywords(gtree, "LATCHED")["durability"]
    assert snapshot_before_commit(methods)
    assert publish_is_guarded_by_rounds(methods)
    assert restore_record_before_subscriptions(methods)
    assert loaded_in_main(tree)
    # The table is passed in, not built in the constructor — the third argument, by name,
    # so a positional shuffle between `store`, `card` and `table` cannot pass silently.
    assert [a.arg for a in methods["__init__"].args.args] == \
        ["self", "store", "card", "table"]


@pytest.mark.mutation
def test_mutant_the_model_counts_the_round_before_the_writer_has_it() -> None:
    """MUTANT: `commit` moves above `write_snapshot`.

    The arrangement in which a refused round is already in the counts, and the next round
    persists it as though it had happened. Both calls are still present, which is why the
    derivation reads statement order and not membership.
    """
    _tree, methods = mutate_memory(
        "            seq = self.store.write_snapshot(rps.ARTIFACT, whole)\n"
        "            self.table.commit(record)",
        "            self.table.commit(record)\n"
        "            seq = self.store.write_snapshot(rps.ARTIFACT, whole)")
    assert not snapshot_before_commit(methods), \
        "the order check cannot tell commit-then-write from write-then-commit"


@pytest.mark.mutation
def test_mutant_the_empty_table_is_published_at_boot() -> None:
    """MUTANT: drop the `if self.table.rounds:` guard.

    The most plausible-looking mutant in this file. It publishes a model with zero
    observations — bit-identical to the one `game_node` just built — and `on_rps_table`
    logs a restore that did not happen, latched, for every later subscriber.
    """
    _tree, methods = mutate_memory("        if self.table.rounds:\n",
                                   "        if True:\n")
    assert not publish_is_guarded_by_rounds(methods), \
        "the guard check passes an unconditional boot publish"


@pytest.mark.mutation
def test_mutant_the_restore_record_lands_after_the_subscriptions() -> None:
    """MUTANT: append `rps_restored` after both `create_subscription` calls.

    Then a round arriving during `__init__` takes a lower sequence number than the record
    describing the table it was added to, and "what did the robot know at boot" is
    answered by a file whose order says otherwise.
    """
    _tree, methods = mutate_memory(
        '        self.store.append("audit", {\n'
        '            "kind": "rps_restored",',
        '        self.create_subscription(String, RPS_ROUND_TOPIC, self.on_rps_round, 10)\n'
        '        self.store.append("audit", {\n'
        '            "kind": "rps_restored",')
    assert not restore_record_before_subscriptions(methods), \
        "the ordering check cannot see a subscription created before the record"


@pytest.mark.mutation
def test_mutant_the_table_is_published_volatile() -> None:
    """MUTANT: `TABLE_QOS` becomes VOLATILE while `game_node` still subscribes latched.

    Invisible in production for the reason the profile card's twin mutant is: memory_node
    publishes and logs, game_node receives nothing, and the coin plays out the session.
    """
    tree, _methods = mutate_memory(
        "TABLE_QOS = QoSProfile(\n    depth=1,\n"
        "    history=HistoryPolicy.KEEP_LAST,\n"
        "    reliability=ReliabilityPolicy.RELIABLE,\n"
        "    durability=DurabilityPolicy.TRANSIENT_LOCAL,\n)",
        "TABLE_QOS = QoSProfile(\n    depth=1,\n"
        "    history=HistoryPolicy.KEEP_LAST,\n"
        "    reliability=ReliabilityPolicy.RELIABLE,\n"
        "    durability=DurabilityPolicy.VOLATILE,\n)")
    assert qos_keywords(tree, "TABLE_QOS")["durability"] != \
        qos_keywords(GAME_TREE, "LATCHED")["durability"], \
        "the QoS comparison cannot tell a volatile publisher from a latched one"


@pytest.mark.mutation
def test_mutant_the_round_topic_is_renamed_on_one_side() -> None:
    """MUTANT: one end of the pair renames the topic. Both nodes still start cleanly."""
    tree, _methods = mutate_memory('RPS_ROUND_TOPIC = "/game/rps_round"',
                                   'RPS_ROUND_TOPIC = "/game/rps_rounds"')
    assert "/game/rps_round" not in topic_literals(tree, parse(
        MEMORY_SRC.replace('RPS_ROUND_TOPIC = "/game/rps_round"',
                           'RPS_ROUND_TOPIC = "/game/rps_rounds"', 1),
        "MemoryNode")[1]), "the pinned copy is not pinned"


@pytest.mark.mutation
def test_mutant_the_table_topic_is_renamed_on_one_side() -> None:
    """MUTANT: the same, on the publish side of the pair."""
    tree, methods = mutate_memory('RPS_TABLE_TOPIC = "/memory/rps_table"',
                                  'RPS_TABLE_TOPIC = "/memory/rps_tables"')
    assert "/memory/rps_table" not in topic_literals(tree, methods), \
        "the pinned copy is not pinned"


@pytest.mark.mutation
def test_mutant_the_table_is_loaded_inside_the_constructor() -> None:
    """MUTANT: `rps_state.load` moves out of `main`.

    Then a refusal is raised by a Node that has already created its publishers, and the
    exit path runs through a partially constructed object.
    """
    tree, _methods = mutate_memory("            table = rps.load(store.root)",
                                   "            table = None")
    assert not loaded_in_main(tree), \
        "the location check does not notice load() leaving main()"


@pytest.mark.mutation
def test_mutant_the_wire_form_is_stored_instead_of_derived() -> None:
    """MUTANT: `snapshot` writes the model's own shape.

    The defect the module exists to prevent: `Opponent.to_dict()`'s tables are aggregates,
    so dropping a round from a list does not decrement a count — a table persisted in the
    model's shape keeps the behavioural fingerprint of a person who asked to be forgotten,
    and `rounds` is the only key an erasure knows how to filter.
    """
    model = rps.replay(played(6))
    with pytest.raises(rc.RecallError):
        rc.parse_rounds(model.to_dict(), where="the model's own shape")
    with pytest.raises(rps.RpsStateError, match="a deletion depends on"):
        rps.snapshot(model.to_dict())


@pytest.mark.mutation
def test_mutant_a_bad_round_is_skipped_instead_of_aborting_the_load() -> None:
    """MUTANT: replay swallows the failure and keeps going.

    Re-implemented rather than patched, because this mutant is about what `replay` would
    be if the `raise` were a `continue` — and the resulting counts are the point: they are
    a legal-looking model of rounds nobody played, indistinguishable from a person who
    played differently.
    """
    rounds = played(6)
    rounds[3] = {"user": "rock", "robot": "lizard", "outcome": "win", "episode": None}
    skipping = opp.Opponent(rng=random.Random(0))
    for entry in rounds:
        try:
            skipping.observe(entry["user"], entry["robot"])
        except opp.OpponentError:
            continue
    assert skipping.evidence_matches_recomputation(), \
        "the skipping model is not even self-consistent, so this mutant proves nothing"
    with pytest.raises(rps.RpsStateError):
        rps.replay(rounds)
    assert skipping.to_dict() != rps.replay(rounds[:3]).to_dict(), \
        "skipping a round is indistinguishable from stopping at it"


@pytest.mark.mutation
def test_mutant_the_episode_field_is_omitted_from_the_record() -> None:
    """MUTANT: `round_record` stops writing `episode`.

    Every round still persists, `recall.parse_rounds` still accepts the file, and R1.3
    inherits a table whose unattributed rounds are indistinguishable from attributed ones
    — which is the failure that cannot be repaired later, because the information was
    never written.
    """
    record = rps.round_record("rock", "paper")
    without = {k: v for k, v in record.items() if k != "episode"}
    body = {"version": rps.VERSION, "rounds": [without]}
    rc.parse_rounds(body, where="a round with no episode key")
    assert "episode" in record, "round_record no longer writes the field at all"
    state = rps.RpsState(model=opp.Opponent(rng=random.Random(0)),
                         rounds=[without], existed=True)
    assert state.orphans() == 1, (
        "orphans() counts a missing key differently from a null one, so dropping the "
        "field would hide the debt instead of reporting it")


@pytest.mark.mutation
def test_mutant_the_restore_record_drops_the_flag_that_separates_two_zeroes() -> None:
    """MUTANT: `existed` leaves the record, so `rounds: 0` has two causes and one shape."""
    record = {"kind": RESTORED_KIND, "existed": False, "rounds": 0, "orphans": 0}
    st.check_audit(record)
    without = {k: v for k, v in record.items() if k != "existed"}
    st.check_audit(without)          # the store cannot see a MISSING declared field
    assert set(st.AUDIT_FIELDS[RESTORED_KIND]) - set(without), (
        "the row no longer declares a field the record omits, so nothing anywhere "
        "notices — which is why the producer control above asserts the row's fields are "
        "all written, and check_audit alone is not enough")
