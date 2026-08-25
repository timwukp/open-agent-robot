"""Wiring between game_node and the opponent model (T-H6, and the T-H5 seam).

Every assertion about `game/game_node.py` here is made against its **source**, not
against an imported module: the node imports `rclpy`, and the suite's first
constraint is that it runs on a laptop with no ROS installed (§6.1). That is a real
limit and it is stated instead of hidden — these tests prove the *shape* of the
wiring, and what proves the node runs is `infra/run_ec2.sh status` on the instance
plus the human UAT in docs/test-design.md §6.4.

**T-H6 is asserted structurally, not as a wall-clock budget.** The plan's criterion
was "p95 arm jitter ≤ current + 5 ms", which collides with this suite's own rule
that no test asserts a duration: a timing assertion that goes red because the
laptop was busy earns itself a `skip` within two days, and a skipped test asserts
nothing. The direct oracle for "the model does not disturb the beat" is that the
beat path *cannot reach the model at all* — a property of the call graph, and
exactly what a mutant moving inference into `on_tts_started` violates. The timing
number is measured and reported instead:
`python3 game/opponent_harness.py --arm-cost` → docs/evidence/track-b/arm-cost.json.

The other half is polarity. The same round now decides what the person hears *and*
what the model trains on, so a swap between the user's view and the robot's view
would make the robot celebrate its losses and learn the opposite of what it said.
That is a one-token mutant, so it gets a named test rather than a review.

Every derivation below is a plain function of (tree, methods), so the mutation
section at the bottom re-runs the identical logic against a textually mutated
source. A derived property that no mutant can break is decoration.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_PY = REPO_ROOT / "game" / "game_node.py"
HAND_PY = REPO_ROOT / "perception" / "hand_node.py"

sys.path.insert(0, str(REPO_ROOT))

from game import rps                                        # noqa: E402
from game.opponent import MOVES, OpponentError              # noqa: E402

NODE_SRC = NODE_PY.read_text()

# The four methods allowed to touch the model. Written down because it is a claim
# about the design; the derivation is what compares the claim to the code, and it
# fails in both directions — a new toucher, or one of these losing its access.
MODEL_METHODS = {"__init__", "_arm", "_learn", "on_rps_table"}
# Callbacks that run between the chant starting and the hand landing on the beat.
BEAT_ENTRYPOINTS = ("on_tts_started",)
# Calls that would make this node a second writer of the model file.
DISK_CALLS = {"open", "write_text", "write_bytes", "replace", "mkdir", "fsync"}


# --------------------------------------------------------------------- parsing
def parse(src: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    tree = ast.parse(src)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "GameNode"), None)
    assert cls is not None, "no GameNode class in game/game_node.py"
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods, ("parsed GameNode but found no methods — this parser no longer "
                     "matches the file it reads, and every assertion would pass")
    return tree, methods


TREE, METHODS = parse(NODE_SRC)


def touches_model(fn: ast.AST) -> bool:
    """Does this function body reach the opponent model?

    Two forms count: any mention of `self.opponent` — reading it, calling through
    it, or assigning it, since constructing the model is as much a touch as
    querying it — and `opp.Opponent`, the class itself (`from_dict`). `opp.MOVES`
    and `opp.OpponentError` deliberately do NOT count: they are the vocabulary and
    the exception type, and counting them would make the coin fallback in `_arm()`
    look like inference.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Attribute):
            continue
        if (node.attr == "opponent"
                and isinstance(node.value, ast.Name) and node.value.id == "self"):
            return True
        if (node.attr == "Opponent"
                and isinstance(node.value, ast.Name) and node.value.id == "opp"):
            return True
    return False


def touchpoints(methods: dict[str, ast.FunctionDef]) -> set[str]:
    return {name for name, fn in methods.items() if touches_model(fn)}


def self_calls(fn: ast.AST, methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Names of GameNode methods this function calls directly."""
    return {n.func.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
            and n.func.attr in methods}


def reachable(start: str, methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Transitive closure of `self_calls` from `start`, including `start`."""
    seen, todo = {start}, [start]
    while todo:
        for name in self_calls(methods[todo.pop()], methods):
            if name not in seen:
                seen.add(name)
                todo.append(name)
    return seen


def beat_path_leak(entry: str, methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Methods reachable from `entry` that touch the model.

    Composed of both derivations rather than intersected with the written-down
    `MODEL_METHODS`, and that is not a style choice: the first version of this
    check compared the closure to `MODEL_METHODS`, so a mutant that put
    `self.opponent.predict()` *inline* in `on_tts_started` created no new named
    toucher and slipped straight through. `test_mutant_inference_on_the_beat_path`
    is what found that; the census and the closure only gate the beat path when
    they are combined.
    """
    return reachable(entry, methods) & touchpoints(methods)


def stmt_index(fn: ast.FunctionDef, predicate) -> int:
    """Index of the first top-level statement of `fn` matching `predicate`."""
    for i, stmt in enumerate(fn.body):
        if any(predicate(n) for n in ast.walk(stmt)):
            return i
    return -1


def learn_order(methods: dict[str, ast.FunctionDef]) -> dict[str, int]:
    """Statement positions inside `resolve` of learning, speaking and banter."""
    resolve = methods["resolve"]
    return {
        "learn": stmt_index(resolve, lambda n: (
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_learn")),
        "say": stmt_index(resolve, lambda n: (
            isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == "clips" and isinstance(n.slice, ast.Name)
            and n.slice.id == "outcome")),
        "banter": stmt_index(resolve, lambda n: (
            isinstance(n, ast.Attribute) and n.attr == "pub_agent")),
    }


def judge_unpack(methods: dict[str, ast.FunctionDef]) -> list[str]:
    """The names `rps.judge`'s two return values are bound to, in order."""
    targets = [s.targets[0] for s in ast.walk(methods["resolve"])
               if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call)
               and isinstance(s.value.func, ast.Attribute)
               and s.value.func.attr == "judge"]
    assert len(targets) == 1, "expected exactly one rps.judge call site in resolve()"
    return [e.id for e in targets[0].elts]           # type: ignore[attr-defined]


def move_tables(tree: ast.Module) -> list[str]:
    """Module-level names whose value's string constants are all moves.

    Any such literal is a second encoding of the 3x3 invariant. `WINS` used to be
    exactly that, and nothing forced it to agree with `opponent.BEATS`.
    """
    found = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(
                stmt.value, (ast.Dict, ast.Set, ast.Tuple, ast.List)):
            continue
        strings = {n.value for n in ast.walk(stmt.value)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if strings and strings <= set(MOVES):
            found.append(getattr(stmt.targets[0], "id", "<unnamed>"))
    return found


def arm_fallback_touches_model(methods: dict[str, ast.FunctionDef]) -> bool:
    """Does `_arm`'s exception handler read the object that just failed?"""
    handlers = [h for n in ast.walk(methods["_arm"]) if isinstance(n, ast.Try)
                for h in n.handlers]
    assert handlers, "_arm no longer handles a failing model — one raise takes the node"
    return any(isinstance(n, ast.Attribute) and n.attr == "opponent"
               for h in handlers for n in ast.walk(h))


def adopts_inside_try(methods: dict[str, ast.FunctionDef]) -> bool:
    """Is the restored model installed before validation has finished?"""
    tries = [n for n in methods["on_rps_table"].body if isinstance(n, ast.Try)]
    assert len(tries) == 1, "on_rps_table should validate in exactly one try block"
    return any(isinstance(t, ast.Attribute) and t.attr == "opponent"
               for n in ast.walk(tries[0]) if isinstance(n, ast.Assign)
               for t in n.targets)


def topic_qos(methods: dict[str, ast.FunctionDef]) -> tuple[dict, dict]:
    """(subscription qos node, publisher qos node) keyed by topic name."""
    subs, pubs = {}, {}
    for node in ast.walk(methods["__init__"]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "create_subscription" and len(node.args) >= 4:
            subs[node.args[1].value] = node.args[3]   # type: ignore[union-attr]
        if node.func.attr == "create_publisher" and len(node.args) >= 3:
            pubs[node.args[1].value] = node.args[2]   # type: ignore[union-attr]
    return subs, pubs


def disk_calls(methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Names of filesystem-mutating calls in the model-touching methods."""
    out = set()
    for name in ("_learn", "on_rps_table", "_arm"):
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Call):
                target = getattr(node.func, "attr", getattr(node.func, "id", None))
                if target in DISK_CALLS:
                    out.add(f"{name}:{target}")
    return out


# --------------------------------------------------------------- T-H6 structural
def test_model_touchpoints_are_exactly_the_four_named_methods():
    """Derived from the source, so a fifth toucher fails without anyone editing me."""
    found = touchpoints(METHODS)
    assert found == MODEL_METHODS, (
        f"methods touching the opponent model are {sorted(found)}, expected "
        f"{sorted(MODEL_METHODS)} — every new toucher has to be argued about, "
        "because the beat-path proof below is a statement about this set")


def test_the_walker_can_actually_see_model_access():
    """Positive control for the two derivations above.

    Without this, a `touches_model` that always returned False and a `reachable`
    that always returned `{start}` would make the beat-path test pass vacuously —
    the most likely way for this file to become decoration.
    """
    assert touches_model(METHODS["_learn"]), \
        "_learn calls self.opponent.observe; a walker that misses it sees nothing"
    assert not touches_model(METHODS["on_hand"]), \
        "on_hand does not touch the model; a walker that says it does says yes to all"
    assert beat_path_leak("resolve", METHODS) == {"_learn"}, \
        "resolve reaches the model through _learn — an empty closure here would " \
        "make the beat-path assertion measure nothing"


@pytest.mark.parametrize("entry", BEAT_ENTRYPOINTS)
def test_beat_path_cannot_reach_the_model(entry: str):
    """T-H6, the gate. Nothing between chant start and the beat may reach inference.

    `tick` is deliberately not in this list even though it eventually calls
    `resolve()`: that call is guarded by `now >= self.t_throw + 0.65`, so it runs
    two thirds of a second *after* the hand has landed. Pretending otherwise would
    force the model out of `resolve()` entirely, which is where it belongs;
    `test_the_model_is_updated_only_after_the_person_has_been_answered` covers that
    path instead.
    """
    leak = beat_path_leak(entry, METHODS)
    assert not leak, (
        f"{entry} can reach the opponent model via {sorted(leak)} — the beat may "
        "not wait for anything, and arm time is 3.5 s earlier for exactly this reason")


def test_the_move_is_committed_at_arm_time():
    """`start_game` picks the move; the beat callbacks only read it."""
    assert "_arm" in self_calls(METHODS["start_game"], METHODS), \
        "start_game must commit the move before the invite (plan D2)"
    for entry in BEAT_ENTRYPOINTS:
        assigns = [t for stmt in ast.walk(METHODS[entry])
                   if isinstance(stmt, ast.Assign) for t in stmt.targets
                   if isinstance(t, ast.Attribute) and t.attr == "robot_move"]
        assert not assigns, f"{entry} assigns robot_move — that is a beat-time decision"


def test_arm_cost_evidence_is_reported_not_asserted():
    """The M-level half of T-H6: a number in a file, with no threshold attached.

    Asserts only that the measurement exists, carries its `n`, and names the tick
    period it should be read against. Asserting the value is the thing this test
    exists to avoid.
    """
    path = REPO_ROOT / "docs" / "evidence" / "track-b" / "arm-cost.json"
    assert path.exists(), (
        "docs/evidence/track-b/arm-cost.json is missing — run "
        "`python3 game/opponent_harness.py --arm-cost` to produce it")
    data = json.loads(path.read_text())
    for key in ("p50", "p95", "max", "samples", "contexts", "tick_period_ms", "unit"):
        assert key in data, f"arm-cost.json has no {key!r}, so it cannot be read"
    assert data["samples"] >= 20, (
        "fewer than 20 samples makes the reported p95 a max in disguise (§6.0 rule 4)")
    assert data["contexts"] > 0, (
        "the timed model had no contexts, so choose() returned on its first branch "
        "and this measured the coin instead of the arithmetic")


# ------------------------------------------------------------ order and polarity
def test_the_model_is_updated_only_after_the_person_has_been_answered():
    """The person hears the result before the model is touched.

    The ordering would still work if reversed today, and that is the point: it is
    what keeps a future slow or throwing model out of the response being waited for.
    """
    order = learn_order(METHODS)
    assert order["learn"] > 0, "resolve() never calls _learn — the model never trains"
    assert order["say"] != -1 and order["banter"] != -1, \
        "resolve() no longer says the result or asks for banter; this test is stale"
    assert order["learn"] > order["say"] and order["learn"] > order["banter"], (
        f"statement order was {order} — a model failure would then delay or "
        "silence the answer the person is waiting for")


def test_judge_result_is_unpacked_user_view_first():
    """The tuple order is the polarity, and swapping it is a one-token mutant.

    `rps.judge` returns (user's outcome, robot's outcome) and the node announces and
    scores the *robot's* view. Unpacking the other way round would make it celebrate
    its own losses while training on the inverted label.
    """
    names = judge_unpack(METHODS)
    assert names[1] == "outcome" and names[0].startswith("_"), (
        f"judge() unpacked as {names} — the second element is the robot's view and "
        "the node speaks and scores that one; the first is the user's and is unused")


def test_no_second_outcome_table_at_module_level():
    """One 3x3 table for the whole system."""
    offenders = move_tables(TREE)
    assert not offenders, (
        f"{offenders} encode(s) the moves again — the round both tells the person "
        "the result and trains the model on it, so two tables would let the robot "
        "announce one outcome and learn another (see game/rps.judge)")


def test_arm_fallback_does_not_ask_the_broken_object():
    """The coin in `_arm`'s handler must not be `self.opponent.uniform_move()`."""
    assert not arm_fallback_touches_model(METHODS), (
        "_arm's fallback reads self.opponent — the fallback exists precisely for "
        "the case where that object is the thing that raised")


# ---------------------------------------------------------- the persistence seam
def test_restored_model_is_adopted_only_after_validation():
    """A malformed table is rejected wholesale, never merged into the live model."""
    fn = METHODS["on_rps_table"]
    assert not adopts_inside_try(METHODS), \
        "on_rps_table installs self.opponent before validation finishes — a " \
        "half-restored table is a model trained on a subset nobody chose"
    tries = [n for n in fn.body if isinstance(n, ast.Try)]
    after = [i for i, stmt in enumerate(fn.body)
             if isinstance(stmt, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == "opponent"
                     for t in stmt.targets)]
    assert after and after[0] > fn.body.index(tries[0]), \
        "on_rps_table never adopts the restored model, so the topic does nothing"
    for h in tries[0].handlers:
        assert any(isinstance(n, ast.Return) for n in ast.walk(h)), \
            "the handler must return, leaving the in-RAM model untouched"


def test_state_topic_is_latched_and_the_round_topic_is_not():
    """The accumulated table is state; a resolved round is an event.

    Latched (`TRANSIENT_LOCAL`, depth 1) means a node that starts after the writer
    still receives the last table instead of waiting for a republish. A round, in
    contrast, must not be redelivered on reconnect — that would train the model on
    the same round twice.
    """
    subs, pubs = topic_qos(METHODS)
    assert "/memory/rps_table" in subs, "the node never subscribes to the saved table"
    qos = subs["/memory/rps_table"]
    assert isinstance(qos, ast.Name) and qos.id == "LATCHED", (
        "the table subscription must use the latched profile, or a table published "
        "before this node started is simply never seen")
    assert "/game/rps_round" in pubs, \
        "no /game/rps_round publisher — nothing would ever persist a round"
    assert isinstance(pubs["/game/rps_round"], ast.Constant), \
        "the round topic should use a plain depth, not the latched profile"

    latched = [s for s in TREE.body if isinstance(s, ast.Assign)
               and getattr(s.targets[0], "id", None) == "LATCHED"]
    assert len(latched) == 1, "LATCHED is defined once at module level"
    kw = {k.arg: k.value for k in latched[0].value.keywords}  # type: ignore[union-attr]
    assert isinstance(kw["depth"], ast.Constant) and kw["depth"].value == 1
    assert kw["durability"].attr == "TRANSIENT_LOCAL"         # type: ignore[union-attr]


def test_the_node_never_writes_the_model_to_disk():
    """Single-writer invariant: memory_node owns the file, game_node owns the RAM.

    Two writers to an append-only file is the defect this invariant exists to make
    unexpressible (T-A1).
    """
    assert not disk_calls(METHODS), (
        f"{sorted(disk_calls(METHODS))} — only memory_node may write the model")
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "publish"
               and isinstance(n.func.value, ast.Attribute)          # type: ignore[union-attr]
               and n.func.value.attr == "pub_round"                 # type: ignore[union-attr]
               for n in ast.walk(METHODS["_learn"])), \
        "_learn must hand the round to the single writer over /game/rps_round"


# ------------------------------------------------------- the reading of the throw
def test_hand_vocabulary_is_exactly_the_moves_plus_none():
    """Derived from `hand_node.classify`, because a written list cannot notice a new
    name — and the name it would fail to notice is the one that reaches
    `Opponent.observe()` and raises inside a ROS callback."""
    vocab = rps.hand_vocabulary(HAND_PY.read_text())
    assert vocab == set(MOVES) | {rps.NO_HAND}, (
        f"hand_node.classify can return {sorted(vocab)}; the game handles "
        f"{sorted(set(MOVES) | {rps.NO_HAND})}. Anything else reaches the model.")


def test_hand_vocabulary_refuses_to_return_an_empty_set():
    """A parser that stops matching must raise, not report "no labels".

    An empty vocabulary satisfies every subset assertion above, which would turn
    the derivation into decoration at exactly the moment it stopped working.
    """
    with pytest.raises(OpponentError):
        rps.hand_vocabulary("def classify(x):\n    return x\n")
    with pytest.raises(OpponentError):
        rps.hand_vocabulary("def other(x):\n    return ('rock', 1)\n")


def test_majority_drops_none_and_reports_nothing_for_an_empty_window():
    """"No samples" and "only `none` samples" take the same path: both are "I did
    not see your hand", and only one of them is distinguishable by a person, so
    distinguishing them in code is a branch nobody can test."""
    assert rps.majority([]) is None
    assert rps.majority(["none", "none", "none"]) is None
    assert rps.majority(["none", "paper", "none"]) == "paper"
    assert rps.majority(["rock", "rock", "paper"]) == "rock"


def test_majority_is_stable_across_hash_seeds():
    """The defect this module was written for, tested the way it was found.

    `max(set(window), key=window.count)` breaks ties on set iteration order, which
    for strings depends on the per-process hash seed: the same window read `paper`
    under PYTHONHASHSEED=0 and `rock` under =2. So one throw could be judged
    differently across two boots of the same binary — and that read now trains the
    transition table, which makes it a nondeterministic *model*, not one bad round.

    The second assertion is the control: it fails if CPython ever makes string set
    order seed-independent, at which point the rationale above has expired and
    somebody should read this docstring rather than trust it.
    """
    window = ["rock", "rock", "paper", "paper"]
    seeds = [str(s) for s in range(8)]
    fixed, incidental = set(), set()
    for seed in seeds:
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, json;"
             f"sys.path.insert(0, {str(REPO_ROOT)!r});"
             "from game import rps;"
             f"w = {window!r};"
             "print(json.dumps([rps.majority(w), list(set(w))[0]]))"],
            capture_output=True, text=True, env={"PYTHONHASHSEED": seed},
        )
        assert out.returncode == 0, f"subprocess failed under seed {seed}: {out.stderr}"
        chosen, set_first = json.loads(out.stdout)
        fixed.add(chosen)
        incidental.add(set_first)
    assert fixed == {"rock"}, (
        f"majority() returned {sorted(fixed)} across hash seeds {seeds} — the "
        "tie-break must come from the order of MOVES, not from the runtime")
    assert len(incidental) > 1, (
        "set iteration order was identical under every hash seed, so this test can "
        "no longer distinguish the fixed tie-break from the old one; the defect "
        "described above may have stopped being reachable")


def test_judge_is_the_exact_inversion_of_the_user_view():
    """All nine pairs, both views, in one place. Counting outcomes rather than
    spot-checking one pair, because flipping `ROBOT_VIEW` leaves the three ties
    correct and a single-pair check could land on one of them."""
    seen = {"win": 0, "lose": 0, "tie": 0}
    for user in MOVES:
        for robot in MOVES:
            u, r = rps.judge(user, robot)
            seen[r] += 1
            if user == robot:
                assert u == r == "tie", f"{user} vs {robot} is a tie for both"
            else:
                assert {u, r} == {"win", "lose"}, (
                    f"{user} vs {robot} gave user={u} robot={r} — exactly one side "
                    "wins a decided round")
    assert seen == {"win": 3, "lose": 3, "tie": 3}, (
        f"robot-view outcomes over the 3x3 grid were {seen}; a rock-paper-scissors "
        "table with any other shape is not this game")


def test_judge_refuses_a_non_move():
    """`none` must raise rather than resolve to a tie: a round whose moves are not
    moves is a round the model must not be trained on, and a silent `tie` would
    train it on a fiction."""
    for bad in (rps.NO_HAND, "", "ROCK", "水"):
        with pytest.raises(OpponentError):
            rps.judge(bad, "rock")
        with pytest.raises(OpponentError):
            rps.judge("rock", bad)


# ------------------------------------------------------------------ T-J4 mutants
# Each row is (id, anchor, replacement, the check that must break). The check is
# the same function the corresponding test above calls, so a mutant surviving means
# that test measures the wrong thing — not that the code is safe.

def mutate(old: str, new: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    """Re-parse game_node.py with one textual substitution applied.

    The anchor assertion matters: without it a stale mutant becomes a no-op, the
    check keeps passing, and the report reads "mutant survived" — blaming the test
    for a broken mutant. Two failures, two messages.
    """
    assert old in NODE_SRC, f"mutant anchor no longer present in game_node.py: {old!r}"
    return parse(NODE_SRC.replace(old, new, 1))


@pytest.mark.mutation
def test_unmutated_control():
    """Round-level control (§6.0 rule 2): with no substitution applied, every check
    below must report the healthy value. A harness that reported failure for all
    inputs would otherwise "kill" every mutant while proving nothing."""
    tree, methods = mutate("class GameNode(Node):", "class GameNode(Node):")
    assert touchpoints(methods) == MODEL_METHODS
    assert not beat_path_leak("on_tts_started", methods)
    assert learn_order(methods)["learn"] > learn_order(methods)["say"]
    assert judge_unpack(methods)[1] == "outcome"
    assert move_tables(tree) == []
    assert not arm_fallback_touches_model(methods)
    assert not adopts_inside_try(methods)
    assert topic_qos(methods)[0]["/memory/rps_table"].id == "LATCHED"
    assert not disk_calls(methods)


@pytest.mark.mutation
def test_mutant_inference_on_the_beat_path():
    """MUTANT: consult the model inside `on_tts_started` (the chant callback)."""
    _, methods = mutate(
        '            self.state = "chanting"',
        '            self.state = "chanting"\n            self.opponent.predict()')
    assert "on_tts_started" in touchpoints(methods), \
        "the touchpoint census missed a new model access on the beat path"
    assert beat_path_leak("on_tts_started", methods), \
        "the beat path reached the model and the leak check did not notice"


@pytest.mark.mutation
def test_mutant_model_trained_before_the_person_is_answered():
    """MUTANT: `_learn` moves above the result clip and the banter request."""
    _, methods = mutate(
        "        self.say(clips[outcome], cache=True)",
        "        self._learn(user)\n        self.say(clips[outcome], cache=True)")
    order = learn_order(methods)
    assert not (order["learn"] > order["say"] and order["learn"] > order["banter"]), \
        f"ordering check accepted {order}, so it does not constrain the order"


@pytest.mark.mutation
def test_mutant_judge_polarity_swapped():
    """MUTANT: unpack judge() the other way round — the robot celebrates its losses."""
    _, methods = mutate("        _user_outcome, outcome = rps.judge(",
                        "        outcome, _user_outcome = rps.judge(")
    assert judge_unpack(methods)[1] != "outcome", \
        "the polarity check cannot tell the two views apart"


@pytest.mark.mutation
def test_mutant_second_outcome_table_reintroduced():
    """MUTANT: bring back `WINS`, a second encoding of the same 3x3 invariant."""
    tree, _ = mutate(
        "LATCHED = QoSProfile(",
        'WINS = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}\n'
        "LATCHED = QoSProfile(")
    assert move_tables(tree) == ["WINS"], \
        "a module-level move table was reintroduced and the census did not see it"


@pytest.mark.mutation
def test_mutant_arm_fallback_asks_the_broken_object():
    """MUTANT: the coin becomes `self.opponent.uniform_move()`."""
    _, methods = mutate("return random.choice(list(opp.MOVES))",
                        "return self.opponent.uniform_move()")
    assert arm_fallback_touches_model(methods), \
        "the fallback now asks the object that just raised and the check allowed it"


@pytest.mark.mutation
def test_mutant_restored_model_adopted_before_validation():
    """MUTANT: install the deserialised model inside the try block."""
    _, methods = mutate(
        "            restored = opp.Opponent.from_dict(json.loads(msg.data),",
        "            self.opponent = opp.Opponent.from_dict(json.loads(msg.data),")
    assert adopts_inside_try(methods), \
        "the model is installed mid-validation and the check did not notice"


@pytest.mark.mutation
def test_mutant_table_subscription_loses_the_latched_profile():
    """MUTANT: default (volatile) QoS on the state topic — a table published before
    this node starts is then never delivered, and the node silently plays the coin
    forever while looking healthy."""
    _, methods = mutate("self.on_rps_table, LATCHED)", "self.on_rps_table, 10)")
    qos = topic_qos(methods)[0]["/memory/rps_table"]
    assert not (isinstance(qos, ast.Name) and qos.id == "LATCHED"), \
        "the QoS check reads something other than the argument actually passed"


@pytest.mark.mutation
def test_mutant_second_writer_in_the_game_node():
    """MUTANT: `_learn` writes the model file itself, breaking the single writer."""
    _, methods = mutate(
        "        self.pub_round.publish(String(data=json.dumps(",
        '        Path("/tmp/oar-rps.json").write_text("{}")\n'
        "        self.pub_round.publish(String(data=json.dumps(")
    assert "_learn:write_text" in disk_calls(methods), \
        "a filesystem write appeared in _learn and the single-writer check missed it"
