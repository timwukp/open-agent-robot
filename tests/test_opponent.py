"""Offline tests for game/opponent.py and game/opponent_harness.py (T-H1..H5).

Reading order, because two of the pass criteria in docs/test-design.md §2.H had to
change and the reasons are the interesting part:

**T-H2 ③ cannot be red on score, for a reason that is not about our code.**
Against an opponent that plays uniformly and ignores history, *every* robot policy
has expected score exactly 0 — the payoff matrix is zero-sum and symmetric, so no
strategy can be punished for being wrong. Removing the confidence gate therefore
cannot make the ungated arm significantly worse than the coin, and the plan's
"remove the gate → ③ must go red" is unachievable in principle rather than
unimplemented. Measured: ungated diff −0.0086 per round, 95% CI [−0.0204, +0.0032]
over 100 seeds — the right *sign* for the −0.0085(a−2) risk the literature reports,
but 1.4 standard errors from zero and consistent with the theoretical 0.
So the assertion is moved onto the claim the gate actually makes: the Ville bound.
Gated, the exploit ever opens on 3 of 100 uniform opponents (δ = 0.05); ungated it
opens on 100 of 100. That goes from 0.03 to 1.00 under the mutant, and it is the
property we would want to know had broken.

**T-H2 ④'s "gate open fraction < 10%" is empirically false, and the truth is
better than the criterion.** The gate opens on 71% of rounds against the adaptive
anti-learner and the model beats it by +0.095 per round, CI [+0.088, +0.102].
The loop is closed: our moves are a function of the user's history, so an opponent
that reacts to our moves is thereby reacting to that history and becomes
predictable from it. The enforced criterion is the one the plan's ④ was protecting
— score not significantly below zero — and the gate fraction is reported, not
gated, because the plan's threshold encoded a mechanism that does not hold.

Everything else follows the house rules: no test asserts a duration, every
assertion names a mutant it must kill, each mutation round runs a no-mutant
control, and the numbers live in docs/evidence/track-b/opponent-h2.json with a
liveness check that re-derives them from source.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "game"
OPPONENT_PY = GAME_DIR / "opponent.py"
HARNESS_PY = GAME_DIR / "opponent_harness.py"
RPS_PY = GAME_DIR / "rps.py"
EVIDENCE_JSON = REPO_ROOT / "docs" / "evidence" / "track-b" / "opponent-h2.json"

from game import opponent as opp  # noqa: E402
from game import opponent_harness as harness  # noqa: E402
from game import rps  # noqa: E402

# The evidence run. Kept as constants so a drifted JSON is a red test rather than
# a quietly different claim; §6.0 rule 6 (numbers by harness, cited from JSON).
EVIDENCE_SEEDS = 100
EVIDENCE_ROUNDS = 300
# Ville: P(the exploit ever opens against a uniform opponent) ≤ δ = 0.05 per
# opponent. Over 100 seeds that is Binomial(100, ≤0.05); 0.12 is comfortably above
# its 99.9th percentile (≈12) and far below the 1.00 an ungated model produces, so
# the bound discriminates without being a coin flip on binomial noise.
VILLE_SEED_FRACTION_CEILING = 0.12
# A 20-seed live re-run only has to agree in direction with the 100-seed record;
# it is a liveness check on the code, not a second estimate.
LIVE_SEEDS = 20

EXPLOITABLE = ("win_stay_lose_shift", "markov1")


# --------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of game/opponent.py.

    Same shape as tests/test_skills_loader.py: the mutant is applied to a copy of
    the real source, and the anchor is asserted present so a refactor that renames
    the mutated line turns into a red anchor rather than a mutant that silently
    patches nothing and "passes".
    """
    dst = tmp_path / "game"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "opponent.py"
    src = OPPONENT_PY.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in opponent.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src)
    name = f"opponent_mutant_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so an
    # unregistered module fails at import for a reason unrelated to the mutation.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


MUTANT_GLOBAL_RNG = (
    ("MOVES[self.rng.randrange(3)]", "MOVES[random.randrange(3)]"),
    ("self.rng.random() >= self.exploit_rate", "random.random() >= self.exploit_rate"),
)
MUTANT_NO_GATE = (("return self.log_evidence() >= math.log(1.0 / self.delta)",
                   "return True"),)
MUTANT_MARGINAL_COUNTER = (
    ('return "|".join(self._token(u, r) for u, r in hist[-k:])', 'return ""'),)
MUTANT_DROP_HISTORY = (
    ('obj.history = [(u, r) for u, r in state["history"]]', "pass"),)
MUTANT_STALE_EVIDENCE = (
    ("self._ctx_e[k][ctx] = _log_polya_evidence(tuple(counts))",
     "pass  # mutant: never fold the round into the cached e-value"),)


# ------------------------------------------------------------ structure/static
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", [OPPONENT_PY, HARNESS_PY, RPS_PY], ids=lambda p: p.name)
def test_pure_module_imports_nothing_that_needs_the_robot(path: Path) -> None:
    """§6.1: a module that imports rclpy or boto3 can only be tested on the GPU
    instance, which means it will not be tested. Mutant: `import boto3` at the top
    of opponent.py → red."""
    banned = {"rclpy", "boto3", "botocore", "torch", "socket", "requests",
              "urllib", "http", "mujoco", "cv2", "numpy"}
    found = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    assert not (found & banned), f"{path.name} imports {sorted(found & banned)}"


def test_the_module_never_reaches_for_the_global_random_source() -> None:
    """T-H1, static half. `random.choice(...)` anywhere in the model would make a
    red unreproducible, which is how a red becomes a "flake" and then a habit.
    `random.Random` (constructing an injectable generator) is the one legal use.

    Mutant: MUTANT_GLOBAL_RNG → red here and in the bit-identical test below.
    """
    offenders = []
    for node in ast.walk(_tree(OPPONENT_PY)):
        if not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "random":
            if node.attr != "Random":
                offenders.append(f"random.{node.attr} at line {node.lineno}")
    assert not offenders, "module-level randomness: " + ", ".join(offenders)


def test_the_scoring_table_has_exactly_one_creator() -> None:
    """Who decides who won: one table, and `BEATS` is it.

    This test used to compare `game_node.WINS` against `opponent.BEATS`, because two
    files each held their own 3x3 relation and nothing forced them to agree — if
    they ever disagreed the robot would announce one result and train on the
    opposite, and no test of either file alone could see it. Wiring the model into
    the game made that unacceptable rather than untidy, so `WINS` is gone and both
    views now come out of `game/rps.judge`, which derives them from `BEATS`.

    So the assertion changed shape with the code: instead of checking that two
    copies agree, it checks that the second copy does not exist and that the
    remaining creator is the one being used. The reintroduction mutant lives in
    tests/test_game_wiring.py::test_mutant_second_outcome_table_reintroduced.
    """
    node_tree = _tree(GAME_DIR / "game_node.py")
    duplicates = []
    for stmt in node_tree.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(
                stmt.value, (ast.Dict, ast.Set, ast.Tuple, ast.List)):
            continue
        strings = {n.value for n in ast.walk(stmt.value)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if strings and strings <= set(opp.MOVES):
            duplicates.append(getattr(stmt.targets[0], "id", "<unnamed>"))
    assert not duplicates, (
        f"game_node holds its own move relation again ({duplicates}); the outcome "
        "the person hears and the label the model learns must come from one table")
    assert any(isinstance(n, ast.Attribute) and n.attr == "judge"
               and isinstance(n.value, ast.Name) and n.value.id == "rps"
               for n in ast.walk(node_tree)), \
        "game_node no longer calls rps.judge, so the surviving table has no user " \
        "and the outcome is being decided somewhere this test cannot see"


def test_the_harness_is_a_declared_standalone_entrypoint() -> None:
    """check_deploy.sh check 8: every `__main__` outside tests/ is either launched
    by run_ec2.sh or declared as run-by-hand. The harness is the second kind.
    Mutant: drop the row from the tsv → check_deploy.sh reports a finding."""
    tsv = (REPO_ROOT / "infra" / "standalone_entrypoints.tsv").read_text()
    assert "game/opponent_harness.py\t" in tsv


# --------------------------------------------------------------- T-H1 pure/det
def _replay(model, rounds: list[tuple[str, str]]) -> None:
    for user, robot in rounds:
        model.observe(user, robot)


def _scripted_rounds(n: int, seed: int = 7) -> list[tuple[str, str]]:
    """A reproducible round log from the win-stay/lose-shift opponent.

    Reuses the harness fixture rather than inventing a second habitual player, and
    the reuse is load-bearing: a *habitual* opponent concentrates its rounds into a
    few contexts, while a near-uniform log spreads 200 rounds over 81 order-2
    buckets and leaves `predict()` starved. An earlier version of this helper did
    exactly that and every prediction assertion below was silently vacuous — the
    gate was open and `predict()` still returned None.
    """
    rng = random.Random(seed)
    rounds: list[tuple[str, str]] = []
    hist: list[dict] = []
    for _ in range(n):
        robot = opp.MOVES[rng.randrange(3)]
        user = harness.opp_win_stay_lose_shift(rng, hist)
        outcome = opp.outcome_for_user(user, robot)
        hist.append({"user": user, "robot": robot, "user_outcome": outcome})
        rounds.append((user, robot))
    return rounds


def test_the_warm_up_cost_is_bounded() -> None:
    """How many rounds before the model says anything at all?

    This is the number that decides whether the wiring into game_node is useful in
    a demo, and it is arithmetic before it is a measurement: the context alphabet
    is 9 symbols (3 moves × 3 outcomes), so order-2 alone has 81 buckets and
    `min_context_count` is 3. A 20-round session cannot fill one, whatever the
    opponent does. Recorded here so a change that makes the warm-up much longer is
    red, and so the wiring PR inherits the number instead of rediscovering it.
    """
    model = opp.Opponent()
    first = None
    for i, (user, robot) in enumerate(_scripted_rounds(300), start=1):
        model.observe(user, robot)
        if first is None and model.predict() is not None:
            first = i
    assert first is not None, "never predicted anything in 300 rounds"
    assert first <= 120, f"first prediction only at round {first}"
    # And the floor, which is structural rather than tunable:
    short = opp.Opponent()
    for user, robot in _scripted_rounds(20):
        short.observe(user, robot)
    assert short.predict() is None, (
        "a 20-round session now predicts — if an order-1 rung was added, update "
        "the warm-up discussion in docs/test-design.md §2.H rather than this "
        "assertion")


def test_same_history_gives_the_same_prediction_and_the_same_table() -> None:
    rounds = _scripted_rounds(200)
    a, b = opp.Opponent(), opp.Opponent()
    _replay(a, rounds)
    _replay(b, list(rounds))
    assert a.predict() == b.predict()
    assert a.log_evidence() == b.log_evidence()
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_the_gate_actually_opens_on_the_scripted_history() -> None:
    """Positive control. Every determinism and prediction assertion above would
    pass on a model that returns None forever, and "no capability" and "no bug"
    look identical from the outside."""
    model = opp.Opponent()
    _replay(model, _scripted_rounds(200))
    assert model.gate_open(), "scripted history did not open the gate — tests above are vacuous"
    assert model.predict() is not None


def test_a_seeded_run_is_bit_identical() -> None:
    """T-H1. Mutant: MUTANT_GLOBAL_RNG → the two digests diverge."""
    def run(seed: int) -> str:
        model = opp.Opponent(rng=random.Random(seed))
        moves = []
        for user, robot in _scripted_rounds(300):
            moves.append(model.choose())
            model.observe(user, robot)
        return hashlib.sha256("".join(moves).encode()).hexdigest()

    assert run(1234) == run(1234)
    assert run(1234) != run(4321), "different seeds produced the same run — rng unused?"


def test_choose_is_valid_and_never_raises_before_the_first_round() -> None:
    """T-H5's other half: a cold model must play, not wait. Mutant: raise instead
    of falling back to uniform in `choose()` → red."""
    model = opp.Opponent(rng=random.Random(5))
    assert model.predict() is None
    assert model.choose() in opp.MOVES


def test_the_incremental_evidence_equals_the_sum_over_contexts() -> None:
    """The running total is an optimisation (10⁸ lgamma calls otherwise), and an
    optimisation that drifts moves the gate without moving any other test.
    Mutant: MUTANT_STALE_EVIDENCE → red."""
    model = opp.Opponent()
    for i, (user, robot) in enumerate(_scripted_rounds(200)):
        model.observe(user, robot)
        if i % 25 == 0:
            assert model.evidence_matches_recomputation(), f"drifted after {i} rounds"
    assert model.evidence_matches_recomputation()


def test_an_empty_context_contributes_exactly_zero_evidence() -> None:
    """An unseen context must be free, or the e-value would grow with the size of
    the context space instead of with the evidence in it."""
    assert opp._log_polya_evidence((0, 0, 0)) == 0.0
    assert opp.Opponent().log_evidence() == 0.0
    assert opp.Opponent().gate_open() is False


def test_a_uniform_history_leaves_the_gate_shut() -> None:
    """The null side of the gate, checked directly rather than only through the
    harness: 300 fair-coin rounds must not look like evidence."""
    rng = random.Random(99)
    model = opp.Opponent()
    for _ in range(300):
        model.observe(opp.MOVES[rng.randrange(3)], opp.MOVES[rng.randrange(3)])
    assert not model.gate_open(), f"e-value {model.evidence():.3g} on a fair coin"


@pytest.mark.parametrize("user,robot,expected", [
    ("rock", "scissors", "win"), ("rock", "paper", "lose"), ("rock", "rock", "tie"),
    ("paper", "rock", "win"), ("paper", "scissors", "lose"), ("paper", "paper", "tie"),
    ("scissors", "paper", "win"), ("scissors", "rock", "lose"),
    ("scissors", "scissors", "tie"),
])
def test_the_full_outcome_table(user: str, robot: str, expected: str) -> None:
    """All nine cells, because a 3×3 table is small enough that sampling it is a
    choice to be wrong somewhere unobserved."""
    assert opp.outcome_for_user(user, robot) == expected


@pytest.mark.parametrize("bad", ["Rock", "", "rock ", "lizard", "spock", None, 5])
def test_a_malformed_move_raises_instead_of_defaulting(bad) -> None:
    """A silently dropped round trains the model on a subset nobody chose.
    Mutant: return "tie" for unknown input → red."""
    with pytest.raises(opp.OpponentError):
        opp.outcome_for_user(bad, "rock")
    with pytest.raises(opp.OpponentError):
        opp.Opponent().observe("rock", bad)


@pytest.mark.parametrize("kwargs", [
    {"orders": ()}, {"orders": (0, 2)}, {"orders": (-1,)},
    {"delta": 0.0}, {"delta": 1.0}, {"delta": -0.1}, {"delta": 2.0},
])
def test_a_nonsensical_configuration_raises_at_construction(kwargs) -> None:
    with pytest.raises(opp.OpponentError):
        opp.Opponent(**kwargs)


# ---------------------------------------------------------- T-H5 persistence
def _digest(state: dict) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_persistence_round_trip_is_bit_identical() -> None:
    """T-H5. Through real JSON, because that is what memory_node will write, and
    JSON turns int keys into strings — a round trip that never serialises would
    miss it. Mutant: MUTANT_DROP_HISTORY → red."""
    model = opp.Opponent()
    _replay(model, _scripted_rounds(200))
    wire = json.dumps(model.to_dict(), sort_keys=True)
    restored = opp.Opponent.from_dict(json.loads(wire))
    assert _digest(restored.to_dict()) == _digest(model.to_dict())
    assert restored.log_evidence() == model.log_evidence()
    assert restored.predict() == model.predict()


def test_two_models_built_in_different_orders_serialise_identically() -> None:
    """A hash comparison over insertion-ordered dicts would test the order of the
    rounds instead of the content of the model. Mutant: drop `sorted` from
    `to_dict` → red."""
    rounds = _scripted_rounds(120)
    a = opp.Opponent()
    _replay(a, rounds)
    b = opp.Opponent.from_dict(json.loads(json.dumps(a.to_dict())))
    # rebuild b's tables by inserting the same contexts in reverse key order
    reordered = {k: dict(reversed(list(v.items()))) for k, v in b.tables.items()}
    b.tables = reordered
    b.rebuild_evidence()
    assert _digest(a.to_dict()) == _digest(b.to_dict())


def test_a_missing_table_falls_back_to_a_fair_coin() -> None:
    """T-H5: no table must mean uniform 1/3, never a crash and never a wait.
    Mutant: `choose()` raising when `predict()` is None → red."""
    model = opp.Opponent(rng=random.Random(2026))
    counts = {m: 0 for m in opp.MOVES}
    for _ in range(3000):
        counts[model.choose()] += 1
    for move, n in counts.items():
        # 3-sigma on Binomial(3000, 1/3): sd ≈ 25.8, so ±90 is ~3.5σ
        assert abs(n - 1000) < 90, f"{move} played {n}/3000 — not a fair coin"


@pytest.mark.parametrize("state", [
    None, [], "{}", {"version": 2, "orders": [2], "tables": {}, "history": []},
    {"orders": [2], "tables": {}, "history": []},                    # no version
    {"version": 1, "tables": {}, "history": []},                     # no orders
    {"version": 1, "orders": [2], "history": []},                    # no tables
    {"version": 1, "orders": [2], "tables": {}},                     # no history
    {"version": 1, "orders": [2], "tables": {"3": {"rw": [1, 0, 0]}}, "history": []},
    {"version": 1, "orders": [2], "tables": {"2": {"rw": [1, 0]}}, "history": []},
    {"version": 1, "orders": [2], "tables": {"2": {"rw": [-1, 0, 1]}}, "history": []},
])
def test_a_malformed_state_raises_rather_than_loading_half_a_model(state) -> None:
    """Half a restored model is worse than none: it plays confidently on counts
    that came from somewhere else. Mutant: replace the checks with `.get(...)`
    defaults → red."""
    with pytest.raises(opp.OpponentError):
        opp.Opponent.from_dict(state)


def test_a_restored_model_keeps_playing_from_where_it_stopped() -> None:
    """The point of persistence is not the hash, it is that round 201 uses what
    rounds 1..200 taught. Mutant: `from_dict` skipping `rebuild_evidence()` →
    the gate reads as shut on a model that has evidence → red."""
    model = opp.Opponent()
    _replay(model, _scripted_rounds(200))
    assert model.gate_open()
    restored = opp.Opponent.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.gate_open()
    assert restored.evidence_matches_recomputation()


# ------------------------------------------------- T-H2 / T-H3 evidence + live
@pytest.fixture(scope="module")
def evidence() -> dict:
    assert EVIDENCE_JSON.exists(), (
        f"{EVIDENCE_JSON.relative_to(REPO_ROOT)} is missing — run "
        "`python3 game/opponent_harness.py` to produce it")
    data = json.loads(EVIDENCE_JSON.read_text())
    assert data["seeds"] == EVIDENCE_SEEDS and data["rounds"] == EVIDENCE_ROUNDS
    return data


def test_the_evidence_file_covers_every_case_the_harness_defines(evidence: dict) -> None:
    """T-F6's sibling: an A/B with only an A. The case list is derived from the
    harness, so a fifth opponent that never gets run is red rather than invisible.
    Mutant: delete one case's row from the JSON → red."""
    declared = set(harness.CASES)
    ran = set(evidence["h2"]) | set(evidence["h3"])
    assert declared == ran, f"defined but never evaluated: {sorted(declared - ran)}"
    assert set(evidence["h2"]) == set(evidence["h2_no_gate"])


@pytest.mark.parametrize("case", EXPLOITABLE)
def test_an_exploitable_opponent_is_exploited(case: str, evidence: dict) -> None:
    """T-H2 ①②: better than the coin it replaced, with a CI that excludes 0."""
    diff = evidence["h2"][case]["diff"]
    assert diff["n"] == EVIDENCE_SEEDS
    assert diff["mean"] > 0, f"{case}: {diff}"
    assert diff["lo"] > 0, f"{case}: CI includes 0 — {diff}"


def test_a_uniform_opponent_is_not_made_worse(evidence: dict) -> None:
    """T-H2 ③, non-inferiority: the CI upper bound must reach 0.

    This is a weak assertion *by necessity* — see the module docstring: expected
    score against a uniform opponent is 0 for every policy, so no policy can fail
    it. It is kept because it would catch a sign error or a scoring inversion, and
    the discriminating assertion for ③ is the Ville bound below.
    """
    model = evidence["h2"]["uniform_random"]["model"]
    assert model["hi"] >= 0.0, f"significantly worse than a coin: {model}"


def test_the_confidence_gate_holds_shut_against_a_fair_coin(evidence: dict) -> None:
    """T-H2 ③, the assertion that actually discriminates. Ville: the exploit may
    open on at most δ = 0.05 of uniform opponents over their whole lifetime, no
    matter how often we look. Mutant: MUTANT_NO_GATE → 1.00 → red."""
    row = evidence["h2"]["uniform_random"]
    assert row["gate_ever_open_fraction"] <= VILLE_SEED_FRACTION_CEILING, row


def test_the_model_survives_an_adaptive_anti_learner(evidence: dict) -> None:
    """T-H2 ④, as amended. The plan asked for a gate-open fraction < 10%; measured
    it is 71% and the model still wins, so the mechanism the threshold encoded
    does not hold and the enforced criterion is the one it was protecting.
    Reported, not gated: the gate fraction. One anti-learner, one-sided (§6.5)."""
    row = evidence["h2"]["anti_learner"]
    assert row["model"]["hi"] >= 0.0, f"loses to the anti-learner: {row['model']}"
    assert 0.0 <= row["gate_open_fraction"]["mean"] <= 1.0


def test_a_conditionally_predictable_opponent_has_uniform_marginals(evidence: dict) -> None:
    """T-H3, fixture validity first. If this opponent's marginals were skewed, the
    exploitation test below could be passed by a move-frequency counter and would
    prove nothing. Mutant: bias the fixture's move distribution → red."""
    marginals = evidence["h3"]["conditional_uniform_marginal"]["user_marginals"]
    for move, p in marginals.items():
        assert abs(p - 1 / 3) < 0.02, f"{move} marginal {p:.4f} is not ~1/3"


def test_conditional_structure_is_exploited_even_with_uniform_marginals(
        evidence: dict) -> None:
    """T-H3. The Round-7 correction as a test: human marginals are 0.36/0.33/0.32,
    statistically indistinguishable from 1/3, so a frequency counter has nothing.
    Mutant: MUTANT_MARGINAL_COUNTER → red."""
    diff = evidence["h3"]["conditional_uniform_marginal"]["diff"]
    assert diff["lo"] > 0, f"conditional structure not exploited: {diff}"


def test_the_order_comparison_is_reported_for_every_arm(evidence: dict) -> None:
    """T-H4 is report-only. Asserting a winner here would promote a fact about our
    synthetic fixtures — which are order-1 processes by construction, and where
    order-1 duly scores highest — into a claim about people. What is checked is
    that all three arms ran, so the table cannot become an A/B with only an A."""
    assert set(evidence["h4"]) == {"order1", "order2_4", "order2_6"}
    for label, row in evidence["h4"].items():
        for case, ci in row.items():
            assert ci["n"] == EVIDENCE_SEEDS, f"{label}/{case} ran {ci['n']} seeds"


@pytest.mark.slow
@pytest.mark.parametrize("case", EXPLOITABLE + ("uniform_random", "anti_learner"))
def test_the_evidence_is_still_derivable_from_the_source(case: str,
                                                         evidence: dict) -> None:
    """Provenance liveness. A recorded number with no way to re-derive it is a
    stamp, not a measurement: it keeps reading as verified after the code that
    produced it changed. 20 seeds, direction only — this checks the code still
    produces that claim, it is not a second estimate of it.
    Mutant: any of the named mutants above → the sign or the gate fraction moves.
    """
    live = harness.run_case(case, seeds=LIVE_SEEDS)
    recorded = evidence["h2"][case]
    if case == "uniform_random":
        assert live["gate_ever_open_fraction"] <= 3 / LIVE_SEEDS
        return
    assert live["diff"]["mean"] > 0, f"{case} no longer beats the coin: {live['diff']}"
    # within a factor of two of the 100-seed record, in the same direction
    assert 0.5 <= live["diff"]["mean"] / recorded["diff"]["mean"] <= 2.0, (
        f"{case}: live {live['diff']['mean']:+.4f} vs recorded "
        f"{recorded['diff']['mean']:+.4f}")


# ------------------------------------------------------------------- mutations
@pytest.mark.mutation
def test_no_mutant_control() -> None:
    """A broken harness reports success too (§6.0 rule 2). The unmutated copy must
    import and behave, or every red below is uninterpretable."""
    mod = load_mutant(Path(_control_dir()))
    model = mod.Opponent(rng=random.Random(1234))
    _replay(model, _scripted_rounds(200))
    assert model.gate_open() and model.predict() is not None
    assert model.choose() in mod.MOVES
    assert _digest(model.to_dict()) == _digest(
        _reference_model().to_dict()), "the unmutated copy already differs"


_CONTROL: dict[str, str] = {}


def _control_dir() -> str:
    import tempfile
    if "dir" not in _CONTROL:
        _CONTROL["dir"] = tempfile.mkdtemp(prefix="opponent_control_")
    return _CONTROL["dir"]


def _reference_model():
    model = opp.Opponent(rng=random.Random(1234))
    _replay(model, _scripted_rounds(200))
    return model


@pytest.mark.mutation
def test_mutant_global_rng_breaks_reproducibility(tmp_path: Path) -> None:
    """Kills: test_the_module_never_reaches_for_the_global_random_source and
    test_a_seeded_run_is_bit_identical."""
    mod = load_mutant(tmp_path, *MUTANT_GLOBAL_RNG)
    # the static check, re-run against the mutated source
    offenders = [n.attr for n in ast.walk(_tree(tmp_path / "game" / "opponent.py"))
                 if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == "random"
                 and n.attr != "Random"]
    assert offenders, "the mutant did not introduce a global random call"

    def run(seed: int, global_seed: int) -> str:
        random.seed(global_seed)      # the source the mutant now reads instead
        model = mod.Opponent(rng=random.Random(seed))
        moves = []
        for user, robot in _scripted_rounds(120):
            moves.append(model.choose())
            model.observe(user, robot)
        return hashlib.sha256("".join(moves).encode()).hexdigest()

    assert run(1234, 0) != run(1234, 999), (
        "same injected seed, different global state, identical run — the mutant "
        "is still reproducible, so this test is not exercising the mutation")


@pytest.mark.mutation
@pytest.mark.slow
def test_mutant_no_gate_breaks_the_ville_bound(tmp_path: Path) -> None:
    """Kills: test_the_confidence_gate_holds_shut_against_a_fair_coin.

    Also records what the mutant does *not* break, because that is the finding:
    the ungated arm's score against a uniform opponent stays statistically
    indistinguishable from the coin's, exactly as the zero-sum symmetry requires.
    """
    mod = load_mutant(tmp_path, *MUTANT_NO_GATE)
    live = harness.run_case("uniform_random", seeds=LIVE_SEEDS,
                            opponent_cls=mod.Opponent)
    assert live["gate_ever_open_fraction"] > VILLE_SEED_FRACTION_CEILING
    assert live["gate_open_fraction"]["mean"] == 1.0
    assert live["model"]["hi"] >= 0.0, (
        "the ungated arm became significantly worse than a coin — that would "
        "contradict the zero-sum symmetry argument in this module's docstring")


@pytest.mark.mutation
@pytest.mark.slow
def test_mutant_marginal_counter_stops_exploiting_conditional_structure(
        tmp_path: Path) -> None:
    """Kills: test_conditional_structure_is_exploited_even_with_uniform_marginals.

    Collapsing every context to one bucket *is* a move-frequency counter, which is
    the design the Round-7 marginal finding ruled out. Against an opponent with
    uniform marginals it must find nothing.
    """
    mod = load_mutant(tmp_path, *MUTANT_MARGINAL_COUNTER)
    live = harness.run_case("conditional_uniform_marginal", seeds=LIVE_SEEDS,
                            opponent_cls=mod.Opponent)
    assert live["diff"]["lo"] <= 0.0, (
        f"a marginal counter still exploited it: {live['diff']} — the fixture's "
        "marginals are probably not uniform, so T-H3 proves nothing")
    healthy = harness.run_case("conditional_uniform_marginal", seeds=LIVE_SEEDS)
    assert healthy["diff"]["lo"] > 0.0, "control arm also found nothing"


@pytest.mark.mutation
def test_mutant_drop_history_breaks_the_round_trip(tmp_path: Path) -> None:
    """Kills: test_persistence_round_trip_is_bit_identical."""
    mod = load_mutant(tmp_path, *MUTANT_DROP_HISTORY)
    model = mod.Opponent()
    _replay(model, _scripted_rounds(120))
    restored = mod.Opponent.from_dict(json.loads(json.dumps(model.to_dict())))
    assert _digest(restored.to_dict()) != _digest(model.to_dict())


@pytest.mark.mutation
def test_mutant_stale_evidence_is_caught_by_the_recomputation_check(
        tmp_path: Path) -> None:
    """Kills: test_the_incremental_evidence_equals_the_sum_over_contexts.

    And note the second assertion: a stale running total leaves the gate shut
    forever, so this mutant would also drain every exploitation test — which is
    why the cheap equality check is worth having at all.
    """
    mod = load_mutant(tmp_path, *MUTANT_STALE_EVIDENCE)
    model = mod.Opponent()
    _replay(model, _scripted_rounds(200))
    assert not model.evidence_matches_recomputation()
    assert not model.gate_open()
    assert model.predict() is None


@pytest.mark.mutation
def test_mutant_flipped_beats_moves_the_judgement_too(tmp_path: Path) -> None:
    """Kills: test_the_scoring_table_has_exactly_one_creator (the second half).

    The old mutant flipped `BEATS` and looked for divergence from `game_node.WINS`.
    With one table there is nothing left to diverge *from* — which is the whole
    point of collapsing them, and also means that mutant can no longer fail. What
    can still fail is delegation: if `game/rps.py` had quietly grown its own copy of
    the relation, flipping `BEATS` here would leave `rps.judge` unmoved.

    So the assertion is that the flip *does* move the judgement. `rps` imports the
    real module, so the comparison is mutant-vs-shipped rather than in-place.
    """
    mod = load_mutant(
        tmp_path,
        ('BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}',
         'BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}'))
    disagreements = [
        (u, r) for u in opp.MOVES for r in opp.MOVES
        if mod.outcome_for_user(u, r) != rps.judge(u, r)[0]]
    assert len(disagreements) == 6, (
        f"flipping BEATS changed {len(disagreements)} of the 9 pairs; the three "
        "ties are invariant under the flip and the other six must all move, or "
        "rps.judge is not reading opponent.BEATS")


def test_the_statistics_helper_refuses_to_average_nothing() -> None:
    """§6.0 rule 5: zero collected cases must be an error, not a pass. A mean over
    an empty seed list would print 0.0 and read like "no effect".
    Mutant: return 0.0 for an empty list → red."""
    with pytest.raises(ValueError):
        harness.mean_ci([])
    single = harness.mean_ci([0.5])
    assert single["n"] == 1 and single["lo"] == single["hi"] == 0.5


def test_the_interval_reports_its_own_n() -> None:
    """A CI without its n invites the reader to forget what it bounds. Here it
    bounds simulation noise over seeds, not agreement with any person."""
    ci = harness.mean_ci([0.1, 0.2, 0.3, 0.4])
    assert ci["n"] == 4
    assert math.isclose(ci["mean"], 0.25)
    assert ci["lo"] < ci["mean"] < ci["hi"]


def teardown_module(module) -> None:  # noqa: ARG001
    if "dir" in _CONTROL:
        shutil.rmtree(_CONTROL["dir"], ignore_errors=True)
