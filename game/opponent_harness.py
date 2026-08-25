"""game/opponent_harness.py — offline evaluation of game/opponent.py (T-H2..H4).

Pure and offline: no ROS, no boto3, no network, no GPU, $0. Every synthetic
opponent is a plain function of the round history plus an injected `random.Random`,
so a case is reproducible from `(case, seed, arm)` alone.

Run it directly to write the evidence JSON that docs/test-design.md cites:

    python3 game/opponent_harness.py            # -> docs/evidence/track-b/opponent-h2.json
    python3 game/opponent_harness.py --arm-cost # -> docs/evidence/track-b/arm-cost.json

`tests/test_opponent.py` imports `run_matrix()` and asserts the pass criteria on
the numbers, so the JSON is the record and pytest is the gate — a number typed
into prose is an unverified number (§6.0 rule 6).

**Every case is judged on its own row.** A mean over the four cases would let a
large win on the exploitable opponents pay for a loss against the anti-learner,
which is precisely the failure the confidence gate exists to prevent.

Two arms per case, on the same seed: `model` plays `Opponent.choose()`, `uniform`
plays a fair 1/3 coin. The paired per-seed difference is the estimate, because the
absolute score depends on how exploitable the synthetic opponent happens to be and
the question is only ever "better than the coin it replaced".
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.opponent import (COUNTER, MOVES, Opponent,  # noqa: E402
                           outcome_for_user)

ROUNDS = 300        # matches the 354-person × 300-round human dataset
SEEDS = 100
EVIDENCE = Path(__file__).resolve().parent.parent / "docs" / "evidence" / "track-b"

_FORWARD = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
_BACKWARD = {v: k for k, v in _FORWARD.items()}


# --------------------------------------------------------------- the opponents
# Each takes (rng, rounds) and returns the user's next move. `rounds` is the list
# of {"user", "robot", "user_outcome"} dicts for the rounds already played.

def opp_uniform(rng: random.Random, rounds: list[dict]) -> str:
    return MOVES[rng.randrange(3)]


def opp_win_stay_lose_shift(rng: random.Random, rounds: list[dict],
                            noise: float = 0.15) -> str:
    """The most-reported human heuristic: keep a winning move, move away from a
    losing one. Noise keeps it from being a trivial deterministic target."""
    if not rounds or rng.random() < noise:
        return opp_uniform(rng, rounds)
    last = rounds[-1]
    if last["user_outcome"] == "win":
        return last["user"]
    if last["user_outcome"] == "lose":
        return COUNTER[last["robot"]]       # what would have beaten them
    return _FORWARD[last["user"]]


def opp_markov1(rng: random.Random, rounds: list[dict],
                stay: float = 0.15) -> str:
    """Order-1 chain on the user's own previous move, cycling forward. Its
    stationary distribution is uniform by symmetry, so a marginal counter sees
    nothing here either — that is deliberate."""
    if not rounds:
        return opp_uniform(rng, rounds)
    prev = rounds[-1]["user"]
    r = rng.random()
    if r < stay:
        return prev
    if r < stay + (1.0 - stay) * 0.82:
        return _FORWARD[prev]
    return _BACKWARD[prev]


def opp_conditional_uniform(rng: random.Random, rounds: list[dict],
                            noise: float = 0.10) -> str:
    """T-H3's opponent: **uniform marginals, strong conditional structure.**

    The move is a deterministic cycle keyed on the previous outcome (win → step
    forward, lose → step back, tie → repeat), which spreads the marginal over all
    three moves while making the transition almost fully predictable given the
    outcome. A move-frequency counter is blind to it by construction; that is the
    Round-7 correction (human marginals 0.36/0.33/0.32) written as a test.
    """
    if not rounds or rng.random() < noise:
        return opp_uniform(rng, rounds)
    last = rounds[-1]
    if last["user_outcome"] == "win":
        return _FORWARD[last["user"]]
    if last["user_outcome"] == "lose":
        return _BACKWARD[last["user"]]
    return last["user"]


class AntiLearner:
    """Case ④: an opponent that models *us* and plays the counter.

    It keys an order-2 table on the robot's own recent moves, so it punishes any
    robot that becomes predictable. Stateful, hence a class rather than a
    function — but the state is rebuilt per (seed, arm), so it never leaks.

    This is the one case where an ungated model is expected to lose money, and it
    is the reason the T-H2 mutant is "remove the gate".
    """

    def __init__(self, order: int = 2, min_count: int = 3) -> None:
        self.order = order
        self.min_count = min_count
        self.table: dict[str, list[int]] = {}

    def _ctx(self, rounds: list[dict], upto: int) -> str | None:
        if upto < self.order:
            return None
        return "|".join(r["robot"][0] for r in rounds[upto - self.order:upto])

    def __call__(self, rng: random.Random, rounds: list[dict]) -> str:
        # learn from the round that just finished
        if len(rounds) >= self.order + 1:
            ctx = self._ctx(rounds, len(rounds) - 1)
            if ctx is not None:
                self.table.setdefault(ctx, [0, 0, 0])[
                    MOVES.index(rounds[-1]["robot"])] += 1
        ctx = self._ctx(rounds, len(rounds))
        counts = self.table.get(ctx) if ctx is not None else None
        if not counts or sum(counts) < self.min_count:
            return opp_uniform(rng, rounds)
        predicted = MOVES[max(range(3), key=lambda i: counts[i])]
        return COUNTER[predicted]


class UngatedOpponent(Opponent):
    """The T-H2 mutant as a real subclass: exploit always, prove nothing first.

    Kept in the harness rather than in a test so the ungated arm is a *player*,
    not a stub — the comparison is only meaningful if the mutant plays as well as
    it can while skipping the demonstration step.
    """

    def gate_open(self) -> bool:
        return True


CASES = {
    "win_stay_lose_shift": lambda: opp_win_stay_lose_shift,
    "markov1": lambda: opp_markov1,
    "uniform_random": lambda: opp_uniform,
    "anti_learner": AntiLearner,
    "conditional_uniform_marginal": lambda: opp_conditional_uniform,
}


# ------------------------------------------------------------------ one match
def play(case: str, seed: int, arm: str, rounds: int = ROUNDS,
         orders: tuple[int, ...] = (2, 3, 4), opponent_cls=Opponent) -> dict:
    """One `rounds`-long match. Returns the per-match summary.

    `opponent_cls` is the mutation seam: a test passes a text-patched module's
    `Opponent` and the whole matrix re-runs against it, so a named mutant is
    evaluated by exactly the harness that produced the evidence rather than by a
    parallel reimplementation of it.
    """
    user_rng = random.Random(0x5EED ^ seed)          # the human's own randomness
    robot_rng = random.Random(0xB07 ^ seed)          # the robot's, kept separate
    opponent_fn = CASES[case]()
    model = opponent_cls(orders=orders, rng=robot_rng)

    history: list[dict] = []
    score = 0
    gate_open_rounds = 0
    user_counts = {m: 0 for m in MOVES}
    for _ in range(rounds):
        if model.gate_open():
            gate_open_rounds += 1
        robot = model.choose() if arm == "model" else MOVES[robot_rng.randrange(3)]
        user = opponent_fn(user_rng, history)
        out = outcome_for_user(user, robot)
        user_counts[user] += 1
        score += {"win": -1, "lose": 1, "tie": 0}[out]   # robot's perspective
        history.append({"user": user, "robot": robot, "user_outcome": out})
        model.observe(user, robot)
    return {
        "score": score,
        "mean_per_round": score / rounds,
        "gate_open_fraction": gate_open_rounds / rounds,
        "gate_ever_open": gate_open_rounds > 0,
        "user_marginals": {m: user_counts[m] / rounds for m in MOVES},
        "final_log_evidence": model.log_evidence(),
    }


# ------------------------------------------------------------------ statistics
def mean_ci(values: list[float], z: float = 1.96) -> dict:
    """Mean with a normal-approximation 95% interval.

    Reported with n, because an interval without its n invites the reader to
    forget that n=100 seeds of a *simulation* bounds simulation noise only — it
    says nothing about how well these four synthetic opponents stand in for a
    person. That limit is stated in docs/test-design.md §6.5, not hidden here.
    """
    n = len(values)
    if n == 0:
        raise ValueError("mean_ci on an empty list — refusing to report a mean of nothing")
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "lo": mean, "hi": mean}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean": mean, "sd": math.sqrt(var),
            "lo": mean - z * se, "hi": mean + z * se}


def run_case(case: str, seeds: int = SEEDS, rounds: int = ROUNDS,
             orders: tuple[int, ...] = (2, 3, 4), opponent_cls=Opponent) -> dict:
    model, uniform, diffs = [], [], []
    gate_fracs, ever_open, marginals = [], 0, {m: [] for m in MOVES}
    for seed in range(seeds):
        m = play(case, seed, "model", rounds, orders, opponent_cls)
        u = play(case, seed, "uniform", rounds, orders, opponent_cls)
        model.append(m["mean_per_round"])
        uniform.append(u["mean_per_round"])
        diffs.append(m["mean_per_round"] - u["mean_per_round"])
        gate_fracs.append(m["gate_open_fraction"])
        ever_open += 1 if m["gate_ever_open"] else 0
        for k in MOVES:                     # the opponent judged against the coin,
            marginals[k].append(u["user_marginals"][k])   # not against our model
    return {
        "case": case, "seeds": seeds, "rounds": rounds,
        "orders": list(orders), "opponent_cls": opponent_cls.__name__,
        "model": mean_ci(model), "uniform": mean_ci(uniform),
        "diff": mean_ci(diffs),
        "gate_open_fraction": mean_ci(gate_fracs),
        "gate_ever_open_seeds": ever_open,
        "gate_ever_open_fraction": ever_open / seeds,
        "user_marginals": {m: mean_ci(marginals[m])["mean"] for m in MOVES},
    }


def run_matrix(seeds: int = SEEDS, rounds: int = ROUNDS) -> dict:
    """T-H2 (four opponents, gated and ungated), T-H3 and T-H4 in one pass."""
    out: dict = {"rounds": rounds, "seeds": seeds, "h2": {}, "h2_no_gate": {},
                 "h3": {}, "h4": {}}
    for case in ("win_stay_lose_shift", "markov1", "uniform_random", "anti_learner"):
        out["h2"][case] = run_case(case, seeds, rounds)
        out["h2_no_gate"][case] = run_case(case, seeds, rounds,
                                          opponent_cls=UngatedOpponent)
    out["h3"]["conditional_uniform_marginal"] = run_case(
        "conditional_uniform_marginal", seeds, rounds)
    # T-H4: report only, never a gate. "Mid-range memory is better" is a
    # conclusion about somebody else's opponents; ours are synthetic, so this
    # table describes our fixtures and nothing more.
    for label, orders in (("order1", (1,)), ("order2_4", (2, 3, 4)),
                          ("order2_6", (2, 3, 4, 5, 6))):
        out["h4"][label] = {
            case: run_case(case, seeds, rounds, orders=orders)["diff"]
            for case in ("win_stay_lose_shift", "conditional_uniform_marginal")
        }
    return out


# ------------------------------------------------------------------ T-H6 (M)
def measure_arm_cost(rounds: int = ROUNDS, samples: int = 2000,
                     seed: int = 0) -> dict:
    """How long one `choose()` takes, on a table warmed by `rounds` real rounds.

    This is the M-level half of T-H6 and it is **not** an assertion. The gate is
    the structural one in tests/test_game_wiring.py (the beat path cannot reach the
    model); this number only says whether the arm-time call is anywhere near the
    20 ms tick period, and no test may go red because a laptop was busy — a timing
    assertion earns itself a `skip` within two days (§6.0 rule 4, and the house
    rule that no test in this suite asserts a duration).

    Written to its own JSON rather than into opponent-h2.json on purpose:
    opponent-h2.json is bit-reproducible from its seeds, and mixing a wall-clock
    measurement into it would destroy that property.
    """
    # The table is warmed against the exploitable opponent rather than a coin: an
    # empty table would make `choose()` return on its first branch and time the
    # fallback instead of the arithmetic we are asking about.
    rng = random.Random(seed)
    model = Opponent(rng=random.Random(seed))
    hist: list[dict] = []
    for _ in range(rounds):
        robot = model.choose()
        user = opp_win_stay_lose_shift(rng, hist)
        model.observe(user, robot)
        hist.append({"user": user, "robot": robot,
                     "user_outcome": outcome_for_user(user, robot)})
    contexts = sum(len(t) for t in model.tables.values())
    per_call: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        model.choose()
        per_call.append((time.perf_counter() - t0) * 1e3)
    per_call.sort()

    def pct(p: float) -> float:
        # nearest-rank on a sorted list; n is reported so p95 can be read with the
        # scepticism it deserves (§6.0 rule 4: n < 20 makes a p95 into a max).
        return per_call[min(len(per_call) - 1, int(math.ceil(p * len(per_call))) - 1)]

    return {"samples": samples, "warmup_rounds": rounds, "seed": seed,
            "contexts": contexts, "gate_open": model.gate_open(),
            "tick_period_ms": 20.0, "unit": "ms",
            "p50": pct(0.50), "p95": pct(0.95), "max": per_call[-1]}


def main() -> int:
    if sys.argv[1:] == ["--arm-cost"]:
        result = measure_arm_cost()
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        path = EVIDENCE / "arm-cost.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"choose() over {result['contexts']} contexts: "
              f"p50 {result['p50']:.4f} ms  p95 {result['p95']:.4f} ms  "
              f"max {result['max']:.4f} ms  (n={result['samples']}, "
              f"tick period {result['tick_period_ms']:.0f} ms)")
        print(f"wrote {path}")
        return 0
    result = run_matrix()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / "opponent-h2.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for case, row in result["h2"].items():
        d, g = row["diff"], row["gate_ever_open_fraction"]
        print(f"{case:>30}  diff {d['mean']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]  gate-ever {g:.2f}")
    h3 = result["h3"]["conditional_uniform_marginal"]
    print(f"{'conditional_uniform_marginal':>30}  diff {h3['diff']['mean']:+.4f} "
          f"[{h3['diff']['lo']:+.4f},{h3['diff']['hi']:+.4f}]  "
          f"marginals {h3['user_marginals']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
