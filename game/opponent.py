"""game/opponent.py — rock-paper-scissors opponent model with an anytime-valid gate.

Zero ROS, zero boto3, zero network, zero global randomness. Replaces
`random.choice(["rock","paper","scissors"])` in game_node — but only after
T-H2/T-H3 are green, because the interesting risk here is that a model makes the
robot *worse* than the coin it replaces.

Why this exists and what shape it has (docs/test-design.md §2.H):

**Marginal frequency counting has nothing to exploit.** Human RPS marginals are
statistically indistinguishable from 1/3 (0.36 / 0.33 / 0.32 over 354 people ×
300 rounds). The signal lives in *conditional* transitions — what a person plays
after they won, lost or tied with a particular move. So the context is a window of
recent (user move, user outcome) pairs, never a running tally of move counts.
T-H3 encodes that: a synthetic opponent with uniform marginals and strong
conditional structure must still be exploited.

**One model, not an ensemble of models with a selector.** Ensembles of RPS
predictors reduce variance without raising the mean (a single 4-strategy agent
scored 47.73 against a 10-agent ensemble's 35.64, SD 21.21 vs 33.12), and a
selector is another thing to be wrong about.

**Exploitation is gated, because a badly fitted model is worse than a coin.**
Sampling 2.4×10⁹ conditional-response parameter sets shows that a randomly chosen
conditional strategy performs *worse* than uniform in expectation. So deviation
from uniform must be demonstrated before it is exploited, and the demonstration
has to survive being checked after every single round — a fixed-n test used as a
stopping rule is not valid under optional continuation, and this is the textbook
optional-continuation setting: we look after every round, forever.

The gate is therefore an **e-value** with a Ville guarantee, not a p-value:

    H0: the user's next move is uniform on {rock, paper, scissors},
        independent of everything observable before it.

Under H0, for any *predictable* predictive distribution q_n, the ratio
q_n(x_n)/(1/3) has conditional expectation 1, so the running product is a
nonnegative martingale with mean 1. Using the Dirichlet(1,1,1) sequential
predictor per context, that product telescopes into a closed form in the counts
alone (the Dirichlet-multinomial / Pólya marginal):

    E_k = ∏_contexts [ 2 · ∏_m n_m! / (n_c + 2)! · 3^{n_c} ]

which is why the persisted state is *only* counts and the e-value is exact rather
than an approximation of some sequential procedure. Orders 2..4 are combined by
averaging, because a mean of e-values is an e-value. Ville's inequality then gives

    P( ever E ≥ 1/δ  |  H0 ) ≤ δ

so `gate_open()` costs at most δ = 0.05 false openings *over the entire lifetime*
of one opponent, no matter how often it is checked. No multiplicity correction
across contexts is needed, and none is applied, because the aggregation happens
inside the martingale rather than across separate tests.

**Why the context is the user's own history and not the robot's move.** Keying the
context on our own moves would hand a counter-learner a lever: it could steer the
context it is predicted from. Keying on the user's own (move, outcome) pairs means
an opponent can only move the context by moving itself.

I first predicted this would keep the gate *shut* against an adaptive
counter-learner — that the anti-learner's moves, conditioned on its own history,
would stay uniform. **The measurement says otherwise** (docs/evidence/track-b/
opponent-h2.json): the gate is open on 71% of rounds against it and the model
still wins, +0.095 per round, 95% CI [+0.088, +0.102]. The mechanism is that the
loop is closed — the robot's moves are a function of the user's own history, so an
opponent reacting to the robot is *thereby* reacting to that history, and becomes
predictable from it. The gate is not what protects us here; being harder to model
than the opponent is. Two limits worth keeping in view: that is one specific
anti-learner (an A-level, one-sided oracle — see docs/test-design.md §6.5), and
the number is a robot-versus-script result with nothing human in it.

**What the gate does buy, measured.** Against a genuinely uniform opponent it
holds the exploit shut on 97% of seeds (0.2% of rounds), which is the Ville
guarantee δ = 0.05 showing up empirically. Its price is small and real: on the
exploitable opponents the ungated arm scores 0.3–0.9 pp per round *better*,
because the gate spends the opening rounds proving what it will later use.

**HRI default (overridable, and a design choice rather than an algorithmic
limit):** the robot exploits only while the gate is open. A robot that always
wins is not fun; `EXPLOIT_RATE` is the dial, not the ceiling of what is possible.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MOVES = ("rock", "paper", "scissors")
# what each move defeats, and what defeats it
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
COUNTER = {beaten: winner for winner, beaten in BEATS.items()}

# Orders 2..4: order-1 is too coarse to hold win-stay/lose-shift plus a repeat
# habit, and beyond 4 a 300-round session cannot fill a context. T-H4 reports
# order-1 / 2..4 / 6 side by side and deliberately does NOT gate on the result —
# "mid-range memory wins" is a conclusion about somebody else's opponents.
ORDERS = (2, 3, 4)
# Lifetime false-opening budget for the gate, in the Ville sense.
DELTA = 0.05
# A context needs this many observations before it is used for prediction. The
# gate says the opponent is non-uniform *somewhere*; this says the particular
# context being asked about has seen enough to answer.
MIN_CONTEXT_COUNT = 3
# 1.0 = always exploit while the gate is open. Lower it to make the robot more
# fun and less sharp; see the HRI note in the module docstring.
EXPLOIT_RATE = 1.0

OUTCOMES = ("win", "lose", "tie")   # from the USER's point of view


class OpponentError(Exception):
    """Malformed move, outcome or persisted state. Raised rather than defaulted:
    a silently ignored round is a model that trains on a subset nobody chose."""


def outcome_for_user(user: str, robot: str) -> str:
    """Who won, from the user's side. Pure 3×3 table, same as game_node's."""
    if user not in MOVES:
        raise OpponentError(f"not a move: {user!r}")
    if robot not in MOVES:
        raise OpponentError(f"not a move: {robot!r}")
    if user == robot:
        return "tie"
    return "win" if BEATS[user] == robot else "lose"


def _log_polya_evidence(counts: tuple[int, int, int]) -> float:
    """log of one context's contribution to the e-value.

    Dirichlet(1,1,1) marginal over the counts, divided by the null's (1/3)^n:

        log[ 2 · ∏ n_m! / (n+2)! ] + n·log 3

    Exactly zero for an empty context, so unseen contexts cost nothing.
    """
    n = sum(counts)
    if n == 0:
        return 0.0
    total = math.log(2.0) - math.lgamma(n + 3)
    total += sum(math.lgamma(c + 1) for c in counts)
    return total + n * math.log(3.0)


@dataclass
class Opponent:
    """Per-opponent conditional model. One instance per human, not per session.

    `rng` is injected so every decision is reproducible from a seed — a model
    that reads the global random source produces reds nobody can reproduce, which
    get filed as flakes and then ignored.
    """
    orders: tuple[int, ...] = ORDERS
    delta: float = DELTA
    min_context_count: int = MIN_CONTEXT_COUNT
    exploit_rate: float = EXPLOIT_RATE
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    # order -> context string -> [rock, paper, scissors] counts of the user's
    # NEXT move. Plain dicts, but every iteration below is over sorted keys:
    # T-H4/T-H1 require bit-identical output across runs, and dict order is a
    # promise Python makes about insertion, not about two differently-built
    # tables that should be equal.
    tables: dict[int, dict[str, list[int]]] = field(default_factory=dict)
    history: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.orders or min(self.orders) < 1:
            raise OpponentError(f"orders must be positive and non-empty: {self.orders}")
        if not 0.0 < self.delta < 1.0:
            raise OpponentError(f"delta must be in (0,1): {self.delta}")
        for k in self.orders:
            self.tables.setdefault(k, {})
        # Per-context contribution to the log e-value, cached so a round costs
        # four lgamma calls per order instead of four per *context* per order —
        # 10⁸ lgamma calls over the harness, the difference between an evaluation
        # that runs in CI and one that does not.
        #
        # The total is `math.fsum` over these values and never a running `+=`.
        # That matters more than it looks: fsum computes the exact sum and rounds
        # once, so it is independent of iteration order, whereas a running total
        # accumulates in *round* order and a rebuild from the table accumulates in
        # *context* order. Those two differ in the last two ulp, and the value is
        # then compared against a threshold — so a model restored from disk could
        # legitimately open the gate that the same model replayed keeps shut.
        # Measured before this was fixed: 35.963475426014185 vs 35.96347542601416.
        self._ctx_e: dict[int, dict[str, float]] = {}
        self._cached: float | None = None
        self.rebuild_evidence()

    # ---------------------------------------------------------------- context
    @staticmethod
    def _token(user: str, robot: str) -> str:
        return f"{user[0]}{outcome_for_user(user, robot)[0]}"

    def _context(self, k: int, upto: int | None = None) -> str | None:
        """The order-k context ending at `upto` (default: now), or None if the
        history is too short. Built from the USER's move+outcome pairs only —
        see the docstring on why the robot's own move is not in here."""
        hist = self.history if upto is None else self.history[:upto]
        if len(hist) < k:
            return None
        return "|".join(self._token(u, r) for u, r in hist[-k:])

    # ---------------------------------------------------------------- observe
    def observe(self, user: str, robot: str) -> None:
        """Record one resolved round. The reward here is measured by the hand
        classifier, not by a language model — that is the whole reason this is
        the one place in the system where learning is admissible."""
        outcome_for_user(user, robot)      # validates both moves, loudly
        idx = MOVES.index(user)
        for k in self.orders:
            ctx = self._context(k)
            if ctx is None:
                continue
            counts = self.tables[k].setdefault(ctx, [0, 0, 0])
            counts[idx] += 1
            self._ctx_e[k][ctx] = _log_polya_evidence(tuple(counts))
        self._cached = None
        self.history.append((user, robot))
        # Only the last max(orders) rounds can ever form a context, so the state
        # stays bounded (≤ Σ 9^k contexts) instead of growing for the life of one
        # person. Episode-level history belongs to memory_node, not here.
        if len(self.history) > max(self.orders):
            del self.history[:-max(self.orders)]

    # --------------------------------------------------------------- evidence
    def _fresh_evidence(self) -> dict[int, dict[str, float]]:
        """Every per-context contribution, derived from the tables alone."""
        return {k: {c: _log_polya_evidence(tuple(self.tables[k][c]))
                    for c in self.tables.get(k, {})}
                for k in self.orders}

    def rebuild_evidence(self) -> None:
        """Recompute every cached contribution from the tables.

        Called on construction and after `from_dict`. Because the total is an
        fsum, a model rebuilt here and the same model reached by replaying rounds
        agree to the bit — that equality is what T-H5 asserts.
        """
        self._ctx_e = self._fresh_evidence()
        self._cached = None

    def evidence_matches_recomputation(self) -> bool:
        """Does the cache still equal what the tables say?

        The cache is an optimisation, and an optimisation that drifts moves the
        gate without moving any other test — so the equality is asserted rather
        than assumed (T-H1). Exact, not tolerance-based: both sides are fsums over
        the same addends, so anything but equality is a bug.

        **Read-only on purpose.** The obvious implementation calls
        `rebuild_evidence()` and compares before to after — and thereby *repairs*
        the drift it was asked to detect. Measured: with that version, the
        stale-cache mutant reported drift and then passed `gate_open()`, because
        asking the question had fixed the answer. A diagnostic you cannot put in
        front of an assertion is not a diagnostic.
        """
        return self._fresh_evidence() == self._ctx_e

    def log_evidence(self) -> float:
        """log of the combined e-value. 0.0 means "exactly no evidence"."""
        if self._cached is not None:
            return self._cached
        self._cached = self._compute_log_evidence()
        return self._cached

    def _compute_log_evidence(self) -> float:
        # fsum, not sum: exact then rounded once, hence order-independent.
        per_order = [math.fsum(self._ctx_e[k].values()) for k in sorted(self.orders)]
        if not per_order:
            return 0.0
        # mean of e-values is an e-value; done in log space
        top = max(per_order)
        if top == -math.inf:
            return 0.0
        return top + math.log(
            sum(math.exp(x - top) for x in per_order) / len(per_order))

    def evidence(self) -> float:
        """The e-value itself. Capped so a long confident session cannot
        overflow into inf and make comparisons meaningless."""
        return math.exp(min(self.log_evidence(), 700.0))

    def gate_open(self) -> bool:
        """Has this opponent demonstrably deviated from uniform?

        Valid at every round without correction (Ville). Closed means "play the
        coin", which is unexploitable — the cost of being wrong here is zero, not
        negative, and that asymmetry is the entire point of the gate.
        """
        return self.log_evidence() >= math.log(1.0 / self.delta)

    # ---------------------------------------------------------------- predict
    def predict(self) -> tuple[float, float, float] | None:
        """Predicted distribution over the user's next move, or None.

        Backs off from the longest order to the shortest, taking the first
        context with `min_context_count` observations. None means either the gate
        is shut or no context is populated enough — both must lead to uniform
        play, and both are exercised by T-H5.
        """
        if not self.gate_open():
            return None
        for k in sorted(self.orders, reverse=True):
            ctx = self._context(k)
            if ctx is None:
                continue
            counts = self.tables.get(k, {}).get(ctx)
            if counts and sum(counts) >= self.min_context_count:
                # Laplace-smoothed, matching the Dirichlet(1,1,1) the gate uses:
                # a raw argmax on 3 observations would commit to 3/3 certainty.
                total = sum(counts) + 3
                return tuple((c + 1) / total for c in counts)  # type: ignore[return-value]
        return None

    # ----------------------------------------------------------------- choose
    def choose(self) -> str:
        """The robot's move for the next round. Never raises, never blocks.

        Called at arm time, off the beat-critical path (T-H6): the beat may not
        wait for anything, and this is pure arithmetic over a dict, but the
        discipline is that the caller decides *when*, not that the call is fast.
        """
        probs = self.predict()
        if probs is None or self.rng.random() >= self.exploit_rate:
            return self.uniform_move()
        best, best_ev = None, -math.inf
        for robot in MOVES:      # sorted implicitly: MOVES is a fixed tuple
            ev = 0.0
            for p, user in zip(probs, MOVES):
                if user == robot:
                    continue
                ev += p if BEATS[robot] == user else -p
            if ev > best_ev:
                best, best_ev = robot, ev
        # An expected value of 0 means the prediction says nothing useful; play
        # the coin rather than a fixed move an adapting opponent can read.
        return best if best is not None and best_ev > 0.0 else self.uniform_move()

    def uniform_move(self) -> str:
        """The unexploitable fallback. Uses the injected rng, so a seeded run is
        bit-reproducible end to end."""
        return MOVES[self.rng.randrange(3)]

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        """Serialisable state. Keys sorted at every level so two tables built in
        different orders serialise identically — T-H5 compares hashes, and a
        hash comparison over insertion-ordered dicts tests the order of the
        rounds instead of the content of the model."""
        return {
            "version": 1,
            "orders": list(sorted(self.orders)),
            "delta": self.delta,
            "min_context_count": self.min_context_count,
            "exploit_rate": self.exploit_rate,
            "tables": {str(k): {c: list(self.tables[k][c])
                                for c in sorted(self.tables[k])}
                       for k in sorted(self.tables)},
            "history": [list(r) for r in self.history],
        }

    @classmethod
    def from_dict(cls, state: dict, rng: random.Random | None = None) -> "Opponent":
        if not isinstance(state, dict):
            raise OpponentError(f"state is not an object: {type(state).__name__}")
        if state.get("version") != 1:
            raise OpponentError(f"unsupported state version: {state.get('version')!r}")
        for key in ("orders", "tables", "history"):
            if key not in state:
                raise OpponentError(f"state is missing {key!r}")
        obj = cls(
            orders=tuple(state["orders"]),
            delta=state.get("delta", DELTA),
            min_context_count=state.get("min_context_count", MIN_CONTEXT_COUNT),
            exploit_rate=state.get("exploit_rate", EXPLOIT_RATE),
            rng=rng or random.Random(0),
        )
        for k_str, table in state["tables"].items():
            k = int(k_str)
            if k not in obj.orders:
                raise OpponentError(f"state has an order-{k} table but orders={obj.orders}")
            for ctx, counts in table.items():
                if len(counts) != 3 or any(int(c) < 0 for c in counts):
                    raise OpponentError(f"bad counts for order {k} context {ctx!r}: {counts!r}")
                obj.tables[k][ctx] = [int(c) for c in counts]
        obj.history = [(u, r) for u, r in state["history"]]
        for user, robot in obj.history:
            outcome_for_user(user, robot)      # a restored round must be a legal one
        if len(obj.history) > max(obj.orders):
            raise OpponentError(
                f"history holds {len(obj.history)} rounds but only the last "
                f"{max(obj.orders)} can form a context — this state was not "
                "written by to_dict()")
        obj.rebuild_evidence()
        return obj
