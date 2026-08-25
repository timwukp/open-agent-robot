"""game/rps.py — the pure part of one rock-paper-scissors round.

Zero ROS, zero boto3, zero network, zero global randomness, so the two decisions
that are easy to get quietly wrong can be tested on a laptop instead of inferred
from a node that needs `rclpy` to import at all:

**1. Reading the user's throw out of a sample window must be deterministic.**
The shipped implementation was `max(set(window), key=window.count)`, and on a tie
that is decided by *set iteration order* — which for strings depends on the
per-process hash seed. Measured on the same window `[rock, rock, paper, paper]`:
`PYTHONHASHSEED=0` reads **paper**, `PYTHONHASHSEED=2` reads **rock**. Ties are not
exotic here; a hand moving into position spends real time between two shapes, and
the window is 0.70 s wide. So the same throw could be judged differently on two
boots of the same binary, and now that the read also trains the opponent model, a
nondeterministic read does not just misjudge one round — it writes a
nondeterministic transition table. `majority()` therefore iterates a fixed tuple
and states its tie-break rather than inheriting one from the runtime.

**2. There must be exactly one 3x3 table in the system.** `game_node` had its own
`WINS` set while `opponent.py` had `BEATS`; two encodings of one invariant that
nothing forced to agree. If they ever disagreed the robot would announce one
result and train on another, which is the kind of defect that shows up as "the
model is somehow bad" months later. `judge()` derives both points of view from
`opponent.outcome_for_user`, and the robot's view is the *inversion* of the user's
— the polarity is spelled out in `ROBOT_VIEW` because getting it backwards makes
the robot celebrate its own losses, and that is a one-character mistake.
"""
from __future__ import annotations

from game.opponent import MOVES, OpponentError, outcome_for_user

# The robot's point of view is the mirror of the user's. Written as a table and
# not as `"win" if user_outcome == "lose" else ...` so the mutant that flips it is
# a single edit a test can name.
ROBOT_VIEW = {"win": "lose", "lose": "win", "tie": "tie"}

# What `perception/hand_node.py` publishes on /hand_pose that is not a move.
NO_HAND = "none"


def majority(window: list[str]) -> str | None:
    """The most-sampled move in `window`, or None if it holds no move at all.

    `none` samples are dropped here rather than by the caller, so "the window was
    empty" and "the window was all `none`" cannot take different paths.

    **Tie-break is the order of `MOVES` (rock, paper, scissors), deliberately.**
    It is arbitrary — but an arbitrary rule that is written down beats one that
    comes out of the hash seed, and returning None on every tie would turn a
    hand caught between two shapes into a re-throw, which is worse for the
    person than reading it as rock.
    """
    moves = [g for g in window if g in MOVES]
    if not moves:
        return None
    return max(MOVES, key=moves.count)


def judge(user: str, robot: str) -> tuple[str, str]:
    """(outcome for the user, outcome for the robot). Raises on a non-move.

    Raising rather than defaulting: a round whose moves are not moves is a round
    the model must not be trained on, and silently mapping it to `tie` would train
    it on a fiction. The caller decides what to do with the exception; what it may
    not do is record the round.
    """
    user_outcome = outcome_for_user(user, robot)
    return user_outcome, ROBOT_VIEW[user_outcome]


def hand_vocabulary(source: str) -> set[str]:
    """Every gesture label `perception/hand_node.classify` can return.

    Derived from that function's source by AST walk, not written down here: a
    hand-maintained list cannot notice a new name, and the name it would fail to
    notice is exactly the one that reaches `Opponent.observe()` and raises inside
    a ROS callback. See tests/test_game_wiring.py for the assertion that this set
    is `MOVES | {NO_HAND}` and nothing more.
    """
    import ast

    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "classify":
            labels = set()
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                    first = node.value.elts[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        labels.add(first.value)
            if not labels:
                raise OpponentError(
                    "classify() returned no string literals — this parser no longer "
                    "matches the function it reads, and an empty vocabulary would "
                    "pass every subset assertion")
            return labels
    raise OpponentError("no classify() found in the source given to hand_vocabulary()")
