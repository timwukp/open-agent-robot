"""The RPS table's two shapes: deletable rounds on disk, an opponent model on the wire.

R1.2 wires `/game/rps_round` -> `memory_node` -> `/memory/rps_table`, and the first thing
that wiring ran into is that the two ends of the pair do **not** speak one format, and
neither of them is wrong:

* `game/game_node.py::on_rps_table` calls `game.opponent.Opponent.from_dict`, so the
  **wire** payload has to be `Opponent.to_dict()` — `version/orders/delta/
  min_context_count/exploit_rate/tables/history`.
* `memory/recall.py::parse_rounds` reads `rps_table.json` and **closes the top level to
  `{"version", "rounds"}`**, raising on anything else. Its stated reason is the deletion:
  "`rounds` is the only key an erasure knows how to filter, so a sibling key holding
  per-episode counts would survive every deletion while nothing anywhere said so."

Those two are not reconcilable in one document, and the conflict is real rather than
cosmetic. `Opponent.to_dict()`'s `tables` are *aggregates*: dropping a round from a list
does not decrement a count, so a table persisted in the model's own shape would keep the
behavioural fingerprint of a person who exercised the right to be forgotten, and no gate
would notice. Write the model's shape to disk and `recall.load_rounds` raises on the file;
write recall's shape to the wire and `from_dict` raises on the message ("state is missing
'orders'") and `game_node` logs "rejected persisted model — keeping RAM".

So this module holds the one asymmetry that resolves it:

    disk  = {"version": 1, "rounds": [{user, robot, outcome, episode}, ...]}   deletable
    wire  = Opponent.to_dict() of those rounds replayed in order                derived

**The wire form is derived and never stored**, which is what makes the asymmetry safe:
there is one direction of derivation, rounds -> model, so the aggregate cannot drift from
the rounds it came from, and an erased round is erased from the only copy that exists.

**Why replay reproduces the table exactly, rather than approximately.** The thing that
would break this is already fixed and already commented, in `Opponent.__post_init__`: the
log e-value is `math.fsum` over per-context contributions and never a running `+=`, so a
model built by N `observe()` calls and the same model rebuilt from its own tables agree to
the last bit (the comment records the two values that differed before the fix,
35.963475426014185 vs 35.96347542601416, and why a threshold made the difference visible).
Replay is N `observe()` calls in the recorded order, so `game_node`'s live table, this
module's replayed table, and the `from_dict` of the payload are the same table.
`Opponent.evidence_matches_recomputation()` is the runtime form of that claim and
`tests/test_rps_wiring.py` asserts it on a replayed model rather than assuming it.

**Why `episode` is written now, as `None`.** It cannot be filled until R1.3 mints episode
ids, and a field is only free to mandate before the first writer exists — R1.2 *is* that
writer. So the field is present from the first record and the honest consequence is
written down instead of discovered: `recall._run_rps` maps a null episode to `ORPHAN` and
`deletion_findings` reports every R1.2-era round as unattributed rather than as a clean
zero. That is the designed behaviour of a table with no attribution, not a defect this
module invented, and it is the reason R1.3 comes next.

**Why a present-but-unreadable table stops the node.** `load` raises, and
`memory_node.main` exits non-zero, for the same reason `memory/profile.py` refuses a
malformed card and `store.location_findings` refuses a bad root: the alternative is that
the first round calls `write_snapshot` and atomically replaces a file a human could still
have repaired. A missing file is not an error — that is a first-ever run, and it starts
from no counts, which is the same safe state as a first-ever conversation.

No ROS in here on purpose. Every decision above would otherwise live in
`memory/memory_node.py`, where it could only be tested on the instance, and a test that
only runs on the instance does not get run.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from game import opponent as opp
from memory import recall as rc

# One creator of the file name. `memory/recall.py` owns it because recall was reading this
# file before anything wrote it, and a second spelling here is a second thing to keep in
# step with `erase`'s rebuild.
ARTIFACT = rc.RPS_ARTIFACT

# The on-disk envelope version. Deliberately the same integer `recall.parse_rounds`
# demands and `Opponent.to_dict` stamps, and deliberately *not* shared with either: the
# disk envelope and the model's state are two formats that happen to be at version 1 at
# the same time, and aliasing them would make one bump silently claim the other.
VERSION = 1


class RpsStateError(Exception):
    """The table on disk cannot be turned into a model, with which round said so."""


def round_record(user: str, robot: str, episode: str | None = None) -> dict:
    """One round in the on-disk shape. Validates both moves before anything is written.

    `outcome` is stored rather than re-derived on read because `recall._run_rps` builds
    its hit label from `entry.get("outcome", "?")`, and a hit labelled `rock/?` names the
    round without saying what happened in it. It is still derived from exactly one place
    — `opponent.outcome_for_user`, the same function `judge` and `observe` go through —
    so the stored value cannot disagree with the model's view of the same round.

    This function is also the validator that makes the write order in
    `memory_node.on_rps_round` safe. `Opponent.observe` validates by calling
    `outcome_for_user` and then indexing `MOVES`; both are covered here, so a round this
    function accepts cannot be refused by `observe` afterwards — which is what allows the
    snapshot to be enqueued *before* the model adopts the round (see `candidate` for what
    that ordering does and does not guarantee; it is weaker than "durable").
    `tests/test_rps_wiring.py` asserts that agreement over every legal pair and a spread
    of illegal ones instead of trusting this paragraph.
    """
    outcome = opp.outcome_for_user(user, robot)
    return {"user": user, "robot": robot, "outcome": outcome, "episode": episode}


def snapshot(rounds) -> dict:
    """The whole file, checked against the reader that is not us before it is returned.

    The `parse_rounds` call is not belt-and-braces. Two readers of one format drift when
    each only checks itself, and this is the writer of a file whose *other* reader decides
    whether a deletion can filter it. So the writer asserts, every time, that what it is
    about to hand `store.write_snapshot` is a file `recall.load_rounds` accepts. The cost
    is one O(rounds) validation per round on a node that is not on the beat path; the
    thing it buys is that a shape recall refuses can never reach the disk in the first
    place, where it would present as "the retrieval path is broken" long afterwards.
    """
    body = {"version": VERSION, "rounds": list(rounds)}
    try:
        rc.parse_rounds(body, where=ARTIFACT)
    except rc.RecallError as exc:
        raise RpsStateError(
            f"refusing to write {ARTIFACT}: the reader that a deletion depends on "
            f"rejects it ({exc})") from exc
    return body


def replay(rounds, *, rng: random.Random | None = None) -> "opp.Opponent":
    """Fold every round into a fresh model, in the recorded order.

    A round that cannot be replayed aborts the whole load rather than being skipped, and
    the reason is `game_node.on_rps_table`'s own: a partially restored table is a model
    trained on a subset nobody chose. Skipping would also be the more expensive failure,
    because the resulting counts look exactly like a person who played differently.

    `rng` defaults to a seeded `Random(0)` and that is safe *here* specifically because
    this model never draws: `memory_node` persists and publishes, it does not play. The
    unseeded rng the player needs is `game_node`'s, which `on_rps_table` keeps by passing
    `rng=self.opponent.rng` into `from_dict` — so this seed cannot reach a throw. A seed
    is chosen over `Random()` so a replay is reproducible in a test; if this model ever
    does draw, deterministic-and-wrong is a failure someone can reproduce.
    """
    model = opp.Opponent(rng=rng or random.Random(0))
    for i, entry in enumerate(rounds):
        if not isinstance(entry, dict):
            raise RpsStateError(f"{ARTIFACT}: round {i} is not an object")
        try:
            model.observe(entry["user"], entry["robot"])
        except (KeyError, opp.OpponentError) as exc:
            raise RpsStateError(
                f"{ARTIFACT}: round {i} cannot be replayed ({exc!r}) — refusing a model "
                "trained on the rounds that happened to parse") from exc
    return model


@dataclass
class RpsState:
    """The rounds, the model they build, and whether the file was there at boot.

    `existed` is carried rather than recomputed because it is the difference between "no
    one has played yet" and "the table was read and is empty", and only the second one
    means a later empty publish would be telling `game_node` something.
    """

    model: "opp.Opponent"
    rounds: list[dict] = field(default_factory=list)
    existed: bool = False

    # ------------------------------------------------------------------ writes
    def candidate(self, user: str, robot: str) -> tuple[dict, dict]:
        """`(record, whole file including it)`. Validates; mutates nothing.

        Split from `commit` so every refusal that can be seen without waiting lands
        before the model counts the round. Do it the other way and a refused round is
        already in the counts, and the *next* round persists that as if it were history;
        this way the worst case is a round that was refused and not learned, which is the
        same state as a round that never arrived.

        **The boundary, because the obvious sentence here would be false.**
        `store.write_snapshot` is not synchronous — it validates, enqueues, and returns a
        sequence number, and the writer thread performs the `os.replace`. So "durable
        before the model adopts it" is *not* what this split buys and must not be written
        down as if it were. What it buys is:

        * everything checkable without the disk is checked first — the moves are legal
          (`round_record`), the whole file is one `recall.load_rounds` accepts
          (`snapshot`), the name is plain, the artifact is not locked, the body is
          JSON-serialisable, and **no earlier write has already failed**
          (`Store._raise_if_failed` runs on entry);
        * an asynchronous failure — a full disk, a revoked permission — is therefore
          discovered on the *next* `write_snapshot` or `append`, which raises before that
          round's `commit`.

        Which bounds the drift rather than removing it: the model can be **at most one
        round ahead of the disk**, and the round that discovers it is the one refused.
        A tighter guarantee would need the queue drained per round, on the node that also
        publishes the table — and `tests/test_rps_wiring.py` pins this bound instead of
        the stronger claim, because a docstring is where an unpinned guarantee rots.
        """
        record = round_record(user, robot)
        return record, snapshot(self.rounds + [record])

    def commit(self, record: dict) -> None:
        """Adopt a round whose snapshot is already durable."""
        self.model.observe(record["user"], record["robot"])
        self.rounds.append(record)
        self.existed = True

    # ------------------------------------------------------------------- reads
    def payload(self) -> str:
        """The wire form: `Opponent.to_dict()` as JSON, sorted like every other payload."""
        return json.dumps(self.model.to_dict(), ensure_ascii=False, sort_keys=True)

    def orphans(self) -> int:
        """Rounds with no episode attribution — every round until R1.3, by construction."""
        return sum(1 for r in self.rounds if not r.get("episode"))

    def observations(self) -> dict[int, int]:
        """Per order, the number of counted transitions.

        Per order and not one total, the same way `game_node` logs it: one round
        increments one counter in every order that had a context, so a sum across orders
        reports roughly three times the rounds played, and order k only ever sees
        rounds-k of them, so the numbers legitimately differ.
        """
        return {k: sum(sum(c) for c in self.model.tables[k].values())
                for k in sorted(self.model.orders)}


def load(root) -> RpsState:
    """Read `rps_table.json` under `root` and replay it. Absent file -> empty state.

    Every failure is re-raised as `RpsStateError` so the caller has one type to catch:
    `memory_node.main` turns it into a non-zero exit, and a caller that has to enumerate
    `RecallError` and `OpponentError` is a caller that will miss the third one.
    """
    path = Path(root) / ARTIFACT
    if not path.exists():
        return RpsState(model=opp.Opponent(rng=random.Random(0)), existed=False)
    try:
        rounds = [dict(r) for r in rc.load_rounds(path)]
    except rc.RecallError as exc:
        raise RpsStateError(f"{ARTIFACT} is present and unreadable ({exc})") from exc
    return RpsState(model=replay(rounds), rounds=rounds, existed=True)
