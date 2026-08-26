"""memory_node — the single writer, and in S3 the publisher of the seed profile card.

Publishes /memory/profile (String, JSON) with `transient_local` durability: the
card's `as_dict()`, once, at boot. As of R1.1 it also subscribes to /memory/audit and
appends what arrives there, which is how "one writer" survives a second node having
something to record: `agent_node` composes a turn and publishes the record instead of
opening the file, because a second writer would break the atomicity — and the ordering
guarantee — that `memory/store.py` exists to provide. The episode stream (R1.3) lands
here too, for the same reason and through the same shape.

**Why `transient_local` and why no republish timer.**
`agent_node` needs the card on its very first turn, and it starts in whatever order
the shell loop in `infra/run_ec2.sh` happens to start things. A volatile publisher
would deliver the card to whoever was already listening and leave every later
subscriber with nothing until the next publish. `transient_local` hands the last
sample to each late joiner at subscribe time, which is exactly the "read from RAM,
zero round trips" seam the plan specifies.

There is deliberately no periodic republish. A timer would make a subscriber that
forgot to request `transient_local` durability look like it works — it would receive
the card on the second tick — and then that subscriber silently has no profile for
however long the tick is, on the first turn, which is the turn that matters. The
failure has to be immediate or it will be discovered in production.

**Why the resident text is not in the payload.**
Only `facts` goes on the wire; consumers call `memory.profile.parse(...).render()`.
Shipping a pre-rendered string as well would put a second renderer in the system the
moment one consumer builds the block itself, and the token budget measured against
one of them would not be the budget paid by the other.

**Why an over-budget card is a boot failure rather than a warning.**
The card is prepended to every system prompt for the life of the process, so a card
at twice the budget doubles the input cost of every turn and nothing in a running
robot will ever surface that. It is hand-written, the fix is one edit, and this runs
before any interaction — so the refusal costs a restart and the warning costs money
forever. Same reasoning as `memory.store.location_findings` refusing a bad root
instead of logging about it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String

# run_ec2.sh launches this file directly, so sys.path[0] is memory/, not the repo root.
# Without this line `from memory import profile` raises ModuleNotFoundError — there is no
# `memory` package *inside* memory/ — so the single writer died before `rclpy.init()` on
# every boot, from any working directory, and nothing offline could see it: the four other
# launched nodes that need this had it and this one did not.
#
# It is not a missing nicety. It is why the seams the census reports as `wired` through
# this file were not reachable in the running robot at all, which is a stronger statement
# than any row in infra/dormant_seams.tsv makes. `infra/wiring_census.py` now derives this
# requirement from the launch table instead of assuming it — see `bootstrap_findings`,
# whose existence is the only reason a second missing bootstrap would be noticed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import profile as pf  # noqa: E402
from memory import rps_state as rps  # noqa: E402
from memory import store as st  # noqa: E402

TOPIC = "/memory/profile"
HUD_TOPIC = "/game/events_out"

# The RPS pair, and it is a pair on purpose. `game_node` publishes one resolved round
# fire-and-forget and subscribes latched to the accumulated model, so `infra/graph_gaps.tsv`
# counted the two topics as **one** gap from both sides: wiring one end alone would look
# like progress and change nothing, because a table nobody persists is a table nobody can
# restore. Both names are pinned against `game/game_node.py`'s own literals by AST in
# tests/test_rps_wiring.py, so a rename on one side cannot pass on the other.
RPS_ROUND_TOPIC = "/game/rps_round"
RPS_TABLE_TOPIC = "/memory/rps_table"

# The single writer's inbox. Any node with something for `audit.jsonl` publishes here
# instead of opening the file, which is what keeps "one writer" true for code that has
# never heard of this module. agent_node holds a pinned copy of this name — see
# tests/test_profile_wiring.py, which reads both assignments by AST.
AUDIT_TOPIC = "/memory/audit"

# The character-proxy calibration, measured once against a real `usage.inputTokens`
# and justified where that measurement is recorded, in tests/test_skills_loader.py.
# It is passed into `profile.budget_findings` rather than imported by it, because a
# pure module must not import from tests/ — which does leave this line as a second
# copy of a number. `tests/test_profile.py::test_the_node_uses_the_one_calibration`
# reads this assignment by AST and asserts it equals the original, so the copy cannot
# drift without going red. Copies are acceptable; unpinned copies are not.
PROXY_RATIO = 0.79

PROFILE_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Same four settings as PROFILE_QOS, and a separate object rather than an alias.
#
# The argument for `transient_local` is the profile card's, restated because the consumer
# is different: `game_node` subscribes latched (`game/game_node.py`'s `LATCHED`) and starts
# in whatever order the shell loop in `infra/run_ec2.sh` happens to start things. A
# volatile publisher would hand the restored model to whoever was already listening, and a
# `game_node` that came up second would play the 1/3 coin for its whole session with
# nothing anywhere saying so — the exact failure `infra/graph_gaps.tsv` costed at ~120
# rounds, arriving through wiring instead of through the missing wire.
#
# Not `PROFILE_QOS` itself, and not because two objects are safer: they are two decisions
# that happen to agree today. The card is published once per boot and the table is
# republished after every round, so `depth` is a genuinely separate choice — and an alias
# would let one of them be re-tuned while silently re-tuning the other.
TABLE_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class MemoryNode(Node):
    def __init__(self, store: "st.Store", card: "pf.ProfileCard",
                 table: "rps.RpsState") -> None:
        super().__init__("memory")
        self.store = store
        self.card = card
        # Loaded in `main`, not here, for the same reason the card is: `rps_state.load`
        # can refuse a present-but-unreadable table, and a refusal that happens inside a
        # constructed Node is a refusal that has already created publishers.
        self.table = table
        self.pub = self.create_publisher(String, TOPIC, PROFILE_QOS)
        self.pub_hud = self.create_publisher(String, HUD_TOPIC, 5)
        self.pub_table = self.create_publisher(String, RPS_TABLE_TOPIC, TABLE_QOS)
        payload = json.dumps(card.as_dict(), ensure_ascii=False, sort_keys=True)
        self.pub.publish(String(data=payload))
        self.get_logger().info(
            f"profile card published on {TOPIC}: {len(card)} facts, "
            f"{len(payload)} bytes, read-only until S5")
        self.pub_hud.publish(String(data=json.dumps(
            {"hud": f"🧠 記憶:{len(card)} 條事實"}, ensure_ascii=False)))

        # The card is a snapshot, so the fact that it loaded is itself an auditable
        # event: "what did the robot know at boot" is unanswerable later otherwise.
        self.store.append("audit", {
            "kind": "profile_loaded",
            "facts": len(card),
            "kinds": sorted({f.kind for f in card.facts}),
            "locked": st.is_locked(pf.CARD_NAME),
        })

        # Published only when there are rounds to publish, which is a decision and not a
        # shortcut. An `Opponent` with zero observations is the model `game_node` already
        # built at boot, so a latched empty table carries no information — and it does
        # produce a receipt: `on_rps_table` would log "restored opponent model:
        # observations per order {2: 0, 3: 0, 4: 0}, gate shut" for a restore that did not
        # happen. Because the sample is latched, that false receipt is also what a
        # `game_node` joining an hour later reads.
        if self.table.rounds:
            self.pub_table.publish(String(data=self.table.payload()))
            self.get_logger().info(
                f"rps table published on {RPS_TABLE_TOPIC}: "
                f"{len(self.table.rounds)} rounds, observations per order "
                f"{self.table.observations()}, {self.table.orphans()} unattributed")
        else:
            # `existed` is why `RpsState` carries it: "the table was read and is empty" is
            # a different fact from "there is no table", they produce the same publish
            # (none), and only the log and the audit record can tell them apart.
            self.get_logger().info(
                "no rps table to publish — "
                + ("read and empty" if self.table.existed else "no file yet"))

        # Auditable for the same reason `profile_loaded` is: "what did the robot know at
        # boot" includes the opponent model. `orphans` is here because it is the number
        # that says how much of this table a deletion cannot attribute to an episode —
        # which is *every* round until R1.3 mints episode ids, so this record is where
        # that debt is visible rather than inferred. Appended even when nothing was
        # restored, so a boot that found no table is distinguishable in `audit.jsonl` from
        # a boot where this code did not run at all — which is exactly the confusion
        # R1.1b was: a node that never started looked like a node with nothing to say.
        self.store.append("audit", {
            "kind": "rps_restored",
            "existed": self.table.existed,
            "rounds": len(self.table.rounds),
            "orphans": self.table.orphans(),
        })

        # Both created last, so `profile_loaded` is seq 1 on every boot: the record saying
        # what the robot knew when it started must not be able to land behind a record
        # from a node that started first. `rps_restored` is seq 2 for the same reason and
        # by the same construction — it is appended above, before either subscription
        # exists, so no arriving round can overtake it.
        self.create_subscription(String, AUDIT_TOPIC, self.on_audit, 10)
        self.create_subscription(String, RPS_ROUND_TOPIC, self.on_rps_round, 10)

    def on_audit(self, msg: String) -> None:
        """Append one audit record produced by another node — the single-writer hop.

        This callback deliberately does **not** pre-validate the record. `check_audit`
        runs inside `Store.append`, and a second checker of one format is a second thing
        that has to be kept in step; the writer's refusal is the one that decides what
        is on disk, so it is the only one worth having. What this method does own is that
        a refusal must not take the node down: a malformed record from another node is a
        defect in that node, not a reason for the robot to lose its memory for the rest
        of the session. So it is logged, and the log line names what was refused.

        The `seq` in the success line is the store's own, which is what makes this a
        receipt rather than a hope: `append` returns the sequence number it took, and a
        record refused for its contents never takes one — a gap in `audit.jsonl` is
        indistinguishable from a deleted record.
        """
        try:
            record = json.loads(msg.data)
            seq = self.store.append("audit", record)
        except Exception as exc:                                    # noqa: BLE001
            self.get_logger().error(f"rejected audit record ({exc})")
            return
        self.get_logger().info(
            f"audit record {record.get('kind')!r} appended as seq {seq}")

    def on_rps_round(self, msg: String) -> None:
        """Persist one resolved round, adopt it, republish the model. In that order.

        The order is the whole content of this method. `candidate` validates and builds
        the whole file including the new round while mutating nothing, `write_snapshot`
        takes it, and only then does `commit` let the in-memory model count it. Reversed,
        a round the writer refuses is already in the counts and the *next* round persists
        that as though it had happened.

        **What that ordering guarantees is weaker than "durable first", and the weaker
        statement is the one to hold.** `write_snapshot` enqueues and returns; the
        `os.replace` happens on the writer thread. So the drift is bounded rather than
        removed — the model can be at most one round ahead of the disk, and the round that
        discovers an asynchronous failure is the one refused, because
        `Store._raise_if_failed` runs on entry to the *next* call. `memory/rps_state.py`'s
        `candidate` states the full boundary and `tests/test_rps_wiring.py` pins it.

        A refusal never takes the node down, for `on_audit`'s reason: a malformed round is
        a defect in the sender, and the robot losing its memory for the rest of the session
        is a worse outcome than one unlearned round. The log line names what was refused.

        The republish is unconditional here — unlike the boot publish — because by this
        point `self.table.rounds` is non-empty by construction, so the "an empty table is
        a false receipt" case cannot arise.
        """
        try:
            payload = json.loads(msg.data)
            record, whole = self.table.candidate(payload["user"], payload["robot"])
            seq = self.store.write_snapshot(rps.ARTIFACT, whole)
            self.table.commit(record)
        except Exception as exc:                                    # noqa: BLE001
            self.get_logger().error(f"rejected rps round ({exc})")
            return
        self.pub_table.publish(String(data=self.table.payload()))
        self.get_logger().info(
            f"round {record['user']}/{record['robot']} -> {record['outcome']} "
            f"snapshot queued as seq {seq}; {len(self.table.rounds)} rounds, "
            f"observations per order {self.table.observations()}")


def main() -> None:
    rclpy.init()
    store = st.Store()
    log = get_logger("memory")
    try:
        # The policy — absent vs malformed vs over budget — is in `memory/profile.py`,
        # which imports no ROS: a decision made in this file could only be tested on
        # the instance, and a test that only runs on the instance does not get run.
        card = pf.boot_card(store.root, PROXY_RATIO)
        if card is None:
            log.error(pf.missing_seed_hint(store.root / pf.CARD_NAME))
            # Exit non-zero rather than spinning with nothing to publish:
            # `run_ec2.sh status` then reports a dead node, which is a visible
            # "this robot has no memory" instead of a silent one. The exception
            # propagates through the `finally` below, so the store is closed once.
            raise SystemExit(2)
        # Same policy as the card and the same reason, from `rps_state.load`'s docstring:
        # a table that is present and unreadable stops the node, because the alternative
        # is that the first round calls `write_snapshot` and atomically replaces a file a
        # human could still have repaired. A *missing* file is not an error — that is a
        # first-ever run, which starts from no counts.
        try:
            table = rps.load(store.root)
        except rps.RpsStateError as exc:
            log.error(f"{exc} — refusing to start rather than overwrite it")
            raise SystemExit(2)
        rclpy.spin(MemoryNode(store, card, table))
    finally:
        store.close()


if __name__ == "__main__":
    main()
