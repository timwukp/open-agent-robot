"""memory_node — the single writer, and in S3 the publisher of the seed profile card.

Publishes /memory/profile (String, JSON) with `transient_local` durability: the
card's `as_dict()`, once, at boot. Subscribes to nothing yet — the episode stream
arrives in S4 and this node is where it will land, because a second writer would
break the atomicity `memory/store.py` exists to provide.

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

import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String

from memory import profile as pf
from memory import store as st

TOPIC = "/memory/profile"
HUD_TOPIC = "/game/events_out"

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


class MemoryNode(Node):
    def __init__(self, store: "st.Store", card: "pf.ProfileCard") -> None:
        super().__init__("memory")
        self.store = store
        self.card = card
        self.pub = self.create_publisher(String, TOPIC, PROFILE_QOS)
        self.pub_hud = self.create_publisher(String, HUD_TOPIC, 5)
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
        rclpy.spin(MemoryNode(store, card))
    finally:
        store.close()


if __name__ == "__main__":
    main()
