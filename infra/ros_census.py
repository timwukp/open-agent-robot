"""The one reader for "which nodes are deployed, and which topics do they touch".

`docs/architecture.svg` and `docs/workflow.md` both draw the ROS 2 graph, and
`tests/test_architecture.py` checks that what they draw is what the code does. That is
three consumers of one fact, and this repo has already been bitten by letting each
consumer derive it separately: `tests/test_voices.py`, `tests/test_disclosure.py`,
`tests/test_game_wiring.py` and `tests/test_tools.py` each walk `create_publisher`
calls for their own single-file question, and a fifth private walk — this one, which is
repo-wide — would be a fifth thing to keep in step. So it lives here once and is
imported.

Two design choices are load-bearing, and both are about a census that could quietly
narrow:

**The node set is executed, not parsed.** `deployed_nodes()` runs
`OAR_DRY=1 infra/run_ec2.sh start-all` and reads the inventory out of its output.
`run_ec2.sh` says in its own comments that `OAR_DRY=1` exists so a guard can "verify
the real inventory on a laptop instead of parsing this file and hoping the parse
matches" — and the `yue` defect is what happens when a second list drifts. A node
added to the here-doc appears here on the next run with nothing to update; a node
added to the tree but *not* to the runner is correctly absent, because a node nothing
launches is not part of the architecture.

**An unreadable topic is a finding, not a skip.** `topics()` resolves the topic
argument as a string literal, or as a module-level string constant in the same file
(`memory/memory_node.py` publishes `TOPIC`, defined at line 50). Anything else raises.
The tempting fallback — skip what you cannot resolve — makes the census silently
narrower than the graph, which is the failure mode where a diagram stays green while a
whole topic goes undrawn.

The output deliberately keeps the *direction* per node rather than a flat topic list,
because the interesting facts in this repo's graph are asymmetries: a publisher with no
subscriber is a message written to nobody, and a subscriber with no publisher is a node
waiting for a message that cannot arrive. Both exist today.

`$HOME` never leaves this module. `run_ec2.sh` prints absolute interpreter paths
(`/Users/<user>/venv/bin/python`), which `infra/redact_scan.py` would flag in any
document that quoted them, so only the repo-relative `.py` path is returned.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# `gateway: would run env MUJOCO_GL=egl <venv>/bin/python gateway/ws_gateway.py --xml …`
# (the real line carries an absolute home path; it is spelled `<venv>` here because
# infra/redact_scan.py refuses that pattern in anything that becomes a blob, and an
# example is not a good enough reason to teach the scanner an exception)
DRY_LINE = re.compile(r"^([a-z0-9_-]+): would run (.+)$", re.M)

# The rclpy surface that creates a graph edge. `create_service`/`create_client` are
# here although the tree has none today: a set spelled as "the two calls we currently
# use" is a name list that cannot notice a new name, and the plan's `/memory/recall`
# service is exactly the addition that would slip past one.
EDGE_CALLS = {
    "create_publisher": "pub",
    "create_subscription": "sub",
    "create_service": "srv",
    "create_client": "cli",
}


@dataclass(frozen=True)
class Edge:
    node: str            # runner name from run_ec2.sh, e.g. "agent"
    path: str            # repo-relative source file
    kind: str            # pub | sub | srv | cli
    topic: str
    lineno: int


@dataclass
class Topic:
    name: str
    pub: list[str] = field(default_factory=list)
    sub: list[str] = field(default_factory=list)
    srv: list[str] = field(default_factory=list)
    cli: list[str] = field(default_factory=list)

    @property
    def dead_end(self) -> bool:
        """Published and nobody is listening — a message written to nobody."""
        return bool(self.pub) and not self.sub

    @property
    def orphan(self) -> bool:
        """Subscribed and nobody publishes — a node waiting for what cannot arrive."""
        return bool(self.sub) and not self.pub


def deployed_nodes(root: Path = REPO_ROOT) -> dict[str, str]:
    """Runner name -> repo-relative source file, from the runner's own dry run.

    Raises ValueError if the inventory is empty or a launch line names no `.py` file:
    an empty census and a clean tree have the same shape, so the empty one must be
    loud (docs/test-design.md §6.0 rule 5).
    """
    script = root / "infra" / "run_ec2.sh"
    proc = subprocess.run(
        ["bash", str(script), "start-all"],
        cwd=str(root), capture_output=True, text=True,
        env={**os.environ, "OAR_DRY": "1"},
    )
    if proc.returncode != 0:
        raise ValueError(f"infra/run_ec2.sh start-all exited {proc.returncode} under "
                         f"OAR_DRY=1: {proc.stderr.strip()[:400]}")
    found: dict[str, str] = {}
    for name, cmd in DRY_LINE.findall(proc.stdout):
        scripts = [tok for tok in cmd.split() if tok.endswith(".py")]
        if len(scripts) != 1:
            raise ValueError(f"runner {name!r} launches {len(scripts)} .py files "
                             f"({scripts}) — cannot attribute its topics to one file")
        found[name] = scripts[0]
    if not found:
        raise ValueError("parsed 0 nodes out of `OAR_DRY=1 run_ec2.sh start-all`; a "
                         "census that reads nothing is an error, not an empty result")
    return found


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "string"` bindings, for topics passed by name."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def edges_in(source: str, node: str, path: str) -> list[Edge]:
    """Every graph edge one node file creates.

    The call is matched on attribute name alone, because the receiver differs across
    the tree: `self.create_publisher(...)` in the class-based nodes and
    `node.create_publisher(...)` in `body/body_node.py`, which builds its node in a
    function. Keying on `self.` would have silently dropped `/joint_states`.
    """
    tree = ast.parse(source)
    consts = _module_constants(tree)
    out: list[Edge] = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr not in EDGE_CALLS:
            continue
        if len(call.args) < 2:
            raise ValueError(f"{path}:{call.lineno} calls {func.attr} with "
                             f"{len(call.args)} positional args — cannot read its topic")
        arg = call.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            topic = arg.value
        elif isinstance(arg, ast.Name) and arg.id in consts:
            topic = consts[arg.id]
        else:
            raise ValueError(
                f"{path}:{call.lineno} names its topic as {ast.dump(arg)[:80]}, which "
                f"this census cannot resolve. Skipping it would make the census "
                f"narrower than the graph, so it is a finding: either make the topic a "
                f"literal or a module-level constant, or teach this resolver the form.")
        out.append(Edge(node=node, path=path, kind=EDGE_CALLS[func.attr],
                        topic=topic, lineno=call.lineno))
    return out


def census(root: Path = REPO_ROOT) -> tuple[dict[str, str], list[Edge], dict[str, Topic]]:
    """(runner name -> file, every edge, topic name -> who publishes and subscribes)."""
    nodes = deployed_nodes(root)
    edges: list[Edge] = []
    for name, rel in sorted(nodes.items()):
        src = (root / rel)
        if not src.exists():
            raise ValueError(f"runner {name!r} launches {rel}, which does not exist")
        edges.extend(edges_in(src.read_text(encoding="utf-8"), name, rel))
    if not edges:
        raise ValueError("0 graph edges across the whole runner inventory — the census "
                         "found no publishers or subscribers, which cannot be true")
    topics: dict[str, Topic] = {}
    for e in edges:
        t = topics.setdefault(e.topic, Topic(e.topic))
        getattr(t, e.kind).append(e.node)
    for t in topics.values():
        for kind in EDGE_CALLS.values():
            getattr(t, kind).sort()
    return nodes, edges, dict(sorted(topics.items()))
