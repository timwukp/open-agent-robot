"""agent/tools.py — the closed registry of everything the model may invoke.

Zero ROS, zero boto3, zero network, zero clock. This module is the answer to one
question — *what can the model make this robot do that is not speech?* — and it
exists before any MCP client does, for the same reason `memory/store.py`'s `why`
field was made mandatory before anything wrote to `events`: a boundary is free to
draw exactly once, before the first thing that would have to be migrated across
it (docs/test-design.md §2.C1, `feedback_mandatory_field_timing`).

Five defects this module is built around. All five were read out of the code that
exists today, not imagined:

**1. After MCP, the tool surface stops being derivable from the repo.** Today
"which tools can the model reach" is answered by reading one dict. An MCP client
answers it by asking a server at runtime, which means the offline suite
structurally cannot assert the boundary any more — the same shape as a gate whose
membership is a written-down list rather than a derivation, one level worse,
because the list now lives on another machine. So the registry here is
**declarative and closed**: `reconcile()` treats a tool the server offers but the
repo does not declare as a **finding**, never as a tool. Discovery may *narrow*
what is available. It may never widen it.

**2. Tool specs are resident prompt cost, and nothing budgeted them.** In
`agent_node` today `toolConfig` is attached *only on turns where a skill fired*,
so most turns carry no tool schema at all. A resident MCP catalogue changes that
to every turn, forever, and `skills/loader.py`'s L1 budget (`L1_MAX_CHARS`, item
T-I1) has no equivalent for tool schemas. `spec_findings()` gives them one, and it
charges the *serialised* spec, because that is what is sent.

**3. `MAX_TOOL_HOPS` bounds reads, not side effects.** Three hops of
`read_skill_file` is three file reads; three hops of a remote tool can be three
actions, and two of them may not be undoable. So effect class is a declared field
and `MAX_EFFECT_HOPS` is **0** — a tool that declares `MUTATES` cannot be invoked
at all today. That number is not a placeholder to be raised quietly: raising it is
a change to this file, in a PR, next to this paragraph.

**4. One failure string described four different situations.** `loader.run_tool`
returned `("unknown tool …", False)` for a name it did not recognise, and that was
the whole vocabulary. After MCP, a name that does not resolve is *more often*
server drift than model invention, and a transport failure is neither. Folding
them together is the failure this repo has now hit in five other places (a `[]`
with four causes, `no record` vs `not in the set`, `errored` vs `allowed`): so
there are six outcomes, and **exactly one of them means the boundary acted**.
`REFUSED` is a claim about policy. `UNREACHABLE` is a claim about the network. A
report that counts the second as the first says the robot was protected by an
outage.

**5. Speech is not the only way out, and today it is the only one guarded.**
`agent_node` publishes on four topics and runs one tool; `safety/output_guard.py`
— all of S7 — has **zero callers outside the tests**. So `EGRESS` enumerates every
channel the model's output can reach with the enforcer that actually runs on it,
and it distinguishes `NOT_WIRED` (the enforcer is in the repo and nothing calls
it) from `NO_ENFORCER_TODAY` (there is nothing to call). Those two are the same
absence with very different repair costs, and a table that folded them together
would have described S7 as protection.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

# Same one-module-identity argument as agent/context.py: run_ec2.sh launches nodes
# by file path, so sys.path[0] is the node's directory and an un-bootstrapped
# import would give the node and the tests two different registries.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills import loader as _loader   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------- vocabulary
# Where the code that answers the call runs. The distinction is not cosmetic:
# a LOCAL tool's refusal path is in this repo and covered by this suite, while a
# REMOTE tool's is on someone else's machine and cannot be.
LOCAL = "local"
REMOTE = "remote"
ORIGINS = (LOCAL, REMOTE)

# What invoking it costs if it goes wrong. READ_ONLY is recoverable by not
# looking; MUTATES may not be recoverable at all.
READ_ONLY = "read_only"
MUTATES = "mutates"
EFFECTS = (READ_ONLY, MUTATES)

# Outcomes of one invocation. Six, not a boolean, and the list is the point.
OK = "ok"                      # the tool ran and returned
REFUSED = "refused"            # policy said no
UNKNOWN_TOOL = "unknown_tool"  # the name is not in this registry at all
NOT_OFFERED = "not_offered"    # declared here, absent from the live offer set
UNREACHABLE = "unreachable"    # transport failed; we never asked
ERRORED = "errored"            # the executor raised; we do not know what happened
OUTCOMES = (OK, REFUSED, UNKNOWN_TOOL, NOT_OFFERED, UNREACHABLE, ERRORED)

# The single member that licenses the sentence "the boundary held". Written as a
# tuple over the outcome set rather than as `outcome != OK`, because `!= OK` is
# how an outage becomes a safety claim.
BOUNDARY_ACTED = (REFUSED,)

# Enforcement states for an egress channel. Three values, because "nothing runs
# on this channel" has two causes that cost different amounts to fix.
ENFORCED = "enforced"
NOT_WIRED = "not_wired"                # the enforcer exists; nothing calls it
NO_ENFORCER_TODAY = "no_enforcer_today"  # there is nothing to call
ENFORCEMENT = (ENFORCED, NOT_WIRED, NO_ENFORCER_TODAY)

# How many times in one turn the model may be handed a tool result. Moved here
# from skills/loader.py, where it had stopped being true of what it bounded: the
# loop it limits is every tool hop, and a module that owns skill files owning the
# bound on all tool use is a guard whose scope no longer matches its name.
MAX_TOOL_HOPS = 3

# A tool declaring MUTATES cannot be invoked while this is 0. See defect 3. It is
# a separate number from MAX_TOOL_HOPS on purpose, and T-L4 asserts both, because
# "three reads" and "three irreversible actions" are the same integer describing
# two different exposures.
MAX_EFFECT_HOPS = 0

# Serialised tool schemas, in the character proxy `skills/loader.py` calibrated
# against a real usage.inputTokens reading. Today's single spec measures well
# under this; the ceiling exists so that a resident MCP catalogue has to argue
# with a number instead of appearing for free on every turn.
TOOL_SPEC_CEILING = 400


class ToolError(Exception):
    """A registry or an invocation is malformed. Raised, not returned: a
    malformed *registry* is our bug, while a malformed *call* is the model's and
    comes back as an outcome."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    origin: str
    effect: str
    description: str
    input_schema: Mapping[str, object]
    # None for LOCAL. For REMOTE it names the server whose offer set has to
    # contain this tool before it may be called.
    server: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ToolError("a tool with no name cannot be dispatched")
        if self.origin not in ORIGINS:
            raise ToolError(f"{self.name}: unknown origin {self.origin!r}")
        if self.effect not in EFFECTS:
            raise ToolError(f"{self.name}: unknown effect class {self.effect!r}")
        if (self.origin == REMOTE) != bool(self.server):
            # Both directions are errors: a remote tool with no server has no
            # offer set to be reconciled against, and a local tool with one
            # implies a network round trip this suite would never see.
            raise ToolError(
                f"{self.name}: origin {self.origin!r} and server {self.server!r} "
                f"disagree — a remote tool needs a server, a local one must not have one"
            )


@dataclass(frozen=True)
class Invocation:
    name: str
    outcome: str
    text: str
    effect: str | None = None      # None when the name never resolved

    @property
    def boundary_acted(self) -> bool:
        return self.outcome in BOUNDARY_ACTED


# --------------------------------------------------------------------- registry
# Written out, one member per line, with no default and no wildcard. Today it has
# exactly one entry, and that is the honest state of the system: the robot can
# read its own skill files and do nothing else.
REGISTRY: dict[str, ToolSpec] = {
    "read_skill_file": ToolSpec(
        name="read_skill_file",
        origin=LOCAL,
        effect=READ_ONLY,
        description=(
            "Read one file from the robot's skills directory. `path` is relative "
            "to skills/ and must include the skill directory, e.g. "
            "'critical-thinking/methods/bias-detection.md'. Use this to follow a "
            "routing table in a loaded SKILL.md. Never guess paths: read only what "
            "the skill's own routing table names."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {
                "type": "string",
                "description": "path relative to skills/, e.g. 'critical-thinking/SKILL.md'",
            }},
            "required": ["path"],
        },
    ),
}


def _exec_read_skill_file(args: Mapping[str, object], root: Path | None) -> str:
    """Executor bodies are named `_exec_<tool name>` so that the dispatch table
    is *derived* from this module rather than written twice. A second dict mapping
    names to functions is two readers of one format, and the reader that stops
    being edited is the one that goes quietly stale."""
    return _loader.read_skill_file(str(args.get("path", "")),
                                   root=_loader.SKILLS_ROOT if root is None else root)


def executors() -> dict[str, Callable[[Mapping[str, object], Path | None], str]]:
    """The other side of the gate, derived from this module's own namespace."""
    prefix = "_exec_"
    return {name[len(prefix):]: fn for name, fn in globals().items()
            if name.startswith(prefix) and callable(fn)}


# ------------------------------------------------------------------- reconciling
def reconcile(offered: set[str] | None, registry: Mapping[str, ToolSpec] | None = None,
              server: str | None = None) -> list[str]:
    """Compare a server's live offer set against what this repo declares.

    `offered=None` means we never got an offer set — which is *not* an empty one,
    and the difference matters: an empty set says the server has no tools, while
    no set at all says we do not know. Returns findings; never mutates anything.
    """
    reg = REGISTRY if registry is None else registry
    mine = {n for n, spec in reg.items()
            if spec.origin == REMOTE and (server is None or spec.server == server)}
    if offered is None:
        if mine:
            return [f"no offer set was obtained, so {len(mine)} declared remote "
                    f"tool(s) cannot be confirmed: {sorted(mine)}"]
        return []
    findings = []
    for extra in sorted(set(offered) - mine):
        # The load-bearing direction. A tool the server offers and the repo does
        # not declare is a capability nobody reviewed, and the cheap handling —
        # merge it into the catalogue because the server said so — is exactly how
        # the tool surface stops being a property of this repo.
        findings.append(
            f"{extra!r} is offered by the server and not declared in agent/tools.py — "
            f"refused; discovery may narrow the registry, never widen it")
    for missing in sorted(mine - set(offered)):
        findings.append(
            f"{missing!r} is declared here and not offered by the server — "
            f"calls will report {NOT_OFFERED!r}, which is server drift, not a refusal")
    return findings


# -------------------------------------------------------------------- invocation
def invoke(tool_use: Mapping[str, object],
           offered: set[str] | None = None,
           registry: Mapping[str, ToolSpec] | None = None,
           root: Path | None = None,
           effect_hops_used: int = 0,
           egress: Mapping[str, Channel] | None = None) -> Invocation:
    """Run one toolUse block and say, precisely, what happened.

    `offered` is the live offer set for remote tools. It is ignored for LOCAL
    tools, whose executor is in this repo: there is no server to drift.

    `registry` and `egress` exist to be *injected*, and not only for convenience:
    today REGISTRY declares no remote and no mutating tool, so four of the six
    outcomes and the effect refusal are unreachable from the real tables. A branch
    no test can reach is a branch that does not exist, so the reachability proof
    of T-L2 runs against declared-but-not-real tables — the same argument as the
    segmenter's injected rule set and S7's lying classifiers. (Named, not spelled:
    `tests/test_segment.py` scans production files for that keyword argument by
    substring, and a docstring quoting it is a finding in that census.)
    """
    reg = REGISTRY if registry is None else registry
    name = tool_use.get("name")
    if not isinstance(name, str) or name not in reg:
        return Invocation(str(name), UNKNOWN_TOOL,
                          f"unknown tool {name!r}: not in the declared registry")
    spec = reg[name]
    args = tool_use.get("input") or {}
    if not isinstance(args, dict):
        return Invocation(name, REFUSED, "tool input must be an object", spec.effect)

    if spec.origin == REMOTE:
        # Tied to the egress census on purpose. A remote tool call is a channel
        # out of this robot, and the table below says nothing enforces anything
        # on it. The refusal is the table's consequence rather than a second
        # written-down rule that could disagree with it.
        table = EGRESS if egress is None else egress
        state = table[MCP_TOOL_CALL].enforcement
        if state != ENFORCED:
            return Invocation(name, REFUSED,
                              f"remote tool {name!r} refused: the egress channel "
                              f"{MCP_TOOL_CALL!r} is {state!r}", spec.effect)
        if offered is None:
            return Invocation(name, UNREACHABLE,
                              f"no offer set for {spec.server!r}", spec.effect)
        if name not in offered:
            return Invocation(name, NOT_OFFERED,
                              f"{spec.server!r} no longer offers {name!r}", spec.effect)

    if spec.effect == MUTATES and effect_hops_used >= MAX_EFFECT_HOPS:
        return Invocation(name, REFUSED,
                          f"{name!r} mutates and the effect budget is "
                          f"{MAX_EFFECT_HOPS} (used {effect_hops_used})", spec.effect)

    fn = executors().get(name)
    if fn is None:
        # Not reachable through REGISTRY — a test asserts both sides agree — but a
        # caller may pass its own registry, and a declared tool with no executor
        # must not read as a policy refusal.
        return Invocation(name, ERRORED,
                          f"{name!r} is declared with no executor", spec.effect)
    try:
        return Invocation(name, OK, fn(args, root), spec.effect)
    except _loader.SkillPathError as exc:
        # The one place policy actually acts today: the path fence of T-I3. The
        # reason goes back to the model because it is a routing mistake far more
        # often than an attack, and a silent empty result makes it retry.
        return Invocation(name, REFUSED, f"refused: {exc}", spec.effect)
    except Exception as exc:                     # noqa: BLE001 — see docstring
        return Invocation(name, ERRORED, f"{type(exc).__name__}: {exc}", spec.effect)


# ------------------------------------------------------------------- prompt cost
def tool_config(registry: Mapping[str, ToolSpec] | None = None) -> dict:
    """The Bedrock Converse `toolConfig`, built here and nowhere else."""
    reg = REGISTRY if registry is None else registry
    return {"tools": [{"toolSpec": {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": {"json": dict(spec.input_schema)},
    }} for spec in reg.values()]}


def spec_tokens(registry: Mapping[str, ToolSpec] | None = None) -> int:
    """Approximate input tokens the schemas cost, on the serialised form."""
    return _loader.approx_tokens(
        json.dumps(tool_config(registry), ensure_ascii=False, sort_keys=True))


def spec_findings(registry: Mapping[str, ToolSpec] | None = None,
                  resident: bool = False) -> list[str]:
    """Findings, not an exception, so the first one does not hide the rest."""
    reg = REGISTRY if registry is None else registry
    findings = []
    cost = spec_tokens(reg)
    if cost > TOOL_SPEC_CEILING:
        findings.append(
            f"tool schemas cost ~{cost} tokens, over the {TOOL_SPEC_CEILING} ceiling "
            f"({len(reg)} tool(s))")
    if resident and any(s.origin == REMOTE for s in reg.values()):
        findings.append(
            "a resident catalogue charges every turn for tools most turns never "
            "use; agent_node attaches toolConfig only on turns a skill fired, and "
            "making it resident is a per-turn cost that needs its own measurement")
    return findings


# ------------------------------------------------------------------ egress census
SAY = "/speech/say"
GESTURE = "/body/gesture"
GAME_CMD = "/game/cmd"
INTERRUPT = "/speech/interrupt"
AUDIT = "/memory/audit"
SKILL_TOOL_CALL = "tool_call:read_skill_file"
MCP_TOOL_CALL = "tool_call:mcp"


@dataclass(frozen=True)
class Channel:
    name: str
    carries: str
    # False for a channel the node publishes on its own account. The census is
    # about what the *model's output* can reach, and folding the two together
    # would let a node-owned topic inherit a claim it never needed.
    #
    # This is a *declared* value, and for two of the six rows it is also
    # derivable — see REGISTRY_DERIVED_REACHABILITY. Declaring what can be
    # derived is how a row goes stale, so `reachability_findings()` checks the
    # two against each other rather than trusting whichever was edited last.
    model_reachable: bool
    enforcement: str
    enforcer: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.enforcement not in ENFORCEMENT:
            raise ToolError(f"{self.name}: unknown enforcement {self.enforcement!r}")
        if (self.enforcement == NO_ENFORCER_TODAY) == bool(self.enforcer):
            raise ToolError(
                f"{self.name}: {self.enforcement!r} and enforcer {self.enforcer!r} "
                f"disagree — naming one is what makes the state checkable")


EGRESS: dict[str, Channel] = {
    SAY: Channel(
        name=SAY, carries="words the robot says out loud", model_reachable=True,
        enforcement=NOT_WIRED, enforcer="safety.output_guard",
        reason=("S7 built the guard and nothing calls it: an AST census of the "
                "production files finds zero callers of safety/output_guard.py "
                "outside tests/. Wiring it is S7's runtime half, which needs the "
                "instance. Until then this channel is unguarded in fact, and the "
                "test that keeps this row honest goes red when the wiring lands."),
    ),
    GESTURE: Channel(
        name=GESTURE, carries="whole-body motion", model_reachable=True,
        enforcement=ENFORCED, enforcer="body.gestures.GESTURES",
        reason=("the tag is parsed after the fact against a closed whitelist and "
                "the `\\w+` capture cannot express a colon, so the model cannot "
                "reach chant's parameter form (T-D3)."),
    ),
    GAME_CMD: Channel(
        name=GAME_CMD, carries="game state transitions", model_reachable=True,
        enforcement=ENFORCED, enforcer="game.game_node",
        reason=("game_node owns its own cooldown and stop reflex, so a [game:start] "
                "from the model is a request the reflex layer may decline."),
    ),
    INTERRUPT: Channel(
        name=INTERRUPT, carries="silencing the robot", model_reachable=False,
        enforcement=NO_ENFORCER_TODAY, enforcer=None,
        reason=("published by on_asr on barge-in, never from model text; it carries "
                "no content and its worst case is silence, which is the safe end."),
    ),
    AUDIT: Channel(
        name=AUDIT, carries="the composed-turn record, on its way to the one writer",
        model_reachable=False,
        enforcement=ENFORCED, enforcer="memory.store.check_audit",
        reason=("R1.1 added it, and it is the one egress row whose enforcer is on the "
                "*receiving* side: agent_node holds no Store, so the record travels one "
                "topic hop and `memory_node.on_audit` appends it, where check_audit "
                "refuses any field the `chat` row does not declare. Not model-reachable "
                "— every field is derived from the Assembly (block names, a character "
                "count, findings this repo's own code wrote), and no model output "
                "reaches it. A record the robot could dictate would be an audit stream "
                "the audited party writes."),
    ),
    SKILL_TOOL_CALL: Channel(
        name=SKILL_TOOL_CALL, carries="a filesystem read under skills/",
        model_reachable=True,
        enforcement=ENFORCED, enforcer="skills.loader.read_skill_file",
        reason=("containment is checked on the resolved path after symlinks, and "
                "the refusal comes back as REFUSED rather than as an error (T-I3)."),
    ),
    MCP_TOOL_CALL: Channel(
        name=MCP_TOOL_CALL, carries="whatever a remote MCP server does",
        model_reachable=False,
        enforcement=NO_ENFORCER_TODAY, enforcer=None,
        reason=("no MCP client exists and REGISTRY declares no remote tool. "
                "`invoke` reads this row, so the day a remote tool is declared it "
                "is refused until something is named here — the gate and the "
                "census cannot drift apart because there is only one of them."),
    ),
}


# The two rows whose reachability is a consequence of REGISTRY rather than an
# opinion: a tool channel is reachable exactly when a tool of that origin is
# declared. Mapping channel -> the origin that makes it live.
REGISTRY_DERIVED_REACHABILITY = {
    SKILL_TOOL_CALL: LOCAL,
    MCP_TOOL_CALL: REMOTE,
}


def derived_reachability(registry: Mapping[str, ToolSpec] | None = None) -> dict[str, bool]:
    reg = REGISTRY if registry is None else registry
    return {channel: any(spec.origin == origin for spec in reg.values())
            for channel, origin in REGISTRY_DERIVED_REACHABILITY.items()}


def reachability_findings(egress: Mapping[str, Channel] | None = None,
                          registry: Mapping[str, ToolSpec] | None = None) -> list[str]:
    """Where a declared `model_reachable` disagrees with what REGISTRY implies.

    Kept separate from `egress_findings` on purpose: "this channel is unguarded"
    and "this row is out of date" are two claims, and a function returning both
    lets a stale row be read as a safety gap or the reverse. The load-bearing
    case is declaring the first remote tool — the day that happens
    `MCP_TOOL_CALL` becomes model-reachable, this goes red, and the row cannot be
    left saying otherwise while `invoke` refuses on its authority.
    """
    table = EGRESS if egress is None else egress
    derived = derived_reachability(registry)
    return [f"{channel}: the table declares model_reachable="
            f"{table[channel].model_reachable} and REGISTRY implies {actual} "
            f"({REGISTRY_DERIVED_REACHABILITY[channel]} tool(s) declared: {actual})"
            for channel, actual in derived.items()
            if channel in table and table[channel].model_reachable != actual]


def egress_findings(egress: Mapping[str, Channel] | None = None) -> list[str]:
    """Every model-reachable channel with no enforcer running on it.

    Reported, never raised. Two of these are true right now, and a version of
    this function that threw would have to be disabled to boot — which is how a
    known gap becomes an invisible one.
    """
    table = EGRESS if egress is None else egress
    return [f"{ch.name}: model-reachable and {ch.enforcement!r} "
            f"(enforcer {ch.enforcer!r}) — {ch.reason}"
            for ch in table.values()
            if ch.model_reachable and ch.enforcement != ENFORCED]
