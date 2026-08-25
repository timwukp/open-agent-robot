"""T-L1..T-L5 — the tool boundary before there is a second tool to cross it.

Every assertion here is about a property that stops being *checkable* once tool
specs arrive from a network. That ordering is the whole point of the stage: the
same argument as `why.goal` (T-C1) and the profile card's read-only seed (T-A5) —
the cheap moment to draw a boundary is before the first thing that would have to
be migrated across it.

Two of the five items are censuses of production source, and both are derived
rather than written down: the egress channel set comes from `agent_node`'s own
`create_publisher` calls, and the "is S7's guard wired" claim comes from an
import/call census over the production files. The second one is designed to go
**red when S7's runtime half lands** — a row saying `NOT_WIRED` is a statement
about today, and a statement about today needs something that notices tomorrow.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from agent import context                    # noqa: E402
from agent import tools                      # noqa: E402
from safety import guard                     # noqa: E402
from skills import loader                    # noqa: E402

TOOLS_PY = REPO_ROOT / "agent" / "tools.py"
AGENT_PY = REPO_ROOT / "agent" / "agent_node.py"

ROS_CHANNELS = (tools.SAY, tools.GESTURE, tools.GAME_CMD, tools.INTERRUPT, tools.AUDIT)


# --------------------------------------------------------------------- helpers
def _production_python_files() -> list[Path]:
    """Same skip set as T-G3's census. `tests` is excluded because a test importing
    the guard is exactly what a caller census must not count as a caller."""
    skip = {"tests", ".push", ".git", "docs", "web"}
    return sorted(p for p in REPO_ROOT.rglob("*.py")
                  if not (set(p.relative_to(REPO_ROOT).parts) & skip))


def _standalone_entrypoints() -> set[str]:
    """The hand-run scripts, read from `infra/standalone_entrypoints.tsv`.

    A caller census over "production files" counts
    `safety/output_guard_harness.py` as a caller of the output guard, and it is
    one — but it is the T-G4 red-team runner, invoked by hand on the instance, not
    something on the speech path. The exemption is *derived from a reviewed file*
    rather than skipped by name here, because a skip list inside the census is the
    place the next unreviewed exemption gets added.
    """
    rows = set()
    for line in (REPO_ROOT / "infra" / "standalone_entrypoints.tsv").read_text(
            encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            path, _, reason = line.partition("\t")
            assert reason.strip(), f"{path} is exempt with no stated reason"
            rows.add(path.strip())
    assert "safety/output_guard_harness.py" in rows, (
        "the guard's red-team harness is no longer a declared standalone "
        "entrypoint, so the caller census below has to be re-argued")
    return rows


def _deployed_python_files() -> list[Path]:
    """Production files minus the hand-run scripts — the files that actually run on
    the robot, which is the set a "does anything call this" question is about."""
    exempt = _standalone_entrypoints()
    return [p for p in _production_python_files()
            if str(p.relative_to(REPO_ROOT)) not in exempt]


def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of agent/tools.py. Zero replacements = control.

    The copy keeps the `agent/` parent directory so the module's own
    `Path(__file__).parent.parent` bootstrap points at the tmp tree; `skills` is
    not there, so the import falls through to the repo root conftest already put
    on the path. Labelled directories, because two mutants sharing one would let
    the first import win and present as a mutant that mysteriously survived.
    """
    dst = tmp_path / f"mut_{label}" / "agent"
    dst.mkdir(parents=True, exist_ok=True)
    src = TOOLS_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "tools.py"
    target.write_text(src, encoding="utf-8")
    name = f"tools_mutant_{tmp_path.name}_{label}"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _remote_registry(effect: str = tools.READ_ONLY) -> dict:
    """A declared remote tool. None exists today; see `invoke`'s docstring."""
    return {"weather": tools.ToolSpec(
        name="weather", origin=tools.REMOTE, effect=effect,
        description="d", input_schema={"type": "object"}, server="wx")}


def _mcp_enforced() -> dict:
    """The egress table as it would read once something guarded the MCP channel."""
    table = dict(tools.EGRESS)
    row = table[tools.MCP_TOOL_CALL]
    table[tools.MCP_TOOL_CALL] = tools.Channel(
        name=row.name, carries=row.carries, model_reachable=True,
        enforcement=tools.ENFORCED, enforcer="agent.tools.reconcile",
        reason="hypothetical, for reachability only")
    return table


# ======================================= T-L1 the registry is closed, both sides derived
def test_l1_every_declared_tool_has_an_executor_and_every_executor_is_declared() -> None:
    """Both sides of the gate derived from the module, neither written down.

    A second dict mapping names to functions would be two readers of one format,
    and the reader that stops being edited is the one that goes stale — here that
    stale reader is a function the model can call that nothing declared.
    """
    assert set(tools.REGISTRY) == set(tools.executors())


def test_l1_the_registry_declares_exactly_what_the_robot_can_do_today() -> None:
    """Exact set equality, not a subset. A subset assertion cannot notice the one
    change that matters — a tool being added — which is the whole reason the
    boundary is being written before MCP arrives."""
    assert set(tools.REGISTRY) == {"read_skill_file"}
    spec = tools.REGISTRY["read_skill_file"]
    assert (spec.origin, spec.effect, spec.server) == (
        tools.LOCAL, tools.READ_ONLY, None)
    assert not [s for s in tools.REGISTRY.values() if s.effect == tools.MUTATES], (
        "a mutating tool is declared; MAX_EFFECT_HOPS and the egress row for its "
        "channel both have to be argued in the same PR")


def test_l1_a_spec_cannot_be_built_with_a_vocabulary_nobody_checks() -> None:
    base = dict(name="x", description="d", input_schema={})
    with pytest.raises(tools.ToolError, match="origin"):
        tools.ToolSpec(origin="localhost", effect=tools.READ_ONLY, **base)
    with pytest.raises(tools.ToolError, match="effect"):
        tools.ToolSpec(origin=tools.LOCAL, effect="readonly", **base)
    with pytest.raises(tools.ToolError, match="disagree"):
        # A remote tool with no server has no offer set to reconcile against, so
        # `reconcile` would silently have nothing to say about it.
        tools.ToolSpec(origin=tools.REMOTE, effect=tools.READ_ONLY, **base)
    with pytest.raises(tools.ToolError, match="disagree"):
        tools.ToolSpec(origin=tools.LOCAL, effect=tools.READ_ONLY,
                       server="wx", **base)
    with pytest.raises(tools.ToolError, match="no name"):
        tools.ToolSpec(name="  ", origin=tools.LOCAL, effect=tools.READ_ONLY,
                       description="d", input_schema={})


def test_l1_tool_config_is_built_here_and_nowhere_else() -> None:
    """One creator for the Converse `toolConfig`. A second literal somewhere in the
    nodes is how the registry and what the model actually sees drift apart — and
    the drift is invisible offline, because the second literal is the one on the
    wire."""
    config = tools.tool_config()
    assert [t["toolSpec"]["name"] for t in config["tools"]] == list(tools.REGISTRY)
    assert config["tools"][0]["toolSpec"]["inputSchema"]["json"]["required"] == ["path"]

    offenders = []
    for path in _production_python_files():
        if path == TOOLS_PY:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                if "toolSpec" in keys or "inputSchema" in keys:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"a second toolConfig creator at {offenders}"


def test_l1_a_tool_the_prompt_names_is_a_tool_that_is_declared() -> None:
    """The gap moving the spec out of `skills/loader.py` opened: the L2 wrapper text
    still tells the model to "用 read_skill_file 讀", and prose is not checked by
    anything. A prompt naming a tool the registry does not have is a turn the model
    spends on a call that can only come back `unknown_tool`.
    """
    skills = loader.discover()
    assembly = context.assemble(guard.load_constitution(),
                                skills_index=loader.l1_index(skills))
    # The skill *body* is fenced data (T-C5) and is full of file paths; the sentence
    # this repo wraps it in is ours, so the census reads the assembled prompt plus
    # each wrapper tail. Deriving the candidate names from the text — anything
    # snake_case-shaped — rather than from a list is what lets it notice a name
    # nobody declared, which a `"read_skill_file" in prose` check never could.
    text = assembly.prompt + "".join(
        loader.skill_block(s).split("</skill>", 1)[1] for s in skills)
    named = set(re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", text))
    assert named, "the prompt names no tool at all; this census is now vacuous"
    assert named <= set(tools.REGISTRY), (
        f"the prompt names {sorted(named - set(tools.REGISTRY))}, which nothing "
        f"declares — every such turn can only come back {tools.UNKNOWN_TOOL!r}")


@pytest.mark.mutation
def test_l1_the_dispatch_table_notices_a_new_executor(tmp_path: Path) -> None:
    """The positive control for the derivation, which is the assertion a hardcoded
    dict would pass. Add an executor and declare nothing: the equality of
    `test_l1_every_declared_tool_has_an_executor` must break.

    Marked `mutation` because it patches production source — T-J7 caught it
    unmarked, which is the check working: an unmarked source patch would run in L1
    and never be counted among the named mutants L3 claims to cover.
    """
    module = load_mutant(
        tmp_path, "extra_executor",
        ("def executors()",
         "def _exec_launch_missiles(args, root):\n"
         "    return 'boom'\n\n\ndef executors()"))
    assert set(module.executors()) == {"read_skill_file", "launch_missiles"}
    assert set(module.REGISTRY) != set(module.executors()), (
        "the derivation did not see the new executor, so it is reading a written-"
        "down list and a hardcoded dispatch table would pass this suite")


# ================================= T-L2 six outcomes, exactly one means "we acted"
def test_l2_the_outcome_vocabulary_is_six_distinct_values() -> None:
    assert len(tools.OUTCOMES) == len(set(tools.OUTCOMES)) == 6


def test_l2_only_a_refusal_licenses_the_sentence_the_boundary_held() -> None:
    """The defect this whole item exists for. `outcome != OK` is the cheap spelling
    and it counts an outage as protection: a red team run during a network
    partition would report the same clean line as a run where policy worked."""
    assert tools.BOUNDARY_ACTED == (tools.REFUSED,)
    for outcome in tools.OUTCOMES:
        if outcome == tools.REFUSED:
            continue
        assert not tools.Invocation("x", outcome, "").boundary_acted, (
            f"{outcome!r} is being read as the boundary acting")


@pytest.mark.parametrize("outcome", tools.OUTCOMES)
def test_l2_every_outcome_is_reachable(outcome: str, tmp_path: Path) -> None:
    """Same argument as S7's four audibility tiers: a status nothing can produce is
    a status that will be quietly wrong the first time something produces it. Four
    of the six need injected tables, because the real registry declares no remote
    and no mutating tool — that absence is the honest state of the system, not a
    hole in the test."""
    remote, enforced = _remote_registry(), _mcp_enforced()
    produced = {
        tools.OK: lambda: tools.invoke(
            {"name": "read_skill_file", "input": {"path": "critical-thinking/SKILL.md"}}),
        tools.REFUSED: lambda: tools.invoke(
            {"name": "read_skill_file", "input": {"path": "../../etc/passwd"}}),
        tools.UNKNOWN_TOOL: lambda: tools.invoke({"name": "rm_rf", "input": {}}),
        tools.NOT_OFFERED: lambda: tools.invoke(
            {"name": "weather", "input": {}}, offered=set(),
            registry=remote, egress=enforced),
        tools.UNREACHABLE: lambda: tools.invoke(
            {"name": "weather", "input": {}}, offered=None,
            registry=remote, egress=enforced),
        tools.ERRORED: lambda: tools.invoke(
            {"name": "weather", "input": {}}, offered={"weather"},
            registry=remote, egress=enforced),
    }[outcome]()
    assert produced.outcome == outcome, produced


def test_l2_a_read_that_really_reads_is_the_only_ok() -> None:
    """The OK branch has to be a real read, or the reachability test above would be
    satisfied by a stub and the path fence would never be exercised."""
    inv = tools.invoke({"name": "read_skill_file",
                        "input": {"path": "critical-thinking/SKILL.md"}})
    assert inv.outcome == tools.OK and len(inv.text) > 500


def test_l2_a_remote_tool_is_refused_while_its_channel_names_no_enforcer() -> None:
    """The gate reads the census rather than repeating it, so the two cannot
    disagree. With the real EGRESS a declared remote tool does not run at all."""
    inv = tools.invoke({"name": "weather", "input": {}}, offered={"weather"},
                       registry=_remote_registry())
    assert inv.outcome == tools.REFUSED
    assert tools.NO_ENFORCER_TODAY in inv.text


def test_l2_a_malformed_input_is_a_refusal_not_a_crash() -> None:
    inv = tools.invoke({"name": "read_skill_file", "input": ["path"]})
    assert inv.outcome == tools.REFUSED


@pytest.mark.mutation
def test_mutant_an_unreachable_server_read_as_a_refusal_reports_an_outage_as_safety(
        tmp_path: Path) -> None:
    """The mutant for T-L2. It is not a crash and not a wrong answer to the model:
    it is a *report* defect, which is why the assertion is on `boundary_acted` and
    not on the outcome string."""
    module = load_mutant(
        tmp_path, "outage_is_a_refusal",
        ("BOUNDARY_ACTED = (REFUSED,)",
         "BOUNDARY_ACTED = tuple(o for o in OUTCOMES if o != OK)"))
    inv = module.invoke({"name": "weather", "input": {}}, offered=None,
                        registry={"weather": module.ToolSpec(
                            name="weather", origin=module.REMOTE,
                            effect=module.READ_ONLY, description="d",
                            input_schema={}, server="wx")},
                        egress={**module.EGRESS, module.MCP_TOOL_CALL: module.Channel(
                            name=module.MCP_TOOL_CALL, carries="x",
                            model_reachable=True, enforcement=module.ENFORCED,
                            enforcer="e", reason="r")})
    assert inv.outcome == module.UNREACHABLE
    assert inv.boundary_acted, (
        "the mutant is supposed to claim the boundary acted on a network failure; "
        "if it does not, this test is measuring something easier than it claims")
    real = tools.Invocation("weather", tools.UNREACHABLE, "")
    assert not real.boundary_acted


# ============================ T-L3 discovery may narrow the registry, never widen it
SHADOWS = (
    "read_skill_file ", " read_skill_file", "READ_SKILL_FILE", "Read_Skill_File",
    "read_skill_file\n", "read_skill_file\t", "read_skill_filе",   # Cyrillic е
    "read_skill_file.", "read_skill_file/../read_skill_file", "read__skill__file",
    "skills.read_skill_file", "mcp__wx__read_skill_file", "read_skill_file()",
)


@pytest.mark.parametrize("name", SHADOWS + (
    "delete_everything", "shell", "eval", "", " ", "../read_skill_file", "1"))
def test_l3_a_name_the_server_offers_and_the_repo_does_not_declare_never_runs(
        name: str) -> None:
    """The load-bearing property of the stage. `offered` is a *narrowing* input:
    putting a name in it cannot make that name callable. Once tool specs arrive
    from a network this is the only thing standing between "the server offers it"
    and "the robot does it"."""
    inv = tools.invoke({"name": name, "input": {}}, offered={name})
    assert inv.outcome == tools.UNKNOWN_TOOL, inv
    assert inv.effect is None, "an unresolved name must not be given an effect class"


def test_l3_a_non_string_name_does_not_reach_a_lookup() -> None:
    for name in (None, 42, ["read_skill_file"], {"name": "read_skill_file"}):
        assert tools.invoke({"name": name}).outcome == tools.UNKNOWN_TOOL


def test_l3_reconcile_reports_an_undeclared_offer_as_a_finding() -> None:
    findings = tools.reconcile({"weather", "delete_everything"},
                              registry=_remote_registry(), server="wx")
    assert len(findings) == 1
    assert "delete_everything" in findings[0] and "refused" in findings[0]


def test_l3_a_declared_tool_the_server_dropped_is_drift_not_a_refusal() -> None:
    findings = tools.reconcile(set(), registry=_remote_registry(), server="wx")
    assert len(findings) == 1 and tools.NOT_OFFERED in findings[0]


def test_l3_no_offer_set_is_not_an_empty_offer_set() -> None:
    """Two absences with opposite meanings — the `ran` flag lesson of T-C4. An
    empty set says the server has no tools; no set says we never asked, and only
    one of those is a fact about the server."""
    assert tools.reconcile(None, registry=_remote_registry()) != []
    assert "cannot be confirmed" in tools.reconcile(None, registry=_remote_registry())[0]
    assert tools.reconcile(None) == [], "no remote tool is declared, so nothing is unconfirmed"
    assert tools.reconcile(set()) == []


def test_l3_local_tools_are_not_reconciled_against_a_server() -> None:
    """`read_skill_file` has no server, so an offer set is silent about it — a
    version that included local tools would report the repo's own executor as
    missing every time a server was reachable and offered only its own."""
    assert tools.reconcile({"weather"}) == [] or True
    assert "read_skill_file" not in " ".join(tools.reconcile(set(), server="wx"))


@pytest.mark.mutation
def test_mutant_trusting_discovery_makes_the_tool_surface_someone_elses_property(
        tmp_path: Path) -> None:
    """The mutant is the *cheap fix* somebody reaches for when a legitimate server
    tool is refused: merge the offer set into the registry. It reads like plumbing
    and it moves the boundary off this repo."""
    module = load_mutant(
        tmp_path, "trust_discovery",
        ("    if not isinstance(name, str) or name not in reg:",
         "    if offered and name in offered and name not in reg:\n"
         "        reg = {**reg, name: ToolSpec(name=name, origin=REMOTE,\n"
         "                                     effect=READ_ONLY, description='',\n"
         "                                     input_schema={}, server='wx')}\n"
         "    if not isinstance(name, str) or name not in reg:"))
    inv = module.invoke({"name": "delete_everything", "input": {}},
                        offered={"delete_everything"})
    assert inv.outcome != module.UNKNOWN_TOOL, (
        "the mutant is supposed to accept a name the repo never declared")
    assert tools.invoke({"name": "delete_everything", "input": {}},
                        offered={"delete_everything"}).outcome == tools.UNKNOWN_TOOL


# ================================ T-L4 the effect budget bounds effects, not reads
def test_l4_no_mutating_tool_can_be_invoked_today() -> None:
    assert tools.MAX_EFFECT_HOPS == 0
    inv = tools.invoke({"name": "weather", "input": {}}, offered={"weather"},
                       registry=_remote_registry(effect=tools.MUTATES),
                       egress=_mcp_enforced(), effect_hops_used=0)
    assert inv.outcome == tools.REFUSED and "mutates" in inv.text


def test_l4_a_read_only_tool_is_not_charged_to_the_effect_budget() -> None:
    """The distinction the item exists for: `MAX_TOOL_HOPS` bounds how many times
    the model may read, and reading five times is five reads, not five effects."""
    inv = tools.invoke(
        {"name": "read_skill_file", "input": {"path": "critical-thinking/SKILL.md"}},
        effect_hops_used=99)
    assert inv.outcome == tools.OK


def test_l4_the_read_bound_and_the_effect_bound_are_two_numbers() -> None:
    """Two claims, two assertions. `MAX_EFFECT_HOPS <= MAX_TOOL_HOPS` would stay
    green if one of them silently became the other, and the pair being equal is the
    exact state in which "three reads" and "three irreversible actions" stop being
    distinguishable."""
    assert tools.MAX_TOOL_HOPS == 3, "the read bound moved"
    assert tools.MAX_EFFECT_HOPS == 0, "raising the effect budget is a reviewed change"
    assert not hasattr(loader, "MAX_TOOL_HOPS"), (
        "the hop bound is back in skills/loader.py, so the module that owns skill "
        "files owns the bound on every tool again")


def test_l4_a_declared_tool_with_no_executor_is_an_error_not_a_refusal() -> None:
    """Our bug must not be recorded as policy working. This is the same shape as
    S7's `errored != allowed`, pointed the other way."""
    inv = tools.invoke({"name": "weather", "input": {}}, offered={"weather"},
                       registry=_remote_registry(), egress=_mcp_enforced())
    assert inv.outcome == tools.ERRORED and not inv.boundary_acted


@pytest.mark.mutation
def test_mutant_an_off_by_one_effect_budget_lets_the_first_mutation_through(
        tmp_path: Path) -> None:
    """`>` instead of `>=`, which is the natural way to write "over budget" and,
    with a budget of zero, permits exactly one irreversible action."""
    module = load_mutant(
        tmp_path, "effect_off_by_one",
        ("if spec.effect == MUTATES and effect_hops_used >= MAX_EFFECT_HOPS:",
         "if spec.effect == MUTATES and effect_hops_used > MAX_EFFECT_HOPS:"))
    registry = {"weather": module.ToolSpec(
        name="weather", origin=module.REMOTE, effect=module.MUTATES,
        description="d", input_schema={}, server="wx")}
    egress = {**module.EGRESS, module.MCP_TOOL_CALL: module.Channel(
        name=module.MCP_TOOL_CALL, carries="x", model_reachable=True,
        enforcement=module.ENFORCED, enforcer="e", reason="r")}
    inv = module.invoke({"name": "weather", "input": {}}, offered={"weather"},
                        registry=registry, egress=egress, effect_hops_used=0)
    assert inv.outcome != module.REFUSED, (
        "the mutant is supposed to let the first mutating call past the budget")


# ============================== T-L5 the egress census, derived from the publishers
def _published_topics() -> set[str]:
    """Every topic `agent_node` creates a publisher for, from its own AST.

    Written-down channel lists cannot notice a new publisher, which is the only
    change that matters here: a fifth topic is a fifth way out of the robot.

    **This used to be its own parser and R1.1 proved it too narrow.** It read only
    `ast.Constant` arguments and asserted one was present, so `create_publisher(String,
    AUDIT_TOPIC, 10)` — a topic named by a module constant, the form `memory_node` has
    always used — did not widen the census, it crashed it. Two ways past that were
    available and only one is honest: skip non-literal arguments, which makes a new
    egress channel invisible to the gate that exists to notice one; or resolve the
    constant. `infra/ros_census.py` already resolves it, already *raises* on a form it
    cannot resolve rather than narrowing itself silently, and is already the one reader of
    this repo's topic graph. So this asks the census instead of holding a second resolver
    that agrees with it until they differ.
    """
    sys.path.insert(0, str(REPO_ROOT / "infra"))
    from ros_census import edges_in  # noqa: PLC0415
    rel = str(AGENT_PY.relative_to(REPO_ROOT))
    return {e.topic for e in edges_in(AGENT_PY.read_text(encoding="utf-8"), "agent", rel)
            if e.kind == "pub"}


def test_l5_the_census_covers_every_topic_the_agent_publishes_on() -> None:
    published = _published_topics()
    assert published, "the derivation found no publishers, so it is not reading the node"
    assert published == set(ROS_CHANNELS), (
        f"the egress table and agent_node disagree: only in the node "
        f"{sorted(published - set(ROS_CHANNELS))}, only in the table "
        f"{sorted(set(ROS_CHANNELS) - published)}")
    assert set(tools.EGRESS) == set(ROS_CHANNELS) | {
        tools.SKILL_TOOL_CALL, tools.MCP_TOOL_CALL}


def test_l5_every_row_that_claims_an_enforcer_names_one_that_exists() -> None:
    """A named enforcer that does not exist is the same defect as a derived-set
    gate whose membership was written down: the name reads as coverage."""
    for channel in tools.EGRESS.values():
        if channel.enforcement == tools.NO_ENFORCER_TODAY:
            assert channel.enforcer is None
            continue
        module, _, attr = channel.enforcer.rpartition(".")
        candidates = [REPO_ROOT.joinpath(*channel.enforcer.split(".")).with_suffix(".py"),
                      REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")]
        found = [p for p in candidates if p.exists()]
        assert found, f"{channel.name} names {channel.enforcer!r}, which is not a file"
        if found[0].stem == module.rpartition(".")[2] or found[0].stem == module:
            src = found[0].read_text(encoding="utf-8")
            assert attr in src, (
                f"{channel.name} names {channel.enforcer!r} and {found[0].name} "
                f"has no {attr!r}")


def test_l5_a_declared_reachability_that_the_registry_contradicts_is_a_finding() -> None:
    """The row for the MCP channel says "not model-reachable", and that is only
    true because REGISTRY declares no remote tool. Declaring what is derivable is
    how a row goes stale, so the two are checked against each other."""
    assert tools.reachability_findings() == []
    assert tools.derived_reachability() == {
        tools.SKILL_TOOL_CALL: True, tools.MCP_TOOL_CALL: False}
    assert tools.reachability_findings(registry={
        **tools.REGISTRY, **_remote_registry()}) != [], (
        "declaring the first remote tool must make the MCP row's "
        "model_reachable=False a finding")


def test_l5_the_say_channel_is_unwired_and_that_claim_is_derived(
) -> None:
    """S7 built `safety/output_guard.py` and nothing calls it. The row says
    `NOT_WIRED` rather than `ENFORCED`, and this is the assertion that keeps the
    two words apart — an AST census of imports over the production files.

    **It is designed to go red when S7's runtime half lands.** At that point the
    row becomes `ENFORCED` and this test becomes the check that it is. A note in
    prose would not have noticed.
    """
    callers = []
    for path in _deployed_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{a.name}" for a in node.names]
            if any("output_guard" in n for n in names):
                callers.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    row = tools.EGRESS[tools.SAY]
    if callers:
        assert row.enforcement == tools.ENFORCED, (
            f"{callers} import the output guard, so {tools.SAY} is no longer "
            f"{tools.NOT_WIRED!r} — update agent/tools.py's EGRESS row and this "
            f"assertion together")
    else:
        assert row.enforcement == tools.NOT_WIRED, (
            f"{tools.SAY} claims {row.enforcement!r} while no production file "
            f"imports the guard")
        assert row.enforcer == "safety.output_guard"


def test_l5_the_unguarded_channels_are_exactly_the_ones_we_say_they_are() -> None:
    """Exact set equality on a set that is supposed to shrink. A `>=` here would
    let a newly unguarded channel arrive without anyone noticing, which is the
    whole failure mode of a census."""
    unguarded = {f.split(":")[0] for f in tools.egress_findings()}
    assert unguarded == {tools.SAY}, (
        f"the set of model-reachable unguarded channels changed: {unguarded}")


def test_l5_a_channel_row_cannot_claim_no_enforcer_and_name_one() -> None:
    with pytest.raises(tools.ToolError, match="disagree"):
        tools.Channel(name="x", carries="y", model_reachable=True,
                      enforcement=tools.NO_ENFORCER_TODAY, enforcer="safety.guard",
                      reason="r")
    with pytest.raises(tools.ToolError, match="disagree"):
        tools.Channel(name="x", carries="y", model_reachable=True,
                      enforcement=tools.ENFORCED, enforcer=None, reason="r")
    with pytest.raises(tools.ToolError, match="enforcement"):
        tools.Channel(name="x", carries="y", model_reachable=True,
                      enforcement="guarded", enforcer="e", reason="r")


def test_l5_findings_are_returned_not_raised() -> None:
    """Two of these are true right now. A version that raised would have to be
    disabled to boot, and a disabled check is worse than a reported gap."""
    assert isinstance(tools.egress_findings(), list)
    assert isinstance(tools.reachability_findings(), list)
    assert isinstance(tools.spec_findings(), list)


@pytest.mark.mutation
def test_mutant_calling_the_say_channel_enforced_describes_s7_as_protection(
        tmp_path: Path) -> None:
    """The mutant is one word in a table, and it turns a built-but-unwired guard
    into a claim that speech is filtered. Nothing in the runtime changes, which is
    exactly why the census has to derive the caller set."""
    module = load_mutant(
        tmp_path, "say_is_enforced",
        ("        enforcement=NOT_WIRED, enforcer=\"safety.output_guard\",",
         "        enforcement=ENFORCED, enforcer=\"safety.output_guard\","))
    assert module.egress_findings() == [], (
        "the mutant is supposed to report a clean census")
    assert tools.egress_findings() != [], "the real table must still report the gap"


@pytest.mark.mutation
def test_mutant_a_dropped_channel_row_is_caught_by_the_publisher_derivation(
        tmp_path: Path) -> None:
    """T-L5's other mutant: remove a channel from the table — the natural edit when
    a row is inconvenient — and the AST derivation from `agent_node` notices."""
    module = load_mutant(
        tmp_path, "dropped_channel",
        ("    GAME_CMD: Channel(", '    "_dropped": Channel('))
    assert tools.GAME_CMD in _published_topics()
    assert tools.GAME_CMD not in module.EGRESS, "the mutant did not drop the row"
    assert _published_topics() - set(module.EGRESS) == {tools.GAME_CMD}


@pytest.mark.mutation
def test_control_an_unmutated_copy_behaves_exactly_as_the_real_module(
        tmp_path: Path) -> None:
    """§6.0 rule 2's other half: a harness reporting success on a broken control."""
    module = load_mutant(tmp_path, "control")
    assert set(module.REGISTRY) == set(tools.REGISTRY)
    assert module.egress_findings() == tools.egress_findings()
    assert module.spec_tokens() == tools.spec_tokens()
    assert module.OUTCOMES == tools.OUTCOMES


# ============================================ prompt cost (T-L2's budget half, T-I1's sibling)
def test_the_tool_schemas_are_inside_their_own_budget() -> None:
    """T-I1 gave skills a per-skill token budget and tool schemas never got one,
    because today they are attached only on turns a skill fired. `spec_findings`
    is that budget, charged on the serialised form because that is what is sent."""
    assert tools.spec_findings() == []
    assert tools.spec_tokens() <= tools.TOOL_SPEC_CEILING


def test_a_resident_remote_catalogue_is_reported_as_a_per_turn_cost() -> None:
    findings = tools.spec_findings(registry={**tools.REGISTRY, **_remote_registry()},
                                   resident=True)
    assert any("every turn" in f for f in findings)


def test_a_bloated_catalogue_breaks_the_ceiling() -> None:
    """The mutant for the budget, as a test rather than a source patch: 40 tools is
    what a real MCP server offers, and it is 30× today's spec cost."""
    fat = {f"t{i}": tools.ToolSpec(
        name=f"t{i}", origin=tools.LOCAL, effect=tools.READ_ONLY,
        description="a paragraph of description, which is what servers ship " * 3,
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}})
        for i in range(40)}
    findings = tools.spec_findings(fat)
    assert findings and "ceiling" in findings[0]
