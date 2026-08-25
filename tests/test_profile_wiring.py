"""R1.1 — the profile card reaching the prompt, and the composed turn reaching the log.

Two seams were wired here and both had been correct-and-offline for two stages:
`agent/context.py::assemble(card=)` ran on every turn with its card argument at the
default, and `agent/context.py::audit_payload()` had no caller at all, so a composed
turn left no record anywhere. The first of this project's five promises — "remembers
you" — was a parameter nobody passed.

**Every assertion about the two node files is made against their source, not against an
imported module.** Both import `rclpy`, and this suite's first constraint (§6.1) is that
it runs on a laptop with no ROS. That is a real limit and it is stated rather than
hidden: these tests prove the *shape* of the wiring — that the subscription is declared,
with matching durability, that the card is forwarded, that the prompt is rebuilt — and
they do **not** prove a message flows. What proves that is `infra/run_ec2.sh status` plus
the on-machine walk-through in docs/test-design.md §6.4, which needs the security group.

**What is checked behaviourally, and why those checks matter more than they look.**
`memory/store.py` and `agent/context.py` import no ROS, so the record the node builds can
be built here and handed to the real `check_audit`. That closes the finding this round
started from: `audit_payload`'s docstring named `guard.audit_record` as its consumer for
two stages, and the single writer **refuses that composition in both directions**. A
docstring naming an impossible consumer is worse than no docstring, because the next
person wires it and reads the refusal as a bug in the store.

**Why the QoS keywords are compared and not just read.** memory_node publishes the card
once, at boot, with no republish timer — deliberately, so that a subscriber which forgot
`transient_local` fails immediately instead of appearing to work on the second tick. That
design decision only pays off if something checks the subscriber's side, because the
symptom of getting it wrong is a robot whose log says the card was published and whose
every prompt is card-less. So the two `PROFILE_QOS` assignments are compared keyword by
keyword, in both files, by AST.

Every derivation below is a plain function of a parsed tree, so the mutation section
re-runs the identical logic against a textually mutated source. A derived property that
no mutant can break is decoration.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_PY = REPO_ROOT / "agent" / "agent_node.py"
MEMORY_PY = REPO_ROOT / "memory" / "memory_node.py"

sys.path.insert(0, str(REPO_ROOT))

from agent import context as cx  # noqa: E402
from memory import profile as pf  # noqa: E402
from memory import store as st  # noqa: E402
from safety import guard  # noqa: E402

AGENT_SRC = AGENT_PY.read_text(encoding="utf-8")
MEMORY_SRC = MEMORY_PY.read_text(encoding="utf-8")

# The audit kind for a composed turn. Read off `store.AUDIT_FIELDS` rather than typed,
# so this file cannot disagree with the authority about which row governs the record.
CHAT_KIND = "chat"


# --------------------------------------------------------------------- parsing
def parse(src: str, cls_name: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    tree = ast.parse(src)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    assert cls is not None, f"no {cls_name} class in the source"
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods, (f"parsed {cls_name} but found no methods — this parser no longer "
                     "matches the file it reads, and every assertion would pass")
    return tree, methods


AGENT_TREE, AGENT = parse(AGENT_SRC, "AgentNode")
MEMORY_TREE, MEMORY = parse(MEMORY_SRC, "MemoryNode")


def constants(tree: ast.Module) -> dict[str, str]:
    """Module-level `NAME = "literal"` assignments, so a topic given as a name resolves.

    The same resolution `infra/ros_census.py` performs, and needed for the same reason:
    both nodes name their topics through constants, and a checker that only understood
    literals would report the graph as empty and pass.
    """
    out: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) \
                and isinstance(stmt.value.value, str):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = stmt.value.value
    return out


def topic_edges(tree: ast.Module, methods: dict[str, ast.FunctionDef],
                where: str = "__init__") -> tuple[dict[str, str], dict[str, str]]:
    """(subscriptions, publishers): topic name -> the QoS argument as source text.

    Topics are resolved through `constants`, so `PROFILE_TOPIC` reads as the topic it
    names. The QoS argument is kept as *text* because a `QoSProfile` cannot be built
    without rclpy, and the text is what has to agree between the two files anyway.
    """
    consts = constants(tree)
    subs, pubs = {}, {}

    def topic_of(arg: ast.expr) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.Name):
            return consts.get(arg.id)
        return None

    for node in ast.walk(methods[where]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("create_subscription", "create_publisher"):
            continue
        args = node.args
        if len(args) < 3:
            continue
        name = topic_of(args[1])
        if name is None:
            continue
        target = subs if node.func.attr == "create_subscription" else pubs
        target[name] = ast.unparse(args[3] if node.func.attr == "create_subscription"
                                   else args[2])
    return subs, pubs


def qos_keywords(tree: ast.Module, name: str) -> dict[str, str]:
    """The keyword arguments of a module-level `name = QoSProfile(...)`, as source text."""
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name
                        for t in stmt.targets)):
            continue
        assert isinstance(stmt.value, ast.Call), ast.dump(stmt.value)
        return {kw.arg: ast.unparse(kw.value) for kw in stmt.value.keywords
                if kw.arg is not None}
    raise AssertionError(f"no module-level {name} = QoSProfile(...) assignment")


def call_keywords(fn: ast.AST, attr: str) -> set[str]:
    """The keyword names passed at every call to something ending in `.attr`."""
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == attr:
            out |= {kw.arg for kw in node.keywords if kw.arg is not None}
    return out


def callers_of(methods: dict[str, ast.FunctionDef], attr: str) -> set[str]:
    """Which methods call `self.<attr>` (or anything `.attr`) at least once."""
    out = set()
    for name, fn in methods.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == attr:
                out.add(name)
    return out


def audit_kinds(methods: dict[str, ast.FunctionDef]) -> set[str]:
    """Every `"kind": <literal>` the node puts in an `audit_payload` extra dict."""
    out: set[str] = set()
    for fn in methods.values():
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "audit_payload"):
                continue
            for arg in node.args + [kw.value for kw in node.keywords]:
                if not isinstance(arg, ast.Dict):
                    continue
                for key, value in zip(arg.keys, arg.values):
                    if isinstance(key, ast.Constant) and key.value == "kind" \
                            and isinstance(value, ast.Constant):
                        out.add(value.value)
    return out


def adopts_inside_try(methods: dict[str, ast.FunctionDef], where: str,
                      attr: str) -> bool:
    """Is `self.<attr>` assigned before its validation has finished?

    The same derivation `tests/test_game_wiring.py` makes about `on_rps_table`, and the
    same failure it guards: a value assigned inside the `try` is adopted by a robot even
    when the parse that was supposed to vet it raises halfway through.
    """
    tries = [n for n in methods[where].body if isinstance(n, ast.Try)]
    assert len(tries) == 1, f"{where} should validate in exactly one try block"
    return any(isinstance(t, ast.Attribute) and t.attr == attr
               for n in ast.walk(tries[0]) if isinstance(n, ast.Assign)
               for t in n.targets)


def cardless_turn_audits(methods: dict[str, ast.FunctionDef]) -> bool:
    """Does a turn that finds no card reach `_audit` from inside the None branch?

    Composed of the branch *and* the call rather than either alone. Checking only that
    `_converse` mentions `self.card` would pass on a mutant that read the card and did
    nothing; checking only that `_audit` is called from `_converse` would pass on a
    mutant that audited every turn forever, which is a per-turn cost with no reader.
    """
    for node in ast.walk(methods["_converse"]):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Attribute) and test.left.attr == "card"
                and any(isinstance(op, ast.Is) for op in test.ops)
                and any(isinstance(c, ast.Constant) and c.value is None
                        for c in test.comparators)):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "_audit" for n in ast.walk(node)):
            return True
    return False


def stream_writes(tree: ast.Module) -> set[str]:
    """Every `X.append("<stream>", ...)` in the file, by stream name.

    Keyed on a first argument that is one of `store.STREAMS`, which is what makes this
    derivation able to tell a store write from `self.history.append({...})` — the node
    does the second one on every turn, and a check that counted `append` by name would
    report the brain as a writer of memory and then be relaxed until it stopped.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "append" and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value in st.STREAMS:
            out.add(node.args[0].value)
    return out


def stmt_index(fn: ast.FunctionDef, predicate) -> int:
    """Index of the first top-level statement of `fn` matching `predicate`."""
    for i, stmt in enumerate(fn.body):
        if any(predicate(n) for n in ast.walk(stmt)):
            return i
    return -1


def audit_publisher_before_first_compose(methods: dict[str, ast.FunctionDef]) -> bool:
    """Is `pub_audit` created before the boot composition that already audits?

    A publisher made afterwards drops the boot record on the floor while every log line
    still says the prompt was composed — a missing audit record that looks like a healthy
    start, which is the failure mode an audit log exists to remove.
    """
    init = methods["__init__"]
    pub = stmt_index(init, lambda n: isinstance(n, ast.Attribute)
                     and n.attr == "pub_audit")
    compose = stmt_index(init, lambda n: isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "_compose")
    assert pub >= 0 and compose >= 0, (pub, compose)
    return pub < compose


# ------------------------------------------------------------- the two topic names
def test_the_two_topic_names_are_pinned_copies_of_memory_nodes() -> None:
    """agent_node holds copies, and an unpinned copy drifts.

    Importing the names would be one creator instead of two — and would also make the
    brain import a second ROS node module, which is what would make
    `infra/wiring_census.py` report memory_node as a file the agent reaches. Copies are
    acceptable in this tree (`memory_node.PROXY_RATIO` is the precedent); unpinned copies
    are not, so the two assignments are compared by AST.
    """
    agent, memory = constants(AGENT_TREE), constants(MEMORY_TREE)
    assert agent["PROFILE_TOPIC"] == memory["TOPIC"], (agent.get("PROFILE_TOPIC"),
                                                       memory.get("TOPIC"))
    assert agent["AUDIT_TOPIC"] == memory["AUDIT_TOPIC"], (agent.get("AUDIT_TOPIC"),
                                                           memory.get("AUDIT_TOPIC"))


def test_the_agent_subscribes_the_card_with_the_publishers_durability() -> None:
    """Matching durability, keyword by keyword. A volatile subscriber gets nothing."""
    subs, _pubs = topic_edges(AGENT_TREE, AGENT)
    topic = constants(AGENT_TREE)["PROFILE_TOPIC"]
    assert topic in subs, sorted(subs)
    assert subs[topic] == "PROFILE_QOS", subs[topic]
    mine = qos_keywords(AGENT_TREE, "PROFILE_QOS")
    theirs = qos_keywords(MEMORY_TREE, "PROFILE_QOS")
    assert mine == theirs, (mine, theirs)
    assert "TRANSIENT_LOCAL" in mine["durability"], mine


def test_the_agent_never_publishes_the_card_topic() -> None:
    """One publisher of a latched topic. Two is two cards, and which one a late joiner
    receives depends on subscribe timing — the claim `tests/test_profile.py` derives
    tree-wide from the census; here it is the consumer's own half."""
    _subs, pubs = topic_edges(AGENT_TREE, AGENT)
    assert constants(AGENT_TREE)["PROFILE_TOPIC"] not in pubs, sorted(pubs)


# ------------------------------------------------------- the card reaches the prompt
def test_the_card_is_forwarded_into_assemble() -> None:
    """The seam this PR exists to close: `assemble(card=…)` had no caller passing it."""
    assert "card" in call_keywords(AGENT["_compose"], "assemble"), \
        "the card argument is not passed, so the prompt is card-less by construction"


def test_the_prompt_is_recomposed_when_the_card_arrives() -> None:
    """Boot and arrival, both. `assemble` is pure and the card is asynchronous, so a
    single composition in `__init__` is a prompt permanently missing the profile."""
    callers = callers_of(AGENT, "_compose")
    assert {"__init__", "on_profile"} <= callers, callers


def test_a_rejected_card_is_not_adopted() -> None:
    """Validation finishes before the robot answers from the facts."""
    assert not adopts_inside_try(AGENT, "on_profile", "card")


def test_a_turn_that_beats_the_card_leaves_a_record() -> None:
    """The design decision, on the side that is visible afterwards.

    The robot answers rather than waiting on a latched topic, so the degradation cannot
    be observed in what it says. It is observable in the audit stream or nowhere.
    """
    assert cardless_turn_audits(AGENT)


def test_passing_the_card_actually_changes_the_prompt(tmp_path) -> None:
    """The AST says the argument is passed; this says passing it matters.

    Both halves are needed. A keyword forwarded into a function that ignored it would
    satisfy every structural check in this file, and the promise would still be absent
    from the running robot — which is the exact shape of the defect being closed.
    """
    con = guard.load_constitution()
    # `trust` is read off the module, not typed: `memory/profile.py` owns the closed
    # vocabulary, and a literal here would be a second place that decides what a card
    # may say — which is how the fixture and the parser drift.
    card = pf.parse({"version": 1, "facts": [
        {"kind": "allergy", "value": "peanuts", "trust": pf.TRUST_SEED,
         "source_episode_ids": []}]}, where=str(tmp_path))
    without = cx.assemble(con)
    with_card = cx.assemble(con, card=card)
    assert cx.CARD_BLOCK not in without.blocks, without.blocks
    assert cx.CARD_BLOCK in with_card.blocks, with_card.blocks
    assert "peanuts" in with_card.prompt
    assert "peanuts" not in without.prompt


# ------------------------------------------------- the record reaches the one writer
def test_the_brain_writes_no_stream_itself() -> None:
    """Single writer. The record travels one topic hop instead of one file handle."""
    assert stream_writes(AGENT_TREE) == set(), stream_writes(AGENT_TREE)
    imported = {a.name for n in ast.walk(AGENT_TREE) if isinstance(n, ast.ImportFrom)
                and n.module == "memory" for a in n.names}
    assert "store" not in imported, imported


def test_the_audit_publisher_exists_before_the_first_composition() -> None:
    assert audit_publisher_before_first_compose(AGENT)


def test_the_node_records_the_kind_the_store_declares() -> None:
    """The kind selects the field set, so the literal in the node has to be a row."""
    kinds = audit_kinds(AGENT)
    assert kinds == {CHAT_KIND}, kinds
    assert CHAT_KIND in st.AUDIT_FIELDS, sorted(st.AUDIT_FIELDS)


def test_the_memory_node_appends_what_arrives_on_the_audit_topic() -> None:
    subs, _pubs = topic_edges(MEMORY_TREE, MEMORY)
    assert constants(MEMORY_TREE)["AUDIT_TOPIC"] in subs, sorted(subs)
    assert "on_audit" in MEMORY, sorted(MEMORY)
    assert stream_writes(ast.Module(body=[MEMORY["on_audit"]],
                                    type_ignores=[])) == {"audit"}


def test_a_refused_record_does_not_take_the_node_down() -> None:
    """A malformed record from another node is a defect in that node — not a reason for
    the robot to lose its memory for the rest of the session."""
    handlers = [h for n in ast.walk(MEMORY["on_audit"]) if isinstance(n, ast.Try)
                for h in n.handlers]
    assert handlers, "on_audit no longer handles a refusal — one bad record takes memory"


def test_the_composed_turn_record_is_one_the_single_writer_accepts() -> None:
    """End to end through the two pure modules: what the node builds, the store takes."""
    con = guard.load_constitution()
    assembly = cx.assemble(con)
    record = cx.audit_payload(assembly, {"kind": CHAT_KIND, "findings": ["a finding"]})
    assert st.check_audit(record) == CHAT_KIND
    assert set(record) - {"kind"} == set(st.AUDIT_FIELDS[CHAT_KIND]), sorted(record)


def test_the_consumer_the_docstring_used_to_name_is_refused_both_ways() -> None:
    """`audit_payload` claimed for two stages that its record goes to
    `guard.audit_record`. It cannot, and the refusal is symmetric — which is why the
    correction is a test and not a comment.

    Keep the stamped `kind="audit"` and the three payload fields are undeclared for that
    row. Override the kind to `"chat"` and the three fields `audit_record` mandates are
    undeclared instead. There is no argument that makes the composition legal, so the
    sentence was not merely stale: it described a path that never existed.
    """
    payload = cx.audit_payload(cx.assemble(guard.load_constitution()))
    stamped = guard.audit_record("guard_blocked", "recorded", "a reason", **payload)
    with pytest.raises(st.StoreError) as first:
        st.check_audit(stamped)
    assert "undeclared" in str(first.value), first.value

    overridden = guard.audit_record("guard_blocked", "recorded", "a reason",
                                    kind=CHAT_KIND, **payload)
    with pytest.raises(st.StoreError) as second:
        st.check_audit(overridden)
    assert "undeclared" in str(second.value), second.value
    assert "action" in str(second.value), second.value


# ================================================================ layer L3: mutants
# Each row is (id, anchor, replacement, the check that must break). The check is the
# same function the corresponding test above calls, so a mutant surviving means that
# test measures the wrong thing — not that the code is safe.

def mutate_agent(old: str, new: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    """Re-parse agent_node.py with one textual substitution applied.

    The anchor assertion matters: without it a stale mutant becomes a no-op, the check
    keeps passing, and the report reads "mutant survived" — blaming the test for a
    broken mutant. Two failures, two messages.
    """
    assert old in AGENT_SRC, f"mutant anchor no longer in agent_node.py: {old!r}"
    return parse(AGENT_SRC.replace(old, new, 1), "AgentNode")


def mutate_memory(old: str, new: str) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    assert old in MEMORY_SRC, f"mutant anchor no longer in memory_node.py: {old!r}"
    return parse(MEMORY_SRC.replace(old, new, 1), "MemoryNode")


@pytest.mark.mutation
def test_no_mutant_control() -> None:
    """§6.0 rule 2: with no substitution applied every check must report the healthy
    value. A harness that reported failure for all inputs would otherwise "kill" every
    mutant below while proving nothing."""
    tree, methods = mutate_agent("class AgentNode(Node):", "class AgentNode(Node):")
    mtree, mmethods = mutate_memory("class MemoryNode(Node):", "class MemoryNode(Node):")
    subs, pubs = topic_edges(tree, methods)
    topic = constants(tree)["PROFILE_TOPIC"]
    assert subs[topic] == "PROFILE_QOS"
    assert topic not in pubs
    assert qos_keywords(tree, "PROFILE_QOS") == qos_keywords(MEMORY_TREE, "PROFILE_QOS")
    assert constants(tree)["AUDIT_TOPIC"] == constants(mtree)["AUDIT_TOPIC"]
    assert "card" in call_keywords(methods["_compose"], "assemble")
    assert {"__init__", "on_profile"} <= callers_of(methods, "_compose")
    assert not adopts_inside_try(methods, "on_profile", "card")
    assert cardless_turn_audits(methods)
    assert stream_writes(tree) == set()
    assert audit_publisher_before_first_compose(methods)
    assert audit_kinds(methods) == {CHAT_KIND}
    assert stream_writes(ast.Module(body=[mmethods["on_audit"]],
                                    type_ignores=[])) == {"audit"}


@pytest.mark.mutation
def test_mutant_volatile_card_subscription() -> None:
    """MUTANT: the subscriber asks for VOLATILE durability.

    The one that would be invisible in production: memory_node publishes once, at boot,
    so a volatile subscriber receives nothing, forever, on a robot whose own log says the
    card was published. No exception, no empty payload — just a prompt with no profile.
    """
    tree, _methods = mutate_agent(
        "    durability=DurabilityPolicy.TRANSIENT_LOCAL,\n)",
        "    durability=DurabilityPolicy.VOLATILE,\n)")
    mine = qos_keywords(tree, "PROFILE_QOS")
    assert mine != qos_keywords(MEMORY_TREE, "PROFILE_QOS"), \
        "the QoS comparison cannot tell a volatile subscriber from a latched one"


@pytest.mark.mutation
def test_mutant_profile_topic_typo() -> None:
    """MUTANT: subscribe to a topic no node publishes."""
    tree, _methods = mutate_agent('PROFILE_TOPIC = "/memory/profile"',
                                  'PROFILE_TOPIC = "/memory/profiles"')
    assert constants(tree)["PROFILE_TOPIC"] != constants(MEMORY_TREE)["TOPIC"], \
        "the pinned copy is not pinned"


@pytest.mark.mutation
def test_mutant_audit_topic_typo() -> None:
    """MUTANT: publish the record where the single writer is not listening."""
    tree, _methods = mutate_agent('AUDIT_TOPIC = "/memory/audit"',
                                  'AUDIT_TOPIC = "/memory/audits"')
    assert constants(tree)["AUDIT_TOPIC"] != constants(MEMORY_TREE)["AUDIT_TOPIC"], \
        "the pinned copy is not pinned"


@pytest.mark.mutation
def test_mutant_card_not_forwarded() -> None:
    """MUTANT: the card is held on the node and never handed to `assemble` — exactly
    the state this PR found the robot in, restored in one edit."""
    _tree, methods = mutate_agent(",\n                                    card=self.card)",
                                  ")")
    assert "card" not in call_keywords(methods["_compose"], "assemble"), \
        "the forwarding check does not read the call's keywords"


@pytest.mark.mutation
def test_mutant_the_card_arrives_and_nothing_is_recomposed() -> None:
    """MUTANT: `on_profile` stores the card and leaves the prompt as it was.

    The subtle one, and the reason re-composition is a named decision rather than a
    detail: every structural check about the subscription still passes, the card really
    is in memory, and the prompt sent to Bedrock never contains it.
    """
    _tree, methods = mutate_agent(
        '        self._compose(f"card arrived: {len(card)} facts")',
        "        pass")
    assert "on_profile" not in callers_of(methods, "_compose"), \
        "the re-composition check does not notice a card that changes nothing"


@pytest.mark.mutation
def test_mutant_card_adopted_before_it_is_validated() -> None:
    """MUTANT: assign the card inside the try, so a half-parsed payload is adopted."""
    _tree, methods = mutate_agent(
        "            card = pf.parse(json.loads(msg.data), where=PROFILE_TOPIC)",
        "            card = pf.parse(json.loads(msg.data), where=PROFILE_TOPIC)\n"
        "            self.card = card")
    assert adopts_inside_try(methods, "on_profile", "card"), \
        "the adoption check cannot see an assignment inside the validation"


@pytest.mark.mutation
def test_mutant_a_cardless_turn_says_nothing() -> None:
    """MUTANT: the degraded turn stops being recorded. The robot still answers, so
    nothing anywhere would say the profile was missing."""
    _tree, methods = mutate_agent("            if self.card is None:",
                                  "            if False:")
    assert not cardless_turn_audits(methods), \
        "the degraded-turn check passes on a branch that can never be taken"


@pytest.mark.mutation
def test_mutant_the_brain_becomes_a_second_writer() -> None:
    """MUTANT: append straight to the store from the agent, bypassing the one writer."""
    tree, _methods = mutate_agent(
        "        self.pub_audit.publish(String(data=json.dumps(",
        '        self.store.append("audit", record)\n'
        "        self.pub_audit.publish(String(data=json.dumps(")
    assert stream_writes(tree) == {"audit"}, \
        "the single-writer census does not notice a stream write in the brain"


@pytest.mark.mutation
def test_mutant_the_record_declares_an_undeclared_kind() -> None:
    """MUTANT: `kind` becomes a name no `AUDIT_FIELDS` row governs.

    Two checks have to fire, and they fire in different places: the node's literal is no
    longer a declared row, and the store refuses the record at the append. The second
    one is what makes the first one worth having — a kind chosen by its first writer is
    how a schema gets decided by accident in a file no deletion can rewrite.
    """
    _tree, methods = mutate_agent('{"kind": "chat",', '{"kind": "turn",')
    kinds = audit_kinds(methods)
    assert kinds == {"turn"}, kinds
    assert not kinds <= set(st.AUDIT_FIELDS), sorted(st.AUDIT_FIELDS)
    record = cx.audit_payload(cx.assemble(guard.load_constitution()), {"kind": "turn"})
    with pytest.raises(st.StoreError) as exc:
        st.check_audit(record)
    assert "unknown audit kind" in str(exc.value), exc.value


@pytest.mark.mutation
def test_mutant_the_publisher_is_made_after_the_boot_record() -> None:
    """MUTANT: create `pub_audit` after the boot composition, so the first record is
    published to nothing while every log line reads healthy."""
    _tree, methods = mutate_agent(
        "        self.pub_audit = self.create_publisher(String, AUDIT_TOPIC, 10)\n"
        '        self._compose("boot")',
        '        self._compose("boot")\n'
        "        self.pub_audit = self.create_publisher(String, AUDIT_TOPIC, 10)")
    assert not audit_publisher_before_first_compose(methods), \
        "the ordering check does not constrain the order"


@pytest.mark.mutation
def test_mutant_the_single_writer_drops_the_record() -> None:
    """MUTANT: `on_audit` receives the record and never appends it. The publisher side
    is unchanged, so the brain's logs are identical and `audit.jsonl` is empty."""
    _tree, methods = mutate_memory(
        '            seq = self.store.append("audit", record)',
        "            seq = -1")
    assert stream_writes(ast.Module(body=[methods["on_audit"]],
                                    type_ignores=[])) == set(), \
        "the append check does not read the callback that performs it"
