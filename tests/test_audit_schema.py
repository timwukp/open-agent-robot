"""T-C7 — the content schema on `audit` records (`memory.store.AUDIT_FIELDS`).

Its own file, for the reason `tests/test_why.py` is its own file: the subject is a
*field vocabulary and the census that keeps it honest*, and the census is the substance.

**What this closes was not a latent gap.** `memory/erase.py`'s `RETAINED` said the
consequence out loud ("nothing enforces that an audit record omits user text") while
`safety/guard.audit_record`'s docstring asserted the opposite premise — that the
cascade-delete path covering `events` covers this stream. It does not:
`REWRITABLE["audit"] is False` and `RETAINED` lists `audit.jsonl`. And the contradiction
had already been written into a test: the previous
`tests/test_guard.py::test_the_audit_stream_shares_the_one_writer` built
`audit_record(..., utterance="來揮手")` and asserted it lands in `audit.jsonl` and reads
back — the user's speech in the one stream no deletion can reach, asserted green. So the
defect was live and pinned as contract, and the invitation in the docstring was the
defect rather than any single line of code.

Four things here are worth reading before the assertions.

**A closed field set is only enforceable in the one writer.** An AST census over
production code cannot see through `audit_record(**fields)` — a caller forwarding
`**user_supplied` is opaque to any reader of the source. That is not a gap in the census
so much as the reason the runtime check has to exist, and `test_the_census_states_its
_own_blind_spot` asserts the `**fields` signature is really there, so the day someone
closes it the justification for the belt-and-braces stops being invisible.

**The positive controls are driven producers, not hand-built dicts.** A dict written in
this file proves the checker accepts what this file believes; it cannot notice that
`safety/output_guard.py` sets an eleventh field on one branch. Two of the four kinds are
driven here — `audit` through `OutputGuard` over every branch that decides which
optional fields exist, and `chat` through `context.audit_payload`, which is declared
ahead of its caller and so has nothing else holding it to the shape it will really
receive. `profile_loaded` is written by a ROS node this suite must not import, so it is
covered statically instead — by reading the literal at its call site, which is the same
check the census applies to every other site. `tombstone` is driven by the real
deletion, in `tests/test_erase.py`, which owns the only fixture rich enough to satisfy
`delete`'s preconditions; the assertion was added there rather than reproduced here,
because a second copy of that fixture is a second reader free to drift from the first.
What this file adds for that kind is the cross-module agreement its two validators owe
each other.

**The census reads much less than the walk suggests, and that is measured.** Of the
three audit call sites on disk, exactly one passes a dict an AST can read: `er.delete`
builds its tombstone in a local first, and `agent/context.py` has no call site at all —
it *builds* a payload a caller forwards. So the weakness is written down as data
(`OPAQUE_SITES`, `BUILDER_ONLY`, `NOT_PRODUCERS`), every entry has to name the test that
covers it instead, and that test has to exist. A census that quietly skips the sites it
cannot parse reports the same "clean" as one where every site is fine.

**The ceiling is not a text detector and does not claim to be.** No check on a string's
value distinguishes prose we wrote from prose someone said; that is provenance, and
provenance is checked by walking the call sites. `AUDIT_TEXT_CEILING` bounds *how much*
of an upstream string outlives a deletion — the route that matters is `detail`, which
comes from the classifier's own error message and has no length its author controls.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_PY = REPO_ROOT / "memory" / "store.py"

from memory import erase as er          # noqa: E402
from memory import store as st          # noqa: E402
from safety import guard                # noqa: E402
from safety import output_guard as og    # noqa: E402
from agent import context as cx        # noqa: E402

ANSWER = "user_requested_answer"
WHY = {"goal": ANSWER}
UID = "say:好,我來說一段話"
T0 = 1000.0

# Every production file that may append to the audit stream. Derived below rather than
# trusted: `test_the_census_reads_every_producer` re-derives this set from the imports
# and call sites on disk, because a name list is exactly what cannot notice a new name.
PRODUCERS = ("memory/store.py", "memory/memory_node.py", "memory/erase.py",
             "safety/guard.py", "safety/output_guard.py", "agent/context.py",
             "agent/agent_node.py")

# Floor on the census. A walk that resolves zero call sites reports clean, and clean is
# the same output as "every site is fine" — so the count is asserted, not just the
# findings. Three is what is on disk today; the assertion is `>=`, because a new
# producer must raise this number and never lower it silently. R1.1 raised it from 3 by
# adding `memory_node.on_audit`.
MIN_CENSUS_SITES = 4

# The sites the census resolves but cannot read, each with the test that covers it
# instead. This table is the reason the census is not allowed to just skip them.
#
# Measured, not assumed: of the four call sites on disk, exactly two pass a dict the
# AST can read. That is a much weaker census than the walk suggests, and writing the
# weakness down as data is what makes it fail — a third opaque site is a KeyError here
# until someone either makes it visible or names where it is covered.
OPAQUE_SITES = {
    # `delete` builds the tombstone in a local before appending it, so the argument is
    # a Name and not a Dict. Covered on both sides instead: the write side by
    # `tests/test_erase.py::test_the_tombstone_names_ids_and_counts_and_quotes_no_
    # deleted_text` (which runs the real deletion and calls `check_audit` on the record
    # it produced), and the field set by the read/write agreement asserted above.
    ("memory/erase.py", "?"):
        "tests/test_erase.py::test_the_tombstone_names_ids_and_counts_and_"
        "quotes_no_deleted_text",
    # R1.1: `memory_node.on_audit` appends a record that arrived over `/memory/audit` as
    # JSON, so the argument is a Name whose fields no AST in this repo can know — they
    # were decided in another process. Unreadable *by construction*, which is the point
    # of the single writer: the fields are `agent_node`'s and the acceptance is
    # `check_audit`'s, and the named test drives exactly that pair without ROS.
    ("memory/memory_node.py", "?"):
        "tests/test_profile_wiring.py::test_the_composed_turn_record_is_one_the_"
        "single_writer_accepts",
}

# Producers the census cannot see at all, because the record leaves the file by some
# route other than an append. Named here so each gap is data rather than an omission,
# with the test that covers it instead.
#
# Two distinct shapes, deliberately in one table rather than two: a second table with one
# row is a category invented for a single member, and the rule both rows need is the same
# one — name the coverage, and the coverage must exist.
BUILDER_ONLY = {
    # Builds the payload; a *caller* forwards it. No audit call site in the file at all.
    "agent/context.py":
        "tests/test_audit_schema.py::test_the_chat_payload_agent_node_will_hand_"
        "over_is_already_accepted",
    # R1.1: builds the `chat` record and *publishes* it on `/memory/audit`, because the
    # brain holds no `Store` — the append is `memory_node.on_audit`, in another process.
    # So this file produces audit records and contains no append, which is not a gap in
    # the census but the single-writer invariant showing up in it.
    "agent/agent_node.py":
        "tests/test_profile_wiring.py::test_the_composed_turn_record_is_one_the_"
        "single_writer_accepts",
}

# Files that touch the audit stream in a role other than producing a record. Written
# down because "mentions the audit stream" is what a scan can see and "writes an audit
# record" is what the census needs, and collapsing the two makes the coverage rule
# either vacuous (skip anything unreadable) or wrong (demand coverage of the checker).
NOT_PRODUCERS = {
    # The writer and the checker themselves. `check_audit` lives here; a rule demanding
    # that the census read the gate's own fields is a rule about the wrong file.
    "memory/store.py": "the single writer — it validates records, it does not make any",
    # The creator. `audit_record` is defined here and called from elsewhere, so there is
    # no call site in this file. Its `**fields` signature is the census's blind spot and
    # is asserted by test_the_census_states_its_own_blind_spot.
    "safety/guard.py": "defines audit_record; every call to it is in another file",
}


def make_store(tmp_path: Path, module=st, **kw):
    """A store rooted in tmp_path. Same visible exemption as tests/test_store.py."""
    return module.Store(tmp_path / "data", _allow_ephemeral=True, **kw)


def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/store.py.

    Keyed by `label` for the reason `tests/test_why.py` gives: two mutants with the
    same anchor count inside one test would land in one directory and the first import
    would win, which presents as a mutant that mysteriously survives. Zero
    replacements gives the no-mutant control (§6.0 rule 2).
    """
    dst = tmp_path / f"mut_{label}" / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    src = STORE_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in store.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "store.py"
    target.write_text(src, encoding="utf-8")
    name = f"store_audit_mutant_{tmp_path.name}_{label}"
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


# ----------------------------------------------------------------- mutant anchors
KIND_GATE = "    if kind not in AUDIT_FIELDS:\n"
ALLOWED_LOOKUP = "    allowed = AUDIT_FIELDS[kind]\n"
UNDECLARED_GATE = "    if undeclared:\n"
CEILING = "AUDIT_TEXT_CEILING = 500"
APPEND_HOOK = '        if stream == "audit":\n            check_audit(record)\n'
SEQ_LINE = '        line = json.dumps({"seq": seq'


def read_audit(store) -> list[dict]:
    store.flush()
    path = store.path_for("audit")
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ============================================ the two sides of the gate must agree
def test_every_declared_kind_has_a_named_producer_and_no_field_is_declared_twice()\
        -> None:
    """The declaration is documentation until something derives it.

    Two directions. Every kind names a producer (a row nobody writes is a field set
    that can only be wrong), and no kind redeclares one of the store's own fields —
    a row listing `why` or `ts` would be a second reader of a decision `WHY_REQUIRED`
    and the writer already own, free to drift from the first.
    """
    assert set(st.AUDIT_FIELDS) == {"audit", "chat", "profile_loaded", "tombstone"}
    for kind, fields in st.AUDIT_FIELDS.items():
        assert fields, f"{kind!r} declares no fields at all"
        assert list(fields) == sorted(fields), f"{kind!r} is not sorted: {fields}"
        assert len(set(fields)) == len(fields), f"{kind!r} repeats a field: {fields}"
        clash = sorted(set(fields) & set(st.AUDIT_IMPLICIT_FIELDS))
        assert not clash, (
            f"{kind!r} redeclares the store's own field(s) {clash}: the implicit set "
            f"is {st.AUDIT_IMPLICIT_FIELDS} and a per-kind copy can drift from it")


def test_the_tombstone_row_and_TOMBSTONE_FIELDS_are_the_same_format() -> None:
    """Two readers, one format — so assert the two agree, not that each is self-consistent.

    `memory/erase.py` validates a tombstone on the way back **in**
    (`parse_tombstone`, whose `TOMBSTONE_FIELDS` is the complete on-disk field set);
    `check_audit` validates it on the way **out**. Pinned separately, each stays green
    while the other moves, and the failure that follows is a record the writer accepts
    and the reader rejects — a tombstone that cannot be read is a deletion that cannot
    be proven.
    """
    assert set(er.TOMBSTONE_FIELDS) == (
        set(st.AUDIT_FIELDS["tombstone"]) | set(st.AUDIT_IMPLICIT_FIELDS)), (
        f"read side {er.TOMBSTONE_FIELDS} disagrees with write side "
        f"{st.AUDIT_FIELDS['tombstone']} + {st.AUDIT_IMPLICIT_FIELDS}")


def test_erase_still_says_what_it_retains_and_no_longer_says_it_is_unbounded() -> None:
    """`RETAINED` is on every receipt, so its text is an interface.

    It used to promise the gap: "nothing enforces that an audit record omits user
    text". A receipt that keeps saying that after the schema landed is a receipt
    understating what the deletion achieved, which is the same class of defect as
    overstating it — both are a receipt that does not describe the system.
    """
    text = er.RETAINED["audit.jsonl"]
    assert "AUDIT_FIELDS" in text, (
        "the receipt does not name the schema that bounds what it retains")
    for stale in ("nothing enforces", "not yet closed", "until then"):
        assert stale not in text, (
            f"the receipt still describes the gap as open: {stale!r}")
    # And the rendering every receipt carries actually contains it.
    assert any("AUDIT_FIELDS" in f for f in er.retained_findings())


# ================================================== positive controls: real producers
@pytest.mark.parametrize("blocked", [True, False])
@pytest.mark.parametrize("audio", ["none", "started", "done"])
def test_every_record_the_real_output_guard_builds_is_accepted_and_reads_back(
        tmp_path: Path, blocked: bool, audio: str) -> None:
    """The `audit` kind's positive control, driven rather than written here.

    Parametrised over the branches that decide `audibility` and therefore which
    optional fields are present, because the field set is what is under test and a
    single shape cannot notice that one branch sets one more key. `detail` is the
    field that matters most: it is upstream text, so a branch that starts setting it
    is a branch that starts putting someone else's string in the unerasable stream.
    """
    g = og.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    if audio in ("started", "done"):
        g.audio_started(UID, 0, now=T0 + 0.1)
    if audio == "done":
        g.audio_done(UID, 0, now=T0 + 0.4)
    actions = g.verdict(UID, 0, blocked=blocked, now=T0 + 0.5)
    records = [a.record for a in actions if a.kind == og.AUDIT]
    assert len(records) == 1, [a.kind for a in actions]

    store = make_store(tmp_path, clock=lambda: 1.5)
    try:
        record = dict(records[0], why=WHY)
        # The checker agrees with the writer, and the writer actually wrote it.
        assert st.check_audit(record) == "audit"
        store.append("audit", record)
        rows = read_audit(store)
    finally:
        store.close()
    assert len(rows) == 1
    assert rows[0]["action"] == "guard_blocked"
    # Every key that came back is one the schema declares. Read from disk rather than
    # from `record`, so a writer that adds a field on the way out is caught too.
    surplus = sorted(set(rows[0]) - set(st.AUDIT_IMPLICIT_FIELDS)
                     - set(st.AUDIT_FIELDS["audit"]))
    assert not surplus, f"undeclared field(s) reached audit.jsonl: {surplus}"


def test_the_chat_payload_agent_node_will_hand_over_is_already_accepted() -> None:
    """The `chat` kind's control, driven through `context.audit_payload`.

    This row was declared before its caller was wired, so this is the test that makes
    that early declaration mean something: the payload the composed-turn path really
    builds is accepted today, rather than being discovered to be refused by whoever
    wires `agent_node` up. It also asserts what the row exists to keep out — the prompt
    itself, which is the field in that path that would carry the utterance.
    """
    assembly = cx.Assembly("SYSTEM 來揮手 the whole prompt", ("constitution",),
                           ("one finding",))
    payload = cx.audit_payload(assembly, {"kind": "chat"})
    assert st.check_audit(payload) == "chat"
    assert payload["chars"] == len(assembly.prompt)
    blob = json.dumps(payload, ensure_ascii=False)
    assert "來揮手" not in blob, (
        f"the composed prompt reached the audit payload: {blob}")


def test_the_profile_loaded_literal_in_memory_node_carries_only_declared_fields()\
        -> None:
    """`profile_loaded`'s control, static because its producer is a ROS node.

    §6.1 forbids importing a node here, so the literal at the call site is read
    instead. That is weaker than driving it in exactly one way — it cannot see a field
    added at runtime — and stronger in another: it fails when the source changes,
    without anyone having to remember this file exists.
    """
    src = (REPO_ROOT / "memory" / "memory_node.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "kind" not in keys:
            continue
        kinds = [v.value for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and k.value == "kind"
                 and isinstance(v, ast.Constant)]
        if kinds == ["profile_loaded"]:
            found.append(sorted(keys))
    assert found, ("no profile_loaded literal in memory_node.py — either the producer "
                   "moved, in which case this control is measuring nothing, or the "
                   "kind is now unproduced and its AUDIT_FIELDS row is dead")
    for keys in found:
        surplus = sorted(set(keys) - set(st.AUDIT_IMPLICIT_FIELDS)
                         - set(st.AUDIT_FIELDS["profile_loaded"]))
        assert not surplus, f"memory_node writes undeclared field(s) {surplus}"


# ============================================================ the refusals, per kind
@pytest.mark.parametrize("kind", ["audit", "profile_loaded", "tombstone"])
def test_an_undeclared_field_is_refused_for_every_kind(tmp_path: Path,
                                                       kind: str) -> None:
    """Parametrised over `AUDIT_FIELDS` itself, so a kind cannot be added untested."""
    record = {"kind": kind, "utterance": "來揮手", "why": WHY}
    with pytest.raises(st.StoreError, match="undeclared field"):
        st.check_audit(record)
    store = make_store(tmp_path)
    try:
        with pytest.raises(st.StoreError, match="undeclared field"):
            store.append("audit", record)
        # Refused *before* the file exists: a stream whose first line is a rejected
        # record is a stream that recorded the wrong thing to say nothing happened.
        assert read_audit(store) == []
    finally:
        store.close()


def test_the_field_the_old_test_pinned_as_accepted_is_now_refused(tmp_path: Path)\
        -> None:
    """The exact call the previous `test_the_audit_stream_shares_the_one_writer` made.

    Kept as its own case, spelled the way the defect was spelled, because this is the
    one assertion that names what changed: `utterance=` used to reach `audit.jsonl`
    and read back, and the test asserting so was green.
    """
    record = guard.audit_record("gesture_rejected", "blocked", "not in GESTURES",
                                utterance="來揮手", tag="__nope__")
    assert record["utterance"] == "來揮手"  # the creator still forwards **fields
    store = make_store(tmp_path)
    try:
        with pytest.raises(st.StoreError, match="undeclared field.*utterance"):
            store.append("audit", dict(record, why=WHY))
        assert read_audit(store) == []
        # `tag` on its own is declared and does land: the refusal is about the
        # utterance, not about the record being a rejection record.
        store.append("audit", {k: v for k, v in record.items() if k != "utterance"})
        rows = read_audit(store)
    finally:
        store.close()
    assert len(rows) == 1 and rows[0]["tag"] == "__nope__"


def test_an_unknown_kind_and_a_missing_kind_are_both_refused(tmp_path: Path) -> None:
    """The case a default would have swallowed, in both of its spellings."""
    store = make_store(tmp_path)
    try:
        for record, pattern in (
                ({"kind": "chatlog", "text": "來揮手"}, "unknown audit kind"),
                ({"kind": "", "text": "來揮手"}, "no usable 'kind'"),
                ({"text": "來揮手"}, "no usable 'kind'"),
                ({"kind": 7}, "no usable 'kind'")):
            with pytest.raises(st.StoreError, match=pattern):
                store.append("audit", record)
        with pytest.raises(st.StoreError, match="must be a dict"):
            store.append("audit", ["kind", "audit"])
        assert read_audit(store) == []
    finally:
        store.close()


def test_a_declared_field_is_bounded_on_both_sides_of_the_ceiling(tmp_path: Path)\
        -> None:
    """`detail` at the ceiling is accepted, one character over is refused.

    Both sides, because a check that refuses everything long also satisfies "long is
    refused", and the number is the whole claim.
    """
    store = make_store(tmp_path)
    try:
        at = {"kind": "audit", "action": "guard_blocked", "decision": "recorded",
              "reason": "ok", "detail": "x" * st.AUDIT_TEXT_CEILING}
        store.append("audit", at)
        over = dict(at, detail="x" * (st.AUDIT_TEXT_CEILING + 1))
        with pytest.raises(st.StoreError, match="AUDIT_TEXT_CEILING"):
            store.append("audit", over)
        rows = read_audit(store)
    finally:
        store.close()
    assert len(rows) == 1
    assert len(rows[0]["detail"]) == st.AUDIT_TEXT_CEILING


def test_a_refusal_leaves_no_gap_in_the_audit_log(tmp_path: Path) -> None:
    """The property the placement of the check exists to protect.

    A rejected record that consumed a sequence number leaves a hole, and a hole in
    `audit.jsonl` is indistinguishable from a deleted record — the single ambiguity
    this stream exists to remove. So the check that protects the audit log must not be
    able to dent it.
    """
    store = make_store(tmp_path)
    ok = {"kind": "profile_loaded", "facts": 2, "kinds": 3, "locked": True}
    try:
        store.append("audit", ok)
        for bad in ({"kind": "profile_loaded", "utterance": "來揮手"},
                    {"kind": "chatlog"},
                    {"kind": "audit", "action": "a", "decision": "d",
                     "reason": "r", "detail": "x" * 5000}):
            with pytest.raises(st.StoreError):
                store.append("audit", bad)
        store.append("audit", ok)
        rows = read_audit(store)
    finally:
        store.close()
    assert [r["seq"] for r in rows] == [1, 2], (
        f"three refusals dented the log: {[r['seq'] for r in rows]}")


# ================================================================== the census (T-C7)
def audit_call_sites() -> list[tuple[str, int, str, list[str]]]:
    """Every `store.append("audit", {...})` and `audit_record(...)` in production code.

    Returns `(path, lineno, kind_or_unknown, visible_field_names)`. "Visible" is the
    load-bearing word: a `**kwargs` forwarding site contributes no names, and that is
    reported rather than skipped — see `test_the_census_states_its_own_blind_spot`.
    """
    sites: list[tuple[str, int, str, list[str]]] = []
    for rel in PRODUCERS:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name == "append" and node.args and isinstance(node.args[0], ast.Constant)\
                    and node.args[0].value == "audit":
                arg = node.args[1] if len(node.args) > 1 else None
                if isinstance(arg, ast.Dict):
                    keys = [k.value for k in arg.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    kinds = [v.value for k, v in zip(arg.keys, arg.values)
                             if isinstance(k, ast.Constant) and k.value == "kind"
                             and isinstance(v, ast.Constant)]
                    sites.append((rel, node.lineno,
                                  kinds[0] if kinds else "?", sorted(keys)))
                else:
                    sites.append((rel, node.lineno, "?", []))
            elif name == "audit_record":
                kws = sorted(k.arg for k in node.keywords if k.arg)
                # `audit_record`'s three positional parameters are always these.
                sites.append((rel, node.lineno, "audit",
                              sorted({"action", "decision", "reason", *kws})))
    return sites


def test_the_census_reads_every_producer_and_finds_the_sites_on_disk() -> None:
    """A walk that resolves nothing reports clean, and clean is the wrong answer.

    Also re-derives `PRODUCERS`: every file that mentions the audit stream at all must
    be in the list, so a sixth producer cannot be invisible to the four checks below
    just because nobody added it here.
    """
    mentions = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in list((REPO_ROOT / "memory").glob("*.py"))
        + list((REPO_ROOT / "safety").glob("*.py"))
        + list((REPO_ROOT / "game").glob("*.py"))
        + list((REPO_ROOT / "agent").glob("*.py"))
        + list((REPO_ROOT / "speech").glob("*.py"))
        if '"audit"' in p.read_text(encoding="utf-8")
        or "audit_record" in p.read_text(encoding="utf-8"))
    missing = sorted(set(mentions) - set(PRODUCERS))
    assert not missing, (
        f"file(s) touch the audit stream and are outside the census: {missing}")
    sites = audit_call_sites()
    assert len(sites) >= MIN_CENSUS_SITES, (
        f"the census resolved {len(sites)} call sites, under the floor of "
        f"{MIN_CENSUS_SITES} — a census that finds nothing passes for free")
    # A file in PRODUCERS with no readable site and no entry in BUILDER_ONLY is a
    # producer the next three tests silently do not cover.
    walked = {rel for rel, _, kind, fields in sites if kind != "?" and fields}
    for rel in PRODUCERS:
        if rel in walked or rel in BUILDER_ONLY or rel in NOT_PRODUCERS:
            continue
        opaque = [k for k in OPAQUE_SITES if k[0] == rel]
        assert opaque, (
            f"{rel} is listed as a producer but the census reads no fields from it and "
            f"nothing names where it is covered instead — add it to BUILDER_ONLY or "
            f"OPAQUE_SITES with the test that does cover it, or drop it from PRODUCERS")


def test_the_role_each_producer_plays_is_written_down_and_true() -> None:
    """`NOT_PRODUCERS` excuses a file from the coverage rule, so it must be checked.

    An excuse is the most dangerous kind of documentation: the docstring saying where a
    guard need not look is where the next instance hides. So the claim "this file makes
    no audit record" is asserted against the disk, and no file may hold two roles.
    """
    sites = audit_call_sites()
    for rel, reason in NOT_PRODUCERS.items():
        assert reason, f"{rel} is excused with no reason"
        readable = [(r, ln) for r, ln, kind, fields in sites
                    if r == rel and kind != "?" and fields]
        assert not readable, (
            f"{rel} is listed as making no audit record, but the census reads "
            f"{readable} — the excuse is now false and the file needs covering")
        assert rel not in BUILDER_ONLY, f"{rel} is in two roles"
    for rel in BUILDER_ONLY:
        assert not any(k[0] == rel for k in OPAQUE_SITES), f"{rel} is in two roles"


def test_every_site_the_census_cannot_read_names_the_test_that_covers_it() -> None:
    """No silent caps: an unreadable site must be accounted for, not skipped.

    `test_every_visible_audit_field_in_production_code_is_declared` continues past a
    site whose kind it cannot resolve, which is correct — and is also how a site that
    became opaque stops being checked without anything going red. So the set of opaque
    sites is itself pinned, and the coverage named for each one has to exist.
    """
    found = {(rel, kind) for rel, _, kind, fields in audit_call_sites()
             if kind == "?" or not fields}
    assert found == set(OPAQUE_SITES), (
        f"the opaque call sites changed: on disk {sorted(found)}, "
        f"written down {sorted(OPAQUE_SITES)}")
    for where, test_id in {**OPAQUE_SITES, **BUILDER_ONLY}.items():
        path, _, name = test_id.partition("::")
        src = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert f"def {name}(" in src, (
            f"{where} says it is covered by {test_id}, which does not exist")


def test_every_visible_audit_field_in_production_code_is_declared() -> None:
    """The provenance half: no production site names a field the schema has not.

    This is what the ceiling cannot do. A string check bounds length; only reading the
    call sites shows whether the *thing being passed* is ours or someone's speech.
    """
    bad = []
    for rel, lineno, kind, fields in audit_call_sites():
        if kind == "?":
            continue
        allowed = set(st.AUDIT_FIELDS.get(kind, ())) | set(st.AUDIT_IMPLICIT_FIELDS)
        surplus = sorted(set(fields) - allowed)
        if kind not in st.AUDIT_FIELDS:
            bad.append(f"{rel}:{lineno} writes unknown kind {kind!r}")
        elif surplus:
            bad.append(f"{rel}:{lineno} ({kind}) passes undeclared {surplus}")
    assert not bad, "\n".join(bad)


def test_the_census_states_its_own_blind_spot() -> None:
    """The reason the runtime check is not redundant with the census above.

    `audit_record(**fields)` is opaque to any reader of the source: a caller
    forwarding `**user_supplied` contributes no visible names, so the census would
    pass a site that passes the utterance. Asserted rather than remarked, so that
    closing the signature is what removes the justification — not someone's memory.
    """
    tree = ast.parse((REPO_ROOT / "safety" / "guard.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "audit_record")
    assert fn.args.kwarg is not None, (
        "audit_record no longer takes **kwargs. That is an improvement — the census "
        "in this file can now see every field a caller passes — and it means this "
        "test, and the paragraph in the module docstring above it, are stale")
    # And the docstring must not re-assert the premise that started this.
    doc = ast.get_docstring(fn) or ""
    assert "must not carry what was said" in doc
    assert "AUDIT_FIELDS" in doc, "the creator does not name its own authority"


# ===================================================================== T-J4 mutants
@pytest.mark.mutation
def test_no_mutant_control_accepts_and_refuses_exactly_as_the_real_module_does(
        tmp_path: Path) -> None:
    """§6.0 rule 2's control. If this reds, the harness is broken, not the code."""
    mod = load_mutant(tmp_path, "control")
    assert mod.check_audit({"kind": "tombstone", "episode": "ep-000001",
                            "removed": {}}) == "tombstone"
    with pytest.raises(mod.StoreError):
        mod.check_audit({"kind": "tombstone", "utterance": "來揮手"})
    with pytest.raises(mod.StoreError):
        mod.check_audit({"kind": "chatlog"})


@pytest.mark.mutation
def test_mutant_audit_unknown_field_allowed_lets_the_utterance_through(
        tmp_path: Path) -> None:
    """`audit_unknown_field_allowed` — the gate computed and then not acted on.

    Killed behaviourally through a real `append`, not just by the absence of a raise:
    what the defect does is put the text on disk, so the assertion reads the disk.
    """
    mod = load_mutant(tmp_path, "unknown_field", (UNDECLARED_GATE, "    if False:\n"))
    store = make_store(tmp_path, module=mod)
    try:
        store.append("audit", {"kind": "audit", "action": "gesture_rejected",
                               "decision": "blocked", "reason": "not in GESTURES",
                               "utterance": "來揮手"})
        store.flush()
        text = store.path_for("audit").read_text(encoding="utf-8")
    finally:
        store.close()
    assert "來揮手" in text, "the mutant did not reproduce the defect it names"


@pytest.mark.mutation
def test_mutant_audit_fields_default_open_lets_a_new_kind_choose_its_own_schema(
        tmp_path: Path) -> None:
    """`audit_fields_default_open` — the default nobody decided.

    Two anchors, because an open default needs both halves: the kind gate stops
    refusing, and the lookup stops raising. That is precisely the shape a well-meaning
    `AUDIT_FIELDS.get(kind, ...)` would have.
    """
    mod = load_mutant(tmp_path, "default_open",
                      (KIND_GATE, "    if False:\n"),
                      (ALLOWED_LOOKUP,
                       "    allowed = AUDIT_FIELDS.get(kind, tuple(record))\n"))
    assert mod.check_audit({"kind": "chatlog", "utterance": "來揮手"}) == "chatlog"
    with pytest.raises(st.StoreError):
        st.check_audit({"kind": "chatlog", "utterance": "來揮手"})


@pytest.mark.mutation
def test_mutant_audit_cap_removed_lets_an_unbounded_upstream_string_through(
        tmp_path: Path) -> None:
    """`audit_cap_removed` — the declared field with upstream content."""
    mod = load_mutant(tmp_path, "cap_removed",
                      (CEILING, "AUDIT_TEXT_CEILING = 10 ** 9"))
    record = {"kind": "audit", "action": "guard_blocked", "decision": "recorded",
              "reason": "r", "detail": "來揮手" * 4000}
    assert mod.check_audit(record) == "audit"
    with pytest.raises(st.StoreError, match="AUDIT_TEXT_CEILING"):
        st.check_audit(record)


@pytest.mark.mutation
def test_mutant_audit_check_moved_after_seq_dents_the_log(tmp_path: Path) -> None:
    """`audit_check_moved_after_seq` — right check, wrong side of the counter.

    The record is still refused, so a test asserting only "it raises" passes this
    mutant. What breaks is the log: the refusal consumed a number, so the surviving
    records read 1 and 3 and the gap is indistinguishable from a deleted record.
    """
    mod = load_mutant(tmp_path, "after_seq",
                      (APPEND_HOOK, ""),
                      (SEQ_LINE, '        if stream == "audit":\n'
                                 "            check_audit(record)\n" + SEQ_LINE))
    store = make_store(tmp_path, module=mod)
    ok = {"kind": "profile_loaded", "facts": 2, "kinds": 3, "locked": True}
    try:
        store.append("audit", ok)
        with pytest.raises(mod.StoreError):
            store.append("audit", {"kind": "profile_loaded", "utterance": "來揮手"})
        store.append("audit", ok)
        store.flush()
        seqs = [json.loads(line)["seq"] for line in
                store.path_for("audit").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    finally:
        store.close()
    assert seqs == [1, 3], (
        f"the mutant left no gap ({seqs}), so the placement assertion is measuring "
        "nothing — the real module's [1, 2] would be indistinguishable")
