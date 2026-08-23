"""Offline tests for memory/store.py — the single writer (T-A1..A4 + the exemption).

Three things in here are not what docs/test-design.md §2.A asked for, and the
reasons are the part worth reading.

**T-A3 as written cannot distinguish fsync from flush.** The design says: child
process writes, calls `os._exit(0)` so no cleanup runs, parent re-reads, and the
mutant is "remove flush/fsync". But `os._exit` does not empty the page cache —
data that reached the kernel stays visible to every other process on the machine,
so the parent finds the record whether or not anything was fsynced. Only a power
loss or a kernel panic separates those two, and neither is available to pytest.
So the claim is split in two: a behavioural test for the half that is observable
(the record survives an exit that runs no cleanup — nothing depends on `close()`,
`__del__` or `atexit`, with the missing-flush mutant to prove it bites), and a
*static* test for the half that is not (fsync and the directory fsync are on the
path, asserted against the AST). A single test claiming both would be reporting a
guarantee it never checked.

**T-A2's crash injection is replaced by a property that needs no injection.** The
design monkeypatches `os.replace` into raising. That works for the real code but
is not comparable across the mutant, because the in-place mutant never calls
`os.replace` and so never receives the injected fault — a mutant that survives by
dodging the injection rather than by being safe. The discriminating property is
the one a concurrent reader actually cares about: a reader holding the old file
open must keep seeing the old bytes. Rename satisfies that; in-place truncation
cannot. It is deterministic and needs no monkeypatch. The injection test stays,
but its own mutant is scoped to what it really asserts — that no `.tmp` debris
survives a failure.

**The interleave is scheduled, not hoped for.** The T-A1 bypass mutant ("let each
thread write directly") corrupts the file only if the threads actually interleave.
Relying on the GIL's scheduling makes that mutant pass on a busy laptop and fail
on an idle one, and a flaky mutation test is worse than an artificial one: it
teaches you to re-run. So the bypass writes the same chunks a buffered writer
would (>8 KB record = more than one `write()`) with the interleave written down
explicitly. The pair of tests is what carries the claim: the real store takes the
same 4000-record load from four threads and comes out intact, and the format
itself is shown to be defenceless against interleaving — so the cleanliness is a
property of the writer, not of jsonl.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_PY = REPO_ROOT / "memory" / "store.py"

from memory import store as st  # noqa: E402

# Larger than a Python file object's 8192-byte buffer, so one logical record is
# more than one write() syscall. That is the whole reason a single writer is
# needed; a record that always fits in the buffer would make the bypass look safe.
PAD = "x" * 8300


def make_store(tmp_path: Path, module=st, **kw) -> "st.Store":
    """A store rooted in tmp_path.

    `_allow_ephemeral=True` is the visible exemption: tmp_path is under the system
    temp directory, which `location_findings` correctly refuses in production.
    `test_only_tests_grant_the_ephemeral_exemption` derives from the source that no
    file outside tests/ passes it.
    """
    return module.Store(tmp_path / "data", _allow_ephemeral=True, **kw)


# --------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/store.py.

    Same shape as tests/test_opponent.py. Zero replacements gives the no-mutant
    control required by §6.0 rule 2 — a harness that cannot import a clean copy
    reports every mutant as killed.

    Note the copy's `location_findings` derives "the repo" from its own `__file__`,
    so its notion of the repo tree is tmp_path, not this repository. Tests about
    the repo-tree check must therefore ask about a path under the *copy's* root, or
    the mutant is vacuous.
    """
    dst = tmp_path / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "store.py"
    src = STORE_PY.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in store.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src)
    name = f"store_mutant_{tmp_path.name}_{len(replacements)}"
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


FLUSH_AND_FSYNC = "            fh.flush()\n            os.fsync(fh.fileno())"
ATOMIC_RENAME = "            os.chmod(tmp, 0o600)\n            os.replace(tmp, final)"
TEMP_CLEANUP = ("            try:\n                os.unlink(tmp)\n"
                "            except OSError:\n                pass")
DEFAULT_ROOT = 'return Path(env).expanduser() if env else Path.home() / "oar-data"'
MODE_CHECK = ("    if resolved.exists():\n"
              "        mode = resolved.stat().st_mode & 0o777\n"
              "        if mode != ROOT_MODE:")
REPO_CHECK = "    if resolved == repo or repo in resolved.parents:"
LOCATION_GATE = "        if findings and not _allow_ephemeral:"
RESERVED_GATE = '        for reserved in ("ts", "seq"):'
SORT_KEYS = 'ensure_ascii=False, sort_keys=True) + "\\n"'
JSON_PARSE = "                out.append(json.loads(line))"
SNAPSHOT_NAME = '        if "/" in name or name.startswith("."):'
SYNC_DIR_AFTER_APPEND = "        if touched:\n            self._sync_dir()"
HIGH_WATER = "                    self._publish(high)"


# Every `events` record has to carry a reason from `store.GOALS` (§S4.5), and these
# tests are about the writer rather than the vocabulary — `tests/test_why.py` owns
# that. One named constant instead of a literal per call site, so this file has a
# single place where the enum is read and no second copy of any member.
WHY = {"goal": "user_requested_answer"}


# ------------------------------------------------------- T-A1 single writer (D)
def _produce(store, stream: str, who: int, n: int) -> None:
    for i in range(n):
        store.append(stream, {"who": f"{who:02d}", "n": f"{i:04d}", "pad": PAD,
                              "why": WHY})


@pytest.mark.slow
def test_four_producers_land_exactly_four_thousand_intact_lines(tmp_path):
    """T-A1. Two producers per stream, because audit shares the writer with events.

    The plan scheduled the audit log in S2 and the writer in S4. An audit log with
    its own writer is a second writer, so the writer moved forward and both streams
    are driven here at once — if they did not share, this test is where it shows.
    """
    with make_store(tmp_path) as store:
        threads = [threading.Thread(target=_produce, args=(store, stream, who, 1000))
                   for who, stream in enumerate(("events", "audit", "events", "audit"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        store.flush()

        seqs = []
        for stream in ("events", "audit"):
            lines = store.path_for(stream).read_text(encoding="utf-8").splitlines()
            assert len(lines) == 2000, f"{stream}: {len(lines)} lines, expected 2000"
            records = store.read(stream)          # raises on any unparseable line
            assert len(records) == 2000
            # Byte-exact round trip per line, not a line-width check: `seq` and `ts`
            # are variable-width, so equal widths were never the invariant. Re-dumping
            # catches a splice that happened to parse, and the `pad` comparison catches
            # one that swapped equal-length halves.
            for line, rec in zip(lines, records):
                assert line == json.dumps(rec, ensure_ascii=False, sort_keys=True)
                assert rec["pad"] == PAD, f"{stream} seq {rec['seq']}: payload spliced"
            seqs.extend(r["seq"] for r in records)

    assert sorted(seqs) == list(range(1, 4001)), "sequence numbers are not a bijection"


@pytest.mark.mutation
def test_bypassing_the_single_writer_splices_records(tmp_path):
    """The T-A1 mutant: each producer writes to the file itself.

    Not a text mutant of store.py, because the mutation is architectural — "skip
    the queue" is a different program, not a changed line. What it has to show is
    that jsonl gives no protection of its own, so the previous test's cleanliness is
    attributable to the writer.

    The interleave is explicit. `os.write` per 8192-byte chunk is exactly what a
    buffered file object does with a 8300-byte record; the only thing removed is
    the dependence on which thread the GIL happens to favour.
    """
    path = tmp_path / "bypass.jsonl"
    fds = [os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
           for _ in range(2)]
    try:
        lines = [
            (json.dumps({"who": f"{who:02d}", "n": "0000", "pad": PAD},
                        sort_keys=True) + "\n").encode("utf-8")
            for who in range(2)
        ]
        chunks = [[line[i:i + 8192] for i in range(0, len(line), 8192)]
                  for line in lines]
        assert all(len(c) > 1 for c in chunks), \
            "the record fits in one write(); this mutant would prove nothing"
        for step in range(max(len(c) for c in chunks)):
            for fd, chunk_list in zip(fds, chunks):
                if step < len(chunk_list):
                    os.write(fd, chunk_list[step])
    finally:
        for fd in fds:
            os.close(fd)

    # Read it with the real reader: a partial line must be an error, not a skip.
    store = make_store(tmp_path)
    try:
        os.replace(path, store.path_for("events"))
        with pytest.raises(st.StoreError, match="not valid JSON"):
            store.read("events")
    finally:
        store.close()


@pytest.mark.mutation
def test_a_tolerant_reader_would_hide_the_corruption(tmp_path):
    """Mutant for `read()`'s raise: swallow the decode error and skip the line.

    A reader that skips unparseable lines makes the failure this module exists to
    prevent unobservable — the file gets shorter and nothing anywhere goes red.

    The store is built on a clean root and the file corrupted **afterwards**, because
    `recovered_seq` now refuses a partial line at construction time. Writing the
    corrupt file first would make this pass on the constructor's refusal while never
    calling `read()` at all — the assertion would be green with the mutant applied,
    since both messages contain "not valid JSON". Same reason the mutant cannot be
    constructed over the corrupt root either: it patches `read`, not recovery.
    """
    corrupt = ('{"seq": 1, "ts": 0.0, "kind": "fine"}\n'
               '{"seq": 2, "ts": 0.0, "kind": "trunc\n')
    store = make_store(tmp_path)
    (tmp_path / "data" / "events.jsonl").write_text(corrupt, encoding="utf-8")
    with pytest.raises(st.StoreError, match="two writers"):
        store.read("events")

    mutant_module = load_mutant(tmp_path, (
        JSON_PARSE,
        "                try:\n"
        "                    out.append(json.loads(line))\n"
        "                except Exception:\n"
        "                    pass",
    ))
    mutant = mutant_module.Store(tmp_path / "clean", _allow_ephemeral=True)
    (tmp_path / "clean" / "events.jsonl").write_text(corrupt, encoding="utf-8")
    # The mutant returns the one good record and silently drops the truncated one.
    assert len(mutant.read("events")) == 1


# --------------------------------------------- T-A2 atomic replace, no debris (D)
#
# These use `rps_table.json`, not `profile.json`. The profile card is in
# `PROTECTED_SNAPSHOTS` from S3 onwards, so a snapshot test written against it would
# be exercising the lock rather than the rename — and it would have to unlock first,
# which is a test that quietly proves the lock can be lifted by anybody who finds it
# inconvenient. The RPS table is the other real snapshot caller (Track B) and is not
# locked, so the atomicity claim is tested on a name where atomicity is the only
# thing in the way.
def test_a_snapshot_replaces_the_file_not_its_contents(tmp_path):
    """T-A2. A reader holding the old snapshot keeps reading the old bytes.

    This is the property `os.replace` buys and in-place writing cannot: the rename
    swings the *name* to a new inode, so anybody mid-read finishes on the old one.
    Needs no fault injection, which is why it can be compared against the mutant.
    """
    with make_store(tmp_path) as store:
        store.write_snapshot("rps_table.json", {"name": "old"})
        store.flush()
        path = store.root / "rps_table.json"
        before = path.read_bytes()
        inode_before = path.stat().st_ino

        with path.open("rb") as held:
            store.write_snapshot("rps_table.json", {"name": "new"})
            store.flush()
            assert held.read() == before, "the held reader saw the write land"

        assert json.loads(path.read_text())["name"] == "new"
        assert path.stat().st_ino != inode_before, "the file was rewritten in place"


@pytest.mark.mutation
def test_an_in_place_snapshot_write_is_visible_to_a_held_reader(tmp_path):
    """Mutant for T-A2: `os.replace(tmp, final)` → write the body in place."""
    mutant = load_mutant(tmp_path, (
        ATOMIC_RENAME,
        "            final.write_text(body, encoding='utf-8')",
    ))
    with make_store(tmp_path, module=mutant) as store:
        store.write_snapshot("rps_table.json", {"name": "old"})
        store.flush()
        path = store.root / "rps_table.json"
        before = path.read_bytes()
        with path.open("rb") as held:
            store.write_snapshot("rps_table.json", {"name": "new"})
            store.flush()
            assert held.read() != before, "mutant did not change what the reader sees"


def _fail_the_next_replace(module, monkeypatch) -> None:
    """Make `os.replace` raise, for as short a window as possible.

    `store.py` does `import os`, so the only reachable seam is the os module
    itself — there is no per-instance hook to patch. The window is closed again the
    moment the failure has been observed, because a global `os.replace` that raises
    is a hazard to anything else running in this interpreter.
    """
    def boom(src, dst):
        raise OSError("injected: no space left on device")

    monkeypatch.setattr(module.os, "replace", boom)


def test_a_failed_replace_leaves_the_old_snapshot_and_no_debris(tmp_path, monkeypatch):
    """T-A2's injection half: the old file survives and nothing is left behind.

    A directory slowly filling with `.rps_table.json.*.tmp` is how "the snapshot is
    fine" and "every write has failed since Tuesday" come to look identical.

    No `with` block: the store's writer is dead by the end, so `close()` — which
    flushes — would raise out of `__exit__` and mask the assertions.
    """
    store = make_store(tmp_path)
    store.write_snapshot("rps_table.json", {"name": "old"})
    store.flush()
    before = (store.root / "rps_table.json").read_bytes()

    _fail_the_next_replace(st, monkeypatch)
    store.write_snapshot("rps_table.json", {"name": "new"})
    with pytest.raises(st.StoreError, match="writer thread died"):
        store.flush()
    monkeypatch.undo()

    assert (store.root / "rps_table.json").read_bytes() == before
    debris = [p.name for p in store.root.iterdir() if p.name.endswith(".tmp")]
    assert debris == [], f"temp files survived the failure: {debris}"


@pytest.mark.mutation
def test_without_the_unlink_a_failed_snapshot_leaves_a_temp_file(tmp_path, monkeypatch):
    """Mutant for the debris assertion: drop the `os.unlink(tmp)` on failure."""
    mutant = load_mutant(tmp_path, (TEMP_CLEANUP, "            pass"))
    store = make_store(tmp_path, module=mutant)
    _fail_the_next_replace(mutant, monkeypatch)
    store.write_snapshot("rps_table.json", {"name": "new"})
    with pytest.raises(mutant.StoreError):
        store.flush()
    monkeypatch.undo()
    debris = [p.name for p in store.root.iterdir() if p.name.endswith(".tmp")]
    assert debris, "mutant left no debris, so the assertion proves nothing"


# ------------------------------------------------------------ T-A3 durability
CHILD = """
import os, sys
sys.path.insert(0, {pkg!r})
from memory.store import Store
s = Store({root!r}, _allow_ephemeral=True)
s.append("events", {{"kind": "survives", "why": {{"goal": "user_requested_answer"}}}})
s.flush()
os._exit(0)
"""


def _run_child(pkg: Path, root: Path) -> None:
    proc = subprocess.run([sys.executable, "-c", CHILD.format(pkg=str(pkg),
                                                              root=str(root))],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"child failed: {proc.stdout}\n{proc.stderr}"


def test_a_record_survives_a_child_that_runs_no_cleanup(tmp_path):
    """T-A3, narrowed to what is observable: nothing depends on cleanup running.

    `os._exit(0)` skips `close()`, `__del__` and `atexit`, so a record still in the
    Python buffer is lost. It does *not* skip the page cache, which is why this test
    says nothing about fsync — see the module docstring and the static test below.
    """
    root = tmp_path / "data"
    _run_child(REPO_ROOT, root)
    with st.Store(root, _allow_ephemeral=True) as store:
        records = store.read("events")
    assert [r["kind"] for r in records] == ["survives"]


@pytest.mark.mutation
def test_without_the_buffer_flush_the_record_is_lost(tmp_path):
    """Mutant for T-A3: `fh.flush()` removed, so `flush()` returns a lie."""
    pkg = tmp_path / "pkg"
    (pkg / "memory").mkdir(parents=True)
    src = STORE_PY.read_text()
    assert FLUSH_AND_FSYNC in src
    (pkg / "memory" / "store.py").write_text(
        src.replace(FLUSH_AND_FSYNC, "            os.fsync(fh.fileno())"))

    root = tmp_path / "data"
    _run_child(pkg, root)
    with st.Store(root, _allow_ephemeral=True) as store:
        assert store.read("events") == [], "mutant still made the record durable"


def test_the_fsync_claim_is_asserted_statically_because_ci_cannot_observe_it():
    """The other half of T-A3, and the only honest form for it.

    fsync's guarantee is against power loss. No test that runs on one machine
    without cutting its power can tell fsync from flush, so this asserts the calls
    are on the path — a weaker claim, stated as such, rather than a behavioural test
    that would appear to prove durability while proving buffering.
    """
    tree = ast.parse(STORE_PY.read_text())
    methods = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            methods[fn.name] = fn

    def called(fn) -> set[str]:
        out = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                out.add(ast.unparse(node.func))
        return out

    for name, required in (("_sync", {"os.fsync", "self._sync_dir"}),
                           ("_replace", {"os.fsync", "os.replace", "self._sync_dir"}),
                           ("_sync_dir", {"os.fsync"})):
        assert name in methods, f"memory/store.py has no {name}()"
        missing = required - called(methods[name])
        assert not missing, f"{name}() no longer calls {sorted(missing)}"


@pytest.mark.mutation
@pytest.mark.parametrize("anchor,replacement,gone", [
    (FLUSH_AND_FSYNC, "            fh.flush()", "os.fsync"),
    (SYNC_DIR_AFTER_APPEND, "        if touched:\n            pass", "self._sync_dir"),
])
def test_the_static_fsync_assertion_kills_its_mutants(tmp_path, anchor,
                                                      replacement, gone):
    """Removing either syscall from `_sync` must fail the static assertion."""
    src = STORE_PY.read_text()
    assert anchor in src and src.count(anchor) == 1
    mutated = src.replace(anchor, replacement)
    tree = ast.parse(mutated)
    sync = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_sync")
    calls = {ast.unparse(n.func) for n in ast.walk(sync) if isinstance(n, ast.Call)}
    assert gone not in calls, f"mutant did not remove {gone}"


# ----------------------------------------------------------- T-A4 location (D)
def test_the_default_root_is_acceptable_and_derived_from_the_environment(monkeypatch):
    """T-A4 on the real production path, not on an injected one.

    Injecting the root is what makes the rest of this file runnable; if the check
    only ever saw injected paths it would never see the one that ships.
    """
    monkeypatch.delenv(st.ROOT_ENV, raising=False)
    assert st.location_findings(st.default_root()) == []
    monkeypatch.setenv(st.ROOT_ENV, "/tmp/oar")
    findings = st.location_findings(st.default_root())
    assert any("ephemeral" in f for f in findings), findings


@pytest.mark.parametrize("case,needle", [
    ("ephemeral", "ephemeral"),
    ("repo", "repository tree"),
    ("mode", "mode is 0755"),
])
def test_location_findings_names_each_defect(tmp_path, case, needle):
    if case == "ephemeral":
        path = tmp_path / "data"
    elif case == "repo":
        path = REPO_ROOT / "oar-data"
    else:
        path = tmp_path / "loose"
        path.mkdir()
        os.chmod(path, 0o755)
    findings = st.location_findings(path)
    assert any(needle in f for f in findings), f"{path} → {findings}"


def test_the_constructor_refuses_a_bad_location_and_creates_nothing(tmp_path):
    target = REPO_ROOT / "oar-data-should-never-exist"
    with pytest.raises(st.StoreError, match="repository tree"):
        st.Store(target)
    assert not target.exists(), "a refused store still created its directory"


def test_an_accepted_root_is_created_private(tmp_path):
    with make_store(tmp_path) as store:
        assert store.root.stat().st_mode & 0o777 == st.ROOT_MODE
        store.append("events", {"kind": "x", "why": WHY})
        store.flush()
        assert store.path_for("events").stat().st_mode & 0o777 == 0o600


@pytest.mark.mutation
def test_the_location_mutants_go_red(tmp_path):
    """T-A4's four mutants, each against the assertion that names its defect."""
    control = load_mutant(tmp_path / "c")
    own_repo = Path(control.__file__).resolve().parent.parent
    assert any("repository tree" in f
               for f in control.location_findings(own_repo / "oar-data")), \
        "no-mutant control does not produce the finding; the harness is broken"

    # /tmp/oar — the exact defect in §0-4 of the plan (run_ec2.sh puts runtime
    # files there and /tmp is reaped).
    retargeted = load_mutant(tmp_path / "a", (DEFAULT_ROOT, 'return Path("/tmp/oar")'))
    assert any("ephemeral" in f
               for f in retargeted.location_findings(retargeted.default_root()))

    no_repo_check = load_mutant(tmp_path / "b", (REPO_CHECK, "    if False:"))
    inside = Path(no_repo_check.__file__).resolve().parent.parent / "oar-data"
    assert not any("repository tree" in f
                   for f in no_repo_check.location_findings(inside))

    no_mode_check = load_mutant(tmp_path / "d", (MODE_CHECK, "    if False:\n"
                                                 "        mode = 0\n"
                                                 "        if False:"))
    loose = tmp_path / "loose"
    loose.mkdir()
    os.chmod(loose, 0o755)
    assert not any("mode is" in f for f in no_mode_check.location_findings(loose))

    # The gate itself: the mutant accepts a root the real module refuses, and it is
    # a temp root rather than one inside the repo — a test that has to create a
    # directory in the repo tree to prove a point has created the defect it is
    # testing for.
    ephemeral = tmp_path / "accepted-anyway"
    with pytest.raises(st.StoreError, match="ephemeral"):
        st.Store(ephemeral)
    open_gate = load_mutant(tmp_path / "e", (LOCATION_GATE,
                                             "        if findings and not True:"))
    with open_gate.Store(ephemeral) as store:
        assert store.root.exists(), "mutant did not open the gate"


def test_only_tests_grant_the_ephemeral_exemption():
    """The exemption is a claim about the whole repo, so derive it from the repo.

    A comment saying "tests only" is not enforcement. This walks every Python file
    and looks for the *keyword at a call site* (the parameter's own definition in
    memory/store.py is not a use), so a node that quietly grants itself the
    exemption goes red here rather than in a review nobody runs.
    """
    offenders, granted, scanned = [], [], 0
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                    kw.arg == "_allow_ephemeral" for kw in node.keywords):
                rel = path.relative_to(REPO_ROOT)
                (granted if rel.parts[0] == "tests" else offenders).append(str(rel))
    assert scanned > 10, f"only {scanned} python files scanned — the walk is broken"
    assert granted, "no file passes _allow_ephemeral, so this test proves nothing"
    assert offenders == [], f"the ephemeral exemption is granted outside tests/: {offenders}"


# ------------------------------------------------------- record shape and format
def test_the_store_rejects_what_would_make_the_file_unreadable(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(st.StoreError, match="unknown stream"):
            store.append("episodes", {"kind": "x"})
        with pytest.raises(st.StoreError, match="must be a dict"):
            store.append("events", ["not", "a", "dict"])
        with pytest.raises(st.StoreError, match="not JSON-serialisable"):
            store.append("events", {"kind": object(), "why": WHY})
        for reserved in ("ts", "seq"):
            with pytest.raises(st.StoreError, match=f"'{reserved}' is set by the store"):
                store.append("events", {reserved: 0})
        with pytest.raises(st.StoreError, match="plain file name"):
            store.write_snapshot("../escape.json", {})
        store.flush()
        assert store.read("events") == [], "a rejected record still reached the file"


def test_the_line_format_is_key_sorted_so_a_diff_is_reviewable(tmp_path):
    # `audit` rather than `events` because that stream is exempt from the §S4.5
    # reason requirement, so the asserted line stays a statement about key sorting
    # and nothing else. On `events` the record would have to carry a `why`, and the
    # expected string would then also be asserting the shape of that field — two
    # claims in one literal, where a change to either moves the same line.
    with make_store(tmp_path, clock=lambda: 1.5) as store:
        store.append("audit", {"zebra": 1, "alpha": 2})
        store.flush()
        line = store.path_for("audit").read_text(encoding="utf-8").rstrip("\n")
    assert line == '{"alpha": 2, "seq": 1, "ts": 1.5, "zebra": 1}'


@pytest.mark.mutation
@pytest.mark.parametrize("anchor,replacement,label", [
    (SORT_KEYS, 'ensure_ascii=False, sort_keys=False) + "\\n"', "unsorted keys"),
    (RESERVED_GATE, "        for reserved in ():", "caller can set ts"),
])
def test_the_format_and_rejection_mutants_go_red(tmp_path, anchor, replacement, label):
    mutant = load_mutant(tmp_path, (anchor, replacement))
    with make_store(tmp_path, module=mutant, clock=lambda: 1.5) as store:
        if label == "unsorted keys":
            store.append("audit", {"zebra": 1, "alpha": 2})
            store.flush()
            line = store.path_for("audit").read_text(encoding="utf-8").rstrip("\n")
            assert line != '{"alpha": 2, "seq": 1, "ts": 1.5, "zebra": 1}'
        else:
            store.append("events", {"ts": 99.0, "why": WHY})
            store.flush()
            assert store.read("events")[0]["ts"] == 99.0


@pytest.mark.mutation
def test_without_the_name_check_a_snapshot_escapes_the_root(tmp_path):
    """Mutant for `write_snapshot`'s name check — and a coincidence worth naming.

    The obvious traversal, `../escape.json`, does **not** escape even with the check
    removed: `mkstemp(prefix=f".{name}.")` turns it into `.../escape.json.` and fails
    with ENOENT because no directory named `...` exists. That reads like protection
    and is not — it is a fact about which directories happen to exist, and it
    evaporates for a name whose traversal resolves. `./../escape.json` is such a
    name, and with the check gone the snapshot lands outside the root.

    Which is the argument for the check: an accidental defence that depends on the
    contents of the filesystem is not a defence, and the version of this test that
    used `../escape.json` would have reported the mutant as killed.
    """
    mutant = load_mutant(tmp_path, (SNAPSHOT_NAME, "        if False:"))
    store = make_store(tmp_path, module=mutant)
    escaped = store.root.parent / "escape.json"
    assert not escaped.exists()
    try:
        store.write_snapshot("./../escape.json", {"leaked": True})
        store.flush()
        assert escaped.exists(), "mutant did not escape the root"
    finally:
        store.close()
        escaped.unlink(missing_ok=True)


# ------------------------------------------------------------------- liveness
def test_flush_raises_instead_of_hanging_when_the_writer_dies(tmp_path, monkeypatch):
    """A dead writer must not look like a slow disk.

    The 5 s join is a liveness bound, not a performance budget — the assertion is
    "it terminates", and nothing here compares a duration to a threshold (§3 house
    rule). Without the bound a regression would hang the whole suite instead of
    failing one test.
    """
    def dies(self, touched):
        raise OSError("injected: writer dies here")

    # Patched on the class, not on `os`: a globally raising `os.fsync` is a hazard
    # to every other thread in this interpreter for as long as it is installed.
    monkeypatch.setattr(st.Store, "_sync", dies)
    store = make_store(tmp_path)
    store.append("events", {"kind": "x", "why": WHY})
    box: list[BaseException | None] = []
    worker = threading.Thread(target=lambda: box.append(_capture(store)))
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "flush() never returned after the writer died"
    assert isinstance(box[0], st.StoreError)
    with pytest.raises(st.StoreError, match="writer thread died"):
        store.append("events", {"kind": "y", "why": WHY})


@pytest.mark.mutation
def test_the_writer_publishes_the_batch_high_water_mark(tmp_path):
    """A white-box test for an ordering hazard that only exists under concurrency.

    `append()` takes its sequence number under the lock and enqueues outside it, so
    two producers may enqueue 2 before 1. If the writer published the *last* item it
    drained, `_durable` would sit at 1 with 2 already written, and a `flush()`
    waiting for 2 would never be woken — a hang that presents as a slow disk. The
    reversed enqueue is done directly here because no amount of thread scheduling
    makes it reproducible from the outside.

    The 5 s join is a liveness bound, not a duration budget: the assertion is that
    `flush()` terminates.
    """
    mutant = load_mutant(tmp_path, (HIGH_WATER, "                    self._publish(seq)"))
    for label, module, should_finish in (("real", st, True), ("mutant", mutant, False)):
        store = make_store(tmp_path / label, module=module)
        gate = threading.Event()
        real_handle = store._handle

        def held(stream, _real=real_handle, _gate=gate):
            _gate.wait(timeout=5.0)
            return _real(stream)

        store._handle = held
        try:
            # Enqueued highest-first, which is what two producers racing between
            # "take a sequence number" and "put" produce. The writer is parked
            # inside the first `_handle` call, so both items are certain to be in
            # the same drain batch — no scheduling luck, no flaky mutant.
            store._q.put(("append", "events", '{"n": 2}\n', 2))
            store._q.put(("append", "events", '{"n": 1}\n', 1))
            with store._lock:
                store._seq = 2
            gate.set()
            worker = threading.Thread(target=lambda: _capture(store), daemon=True)
            worker.start()
            worker.join(timeout=5.0)
            assert worker.is_alive() is not should_finish, (
                f"{label}: flush() never returned" if should_finish else
                f"{label}: did not hang, so this test proves nothing")
        finally:
            # Release the hung waiter: it polls the failure flag every 0.5 s, so
            # setting one ends the thread instead of leaving it spinning for the
            # rest of the session.
            store._failure = RuntimeError("test teardown")


def _capture(store) -> BaseException | None:
    try:
        store.flush()
    except BaseException as exc:  # noqa: BLE001 — the value under test
        return exc
    return None


# ------------------------------------- T-C4 the primitive: rewrite_stream (D)
#
# The one way an append-only file gets shorter. Everything here is about the writer
# rather than about the deletion transaction — tests/test_erase.py owns that — and
# the two assertions worth reading twice are the inode one and the in-flight one:
# both name a defect whose only symptom is silence.
def _seqs_on_disk(root: Path, stream: str = "events") -> list[int]:
    text = (root / f"{stream}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line)["seq"] for line in text.splitlines() if line.strip()]


def test_a_rewrite_drops_exactly_the_named_seqs_and_copies_the_rest_byte_for_byte(
        tmp_path):
    """Records go whole or not at all: the survivors are the same bytes.

    Compared line by line against the pre-rewrite text rather than by re-reading the
    parsed records, because a re-serialise-and-compare would agree with itself: the
    writer would produce the same JSON from the same dict even if the rewrite had
    rebuilt every line. `rewrite_stream` is meant to be incapable of altering a
    record's text, and only the raw bytes can show that.
    """
    store = make_store(tmp_path)
    try:
        for i in range(6):
            store.append("events", {"text": f"line {i}", "why": WHY})
        store.flush()
        path = store.path_for("events")
        before = path.read_text(encoding="utf-8").splitlines()
        assert len(before) == 6
        dropped = store.rewrite_stream("events", [2, 4])
        assert dropped == (2, 4)
        after = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in before if json.loads(line)["seq"] not in (2, 4)]
        assert after == kept, "surviving records were re-rendered, not copied"
        assert _seqs_on_disk(store.root) == [1, 3, 5, 6]
    finally:
        store.close()


def test_a_rewrite_reports_what_the_file_held_not_what_it_was_asked_for(tmp_path):
    """Idempotence, and the reason the return value exists.

    A second deletion of the same episode is the state a crash between the tombstone
    and the rewrite leaves, so it has to be a normal call rather than an error. What
    makes it *idempotent* rather than merely repeatable is that the second call
    returns `()`: the caller can tell "already gone" from "just removed" and report
    the difference, which is what memory/erase.py turns into a finding.
    """
    store = make_store(tmp_path)
    try:
        for i in range(3):
            store.append("events", {"text": f"line {i}", "why": WHY})
        store.flush()
        assert store.rewrite_stream("events", [2]) == (2,)
        assert store.rewrite_stream("events", [2]) == ()
        assert store.rewrite_stream("events", [99]) == ()
        assert _seqs_on_disk(store.root) == [1, 3]
    finally:
        store.close()


def test_appends_after_a_rewrite_land_in_the_file_the_readers_open(tmp_path):
    """The inode trap, asserted on the inode.

    `os.replace` puts a new inode behind the name and the kept-open append fd still
    points at the unlinked old one. An append through that fd is accepted, fsynced,
    reported durable by `flush()` — and invisible to every reader forever, with disk
    usage as the only symptom. A record-count assertion would pass on the broken
    version for the first records and fail confusingly later, so the assertion is on
    the identity of the file: the handle the writer holds must be the file the name
    resolves to.
    """
    store = make_store(tmp_path)
    try:
        store.append("events", {"text": "before", "why": WHY})
        store.flush()
        path = store.path_for("events")
        store.rewrite_stream("events", [1])
        assert path.read_text(encoding="utf-8") == ""
        store.append("events", {"text": "after", "why": WHY})
        store.flush()
        held = store._handles["events"]
        assert os.fstat(held.fileno()).st_ino == os.stat(path).st_ino, (
            "the writer is appending into the inode os.replace unlinked")
        assert _seqs_on_disk(store.root) == [3]
    finally:
        store.close()


@pytest.mark.mutation
def test_the_stale_handle_mutant_makes_later_appends_invisible(tmp_path):
    """The mutant for the test above: keep the handle instead of closing it."""
    module = load_mutant(tmp_path, (
        "        fh = self._handles.pop(stream, None)\n"
        "        if fh is not None:\n"
        "            fh.close()",
        "        fh = self._handles.get(stream)\n"
        "        if fh is not None:\n"
        "            fh.flush()"))
    store = make_store(tmp_path, module=module)
    try:
        store.append("events", {"text": "before", "why": WHY})
        store.flush()
        store.rewrite_stream("events", [1])
        store.append("events", {"text": "after", "why": WHY})
        store.flush()
        path = store.path_for("events")
        held = store._handles["events"]
        assert os.fstat(held.fileno()).st_ino != os.stat(path).st_ino
        assert _seqs_on_disk(store.root) == [], (
            "mutant did not reproduce the defect, so the real test proves nothing")
    finally:
        store.close()


def test_a_rewrite_does_not_destroy_a_record_still_in_flight(tmp_path):
    """Checked by looking for the survivor rather than for the target.

    `rewrite_stream` blocks, but the records queued before it may still be in the
    writer's buffer when it runs. If the file on disk is short at that moment the copy
    keeps nothing, and the buffered append is then flushed into an already-replaced
    inode — a record destroyed by a deletion that never named it.

    The assertion is therefore about record 2, which was never a target. Asserting
    only that record 1 is gone passes on every broken version of this.
    """
    store = make_store(tmp_path)
    try:
        store.append("events", {"text": "target", "why": WHY})
        store.append("events", {"text": "bystander", "why": WHY})
        # No flush: both records may still be buffered when the rewrite runs.
        assert store.rewrite_stream("events", [1]) == (1,)
        store.flush()
        assert _seqs_on_disk(store.root) == [2], "an unnamed record was deleted"
    finally:
        store.close()


@pytest.mark.mutation
def test_without_the_flush_and_the_drain_a_rewrite_deletes_a_bystander(tmp_path):
    """A deliberately compound mutant, and what it does and does not show.

    Two independent mechanisms put a buffered append in the file before the copy
    reads it: `_rewrite` closing the handle (close flushes) and `_run` draining the
    batch with `_sync(touched)`. Removing either alone changes nothing observable, so
    a single-line mutant here would survive and read as "the assertion is wrong".
    Removing both reproduces the defect, which shows the property is protected —
    twice — while attributing it to neither line. Stated rather than dressed up as a
    normal one-line mutant, per §6.0 rule 2.
    """
    module = load_mutant(
        tmp_path,
        ("        fh = self._handles.pop(stream, None)\n"
         "        if fh is not None:\n"
         "            fh.close()",
         "        fh = self._handles.pop(stream, None)\n"
         "        if fh is not None:\n"
         "            pass"),
        ("                    self._sync(touched)\n"
         "                    touched.clear()\n"
         "                    dropped = self._rewrite(target, body)",
         "                    touched.clear()\n"
         "                    dropped = self._rewrite(target, body)"))
    store = make_store(tmp_path, module=module)
    try:
        store.append("events", {"text": "target", "why": WHY})
        store.append("events", {"text": "bystander", "why": WHY})
        # Both halves of the defect, and the pairing is the worst case: the target is
        # reported absent, so a receipt would read "already gone" — while the
        # bystander, which no plan named, is the record that actually went.
        assert store.rewrite_stream("events", [1]) == ()
        store.flush()
        assert _seqs_on_disk(store.root) == [], (
            "mutant kept the bystander, so the in-flight test proves nothing")
    finally:
        # The mutant leaks the un-closed handle; `close()` only knows about handles
        # still in `_handles`, and this one was popped.
        store.close()


def test_the_rewrite_branch_drains_the_batch_before_replacing_a_file():
    """The tombstone-before-removal ordering, asserted statically.

    Durability ordering is not observable without cutting power — the same limit as
    T-A3's fsync half — so this is the honest form: the drain is on the path, ahead of
    the rewrite, checked against the AST. What it protects is the ordering
    `memory.erase.delete` depends on: the tombstone is an `audit` append, and it has
    to be on the disk before the file it describes is replaced. A behavioural test
    here would be a test of `flush()`, which the erasure calls anyway, and would go
    green on a writer that had inverted the order.
    """
    tree = ast.parse(STORE_PY.read_text())
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_run")
    branch = None
    for node in ast.walk(run):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "op == 'rewrite'":
            branch = node.body
    assert branch is not None, "memory/store.py has no rewrite branch in _run()"
    calls = [ast.unparse(n.func) for stmt in branch for n in ast.walk(stmt)
             if isinstance(n, ast.Call)]
    assert "self._sync" in calls and "self._rewrite" in calls, calls
    assert calls.index("self._sync") < calls.index("self._rewrite"), (
        "the batch is drained after the file is replaced, so a tombstone can reach "
        "the disk later than the removal it describes")


def test_the_audit_stream_may_not_be_rewritten(tmp_path):
    """`REWRITABLE["audit"] is False`, enforced in the one writer.

    The consequence is the point: an audit log the deletion path can shorten cannot
    hold the record that a deletion happened, because the next deletion could remove
    it. Checked here rather than in memory/erase.py, since any of the nodes can reach
    the store without importing the erasure.
    """
    store = make_store(tmp_path)
    try:
        store.append("audit", {"kind": "tombstone"})
        store.flush()
        with pytest.raises(st.StoreError, match="not an audit log"):
            store.rewrite_stream("audit", [1])
        assert _seqs_on_disk(store.root, "audit") == [1]
    finally:
        store.close()


@pytest.mark.parametrize("stream,drop,needle", [
    ("nope", [1], "unknown stream"),
    ("events", [1.5], "sequence numbers"),
    ("events", [True], "sequence numbers"),
    ("events", ["1"], "sequence numbers"),
])
def test_rewrite_refuses_what_it_cannot_address_by_identity(tmp_path, stream, drop,
                                                            needle):
    """A bad target is refused before a sequence number is taken.

    `True` is in the list on purpose: `isinstance(True, int)` is true in Python, so a
    bool would slip through a plain int check and match `seq == 1` — a deletion
    addressed by an accident of the type system.
    """
    store = make_store(tmp_path)
    try:
        store.append("events", {"text": "keep", "why": WHY})
        store.flush()
        before = store._seq
        with pytest.raises(st.StoreError, match=needle):
            store.rewrite_stream(stream, drop)
        assert store._seq == before, (
            "a refused rewrite consumed a sequence number, so it left a gap that "
            "looks exactly like a deletion")
        assert _seqs_on_disk(store.root) == [1]
    finally:
        store.close()


def test_a_rewrite_refuses_a_partial_line_instead_of_deleting_it(tmp_path):
    """Corruption is reported, not quietly repaired — and it stops the writer.

    A rewrite that skipped an unparseable line would *delete* it, making the deletion
    path the one place that silently fixes a torn write. The torn write is the
    evidence of a second writer, so both halves are asserted: the caller is told, and
    the store stops. The second half is the consequence and it is checked rather than
    discovered later — one torn line ends this store's logging, including audit,
    until a person looks at the file.
    """
    store = make_store(tmp_path)
    store.append("events", {"text": "good", "why": WHY})
    store.flush()
    with store.path_for("events").open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "text": "tor')
    with pytest.raises(st.StoreError, match="not valid JSON"):
        store.rewrite_stream("events", [1])
    assert '"good"' in store.path_for("events").read_text(encoding="utf-8")
    # No `close()` in a finally: the writer is gone and closing re-raises. Every
    # later call has to say so rather than accepting a record it cannot write.
    with pytest.raises(st.StoreError, match="writer thread died"):
        store.append("events", {"text": "after", "why": WHY})
    with pytest.raises(st.StoreError, match="writer thread died"):
        store.close()


def test_a_rewrite_of_an_absent_file_is_empty_rather_than_an_error(tmp_path):
    """Nothing on disk means nothing was removed, and it is not a failure.

    A deletion runs against whatever the store happens to hold; a store that has
    never written an event is a real state, and raising here would make an erasure on
    a fresh root fail instead of reporting zero.
    """
    store = make_store(tmp_path)
    try:
        assert store.rewrite_stream("events", [1, 2]) == ()
        assert not store.path_for("events").exists()
    finally:
        store.close()


# ---------------------------------------- the sequence counter survives a restart
#
# Not in docs/test-design.md §2.A, and it belongs with T-A1 rather than beside it:
# the single-writer guarantee is about one process at a time, and this is what makes
# the *sequence numbers* unique across processes. It became load-bearing when
# `rewrite_stream` arrived, because a deletion addresses records by seq.
def test_a_reopened_store_resumes_the_sequence_counter(tmp_path):
    """Two sessions, no repeated sequence number.

    The assertion is on the set of seqs in the file rather than on the counter, since
    the defect's whole symptom is two records sharing a number: `_seq` could be
    restored correctly and still be spent by a path that does not use it.
    """
    first = make_store(tmp_path)
    try:
        for i in range(3):
            first.append("events", {"text": f"a{i}", "why": WHY})
        first.flush()
    finally:
        first.close()
    second = st.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        for i in range(2):
            second.append("events", {"text": f"b{i}", "why": WHY})
        second.flush()
        seqs = _seqs_on_disk(second.root)
        assert seqs == [1, 2, 3, 4, 5], seqs
        assert len(set(seqs)) == len(seqs), "a sequence number was issued twice"
    finally:
        second.close()


def test_the_counter_is_shared_across_streams_on_recovery(tmp_path):
    """An `audit`-only session still moves the number `events` uses next.

    One counter issues both streams, so recovering from `events` alone would re-issue
    every number an audit-heavy session spent — and the tombstone is an audit record,
    which makes this the deletion transaction's own failure mode.
    """
    first = make_store(tmp_path)
    try:
        first.append("audit", {"kind": "guard"})
        first.append("audit", {"kind": "guard"})
        first.flush()
    finally:
        first.close()
    assert st.recovered_seq(tmp_path / "data") == 2
    second = st.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        assert second.append("events", {"text": "after", "why": WHY}) == 3
        second.flush()
        assert _seqs_on_disk(second.root) == [3]
    finally:
        second.close()


def test_a_restart_after_a_deletion_does_not_reuse_the_deleted_number(tmp_path):
    """The interaction that makes this a correctness bug rather than untidiness.

    A rewrite leaves a gap, and a counter recovered as "max on disk" would reissue the
    numbers above the gap if the highest records were the ones deleted. A reissued 3
    would make a later `rewrite_stream([3])` delete a record from a different session,
    and the tombstone naming 3 would then describe two removals.

    The arithmetic is worth following, because it shows the documented lossiness:
    events take 1, 2, 3; the rewrite consumes **4** while writing no record; the audit
    tombstone takes 5. Recovery sees 5 (from `audit.jsonl`, not from the events file it
    just shortened) and the next session starts at 6 — 4 is never reused because
    nothing on disk needs it to be, and 3 is never reused because the audit record
    outlives the event it removed.
    """
    first = make_store(tmp_path)
    try:
        for i in range(3):
            first.append("events", {"text": f"a{i}", "why": WHY})
        first.flush()
        assert first.rewrite_stream("events", [3]) == (3,)
        first.append("audit", {"kind": "tombstone", "removed": [3]})
        first.flush()
    finally:
        first.close()
    assert st.recovered_seq(tmp_path / "data") == 5
    second = st.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        assert second.append("events", {"text": "later", "why": WHY}) == 6
        second.flush()
        assert _seqs_on_disk(second.root) == [1, 2, 6]
    finally:
        second.close()


@pytest.mark.mutation
def test_the_zero_start_mutant_reissues_a_live_sequence_number(tmp_path):
    """The mutant: `self._seq = 0`, which is what this module shipped with."""
    module = load_mutant(tmp_path, (
        "        self._seq = recovered_seq(self.root)",
        "        self._seq = 0"))
    first = make_store(tmp_path, module=module)
    try:
        first.append("events", {"text": "old", "why": WHY})
        first.flush()
    finally:
        first.close()
    second = module.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        second.append("events", {"text": "new", "why": WHY})
        second.flush()
        seqs = _seqs_on_disk(second.root)
        assert seqs == [1, 1], seqs
        # And the consequence, spelled out: one named target, two records removed.
        assert second.rewrite_stream("events", [1]) == (1, 1)
    finally:
        second.close()


def test_recovery_refuses_a_partial_line_rather_than_guessing(tmp_path):
    """A torn line stops the boot instead of silently resuming below it.

    Resuming at 0 from a file whose lines cannot all be read is the defect above with
    a corrupt file as its cause, so this is refused for the same reason `read` refuses
    it. The cost is stated in `recovered_seq`: a power cut can tear the final line and
    this treats that like a second writer.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "events.jsonl").write_text(
        '{"seq": 1, "text": "good"}\n{"seq": 2, "text": "tor', encoding="utf-8")
    with pytest.raises(st.StoreError, match="partial line"):
        st.recovered_seq(root)
    with pytest.raises(st.StoreError, match="partial line"):
        st.Store(root, _allow_ephemeral=True)


def test_recovery_refuses_a_record_that_cannot_bound_the_counter(tmp_path):
    """A line with no usable seq is refused, not skipped.

    Skipping it would recover a number below one already in use whenever the missing
    seq was the highest — the same collision, reached by tolerating a record instead
    of a byte.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "events.jsonl").write_text('{"seq": 1}\n{"text": "no seq"}\n',
                                       encoding="utf-8")
    with pytest.raises(st.StoreError, match="cannot bound the counter"):
        st.recovered_seq(root)


def test_opening_a_populated_store_and_flushing_nothing_returns(tmp_path):
    """The liveness half of recovery, and it is not obvious from either variable.

    `flush()` defaults to waiting for `self._seq`, and recovery moves that number
    without any writer having published it. So a store opened over a non-empty root
    and closed without a single write waits for a number no thread will ever publish —
    a hang on the most ordinary path there is (a node that boots, reads, and shuts
    down). Asserting on `flush()` returning rather than on `_durable`, since the
    variable could be right while the wait is computed from something else.
    """
    first = make_store(tmp_path)
    try:
        first.append("events", {"text": "old", "why": WHY})
        first.flush()
    finally:
        first.close()
    second = st.Store(tmp_path / "data", _allow_ephemeral=True)
    finished = threading.Event()
    threading.Thread(target=lambda: (second.flush(), finished.set()),
                     daemon=True).start()
    assert finished.wait(timeout=5.0), "flush() on a freshly opened store hung"
    second.close()
