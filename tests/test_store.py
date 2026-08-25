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


# ------------------------------------------------------- T-A1 single writer (D)
def _produce(store, stream: str, who: int, n: int) -> None:
    for i in range(n):
        store.append(stream, {"who": f"{who:02d}", "n": f"{i:04d}", "pad": PAD})


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
    """
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "events.jsonl").write_text(
        '{"seq": 1, "ts": 0.0, "kind": "fine"}\n{"seq": 2, "ts": 0.0, "kind": "trunc\n',
        encoding="utf-8")

    with pytest.raises(st.StoreError, match="not valid JSON"):
        make_store(tmp_path).read("events")

    mutant = load_mutant(tmp_path, (
        JSON_PARSE,
        "                try:\n"
        "                    out.append(json.loads(line))\n"
        "                except Exception:\n"
        "                    pass",
    ))
    # The mutant returns the one good record and silently drops the truncated one.
    assert len(mutant.Store(tmp_path / "data", _allow_ephemeral=True)
               .read("events")) == 1


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
s.append("events", {{"kind": "survives"}})
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
        store.append("events", {"kind": "x"})
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
            store.append("events", {"kind": object()})
        for reserved in ("ts", "seq"):
            with pytest.raises(st.StoreError, match=f"'{reserved}' is set by the store"):
                store.append("events", {reserved: 0})
        with pytest.raises(st.StoreError, match="plain file name"):
            store.write_snapshot("../escape.json", {})
        store.flush()
        assert store.read("events") == [], "a rejected record still reached the file"


def test_the_line_format_is_key_sorted_so_a_diff_is_reviewable(tmp_path):
    with make_store(tmp_path, clock=lambda: 1.5) as store:
        store.append("events", {"zebra": 1, "alpha": 2})
        store.flush()
        line = store.path_for("events").read_text(encoding="utf-8").rstrip("\n")
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
            store.append("events", {"zebra": 1, "alpha": 2})
            store.flush()
            line = store.path_for("events").read_text(encoding="utf-8").rstrip("\n")
            assert line != '{"alpha": 2, "seq": 1, "ts": 1.5, "zebra": 1}'
        else:
            store.append("events", {"ts": 99.0})
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
    store.append("events", {"kind": "x"})
    box: list[BaseException | None] = []
    worker = threading.Thread(target=lambda: box.append(_capture(store)))
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "flush() never returned after the writer died"
    assert isinstance(box[0], st.StoreError)
    with pytest.raises(st.StoreError, match="writer thread died"):
        store.append("events", {"kind": "y"})


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
