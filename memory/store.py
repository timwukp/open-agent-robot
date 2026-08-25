"""memory/store.py — the single writer, and the only code that touches the disk.

Zero ROS, zero boto3, zero network. The path and the clock are injected, so the
whole thing runs under pytest on a laptop; nothing here needs the instance.

**Why a writer thread rather than "just append, it's O_APPEND".**
Four nodes wanting to record events is four processes' worth of interleaving, and
the usual reassurance ("append writes are atomic") does not cover this case: a
Python file object flushes at its buffer boundary, so one logical 9 KB record is
already two `write()` calls before the kernel sees it. Two producers then produce
a *fragment* of one record followed by a fragment of another, and the corruption
is invisible until something reads the file back weeks later. So exactly one
thread in exactly one process performs every write, and everyone else enqueues.
`tests/test_store.py` builds the bypass version and shows it truncating lines.

**Why the audit log shares this writer.**
The plan gave S2 an audit log and put the writer in S4, which cannot work: an
audit log with its own writer *is* a second writer, and the invariant this module
exists to hold would be broken by the first feature that depends on it. So the
streams are a closed enum and the writer is one thread. What S4 adds is episodes,
not a second way to write.

**What durability means here, precisely.**
`flush()` guarantees the record has been handed to the kernel *and* fsynced, so
another process can read it and a power loss will not lose it. The test suite can
only prove the first half. Killing a process with `os._exit` does **not** empty the
page cache — the data stays visible to every other process — so no in-CI test can
tell `fsync` apart from `flush`. The behavioural test therefore asserts what it can
(a record survives an exit that runs no cleanup, i.e. nothing depends on
`close()`, `__del__` or `atexit`), and the fsync claim is asserted *statically*
against this source. Two claims, two assertions; pretending one test covers both
would be the more comfortable lie.
"""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import IO


class StoreError(Exception):
    """Raised for a rejected record, a bad location, or a dead writer thread."""


class ReadOnlyError(StoreError):
    """Raised by any write against a read-only seed file (design item T-A5)."""


# Closed enum, like `why.goal` and the semantic predicates. A free-form stream
# name would let a caller invent a file nothing knows how to retain or delete, and
# everything written here is personal data with a retention period attached.
STREAMS = ("events", "audit")

# Snapshot names nothing in this repo may write yet (design item T-A5).
#
# This is the read-only guarantee for the S3 profile card, and it is *here* rather
# than in `memory/profile.py` on purpose. The plan phrases T-A5 as "call any write
# API on the seed card and it raises", but the write API is `write_snapshot` on
# this class — a profile module that simply declines to call it constrains only the
# callers that go through it, and any of the nine nodes can reach the store
# directly. A guarantee enforced in the reader is a comment; enforced in the one
# writer, it holds for code that has never heard of the profile module.
#
# Why the card is locked at all: a semantic fact must carry `source_episode_ids`
# so that deleting an episode deletes what it produced, and there are no episodes
# until S4. A card written now would hold facts that no deletion can reach — see
# plan §0-5, which is the defect the build order would otherwise land.
PROTECTED_SNAPSHOTS = frozenset({"profile.json"})

# Lifted by S5, which is when automatic writes become correct (an episode store
# exists, so a written fact can name its sources). A named function rather than a
# mutable module global: `store.PROTECTED_SNAPSHOTS = frozenset()` anywhere in the
# tree would be a silent, greppable-only unlock, whereas a call site is a line a
# reviewer sees in a diff and `tests/test_profile.py` can assert nothing outside
# tests/ contains.
_UNLOCKED: set[str] = set()


def unlock_for_s5(name: str) -> None:
    """Allow snapshot writes to `name`. The only sanctioned way past the lock.

    Not a context manager and not reversible on purpose: a scoped unlock invites
    "just for this one write", which is how a lock stops describing the system. When
    S5 lands, it calls this once at startup and the audit log records that it did.
    """
    if name not in PROTECTED_SNAPSHOTS:
        raise StoreError(f"{name!r} is not a protected snapshot; the protected set "
                         f"is {sorted(PROTECTED_SNAPSHOTS)}")
    _UNLOCKED.add(name)


def is_locked(name: str) -> bool:
    return name in PROTECTED_SNAPSHOTS and name not in _UNLOCKED

# Where the data lives in production. Overridable by env var for the instance,
# because the home directory differs between the laptop and the AMI's user — but
# the override is validated exactly like the default, so it cannot be used to
# move the data somewhere that gets reaped.
ROOT_ENV = "OAR_DATA_DIR"

# Directory mode is set on the root itself, not left to `makedirs(mode=...)`:
# that argument applies to the leaf only, so a run that has to create two levels
# would leave the parent world-readable.
ROOT_MODE = 0o700

_EPHEMERAL_PREFIXES = ("/tmp", "/var/tmp", "/var/folders", "/dev/shm")


def _ephemeral_roots() -> list[Path]:
    """The ephemeral directories, **resolved**, plus whatever tempfile says.

    Resolving matters and is not cosmetic: on macOS `/tmp` is a symlink to
    `/private/tmp` and `/var/folders` to `/private/var/folders`, so a literal
    string prefix check passes happily on the resolved path that the constructor
    actually uses. That is a check that only works on the platform it was written
    on, which for this repo means it works in CI and not on the instance, or the
    reverse.
    """
    roots = []
    for prefix in (*_EPHEMERAL_PREFIXES, tempfile.gettempdir()):
        p = Path(prefix)
        if p.exists():
            roots.append(p.resolve())
    return roots


def default_root() -> Path:
    """The production data directory. Never inside the repo, never under /tmp.

    `run_ec2.sh` puts runtime files in `/tmp/oar`, which is correct for logs and
    pid files and wrong for the only copy of what somebody told the robot: /tmp is
    reaped. That is why this is a function with a test on it rather than a constant
    someone can retarget in one character.
    """
    env = os.environ.get(ROOT_ENV)
    return Path(env).expanduser() if env else Path.home() / "oar-data"


def location_findings(path: Path) -> list[str]:
    """Everything wrong with `path` as a home for durable personal data.

    A list rather than a bool, and a *pure function* rather than an assertion
    inside the constructor, so the node can log each finding and the test can
    assert on the finding text instead of on an exit code — a rejection for the
    right reason and a rejection for the wrong one are otherwise the same
    observation.
    """
    findings: list[str] = []
    resolved = path.expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    if resolved == repo or repo in resolved.parents:
        findings.append(
            f"inside the repository tree ({repo}) — memory content must never "
            "become a blob, and a path under the repo will be pushed by the first "
            "person who runs a wildcard add")
    for root in _ephemeral_roots():
        if resolved == root or root in resolved.parents:
            findings.append(
                f"under an ephemeral directory ({root}) — it is reaped on reboot "
                "and this is the only copy")
            break
    if resolved.exists():
        mode = resolved.stat().st_mode & 0o777
        if mode != ROOT_MODE:
            findings.append(f"mode is {mode:04o}, expected {ROOT_MODE:04o} "
                            "— this directory holds what a person said out loud")
    return findings


class Store:
    """The one object allowed to write. One writer thread, N append-only streams.

    `_allow_ephemeral` exists for the test suite, which necessarily works in
    `tmp_path`. It is a reviewable argument rather than a comment about where the
    check need not look, and `tests/test_store.py` derives from the source that no
    file outside `tests/` passes it — an exemption nobody can grant quietly.
    """

    def __init__(self, root: Path | None = None, *, clock=time.time,
                 _allow_ephemeral: bool = False) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None \
            else default_root().resolve()
        self._clock = clock
        findings = location_findings(self.root)
        if findings and not _allow_ephemeral:
            raise StoreError(f"refusing to store data in {self.root}: "
                             + "; ".join(findings))
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, ROOT_MODE)

        self._q: queue.Queue = queue.Queue()
        self._handles: dict[str, IO[str]] = {}
        self._seq = 0
        self._durable = 0
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self._progress = threading.Condition(self._lock)
        self._writer = threading.Thread(target=self._run, name="oar-store-writer",
                                        daemon=True)
        self._writer.start()

    # ----------------------------------------------------------- producer side
    def append(self, stream: str, record: dict) -> int:
        """Enqueue one record. Returns its sequence number; does not block on disk.

        Serialisation happens **here**, in the caller's thread, so a record that
        cannot be encoded raises at the call site instead of killing the writer
        thread three records later — a producer that gets no exception assumes it
        was recorded.
        """
        if stream not in STREAMS:
            raise StoreError(f"unknown stream {stream!r}; the streams are {STREAMS}")
        if not isinstance(record, dict):
            raise StoreError(f"a record must be a dict, got {type(record).__name__}")
        for reserved in ("ts", "seq"):
            if reserved in record:
                raise StoreError(
                    f"{reserved!r} is set by the store; a caller-supplied value "
                    "would make the ordering of the file a claim nobody checks")
        self._raise_if_failed()
        # Everything that can fail happens **before** a sequence number is taken,
        # and it costs a second `json.dumps` to arrange. Two reasons, both worth the
        # microseconds: a rejected record that consumed a number leaves a gap in the
        # audit log, and a gap is indistinguishable from a deleted record — exactly
        # the ambiguity an audit log exists to remove. And the number would never be
        # published, so the next `flush()` would wait for it forever and present a
        # rejected record as a hung disk.
        try:
            ts = float(self._clock())
            json.dumps(record, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise StoreError(f"record is not JSON-serialisable: {exc}") from exc
        with self._lock:
            self._seq += 1
            seq = self._seq
        line = json.dumps({"seq": seq, "ts": ts, **record},
                          ensure_ascii=False, sort_keys=True) + "\n"
        self._q.put(("append", stream, line, seq))
        return seq

    def write_snapshot(self, name: str, obj) -> int:
        """Replace a whole file atomically (profile card, semantic layer, RPS table).

        Goes through the same queue as the appends: a snapshot written by another
        thread would be a second writer touching the same directory, and the
        `os.replace` is only atomic with respect to readers, not with respect to a
        concurrent writer of the same temp name.
        """
        if "/" in name or name.startswith("."):
            raise StoreError(f"snapshot name {name!r} must be a plain file name")
        # Before `_raise_if_failed`, so a locked write is refused for the reason it
        # was refused even when the writer thread has already died: two different
        # causes reported as one message is how a lock gets removed to "fix" a disk.
        if is_locked(name):
            raise ReadOnlyError(
                f"{name!r} is a read-only seed until S5: a fact written now would "
                "have no source episode (episodes arrive in S4) and no deletion "
                "could ever reach it. memory.store.unlock_for_s5 is the one way past")
        self._raise_if_failed()
        try:
            body = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        except (TypeError, ValueError) as exc:
            raise StoreError(f"snapshot is not JSON-serialisable: {exc}") from exc
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._q.put(("snapshot", name, body, seq))
        return seq

    def flush(self, seq: int | None = None) -> None:
        """Block until `seq` (default: everything enqueued so far) is durable."""
        with self._lock:
            target = self._seq if seq is None else seq
        while True:
            with self._progress:
                self._raise_if_failed()
                if self._durable >= target:
                    return
                # A timeout rather than an indefinite wait: if the writer dies
                # between the failure check and the wait, an untimed wait would
                # hang the caller forever and look like a slow disk.
                self._progress.wait(timeout=0.5)
            if not self._writer.is_alive():
                self._raise_if_failed()
                raise StoreError("the writer thread is gone and the record is not "
                                 "durable — refusing to report success")

    def close(self) -> None:
        self.flush()
        self._q.put(("stop", None, None, self._seq))
        self._writer.join(timeout=5.0)
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._handles.clear()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----------------------------------------------------------- reader side
    def path_for(self, stream: str) -> Path:
        if stream not in STREAMS:
            raise StoreError(f"unknown stream {stream!r}; the streams are {STREAMS}")
        return self.root / f"{stream}.jsonl"

    def read(self, stream: str) -> list[dict]:
        """Every record in `stream`. Raises on a line that does not parse.

        Raising is the point: a truncated line is the failure this module is built
        to prevent, and a reader that skipped unparseable lines would make that
        failure unobservable — which is how a corrupt log survives to production.
        """
        path = self.path_for(stream)
        if not path.exists():
            return []
        out = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StoreError(
                    f"{path.name} line {n} is not valid JSON ({exc}) — a partial "
                    f"line means two writers, {len(line)} bytes: {line[:80]!r}…"
                ) from exc
        return out

    # ------------------------------------------------------------ writer side
    def _run(self) -> None:
        touched: set[str] = set()
        high = 0
        while True:
            op, target, body, seq = self._q.get()
            try:
                if op == "stop":
                    self._sync(touched)
                    return
                if op == "append":
                    fh = self._handle(target)
                    fh.write(body)
                    touched.add(target)
                elif op == "snapshot":
                    self._replace(target, body)
                # The **high-water mark of the batch**, not the last item drained.
                # `append` takes its sequence number under the lock and then enqueues
                # outside it, so two producers can enqueue 6 before 5; publishing the
                # last item's seq would leave `_durable` at 5 with 6 already on disk
                # and a `flush()` waiting for 6 would never be woken. That is a hang,
                # not a slow disk, and it only appears under concurrency — see
                # tests/test_store.py::test_the_writer_publishes_the_batch_high_water_mark.
                high = max(high, seq)
                if self._q.empty():
                    # Batch the fsync to the point the queue drains: per-record
                    # fsync would make a burst of events cost one disk round trip
                    # each, and `flush()` is what a caller uses when it needs the
                    # guarantee now.
                    self._sync(touched)
                    touched.clear()
                    self._publish(high)
            except BaseException as exc:                          # noqa: BLE001
                with self._progress:
                    self._failure = exc
                    self._progress.notify_all()
                return
            finally:
                self._q.task_done()

    def _handle(self, stream: str) -> IO[str]:
        fh = self._handles.get(stream)
        if fh is None:
            path = self.path_for(stream)
            # Opened once and kept open: reopening per record would make every
            # append an O_CREAT|O_APPEND syscall pair, and the mode is set here so
            # a file created on the instance is not group-readable.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            fh = os.fdopen(fd, "a", encoding="utf-8")
            self._handles[stream] = fh
        return fh

    def _replace(self, name: str, body: str) -> None:
        """Write `body` to a temp file in the same directory, then rename over.

        Same directory because `os.replace` is only atomic within a filesystem,
        and `/tmp` is a different one on the instance. The old file is untouched
        until the rename, so a crash anywhere before it leaves the previous
        snapshot complete rather than a half-written one.
        """
        final = self.root / name
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, final)
        except BaseException:
            # The temp file must not outlive the failure: a directory accumulating
            # `.profile.json.*.tmp` is how "the snapshot is fine" and "every write
            # has failed since Tuesday" come to look the same.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._sync_dir()

    def _sync(self, touched: set[str]) -> None:
        for stream in touched:
            fh = self._handles[stream]
            fh.flush()
            os.fsync(fh.fileno())
        if touched:
            self._sync_dir()

    def _sync_dir(self) -> None:
        """fsync the directory itself, so a *rename* survives too.

        Syncing the file's contents does not make its name durable; the rename
        lives in the directory. Omitting this is the classic version of "the data
        was fine, the file was missing".
        """
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _publish(self, seq: int) -> None:
        with self._progress:
            self._durable = max(self._durable, seq)
            self._progress.notify_all()

    def _raise_if_failed(self) -> None:
        exc = self._failure
        if exc is not None:
            raise StoreError(f"the writer thread died: {exc!r}") from exc
