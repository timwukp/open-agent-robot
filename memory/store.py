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

**Why `why.goal` becomes mandatory in this stage and not in the one that needs it.**
S5 is the stage that reads the goal, so the tempting order is to add the field when
something consumes it. But a field is mandatory for free exactly once: before the
first writer exists. Today `memory_node` appends to `audit` only, so making a reason
compulsory on `events` changes no production call site at all — it changes twelve
lines of test suite. Adding the same requirement after S5 ships would instead be a
migration over a file of personal data that cannot be re-derived, and the pressure
in that moment is to make the field optional "for the old records", which is the
version where the boundary never existed. Plan §S4.5 put this stage here; what was
measurable is that the cost of putting it here is zero and the cost of putting it
one stage later is unbounded.

**Why an append-only log has a way to get shorter.**
`rewrite_stream` looks like the opposite of everything above, and the resolution is
that "append-only" is a property of the *turn loop*, not a property of the file for
all time. Nothing in a conversation may edit history; a deletion transaction must,
because 「忘記我說的 X」 has to remove the sentence from the file that holds it and a
tombstone that leaves the text on disk is retrieval suppression wearing deletion's
name — the thing plan §1 refused when it refused Ebbinghaus decay as a right to
erasure. So the shortening exists, it is one method, it goes through the same single
writer, it can only drop whole records addressed by sequence number (never alter
one), and `REWRITABLE` says which streams it may touch. The seq gaps it leaves are
what `memory/erase.py`'s tombstones exist to account for: after this method exists,
"a gap means a deletion" is only true if every gap has a tombstone naming it, which
is a two-directional check rather than a comment.

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

# --------------------------------------------------------- why.goal (plan §S4.5)
#
# The closed enumeration three comments in this tree already cite as the example of
# one — the `STREAMS` comment directly above, `memory/segment.py`'s `KINDS`,
# `safety/guard.py`'s enums — and which design item T-C1 lists a free-text `goal`
# in the rejection set for. Until this tuple existed, the invariant those comments
# assert had no creator anywhere in the repo.
#
# **Why the plan's own notation had to be expanded before it could be enforced.**
# §S4.5 writes the families as `user_requested_*` / `game_*` / `vision_event` /
# `scheduled_*`, and a `*` is not a closed set. Implemented as a prefix test,
# `user_requested_ignore_stop_words` is accepted and the field becomes exactly the
# hole it was added to close — the free text returns as the suffix. A prefix is the
# right shape for a *scanner*, where failing to notice a new name is the failure
# mode; it is the wrong shape for a *gate*, where admitting one is. Same pattern,
# opposite direction, because the failure lands on the other side of the test.
#
# So the families are documentation and the members are the contract, and the gate
# below contains no prefix test at all — not even to build a friendlier error
# message. A mutant therefore has to *add* code rather than change it. Note that the
# claim is about `check_why` and is asserted by walking that function's AST, not by
# grepping this file: `write_snapshot` uses `startswith` legitimately on a file name,
# so a file-level grep would need an allowlist, and an allowlist that can be taught
# one exception can be taught the second one quietly.
# `tests/test_why.py` derives the four family names back out of the members, so a
# family that loses its last member reds instead of quietly ceasing to exist.
#
# **Why it is nested under `why` rather than a flat `goal` field.**
# `memory/profile.py` already has a `goal` *predicate*, and its value is a sentence
# the person said out loud (`memory/profile.example.json`: 練粵語聽力). A flat field
# would give a user-authored string and a closed enum the same key in two files that
# S5 joins, and the first join would mix them with nothing anywhere to notice. That
# collision is asserted disjoint in the tests rather than avoided by naming
# discipline — 練粵語聽力 is a row in the frozen rejection set.
GOALS = (
    # user_requested_* — the person asked for this, in words.
    "user_requested_answer",     # an ordinary turn they opened
    "user_requested_remember",   # 「記住這件事」
    "user_requested_forget",     # 「忘記我說的 X」 — the deletion transaction
    "user_requested_recall",     # 「我上次說什麼」
    "user_requested_stop",       # the L0 stop reflex fired on their word
    # game_* — the round loop, the one place a reward is measured by a sensor.
    "game_start",
    "game_round",
    "game_stop",
    # vision_event — a detector fired with nobody speaking.
    "vision_event",
    # scheduled_* — nothing in the room caused it.
    "scheduled_consolidation",   # S5's nightly summarise-and-embed pass
    "scheduled_retention",       # the retention-period sweep, which deletes
)

# Which streams must carry a reason. Written out for every member of `STREAMS`
# rather than looked up with a default, because `WHY_REQUIRED.get(stream, False)`
# lets the *next* stream default to exempt, and the decision nobody made is then
# indistinguishable from the decision to exempt it.
#
# `audit` is exempt deliberately. A guard's audit record has no reason of its own —
# the event it is about has one. Demanding one would make callers invent a member,
# and an invented enum value is worse than an absent field: it is indistinguishable
# from a real one, so the enum silently acquires an "unknown" member spelled like a
# goal, and every census and retention sweep counts it as real.
WHY_REQUIRED = {"events": True, "audit": False}

# Which streams a deletion may shorten (design item T-C4, plan §3 S5).
#
# Written out for every member of `STREAMS` for the same reason `WHY_REQUIRED` is,
# and the `audit` entry is the substantive half. A right-to-erasure transaction has
# to remove what a person said from the file that holds it — a tombstone that leaves
# the text on disk is retrieval suppression, which plan §1 already refused under its
# other name ("Ebbinghaus decay is not deletion"). So `events` is rewritable.
#
# `audit` is not, and this is the one place that decision can be enforced: an audit
# log the deletion path can rewrite is not an audit log, because the record that a
# deletion happened would be removable by the next deletion. The consequence used to
# be stated and left open here — see `memory/erase.py`'s `RETAINED` — because nothing
# enforced that an audit record omits user text. `AUDIT_FIELDS` below closes it.
REWRITABLE = {"events": True, "audit": False}

# What an `audit` record may contain, per record kind (closes the gap `REWRITABLE`
# names above; plan §3 S2's audit obligation and T-C4's "no trace of E").
#
# The argument is `REWRITABLE["audit"] is False`: this is the one stream a deletion
# cannot rewrite, so whatever an audit record carries is a copy that outlives every
# future erasure. `memory/erase.py` already closed `TOMBSTONE_FIELDS` for exactly
# that reason — but only for tombstones, and only on the read side, which covered one
# of the three producers. The other two were open: `memory_node` writes
# `profile_loaded`, and `safety/guard.audit_record` takes arbitrary `**fields` while
# its own docstring asserted that "the same cascade-delete path that covers `events`
# covers this stream". That premise is false, and it is the more dangerous half of
# the defect: a comment inviting the next writer to pass the text.
#
# Declared per kind with **no default**, so a new audit kind is refused until someone
# writes down what it may carry. The timing is the point and it is the same argument
# as `WHY_REQUIRED`: a field set is only free to mandate before the first writer
# exists, and a default either way — open or closed — is the decision nobody made.
#
# Fields no row lists because they are the store's, not a kind's. `kind` selects the
# row; `seq` and `ts` are stamped by the writer and refused from callers above; and
# `why` is governed by `check_why` and `WHY_REQUIRED` — a per-kind copy of that
# decision would be a second reader of one format, free to drift from the first.
AUDIT_IMPLICIT_FIELDS = ("kind", "seq", "ts", "why")

AUDIT_FIELDS = {
    # safety/guard.audit_record, written by safety/output_guard.py
    # `tag` is the rejected gesture tag and it is declared on purpose: the `\w+`
    # regex that produces it admits no spaces and no punctuation, so it cannot be an
    # utterance, and "which tag was refused" is the one thing that record exists to
    # say. `utterance` is deliberately absent — see check_audit.
    "audit": ("action", "audibility", "decision", "detail", "guard_decision",
              "interrupted", "latency_s", "reason", "span_index", "tag", "uid",
              "uninterruptible"),
    # agent/context.py's `audit_payload`, which its own docstring describes as "the
    # dict `agent_node` hands to `guard.audit_record` for a composed turn". Declared
    # here although **nothing calls it yet** outside its test, and that is the reason
    # rather than an oversight: this row is free to mandate exactly once, before the
    # first record exists. Wire the call up first and the choice is already made — the
    # composed-turn record would be refused at the append, and the cheapest way past a
    # refusal is to widen the set to whatever the caller happens to pass.
    #
    # `chars` is `len(prompt)`, not the prompt: `Assembly.as_dict` deliberately keeps
    # the prompt out, and the prompt is the one field here that would contain the
    # utterance. `blocks` and `findings` are block *names* and our own finding strings
    # from `context._put`, neither of which is quoted input.
    "chat": ("blocks", "chars", "findings"),
    # memory/memory_node.py, at card load
    "profile_loaded": ("facts", "kinds", "locked"),
    # memory/erase.py. `TOMBSTONE_FIELDS` is the read-side twin: it lists the
    # *complete* on-disk field set, so it equals this row plus the implicit ones, and
    # `tests/test_audit_schema.py` asserts that equality rather than trusting it.
    "tombstone": ("episode", "removed"),
}

# Longest string any audit record may carry, per field.
#
# The field-set closure above stops a *new* field from arriving. It does not bound
# the fields already declared, and one of them is not ours: `detail` is set from the
# classifier's own error message (`safety/output_guard.py`), so its content comes from
# upstream and has no length its author controls. An unbounded upstream string in the
# one unerasable stream is the same failure as a quoted utterance, reached by a
# different route.
#
# Stated honestly, because the number is the whole claim: this bounds *how much*
# outlives a deletion. It does not make the record text-free, and it cannot — no
# check on a string's value can tell prose we wrote from prose someone said. That
# distinction is provenance, not content, and it is checked where provenance lives:
# `tests/test_audit_schema.py` walks the call sites.
#
# Second limit, and this one is a shape rather than a judgement: the check below reads
# the record's own values, so it bounds a *top-level* string and not a string inside a
# list or dict. Every container-valued declared field today holds machine-generated
# content — `removed` is sequence numbers, `blocks` is block names, `kinds` is a count
# — and a tombstone's `removed` is legitimately large, so a blanket recursive size
# bound would refuse a correct deletion receipt. Walking containers is therefore a
# separate change with its own mutant, and until it lands the guarantee is: no
# declared *string* field exceeds this, and no declared container field is fed from
# user input at any site `tests/test_audit_schema.py`'s census can see.
AUDIT_TEXT_CEILING = 500

# Record fields that are meaningless without a companion field (plan §S3.5).
#
# `salience` is a number produced by a formula, and the same number from a different
# formula is a different quantity. A score written without its producer's id is a
# value nothing can ever compare: when S6 changes the weights, "these two records
# disagree" and "these two records are not the same measurement" become the same
# observation, and no gate can tell them apart afterwards because the information was
# never written down. Refusing the record is the only point at which that is fixable.
#
# Declared as data rather than by importing `memory.salience`: the one writer must not
# acquire a dependency on every module that derives a field, and the *value* of the id
# belongs to the module that computes it — `tests/test_salience.py` asserts there is
# exactly one creator of it, the same way `PREDICATES` has exactly one.
RECORD_FIELD_DEPS = {"salience": "salience_formula"}

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


def recovered_seq(root: Path) -> int:
    """The highest sequence number already on disk under `root`, across all streams.

    A `Store` resumes from this instead of starting at zero, and the reason is the
    deletion transaction. `rewrite_stream` addresses records by sequence number, and
    `memory.consolidate.episode_id` derives an episode's identity from the sequence of
    its first record. A counter that restarts issues numbers the files already hold,
    so after one reboot "delete seq 5" names two records written weeks apart, in two
    different episodes — a right-to-erasure request that removes somebody's unrelated
    sentence, silently, with a receipt that reads clean. Two episodes would also mint
    the same id, which the warm layer's duplicate check would report as corruption
    long after the cause.

    One counter across **both** streams, matching the one that issues them: the max is
    taken over every member of `STREAMS`, so an `audit`-only session still moves the
    number that `events` will use next.

    Snapshot and rewrite operations consume a sequence number and write no record, so
    the recovered value can be lower than the highest ever issued. That is harmless
    and deliberate — nothing addresses a snapshot by seq, and the property that
    matters is only that no number on disk is issued twice.

    Raises on an unparseable line, exactly like `read`, which means a torn line stops
    the node from booting rather than being repaired by guessing. The consequence is
    stated because it is a real cost: a power cut during a write can leave a torn
    final line, and this treats that the same as a second writer. Distinguishing them
    (tolerating a torn *tail*, refusing a torn *middle*) is a policy decision with its
    own audit obligation, and inventing it quietly here would be the wrong place.
    """
    root = Path(root)
    high = 0
    for stream in STREAMS:
        path = root / f"{stream}.jsonl"
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StoreError(
                    f"{path.name} line {n} is not valid JSON ({exc}) — refusing to "
                    f"resume the sequence counter from a file with a partial line, "
                    f"{len(line)} bytes. Resuming at 0 would re-issue numbers this "
                    "file already holds, and a deletion addressed by sequence number "
                    "would then remove another episode's record") from exc
            seq = record.get("seq") if isinstance(record, dict) else None
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
                raise StoreError(
                    f"{path.name} line {n} has seq {seq!r}: a record with no sequence "
                    "number cannot bound the counter, and resuming below it would "
                    "issue a number that is already in use")
            high = max(high, seq)
    return high

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


def check_why(why) -> str:
    """Validate a `why` block and return its goal. Raises `StoreError` otherwise.

    A function rather than a method, for the same reason `location_findings` is one:
    the producer that is about to send a ROS message can check the reason it is
    about to claim without holding the writer, and a test can assert on the message
    instead of on an exit code.
    """
    if not isinstance(why, dict):
        raise StoreError(f"why must be a dict, got {type(why).__name__}: "
                         f"{why!r} — the shape is {{'goal': <one of GOALS>}}")
    extra = sorted(k for k in why if k != "goal")
    if extra:
        raise StoreError(
            f"why carries {extra} beyond 'goal': a companion key is where the free "
            "text this enum exists to refuse comes back one level down, and it "
            "would be recalled into a prompt as data the model reads")
    if "goal" not in why:
        raise StoreError("why has no 'goal': an empty reason block satisfies every "
                         "check that only looks at what is present")
    goal = why["goal"]
    if not isinstance(goal, str):
        raise StoreError(f"why.goal must be a str, got {type(goal).__name__}: "
                         f"{goal!r}")
    if goal not in GOALS:
        raise StoreError(
            f"unknown why.goal {goal!r}: the set is closed, lives in "
            f"memory.store.GOALS ({len(GOALS)} entries), and belonging to a family "
            f"grants nothing — the members are {GOALS}")
    return goal


def check_audit(record) -> str:
    """Validate an `audit` record's contents and return its kind.

    A function rather than a method, for the same reason `check_why` is one: the
    producer can check the record it is about to claim without holding the writer,
    and `memory/erase.py` already does that with `parse_tombstone` before appending.

    Both directions matter and they fail differently. An undeclared *field* is the
    "helpful excerpt" case: someone adds `text` to make a receipt readable, and
    because `REWRITABLE['audit']` is False that text is now the one copy of a deleted
    utterance no deletion can reach. An undeclared *kind* is the case a default would
    have swallowed: a new producer inherits either a closed set that does not describe
    it, or an open one — and an open default is how the first writer of a kind gets to
    decide the schema by accident.
    """
    if not isinstance(record, dict):
        raise StoreError(f"an audit record must be a dict, got "
                         f"{type(record).__name__}")
    kind = record.get("kind")
    if not isinstance(kind, str) or not kind:
        raise StoreError(
            f"audit record has no usable 'kind' ({kind!r}): the field set is per kind, "
            "so a record without one cannot be checked at all, and an unchecked audit "
            "record is exactly the copy that outlives every deletion")
    if kind not in AUDIT_FIELDS:
        raise StoreError(
            f"unknown audit kind {kind!r}: the kinds are {tuple(AUDIT_FIELDS)}. "
            "A new kind has to write down what it may carry before its first record "
            "exists — audit.jsonl is not erasable, so a field set chosen afterwards "
            "cannot be applied to what is already on disk")
    allowed = AUDIT_FIELDS[kind]
    # Named `undeclared`, not `unknown`: `check_why` below already owns the line
    # `if unknown:` and T-J4's mutants anchor on that exact text, so a second copy
    # here makes the anchor ambiguous and the harness refuses to build the mutant.
    # The distinct name is also the more accurate one — an undeclared field on a
    # known kind is a different refusal from an unknown goal.
    undeclared = sorted(set(record) - set(AUDIT_IMPLICIT_FIELDS) - set(allowed))
    if undeclared:
        raise StoreError(
            f"audit record of kind {kind!r} carries undeclared field(s) {undeclared}: "
            f"audit.jsonl is outside the erasure (memory.store.REWRITABLE), so "
            f"anything here outlives every future deletion. The fields are {allowed}")
    for field in sorted(allowed):
        value = record.get(field)
        if isinstance(value, str) and len(value) > AUDIT_TEXT_CEILING:
            raise StoreError(
                f"audit record of kind {kind!r} has {field!r} at {len(value)} chars, "
                f"over AUDIT_TEXT_CEILING={AUDIT_TEXT_CEILING}. Not every declared "
                "field's content is ours — 'detail' comes from the classifier — and "
                "an unbounded upstream string in the one unerasable stream is a "
                "quoted utterance reached by another route")
    return kind


def goal_of(record: dict) -> str:
    """The validated goal of one record read back off disk.

    Reading is exactly as strict as writing, and returning `None` for a record with
    no reason is the version that fails silently: such a record would be absent from
    every goal filter and every census, which is the same observation as its not
    existing. A record in a stream that requires a reason and lacks one was not
    written by this module — the same conclusion `read()` draws from a line that
    does not parse, and the same reason it raises instead of skipping.
    """
    if "why" not in record:
        raise StoreError(
            "record has no 'why' — it was not written through Store.append on a "
            f"stream that requires one: {sorted(record)[:6]}")
    return check_why(record["why"])


def filter_by_goal(records, goals) -> list[dict]:
    """The records whose goal is in `goals`. Raises on an unknown goal in the query.

    The refusal is the point. An unknown goal returns the same empty list as a goal
    nothing has yet, so a typo, a rename, or a stale caller reads as "there were
    none of those" — the reading a retention sweep or a deletion audit would act on.
    """
    if isinstance(goals, str):
        raise StoreError(
            f"pass a sequence of goals, not the string {goals!r}: a string is "
            "iterable, so this would quietly ask for its characters and match none")
    wanted = tuple(goals)
    unknown = sorted({g for g in wanted if g not in GOALS})
    if unknown:
        raise StoreError(f"unknown goal(s) {unknown} in a query; the set is {GOALS}")
    return [r for r in records if goal_of(r) in wanted]


def goal_census(records) -> dict[str, int]:
    """Records per goal — **every member present, zeros included**.

    Omitting the zeros makes "nothing carried this goal" and "this goal is no longer
    in the set" the same output, and both a renamed member and a deleted one then
    look like a quiet day. Keys come out in `GOALS` order, so two runs of the same
    input produce byte-identical JSON.
    """
    counts = {goal: 0 for goal in GOALS}
    for record in records:
        counts[goal_of(record)] += 1
    return counts


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
        # Resumed from the disk, not started at zero. See `recovered_seq`.
        #
        # `_durable` starts at the same value, and it has to: every recovered number is
        # by definition already on the disk, while `_durable` means "published by the
        # writer thread of *this* process". Leaving it at 0 makes `flush()` — whose
        # default target is `self._seq` — wait for numbers no writer will ever publish,
        # so opening a store over a non-empty root and closing it without writing
        # anything hangs for five seconds and then leaves the thread waiting. That is
        # the shape of defect this pair invites: two variables holding one fact, only
        # one of them updated.
        self._seq = recovered_seq(self.root)
        self._durable = self._seq
        self._failure: BaseException | None = None
        # Outcome of each rewrite, keyed by its sequence number. The dropped set is
        # only knowable in the writer thread (it is what the file actually held), and
        # the tombstone has to be built from what was removed rather than from what
        # the caller intended — two numbers, two claims.
        self._rewrites: dict[int, tuple[int, ...]] = {}
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
        # Before `_raise_if_failed`, like the snapshot lock and for the same reason: a
        # record refused for what it contains must not be reported as a broken disk.
        for field, companion in sorted(RECORD_FIELD_DEPS.items()):
            if field in record and companion not in record:
                raise StoreError(
                    f"a record carrying {field!r} must also carry {companion!r}: a "
                    f"derived value with no record of what produced it cannot be "
                    f"compared with the next one — use memory.salience.stamp()")
        # Same placement as the field-dependency check above, and for both of its
        # reasons: before `_raise_if_failed` so a record refused for what it contains
        # is never reported as a broken disk, and before a sequence number is taken
        # so a refusal leaves no gap — a gap is indistinguishable from a deletion,
        # which is the one ambiguity an audit log exists to remove.
        if stream not in WHY_REQUIRED:
            raise StoreError(
                f"stream {stream!r} is in STREAMS but not in WHY_REQUIRED: a new "
                "stream has to decide whether a reason is mandatory, and a default "
                "either way is the decision nobody made")
        # Same placement and both of the same reasons as the checks above: a record
        # refused for what it contains must not be reported as a broken disk, and a
        # refusal must not consume a sequence number. The second one carries extra
        # weight here — a gap in `audit.jsonl` is indistinguishable from a deleted
        # record, which is the single ambiguity this stream exists to remove, so the
        # check that protects the audit log must not be able to dent it.
        if stream == "audit":
            check_audit(record)
        if "why" in record:
            check_why(record["why"])
        elif WHY_REQUIRED[stream]:
            raise StoreError(
                f"a {stream!r} record must carry why={{'goal': ...}}: an optional "
                "reason is how the retrieval key and the safety boundary both stop "
                f"existing while records keep flowing. The goals are {GOALS}")
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

    def rewrite_stream(self, stream: str, drop) -> tuple[int, ...]:
        """Remove the records of `stream` whose seq is in `drop`. Returns what went.

        The only way to make an append-only file shorter, and the deletion
        transaction's one write into a stream. Four deliberate shapes:

        **It takes seqs, not lines.** A `rewrite(stream, lines)` API is an
        edit-history API: it can alter a record's text as easily as remove it, and an
        append-only log exists to make exactly that impossible. Addressing records by
        sequence number means the survivors are copied through byte for byte, so this
        primitive can drop whole records and can do nothing else.

        **It names what goes, not what stays.** A keep-set is the more natural way to
        write the filter and it is wrong here: any record appended between the moment
        the caller computed its plan and the moment this runs is absent from the
        keep-set, so it would be deleted — silently, and by a transaction that never
        examined it. Under-deleting is visible (the receipt says a reference is still
        there) and fixed by running the transaction again; over-deleting personal data
        is the one direction nothing can repair. So the caller names its targets and a
        record it has not heard of survives.

        **It returns the seqs actually dropped rather than accepting them.** What the
        file held is only knowable here, and a tombstone built from the caller's plan
        would claim a deletion the disk may not have performed. The difference between
        the request and the return is itself information: on a second run of the same
        deletion it is the whole request, which is what makes the transaction
        idempotent rather than merely repeatable.

        **It blocks.** Every other producer call is fire-and-forget because the speech
        loop cannot wait on a disk. A deletion is a transaction whose next step must
        not start until this one is durable, and it runs on the retention sweep or on
        「忘記我說的 X」 — never inside a turn.
        """
        if stream not in STREAMS:
            raise StoreError(f"unknown stream {stream!r}; the streams are {STREAMS}")
        if stream not in REWRITABLE:
            raise StoreError(
                f"stream {stream!r} is in STREAMS but not in REWRITABLE: a new stream "
                "has to decide whether a deletion may shorten it, and a default "
                "either way is the decision nobody made")
        if not REWRITABLE[stream]:
            raise StoreError(
                f"{stream!r} may not be rewritten: an audit log the deletion path can "
                "shorten is not an audit log, because the record that a deletion "
                "happened would be removable by the next one. See memory.erase.RETAINED "
                "for what that costs and what is not covered by an erasure")
        drop_set = frozenset(drop)
        bad = sorted(repr(s) for s in drop_set
                     if isinstance(s, bool) or not isinstance(s, int))
        if bad:
            raise StoreError(
                f"drop must contain store sequence numbers, got {bad} — a list "
                "position here would remove the wrong records and the file would "
                "still parse")
        self._raise_if_failed()
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._q.put(("rewrite", stream, drop_set, seq))
        self.flush(seq)
        with self._lock:
            if seq not in self._rewrites:
                raise StoreError(
                    f"rewrite {seq} of {stream!r} reported durable with no outcome "
                    "recorded — refusing to build a tombstone from an absent result")
            return self._rewrites.pop(seq)

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
                elif op == "rewrite":
                    # Drain first, for the *other* stream. `_rewrite` closes the
                    # handle it is about, and closing flushes, so this is not what
                    # makes an in-flight append to `events` visible to the copy. What
                    # it does is make everything else in the batch durable before a
                    # file gets replaced — above all the tombstone, which is an
                    # `audit` append that must reach the disk before the removal it
                    # describes. `memory.erase.delete` also flushes it explicitly;
                    # this is the same ordering made structural, so a future caller
                    # that forgets does not silently invert it.
                    #
                    # `touched.clear()` is load-bearing, not tidiness: `_sync` looks
                    # each stream up in `self._handles`, and `_rewrite` pops the one
                    # it closed, so a stale entry here raises KeyError at the end of
                    # the batch and kills the writer.
                    self._sync(touched)
                    touched.clear()
                    dropped = self._rewrite(target, body)
                    with self._lock:
                        self._rewrites[seq] = dropped
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

    def _rewrite(self, stream: str, drop: frozenset) -> tuple[int, ...]:
        """Copy `stream` through, omitting `drop`. Writer thread only.

        **Why the handle is closed at the end and not reused.** `_handle` keeps the
        append fd open across records, and `os.replace` swaps the *name* onto a new
        inode — the old fd keeps pointing at the unlinked file. Reusing it would send
        every subsequent append into a file with no name, so the records would be
        accepted, fsynced, reported durable, and invisible to every reader, with the
        disk usage as the only symptom. Dropping the handle makes the next append
        reopen by name. `tests/test_store.py` asserts on the inode rather than on the
        record count, because a count can pass while pointing at the wrong file.

        **Why it does not flush or fsync first, even though it must not read a stale
        file.** A buffered append that is not in the file yet would be copied away —
        a record destroyed by a deletion that never named it, which is the one
        direction nothing can repair. Two mechanisms already prevent it and neither is
        this one: `close()` above flushes the Python buffer, and `_run` drains the
        batch with `_sync(touched)` before calling here. Adding a third would put a
        second `fh.flush()` / `os.fsync(fh.fileno())` pair in this module, and the
        mutation harness anchors on that exact pair and requires it to be unique — so
        the duplicate would make "remove the fsync" ambiguous about which site it
        removed, weakening a live test to buy a redundancy. That the property holds is
        shown by removing *both* mechanisms at once
        (tests/test_store.py::test_without_the_flush_and_the_drain_a_rewrite_deletes_a_bystander),
        which is honest about what that mutant demonstrates: the property is protected
        twice over, and the mutant does not attribute it to one line.

        **A torn line here kills the writer, and that is the intended response.** The
        refusal below runs in the writer thread, so `_failure` is set and no further
        record — including audit — is written by this store. The caller does learn:
        `rewrite_stream` blocks on `flush(seq)`, which re-raises. It is the right
        trade because a partial line means a second process wrote to this file, so the
        single-writer invariant the module exists to provide is already gone, and the
        alternative is to keep appending to a file known to be corrupt. The
        availability cost is real and is not hidden: one torn line stops the memory
        node's logging until a person looks at the file.
        """
        path = self.path_for(stream)
        fh = self._handles.pop(stream, None)
        if fh is not None:
            fh.close()
        if not path.exists():
            return ()
        kept_lines: list[str] = []
        dropped: list[int] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # The same refusal `read()` makes, and it matters more here: skipping
                # an unparseable line would *delete* it, so a deletion would quietly
                # repair the corruption it should have reported.
                raise StoreError(
                    f"{path.name} line {n} is not valid JSON ({exc}) — refusing to "
                    f"rewrite a file with a partial line, {len(line)} bytes") from exc
            seq = record.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int):
                raise StoreError(
                    f"{path.name} line {n} has seq {seq!r}: a record with no sequence "
                    "number cannot be kept or dropped by identity, and rewriting the "
                    "file would decide its fate by position")
            if seq in drop:
                dropped.append(seq)
            else:
                kept_lines.append(line)
        self._replace(path.name, "".join(f"{line}\n" for line in kept_lines))
        return tuple(dropped)

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
