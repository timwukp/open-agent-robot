"""memory/erase.py — the deletion transaction: what 「忘記我說的 X」 actually removes.

Zero ROS, zero boto3, zero network, zero LLM. Everything is injected — the store,
the clock, the retention period — so the whole transaction runs under pytest on a
laptop against a `tmp_path` root.

This is the half of design item T-C4 that did not exist: `memory/recall.py` shipped
the five paths and `deletion_findings()`, which is the *evidence* side, and the plan's
own §S5 row notes that `store.py` had "no tombstone, no delete, no retention". So the
check existed before the thing it checks, on purpose (the criterion was satisfiable by
a path that never ran, which had to be fixed before anything relied on it). This module
is the action.

**Why a tombstone is not a deletion, and what the tombstone is for instead.**
The obvious implementation of "delete an episode" in an append-only store is to append
a marker and teach the readers to skip it. That is retrieval suppression, which plan
§1 already refused under its other name: 「衰減是機率性檢索抑制,不是刪除」. The sentence
the person asked to be forgotten would still be in `events.jsonl` — visible to `grep`,
to a backup, and to the next node somebody writes that reads the file directly. So the
records are removed from the file (`Store.rewrite_stream`), and the tombstone is the
*audit* record that the removal happened. It carries ids and counts and no text: it
lives in `audit.jsonl`, which is deliberately **not** erasable, so a tombstone quoting
what it deleted would be the one copy of the deleted text that no deletion can reach.
`TOMBSTONE_FIELDS` is closed for that reason and validated on the way back in, so a
future writer that adds a helpful `"text"` field reds instead of shipping.

**Why the whole plan is computed before anything is written.**
Erasing `episodes.json` first is the natural order — it is the record of the episode —
and it destroys the map from event to episode, so the events that belonged to E become
unattributable and the transaction can no longer find them. That is not a hypothetical:
it is exactly the `ORPHAN` state `recall.py` describes as the cheapest way to pass a
cascade test without deleting anything, reached here by accident rather than by malice.
So every artifact's removal is planned against the pre-deletion state, and only then
applied — and `erase_order()` puts the artifact that *carries the attribution map*
last, derived from `ATTRIBUTION_ARTIFACT` rather than written down, so that a crash
mid-transaction leaves a state the same transaction can finish. Alphabetical order puts
`episodes.json` first, which is why the order is a function with a test rather than a
`sorted()` at the use site.

**Why the tombstone is appended before the removals, not after.**
Crash between the two, and the two orders leave different worlds. Tombstone first: the
audit log claims a removal the disk has not performed, `tombstone_findings()` says so,
and re-running the transaction completes it. Removals first: the records are gone with
nothing to say why, which is indistinguishable from the corruption this store exists to
prevent, and unrecoverable — nobody can reconstruct what a missing record was. So the
claim is written first and the world is made to match it.

**Why the eraser registry is derived from the disk, not from `recall.PATHS`.**
`PATHS` is what is *retrievable*; erasure has to cover what is *stored*, and those sets
differ today by exactly one file. `recall.coverage_findings()` on a real store root
already returns a finding for `audit.jsonl`: it is a durable artifact no retrieval path
reads. A registry derived from `PATHS` would therefore have silently left the audit log
out of the erasure forever, which is the one artifact where that is a decision worth
making rather than an omission. So `registry_findings()` joins the disk against
`ERASERS ∪ RETAINED` and refuses to run a deletion while any artifact is in neither —
a new file written by a future node forces a decision instead of inheriting one.

**What an erasure does not cover, stated instead of implied.**
`RETAINED` is a written decision per artifact, and every receipt carries it. The audit
stream is retained because an audit log the deletion path can shorten is not an audit
log. The cost is real and is not talked around: nothing today stops an audit record
from quoting user text, so the erasure cannot claim the store holds no trace of E. The
missing piece is a content schema on `audit` records, which is a separate change with
its own mutant, and until it lands `Receipt.retained` says so on every deletion rather
than the absence being mistaken for completeness.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from memory import recall
from memory.consolidate import is_episode_id
from memory.profile import ProfileError
from memory.profile import load as load_card
from memory.profile import parse as parse_card
from memory.store import GOALS, Store, StoreError, is_locked


class EraseError(StoreError):
    """A deletion that must not proceed: a bad target, an unclassified artifact,
    a locked card, or a plan whose output its own reader would refuse."""


# ------------------------------------------------------------------------ goals
#
# The two reasons a deletion may happen, and both are already members of
# `store.GOALS` — the enum that exists so a reason cannot be free text. Named here
# rather than spelled at the call sites so the tombstone's goal and the audit
# record's goal cannot drift apart.
#
# **Why the check is inside the constant's creator rather than a loop underneath it.**
# The membership has to be checked when the module is imported, not at the first
# deletion: a goal this module names that `store.GOALS` does not contain makes every
# tombstone unappendable, and finding that out when somebody asks to be forgotten is
# strictly worse than finding it out when the test suite imports the module. A rename
# produces exactly that, silently.
#
# Written first as a `for` loop over `DELETION_GOALS`, which was correct and had a
# consequence I had not thought about: `infra/entrypoints.py` classifies any top-level
# statement that is not a declaration as import-time work, so the loop made this
# library look like a runnable script and check 8 reported it as an entrypoint nothing
# launches. The detector is right by its own definition — it cannot tell "validates its
# own constants" from "pushes blobs on import", and widening its exclusion list is how
# an allowlist gets taught its second exception quietly. Declaring the file in
# `infra/standalone_entrypoints.tsv` would have been worse: it is not standalone, so
# the declaration would be false.
#
# So the validation moved to the one place that *creates* the value. A constant cannot
# be defined without being checked, which is a stronger invariant than a loop that a
# third goal could be added below — and to the AST it is an assignment. Do not
# "simplify" this back into a loop; the deploy guard will red and this comment is the
# only thing that explains why.
def _deletion_goal(goal: str) -> str:
    """`goal`, if `store.GOALS` contains it. Raises at definition time otherwise."""
    if goal not in GOALS:
        raise EraseError(
            f"{goal!r} is not a member of memory.store.GOALS: a deletion whose "
            f"reason is not a valid reason cannot write its own audit record. "
            f"The members are {GOALS}")
    return goal


FORGET_GOAL = _deletion_goal("user_requested_forget")   # 「忘記我說的 X」, said aloud
RETENTION_GOAL = _deletion_goal("scheduled_retention")  # the sweep; nothing caused it
DELETION_GOALS = (FORGET_GOAL, RETENTION_GOAL)

# -------------------------------------------------------------------- tombstones
TOMBSTONE_KIND = "tombstone"

# Closed, and enforced in both directions by `parse_tombstone`. The absent field is
# the point: there is no `text`, no `summary`, no `excerpt`. `audit.jsonl` is retained
# (see `RETAINED`), so anything quoted in a tombstone is a copy of the deleted content
# that survives every future deletion — the erasure would be undone by its own receipt.
TOMBSTONE_FIELDS = ("episode", "kind", "removed", "seq", "ts", "why")

# ---------------------------------------------------------------------- artifacts
#
# The artifact whose file *is* the attribution map. `erase_order()` derives the
# ordering from this rather than hard-coding "episodes last", so a future layout that
# moves the map somewhere else moves the ordering with it.
ATTRIBUTION_ARTIFACT = recall.EPISODES_ARTIFACT

# Artifacts a deletion deliberately does **not** touch, each with the reason written
# down. A dict rather than a set: a retained artifact with no reason is
# indistinguishable from one nobody thought about, and this is the field a reviewer
# reads when they ask what "deleted" means here.
RETAINED = {
    "audit.jsonl":
        "the audit log is append-only for the deletion path too: a log the deletion "
        "can shorten cannot hold the record that a deletion happened. Consequence, "
        "not yet closed — nothing enforces that an audit record omits user text, so "
        "an erasure cannot claim the store retains no trace of the episode. The fix "
        "is a content schema on audit records; until then every receipt says this.",
}

# ------------------------------------------------------------------- retention
#
# A policy default, and labelled as one. No regulation names a number, nothing here
# has measured what a person expects, and a value dressed up as a finding would be a
# number in prose (§6.0 rule 6). What *is* tested is the mechanism: the boundary is
# exact, an episode one second either side of it lands on the right side, and an
# episode that cannot be dated is reported rather than silently kept or silently
# deleted. The period is a parameter on every function that uses it.
RETENTION_DAYS = 90
RETENTION_S = RETENTION_DAYS * 24 * 60 * 60


class Removal:
    """One artifact's planned change: what goes, and the exact bytes-to-be.

    Planned and applied in two steps because the plan has to be complete before the
    first write lands — see the module docstring on ordering. `body` is a snapshot
    object and `drop_seqs` is a stream rewrite; a `Removal` carries at most one of
    them, and neither means this artifact needs no change.
    """

    __slots__ = ("artifact", "refs", "body", "drop_seqs", "findings")

    def __init__(self, artifact: str, refs: Sequence = (), *, body=None,
                 drop_seqs: Sequence[int] | None = None,
                 findings: Sequence[str] = ()) -> None:
        if body is not None and drop_seqs is not None:
            raise EraseError(
                f"{artifact}: a removal is either a snapshot replacement or a stream "
                "rewrite, and one that claims both would be applied twice")
        self.artifact = artifact
        self.refs = tuple(refs)
        self.body = body
        self.drop_seqs = None if drop_seqs is None else tuple(drop_seqs)
        self.findings = tuple(findings)

    @property
    def changes_anything(self) -> bool:
        return self.body is not None or bool(self.drop_seqs)


class Receipt:
    """What one deletion did, and what it could not prove. Never any deleted text.

    `removed` and `findings` are two claims and they are derived separately.
    `removed` comes back from the disk — `rewrite_stream` returns the sequence numbers
    it actually dropped, not the ones it was asked for — so a receipt that says a
    record went is a statement about the file rather than about the plan.
    """

    __slots__ = ("episode", "goal", "tombstone_seq", "removed", "findings", "retained")

    def __init__(self, episode: str, goal: str, tombstone_seq: int,
                 removed: Mapping[str, Sequence], findings: Sequence[str],
                 retained: Mapping[str, str]) -> None:
        self.episode = episode
        self.goal = goal
        self.tombstone_seq = int(tombstone_seq)
        self.removed = {k: tuple(v) for k, v in sorted(removed.items())}
        self.findings = tuple(findings)
        self.retained = dict(sorted(retained.items()))

    def count(self, artifact: str) -> int:
        return len(self.removed.get(artifact, ()))

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.removed.values())

    def as_dict(self) -> dict:
        return {"episode": self.episode, "goal": self.goal,
                "tombstone_seq": self.tombstone_seq,
                "removed": {k: list(v) for k, v in self.removed.items()},
                "findings": list(self.findings),
                "retained": dict(self.retained)}


# ------------------------------------------------------------------- the planners
#
# One `_plan_<artifact>` per entry in `ERASERS`, and the equality of those two sets is
# asserted in `tests/test_erase.py` — the same two-directional derivation `recall.py`
# uses for `_run_<path>`. A planner with no registration is unreachable; a
# registration with no planner reds instead of skipping an artifact.
def _plan_events(root: Path, episode: str, *, attrib: Mapping[int, str]) -> Removal:
    """The event records attributed to `episode`.

    Attribution comes from the caller, computed once against the pre-deletion warm
    layer, so this planner cannot accidentally read a map that a sibling planner has
    already invalidated.
    """
    path = root / recall.EVENTS_ARTIFACT
    if not path.exists():
        return Removal(recall.EVENTS_ARTIFACT)
    events = recall.load_events(root)
    drop = [e["seq"] for e in events if attrib.get(e["seq"]) == episode]
    unattributed = [e["seq"] for e in events if e["seq"] not in attrib]
    findings = []
    if unattributed:
        findings.append(
            f"{recall.EVENTS_ARTIFACT}: {len(unattributed)} record(s) belong to no "
            f"episode ({unattributed[:5]}…), so this erasure cannot show they were "
            f"not {episode}'s. A sweep runs when no episode is open; an open one "
            "legitimately has unattributed events")
    return Removal(recall.EVENTS_ARTIFACT, drop, drop_seqs=drop, findings=findings)


def _plan_episodes(root: Path, episode: str, *, attrib: Mapping[int, str]) -> Removal:
    """The warm-layer entry for `episode`: its summary, its vector, its seq list."""
    path = root / ATTRIBUTION_ARTIFACT
    if not path.exists():
        return Removal(ATTRIBUTION_ARTIFACT)
    entries = recall.load_episodes(path)
    kept = [e for e in entries if e["id"] != episode]
    if len(kept) == len(entries):
        return Removal(ATTRIBUTION_ARTIFACT)
    body = {"version": 1, "episodes": kept}
    # Validated with the retrieval path's own parser before it is written. An erasure
    # that emits a file its reader refuses turns into a corrupt store on the next
    # boot, several hours from the write that caused it.
    recall.parse_episodes(body, where=f"<planned {ATTRIBUTION_ARTIFACT}>")
    return Removal(ATTRIBUTION_ARTIFACT, (episode,), body=body)


def _plan_card(root: Path, episode: str, *, attrib: Mapping[int, str]) -> Removal:
    """The profile card, fact by fact. Two branches, and both have to exist.

    A fact sourced *only* to `episode` goes entirely. A fact with a second source
    stays and loses one citation — and getting that wrong in either direction is a
    defect with no symptom: dropping the multi-source fact deletes something the
    person never asked to have deleted, and keeping the single-source one leaves a
    fact whose only evidence is gone. Design item T-C4 requires the fixture to prove
    both branches occurred with a counting assertion, which is why the refs
    distinguish them (`#i` for a dropped fact, `#i@eid` for a dropped citation).

    Seed facts are hand-written and cite nothing (`profile.TRUST_SEED`), so they are
    outside the cascade by construction and are never touched here.
    """
    path = root / recall.CARD_ARTIFACT
    if not path.exists():
        return Removal(recall.CARD_ARTIFACT)
    card = load_card(path)
    refs, facts, changed = [], [], False
    for i, fact in enumerate(card.facts):
        if episode not in fact.source_episode_ids:
            facts.append(fact.as_dict())
            continue
        changed = True
        remaining = [e for e in fact.source_episode_ids if e != episode]
        if remaining:
            refs.append(f"{recall.CARD_ARTIFACT}#{i}@{episode}")
            entry = fact.as_dict()
            entry["source_episode_ids"] = remaining
            facts.append(entry)
        else:
            # A non-seed fact may not have an empty source list — `profile.Fact`
            # refuses it, because a fact with no episode behind it can never be
            # removed by a deletion. So the fact goes rather than being kept
            # sourceless, which is the same rule stated from the other side.
            refs.append(f"{recall.CARD_ARTIFACT}#{i}")
    if not changed:
        return Removal(recall.CARD_ARTIFACT)
    body = {"version": card.version, "facts": facts}
    try:
        parse_card(body, where=f"<planned {recall.CARD_ARTIFACT}>")
    except ProfileError as exc:
        # The realistic instance is an emptied card: `profile.parse` refuses a card
        # with no facts, because an empty one loads fine and makes every recall test
        # pass vacuously. Refusing here, before any write, is the difference between
        # a deletion that declines and a store that will not boot.
        raise EraseError(
            f"erasing {episode} would leave a profile card its own reader refuses "
            f"({exc}). The seed facts are hand-written and cite no episode, so a card "
            "reduced to nothing means every fact in it came from this one episode — "
            "edit the seed card rather than making the reader accept an empty one"
        ) from exc
    return Removal(recall.CARD_ARTIFACT, refs, body=body)


def _plan_rps(root: Path, episode: str, *, attrib: Mapping[int, str]) -> Removal:
    """The opponent table's rounds. Behavioural data about one person (plan §4).

    A round with no `episode` field is a real state — a table written before the
    attribution field existed — and it is kept, with a finding. It cannot be shown to
    belong to `episode` and it cannot be shown not to, and deleting it on suspicion
    would destroy another episode's data to make this receipt look clean.
    """
    path = root / recall.RPS_ARTIFACT
    if not path.exists():
        return Removal(recall.RPS_ARTIFACT)
    rounds = recall.load_rounds(path)
    kept, refs, unattributed = [], [], 0
    for i, entry in enumerate(rounds):
        eid = entry.get("episode")
        if eid == episode:
            refs.append(f"{recall.RPS_ARTIFACT}#{i}")
            continue
        if eid is None:
            unattributed += 1
        kept.append(entry)
    findings = []
    if unattributed:
        findings.append(
            f"{recall.RPS_ARTIFACT}: {unattributed} round(s) name no episode, so this "
            f"erasure cannot show they were not {episode}'s. They are kept: deleting "
            "them on suspicion would remove another episode's rounds to make this "
            "receipt read clean")
    if not refs:
        return Removal(recall.RPS_ARTIFACT, findings=findings)
    body = {"version": 1, "rounds": kept}
    recall.parse_rounds(body, where=f"<planned {recall.RPS_ARTIFACT}>")
    return Removal(recall.RPS_ARTIFACT, refs, body=body, findings=findings)


ERASERS: dict[str, Callable[..., Removal]] = {
    recall.EVENTS_ARTIFACT: _plan_events,
    ATTRIBUTION_ARTIFACT: _plan_episodes,
    recall.CARD_ARTIFACT: _plan_card,
    recall.RPS_ARTIFACT: _plan_rps,
}


def erase_order() -> tuple[str, ...]:
    """The artifacts in the order they must be written. Attribution map last.

    Derived, not written down. If `episodes.json` is erased before the events it
    attributes, a crash in between leaves events that belonged to a deleted episode
    with nothing left to say so — unrecoverable, because no later run can work out
    which orphans were E's. With the map written last, a crash leaves a warm layer
    naming an episode whose records are already gone, which re-running the same
    deletion finishes. Note that plain alphabetical order gets this exactly wrong:
    `episodes.json` sorts before `events.jsonl`.
    """
    rest = tuple(sorted(a for a in ERASERS if a != ATTRIBUTION_ARTIFACT))
    return rest + (ATTRIBUTION_ARTIFACT,)


def registry_findings(root: Path) -> list[str]:
    """Every artifact on disk that is neither erasable nor deliberately retained.

    Both directions, because either alone leaves a hole:

    * an artifact **on disk** in neither `ERASERS` nor `RETAINED` is a file a future
      node writes and a deletion silently walks past. This is the check that
      `recall.PATHS` could not provide: erasure has to cover what is stored, and
      `audit.jsonl` is stored and retrievable by nothing.
    * an artifact a **retrieval path reads** that is not in `ERASERS` is worse than
      uncovered — it is reachable. `deletion_findings` would report a clean five-path
      zero for a file no erasure ever touched, if the query happened not to match.
    """
    findings = []
    classified = set(ERASERS) | set(RETAINED)
    for name in recall.artifacts(root):
        if name not in classified:
            findings.append(
                f"{name} is a durable artifact that is neither erasable nor "
                f"deliberately retained — a deletion would walk past it and report "
                f"success. Add a planner to memory.erase.ERASERS or a written reason "
                f"to RETAINED. Classified: {sorted(classified)}")
    for name in sorted({a for spec in recall.PATHS.values() for a in spec.reads}):
        if name not in ERASERS:
            findings.append(
                f"{name} is read by a retrieval path but has no eraser: a deletion "
                "cannot remove anything from a file it does not touch, and the path "
                "would keep returning the episode")
    overlap = sorted(set(ERASERS) & set(RETAINED))
    if overlap:
        findings.append(
            f"{overlap} are both erasable and retained — one of the two is a claim "
            "nobody checks, and which one wins depends on iteration order")
    return findings


def retained_findings() -> list[str]:
    """`RETAINED`, rendered. What a deletion does not reach, on every receipt.

    A function rather than a constant string so the reasons cannot drift from the
    dict, and reported on every receipt rather than documented once: an omission
    mentioned in a docstring is an omission the next reader of a green receipt does
    not see.
    """
    return [f"{name} is not erased — {reason}"
            for name, reason in sorted(RETAINED.items())]


# ------------------------------------------------------------------- tombstones
def parse_tombstone(record: Mapping, where: str = "<record>") -> dict:
    """Validate one tombstone read back off the audit log.

    Strict in both directions, and the closed field set is the substantive half: a
    tombstone carrying a `text` or `summary` field would be a copy of the deleted
    content living in the one stream no erasure may shorten. A future writer that adds
    a helpful excerpt reds here rather than shipping.
    """
    if record.get("kind") != TOMBSTONE_KIND:
        raise EraseError(f"{where}: not a tombstone (kind={record.get('kind')!r})")
    unknown = sorted(set(record) - set(TOMBSTONE_FIELDS))
    if unknown:
        raise EraseError(
            f"{where}: tombstone has unknown field(s) {unknown} — audit.jsonl is not "
            "erasable, so any content quoted here outlives every deletion. The fields "
            f"are {TOMBSTONE_FIELDS}")
    episode = record.get("episode")
    if not is_episode_id(episode):
        raise EraseError(f"{where}: tombstone episode {episode!r} is not a minted id, "
                         "so nothing can check whether it is still retrievable")
    removed = record.get("removed")
    if not isinstance(removed, dict):
        raise EraseError(f"{where}: tombstone 'removed' must be an object mapping "
                         f"artifact to references, got {type(removed).__name__}")
    for artifact, refs in sorted(removed.items()):
        if artifact not in ERASERS:
            raise EraseError(
                f"{where}: tombstone claims a removal from {artifact!r}, which has no "
                f"eraser — a claim about a file nothing erases cannot be verified")
        if not isinstance(refs, list):
            raise EraseError(f"{where}: 'removed[{artifact}]' must be a list")
    seqs = removed.get(recall.EVENTS_ARTIFACT, [])
    bad = sorted(repr(s) for s in seqs
                 if isinstance(s, bool) or not isinstance(s, int))
    if bad:
        raise EraseError(
            f"{where}: event references must be store sequence numbers, got {bad} — "
            "a reference that is not a seq cannot be checked against the file")
    return dict(record)


def tombstones(root: Path) -> tuple[dict, ...]:
    """Every validated tombstone in the audit log, oldest first.

    Reads `audit.jsonl` directly rather than through `Store.read`, for the reason
    `recall.load_events` does: building a `Store` starts the single writer thread, and
    a verification pass that spawns a writer is a second writer waiting to happen.
    """
    path = Path(root) / "audit.jsonl"
    if not path.exists():
        return ()
    import json                                    # local: the only use in this module
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EraseError(
                f"audit.jsonl line {n} is not valid JSON ({exc}) — a partial line "
                f"means two writers, {len(line)} bytes") from exc
        if isinstance(record, dict) and record.get("kind") == TOMBSTONE_KIND:
            out.append(parse_tombstone(record, where=f"audit.jsonl line {n}"))
    return tuple(out)


def tombstone_findings(root: Path) -> list[str]:
    """Every deletion this store claims to have performed, re-checked against the disk.

    A store-wide invariant rather than a post-condition of one call, and that is what
    makes a tombstone worth writing: it catches a restore-from-backup that reintroduces
    deleted records, a crash between the tombstone and the removals, and an erasure
    that skipped an artifact. Checked **by content**, not by the refs the tombstone
    lists: after a removal every position shifts, so a positional re-check would
    either need the pre-deletion file or agree with itself.

    **The converse is not checkable and is not claimed.** "Every gap in `events.jsonl`
    is explained by a tombstone" would be the stronger invariant, and it is false by
    construction here: sequence numbers are drawn from one counter shared by both
    streams, and `write_snapshot` and `rewrite_stream` each consume one while writing
    no record at all. So a gap has at least four innocent causes and is evidence of
    nothing — the same shape as an empty retrieval result, and the same fix if it is
    ever wanted (make the other consumers of a sequence number record that they took
    one). Until then this function checks the direction that is derivable.
    """
    root = Path(root)
    findings = []
    stones = tombstones(root)
    if not stones:
        return findings
    events = {e["seq"] for e in recall.load_events(root)} \
        if (root / recall.EVENTS_ARTIFACT).exists() else set()
    warm = {e["id"] for e in recall.load_episodes(root / ATTRIBUTION_ARTIFACT)} \
        if (root / ATTRIBUTION_ARTIFACT).exists() else set()
    card = load_card(root / recall.CARD_ARTIFACT) \
        if (root / recall.CARD_ARTIFACT).exists() else None
    rounds = recall.load_rounds(root / recall.RPS_ARTIFACT) \
        if (root / recall.RPS_ARTIFACT).exists() else ()
    for stone in stones:
        episode = stone["episode"]
        back = sorted(set(stone["removed"].get(recall.EVENTS_ARTIFACT, [])) & events)
        if back:
            findings.append(
                f"{episode}: {recall.EVENTS_ARTIFACT} still holds seq {back} that a "
                "tombstone says were removed — either the removals never ran or the "
                "file was restored over them")
        if episode in warm:
            findings.append(
                f"{episode}: {ATTRIBUTION_ARTIFACT} still has an entry for a "
                "tombstoned episode, so its summary and vector are still retrievable")
        if card is not None:
            citing = [i for i, f in enumerate(card.facts)
                      if episode in f.source_episode_ids]
            if citing:
                findings.append(
                    f"{episode}: {recall.CARD_ARTIFACT} fact(s) {citing} still cite a "
                    "tombstoned episode, so the card asserts a fact whose evidence "
                    "was deleted")
        naming = [i for i, r in enumerate(rounds) if r.get("episode") == episode]
        if naming:
            findings.append(
                f"{episode}: {recall.RPS_ARTIFACT} round(s) {naming} still name a "
                "tombstoned episode")
    return findings


# -------------------------------------------------------------------- retention
def datable(episodes: Iterable[Mapping]) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    """Split the warm layer into episodes that can be aged and ids that cannot.

    Two return values because they are two claims and one of them is a finding.
    `end_ts` is optional in the warm-layer schema — an episode summarised by a build
    that did not record it is a real state — and an undatable episode must be neither
    retained forever in silence nor deleted on the assumption that missing means old.
    A single filtered list would make the second choice look like the first.
    """
    dated, undatable = [], []
    for entry in episodes:
        end = entry.get("end_ts")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            undatable.append(entry["id"])
        else:
            dated.append(entry)
    return tuple(dated), tuple(sorted(undatable))


def expired(episodes: Iterable[Mapping], now: float, *,
            retention_s: float = RETENTION_S) -> tuple[str, ...]:
    """The ids whose retention period has elapsed at `now`, oldest first.

    The boundary is `now - end_ts >= retention_s`, and it is `>=` rather than `>` so
    that the period is the length of time the data is kept rather than that length
    plus one tick of whatever clock happens to be injected. Tested from both sides at
    one-second resolution, because an off-by-one here is a day of somebody's data.
    """
    if retention_s <= 0:
        raise EraseError(
            f"retention_s must be positive, got {retention_s!r}: a non-positive "
            "period expires every episode including the one that just closed, which "
            "reads in a sweep's receipt as a very thorough deletion")
    dated, _ = datable(episodes)
    due = [(e["end_ts"], e["id"]) for e in dated
           if float(now) - float(e["end_ts"]) >= retention_s]
    return tuple(eid for _end, eid in sorted(due))


# ------------------------------------------------------------------ verification
def verification_queries(root: Path, episode: str) -> tuple[str, ...]:
    """The queries a deletion of `episode` must be checked with. Call this **first**.

    `recall.deletion_findings` names the gap this closes: the lexical path drops
    zero-scoring documents, so a query bearing no resemblance to the deleted episode
    returns a clean zero from a path that genuinely ran. The queries therefore have to
    be drawn from the episode's own text — and that text only exists *before* the
    deletion, which is why this is a separate function the caller runs first rather
    than something a receipt could hand back. A receipt carrying the deleted text
    would be a copy of it, written into a log (see `TOMBSTONE_FIELDS`).

    Raises on an empty result rather than returning `()`: a verification with no
    queries passes for the same reason a broken one does.
    """
    root = Path(root)
    if not is_episode_id(episode):
        raise EraseError(f"{episode!r} is not a minted episode id")
    texts: list[str] = []
    path = root / ATTRIBUTION_ARTIFACT
    if path.exists():
        for entry in recall.load_episodes(path):
            if entry["id"] == episode:
                texts.append(entry["summary"])
    if (root / recall.EVENTS_ARTIFACT).exists() and path.exists():
        attrib = recall.attribution(recall.load_episodes(path))
        for event in recall.load_events(root):
            if attrib.get(event["seq"]) == episode:
                text = str(event.get("text", "")).strip()
                if text:
                    texts.append(text)
    if (root / recall.CARD_ARTIFACT).exists():
        card = load_card(root / recall.CARD_ARTIFACT)
        for fact in card.facts:
            if episode in fact.source_episode_ids:
                texts.append(fact.line())
    out = tuple(sorted({t for t in texts if t.strip()}))
    if not out:
        raise EraseError(
            f"{episode} has no text anywhere under {root}: there is no query that "
            "could distinguish 'deleted' from 'never asked about', so a verification "
            "built on this would pass for the wrong reason. Check the episode id")
    return out


# ---------------------------------------------------------------------- the act
def delete(store: Store, episode: str, *, goal: str,
           _after_step: Callable[[str], None] | None = None) -> Receipt:
    """Remove every trace of `episode` that this store knows how to reach.

    The root is `store.root` and there is no way to pass a different one: a deletion
    that read one directory and wrote another would report a clean receipt about a
    store it never touched, and no argument makes that impossible if the argument
    exists.

    Idempotent. A second call finds nothing to remove, appends a second tombstone
    saying so, and returns a receipt whose counts are zero — which is the state a
    crash mid-transaction leaves, so "run it again" is the whole recovery procedure.

    `_after_step` is the crash injection point, called with each step's name after it
    completes. A reviewable argument rather than a monkeypatch of an internal, for the
    reason `Store._allow_ephemeral` is one — and `tests/test_erase.py` derives from the
    source that nothing outside `tests/` passes it.
    """
    if goal not in DELETION_GOALS:
        raise EraseError(
            f"a deletion's reason must be one of {DELETION_GOALS}, got {goal!r}: "
            "every removal of personal data has to say which of the two things it is, "
            "because the sweep and a person's request have different audit obligations")
    if not is_episode_id(episode):
        raise EraseError(
            f"{episode!r} is not a minted episode id; memory.consolidate.episode_id is "
            "the one creator of that format. A deletion addressed by anything else "
            "would match nothing and report a clean receipt")
    root = Path(store.root)

    # ---- preconditions, all of them, before the first write.
    findings = registry_findings(root)
    if findings:
        raise EraseError(
            "refusing to delete while an artifact is unclassified — a receipt from "
            "this store would claim a completeness it cannot have: " + "; ".join(findings))
    if is_locked(recall.CARD_ARTIFACT):
        # Unconditional, not "only if the card cites this episode". A precondition that
        # fires only when there is something to delete is a precondition first tested
        # by the deletion that could not afford to fail; this way every deletion in the
        # store's life discovers it at the same moment. The lock is not lifted here
        # either: a retention sweep that could quietly unlock the profile card is a
        # lock that describes nothing.
        raise EraseError(
            f"{recall.CARD_ARTIFACT} is still a read-only seed, so a fact citing "
            f"{episode} could not be removed from it and this deletion would be "
            "incomplete while reporting success. memory.store.unlock_for_s5 is the "
            "one way past, and it belongs at node start-up, not inside a deletion")
    if not (root / ATTRIBUTION_ARTIFACT).exists():
        raise EraseError(
            f"no {ATTRIBUTION_ARTIFACT} under {root}: without the attribution map no "
            f"event can be shown to belong to {episode}, so a deletion would remove "
            "nothing from the event stream and say it was clean")
    attrib = recall.attribution(recall.load_episodes(root / ATTRIBUTION_ARTIFACT))

    # ---- plan everything against the pre-deletion state.
    plans = {name: ERASERS[name](root, episode, attrib=attrib)
             for name in erase_order()}
    soft = [f for name in erase_order() for f in plans[name].findings]

    # ---- the claim, first and durable. See the module docstring on this ordering.
    tomb = {"kind": TOMBSTONE_KIND, "episode": episode, "why": {"goal": goal},
            "removed": {name: list(plans[name].refs) for name in erase_order()
                        if plans[name].refs}}
    parse_tombstone({**tomb, "seq": 0, "ts": 0.0}, where="<planned tombstone>")
    tombstone_seq = store.append("audit", tomb)
    store.flush(tombstone_seq)
    if _after_step is not None:
        _after_step("tombstone")

    # ---- then make the world match it.
    removed: dict[str, tuple] = {}
    for name in erase_order():
        plan = plans[name]
        if plan.drop_seqs is not None:
            dropped = store.rewrite_stream("events", plan.drop_seqs)
            removed[name] = dropped
            absent = sorted(set(plan.drop_seqs) - set(dropped))
            if absent:
                # Not an error: this is what the second run of the same deletion
                # looks like, and it is the difference between idempotent and merely
                # repeatable. Reported because "already gone" and "never there" are
                # the same observation and only one of them is expected.
                soft.append(
                    f"{name}: seq {absent} were planned for removal and were already "
                    "absent — expected on a re-run after a crash, and a finding on a "
                    "first run because it means something else edited the stream")
        elif plan.body is not None:
            store.write_snapshot(name, plan.body)
            store.flush()
            removed[name] = plan.refs
        else:
            removed[name] = ()
        if _after_step is not None:
            _after_step(name)

    return Receipt(episode, goal, tombstone_seq, removed, soft, RETAINED)


def sweep(store: Store, *, now: float | None = None,
          retention_s: float = RETENTION_S,
          clock: Callable[[], float] = time.time) -> tuple[Receipt, ...]:
    """Delete every episode whose retention period has elapsed. Oldest first.

    Oldest first so that an interruption leaves the *newest* data — a sweep killed
    half way through has deleted the most-expired episodes rather than an arbitrary
    subset, which is the only ordering under which "run it again" converges on the
    same state as one uninterrupted run.

    One receipt per episode rather than one for the sweep: a per-episode failure must
    fail that episode and not the run (§6.0 rule 5), and a single aggregate receipt
    cannot say which of forty episodes the finding belongs to.
    """
    root = Path(store.root)
    path = root / ATTRIBUTION_ARTIFACT
    if not path.exists():
        return ()
    entries = recall.load_episodes(path)
    _dated, undatable = datable(entries)
    at = clock() if now is None else float(now)
    receipts = []
    for episode in expired(entries, at, retention_s=retention_s):
        receipt = delete(store, episode, goal=RETENTION_GOAL)
        if undatable:
            # Attached to every receipt rather than returned separately: an episode
            # with no `end_ts` is never swept, and a sweep that reported "3 deleted"
            # with the undatable ones counted nowhere is how data outlives its
            # retention period without anything looking wrong.
            receipt = Receipt(
                receipt.episode, receipt.goal, receipt.tombstone_seq, receipt.removed,
                (*receipt.findings,
                 f"{len(undatable)} episode(s) have no end_ts and can never expire: "
                 f"{list(undatable[:5])}… — they are outside every retention period, "
                 "and a sweep cannot decide whether missing means new or ancient"),
                receipt.retained)
        receipts.append(receipt)
    return tuple(receipts)
