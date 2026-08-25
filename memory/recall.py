"""memory/recall.py — the retrieval side of memory, and the five paths a deletion must reach.

Zero ROS, zero boto3, zero network, zero LLM. The store root, the clock and the
text encoder are all injected, so every path in here runs under pytest on a
laptop; nothing in this module needs the instance or a model.

**Why a path reports a status and not just a list of hits.**
Design item T-C4 states the deletion criterion as "delete episode E, query all
five retrieval paths, **0 hits each**". A path that never ran returns an empty
list, so that criterion is satisfied — byte-for-byte, in the passing direction —
by a semantic path with no index built, an artifact that is missing, or a path
skipped by the deadline. And that is precisely the defect T-C4's own named mutant
is supposed to catch ("remove the index rebuild → red"): with the index left
holding the deleted episode, an unbuilt-and-therefore-empty semantic path reports
a clean zero while the data is still on disk. So a query returns a status per
path, `0 hits` is only meaningful together with `ran`, and
`deletion_findings()` reports a path that did not run as a finding rather than as
a clean zero. This is the same rule `infra/check_deploy.sh` encodes in its exit
code 3: a guard that could not run must never report clean.

**Why the five paths are derived rather than written down, on both sides.**
"All five retrieval paths" is a five-name list, and a five-name list cannot notice
a sixth. Two derivations, because one of them alone leaves a hole:

* A path can only exist as a `_run_<name>` method on `Retriever`, and `query()`
  dispatches by iterating the registry. A retrieval function added outside the
  registry has nowhere to live, and `tests/test_recall.py` asserts the method set
  and `PATHS` are equal — so a new method without a registration reds, and a
  registration without a method reds.
* Each path declares the durable artifact(s) it reads, and `coverage_findings()`
  compares that against the artifacts actually **on disk**. An artifact nothing
  reads is a finding, because a deletion cannot prove it removed anything from a
  file no path queries. Note there is nothing to derive this from statically:
  `Store.write_snapshot` accepts any plain file name, so there is no enum of
  snapshots anywhere in the tree — the disk is the only honest producer.

**Why the deadline is a parameter, and why the card is exempt from it.**
Plan §2 gives recall a 150 ms hard deadline after which the turn proceeds on the
profile card alone (T-F3). A skipped path therefore has to be distinguishable from
an empty one for the reason above, so the deadline sets a status. The card path is
exempt on purpose: it is the fallback the deadline degrades *to*, and a deadline that
is allowed to skip the fallback leaves the degraded turn with nothing. The exemption
is a field on the path's own spec, not a position in the run order, because an order
does not *create* that invariant — move `card` last and the exemption still runs it,
which is the only version of this claim that a mutant can be pointed at.

**Why there is no fused ranking across paths.**
The plan's T-F1 says "top-5 hit rate", which reads like one ranked list. BM25 is
unbounded and cosine is at most 1.0, so a single list ordered by score is ordered
by a quantity that does not exist: it would rank a lexical hit above a semantic one
because of the scoring scale, not because of relevance, and the calibration that
would make them comparable has never been measured here. So hits are ranked
**within** a path, `hits()` orders deterministically by `(path, -score, ref)`, and
the 30 frozen queries score each path separately. A fused list can be added the
day somebody measures the calibration; asserting one now would be a number in
prose (§6.0 rule 6) wearing a function's clothes.

**Why there is no in-process cache.**
Reloading the artifacts per query costs I/O inside a 150 ms budget, and caching
them is the obvious fix. It is also exactly the T-C4 mutant: a cached index that
survives a deletion returns the deleted episode while the file on disk is clean,
and `deletion_findings()` — reading the cache — would report zero. A cache is
therefore a second claim (an invalidation contract with the deletion
transaction), not an optimisation, and it does not land in the same stage as the
paths it would invalidate.

**Why the tokeniser is imported and not written here.**
`memory.salience.bigrams` already decided this: character bigrams, because 中文
needs a segmenter and 粵語 segmenters are worse. Two tokenisers is two definitions
of what a term is, and the retrieval bucket would split by which module produced
it — the same failure §S4.5 refused when it declined to normalise `why.goal`.
Honest boundary: BM25 over character bigrams is not word BM25. It has no published
baseline on this kind of text and it is not claimed to; its job is to be a
deterministic, regression-testable lexical path.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from memory.consolidate import is_episode_id
from memory.profile import TRUST_SEED, ProfileError
from memory.profile import parse as parse_card
from memory.salience import bigrams
from memory.store import StoreError, filter_by_goal


class RecallError(StoreError):
    """A malformed artifact, an unusable query, or an encoder that will not answer."""


# ------------------------------------------------------------------ artifacts
#
# The durable files this module reads. Named constants rather than literals at the
# use sites, because `coverage_findings()` joins these names against the directory
# listing and a typo would silently become "an artifact nothing reads".
EVENTS_ARTIFACT = "events.jsonl"
EPISODES_ARTIFACT = "episodes.json"   # S5's warm layer: one entry per closed episode
CARD_ARTIFACT = "profile.json"
RPS_ARTIFACT = "rps_table.json"

# What counts as a durable artifact when censusing the store root. Suffix-based on
# purpose: a *new* file written by a future node is the case this has to notice, and
# a name list could not. Temp files and dot-files are excluded because `Store`
# creates `.profile.json.<rand>.tmp` during an atomic replace, and a census that
# flagged those would report a finding for a healthy write in progress.
ARTIFACT_SUFFIXES = (".json", ".jsonl")

# ------------------------------------------------------------------- statuses
RAN = "ran"
NO_ARTIFACT = "no_artifact"    # the file this path reads does not exist yet
NO_ENCODER = "no_encoder"      # no embedding function was injected
DEADLINE = "deadline"          # the 150 ms budget was gone before this path started
STATUSES = (RAN, NO_ARTIFACT, NO_ENCODER, DEADLINE)

# ----------------------------------------------------------------- attribution
#
# What a hit says when it does not name an episode. Two markers rather than one
# `None`, because an absent episode means two different things and exactly one of
# them is a defect:
#
#   SEED    a hand-written seed fact. `profile.Fact` exempts `trust == "seed"` from
#           `source_episode_ids` on purpose, so this hit is outside the cascade by
#           construction. Nothing is wrong.
#   ORPHAN  an item that should have named an episode and does not. This is the
#           cheapest way to pass a cascade test without deleting anything: remove
#           the episode entry, leave the events, and every reference to the deleted
#           episode silently becomes a reference to nothing.
#
# A single `None` would merge the case that is fine with the case that is the bug,
# and `deletion_findings` would then have to either flag every seed fact forever or
# stay quiet about every orphan.
SEED = "seed"
ORPHAN = "orphan"
NOT_AN_EPISODE = (SEED, ORPHAN)

# Plan §2's hard deadline for the recall service, and the default top-k.
DEADLINE_S = 0.150
TOP_K = 5

# Standard BM25 constants. Written out because a "tuned" value would be a number in
# prose: nothing here has measured them on this corpus, and these are the textbook
# defaults, which is a claim this module can actually make.
BM25_K1 = 1.5
BM25_B = 0.75


class PathSpec:
    """One retrieval path's declaration: what it reads and what it needs to run."""

    __slots__ = ("name", "reads", "needs_encoder", "deadline_exempt")

    def __init__(self, name: str, reads: tuple[str, ...], *,
                 needs_encoder: bool = False,
                 deadline_exempt: bool = False) -> None:
        if not reads:
            raise RecallError(
                f"path {name!r} declares no artifact: a path that reads nothing is "
                "one a deletion can never prove it emptied")
        self.name = name
        self.reads = reads
        self.needs_encoder = needs_encoder
        self.deadline_exempt = deadline_exempt


PATHS: dict[str, PathSpec] = {
    # The profile card. Exempt from the deadline because it *is* the degraded turn.
    "card": PathSpec("card", (CARD_ARTIFACT,), deadline_exempt=True),
    # Structured filter: goal and episode, no scoring. The cheapest path and the
    # only one whose answer is a set rather than a ranking.
    "structured": PathSpec("structured", (EVENTS_ARTIFACT, EPISODES_ARTIFACT)),
    # Lexical: BM25 over character bigrams of the event text.
    "lexical": PathSpec("lexical", (EVENTS_ARTIFACT, EPISODES_ARTIFACT)),
    # Semantic: cosine over the episode summaries' stored vectors.
    "semantic": PathSpec("semantic", (EPISODES_ARTIFACT,), needs_encoder=True),
    # The opponent transition table. Personal behavioural data about one person, so
    # it is inside the same retention period and the same cascade (plan §4).
    "rps": PathSpec("rps", (RPS_ARTIFACT,)),
}

# Declared, not `sorted(PATHS)`, and asserted set-equal to `PATHS` in tests — so a
# path that is registered but not ordered reds instead of being skipped, and one that
# is ordered but not registered reds instead of being silently absent.
#
# What this tuple does *not* buy: it is not what keeps the fallback alive under the
# deadline. `deadline_exempt` is. Running `card` first only means the fallback does not
# spend the budget it is the fallback for; move it last and it still runs, because the
# exemption is checked per path rather than inferred from position. Saying the order
# protects the fallback would be an invariant with no line that creates it.
PATH_ORDER = ("card", "structured", "lexical", "semantic", "rps")


def paths() -> tuple[str, ...]:
    """The retrieval paths, in run order. The registry, which deletion iterates."""
    return PATH_ORDER


def artifacts(root: Path) -> tuple[str, ...]:
    """The durable artifacts present in `root`, sorted. Derived from the disk.

    Derived rather than declared because the case worth catching is a file some
    future node writes that no retrieval path reads — and a declared list is
    exactly what cannot notice that.
    """
    root = Path(root)
    if not root.exists():
        return ()
    found = [p.name for p in root.iterdir()
             if p.is_file() and not p.name.startswith(".")
             and p.suffix in ARTIFACT_SUFFIXES]
    return tuple(sorted(found))


def coverage_findings(root: Path) -> list[str]:
    """Artifacts in `root` that no retrieval path reads. Empty list = full coverage.

    This is the half of T-C4 that a hand-written five-path list cannot provide. A
    findings list rather than a raise, like `store.location_findings` and
    `profile.budget_findings`: the node logs each one and the test asserts on the
    text, so an uncovered artifact and a missing directory are not the same
    observation.
    """
    read = {name for spec in PATHS.values() for name in spec.reads}
    findings = []
    for name in artifacts(root):
        if name not in read:
            findings.append(
                f"{name} is a durable artifact that no retrieval path reads — a "
                f"deletion cannot show it removed anything from it. The paths read "
                f"{sorted(read)}")
    return findings


# --------------------------------------------------------------------- results
class Hit:
    """One retrieved item. Immutable, and it always accounts for its provenance.

    `episode` is either a minted episode id or one of `NOT_AN_EPISODE`. There is no
    `None`: every producer has to *decide* which of the two non-episode cases it is
    looking at, and `attrib.get(seq)` quietly returning `None` is exactly the
    decision this refuses to let anyone skip.
    """

    __slots__ = ("path", "episode", "text", "score", "ref")

    def __init__(self, path: str, episode: str, text: str, score: float,
                 ref: str) -> None:
        if path not in PATHS:
            raise RecallError(f"unknown path {path!r}; the registry is {paths()}")
        if episode not in NOT_AN_EPISODE and not is_episode_id(episode):
            raise RecallError(
                f"{path}: episode {episode!r} is neither a minted episode id nor one "
                f"of {NOT_AN_EPISODE} — memory.consolidate.episode_id is the one "
                "creator of that format, and 'no episode' has to say which kind")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "score", float(score))
        object.__setattr__(self, "ref", str(ref))

    def __setattr__(self, name: str, value: Any) -> None:
        raise RecallError(
            f"a Hit is immutable ({name}={value!r}): the same object is handed to the "
            "prompt assembler and to the deletion check, and an edit between those "
            "two readers would make them disagree with nothing in any log to say so")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def as_dict(self) -> dict:
        return {"path": self.path, "episode": self.episode, "text": self.text,
                "score": self.score, "ref": self.ref}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Hit({self.path!r}, {self.episode!r}, {self.score:.4f}, {self.ref!r})"


class PathResult:
    """What one path did: its status, and its hits if it ran."""

    __slots__ = ("name", "status", "hits")

    def __init__(self, name: str, status: str, hits: Sequence[Hit] = ()) -> None:
        if name not in PATHS:
            raise RecallError(f"unknown path {name!r}; the registry is {paths()}")
        if status not in STATUSES:
            raise RecallError(f"unknown status {status!r}; the set is {STATUSES}")
        if status != RAN and hits:
            raise RecallError(
                f"{name}: status {status!r} with {len(hits)} hit(s) — a path that did "
                "not run cannot have found anything, and this combination is how a "
                "stale result gets reported as a fresh one")
        self.name = name
        self.status = status
        self.hits = tuple(hits)

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "hits": [h.as_dict() for h in self.hits]}


class Result:
    """One query's answer across every registered path."""

    __slots__ = ("query", "results")

    def __init__(self, query: str, results: Sequence[PathResult]) -> None:
        names = [r.name for r in results]
        if sorted(names) != sorted(paths()):
            raise RecallError(
                f"a result must carry every registered path exactly once; got "
                f"{sorted(names)} against {sorted(paths())} — a missing path is a "
                "path a deletion check would silently not examine")
        self.query = query
        self.results = tuple(results)

    def status_of(self, name: str) -> str:
        for r in self.results:
            if r.name == name:
                return r.status
        raise RecallError(f"unknown path {name!r}; the registry is {paths()}")

    def hits(self, k: int | None = TOP_K) -> tuple[Hit, ...]:
        """Every hit, ranked within its path. Deterministic, and never fused.

        `k` is applied **per path**, not to the concatenation: taking the global top
        five off a list ordered by `(path, …)` would return five hits from whichever
        path sorts first, which is an ordering by name dressed up as relevance.
        `k=None` means every hit, and the deletion check uses it — a truncated view
        is a check that stops looking after five, and the sixth hit is still a
        reference to the episode that was supposed to be gone.
        """
        out: list[Hit] = []
        for name in paths():
            for r in self.results:
                if r.name == name:
                    ranked = sorted(r.hits, key=lambda h: (-h.score, h.ref))
                    out.extend(ranked if k is None else ranked[:int(k)])
        return tuple(out)

    def census(self) -> dict[str, int]:
        """Paths per status — **every member of `STATUSES`, zeros included**.

        Omitting the zeros makes "no path was skipped" and "the skipped-path counter
        was removed" the same output, which is the reading that lets a degraded turn
        look like a healthy one for as long as nobody re-reads the code.
        """
        counts = {status: 0 for status in STATUSES}
        for r in self.results:
            counts[r.status] += 1
        return counts

    def as_dict(self) -> dict:
        return {"query": self.query,
                "results": [r.as_dict() for r in self.results],
                "census": self.census()}


def deletion_findings(result: Result, episode: str) -> list[str]:
    """Everything wrong with `result` as evidence that `episode` is gone.

    This is the primitive T-C4 is built on, and it exists in this stage — before any
    deletion path does — because the criterion it replaces is satisfied by the
    failure. Three findings, and the third is the one that is easy to miss:

    * **A path that did not run.** "0 hits" from a path with no index, no artifact,
      or no time left is not evidence of anything.
    * **A hit that still names the episode.** The obvious one.
    * **An `ORPHAN` hit.** Deleting the *episode entry* without deleting the events
      it covered converts every one of those events from "attributed to E" into
      "attributed to nothing", so an attribution-based check reads 0 hits for E while
      the text is still retrievable. Orphaning is the cheapest way to pass a cascade
      test without deleting anything. `SEED` hits are clean: a hand-written fact is
      outside the cascade by construction, which is why those are two markers and not
      one absent value.

    Note the consequence for *when* a sweep may run: an episode that is still open
    legitimately has events in no episode, so a check run mid-conversation reports
    orphan findings. That is correct for a check — the sweep's goal is
    `scheduled_retention`, and it runs when nothing is open.

    **What this function cannot check, stated rather than left implicit.** It reads a
    `Result`, so it is only evidence about the query that produced it. The lexical
    path drops zero-scoring documents, so a query bearing no resemblance to the
    deleted episode returns an empty lexical result from a path that genuinely `ran` —
    a clean zero, obtained by asking the wrong question. The caller therefore has to
    build its queries out of the deleted episode's **own** text, and T-C4's harness is
    where that requirement gets a counting assertion (the fixture must prove the query
    would have hit before the deletion, or the test is green for the same reason a
    broken one is). Naming the gap here because the place a guard says it need not look
    is where the next instance hides.
    """
    if not is_episode_id(episode):
        raise RecallError(f"{episode!r} is not a minted episode id")
    findings = []
    for name in paths():
        status = result.status_of(name)
        if status != RAN:
            findings.append(
                f"{name} did not run ({status}) — 0 hits from a path that never "
                "executed is not evidence that anything was removed from it")
    for hit in result.hits(k=None):
        if hit.episode == episode:
            findings.append(
                f"{hit.path} still returns {episode} ({hit.ref}): "
                f"{hit.text[:40]!r}")
        elif hit.episode == ORPHAN:
            findings.append(
                f"{hit.path} returns an unattributed item ({hit.ref}) — deleting an "
                "episode entry without its events turns a reference into an orphan, "
                "and an orphan satisfies this check while staying retrievable")
    return findings


# --------------------------------------------------------------------- scoring
def _pack(text: str) -> tuple[str, ...]:
    """The terms of one document. One creator, imported from `memory.salience`."""
    return bigrams(text)


def bm25(query: str, documents: Sequence[str]) -> tuple[float, ...]:
    """Okapi BM25 of `query` against each document. Deterministic, no dependencies.

    Returns a score per document in the documents' order — not a ranking — so the
    caller decides the tie-break and the ordering never depends on a `dict` or `set`
    iteration order (the T-B4 defect one layer up).
    """
    docs = [_pack(d) for d in documents]
    if not docs:
        return ()
    lengths = [len(d) for d in docs]
    total = sum(lengths)
    avg = total / len(docs) if total else 0.0
    q_terms = sorted(set(_pack(query)))
    scores = [0.0] * len(docs)
    for term in q_terms:
        containing = sum(1 for d in docs if term in d)
        if not containing:
            continue
        # The +0.5/+0.5 form, which stays positive for a term in every document.
        # The textbook `log(N/df)` goes to zero there and the "classic" variant goes
        # negative, which would let a common term *reduce* a document's score below
        # a document that does not contain it at all.
        idf = math.log(1.0 + (len(docs) - containing + 0.5) / (containing + 0.5))
        for i, doc in enumerate(docs):
            tf = doc.count(term)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (lengths[i] / avg if avg else 0.0))
            scores[i] += idf * (tf * (BM25_K1 + 1.0)) / denom
    return tuple(scores)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Raises on a dimension mismatch rather than scoring it.

    A mismatch scored as 0.0 is a semantic path that runs, reports `ran`, and is
    blind — the exact combination the status field exists to make impossible. This
    happens for real when the encoder is swapped without rebuilding the index, which
    is the same class of defect as the cached index this module refuses.
    """
    if len(a) != len(b):
        raise RecallError(
            f"embedding dimension mismatch: query {len(a)} vs stored {len(b)} — the "
            "index was built by a different encoder, and scoring that as 0.0 would "
            "make a blind path indistinguishable from an empty one")
    if not a:
        raise RecallError("an empty embedding cannot be compared")
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise RecallError("a zero vector has no direction; refusing to score it")
    return dot / (na * nb)


# ------------------------------------------------------------------- artifacts
def load_episodes(path: Path) -> tuple[dict, ...]:
    """The warm layer: one validated entry per closed episode.

    Validation is as strict as `profile.parse` and for the same reasons: an unknown
    field is a setting its author believes is on, and an id that is not a minted one
    is a reference that resolves to nothing. `embedding` is optional — an episode
    summarised before the encoder existed is a real state — but a *present* one must
    be a non-empty list of numbers, because a `null` embedding scored as a zero
    vector is another blind-but-`ran` path.
    """
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecallError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(obj, dict) or obj.get("version") != 1:
        raise RecallError(f"{path}: expected a v1 object, got {type(obj).__name__} "
                          f"with version {obj.get('version') if isinstance(obj, dict) else None!r}")
    raw = obj.get("episodes")
    if not isinstance(raw, list):
        raise RecallError(f"{path}: 'episodes' must be a list")
    known = {"id", "seqs", "summary", "embedding", "activity", "start_ts", "end_ts"}
    out = []
    for n, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise RecallError(f"{path}: episode {n} is not an object")
        unknown = sorted(set(entry) - known)
        if unknown:
            raise RecallError(f"{path}: episode {n} has unknown field(s) {unknown} — "
                             "a silently ignored field is a setting its author "
                             "believes is on")
        eid = entry.get("id")
        if not is_episode_id(eid):
            raise RecallError(
                f"{path}: episode {n} has id {eid!r}, which is not a minted episode "
                "id; memory.consolidate.episode_id is the one creator of that format")
        seqs = entry.get("seqs")
        if not isinstance(seqs, list) or not seqs or \
                not all(isinstance(s, int) and not isinstance(s, bool) for s in seqs):
            raise RecallError(f"{path}: episode {n} ({eid}) needs a non-empty list of "
                              f"integer store sequence numbers, got {seqs!r}")
        summary = entry.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise RecallError(f"{path}: episode {n} ({eid}) has no summary — an empty "
                             "summary makes every semantic query miss it silently")
        vec = entry.get("embedding")
        if vec is not None:
            if not isinstance(vec, list) or not vec or \
                    not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            for x in vec):
                raise RecallError(
                    f"{path}: episode {n} ({eid}) has embedding {type(vec).__name__} "
                    "— a present embedding must be a non-empty list of numbers, "
                    "because a null one scored as a zero vector is a blind path "
                    "reporting that it ran")
        out.append(dict(entry))
    ids = [e["id"] for e in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RecallError(f"{path}: duplicate episode id(s) {dupes} — two entries "
                          "under one id means a deletion removes one and leaves one")
    return tuple(out)


def attribution(episodes: Iterable[Mapping]) -> dict[int, str]:
    """seq → episode id. Raises if one sequence number belongs to two episodes.

    The raise is not defensive: the segmenter partitions the stream, so an overlap
    means the warm layer was written by something that is not the segmenter, and a
    deletion driven by an ambiguous map would remove one reference and leave the
    other.
    """
    out: dict[int, str] = {}
    for entry in episodes:
        for seq in entry["seqs"]:
            if seq in out and out[seq] != entry["id"]:
                raise RecallError(
                    f"store sequence {seq} is claimed by both {out[seq]} and "
                    f"{entry['id']} — episodes partition the stream, so this map "
                    "was not written by memory.segment")
            out[seq] = entry["id"]
    return out


# ------------------------------------------------------------------- retriever
class Retriever:
    """Every retrieval path, over one store root. No ROS, no network, no cache.

    `clock` is `time.monotonic` by default and injected everywhere else, so the
    deadline tests assert an exact schedule instead of racing the machine — the
    `feedback_harness_test_measures_the_machine` lesson, applied before it can bite.

    `encoder` is a `Callable[[str], Sequence[float]]` and is **absent by default**.
    In production it is `Alibaba-NLP/gte-multilingual-base` at a pinned commit
    (`trust_remote_code`), which is a torch import and a 305M-parameter download:
    exactly the kind of dependency §6.1 says a pure module must not acquire. Absent
    means the semantic path reports `no_encoder`, which is a status, not an empty
    result.
    """

    def __init__(self, root: Path, *, encoder: Callable[[str], Sequence[float]] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 deadline_s: float = DEADLINE_S) -> None:
        self.root = Path(root)
        self.encoder = encoder
        self._clock = clock
        self.deadline_s = float(deadline_s)
        if self.deadline_s <= 0.0:
            raise RecallError(
                f"deadline_s must be positive, got {self.deadline_s!r}: a "
                "non-positive budget skips every non-exempt path on every turn, "
                "which reads in the census as a machine that is merely slow")

    # ------------------------------------------------------------- public API
    def query(self, text: str, *, goals: Sequence[str] | None = None,
              k: int = TOP_K) -> Result:
        """Run every registered path against `text` and return their statuses and hits.

        An empty or whitespace-only query is refused. Precisely: it silences the two
        query-scored paths — `lexical` scores every document 0.0 and `card` shares no
        bigram with anything — while both still report `ran`, which is the one
        combination that is indistinguishable from a store holding nothing. (It does
        *not* silence `structured`, which is a filter and returns the same records for
        any query; that asymmetry is the reason the refusal is stated in terms of the
        scored paths rather than "every path".) Same failure as a valid-but-wrong Cost
        Explorer filter returning $0 exactly like an ingestion lag.
        """
        if not isinstance(text, str) or not text.strip():
            raise RecallError(
                f"refusing an empty query ({text!r}): the scored paths return nothing "
                "from it while reporting that they ran, which is indistinguishable "
                "from a store that holds nothing")
        start = self._clock()
        results = []
        for name in paths():
            spec = PATHS[name]
            if not spec.deadline_exempt and self._clock() - start >= self.deadline_s:
                results.append(PathResult(name, DEADLINE))
                continue
            results.append(self._dispatch(spec, text, goals, k))
        return Result(text, results)

    def _dispatch(self, spec: PathSpec, text: str, goals, k: int) -> PathResult:
        # Status precedence is this code order — encoder, then artifact — and it is
        # fixed on purpose rather than left to whichever check happens to be written
        # first: a path missing both would otherwise report a different status
        # depending on an edit, and the census is compared across runs.
        if spec.needs_encoder and self.encoder is None:
            return PathResult(spec.name, NO_ENCODER)
        for artifact in spec.reads:
            if not (self.root / artifact).exists():
                return PathResult(spec.name, NO_ARTIFACT)
        runner = getattr(self, f"_run_{spec.name}")
        return PathResult(spec.name, RAN, runner(text, goals, k))

    # ------------------------------------------------------------- the paths
    #
    # One `_run_<name>` per registry entry, and the equality of those two sets is
    # asserted in tests. A retrieval function that is not a `_run_` method is not
    # reachable from `query`, and one that is has to be registered to be called.
    def _run_card(self, text: str, goals, k: int) -> list[Hit]:
        """The resident profile card, matched on shared bigrams with the query.

        No scoring: the card is ~17 facts and it is resident in the prompt anyway, so
        a ranking over it would be a ranking over noise. Every hit gets `score=1.0`,
        which is honest — it is a filter, not a ranker.

        **One hit per source episode, not per fact.** A fact cites
        `source_episode_ids`, plural, and a fact sourced to `[ep-000001, ep-000007]`
        that reported only the first would tell a deletion of `ep-000007` that the
        card is clean while the row still cites it. Seed facts have no sources by
        construction (`profile.TRUST_SEED`) and get the `SEED` marker.
        """
        try:
            obj = json.loads((self.root / CARD_ARTIFACT).read_text(encoding="utf-8"))
            card = parse_card(obj, where=str(self.root / CARD_ARTIFACT))
        except (json.JSONDecodeError, ProfileError) as exc:
            raise RecallError(f"{CARD_ARTIFACT}: {exc}") from exc
        terms = set(bigrams(text))
        hits = []
        for i, fact in enumerate(card.facts):
            line = fact.line()
            if not (terms & set(bigrams(line))):
                continue
            sources = list(fact.source_episode_ids)
            if not sources:
                # `Fact` already guarantees this only happens for a seed; the branch
                # is explicit so a future trust level cannot arrive here silently.
                marker = SEED if fact.trust == TRUST_SEED else ORPHAN
                hits.append(Hit("card", marker, line, 1.0, f"{CARD_ARTIFACT}#{i}"))
                continue
            for eid in sources:
                hits.append(Hit("card", eid, line, 1.0,
                                f"{CARD_ARTIFACT}#{i}@{eid}"))
        return hits

    def _run_structured(self, text: str, goals, k: int) -> list[Hit]:
        """Goal filter over the event stream. The one path whose answer is a set.

        `goals=None` means every goal, and it is spelled as `None` rather than as an
        empty sequence: `filter_by_goal(records, [])` legitimately returns nothing,
        so an accidentally-empty list would be an all-paths-empty result that
        reported `ran`.
        """
        events = self._events()
        if goals is not None:
            events = filter_by_goal(events, goals)
        attrib = attribution(load_episodes(self.root / EPISODES_ARTIFACT))
        hits = []
        for event in events:
            hits.append(Hit("structured", attrib.get(event["seq"], ORPHAN),
                            str(event.get("text", "")), 1.0,
                            f"{EVENTS_ARTIFACT}#{event['seq']}"))
        return hits

    def _run_lexical(self, text: str, goals, k: int) -> list[Hit]:
        """BM25 over the event text. Scores every event, returns those above zero."""
        events = self._events()
        if goals is not None:
            events = filter_by_goal(events, goals)
        attrib = attribution(load_episodes(self.root / EPISODES_ARTIFACT))
        docs = [str(e.get("text", "")) for e in events]
        scores = bm25(text, docs)
        hits = []
        for event, doc, score in zip(events, docs, scores):
            if score > 0.0:
                hits.append(Hit("lexical", attrib.get(event["seq"], ORPHAN), doc,
                                score, f"{EVENTS_ARTIFACT}#{event['seq']}"))
        return hits

    def _run_semantic(self, text: str, goals, k: int) -> list[Hit]:
        """Cosine over the stored episode vectors.

        An episode with no embedding is skipped **and that is a reportable state**,
        not a silent one: `unembedded()` names them, because an index missing half
        its episodes answers every query without ever saying so.
        """
        episodes = load_episodes(self.root / EPISODES_ARTIFACT)
        vector = list(self.encoder(text))
        if not vector:
            raise RecallError("the encoder returned an empty vector for the query")
        hits = []
        for entry in episodes:
            stored = entry.get("embedding")
            if stored is None:
                continue
            hits.append(Hit("semantic", entry["id"], entry["summary"],
                            cosine(vector, stored), f"{EPISODES_ARTIFACT}#{entry['id']}"))
        return hits

    def _run_rps(self, text: str, goals, k: int) -> list[Hit]:
        """The opponent transition table, as one hit per episode that fed it.

        The table is behavioural data about one person, so plan §4 puts it inside the
        same retention period and the same cascade as everything else. What makes it
        a *retrieval path* rather than a file is that a deletion has to be able to
        show it no longer contains the deleted episode's rounds — which requires the
        table to record which episode each count came from. A table that stores only
        counts cannot answer that, so this path reports what the table actually
        carries: if the file has no per-episode attribution, every hit is
        unattributed and `deletion_findings` says so instead of reporting a clean zero.
        """
        path = self.root / RPS_ARTIFACT
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecallError(f"{RPS_ARTIFACT}: not valid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise RecallError(f"{RPS_ARTIFACT}: expected an object, "
                              f"got {type(obj).__name__}")
        rounds = obj.get("rounds")
        if not isinstance(rounds, list):
            raise RecallError(
                f"{RPS_ARTIFACT}: expected a 'rounds' list. A table of bare counts "
                "cannot say which episode a count came from, so no deletion could "
                "ever prove it removed one")
        hits = []
        for i, entry in enumerate(rounds):
            if not isinstance(entry, dict):
                raise RecallError(f"{RPS_ARTIFACT}: round {i} is not an object")
            eid = entry.get("episode")
            if eid is None:
                # Not an error: a table written before the attribution field existed
                # is a real state. It is an ORPHAN, so the deletion check reports it
                # instead of counting it as a clean zero.
                eid = ORPHAN
            elif not is_episode_id(eid):
                raise RecallError(f"{RPS_ARTIFACT}: round {i} has episode {eid!r}, "
                                  "which is not a minted episode id")
            label = f"{entry.get('user', '?')}/{entry.get('outcome', '?')}"
            hits.append(Hit("rps", eid, label, 1.0, f"{RPS_ARTIFACT}#{i}"))
        return hits

    # ------------------------------------------------------------- internals
    def _events(self) -> list[dict]:
        """Every event record, validated the way `Store.read` validates it.

        Deliberately not `Store.read`: constructing a `Store` starts a writer
        thread, and a retrieval module that spawns the single writer in order to read
        is a second writer waiting to happen. The parse strictness is the same — a
        line that does not parse raises, because a truncated line is the failure the
        store exists to prevent and skipping it would make that failure unobservable.
        """
        path = self.root / EVENTS_ARTIFACT
        out = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecallError(
                    f"{EVENTS_ARTIFACT} line {n} is not valid JSON ({exc}) — a "
                    f"partial line means two writers, {len(line)} bytes") from exc
            if "seq" not in record:
                raise RecallError(
                    f"{EVENTS_ARTIFACT} line {n} has no 'seq': the store stamps every "
                    "record, so this did not come from it, and it cannot be attributed "
                    "to an episode or removed by one")
            out.append(record)
        return out

    def unembedded(self) -> tuple[str, ...]:
        """Episode ids in the warm layer with no vector. Reported, never assumed away.

        An index that silently covers 40% of the episodes answers every query and is
        wrong about all of them; T-F1's regression line would move and nothing would
        say why.
        """
        path = self.root / EPISODES_ARTIFACT
        if not path.exists():
            return ()
        return tuple(e["id"] for e in load_episodes(path)
                     if e.get("embedding") is None)
