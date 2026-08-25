"""memory/segment.py — where one episode ends and the next begins. Deterministic.

Zero ROS, zero boto3, zero network, **and no clock at all**: `tick(now)` takes the
time as an argument. §6.1 asks for an injectable clock; a parameter is the strongest
form of that, because there is no default to forget to inject and no monkeypatching
in the tests. The node owns the timer, this module owns the decision.

**Why there is a `tick()` and not only a `feed()`.**
`asr_node` sends nothing while nobody is talking (`asr_node.py:79-87` publishes only
when there is text or when a segment ends), so there is no "silence event" to
subscribe to. A segmenter driven only by arriving events still produces the *same
boundary index* — it notices the 45 s gap when the next event finally arrives — which
is exactly why "a boundary is produced" is too weak an oracle for T-B2: it is
satisfied by the version that cuts at the right place at the wrong *time*. The
difference matters because closing an episode is what triggers summarisation and
embedding (S5); a gap-driven segmenter defers that work until the person speaks
again, so a robot left alone overnight wakes up with the whole evening unconsolidated
and the summary of "what we did last night" arrives after the next thing is said.
`tests/test_segment.py` asserts the episode closes with **no later event in the
stream at all**, and separately shows that the index-only assertion cannot tell the
two designs apart.

**Why the span cap is exclusive and the turn cap is inclusive.**
The plan states both as invariants: no episode longer than 20 turns or 5 minutes. For
turns that permits equality, and closing *after* appending the 20th turn yields
exactly 20. For the span it does not: events arriving every 40 s pass the silence
rule, and the one that lands at 320 s would close an episode spanning 320 s if the
cap were inclusive — the rule that exists to bound the span would be the reason it is
exceeded. So an event that *would* push the span past the cap closes the previous
episode first and starts the next one. The asymmetry is not a style choice: a turn
count is an integer this module increments, and the span is a value the event brings
with it.

**Why every boundary records its reason.**
An episode that knows it ended but not why makes "the person stopped talking" and "we
hit a ceiling" the same observation, and those call for different behaviour in S5: a
silence boundary is a real conversational unit worth summarising, a `max_turns`
boundary is an arbitrary cut through the middle of one. It is also the only way to
check the hard-cap invariant on real data without re-deriving the segmentation.

**Why an unknown event kind is refused.**
Treating an unrecognised kind as "contributes nothing" is how a typo in a producer
silently switches off the turn cap: the events still flow, no episode ever reaches 20
turns, and every gate stays green. The kinds are a closed set for the same reason
`why.goal` and the semantic predicates are.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from memory.store import StoreError


class SegmentError(StoreError):
    """An event this module cannot segment: unknown kind, missing ts, bad order."""


# Closed set. `turn` is the only kind that counts against the turn cap — a scene
# change is a boundary signal, not a turn, and letting it consume turn budget would
# make the cap depend on how busy the camera was.
TURN_KIND = "turn"
SCENE_KIND = "scene_change"
KINDS = (TURN_KIND, SCENE_KIND, "gesture", "game_round", "interrupt")

# Closed set, and the reason an episode has one at all: 「活動轉換」 is a boundary
# rule, so the thing that changed has to be a value from a set rather than free text.
# Free-text activity would make the rule fire on a spelling difference.
ACTIVITIES = ("conversation", "game")

# Plan §2 / §3-S4. Seconds and turns.
SILENCE_S = 45.0
MAX_TURNS = 20
MAX_SPAN_S = 300.0

# Boundary reasons, in the order the rules are evaluated. The order is part of the
# specification, not an implementation detail: an event can trip two rules at once
# (a scene change that is also an activity change), and without a fixed order the
# recorded reason would depend on dict iteration — which is what T-B4 forbids.
#
# `silence` is first because it is a statement about the *previous* episode's end,
# decided before this event is considered as a member of anything. `end_of_stream` is
# last because it is not a rule; it is what `flush()` reports.
RULES = ("silence", "activity", "scene", "max_span", "max_turns")
REASONS = RULES + ("end_of_stream",)

# Rules that decide *before* the event joins the episode, and therefore start the new
# episode with it. `max_turns` is the only one that closes after appending, because a
# turn cap of 20 is satisfied by an episode holding exactly 20.
PRE_RULES = ("silence", "activity", "scene", "max_span")


class Episode:
    """One closed episode. Immutable, and it knows why it ended.

    `indices` are positions in the replayed sequence and exist for the tests; the
    live node has `seqs`, which are store sequence numbers and therefore stable
    across a restart. Both are recorded because a boundary asserted by index in a
    fixture and a boundary stored on disk are two different claims.
    """

    __slots__ = ("reason", "seqs", "indices", "start_ts", "end_ts", "turns", "activity")

    def __init__(self, reason: str, seqs: tuple[int, ...], indices: tuple[int, ...],
                 start_ts: float, end_ts: float, turns: int, activity: str) -> None:
        if reason not in REASONS:
            raise SegmentError(f"unknown boundary reason {reason!r}; the set is {REASONS}")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "seqs", seqs)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "start_ts", start_ts)
        object.__setattr__(self, "end_ts", end_ts)
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "activity", activity)

    def __setattr__(self, name, value):
        raise SegmentError("an Episode is immutable once closed; a boundary that can "
                           "be edited after the fact is not a record of anything")

    @property
    def span(self) -> float:
        return self.end_ts - self.start_ts

    def as_dict(self) -> dict:
        return {"reason": self.reason, "seqs": list(self.seqs),
                "start_ts": self.start_ts, "end_ts": self.end_ts,
                "turns": self.turns, "activity": self.activity}

    def __repr__(self) -> str:
        return (f"Episode({self.reason}, seqs={self.seqs[:1]}..{self.seqs[-1:]}, "
                f"turns={self.turns}, span={self.span:.1f}s)")


def check_event(event: Mapping) -> None:
    """Everything this module requires of a stored event. Raises, never assumes."""
    kind = event.get("kind")
    if kind not in KINDS:
        raise SegmentError(
            f"unknown event kind {kind!r}; the set is closed ({KINDS}) because an "
            "unrecognised kind that contributes nothing switches off the turn cap "
            "without switching off the events")
    ts = event.get("ts")
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        raise SegmentError(f"event {kind!r} has ts={ts!r}; the store stamps every "
                           "record, so a missing ts means this did not come from it")
    activity = event.get("activity")
    if activity not in ACTIVITIES:
        raise SegmentError(f"unknown activity {activity!r}; the set is {ACTIVITIES}")


class Segmenter:
    """Feed it stored events and ticks; it hands back episodes as they close.

    `rules` is a reviewable argument rather than a comment, for the same reason
    `Store._allow_ephemeral` is: T-B1 has to be able to disable exactly one rule and
    show that only its own case goes red, and a text-patched mutant could not show
    that the four assertions do not mask each other. `tests/test_segment.py` derives
    from the source that nothing outside `tests/` passes it, so the exemption cannot
    be taken quietly in production.
    """

    def __init__(self, *, silence_s: float = SILENCE_S, max_turns: int = MAX_TURNS,
                 max_span_s: float = MAX_SPAN_S, rules: Sequence[str] = RULES) -> None:
        unknown = [r for r in rules if r not in RULES]
        if unknown:
            raise SegmentError(f"unknown rule(s) {unknown}; the set is {RULES}")
        if silence_s <= 0 or max_turns <= 0 or max_span_s <= 0:
            raise SegmentError("silence_s, max_turns and max_span_s must be positive")
        self.silence_s = float(silence_s)
        self.max_turns = int(max_turns)
        self.max_span_s = float(max_span_s)
        # A tuple in RULES order, not the caller's order: the evaluation order is
        # part of the specification, so it must not be re-orderable by an argument.
        self.rules = tuple(r for r in RULES if r in rules)
        self._open: list[tuple[int, int, Mapping]] = []   # (seq, index, event)
        self._turns = 0
        self._activity: str | None = None
        self._last_ts: float | None = None
        self._first_ts: float | None = None

    # ------------------------------------------------------------------ queries
    @property
    def open_turns(self) -> int:
        return self._turns

    def is_open(self) -> bool:
        return bool(self._open)

    # ------------------------------------------------------------------ drivers
    def tick(self, now: float) -> list[Episode]:
        """Time passed and nothing arrived. Returns the episodes that closed.

        This is the whole of T-B2: with no `tick`, a silent evening closes only when
        somebody next speaks, and the consolidation it should have triggered is
        deferred to a moment that has nothing to do with it.
        """
        if not self._open:
            return []
        out = []
        if "silence" in self.rules and self._last_ts is not None \
                and now - self._last_ts >= self.silence_s:
            out.append(self._close("silence"))
        elif "max_span" in self.rules and self._first_ts is not None \
                and now - self._first_ts >= self.max_span_s:
            out.append(self._close("max_span"))
        return out

    def feed(self, event: Mapping, index: int = -1) -> list[Episode]:
        """Add one stored event. Returns the episodes that closed because of it."""
        check_event(event)
        seq = event.get("seq", index)
        ts = float(event["ts"])
        if self._last_ts is not None and ts < self._last_ts:
            raise SegmentError(
                f"event ts {ts!r} precedes the previous {self._last_ts!r} — the store "
                "stamps in order, so out-of-order input means two writers or a "
                "hand-edited fixture, and segmenting it would invent a boundary")
        out: list[Episode] = []
        reason = self._pre_boundary(event, ts)
        if reason is not None:
            out.append(self._close(reason))
        if not self._open:
            self._first_ts = ts
            self._activity = event["activity"]
        self._open.append((seq, index, event))
        self._last_ts = ts
        if event["kind"] == TURN_KIND:
            self._turns += 1
        if "max_turns" in self.rules and self._turns >= self.max_turns:
            out.append(self._close("max_turns"))
        return out

    def flush(self) -> list[Episode]:
        """Close whatever is open, for shutdown and for replaying a finite fixture."""
        return [self._close("end_of_stream")] if self._open else []

    # ------------------------------------------------------------------ internal
    def _pre_boundary(self, event: Mapping, ts: float) -> str | None:
        """The first pre-rule this event trips, in RULES order, or None."""
        if not self._open:
            return None
        for rule in self.rules:
            if rule not in PRE_RULES:
                continue
            if rule == "silence":
                if self._last_ts is not None and ts - self._last_ts >= self.silence_s:
                    return "silence"
            elif rule == "activity":
                if event["activity"] != self._activity:
                    return "activity"
            elif rule == "scene":
                if event["kind"] == SCENE_KIND:
                    return "scene"
            elif rule == "max_span":
                # Exclusive: see the module docstring. An inclusive cap is exceeded by
                # the very event that trips it.
                if self._first_ts is not None and ts - self._first_ts > self.max_span_s:
                    return "max_span"
        return None

    def _close(self, reason: str) -> Episode:
        ep = Episode(
            reason=reason,
            seqs=tuple(seq for seq, _, _ in self._open),
            indices=tuple(idx for _, idx, _ in self._open),
            start_ts=float(self._first_ts if self._first_ts is not None else 0.0),
            end_ts=float(self._last_ts if self._last_ts is not None else 0.0),
            turns=self._turns,
            activity=str(self._activity),
        )
        self._open = []
        self._turns = 0
        self._activity = None
        self._first_ts = None
        # `_last_ts` deliberately survives, and not for the silence rule: a boundary
        # is decided while the episode is still open, so `_pre_boundary` has already
        # returned by the time this runs. What it carries across is the monotonicity
        # guard. Resetting it would make the first event after every boundary
        # unorderable — an event timestamped *before* the episode that just closed
        # would be accepted, and the store's ordering assumption would be violated
        # exactly at the seam where nothing is watching.
        return ep


def replay(events: Iterable[Mapping], ticks: Iterable[float] = (), **kw) -> list[Episode]:
    """Segment a finite fixture. Ticks are merged by time; a tie puts the tick first.

    Tie-break stated rather than inherited from a sort: a tick at exactly the same
    instant as an event means "time reached here with nothing new yet", which is true
    up to but not including the event. Sorting stably the other way would let a
    45-second silence be cancelled by the event that ends it.
    """
    seg = Segmenter(**kw)
    out: list[Episode] = []
    pending = sorted(float(t) for t in ticks)
    for index, event in enumerate(events):
        ts = float(event["ts"]) if isinstance(event.get("ts"), (int, float)) else None
        while pending and ts is not None and pending[0] <= ts:
            out.extend(seg.tick(pending.pop(0)))
        out.extend(seg.feed(event, index))
    for t in pending:
        out.extend(seg.tick(t))
    out.extend(seg.flush())
    return out


def cap_findings(episodes: Sequence[Episode], *, max_turns: int = MAX_TURNS,
                 max_span_s: float = MAX_SPAN_S) -> list[str]:
    """The hard-cap invariant, as a list of violations. Empty is the passing value.

    Separate from the segmenter so it can be run against episodes that came off disk
    rather than only against ones this process just produced — a self-checking
    segmenter proves its own arithmetic, not the contents of the store.
    """
    findings: list[str] = []
    for n, ep in enumerate(episodes):
        if ep.turns > max_turns:
            findings.append(f"episode {n} ({ep.reason}) holds {ep.turns} turns, "
                            f"cap is {max_turns}")
        if ep.span > max_span_s:
            findings.append(f"episode {n} ({ep.reason}) spans {ep.span:.1f}s, "
                            f"cap is {max_span_s}")
    return findings
