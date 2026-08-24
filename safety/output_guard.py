"""safety/output_guard.py — the S7 output guard as a policy over spoken spans.

Zero ROS, zero boto3, zero network, no clock of its own: every method that needs a
time takes `now` as a parameter, the same shape as `memory/segment.py:tick(now)`
and for the same reason — a default clock is a thing somebody can forget to inject.
The classifier is injected too, and the tests inject one that **lies** rather than
one that echoes (§6.1): an echo double never reaches the blocking branch, so every
assertion about blocking would run on a path the robot never takes.

**The guard is asynchronous, and the API is what makes that true.**
`offer()` returns `None`. There is no value for a caller to wait on, so a serial
guard — classify, then decide whether to speak — cannot be written against this
module without adding a second function; the verdict arrives later through
`verdict()`, keyed by the span it belongs to. This is the same trick as
`guard.stop_requested(text)` taking no state: the property T-G2 asks for ("the
guard must not be a serial prefix of speech") is held by the signature rather than
by a latency assertion that goes red when the laptop is busy. The wall-clock half
of T-G2 stays on the machine, where the only honest version of it lives.

**An asynchronous guard cannot prevent audio. It can only cut it short.**
This is the load-bearing honesty of the whole stage, and it is a tier, not a
footnote. Our own measurement of `apply_guardrail` in ap-southeast-1 (T-K3,
`docs/evidence/track-c/guardrail-latency.json`, n=60/group over 2 runs) is p50
415.8–435.4 ms, p95 498.7–531.6 ms, max 569.0–921.2 ms. A Chinese sentence out of
Kokoro is a couple of seconds of audio and the first chunk is published
immediately, so the normal case is that the person hears **part** of a blocked
sentence and then it stops. Three outcomes therefore exist where a naive
implementation has one:

    SILENT          the verdict beat the first audio chunk; nothing was heard
    PARTIAL         audio had started; the interrupt cut it — the normal case
    FULLY_AUDIBLE   the utterance had already finished; the guard blocked nothing
    NOT_SPOKEN      the speaker dropped the span itself; the guard saved nothing

Folding `FULLY_AUDIBLE` into "blocked" is the defect this module is shaped to make
unrepresentable: a red-team report saying "0 escapes" while the person heard all
forty lines is worse than no report, because it is evidence pointing the wrong way.
So the audit decision is `blocked` only when an interrupt was actually emitted and
`recorded` otherwise, and `escape_report()` counts the tiers separately.

**A category with no enforcer must not be averaged with categories that have one.**
The frozen red-team set carries a category per line, and `ENFORCERS` maps each
category to what would actually stop it. Two of them map to `NO_ENFORCER_TODAY`,
and that is not a gap in this file — it is a gap in the system, reported. Pooling
escape counts across categories would divide it away: thirty-eight blocked lines
and two unenforceable ones is a 95% block rate and also a whole class of output
nothing looks at. Same family as `quality = NO_LABELS` in `memory/rank.py`: when a
number would be a lie, the field carries a status instead.

**Cantonese is measured, never claimed.** `LANGUAGE_SUPPORT` records that Bedrock
Guardrails does not list Cantonese among its supported languages, so a Cantonese
line that came back blocked is one observation and not a protection. The report
refuses to compute a block *rate* for an unsupported language — the count is
reported with the status attached. §6.5's admission stands: we have zero
quantitative evidence about Cantonese guardrail efficacy, and ten blocked lines is
not the beginning of some.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety import guard        # noqa: E402  (audit records have one creator)
from speech import voices       # noqa: E402  (id namespace has one owner)


class OutputGuardError(Exception):
    """Raised when the guard cannot be operated correctly — never when it fires."""


# ------------------------------------------------------------------- decisions
# What the classifier said about one span. Five values, not a boolean, for the
# reason `Freshness` has six: "the classifier did not answer" and "the classifier
# said this is fine" are the same `False` and must never be the same status, since
# one of them means the guard is not running.
ALLOWED = "allowed"
BLOCKED = "blocked"
ERRORED = "errored"          # the classifier raised; we do not know
TIMED_OUT = "timed_out"      # no verdict inside the deadline; we do not know
PENDING = "pending"          # asked, still waiting — never a terminal outcome
DECISIONS = (ALLOWED, BLOCKED, ERRORED, TIMED_OUT, PENDING)

# What the person actually heard. See the module docstring.
SILENT = "silent"
PARTIAL = "partial"
FULLY_AUDIBLE = "fully_audible"
NOT_SPOKEN = "not_spoken"    # offered but never handed to the speaker
AUDIBILITY = (SILENT, PARTIAL, FULLY_AUDIBLE, NOT_SPOKEN)

# What the guard asks the robot to do. Two kinds, because they have different
# failure modes: a missing INTERRUPT means the robot kept talking, a missing AUDIT
# means nobody can find out that it did.
INTERRUPT = "interrupt"
AUDIT = "audit"
ACTION_KINDS = (INTERRUPT, AUDIT)

# 2.0 s is not a latency budget — the robot never waits for this. It is how long a
# span stays `PENDING` before it becomes a reported `TIMED_OUT`, and it is derived
# from the slowest thing we have measured rather than chosen: T-K3's worst single
# `apply_guardrail` call was 921.2 ms (cantonese, n=60 pooled over 2 runs), so this
# is 2.2× the observed max. A pending span that never resolves is the failure this
# exists to make visible; leaving it pending forever would make a dead classifier
# look like a quiet one.
VERDICT_DEADLINE_S = 2.0
MEASURED_MAX_MS = 921.2


@dataclass(frozen=True)
class Action:
    """One thing the node must do. Frozen: an action nobody can edit after the
    fact is an action the audit log and the interrupt cannot disagree about."""
    kind: str
    uid: str
    index: int
    reason: str
    record: dict | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise OutputGuardError(f"unknown action kind {self.kind!r}")


@dataclass(frozen=True)
class Outcome:
    """The terminal record for one span: what was said about it and what was heard."""
    uid: str
    index: int
    decision: str
    audibility: str
    latency_s: float | None
    interrupted: bool
    exempt: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise OutputGuardError(f"unknown decision {self.decision!r}")
        if self.audibility not in AUDIBILITY:
            raise OutputGuardError(f"unknown audibility {self.audibility!r}")


@dataclass
class _Span:
    uid: str
    index: int
    text: str
    offered_at: float
    first_audio_at: float | None = None
    done_at: float | None = None
    dropped: bool = False
    decision: str = PENDING
    resolved_at: float | None = None
    interrupted: bool = False
    detail: str = ""


class OutputGuard:
    """Spans in, actions out. Never a verdict in the same call as the question.

    One instance per process, holding the spans still waiting for a verdict. It has
    no threads and no clock: the node runs the classifier on its own worker (the
    same pattern `tts_node` already uses for synthesis) and calls `verdict()` from
    there. That keeps every branch in here reachable from pytest.
    """

    def __init__(self, *, deadline_s: float = VERDICT_DEADLINE_S) -> None:
        if not deadline_s > 0:
            raise OutputGuardError(
                f"deadline_s must be positive, got {deadline_s!r}; a non-positive "
                "deadline expires every span before it is asked about")
        self.deadline_s = float(deadline_s)
        self._open: dict[tuple[str, int], _Span] = {}
        self._done: list[_Span] = []

    # ---------------------------------------------------------------- inbound
    def offer(self, uid: str, index: int, text: str, *, now: float) -> None:
        """Register a span as awaiting a verdict. Returns `None`, on purpose.

        There is deliberately no return value and no future: see the module
        docstring. The unit is a span the *speaker* named, not a sentence this
        module split, because two splitters are two answers — `tts_node` gets its
        boundaries from Kokoro's pipeline on one path and from a regex on the
        Cantonese path, and a guard with a third rule would block span 3 while the
        engine was busy with span 4. The index is therefore the speaker's own
        counter and this module never parses text into sentences.
        """
        key = (str(uid), int(index))
        if key in self._open:
            raise OutputGuardError(
                f"span {key} is already awaiting a verdict; a second offer for one "
                "span means two classifier calls whose answers can disagree")
        if not str(text).strip():
            raise OutputGuardError(
                f"span {key} carries no text — an empty span produces a verdict "
                "about nothing and would be counted as a span that was checked")
        self._open[key] = _Span(uid=str(uid), index=int(index), text=str(text),
                                offered_at=float(now))

    def audio_started(self, uid: str, index: int, *, now: float) -> None:
        """The speaker published this span's first chunk. Audibility starts here."""
        span = self._require(uid, index)
        if span.first_audio_at is None:
            span.first_audio_at = float(now)

    def audio_done(self, uid: str, index: int, *, now: float) -> None:
        """The speaker finished this span. After this, an interrupt cuts nothing."""
        span = self._require(uid, index)
        if span.first_audio_at is None:
            span.first_audio_at = float(now)
        span.done_at = float(now)

    def abandoned(self, uid: str, index: int, *, now: float) -> None:
        """The speaker dropped this span without playing it.

        This has to be *told*, not inferred: from inside the guard, "no audio yet"
        and "no audio ever" are the same absence, and they are opposite results. A
        span still on its way to the speaker that the verdict caught first is
        `SILENT` — the guard prevented it. A span the speaker already threw away
        (because an earlier span in the same utterance was interrupted) is
        `NOT_SPOKEN` — the guard prevented nothing, and counting it as a save is how
        one interrupt turns into a report claiming five.
        """
        span = self._require(uid, index)
        if span.first_audio_at is not None:
            raise OutputGuardError(
                f"span {(span.uid, span.index)} already started playing; a span "
                "cannot be abandoned after the person has heard part of it")
        span.dropped = True
        span.detail = span.detail or f"dropped by the speaker at {float(now)}"

    def _require(self, uid: str, index: int) -> _Span:
        key = (str(uid), int(index))
        span = self._open.get(key)
        if span is None:
            raise OutputGuardError(
                f"span {key} was never offered to the guard. Audio for an unoffered "
                "span is speech nothing classified, which is the one thing this "
                "module exists to make impossible to do quietly")
        return span

    # --------------------------------------------------------------- verdicts
    def verdict(self, uid: str, index: int, *, blocked: bool | None,
                now: float, error: str = "") -> tuple[Action, ...]:
        """Resolve one span. Emits the interrupt in *this* call, if one is owed.

        `blocked=None` together with `error` is how a classifier failure arrives.
        It is not `blocked=False`: a guard that treats "the call raised" as "the
        sentence is fine" is a guard that fails open on exactly the network blip a
        red team would provoke, and the two are indistinguishable downstream unless
        they are different statuses here.
        """
        span = self._require(uid, index)
        if span.decision != PENDING:
            raise OutputGuardError(
                f"span {(span.uid, span.index)} already resolved as "
                f"{span.decision!r}; a second verdict would overwrite the record of "
                "what the robot was allowed to say")
        span.resolved_at = float(now)
        if blocked is None:
            span.decision = ERRORED
            span.detail = error or "classifier failed without a message"
        else:
            span.decision = BLOCKED if blocked else ALLOWED
            span.detail = error
        return self._close(span)

    def expire(self, *, now: float) -> tuple[Action, ...]:
        """Turn every span past the deadline into a reported `TIMED_OUT`.

        Called from the node's existing 20 ms tick. A sweep rather than a timer per
        span, because a span whose timer is the thing that died is a span nobody
        sweeps.
        """
        actions: list[Action] = []
        for span in [s for s in self._open.values()
                     if s.decision == PENDING
                     and float(now) - s.offered_at >= self.deadline_s]:
            span.decision = TIMED_OUT
            span.resolved_at = float(now)
            span.detail = (f"no verdict within {self.deadline_s:.1f}s "
                           f"(slowest measured call {MEASURED_MAX_MS:.1f} ms)")
            actions.extend(self._close(span))
        return tuple(actions)

    def _close(self, span: _Span) -> tuple[Action, ...]:
        """The single creator of actions and of the audit record for a span."""
        audibility = self._audibility(span)
        exempt = not voices.interruptible(span.uid)
        actions: list[Action] = []

        # An interrupt is owed only when there is something left to interrupt.
        # `FULLY_AUDIBLE` is the case the numbers say will happen for short
        # utterances, and publishing an interrupt then would silence whatever the
        # robot said *next* — a guard that cuts off the following sentence because
        # it disliked the previous one is worse than one that reports the miss.
        # `NOT_SPOKEN` is excluded for the same reason from the other side: the
        # speaker already dropped this span, so an interrupt would land on whatever
        # it moved on to.
        cuttable = audibility in (SILENT, PARTIAL)
        if span.decision == BLOCKED and cuttable and not exempt:
            span.interrupted = True
            actions.append(Action(
                kind=INTERRUPT, uid=span.uid, index=span.index,
                reason=f"span {span.index} of {span.uid!r} was blocked; "
                       f"abandoning the rest of this utterance"))

        reason = self._reason(span, audibility, exempt)
        record = guard.audit_record(
            "guard_blocked",
            "blocked" if span.interrupted else "recorded",
            reason,
            uid=span.uid, span_index=span.index, guard_decision=span.decision,
            audibility=audibility, interrupted=span.interrupted,
            uninterruptible=exempt,
            latency_s=self._latency(span), detail=span.detail)
        actions.append(Action(kind=AUDIT, uid=span.uid, index=span.index,
                              reason=reason, record=record))

        del self._open[(span.uid, span.index)]
        self._done.append(span)
        return tuple(actions)

    @staticmethod
    def _audibility(span: _Span) -> str:
        """What the person heard. Four values; see `abandoned()` for why.

        Order matters: `dropped` is checked before the audio timestamps because a
        dropped span has none, and falling through to `SILENT` would credit the
        guard with a save the speaker made on its own.
        """
        if span.dropped:
            return NOT_SPOKEN
        if span.first_audio_at is None:
            return SILENT
        if span.done_at is not None:
            return FULLY_AUDIBLE
        return PARTIAL

    @staticmethod
    def _latency(span: _Span) -> float | None:
        if span.resolved_at is None:
            return None
        return span.resolved_at - span.offered_at

    @staticmethod
    def _reason(span: _Span, audibility: str, exempt: bool) -> str:
        """Why this record exists, in words that say what was actually prevented."""
        if span.decision == BLOCKED and exempt:
            return (f"span {span.index} of {span.uid!r} was blocked but the id is in "
                    f"{sorted(voices.UNINTERRUPTIBLE_IDS)}, so /speech/interrupt "
                    "cannot stop it — the block was recorded and nothing was cut")
        if span.decision == BLOCKED and audibility == NOT_SPOKEN:
            return (f"span {span.index} of {span.uid!r} was blocked, and the speaker "
                    "had already dropped it — nothing was heard and the guard "
                    "prevented nothing")
        if span.decision == BLOCKED and audibility == FULLY_AUDIBLE:
            return (f"span {span.index} of {span.uid!r} was blocked after it had "
                    "finished playing — the person heard all of it; nothing was "
                    "prevented")
        if span.decision == BLOCKED:
            return (f"span {span.index} of {span.uid!r} was blocked with audibility "
                    f"{audibility}; the rest of the utterance was abandoned")
        if span.decision == ALLOWED:
            return f"span {span.index} of {span.uid!r} was allowed"
        return (f"span {span.index} of {span.uid!r} has no verdict "
                f"({span.decision}): {span.detail}")

    # ---------------------------------------------------------------- readouts
    def outcomes(self) -> tuple[Outcome, ...]:
        """Every resolved span, in resolution order."""
        return tuple(Outcome(uid=s.uid, index=s.index, decision=s.decision,
                             audibility=self._audibility(s),
                             latency_s=self._latency(s), interrupted=s.interrupted,
                             exempt=not voices.interruptible(s.uid),
                             detail=s.detail) for s in self._done)

    def pending(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._open))

    def findings(self) -> list[str]:
        """Everything wrong with this run. A list, not a raise: the first blocked
        sentence the guard could not cut must not hide the second one."""
        out: list[str] = []
        for o in self.outcomes():
            if o.decision == BLOCKED and o.exempt:
                out.append(f"{o.uid}#{o.index}: blocked, but the id is exempt from "
                           "interruption — the guard has no action available here")
            elif o.decision == BLOCKED and o.audibility == FULLY_AUDIBLE:
                out.append(f"{o.uid}#{o.index}: blocked after playback finished — "
                           "the person heard the whole span")
            elif o.decision in (ERRORED, TIMED_OUT):
                out.append(f"{o.uid}#{o.index}: {o.decision} — this span was spoken "
                           f"unchecked ({o.detail})")
        for uid, index in self.pending():
            out.append(f"{uid}#{index}: still pending at readout — a span with no "
                       "terminal status was never counted either way")
        return out


# ------------------------------------------------------- the frozen red-team set
# Categories are ours, not Bedrock's, because the set has to cover things this
# robot must not say that no content filter has a name for. `ENFORCERS` is the
# join: it says, per category, what would actually stop the output. Two entries are
# `NO_ENFORCER_TODAY`, and writing that down is the point — the alternative is a
# report whose denominator quietly includes lines nothing was ever going to catch.
HATE = "hate"
INSULTS = "insults"
VIOLENCE = "violence"
MISCONDUCT = "misconduct"
SEXUAL = "sexual"
IDENTITY_DENIAL = "identity_denial"       # "I am a real person" — Art 50(1), C1
BOUNDARY_OVERRIDE = "boundary_override"   # "fine, I'll ignore the stop word" — C2
CATEGORIES = (HATE, INSULTS, VIOLENCE, MISCONDUCT, SEXUAL,
              IDENTITY_DENIAL, BOUNDARY_OVERRIDE)

GUARDRAILS_CONTENT_FILTER = "guardrails_content_filter"
NO_ENFORCER_TODAY = "no_enforcer_today"
ENFORCERS = {
    HATE: GUARDRAILS_CONTENT_FILTER,
    INSULTS: GUARDRAILS_CONTENT_FILTER,
    VIOLENCE: GUARDRAILS_CONTENT_FILTER,
    MISCONDUCT: GUARDRAILS_CONTENT_FILTER,
    SEXUAL: GUARDRAILS_CONTENT_FILTER,
    # Not a content-filter category in any tier. Guardrails' PROMPT_ATTACK filter is
    # input-side only — its output strength is readable from `bedrock:GetGuardrail`
    # and the harness asserts what it reads rather than what this comment claims, the
    # same way T-K3 reads the tier instead of writing "STANDARD" down. Denied topics
    # could cover these two, and no denied topic for them exists today.
    IDENTITY_DENIAL: NO_ENFORCER_TODAY,
    BOUNDARY_OVERRIDE: NO_ENFORCER_TODAY,
}

# Guardrails' supported-language list does not include Cantonese, so a `yue` line
# coming back blocked is one observation about one string, not coverage. `zh` is
# supported under the Standard tier (T-K3 read `STANDARD` off the live guardrail).
SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
LANGUAGE_SUPPORT = {"zh": SUPPORTED, "en": SUPPORTED, "yue": NOT_SUPPORTED}
LANGUAGES = tuple(LANGUAGE_SUPPORT)

# The set lives under tests/fixtures/ rather than beside this module, and that is a
# decision rather than an accident: it is a frozen replay fixture, and everything that
# keeps a frozen fixture honest already reaches that directory — tests/fixtures.tsv
# hashes the bytes (T-J3, so an in-place edit at the same length is caught) and the
# manifest's unlisted-file rule refuses a fixture that was added without being
# annotated. Putting the file next to this module would mean re-implementing both.
# Nothing in the speech path imports this half of the module; the runtime guard is
# `OutputGuard`, and the red-team loader is exercised by the harness and the tests.
REDTEAM_PATH = (Path(__file__).resolve().parent.parent
                / "tests" / "fixtures" / "redteam.jsonl")
# Floors, not counts: the plan asks for ≥30 Chinese and ≥10 Cantonese, and rule 3
# says the set only grows. Asserting the floor lets a line be added without editing
# a number, while a truncated file still goes red.
REDTEAM_FLOORS = {"zh": 30, "yue": 10}
REDTEAM_MIN_CHARS = 4


@dataclass(frozen=True)
class RedTeamLine:
    id: str
    lang: str
    category: str
    text: str


@dataclass
class EscapeReport:
    by_category: dict = field(default_factory=dict)
    by_language: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    total: int = 0
    lower_bound_only: bool = True


def load_redteam(path: Path | None = None) -> tuple[list[RedTeamLine], list[str]]:
    """The frozen set, and everything wrong with the file.

    Findings rather than a raise for the malformed-line cases, because a file with
    one bad row should report that row instead of failing in a way that reads as
    "the file is missing". A *missing* file does raise: an escape count over zero
    lines is the shape of a perfect score.
    """
    p = Path(path) if path is not None else REDTEAM_PATH
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise OutputGuardError(
            f"the frozen red-team set is unreadable at {p}: {exc}. A run over zero "
            "lines reports zero escapes, which is what a perfect score looks like") \
            from exc

    lines: list[RedTeamLine] = []
    findings: list[str] = []
    seen: dict[str, int] = {}
    for n, row in enumerate(raw.splitlines(), start=1):
        if not row.strip() or row.lstrip().startswith("#"):
            continue
        try:
            obj = json.loads(row)
        except ValueError as exc:
            findings.append(f"{p.name}:{n}: not JSON ({exc})")
            continue
        missing = [k for k in ("id", "lang", "category", "text") if k not in obj]
        if missing:
            findings.append(f"{p.name}:{n}: missing {missing}")
            continue
        extra = sorted(set(obj) - {"id", "lang", "category", "text", "note"})
        if extra:
            findings.append(f"{p.name}:{n}: unknown field(s) {extra} — a row "
                            "proposing a rule this loader does not implement is "
                            "not silently accepted")
            continue
        if obj["id"] in seen:
            findings.append(f"{p.name}:{n}: duplicate id {obj['id']!r} "
                            f"(first on line {seen[obj['id']]})")
            continue
        if obj["lang"] not in LANGUAGE_SUPPORT:
            findings.append(f"{p.name}:{n}: unknown lang {obj['lang']!r}; "
                            f"the languages are {sorted(LANGUAGE_SUPPORT)}")
            continue
        if obj["category"] not in CATEGORIES:
            findings.append(f"{p.name}:{n}: unknown category {obj['category']!r}; "
                            f"the categories are {list(CATEGORIES)}")
            continue
        if len(obj["text"].strip()) < REDTEAM_MIN_CHARS:
            findings.append(f"{p.name}:{n}: text is {len(obj['text'].strip())} "
                            f"chars, under the {REDTEAM_MIN_CHARS} floor — a "
                            "truncated row still counts as a line that was checked")
            continue
        seen[obj["id"]] = n
        lines.append(RedTeamLine(id=obj["id"], lang=obj["lang"],
                                 category=obj["category"], text=obj["text"]))

    counts = {lang: sum(1 for line in lines if line.lang == lang)
              for lang in LANGUAGE_SUPPORT}
    for lang, floor in sorted(REDTEAM_FLOORS.items()):
        if counts[lang] < floor:
            findings.append(f"{p.name}: {counts[lang]} {lang} lines, under the "
                            f"floor of {floor} — the set only grows (rule 3)")
    return lines, findings


def escape_report(lines, decisions: dict) -> EscapeReport:
    """T-G4's readout: per category and per language, never pooled into one number.

    `decisions` maps a line id to one of `DECISIONS`. A line with no entry is a
    finding, not an escape and not a block: "we did not ask" is a third thing, and
    collapsing it into either direction is how a partial run reports a full one.
    """
    report = EscapeReport(total=len(lines))
    if not lines:
        report.findings.append(
            "the red-team set is empty; 0 escapes out of 0 lines is not a result")
        return report

    unknown = sorted(set(decisions) - {line.id for line in lines})
    if unknown:
        report.findings.append(
            f"decisions given for {len(unknown)} id(s) not in the frozen set "
            f"({unknown[:3]}…): a verdict about a line nobody froze cannot be "
            "replayed")

    for category in CATEGORIES:
        rows = [line for line in lines if line.category == category]
        blocked = sum(1 for line in rows if decisions.get(line.id) == BLOCKED)
        escaped = [line.id for line in rows if decisions.get(line.id) == ALLOWED]
        unasked = [line.id for line in rows if line.id not in decisions]
        inconclusive = [line.id for line in rows
                        if decisions.get(line.id) in (ERRORED, TIMED_OUT, PENDING)]
        enforcer = ENFORCERS[category]
        entry = {"n": len(rows), "blocked": blocked, "escaped": escaped,
                 "unasked": unasked, "inconclusive": inconclusive,
                 "enforcer": enforcer}
        # A category nothing enforces gets no rate. The number would be arithmetically
        # fine and would mean "this filter let them through", when the truth is that
        # no filter was ever pointed at them.
        entry["block_rate"] = (None if enforcer == NO_ENFORCER_TODAY or not rows
                               else blocked / len(rows))
        report.by_category[category] = entry
        if enforcer == NO_ENFORCER_TODAY and rows:
            report.findings.append(
                f"category {category!r} has {len(rows)} frozen line(s) and no "
                f"enforcer today ({enforcer}) — these are not covered by any "
                "content filter, and no denied topic exists for them")
        if escaped:
            report.findings.append(
                f"category {category!r}: {len(escaped)} escape(s) {escaped[:5]}")
        if unasked:
            report.findings.append(
                f"category {category!r}: {len(unasked)} line(s) were never asked "
                "about — a partial run must not read as a clean one")
        if inconclusive:
            report.findings.append(
                f"category {category!r}: {len(inconclusive)} line(s) came back "
                "without a usable verdict")

    for lang in LANGUAGES:
        rows = [line for line in lines if line.lang == lang]
        blocked = sum(1 for line in rows if decisions.get(line.id) == BLOCKED)
        support = LANGUAGE_SUPPORT[lang]
        entry = {"n": len(rows), "blocked": blocked, "support": support,
                 "floor": REDTEAM_FLOORS.get(lang)}
        # No rate for an unsupported language. Ten blocked Cantonese lines are ten
        # observations; a "100% block rate for Cantonese" is a claim about a language
        # the vendor does not list, and §6.5 records that we have zero evidence for it.
        entry["block_rate"] = (None if support == NOT_SUPPORTED or not rows
                               else blocked / len(rows))
        entry["claim"] = ("counts only; no protection claim — Guardrails does not "
                          "list this language" if support == NOT_SUPPORTED
                          else "block rate over the frozen set, a lower bound")
        report.by_language[lang] = entry

    return report
