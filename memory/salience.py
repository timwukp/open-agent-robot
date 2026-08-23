"""memory/salience.py — how much a turn is worth remembering. Text and intent only.

Zero ROS, zero boto3, zero network, no clock, no disk. Scoring a turn is a pure
function of text, so the whole of T-C6 runs under pytest on a laptop.

**Why this lands before S4 rather than with it.**
Plan §S3.5: 「公式一旦寫進資料就難改」. The reason is not that the code is hard to
edit — it is three lines — but that the *values* are durable. A salience number in
`events.jsonl` is only comparable to another one if the same formula produced both,
and a formula changed in S6 silently makes every earlier record incomparable with
every later one while every gate stays green. So two things exist here before
anything writes: the formula, and `FORMULA` — an id that travels with each value.
`memory.store` refuses a record that carries a score without one (see
`store.RECORD_FIELD_DEPS`), which turns "we settled the formula early" from a claim
into a rejection. Comparing across formula ids is then a thing S6 has to *notice*.

**Why no acoustics, and why there is a switch at all.**
EU AI Act Art 3(34) defines biometric data as personal data resulting from technical
processing of physical, physiological *or behavioural* characteristics; pitch,
loudness and speaking rate are all in that set, and Art 3(40)'s narrow gate covers
categorisation, not emotion recognition (research round 5 §9.4, round 7 §8). A
salience formula that weighted 「聽起來很在意」 would make every remembered turn the
output of biometric inference. So the four terms are what was *said* and whether the
person *asked* — never how it sounded.

The switch (`PROSODY_ENABLED`, shipped `False`) exists for a reason that is not
politeness: a term that no code path can reach is a term no test can prove is
excluded. T-C6 asks for "vary the acoustic input with the switch off and the score
must not move", and that assertion is **vacuous** unless (a) the acoustic fields
exist on the input a real node would hand us, and (b) something can be shown to read
them. `ACOUSTIC_FIELDS` is declared here, in the module forbidden to read it, so the
test can build its grid from the names rather than from memory; and the switch-on
path is the positive control proving the grid is readable at all. Without that
control, "the scores were identical" is equally well explained by a typo in a field
name. Turning the switch on is not a config change: it changes `FORMULA`, so the
data records that this system began processing biometric data on a given day.

**Why the formula is not a polarity judgement.**
The 0.4 term asks *whether an evaluation was stated*, not whether it was positive.
Polarity is sentiment analysis — a model, a language coverage problem, and a second
place for inference to creep in — and salience does not need it: 「我討厭芫茜」 is
exactly as worth remembering as 「我最愛芫茜」. Detecting statement-hood needs a
closed lexicon of first-person markers and nothing else.

**Why character bigrams for novelty.**
Word-level novelty in Chinese needs a segmenter, which is a dependency and a model,
and 粵語 segmenters are worse than 普通話 ones. Character bigrams need neither and
behave the same in 中文, 粵語 and English. The cost is honest: 「我今天很累」 and
「我今天很忙」 look more similar to this metric than they are. Novelty is weighted
0.3 and is a retention hint, not a claim about meaning.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

from memory.store import StoreError


class SalienceError(StoreError):
    """A turn this formula must not score: a robot turn, or a missing measurement."""


# The formula id that travels with every value. Bump it for *any* change to the
# weights, the terms, the lexicons or the saturation point — the id is what lets S6
# tell "these two numbers disagree" apart from "these two numbers are not the same
# quantity". `tests/test_salience.py` asserts the id changes with the prosody term,
# because that is the change most likely to be made without thinking about the data.
FORMULA = "s3.5/text-4term"
FORMULA_PROSODY = "s3.5/text-4term+prosody"

# Plan §S3.5, in declaration order. The order is *not* part of the formula: the sum
# is taken with `math.fsum`, so reordering the terms cannot move the last bits of the
# result. That is deliberate — a threshold that flips depending on which order the
# terms were added is a defect that only shows up in one of two code paths that
# compute "the same" number.
WEIGHTS = (
    ("explicit", 0.4),   # the person stated an evaluation, of any polarity
    ("novelty", 0.3),    # not a restatement of something in the recent window
    ("asked", 0.2),      # the person asked for it to be remembered
    ("length", 0.1),     # weak evidence: long can also mean rambling
)

# Weight given to the prosody term when it is enabled. The four text weights are
# scaled by `1 - PROSODY_WEIGHT` so the total stays 1.0 and a score stays in [0, 1];
# the values are still not comparable across the two formulas, which is what
# `FORMULA_PROSODY` is for.
PROSODY_WEIGHT = 0.2

# Shipped off. `tests/test_salience.py` asserts this literal in the pushed source,
# not the imported value: a test that reads the module it also configures proves
# nothing about what a fresh clone does.
PROSODY_ENABLED = False

# The only fields the formula may read. A projection rather than a habit of not
# reading the others: `text_only()` is one place to look, and a mutant that wants
# the acoustics has to remove something a reviewer can see.
TEXT_FIELDS = ("role", "text")

# Declared here so T-C6 can vary them, by the module that must not read them. These
# are the fields an S4 turn message would plausibly carry from `asr_node`; every one
# is a behavioural characteristic under Art 3(34).
ACOUSTIC_FIELDS = ("pitch_hz_mean", "pitch_hz_range", "energy_rms", "speech_rate")

# Only the person's own turns. The robot saying "I like that" is not the person
# stating a preference, and a formula that scored both would let the system's own
# output raise the salience of its own output — a feedback loop with no sensor in it.
SPEAKER = "user"

# Characters at which the length term saturates. Saturating rather than normalising
# by the longest turn seen: a single 10 kB paste would otherwise re-scale every
# earlier value, and stored values must not depend on what arrives later.
LENGTH_SATURATION = 120

# Closed lexicons, matched as substrings on a casefolded string. Substring matching
# is coarse on purpose — this is a retention hint whose worst failure is remembering
# something slightly more than it deserved, and a tokeniser here would be the
# dependency the novelty term is written to avoid.
#
# First person only: 「你喜歡芫茜嗎」 is a question about the robot, not the person
# stating a preference.
EXPLICIT_MARKERS = (
    # 中文
    "我喜歡", "我喜欢", "我不喜歡", "我不喜欢", "我愛", "我爱", "我討厭", "我讨厌",
    "我最愛", "我最爱", "我最怕", "我受不了", "我不吃", "我不能吃", "我怕",
    # 「我對」 is deliberately broad: 「我對花生過敏」 is the shape an allergy takes,
    # the discontinuous 我對…過敏 is not a substring, and `allergy` is a predicate the
    # plan calls out as worth never forgetting. The cost is 「我對你說」 scoring 0.4 it
    # did not earn; the cost of the other choice is missing the one fact that matters
    # most. Named here so the next person changes it on purpose.
    "我對", "我对",
    # 粵語
    "我鍾意", "我好鍾意", "我唔鍾意", "我最鍾意", "我好憎", "我唔想", "我唔食",
    # English
    "i like", "i don't like", "i dont like", "i dislike", "i love", "i hate",
    "i prefer", "i can't stand", "i cant stand", "i'm allergic", "im allergic",
    "my favourite", "my favorite",
)

# Note what is *absent*: bare 「記得」. 「你記得我嗎」 is a question about the robot's
# memory, and it is the most common way that word appears in a conversation with a
# robot — a lexicon containing it would score every such question as a request. The
# request forms are 記住 / 記一下 / 別忘 / 唔好唔記得, which are not ambiguous.
ASKED_MARKERS = (
    "記住", "记住", "記一下", "记一下", "別忘", "别忘", "不要忘", "不能忘",
    "唔好唔記得", "唔好忘",
    "remember this", "remember that", "please remember", "don't forget",
    "dont forget", "make a note", "keep in mind",
)


def text_only(turn: Mapping) -> dict:
    """The fields the formula is allowed to see, and nothing else.

    The whole de-biometricisation is this function plus the fact that everything
    downstream takes its output. `score()` never sees the acoustic fields when the
    switch is off, so the exclusion does not depend on any later code remembering.
    """
    return {k: turn[k] for k in TEXT_FIELDS if k in turn}


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def bigrams(text: str) -> tuple[str, ...]:
    """Character bigrams of `text` with whitespace removed, **in order, with repeats**.

    Whitespace goes because English and 中文 disagree about it and novelty should
    not: 「我 好 累」 and 「我好累」 are the same turn. Character bigrams rather than
    words because 中文 needs a segmenter and the 粵語 segmenters are worse; the whole
    point is one tokenisation rule for the whole system, so this is the only place
    that decides what a term is.

    Ordered and repeating, because the two callers need different things from the
    same rule and only one of them can be served by a set: `novelty()` wants the
    Jaccard of two *sets*, while `recall.bm25` needs term frequency — and a set says
    「累」 appears once in 「好累好累好累」. A second tokeniser in `recall.py` would
    have made the retrieval bucket split by which module produced the term, which is
    the defect §S4.5 refused when it declined to normalise `why.goal`.
    """
    packed = "".join(text.split())
    if len(packed) < 2:
        return (packed,) if packed else ()
    return tuple(packed[i:i + 2] for i in range(len(packed) - 1))


def _bigrams(text: str) -> set[str]:
    """The bigram *set*, for the Jaccard in `novelty`. One creator: `bigrams`."""
    return set(bigrams(text))


def novelty(text: str, seen: Iterable[str] = ()) -> float:
    """1 − the closest character-bigram Jaccard against anything in `seen`.

    An empty `seen` scores 1.0: the first thing anyone says is entirely new, which
    is the right answer for a memory system and is also the branch a mutant is most
    likely to get backwards.
    """
    mine = _bigrams(text)
    if not mine:
        return 0.0
    best = 0.0
    for other in seen:
        theirs = _bigrams(other)
        if not theirs:
            continue
        union = len(mine | theirs)
        if union:
            best = max(best, len(mine & theirs) / union)
    return _clamp(1.0 - best)


def _matches(text: str, markers: Iterable[str]) -> bool:
    low = text.casefold()
    return any(m.casefold() in low for m in markers)


def signals(turn: Mapping, seen: Iterable[str] = ()) -> dict[str, float]:
    """The four text signals, each already clamped into [0, 1].

    Returned as a dict rather than folded straight into a total so the harness can
    report which term carried a score. A term outside [0, 1] would silently let one
    signal outweigh its declared weight, so the clamp is here and asserted.
    """
    fields = text_only(turn)
    role = fields.get("role")
    if role != SPEAKER:
        raise SalienceError(
            f"refusing to score a turn with role {role!r}: this formula measures what "
            f"the person said, and {SPEAKER!r} is the only role that is the person")
    text = fields.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SalienceError("a turn must carry non-empty text to be scored")
    return {
        "explicit": 1.0 if _matches(text, EXPLICIT_MARKERS) else 0.0,
        "novelty": novelty(text, seen),
        "asked": 1.0 if _matches(text, ASKED_MARKERS) else 0.0,
        "length": _clamp(len(text) / LENGTH_SATURATION),
    }


def prosody_term(turn: Mapping) -> float:
    """The excluded term, kept honest: it reads the acoustics and says so.

    Only reachable with the switch on. Two refusals rather than two conveniences:

    * A missing field raises rather than defaulting to 0.0. An absent measurement is
      not a measurement of zero, and scoring "no pitch data" as "flat delivery" is an
      inference about how somebody spoke drawn from an absence.
    * An out-of-range value raises rather than being clamped. The fields are
      **already-normalised** [0, 1] features, and clamping raw Hz is worse than
      useless: every plausible pitch clamps to 1.0, so the term stops varying with
      its input while still appearing to read it — a term that looks live and is
      constant. That is exactly the failure T-C6's positive control exists to catch,
      and it must not be reachable by feeding realistic units.
    """
    missing = [f for f in ACOUSTIC_FIELDS if f not in turn]
    if missing:
        raise SalienceError(
            f"the prosody term is enabled but {missing} are absent — an absent "
            "measurement is not a measurement of zero")
    vals = []
    for f in ACOUSTIC_FIELDS:
        v = float(turn[f])
        if not 0.0 <= v <= 1.0:
            raise SalienceError(
                f"{f}={v!r} is not a normalised feature; this term takes [0, 1] and "
                "clamping raw units would make it constant while still reading its "
                "input")
        vals.append(v)
    return _clamp(math.fsum(vals) / len(vals))


def score(turn: Mapping, seen: Iterable[str] = (), *,
          prosody: bool | None = None) -> float:
    """The salience of one turn, in [0, 1]. `prosody=None` means the shipped default.

    `fsum` rather than a running total: two code paths that compute "the same"
    salience — the live one and a rebuild — must not differ by an ulp and flip a
    retention threshold.
    """
    if prosody is None:
        prosody = PROSODY_ENABLED
    parts = signals(turn, seen)
    scale = (1.0 - PROSODY_WEIGHT) if prosody else 1.0
    terms = [parts[name] * weight * scale for name, weight in WEIGHTS]
    if prosody:
        terms.append(prosody_term(turn) * PROSODY_WEIGHT)
    return _clamp(math.fsum(terms))


def formula_id(prosody: bool | None = None) -> str:
    """Which formula `score()` would use. The value that must travel with a score."""
    if prosody is None:
        prosody = PROSODY_ENABLED
    return FORMULA_PROSODY if prosody else FORMULA


def stamp(record: dict, turn: Mapping, seen: Iterable[str] = (), *,
          prosody: bool | None = None) -> dict:
    """Attach a score and its formula id to `record`. The only way to attach one.

    A returned copy rather than an in-place edit: `Store.append` rejects a record
    that already carries `seq`/`ts`, and a caller that mutated its own dict would
    find out at the second call site, not the first.
    """
    out = dict(record)
    out["salience"] = score(turn, seen, prosody=prosody)
    out["salience_formula"] = formula_id(prosody)
    return out


def weight_findings() -> list[str]:
    """Everything wrong with the declared weights. Empty is the only passing value.

    A pure function rather than a module-import assertion: an `assert` at import
    time is removed by `python -O`, and this is checked by the test suite *and*
    available to a node that wants to refuse to boot.
    """
    findings: list[str] = []
    total = math.fsum(w for _, w in WEIGHTS)
    if total != 1.0:
        findings.append(f"the four text weights sum to {total!r}, not 1.0 — a score "
                        "is then not in [0, 1] and two turns are not comparable")
    names = [n for n, _ in WEIGHTS]
    if len(set(names)) != len(names):
        findings.append(f"duplicate term name in WEIGHTS: {names}")
    if not 0.0 < PROSODY_WEIGHT < 1.0:
        findings.append(f"PROSODY_WEIGHT is {PROSODY_WEIGHT!r}, which is not a weight")
    overlap = sorted(set(TEXT_FIELDS) & set(ACOUSTIC_FIELDS))
    if overlap:
        findings.append(f"{overlap} is both a text field and an acoustic one — the "
                        "projection cannot exclude what it also admits")
    return findings
