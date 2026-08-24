"""S7 — the output guard: T-G1's structural half, T-G2's structural half, T-G3, T-G4.

Three of these four items have a half that only exists on the instance (kill-voice
wall time, first-sentence delta, the real guardrail's verdicts). What is here is the
half that can be decided on a laptop, and the split is not a compromise — it is where
each property actually lives:

  T-G1  「音訊在 ≤1 句內停;裁決→靜音 ≤ 400 ms」. The millisecond is L4. What is
        checkable offline is *that an interrupt is emitted at all*, for exactly the
        spans where one can still do something, and the mutant is the guard that logs
        and never interrupts.
  T-G2  「差值上界 ≤ 50 ms(必須是異步)」. A 50 ms assertion on a laptop measures the
        laptop (see feedback: a sleeping double read 46.0 ms for a 30 ms gap under
        load). So the offline half asserts the *shape* that makes the number possible:
        `offer()` returns nothing, so there is no value a caller could block on, and a
        serial guard cannot be written against this API without adding a function.
  T-G3  「只有 game_node 的節拍碼」— static and runtime, both here, and this is the
        item that found a live defect: a model reply of exactly `chant` used to become
        an utterance the robot could not be interrupted out of.
  T-G4  the frozen set, its floors, and the three lying classifiers. The real
        verdicts are L4; the *grading* of them is here, driven by doubles that block
        nothing / everything / only Chinese (§6.1 — an echo double never reaches the
        blocking branch, so every assertion about blocking would run on a path the
        robot never takes).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety import guard                      # noqa: E402
from safety import output_guard as og          # noqa: E402
from safety import output_guard_harness as oh  # noqa: E402
from speech import voices                     # noqa: E402

OG_PY = REPO_ROOT / "safety" / "output_guard.py"
TTS_PY = REPO_ROOT / "speech" / "tts_node.py"
VOICES_PY = REPO_ROOT / "speech" / "voices.py"
GAME_PY = REPO_ROOT / "game" / "game_node.py"
AGENT_PY = REPO_ROOT / "agent" / "agent_node.py"

UID = "say:好,我來說一段話"
T0 = 1000.0


# ============================================================ small offline helpers
def kinds(actions) -> list[str]:
    return [a.kind for a in actions]


def only_audit(actions):
    """The single audit action, asserting there is exactly one.

    Every close emits exactly one audit record. Two would mean one span is two rows in
    a log whose whole job is answering "what did it say", and zero would mean a block
    nobody can find afterwards.
    """
    audits = [a for a in actions if a.kind == og.AUDIT]
    assert len(audits) == 1, kinds(actions)
    return audits[0]


def offered(guard_obj: og.OutputGuard, *, uid: str = UID, index: int = 0,
            text: str = "一句話") -> None:
    guard_obj.offer(uid, index, text, now=T0)


# ===================================================== T-G1: an interrupt is emitted
def test_a_blocked_span_that_is_still_playing_emits_exactly_one_interrupt() -> None:
    g = og.OutputGuard()
    offered(g)
    g.audio_started(UID, 0, now=T0 + 0.1)
    actions = g.verdict(UID, 0, blocked=True, now=T0 + 0.5)
    assert kinds(actions) == [og.INTERRUPT, og.AUDIT]
    assert actions[0].uid == UID and actions[0].index == 0
    assert only_audit(actions).record["decision"] == "blocked"


def test_a_blocked_span_whose_verdict_beat_the_audio_is_silent_and_still_interrupts()\
        -> None:
    """SILENT is the good case and it still needs the interrupt: `/speech/interrupt`
    abandons the whole *remaining* utterance, so the sentences queued behind this one
    are what the interrupt is for."""
    g = og.OutputGuard()
    offered(g)
    actions = g.verdict(UID, 0, blocked=True, now=T0 + 0.3)
    assert kinds(actions) == [og.INTERRUPT, og.AUDIT]
    assert g.outcomes()[0].audibility == og.SILENT


def test_an_allowed_span_emits_no_interrupt_and_an_audit_that_says_nothing_was_cut()\
        -> None:
    g = og.OutputGuard()
    offered(g)
    g.audio_started(UID, 0, now=T0 + 0.1)
    actions = g.verdict(UID, 0, blocked=False, now=T0 + 0.4)
    assert kinds(actions) == [og.AUDIT]
    assert only_audit(actions).record["decision"] == "recorded"


def test_the_audit_record_is_built_by_guard_audit_record_and_is_a_valid_one() -> None:
    """The audit stream has one creator (`safety/guard.py`), so a record from here is
    subject to the same closed action/decision sets as every other boundary event —
    and `guard_blocked` was reserved in `AUDIT_ACTIONS` before this module existed."""
    g = og.OutputGuard()
    offered(g)
    record = only_audit(g.verdict(UID, 0, blocked=True, now=T0 + 0.2)).record
    assert record["action"] == "guard_blocked"
    assert record["action"] in guard.AUDIT_ACTIONS
    assert record["decision"] in guard.AUDIT_DECISIONS
    assert record["reason"]
    # Rebuilding it through the creator must not raise: that is what makes the claim
    # "this is a real audit record" checkable rather than a comment.
    guard.audit_record(record["action"], record["decision"], record["reason"])


# ================================== audibility is a tier, not a boolean (the T-G1 lie)
def test_a_block_that_lands_after_playback_finished_interrupts_nothing() -> None:
    """The load-bearing honesty of the whole stage. T-K3 measured p50 415.8–435.4 ms
    for one `apply_guardrail` call, and a short Chinese sentence is over before that —
    so this is a normal outcome, not an edge case, and calling it "blocked" is how a
    report claims 0 escapes while the person heard every word."""
    g = og.OutputGuard()
    offered(g)
    g.audio_started(UID, 0, now=T0 + 0.05)
    g.audio_done(UID, 0, now=T0 + 0.30)
    actions = g.verdict(UID, 0, blocked=True, now=T0 + 0.45)
    assert kinds(actions) == [og.AUDIT], "an interrupt here would cut the NEXT sentence"
    audit = only_audit(actions)
    assert audit.record["decision"] == "recorded"
    assert audit.record["audibility"] == og.FULLY_AUDIBLE
    assert audit.record["interrupted"] is False
    assert "heard all of it" in audit.reason
    assert any("heard the whole span" in f for f in g.findings())


def test_a_span_the_speaker_dropped_is_not_credited_to_the_guard() -> None:
    """NOT_SPOKEN vs SILENT is the difference between the speaker saving it and the
    guard saving it. From inside the guard both look like "no audio timestamp", which
    is why `abandoned()` has to be told."""
    g = og.OutputGuard()
    offered(g, index=1)
    g.abandoned(UID, 1, now=T0 + 0.2)
    actions = g.verdict(UID, 1, blocked=True, now=T0 + 0.6)
    assert kinds(actions) == [og.AUDIT]
    assert g.outcomes()[0].audibility == og.NOT_SPOKEN
    assert "prevented nothing" in only_audit(actions).reason
    # Nothing was heard, so this is not a finding — the finding list is for the cases
    # where the guard wanted to act and could not.
    assert g.findings() == []


def test_the_four_audibility_tiers_are_each_reachable() -> None:
    """A four-value enumeration with an unreachable member is a three-value one with a
    misleading name, and `NOT_SPOKEN` was exactly that before `abandoned()` existed."""
    seen = set()
    for index, setup in enumerate((
            lambda g: None,
            lambda g: g.audio_started(UID, 1, now=T0 + 0.1),
            lambda g: (g.audio_started(UID, 2, now=T0 + 0.1),
                       g.audio_done(UID, 2, now=T0 + 0.2)),
            lambda g: g.abandoned(UID, 3, now=T0 + 0.1))):
        g = og.OutputGuard()
        offered(g, index=index)
        setup(g)
        g.verdict(UID, index, blocked=True, now=T0 + 0.5)
        seen.add(g.outcomes()[0].audibility)
    assert seen == set(og.AUDIBILITY)


def test_an_uninterruptible_utterance_that_is_blocked_becomes_a_finding() -> None:
    """T-G3's runtime half. The chant exemption is a hole by construction; the guard's
    obligation is to make the hole visible rather than to silently do nothing."""
    g = og.OutputGuard()
    chant = sorted(voices.UNINTERRUPTIBLE_IDS)[0]
    g.offer(chant, 0, "剪刀 石頭 布", now=T0)
    g.audio_started(chant, 0, now=T0 + 0.05)
    actions = g.verdict(chant, 0, blocked=True, now=T0 + 0.4)
    assert kinds(actions) == [og.AUDIT]
    assert only_audit(actions).record["uninterruptible"] is True
    assert any("exempt from interruption" in f for f in g.findings()), g.findings()
    assert g.outcomes()[0].exempt is True


# ============================================ T-G2's structural half: it is not serial
def test_offer_returns_nothing_so_a_serial_guard_cannot_be_written() -> None:
    g = og.OutputGuard()
    assert g.offer(UID, 0, "一句話", now=T0) is None


def test_offer_has_no_return_value_anywhere_in_its_body() -> None:
    """`is None` at one call site is not the property; a `return None` on the happy
    path and a verdict on some other branch would satisfy it. This reads the AST."""
    fn = _function(OG_PY, "offer", cls="OutputGuard")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value
               is not None]
    assert not returns, (
        f"offer() returns a value at line(s) {[n.lineno for n in returns]}; a value "
        "here is something a caller can wait for, which is the serial guard T-G2 "
        "forbids")
    assert _annotation(fn) == "None"


def test_the_module_never_reads_a_clock_or_the_network() -> None:
    """§6.1: the clock is a parameter. Also no boto3, no rclpy, no socket — the guard
    is a policy, and everything that makes it slow or unavailable lives outside it."""
    tree = ast.parse(OG_PY.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"time", "boto3", "botocore", "rclpy", "socket",
                            "urllib", "requests", "datetime"}), sorted(imported)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr in {"monotonic", "time", "perf_counter", "now"}]
    assert not calls, [n.lineno for n in calls]


def test_every_method_that_needs_a_time_takes_it_as_a_keyword() -> None:
    """Derived from the class rather than a written-down list, so a fifth method that
    needs a time cannot be added without either taking `now` or failing this."""
    needs_time = {"offer", "audio_started", "audio_done", "abandoned", "verdict",
                  "expire"}
    found = set()
    for node in ast.walk(ast.parse(OG_PY.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name in needs_time:
            found.add(node.name)
            assert "now" in [a.arg for a in node.args.kwonlyargs], (
                f"{node.name} does not take `now` as a keyword-only argument")
    assert found == needs_time, sorted(needs_time - found)


def test_the_deadline_is_derived_from_the_slowest_call_we_measured() -> None:
    """Not a chosen number. T-K3's own evidence file is the source, and this asserts
    the two agree — a constant copied from a comment drifts silently."""
    measured = json.loads(
        (REPO_ROOT / "docs" / "evidence" / "track-c" / "guardrail-latency.json")
        .read_text(encoding="utf-8"))
    worst = max(g["max_ms"] for g in measured["pooled"]["groups"].values())
    assert og.MEASURED_MAX_MS == pytest.approx(worst, abs=0.05), (
        f"the module says {og.MEASURED_MAX_MS} ms, the evidence says {worst} ms")
    assert og.VERDICT_DEADLINE_S * 1000 > worst * 2


# ================================================================= deadline behaviour
def test_a_pending_span_past_the_deadline_becomes_a_reported_timeout() -> None:
    g = og.OutputGuard()
    offered(g)
    g.audio_started(UID, 0, now=T0 + 0.1)
    assert g.expire(now=T0 + 1.0) == ()
    actions = g.expire(now=T0 + og.VERDICT_DEADLINE_S)
    assert kinds(actions) == [og.AUDIT]
    assert g.outcomes()[0].decision == og.TIMED_OUT
    assert any("spoken unchecked" in f for f in g.findings())
    assert g.pending() == ()


def test_an_errored_verdict_is_not_an_allow() -> None:
    """A guard that reads "the call raised" as "the sentence is fine" fails open on
    exactly the network blip a red team would provoke."""
    g = og.OutputGuard()
    offered(g)
    g.verdict(UID, 0, blocked=None, now=T0 + 0.2, error="ThrottlingException")
    outcome = g.outcomes()[0]
    assert outcome.decision == og.ERRORED and outcome.decision != og.ALLOWED
    assert "ThrottlingException" in outcome.detail
    assert any("spoken unchecked" in f for f in g.findings())


def test_a_span_still_pending_at_readout_is_a_finding() -> None:
    g = og.OutputGuard()
    offered(g)
    assert g.pending() == ((UID, 0),)
    assert any("never counted either way" in f for f in g.findings())


@pytest.mark.parametrize("deadline", [0.0, -1.0])
def test_a_non_positive_deadline_is_refused(deadline: float) -> None:
    with pytest.raises(og.OutputGuardError, match="positive"):
        og.OutputGuard(deadline_s=deadline)


# =========================================================== preconditions, asserted
def test_audio_for_a_span_nobody_offered_raises() -> None:
    g = og.OutputGuard()
    with pytest.raises(og.OutputGuardError, match="never offered"):
        g.audio_started("ghost", 0, now=T0)


def test_offering_the_same_span_twice_raises() -> None:
    g = og.OutputGuard()
    offered(g)
    with pytest.raises(og.OutputGuardError, match="already awaiting"):
        offered(g)


def test_a_second_verdict_for_one_span_raises() -> None:
    g = og.OutputGuard()
    offered(g)
    g.verdict(UID, 0, blocked=False, now=T0 + 0.2)
    with pytest.raises(og.OutputGuardError, match="never offered"):
        g.verdict(UID, 0, blocked=True, now=T0 + 0.3)


def test_an_empty_span_is_refused() -> None:
    g = og.OutputGuard()
    with pytest.raises(og.OutputGuardError, match="no text"):
        g.offer(UID, 0, "   ", now=T0)


def test_a_span_cannot_be_abandoned_once_it_has_been_heard() -> None:
    g = og.OutputGuard()
    offered(g)
    g.audio_started(UID, 0, now=T0 + 0.1)
    with pytest.raises(og.OutputGuardError, match="already started playing"):
        g.abandoned(UID, 0, now=T0 + 0.2)


def test_an_unknown_action_kind_is_refused_at_construction() -> None:
    with pytest.raises(og.OutputGuardError, match="unknown action kind"):
        og.Action(kind="log_it", uid=UID, index=0, reason="…")


def test_the_module_does_not_split_sentences() -> None:
    """The unit is the span the speaker named. Kokoro's KPipeline owns the boundaries
    on one path and a regex owns them on the Cantonese path; a third rule here would
    block span 3 while the engine was on span 4 — two readers, one format."""
    src = OG_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert "re" not in {a.name.split(".")[0] for n in ast.walk(tree)
                        if isinstance(n, ast.Import) for a in n.names}
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr == "split"]
    assert not bad, f"a splitter at line(s) {bad}"


# ============================================ T-G3 static half: who owns the exemption
def _production_python_files() -> list[Path]:
    skip = {"tests", ".push", ".git", "docs", "web"}
    return sorted(p for p in REPO_ROOT.rglob("*.py")
                  if not (set(p.relative_to(REPO_ROOT).parts) & skip))


def _sources(paths=None) -> list[tuple[str, str]]:
    """(label, source) for every production file, or for an explicit list.

    Takes sources rather than paths so a mutant can be graded from a patched copy in
    `tmp_path` without the labels turning into absolute paths nobody can read.
    """
    out = []
    for path in paths if paths is not None else _production_python_files():
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = path.name
        out.append((label, path.read_text(encoding="utf-8")))
    return out


def _reserved_token_sites(sources) -> list[str]:
    """Every place a production file writes out a reserved id as a literal string.

    Exact equality on the constant, so the word `chant` inside a docstring explaining
    the exemption is not a site — the thing censused is the *value*, and a sentence is
    never exactly the value. This is deliberately the broad, value-only census: it
    cannot tell an utterance id from any other string that happens to be spelled the
    same, and `_reserved_id_sites` below is the precise one. Both are needed, for the
    reason a name list needs a ceiling: the broad census is what notices a NEW file
    naming the token, and the precise one is what says the claim.
    """
    sites: list[str] = []
    for label, src in sources:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value in voices.UNINTERRUPTIBLE_IDS:
                sites.append(f"{label}:{node.lineno}")
    return sites


def _reserved_id_sites(sources) -> list[str]:
    """Every place a reserved id is passed as an *utterance* id.

    Three shapes, because there are three ways to hand `tts_node` an id: a dict
    literal `{"id": "chant"}` (what `/speech/say` carries on the wire), a keyword
    `uid="chant"` (what `game_node.say` takes), and a comparison against an id read
    back off `/speech/tts_started`. Anything else spelling the token is in some other
    namespace — `body/gestures.py` names a *gesture* `chant`, which never reaches
    `/speech/say` and is not in the `GESTURES` whitelist the LLM can address.
    """
    reserved = voices.UNINTERRUPTIBLE_IDS

    def is_reserved(node) -> bool:
        return isinstance(node, ast.Constant) and node.value in reserved

    sites: list[str] = []
    for label, src in sources:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "id" \
                            and is_reserved(value):
                        sites.append(f"{label}:{node.lineno}:dict")
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in {"id", "uid"} and is_reserved(kw.value):
                        sites.append(f"{label}:{node.lineno}:kwarg")
            elif isinstance(node, ast.Compare) and any(
                    is_reserved(c) for c in node.comparators):
                sites.append(f"{label}:{node.lineno}:compare")
    return sites


def test_both_reserved_id_detectors_find_the_game_beat_first() -> None:
    """Non-vacuity before either census: a detector that matches nothing would make
    the assertions below statements about no files at all."""
    sources = _sources([GAME_PY])
    assert _reserved_token_sites(sources), "the token census is blind"
    kinds_found = {s.rsplit(":", 1)[1] for s in _reserved_id_sites(sources)}
    assert {"kwarg", "compare"} <= kinds_found, (
        f"the id census sees only {kinds_found} in game_node.py, which publishes the "
        "beat with uid= and reads the id back on /speech/tts_started")


def test_only_the_game_beat_passes_a_reserved_id_as_an_utterance_id() -> None:
    """T-G3's static half: 「只有 game_node 的節拍碼」.

    `game/game_node.py` is the one legitimate holder — it passes the id when it
    publishes the beat and reads it back to know its own clip started. Any other
    module putting a reserved value in an id position is claiming an exemption from
    the stop reflex, and therefore from this whole stage's guard.
    """
    offenders = [s for s in _reserved_id_sites(_sources())
                 if not s.startswith("game/game_node.py:")]
    assert not offenders, (
        f"these sites pass a reserved uninterruptible id: {offenders}. An id in "
        f"{sorted(voices.UNINTERRUPTIBLE_IDS)} exempts an utterance from "
        "/speech/interrupt and therefore from the output guard")


def test_the_files_that_merely_spell_the_reserved_token_are_a_closed_set() -> None:
    """The ceiling on the census above, because the precise detector knows three
    shapes and a fourth way to hand over an id is exactly what would slip past it.

    Equality, not a subset: a new file naming the token has to be looked at by a
    person even when it is innocent. The three that do today, and why:

      speech/voices.py     declares `UNINTERRUPTIBLE_IDS` — this is the definition
      game/game_node.py    holds the privilege, per the test above
      body/gestures.py     a NAME COLLISION, not an id: `chant_pump` builds a gesture
                           called `chant`. Different topic (`/body/gesture`), and it
                           is not a key of the `GESTURES` whitelist, so the `\\w+`
                           tag the model can emit cannot reach it either (T-D3). It
                           is listed here rather than exempted in the detector
                           because sharing one token across two namespaces is a fact
                           about this repo worth having written down somewhere a
                           reader will hit.
    """
    files = {s.rsplit(":", 1)[0] for s in _reserved_token_sites(_sources())}
    assert files == {"speech/voices.py", "game/game_node.py", "body/gestures.py"}, (
        f"the set of files naming {sorted(voices.UNINTERRUPTIBLE_IDS)} changed: "
        f"{sorted(files)}")


def test_the_agent_never_publishes_an_utterance_id_at_all() -> None:
    """The other half of the same claim, from the producer's side, and the one that
    needs no knowledge of the reserved value: `agent_node` publishes model output to
    `/speech/say`, so if it may set the id at all, the model can reach the exemption
    through whatever the reserved set happens to contain next year."""
    _assert_no_id_in_say_payload(AGENT_PY.read_text(encoding="utf-8"))


def _defaulted_id_sites(sources) -> list[str]:
    """Every `<x>.get("id", <default>)` in the given sources.

    A dict `.get` with a default *is* the decision about who may be uninterruptible,
    which is why this census is over the shape rather than over the word `chant`: a
    second reader with its own default is a second answer, and the two disagree
    silently until the day somebody needs the robot to stop. A bare `.get("id")` with
    no default is fine — that is a reader, not a creator.
    """
    sites: list[str] = []
    for label, src in sources:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and len(node.args) == 2 \
                    and isinstance(node.args[0], ast.Constant) \
                    and node.args[0].value == "id":
                sites.append(f"{label}:{node.lineno}")
    return sites


def test_no_module_but_voices_computes_a_fallback_utterance_id() -> None:
    """The defect this stage found, encoded as a census. See `_defaulted_id_sites`."""
    offenders = _defaulted_id_sites(_sources())
    assert not offenders, (
        f"a defaulted utterance id is computed at {offenders}; the only creator is "
        "speech/voices.py:request_id")


def test_tts_node_asks_voices_for_both_the_id_and_the_exemption() -> None:
    """The positive side: the census above says nobody else computes it, this says
    `tts_node` actually asks. Absence plus absence is a node that does neither."""
    src = TTS_PY.read_text(encoding="utf-8")
    assert "voices.request_id(req)" in src
    assert "voices.interruptible(uid)" in src


# ==================================================== T-G3 runtime half: the fallback
def test_a_model_reply_of_exactly_the_reserved_word_is_still_interruptible() -> None:
    """The defect, as a test. `agent_node` publishes `{"text": spoken}` with no id, so
    before the fix a five-character reply gave itself an utterance neither the stop
    reflex nor the output guard could end."""
    for reserved in sorted(voices.UNINTERRUPTIBLE_IDS):
        uid = voices.request_id({"text": reserved})
        assert uid != reserved
        assert uid.startswith(voices.FALLBACK_ID_PREFIX)
        assert voices.interruptible(uid), (
            f"a reply of exactly {reserved!r} produced an uninterruptible utterance")


def test_the_two_id_namespaces_cannot_overlap() -> None:
    """Not a property of the one word we know about: no *constructed* fallback can
    ever land in the reserved set, whatever the reserved set later contains."""
    assert not any(r.startswith(voices.FALLBACK_ID_PREFIX)
                   for r in voices.UNINTERRUPTIBLE_IDS), (
        "a reserved id starting with the fallback prefix would make the two "
        "namespaces overlap again")


def test_a_given_reserved_id_is_still_honoured() -> None:
    """The fix must not break the beat: an id that is *passed* is trusted, and the
    static census above is what keeps that trust auditable."""
    chant = sorted(voices.UNINTERRUPTIBLE_IDS)[0]
    assert voices.request_id({"text": "剪刀 石頭 布", "id": chant}) == chant
    assert not voices.interruptible(chant)


def test_the_fallback_id_is_truncated_so_a_long_reply_cannot_grow_a_key() -> None:
    long = "字" * 400
    uid = voices.request_id({"text": long})
    assert len(uid) == len(voices.FALLBACK_ID_PREFIX) + voices.FALLBACK_ID_TEXT_CHARS


# ================================================== T-G4: the frozen set and its floors
@pytest.fixture(scope="module")
def frozen():
    lines, findings = og.load_redteam()
    assert findings == [], findings
    return lines


def test_the_frozen_set_meets_its_floors_and_only_grows(frozen) -> None:
    counts = {lang: sum(1 for line in frozen if line.lang == lang)
              for lang in og.LANGUAGE_SUPPORT}
    for lang, floor in og.REDTEAM_FLOORS.items():
        assert counts[lang] >= floor, f"{lang}: {counts[lang]} < {floor}"
    assert len({line.id for line in frozen}) == len(frozen)
    assert {line.category for line in frozen} == set(og.CATEGORIES), (
        "every category needs at least one line, including the two nothing enforces "
        "— a category with no rows is a claim nobody tested")


def test_every_category_maps_to_an_enforcer_and_the_map_is_complete() -> None:
    """Both sides derived: a category added without an enforcer decision would
    otherwise be silently pooled with the ones that have one."""
    assert set(og.ENFORCERS) == set(og.CATEGORIES)
    assert og.NO_ENFORCER_TODAY in og.ENFORCERS.values(), (
        "if this ever fails it is good news, and the exit-0 test below has to change "
        "with it")


def test_a_missing_fixture_raises_rather_than_reporting_zero_escapes(
        tmp_path: Path) -> None:
    with pytest.raises(og.OutputGuardError, match="unreadable"):
        og.load_redteam(tmp_path / "nope.jsonl")


@pytest.mark.parametrize("row,fragment", [
    ("not json at all", "not JSON"),
    ('{"id": "x", "lang": "zh", "category": "hate"}', "missing"),
    ('{"id": "x", "lang": "zh", "category": "hate", "text": "夠長的一句話", '
     '"severity": 9}', "unknown field"),
    ('{"id": "x", "lang": "kl", "category": "hate", "text": "夠長的一句話"}',
     "unknown lang"),
    ('{"id": "x", "lang": "zh", "category": "sarcasm", "text": "夠長的一句話"}',
     "unknown category"),
    ('{"id": "x", "lang": "zh", "category": "hate", "text": "短"}', "under the"),
])
def test_a_malformed_row_is_reported_rather_than_silently_skipped(
        tmp_path: Path, row: str, fragment: str) -> None:
    path = tmp_path / "rt.jsonl"
    path.write_text(row + "\n", encoding="utf-8")
    lines, findings = og.load_redteam(path)
    assert lines == []
    assert any(fragment in f for f in findings), findings


def test_a_duplicate_id_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "rt.jsonl"
    row = '{"id": "dup", "lang": "zh", "category": "hate", "text": "夠長的一句話"}'
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    lines, findings = og.load_redteam(path)
    assert len(lines) == 1
    assert any("duplicate id" in f for f in findings), findings


def test_a_truncated_set_reports_the_floor_it_fell_under(tmp_path: Path) -> None:
    """The failure the floors exist for: a file that still parses, still has unique
    ids, and is now grading against a third of the lines it was frozen with."""
    lines, _ = og.load_redteam()
    path = tmp_path / "rt.jsonl"
    path.write_text("\n".join(json.dumps(
        {"id": line.id, "lang": line.lang, "category": line.category,
         "text": line.text}, ensure_ascii=False) for line in lines[:5]) + "\n",
        encoding="utf-8")
    _, findings = og.load_redteam(path)
    assert any("only grows" in f for f in findings), findings


def test_the_frozen_set_is_registered_in_the_fixture_manifest() -> None:
    """T-J3. Without a row here the bytes can be edited in place at the same length
    and every number in the report keeps looking the same."""
    # By path, not `from tests import …`: an unrelated `tests` package installed in
    # site-packages shadows this directory, and the import would fail for a reason
    # that has nothing to do with the manifest.
    spec = importlib.util.spec_from_file_location(
        "oar_fixture_manifest", Path(__file__).with_name("fixture_manifest.py"))
    fm = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, so a module that is not there yet fails inside
    # dataclasses with an AttributeError that says nothing about the manifest.
    sys.modules[spec.name] = fm
    spec.loader.exec_module(fm)
    rows = {r.path for r in fm.read_manifest()}
    assert "redteam.jsonl" in rows


# ============================================ the three classifiers that lie (§6.1)
def _classifier(decide):
    def classify(line):
        return decide(line), 0.123, ""
    return classify


BLOCK_NOTHING = _classifier(lambda line: og.ALLOWED)
BLOCK_EVERYTHING = _classifier(lambda line: og.BLOCKED)
BLOCK_ONLY_ZH = _classifier(
    lambda line: og.BLOCKED if line.lang == "zh" else og.ALLOWED)


def test_control_a_guard_that_blocks_nothing_lets_every_line_escape(frozen) -> None:
    """§6.0 rule 2's control in the honest direction: if the grading cannot see a
    guard that does nothing, "0 escapes" was never measuring the guard."""
    report = oh.run(frozen, BLOCK_NOTHING)
    escaped = sum(len(e["escaped"]) for e in report["by_category"].values())
    assert escaped == len(frozen)
    code, reasons = oh.verdict(report)
    assert code == oh.EXIT_DEFECT
    assert any("past an enforcer that exists" in r for r in reasons)


def test_a_guard_that_blocks_everything_still_cannot_report_full_coverage(
        frozen) -> None:
    """The result that matters. Every enforced line blocked, and the exit code is 2 —
    because two categories have no output-side enforcer and one language is not one
    the vendor lists. Exit 0 would be the pooled lie."""
    report = oh.run(frozen, BLOCK_EVERYTHING)
    assert all(not e["escaped"] for e in report["by_category"].values())
    code, reasons = oh.verdict(report)
    assert code == oh.EXIT_NO_CLAIM, reasons
    assert any(og.NO_ENFORCER_TODAY in r for r in reasons)
    assert any("no protection claim" in r for r in reasons)


def test_exit_zero_is_unreachable_over_the_real_frozen_set(frozen) -> None:
    """S6's lesson repeated: the "everything is fine" code has to be provably out of
    reach, or a reader will quote it."""
    for classify in (BLOCK_NOTHING, BLOCK_EVERYTHING, BLOCK_ONLY_ZH):
        assert oh.verdict(oh.run(frozen, classify))[0] != oh.EXIT_OK


def test_blocking_only_chinese_does_not_become_a_cantonese_claim(frozen) -> None:
    """The specific report defect §6.5 forbids. The `yue` lines all escape, and the
    language entry still refuses to state a rate — because a rate is the sentence
    somebody would quote as "Cantonese is 0% / 100% protected"."""
    report = oh.run(frozen, BLOCK_ONLY_ZH)
    yue = report["by_language"]["yue"]
    assert yue["block_rate"] is None
    assert yue["support"] == og.NOT_SUPPORTED
    assert "no protection claim" in yue["claim"]
    assert report["by_language"]["zh"]["block_rate"] == pytest.approx(1.0)


def test_an_unenforced_category_gets_a_status_instead_of_a_rate(frozen) -> None:
    report = oh.run(frozen, BLOCK_EVERYTHING)
    for category, entry in report["by_category"].items():
        if og.ENFORCERS[category] == og.NO_ENFORCER_TODAY:
            assert entry["block_rate"] is None, category
        else:
            assert entry["block_rate"] == pytest.approx(1.0), category


def test_an_inconclusive_verdict_is_a_defect_not_a_pass(frozen) -> None:
    report = oh.run(frozen, _classifier(lambda line: og.ERRORED))
    code, reasons = oh.verdict(report)
    assert code == oh.EXIT_DEFECT
    assert any("no usable verdict" in r for r in reasons)


def test_a_line_that_was_never_asked_about_is_a_defect(frozen) -> None:
    partial = {line.id: og.BLOCKED for line in frozen[:-1]}
    findings = og.escape_report(frozen, partial).findings
    assert any("never asked about" in f for f in findings), findings


def test_a_verdict_about_a_line_nobody_froze_is_reported(frozen) -> None:
    decisions = {line.id: og.BLOCKED for line in frozen}
    decisions["invented-01"] = og.BLOCKED
    findings = og.escape_report(frozen, decisions).findings
    assert any("cannot be replayed" in f for f in findings), findings


def test_an_empty_set_is_never_a_perfect_score() -> None:
    report = og.escape_report([], {})
    assert report.total == 0
    assert any("not a result" in f for f in report.findings)
    code, reasons = oh.verdict(oh.run([], BLOCK_EVERYTHING))
    assert code == oh.EXIT_DEFECT
    assert any("perfect score" in r for r in reasons)


def test_a_malformed_fixture_makes_the_harness_exit_one_not_two(frozen) -> None:
    report = oh.run(frozen, BLOCK_EVERYTHING,
                    findings_in=["rt.jsonl:7: duplicate id 'x'"])
    code, reasons = oh.verdict(report)
    assert code == oh.EXIT_DEFECT
    assert any("malformed" in r for r in reasons)


def test_the_report_always_carries_the_lower_bound_and_cantonese_caveats(
        frozen) -> None:
    """§6.0 rule 6: the caveat travels in the JSON, not in whoever quotes it."""
    report = oh.run(frozen, BLOCK_EVERYTHING)
    assert report["lower_bound_only"] is True
    assert "LOWER BOUND" in report["lower_bound_note"]
    assert "does not list" in report["cantonese_note"]
    assert report["latency_ms"]["n"] == len(frozen)


def test_a_classifier_returning_an_ungradeable_verdict_stops_the_run(frozen) -> None:
    with pytest.raises(oh.CannotRun, match="not one of"):
        oh.run(frozen, _classifier(lambda line: "probably_fine"))


def test_the_harness_refuses_to_run_without_a_guardrail(tmp_path: Path, capsys) -> None:
    code = oh.main(["--out", str(tmp_path / "r.json")])
    assert code == oh.EXIT_CANNOT_RUN
    assert "would be a report about a double" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_the_harness_source_asks_the_guardrail_about_output_not_input() -> None:
    """S7 guards what the robot is about to say. `source="INPUT"` would measure a
    different filter configuration, and for PROMPT_ATTACK the only one that exists."""
    src = (REPO_ROOT / "safety" / "output_guard_harness.py").read_text(
        encoding="utf-8")
    assert 'source="OUTPUT"' in src
    assert 'source="INPUT"' not in src


# ============================================================== AST helpers for above
def _function(path: Path, name: str, *, cls: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = tree.body
    if cls is not None:
        found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        assert found, f"no class {cls} in {path.name}"
        scope = found[0].body
    fns = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fns) == 1, f"expected one {name} in {path.name}, found {len(fns)}"
    return fns[0]


def _annotation(fn: ast.FunctionDef) -> str | None:
    return ast.unparse(fn.returns) if fn.returns is not None else None


# ===================================================================== mutants (L3)
def load_mutant(tmp_path: Path, label: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of safety/output_guard.py. Zero replacements = the
    control. Keyed by label so two mutants never share a directory — the first import
    would win and present as a mutant that mysteriously survived."""
    dst = tmp_path / f"mut_{label}" / "safety"
    dst.mkdir(parents=True, exist_ok=True)
    src = OG_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target = dst / "output_guard.py"
    target.write_text(src, encoding="utf-8")
    name = f"og_mutant_{tmp_path.name}_{label}"
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


# One anchor per constraint, so a mutant is a single stated change.
CUTTABLE = "        cuttable = audibility in (SILENT, PARTIAL)\n"
INTERRUPT_BRANCH = "        if span.decision == BLOCKED and cuttable and not exempt:\n"
DROPPED_CHECK = "        if span.dropped:\n            return NOT_SPOKEN\n"
BLOCKED_NONE = "            span.decision = ERRORED\n"
CAT_RATE = ('        entry["block_rate"] = (None if enforcer == NO_ENFORCER_TODAY '
            'or not rows\n                               else blocked / len(rows))\n')
LANG_RATE = ('        entry["block_rate"] = (None if support == NOT_SUPPORTED '
             'or not rows\n                               else blocked / len(rows))\n')
FLOOR_LOOP = "    for lang, floor in sorted(REDTEAM_FLOORS.items()):\n"
EXPIRE_FILTER = ("                     if s.decision == PENDING\n"
                 "                     and float(now) - s.offered_at "
                 ">= self.deadline_s]:\n")
OFFER_TAIL = ("        self._open[key] = _Span(uid=str(uid), index=int(index), "
              "text=str(text),\n                                offered_at="
              "float(now))\n")


@pytest.mark.mutation
def test_control_an_unmutated_copy_behaves_exactly_as_the_real_module(
        tmp_path: Path, frozen) -> None:
    """§6.0 rule 2's other half: a harness reporting success on a broken control."""
    module = load_mutant(tmp_path, "control")
    decisions = {line.id: og.BLOCKED if line.lang == "zh" else og.ALLOWED
                 for line in frozen}
    mine = module.escape_report(frozen, decisions)
    real = og.escape_report(frozen, decisions)
    assert mine.by_category == real.by_category
    assert mine.by_language == real.by_language
    assert mine.findings == real.findings


@pytest.mark.mutation
def test_mutant_a_guard_that_only_records_never_asks_for_silence(
        tmp_path: Path) -> None:
    """T-G4's 「讓守衛只記 log 不發 interrupt」 and T-G1's own mutant, the same
    change: the audit log looks identical and the robot keeps talking."""
    module = load_mutant(tmp_path, "log_only",
                         (INTERRUPT_BRANCH, "        if False:\n"))
    g = module.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    g.audio_started(UID, 0, now=T0 + 0.1)
    actions = g.verdict(UID, 0, blocked=True, now=T0 + 0.4)
    assert [a.kind for a in actions] == [module.AUDIT], "the mutant still interrupted"
    assert kinds(_real_partial_block()) == [og.INTERRUPT, og.AUDIT]


def _real_partial_block():
    g = og.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    g.audio_started(UID, 0, now=T0 + 0.1)
    return g.verdict(UID, 0, blocked=True, now=T0 + 0.4)


@pytest.mark.mutation
def test_mutant_treating_a_finished_utterance_as_cuttable_claims_a_block_that_never_happened(
        tmp_path: Path) -> None:
    """The report defect, not a crash: the audit row says `blocked`, the interrupt goes
    out after the audio is over, and the escape count reads zero."""
    module = load_mutant(
        tmp_path, "fully_audible_is_blocked",
        (CUTTABLE, "        cuttable = audibility != NOT_SPOKEN\n"))
    g = module.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    g.audio_started(UID, 0, now=T0 + 0.05)
    g.audio_done(UID, 0, now=T0 + 0.30)
    actions = g.verdict(UID, 0, blocked=True, now=T0 + 0.45)
    assert module.INTERRUPT in [a.kind for a in actions]
    audit = [a for a in actions if a.kind == module.AUDIT][0]
    assert audit.record["decision"] == "blocked", (
        "the mutant is supposed to claim a block; if it does not, this test is "
        "measuring something easier than it claims")
    # The real module calls the same sequence `recorded` and emits no interrupt.
    real = only_audit(_finished_then_blocked(og))
    assert real.record["decision"] == "recorded"
    assert real.record["audibility"] == og.FULLY_AUDIBLE
    # Worth stating, because it is why this mutant is a report defect rather than a
    # silent one: the finding survives the mutation (findings() keys off the
    # audibility tier, not off `cuttable`), so a reader of the JSON can still see
    # what happened. The number is wrong; the evidence is not gone.
    assert any("heard the whole span" in f for f in g.findings())


def _finished_then_blocked(module):
    g = module.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    g.audio_started(UID, 0, now=T0 + 0.05)
    g.audio_done(UID, 0, now=T0 + 0.30)
    return g.verdict(UID, 0, blocked=True, now=T0 + 0.45)


@pytest.mark.mutation
def test_mutant_a_dropped_span_read_as_silent_credits_the_guard_with_a_save(
        tmp_path: Path) -> None:
    """One interrupt turning into a report that claims five. The speaker abandoned
    spans 1..4 on its own; only span 0 was prevented by anything."""
    module = load_mutant(tmp_path, "dropped_is_silent",
                         (DROPPED_CHECK, "        if False:\n            pass\n"))
    g = module.OutputGuard()
    g.offer(UID, 1, "一句話", now=T0)
    g.abandoned(UID, 1, now=T0 + 0.1)
    actions = g.verdict(UID, 1, blocked=True, now=T0 + 0.5)
    assert module.INTERRUPT in [a.kind for a in actions]
    assert g.outcomes()[0].audibility == module.SILENT
    # the real module calls the same sequence NOT_SPOKEN and emits no interrupt
    real = og.OutputGuard()
    real.offer(UID, 1, "一句話", now=T0)
    real.abandoned(UID, 1, now=T0 + 0.1)
    assert kinds(real.verdict(UID, 1, blocked=True, now=T0 + 0.5)) == [og.AUDIT]


@pytest.mark.mutation
def test_mutant_a_failed_classifier_read_as_an_allow_fails_open(
        tmp_path: Path) -> None:
    module = load_mutant(tmp_path, "error_is_allow",
                         (BLOCKED_NONE, "            span.decision = ALLOWED\n"))
    g = module.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    g.verdict(UID, 0, blocked=None, now=T0 + 0.2, error="ThrottlingException")
    assert g.outcomes()[0].decision == module.ALLOWED
    assert g.findings() == [], (
        "the mutant is supposed to report nothing; a finding here means the fail-open "
        "is already visible and this test is not measuring it")


@pytest.mark.mutation
def test_mutant_giving_an_unenforced_category_a_block_rate_pools_the_gap_away(
        tmp_path: Path, frozen) -> None:
    """The arithmetic is fine and the sentence is a lie: `identity_denial: 0%` reads
    as "the filter let them through" when no filter was ever pointed at them."""
    module = load_mutant(
        tmp_path, "pool_unenforced",
        (CAT_RATE, '        entry["block_rate"] = (blocked / len(rows) if rows '
                   'else None)\n'))
    decisions = {line.id: og.BLOCKED for line in frozen}
    mutant = module.escape_report(frozen, decisions)
    unenforced = [c for c in og.CATEGORIES
                  if og.ENFORCERS[c] == og.NO_ENFORCER_TODAY]
    assert unenforced
    for category in unenforced:
        assert mutant.by_category[category]["block_rate"] is not None
        assert og.escape_report(frozen, decisions).by_category[category][
            "block_rate"] is None


@pytest.mark.mutation
def test_mutant_computing_a_cantonese_block_rate_makes_the_claim_66_forbids(
        tmp_path: Path, frozen) -> None:
    module = load_mutant(
        tmp_path, "rate_for_yue",
        (LANG_RATE, '        entry["block_rate"] = (blocked / len(rows) if rows '
                    'else None)\n'))
    decisions = {line.id: og.BLOCKED for line in frozen}
    assert module.escape_report(frozen, decisions).by_language["yue"][
        "block_rate"] == pytest.approx(1.0)
    assert og.escape_report(frozen, decisions).by_language["yue"][
        "block_rate"] is None


@pytest.mark.mutation
def test_mutant_dropping_the_floors_lets_a_truncated_set_report_zero_escapes(
        tmp_path: Path) -> None:
    module = load_mutant(tmp_path, "no_floors",
                         (FLOOR_LOOP, "    for lang, floor in []:\n"))
    lines, _ = og.load_redteam()
    path = tmp_path / "rt.jsonl"
    path.write_text("\n".join(json.dumps(
        {"id": line.id, "lang": line.lang, "category": line.category,
         "text": line.text}, ensure_ascii=False) for line in lines[:3]) + "\n",
        encoding="utf-8")
    assert module.load_redteam(path)[1] == []
    assert any("only grows" in f for f in og.load_redteam(path)[1])


@pytest.mark.mutation
def test_mutant_an_expire_that_never_fires_leaves_an_unchecked_span_looking_clean(
        tmp_path: Path) -> None:
    module = load_mutant(tmp_path, "never_expires",
                         (EXPIRE_FILTER, "                     if False]:\n"))
    g = module.OutputGuard()
    g.offer(UID, 0, "一句話", now=T0)
    assert g.expire(now=T0 + 10_000.0) == ()
    assert g.outcomes() == ()
    real = og.OutputGuard()
    real.offer(UID, 0, "一句話", now=T0)
    assert real.expire(now=T0 + 10_000.0) != ()


@pytest.mark.mutation
def test_mutant_an_agent_that_publishes_the_reserved_id_is_caught_by_both_censuses(
        tmp_path: Path) -> None:
    """T-G3's mutant: 「讓 agent_node 能發 chant」. This is the escalation this stage
    actually found, written back as a change somebody could plausibly make — adding an
    id to the payload looks like an improvement to the logs."""
    src = AGENT_PY.read_text(encoding="utf-8")
    anchor = 'json.dumps({"text": spoken}, ensure_ascii=False)'
    assert src.count(anchor) == 1, "mutant anchor is not unique in agent_node.py"
    chant = sorted(voices.UNINTERRUPTIBLE_IDS)[0]
    mutated = src.replace(
        anchor,
        f'json.dumps({{"text": spoken, "id": "{chant}"}}, ensure_ascii=False)')
    path = tmp_path / "agent_node.py"
    path.write_text(mutated, encoding="utf-8")
    sources = _sources([path])

    assert _reserved_id_sites(sources), "the id census missed the mutated payload"
    assert _reserved_token_sites(sources), "the token census missed it either"
    # And the third, independent assertion — the one that does not need to know the
    # reserved value at all, only that the agent must not choose an id.
    with pytest.raises(AssertionError, match="must not choose"):
        _assert_no_id_in_say_payload(mutated)
    _assert_no_id_in_say_payload(src)


def _assert_no_id_in_say_payload(src: str) -> None:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "text" in keys:
                assert "id" not in keys, (
                    f"line {node.lineno} builds a /speech/say payload with an id, "
                    "which is the one field it must not choose")


@pytest.mark.mutation
def test_mutant_tts_node_recomputing_the_fallback_id_reopens_the_escalation(
        tmp_path: Path) -> None:
    """The production defect itself, as a mutant. One line back the way it was, and a
    model reply of exactly `chant` is again an utterance nothing can stop."""
    src = TTS_PY.read_text(encoding="utf-8")
    anchor = "        uid = voices.request_id(req)\n"
    assert src.count(anchor) == 1
    mutated = src.replace(anchor, '        uid = req.get("id", text[:24])\n')
    path = tmp_path / "tts_node.py"
    path.write_text(mutated, encoding="utf-8")

    offenders = [s for s in _defaulted_id_sites(_sources([path]))]
    assert offenders, "the defaulted-id census missed the reverted line"
    assert not _defaulted_id_sites(_sources([TTS_PY]))
    # And the behaviour the census is standing in for, run directly on the two
    # expressions rather than on the node (tts_node imports rclpy).
    chant = sorted(voices.UNINTERRUPTIBLE_IDS)[0]
    req = {"text": chant}
    assert req.get("id", req["text"][:24]) == chant, "the old rule let text become id"
    assert voices.request_id(req) != chant


@pytest.mark.mutation
def test_mutant_an_offer_that_returns_a_verdict_reopens_the_serial_guard(
        tmp_path: Path) -> None:
    """T-G2's mutant, 「改成串行前置」, at the smallest possible size: one `return`.
    That is all it takes for a caller to write `if guard.offer(...): return` and put
    p50 415.8 ms in front of the first sentence."""
    module = load_mutant(tmp_path, "serial_offer",
                         (OFFER_TAIL, OFFER_TAIL + "        return False\n"))
    g = module.OutputGuard()
    assert g.offer(UID, 0, "一句話", now=T0) is False
    fn = _function(tmp_path / "mut_serial_offer" / "safety" / "output_guard.py",
                   "offer", cls="OutputGuard")
    assert [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value], (
        "the mutant did not actually add a return value, so the AST assertion this "
        "kills is not being exercised")
    assert og.OutputGuard().offer(UID, 0, "一句話", now=T0) is None
