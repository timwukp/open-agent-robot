"""T-E2 — every synthetic utterance carries the Art 50(2) marking.

The plan asks for "enumerate every branch of `tts_node` (streaming, cached, chant,
Cantonese yue), branch coverage 100% asserted from a branch list". A written branch
list is the thing this suite has been burned by before: a list cannot notice a new
name, so it certifies coverage of the branches that existed when it was typed. So
coverage is **derived** here, in two halves that meet in the middle:

  * the *engine* half comes from `voices.ENGINES` — a census, with a sample text per
    engine id and an equality assertion between the two sets, so an engine added
    without a sample turns this file red instead of quietly going unmeasured;
  * the *publish-path* half comes from `tts_node.py`'s AST — there must be exactly
    one creator of a `/speech/tts_started` payload, and both sides of the
    `if req.get("cache")` branch must reach it. With one creator, "did we mark this
    path" stops being a question a future path can answer wrongly.

`chant` is not a branch of the marking at all, and that is worth stating rather than
listing: `uid == "chant"` only decides interruptibility, and the payload is built
from `uid` and `text` alone, so the chant travels the same creator as everything
else. The assertion that keeps it that way is that the payload builder's arguments
never mention `interruptible`.

`tts_node` is read as source, never imported: it pulls in numpy, rclpy and Kokoro.
That is the same trade as test_game_wiring.py — structural assertions on the node,
behavioural assertions on the pure module it delegates to.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from speech import voices

REPO_ROOT = Path(__file__).resolve().parent.parent
TTS_NODE = REPO_ROOT / "speech" / "tts_node.py"
GATEWAY = REPO_ROOT / "gateway" / "ws_gateway.py"
INDEX_HTML = REPO_ROOT / "web" / "robot-view" / "index.html"

# One sample per engine. Compared against the census below, so this is a coverage
# obligation rather than a list of cases someone felt like writing.
SAMPLES = {
    "kokoro-82m/zh": "你好,今天天氣不錯。",
    "kokoro-82m/en": "Hello, the weather is fine today.",
    "cosyvoice-300m-sft/yue": "你好,今日天氣唔錯。",
}

# --------------------------------------------------------------- the engine census


def test_the_engine_census_is_non_empty_and_ids_are_unique() -> None:
    assert voices.ENGINES, "an empty census would make every coverage assertion vacuous"
    ids = [e.id for e in voices.ENGINES]
    assert len(ids) == len(set(ids)), ids
    assert set(voices.ENGINES_BY_ID) == set(ids)


def test_every_engine_has_a_sample_and_every_sample_names_a_real_engine() -> None:
    """The join that makes coverage derived instead of declared."""
    assert set(SAMPLES) == {e.id for e in voices.ENGINES}, (
        "a new engine needs a sample text here, or its marking is never exercised")


def test_engine_ids_are_a_stable_interface() -> None:
    """These strings land in logs, in the browser HUD and in the audit trail, so
    changing one rewrites the vocabulary of recorded history. Pinned deliberately."""
    assert {e.id for e in voices.ENGINES} == {
        "kokoro-82m/zh", "kokoro-82m/en", "cosyvoice-300m-sft/yue"}


def test_exactly_one_engine_lives_outside_this_process() -> None:
    """`in_process` is what tts_node branches on; the Cantonese engine is the only
    engine reached over HTTP, and anything else claiming that needs a code path."""
    out = [e for e in voices.ENGINES if not e.in_process]
    assert out == [voices.COSYVOICE_YUE]


def test_every_engine_declares_a_voice() -> None:
    for e in voices.ENGINES:
        assert e.voice, e.id
        assert (e.lang_code is not None) == e.in_process, (
            f"{e.id}: lang_code is Kokoro's KPipeline argument, so it should be "
            "present exactly for the in-process engines")


# ------------------------------------------------------------------------- routing


@pytest.mark.parametrize("engine_id,text", sorted(SAMPLES.items()))
def test_engine_for_routes_each_sample_to_its_engine(engine_id: str, text: str) -> None:
    assert voices.engine_for(text).id == engine_id


def test_the_cantonese_test_must_come_first_and_here_is_why() -> None:
    """A Cantonese sentence is also full of Han characters.

    Asserting that `HAS_CJK` matches the Cantonese sample is what makes the
    ordering in `engine_for` load-bearing rather than incidental: if the Mandarin
    test ran first it would match, and every Cantonese utterance would be read by
    the Mandarin voice — intelligible enough to survive a listening test.
    """
    yue = SAMPLES["cosyvoice-300m-sft/yue"]
    assert voices.HAS_CJK.search(yue), "the sample must trip the Mandarin test too"
    assert voices.HAS_YUE.search(yue)
    assert voices.engine_for(yue) is voices.COSYVOICE_YUE


def test_mixed_script_routes_by_the_presence_of_han_not_by_majority() -> None:
    assert voices.engine_for("OK,我知道了").id == "kokoro-82m/zh"
    assert voices.engine_for("Kokoro is 82M parameters").id == "kokoro-82m/en"


def test_empty_text_still_returns_an_engine_and_never_none() -> None:
    """Unreachable in production — `on_say` drops falsy text (asserted below) — but
    a router that can return None makes every caller carry a branch it cannot test."""
    assert voices.engine_for("") is voices.KOKORO_EN


def test_on_say_drops_empty_text_so_the_empty_route_is_unreachable() -> None:
    src = TTS_NODE.read_text(encoding="utf-8")
    assert 'if req.get("text"):' in src, (
        "the empty-text guard in on_say is what makes the empty routing case "
        "unreachable rather than merely untested")


# ---------------------------------------------------------------- the marking itself


@pytest.mark.parametrize("engine_id,text", sorted(SAMPLES.items()))
@pytest.mark.parametrize("dur_s", [0.0, 1.234])
def test_every_engine_on_both_duration_paths_is_marked(engine_id: str, text: str,
                                                       dur_s: float) -> None:
    """The full cross product: 3 engines × {streaming placeholder, exact cached}."""
    p = voices.started_payload(uid="u1", text=text, dur_s=dur_s, t_us=1_700_000_000_000)
    assert p[voices.SYNTHETIC_KEY] is True
    assert p["engine"] == engine_id
    assert p["voice"] == voices.ENGINES_BY_ID[engine_id].voice
    assert p["dur_s"] == dur_s
    assert p["id"] == "u1"
    assert p["t_us"] == 1_700_000_000_000


def test_the_payload_key_set_is_exactly_what_the_interface_declares() -> None:
    """`PAYLOAD_KEYS` is hand-written and this compares it to a real payload.

    Deriving `PAYLOAD_KEYS` from `started_payload` would make the comparison
    vacuous — it would agree with any change, including a dropped marking.
    """
    p = voices.started_payload(uid="u", text="hi", dur_s=0.0, t_us=1)
    assert set(p) == set(voices.PAYLOAD_KEYS)
    assert voices.SYNTHETIC_KEY in voices.PAYLOAD_KEYS


def test_the_marking_does_not_depend_on_the_utterance_id() -> None:
    """`chant` is exempt from *interruption*, never from marking."""
    for uid in ("chant", "disclosure_zh", "reply", ""):
        p = voices.started_payload(uid=uid, text="你好", dur_s=0.0, t_us=1)
        assert p[voices.SYNTHETIC_KEY] is True, uid


def test_the_reported_engine_is_the_one_that_will_speak(monkeypatch) -> None:
    """MUTANT-adjacent property, kept as a normal test because it is the whole point.

    The payload must not re-derive the engine with a second copy of the rule. Making
    `HAS_YUE` never match changes *both* the routing and the marking together — if
    the marking had its own copy of the rule, one of them would stay behind.
    """
    monkeypatch.setattr(voices, "HAS_YUE", re.compile(r"(?!x)x"))
    yue = SAMPLES["cosyvoice-300m-sft/yue"]
    assert voices.engine_for(yue) is voices.KOKORO_ZH, "the mutant is not in effect"
    p = voices.started_payload(uid="u", text=yue, dur_s=0.0, t_us=1)
    assert p["engine"] == "kokoro-82m/zh", (
        "the marking followed the routing, which is only possible if there is one rule")


# ------------------------------------------------- tts_node: one creator, both paths


def _dotted(node: ast.AST) -> str:
    """`self.pub_started.publish` for an Attribute chain, "" for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _call_names(node: ast.AST) -> list[str]:
    return [_dotted(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


def _func(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"no function named {name!r} — the anchor moved, so this "
                         "check is not measuring what it claims")


def assert_single_creator(src: str) -> None:
    """Exactly one `/speech/tts_started` publish, inside `_publish_started`.

    Raising rather than asserting inline so the mutant tests below can require this
    very function to go red — a checker no mutant is run against is a claim.

    Every violation is collected and reported together: a checker that stops at the
    first problem makes a mutant's *other* symptoms invisible, and then the mutant
    test that names one of them matches on whichever check happened to run first.
    """
    problems: list[str] = []
    tree = ast.parse(src)
    publishes = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and _dotted(n.func) == "self.pub_started.publish"]
    if len(publishes) != 1:
        problems.append(
            f"expected exactly 1 self.pub_started.publish call, found {len(publishes)}; "
            "two creators is how one path ends up unmarked")
    creator = _func(tree, "_publish_started")
    if publishes and not any(n is publishes[0] for n in ast.walk(creator)):
        problems.append("the publish call is not inside _publish_started")
    if "voices.started_payload" not in _call_names(creator):
        problems.append("_publish_started must build its payload via "
                        "voices.started_payload, not by hand")
    # The payload's own key strings must not appear anywhere in the node, or it can
    # assemble a payload of its own. "id" is exempt: it is also a /speech/say request
    # key that on_say legitimately reads.
    forbidden = set(voices.PAYLOAD_KEYS) - {"id"}
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    leaked = sorted(forbidden & literals)
    if leaked:
        problems.append(f"tts_node spells payload keys itself: {leaked}")
    if problems:
        raise AssertionError("; ".join(problems))


def assert_both_branches_marked(src: str) -> None:
    """Both sides of `if req.get("cache")` in `speak` reach the creator."""
    speak = _func(ast.parse(src), "speak")
    ifs = [n for n in ast.walk(speak)
           if isinstance(n, ast.If) and "cache" in ast.unparse(n.test)]
    if len(ifs) != 1:
        raise AssertionError(f"expected 1 cache branch in speak(), found {len(ifs)}")
    branch = ifs[0]
    problems: list[str] = []
    for side, body in (("cached", branch.body), ("streaming", branch.orelse)):
        if not body:
            problems.append(f"the {side} side of the cache branch is empty")
            continue
        calls = [c for stmt in body for c in _call_names(stmt)]
        if "self._publish_started" not in calls:
            problems.append(
                f"the {side} path does not publish a tts_started marking: {calls}")
    if problems:
        raise AssertionError("; ".join(problems))


def test_tts_node_has_exactly_one_marking_creator() -> None:
    assert_single_creator(TTS_NODE.read_text(encoding="utf-8"))


def test_both_the_cached_and_the_streaming_path_are_marked() -> None:
    assert_both_branches_marked(TTS_NODE.read_text(encoding="utf-8"))


def test_the_marking_call_does_not_consult_the_interrupt_state() -> None:
    """Why `chant` needs no branch of its own: the payload is a function of uid and
    text, so interruptibility cannot leak into whether something is marked."""
    src = TTS_NODE.read_text(encoding="utf-8")
    creator = _func(ast.parse(src), "_publish_started")
    names = {n.id for n in ast.walk(creator) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(creator) if isinstance(n, ast.Attribute)}
    assert "interruptible" not in names
    assert "_interrupt" not in attrs


def test_the_engine_pipelines_are_derived_from_the_census() -> None:
    """A voice declared in voices.py but not loaded here would be a silent 404."""
    src = TTS_NODE.read_text(encoding="utf-8")
    assert "voices.ENGINES" in src, (
        "tts_node must build its pipelines from the census, or the census becomes "
        "documentation instead of the source of truth")
    assert "zf_xiaoxiao" not in src and "af_heart" not in src, (
        "voice names belong to the census only")


# ------------------------------------------------- the gateway forwards the marking


def test_the_gateway_subscribes_to_tts_started_and_fans_it_out() -> None:
    tree = ast.parse(GATEWAY.read_text(encoding="utf-8"))
    subs = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _dotted(n.func) == "self.create_subscription"
            and any(isinstance(a, ast.Constant) and a.value == "/speech/tts_started"
                    for a in n.args)]
    assert len(subs) == 1, "expected exactly one /speech/tts_started subscription"
    handler = _func(tree, "on_tts_started")
    assert "self._fanout" in _call_names(handler), (
        "the marking must reach both the daemon terminal log and the browser face; "
        "_fanout is the single path that does both")
    fanout = _func(tree, "_fanout")
    assert "self._send_daemon" in _call_names(fanout)
    assert any(_dotted(n) == "self.browser_ws" for n in ast.walk(fanout)), (
        "the browser half of the fan-out is missing")


def test_the_gateway_does_not_rebuild_the_marking() -> None:
    literals = {n.value for n in ast.walk(ast.parse(GATEWAY.read_text(encoding="utf-8")))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert voices.SYNTHETIC_KEY not in literals and "engine" not in literals, (
        "the gateway forwards the payload; a second version of the marking here "
        "could disagree with the engine that actually spoke")


def test_the_event_type_the_gateway_sends_is_the_one_the_browser_checks() -> None:
    """A cross-file string agreement, because nothing else would notice a rename."""
    tree = ast.parse(GATEWAY.read_text(encoding="utf-8"))
    handler = _func(tree, "on_tts_started")
    sent = {n.value for n in ast.walk(handler)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    html = INDEX_HTML.read_text(encoding="utf-8")
    checked = set(re.findall(r"ev\.type\s*===\s*'([^']+)'", html))
    assert checked, "no ev.type check found in index.html — the pattern went stale"
    assert checked <= sent, (f"the browser checks {sorted(checked)} but the gateway "
                             f"sends {sorted(sent)}")


def test_the_browser_reads_the_marking_and_flags_its_absence() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert f"ev.{voices.SYNTHETIC_KEY}" in html
    assert "ev.engine" in html and "ev.voice" in html
    assert "NOT marked synthetic" in html, (
        "an unmarked utterance should be visibly wrong on the face, not silently "
        "rendered as if it were marked")


# --------------------------------------------------------------------------- mutants


MUTANT_INLINE_PAYLOAD = (
    'self._publish_started(uid, text, 0.0)',
    'self.pub_started.publish(String(data=json.dumps(\n'
    '                {"id": uid, "dur_s": 0.0, "t_us": time.time_ns() // 1000})))',
)
# The indentation is part of the anchor: deleting the call without its own leading
# whitespace would merge two lines into an IndentationError, and the mutant test
# would then "pass" on a syntax error instead of on the missing marking.
MUTANT_DROP_CACHED_MARKING = ('            self._publish_started(uid, text, dur)\n', '')


def _mutate(anchor: str, replacement: str) -> str:
    src = TTS_NODE.read_text(encoding="utf-8")
    assert anchor in src, f"mutant anchor no longer present: {anchor!r}"
    return src.replace(anchor, replacement, 1)


@pytest.mark.mutation
def test_mutant_a_second_inline_payload_breaks_the_single_creator() -> None:
    """The shape the code had before S2.5: each path building its own dict."""
    mutated = _mutate(*MUTANT_INLINE_PAYLOAD)
    with pytest.raises(AssertionError, match="exactly 1 self.pub_started.publish"):
        assert_single_creator(mutated)


@pytest.mark.mutation
def test_mutant_marking_only_the_streaming_path_is_caught() -> None:
    """The plan's named mutant for T-E2: "only the non-cached branch is marked"."""
    mutated = _mutate(*MUTANT_DROP_CACHED_MARKING)
    with pytest.raises(AssertionError, match="cached path does not publish"):
        assert_both_branches_marked(mutated)


@pytest.mark.mutation
def test_mutant_a_hand_built_payload_leaks_the_key_strings() -> None:
    """Even a mutant that keeps one publish site is caught, because building the
    payload locally means spelling its keys locally."""
    mutated = _mutate('voices.started_payload(\n            uid=uid, text=text, '
                      'dur_s=dur_s, t_us=time.time_ns() // 1000)',
                      '{"id": uid, "dur_s": dur_s, "t_us": time.time_ns() // 1000}')
    with pytest.raises(AssertionError, match="spells payload keys itself"):
        assert_single_creator(mutated)


@pytest.mark.mutation
def test_control_the_unmutated_node_passes_both_checkers() -> None:
    """Without this, a stale anchor would make every mutant above 'pass'."""
    src = TTS_NODE.read_text(encoding="utf-8")
    assert_single_creator(src)
    assert_both_branches_marked(src)
    assert src != _mutate(*MUTANT_INLINE_PAYLOAD), "the mutation was a no-op"
