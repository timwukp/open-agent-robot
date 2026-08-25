"""T-E1 / T-E3 — the Art 50(1) disclosure happens without the model, and is visible.

The plan states T-E1 as "with a Bedrock double returning an empty string, the
disclosure is still emitted and is the first utterance after a cold start". Both
halves are strengthened here, because both can be proven more cheaply than the
wording suggests:

  * *"without the model"* is asserted as **no dependency at all**, not as one double
    behaving badly. `safety/disclosure.py` is imported in a subprocess with `boto3`,
    `botocore` and `rclpy` blocked at the import hook, and the latch still yields its
    utterances. An empty-string double proves the code survives one reply; a blocked
    import proves the reply is not on the path.
  * *"the first utterance"* is asserted as **statement order in the AST**, not as
    timing. `gw.disclose()` sits in `handler` before the `async for data in ws`
    loop, so no microphone frame can be published ahead of it — a property of where
    the call is, which stays true on a slow machine and under any scheduling. A
    latency test could only have shown it was true once.

T-E3's split follows §6.3, which puts the browser check at L5: what CI can assert is
*structure* — the badge is static markup rather than event-driven, and it declares a
colour that differs from what is behind it. Whether it actually rendered is a
computed-style question that needs a real browser, and that walk-through is recorded
in docs/evidence/ rather than pretended at here. A class token existing in a
stylesheet is not evidence that a colour appeared on screen.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from safety import disclosure, guard

REPO_ROOT = Path(__file__).resolve().parent.parent
GATEWAY = REPO_ROOT / "gateway" / "ws_gateway.py"
TTS_NODE = REPO_ROOT / "speech" / "tts_node.py"
INDEX_HTML = REPO_ROOT / "web" / "robot-view" / "index.html"
CONSTITUTION = REPO_ROOT / "safety" / "constitution.md"

# --------------------------------------------------------------------- the text itself


def test_the_chinese_disclosure_states_both_required_facts() -> None:
    """Art 50(1) (an AI system) and Art 50(2) (the voice is generated).

    A keyword oracle, and deliberately a weak one about *wording* — it cannot judge
    whether a sentence is clear to a child. What it can do is fail if a rewrite drops
    one of the two facts the article requires, which is the failure that matters.
    """
    for token in ("AI", "不是真人", "合成"):
        assert token in disclosure.DISCLOSURE_ZH, token


def test_the_english_disclosure_states_both_required_facts() -> None:
    low = disclosure.DISCLOSURE_EN.lower()
    for token in ("ai system", "not a human", "synthetic"):
        assert token in low, token


def test_both_sentences_are_short_enough_to_not_be_talked_over() -> None:
    """Spoken before the person has asked for anything; a paragraph gets ignored."""
    assert len(disclosure.DISCLOSURE_ZH) <= 40
    assert len(disclosure.DISCLOSURE_EN) <= 90


# ------------------------------------------------------- routing: no hidden dependency


def test_neither_sentence_routes_to_the_out_of_process_engine() -> None:
    """The disclosure must not depend on another service having started.

    A Cantonese marker character anywhere in the Chinese sentence would route it to
    the CosyVoice microservice, which is a separate venv on a separate port that is
    not up at boot — and the failure would be silent: the queue entry raises inside
    the TTS worker and the person is simply never told anything.
    """
    from speech import voices
    for text in (disclosure.DISCLOSURE_ZH, disclosure.DISCLOSURE_EN):
        engine = voices.engine_for(text)
        assert engine.in_process, f"{engine.id} is reached over HTTP: {text!r}"


def test_each_sentence_reaches_the_voice_for_its_own_language() -> None:
    """Why there are two utterances instead of one bilingual string: routing is
    per-text, so a mixed sentence would be read end-to-end by one voice."""
    from speech import voices
    assert voices.engine_for(disclosure.DISCLOSURE_ZH).id == "kokoro-82m/zh"
    assert voices.engine_for(disclosure.DISCLOSURE_EN).id == "kokoro-82m/en"


def test_the_chinese_sentence_is_spoken_first() -> None:
    """A stated choice, not dict order: it is the user's language and the HUD's."""
    assert [u["text"] for u in disclosure.UTTERANCES] == [
        disclosure.DISCLOSURE_ZH, disclosure.DISCLOSURE_EN]


# ------------------------------------------------- the request shape tts_node accepts


def _request_keys_tts_node_reads() -> set[str]:
    """Derive the accepted `/speech/say` keys from tts_node's own source.

    Written-down interface lists go stale in the direction that matters: a key the
    node stopped reading still looks supported. This reads every `req["k"]` and
    `req.get("k")` in the node instead.
    """
    tree = ast.parse(TTS_NODE.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "req" and isinstance(n.slice, ast.Constant)):
            keys.add(n.slice.value)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "req" and n.args
                and isinstance(n.args[0], ast.Constant)):
            keys.add(n.args[0].value)
    return keys


def test_the_derived_request_key_census_is_not_empty() -> None:
    """A scan that finds nothing would make the subset assertion below vacuous."""
    keys = _request_keys_tts_node_reads()
    assert "text" in keys, keys
    assert len(keys) >= 3, keys


def test_every_utterance_uses_only_keys_tts_node_reads() -> None:
    keys = _request_keys_tts_node_reads()
    for u in disclosure.UTTERANCES:
        assert set(u) <= keys, f"{sorted(set(u) - keys)} would be silently ignored"


def test_the_utterances_are_cached_and_uniquely_identified() -> None:
    assert len(disclosure.UTTERANCES) == len(disclosure.DISCLOSURE_IDS)
    for u in disclosure.UTTERANCES:
        assert u["cache"] is True, "synthesized once, replayed on every reconnect"
        assert u["id"] in disclosure.DISCLOSURE_IDS
        assert u["text"]


# ------------------------------------------------------------------------- the latch


def test_the_latch_speaks_once_and_then_stays_quiet() -> None:
    latch = disclosure.DisclosureLatch()
    assert latch.fired is False
    assert latch.utterances() == disclosure.UTTERANCES
    assert latch.fired is True
    assert latch.utterances() == ()
    assert latch.utterances() == ()


def test_over_many_calls_exactly_one_returns_utterances() -> None:
    latch = disclosure.DisclosureLatch()
    spoke = [bool(latch.utterances()) for _ in range(50)]
    assert spoke.count(True) == 1
    assert spoke[0] is True, "and it is the first call, not an arbitrary one"


def test_a_new_process_discloses_again() -> None:
    """Latched per process, not per connection: a reconnect is the same person in
    the same room, but a restarted process cannot know who is in front of it."""
    first, second = disclosure.DisclosureLatch(), disclosure.DisclosureLatch()
    assert first.utterances()
    assert second.utterances(), "a second process must disclose to whoever is there"


def test_the_module_holds_no_process_global_latch() -> None:
    """A module-level latch would make the two objects above share state, and the
    test right above would pass for the wrong reason on the first run only."""
    assert not any(isinstance(v, disclosure.DisclosureLatch)
                   for v in vars(disclosure).values())


# ------------------------------------------------- no LLM anywhere on the disclosure path


BLOCKED_IMPORT_PROBE = """
import sys

BLOCKED = {"boto3", "botocore", "rclpy", "websockets"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked by the probe: " + name)
        return None


sys.meta_path.insert(0, Blocker())
from safety import disclosure

latch = disclosure.DisclosureLatch()
first = latch.utterances()
print(len(first), len(latch.utterances()), latch.fired,
      first[0]["id"], sep="|")
"""


def test_the_disclosure_works_with_boto3_and_rclpy_unimportable() -> None:
    """Stronger than a lying Bedrock double: the model is not on this path at all.

    A double that returns an empty string only shows the code survives one bad
    reply. Blocking the import shows there is no reply to survive — which is the
    property Art 50(1) needs when Bedrock is throttled, misconfigured or down.
    """
    proc = subprocess.run([sys.executable, "-c", BLOCKED_IMPORT_PROBE],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2|0|True|disclosure_zh", proc.stdout


def test_the_probe_would_notice_a_blocked_import(monkeypatch) -> None:
    """The probe above is worthless until it is shown to actually block something.

    Without this, a typo in the blocker would make the clean result meaningless —
    a probe that never reached the mechanism reporting success.
    """
    probe = BLOCKED_IMPORT_PROBE.replace(
        "from safety import disclosure",
        "import boto3\nfrom safety import disclosure")
    proc = subprocess.run([sys.executable, "-c", probe],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "blocked by the probe: boto3" in proc.stderr


def test_the_module_imports_nothing_that_could_reach_a_network() -> None:
    tree = ast.parse((REPO_ROOT / "safety" / "disclosure.py").read_text(encoding="utf-8"))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert imported <= {"__future__"}, imported


# ------------------------------------------------ the gateway fires it, before the loop


def _dotted(node: ast.AST) -> str:
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


def assert_disclosure_precedes_the_frame_loop(src: str) -> None:
    """`handler` must call `disclose()` before the `async for` over WebSocket frames.

    Statement *order* rather than a substring search: the substring "disclose"
    appears in this file's own comments and in the module name, so a text scan would
    be satisfied by a mention. Order is the property — the loop is where microphone
    frames start reaching ROS, so a call before it cannot be beaten by one.
    """
    handler = _func(ast.parse(src), "handler")
    disclose_at = loop_at = None
    for i, stmt in enumerate(handler.body):
        if disclose_at is None and any(n.endswith(".disclose") for n in _call_names(stmt)):
            disclose_at = i
        if loop_at is None and any(isinstance(n, ast.AsyncFor) for n in ast.walk(stmt)):
            loop_at = i
    if loop_at is None:
        raise AssertionError("no `async for` frame loop found in handler — this check "
                             "no longer knows where the interaction begins")
    if disclose_at is None:
        raise AssertionError("handler never calls disclose(): nothing tells the person "
                             "they are talking to an AI system")
    if disclose_at >= loop_at:
        raise AssertionError(
            f"disclose() is at statement {disclose_at} and the frame loop at {loop_at}: "
            "the disclosure must be queued before any microphone frame is published")


def test_the_gateway_discloses_before_the_first_frame() -> None:
    assert_disclosure_precedes_the_frame_loop(GATEWAY.read_text(encoding="utf-8"))


def test_there_is_exactly_one_disclosure_trigger() -> None:
    """Two triggers would each be individually latched-safe and jointly unreviewable."""
    calls = _call_names(ast.parse(GATEWAY.read_text(encoding="utf-8")))
    assert [c for c in calls if c.endswith(".disclose")] == ["gw.disclose"]


def test_disclose_publishes_the_utterances_to_speech_say() -> None:
    tree = ast.parse(GATEWAY.read_text(encoding="utf-8"))
    pubs = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _dotted(n.func) == "self.create_publisher"
            and any(isinstance(a, ast.Constant) and a.value == "/speech/say" for a in n.args)]
    assert len(pubs) == 1, "the gateway needs exactly one /speech/say publisher"
    fn = _func(tree, "disclose")
    assert "self.pub_say.publish" in _call_names(fn)
    assert "self.latch.utterances" in _call_names(fn), (
        "disclose() must go through the latch, or a reconnect re-announces")


def test_the_gateway_does_not_carry_its_own_copy_of_the_text() -> None:
    src = GATEWAY.read_text(encoding="utf-8")
    assert disclosure.DISCLOSURE_ZH not in src
    assert "AI 系統" not in src, "the text has one home: safety/disclosure.py"
    assert "from safety import disclosure" in src


def test_the_disclosure_reaches_both_faces() -> None:
    fn = _func(ast.parse(GATEWAY.read_text(encoding="utf-8")), "disclose")
    assert "self._fanout" in _call_names(fn), (
        "the daemon terminal and the browser both have to record it")
    assert "disclosure.browser_event" in _call_names(fn)


def test_the_browser_event_is_logged_by_the_c_daemon_unchanged() -> None:
    """daemon/main.c prints any event that is not a hud-only refresh, so a payload
    with a `text` key lands in the Mac-side log with no change to the C source —
    which is what keeps this out of T-J5's rebuild-staleness territory."""
    evt = disclosure.browser_event()
    assert evt["text"] and evt["hud"]
    assert evt["disclosure"]["art"] == "50(1)"
    src = (REPO_ROOT / "daemon" / "main.c").read_text(encoding="utf-8", errors="replace")
    assert '"{\\"hud\\""' in src, (
        "the daemon's hud-only exemption is what this relies on; if that check "
        "changed shape, re-derive whether a text event is still printed")


# ----------------------------------------------------- the constitution agrees with the code


def test_the_clause_note_points_at_the_code_that_enforces_it() -> None:
    assert "safety/disclosure.py" in guard.REQUIRED_CLAUSES["C1"]
    assert "S2.5" not in guard.REQUIRED_CLAUSES["C1"], (
        "the note still promises the deterministic path as future work")


def test_the_constitution_tells_the_model_the_system_handles_the_disclosure() -> None:
    """The clause the model reads must match what the code does, or C1 becomes an
    instruction to duplicate the disclosure on every turn."""
    text = CONSTITUTION.read_text(encoding="utf-8")
    assert "確定性路徑" in text


def test_the_audit_vocabulary_has_an_action_for_it() -> None:
    assert "disclosure_sent" in guard.AUDIT_ACTIONS


# ------------------------------------------------------- T-E3, the half CI can assert


def _css_rule(html: str, selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", html)
    if m is None:
        raise AssertionError(
            f"no CSS rule for {selector} — a missing rule is a failure, not a pass: "
            "the badge would render in the page's own colours")
    return m.group(1)


def _rgba(value: str) -> tuple[int, int, int, int]:
    """Normalise a 3/4/6/8-digit hex colour to (r, g, b, a).

    String comparison is not enough: `#fff` and `#ffffff` are the same colour, and —
    the case that actually bit — `#10141800` is *fully transparent*, so it compares
    unequal to the page background while rendering as exactly the page background.
    That hole was found in the browser (docs/evidence/t-e3-browser-disclosure.md),
    not here, which is why the normaliser exists at all.
    """
    h = value.lstrip("#").lower()
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        raise AssertionError(f"unparseable colour {value!r}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]


def _declared(rule: str, prop: str, selector: str) -> tuple[int, int, int, int]:
    m = re.search(prop + r"\s*:\s*(#[0-9a-fA-F]{3,8})", rule)
    if m is None:
        raise AssertionError(f"{selector} declares no explicit {prop}")
    return _rgba(m.group(1))


def assert_badge_is_visible(html: str) -> None:
    """The structural half of T-E3: static markup, explicit contrasting colours."""
    head, _, tail = html.partition("<script")
    if 'id="disclosure"' not in head:
        raise AssertionError(
            "the disclosure badge is not static markup — an element created by the "
            "event handler is absent exactly when the gateway is down")
    if disclosure.DISCLOSURE_HUD not in head:
        raise AssertionError("the badge text is not the one in safety/disclosure.py")
    rule = _css_rule(html, "#disclosure")
    fg = _declared(rule, "color", "#disclosure")
    bg = _declared(rule, "background", "#disclosure")
    body_bg = _declared(_css_rule(html, "body"), "background", "body")
    if bg[3] == 0:
        raise AssertionError(
            "the badge background is fully transparent, so the page background shows "
            "through: a colour that is unequal as a string and identical as pixels")
    if fg[3] == 0:
        raise AssertionError("the badge text colour is fully transparent")
    if fg[:3] == bg[:3]:
        raise AssertionError(f"the badge text colour equals its own background ({fg})")
    if bg[:3] == body_bg[:3]:
        raise AssertionError(
            f"the badge background equals the page background ({bg}): visually there "
            "is no badge, only text")
    if fg[:3] == body_bg[:3]:
        raise AssertionError("the badge text colour is the page background colour")
    # An overlay can cover a correctly-coloured badge — #hud did, when a dead camera
    # stream collapsed #stage. Only a browser can measure that, so what CI holds is
    # the two declarations that fixed it; the hit test lives in the evidence file.
    if "z-index" not in _css_rule(html, "#disclosure, #voice"):
        raise AssertionError("the Art 50 rows declare no stacking order, so a "
                             "positioned overlay can be painted over them")
    if "min-height" not in _css_rule(html, "#stage"):
        raise AssertionError("#stage can collapse when the camera stream is down, "
                             "which puts #hud on top of the Art 50 rows")


def test_the_disclosure_badge_is_static_and_declares_contrasting_colours() -> None:
    assert_badge_is_visible(INDEX_HTML.read_text(encoding="utf-8"))


def test_the_synthetic_voice_indicator_is_present_before_any_event() -> None:
    """Art 50(2) on the face: the indicator has an idle state, so a person can see
    that it exists rather than discovering it only while the robot is talking."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    head, _, _ = html.partition("<script")
    assert 'id="voice"' in head
    assert "idle" in head
    _css_rule(html, "#voice")


def test_the_computed_colour_half_of_t_e3_is_recorded_not_asserted_here() -> None:
    """§6.3 puts the browser check at L5. Naming the evidence file keeps that an
    explicit hand-off instead of a gap: a class token in the stylesheet is not
    proof that a colour rendered, and only a real browser can settle it."""
    evidence = REPO_ROOT / "docs" / "evidence" / "t-e3-browser-disclosure.md"
    assert evidence.exists(), (
        "the browser walk-through evidence is missing; CI has only checked structure")
    body = evidence.read_text(encoding="utf-8")
    assert "computed" in body.lower()
    assert len(body) > 400, "an evidence stub is not evidence"
    # The write-up must name what to run to get the numbers again, and the screenshot
    # must actually be there: a prose claim about pixels with no pixels attached is
    # the thing this hand-off is supposed to replace.
    assert "stub_ws_server.py" in body, "the evidence does not say how to reproduce it"
    png = evidence.with_suffix(".png")
    assert png.name in body, "the evidence does not reference its own screenshot"
    assert png.exists() and png.stat().st_size > 20_000, (
        f"{png.name} is missing or too small to be a viewport screenshot")


# --------------------------------------------------------------------------- mutants


@pytest.mark.mutation
def test_mutant_the_disclosure_only_in_the_system_prompt_is_caught() -> None:
    """The plan's named mutant for T-E1: delete the deterministic trigger and leave
    the constitution clause as the only thing asking for a disclosure."""
    src = GATEWAY.read_text(encoding="utf-8")
    anchor = "        gw.disclose()\n"
    assert anchor in src, "mutant anchor no longer present"
    with pytest.raises(AssertionError, match="never calls disclose"):
        assert_disclosure_precedes_the_frame_loop(src.replace(anchor, "", 1))


@pytest.mark.mutation
def test_mutant_disclosing_inside_the_frame_loop_is_caught() -> None:
    """The subtler failure: the disclosure happens, but a microphone frame can be
    published first, so the person speaks to an undisclosed system."""
    src = GATEWAY.read_text(encoding="utf-8")
    moved = src.replace("        gw.disclose()\n", "", 1).replace(
        "                if isinstance(data, bytes):",
        "                gw.disclose()\n                if isinstance(data, bytes):", 1)
    assert "gw.disclose()" in moved, "the mutant lost the call entirely"
    with pytest.raises(AssertionError, match="must be queued before"):
        assert_disclosure_precedes_the_frame_loop(moved)


@pytest.mark.mutation
def test_mutant_a_badge_with_a_class_but_no_colour_is_caught() -> None:
    """"Only added a class, no style" — the shape T-E3 exists to catch."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    anchor = "background: #2b2113; color: #ffd479;"
    assert anchor in html, "mutant anchor no longer present"
    with pytest.raises(AssertionError, match="declares no explicit color"):
        assert_badge_is_visible(html.replace(anchor, "background: #2b2113;", 1))


@pytest.mark.mutation
def test_mutant_a_badge_painted_the_page_background_is_caught() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    with pytest.raises(AssertionError, match="badge background equals the page"):
        assert_badge_is_visible(html.replace("background: #2b2113;",
                                             "background: #101418;", 1))


@pytest.mark.mutation
def test_mutant_a_transparent_badge_background_is_caught() -> None:
    """The hole the browser found, now closed here.

    `#10141800` is the page background with alpha `00`. As a *string* it differs from
    every other colour in the stylesheet, so the string-comparison version of this
    checker called it visible; on screen it is the page, unchanged. The browser
    reported `rgba(0, 0, 0, 0)` for the same shape when the whole rule was deleted.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    with pytest.raises(AssertionError, match="fully transparent"):
        assert_badge_is_visible(html.replace("background: #2b2113;",
                                             "background: #10141800;", 1))


@pytest.mark.mutation
def test_mutant_dropping_the_stacking_order_is_caught() -> None:
    """`#hud` is absolutely positioned; without a stacking order it paints over the
    Art 50 rows, which is what the screenshot showed before the fix."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    anchor = "#disclosure, #voice { position: relative; z-index: 2; }"
    assert anchor in html, "mutant anchor no longer present"
    with pytest.raises(AssertionError, match="no stacking order"):
        assert_badge_is_visible(html.replace(
            anchor, "#disclosure, #voice { position: relative; }", 1))


@pytest.mark.mutation
def test_mutant_a_collapsible_stage_is_caught() -> None:
    """No `min-height` and a dead camera stream: the <img> shrinks to its alt text and
    drags `#hud` up onto the synthetic-voice line. Measured, not imagined."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    anchor = "#stage { position: relative; min-height: 148px; }"
    assert anchor in html, "mutant anchor no longer present"
    with pytest.raises(AssertionError, match="can collapse"):
        assert_badge_is_visible(html.replace(
            anchor, "#stage { position: relative; }", 1))


@pytest.mark.mutation
def test_mutant_a_javascript_created_badge_is_caught() -> None:
    """Moving the badge into the event handler: correct-looking, and invisible in
    the one situation that matters (no gateway, no events, no badge)."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    anchor = f'<div id="disclosure">{disclosure.DISCLOSURE_HUD}</div>'
    assert anchor in html, "mutant anchor no longer present"
    with pytest.raises(AssertionError, match="not static markup"):
        assert_badge_is_visible(html.replace(anchor, "", 1))


@pytest.mark.mutation
def test_control_the_unmutated_sources_pass_every_checker() -> None:
    """Without this, a stale anchor makes each mutant above pass for free."""
    assert_disclosure_precedes_the_frame_loop(GATEWAY.read_text(encoding="utf-8"))
    assert_badge_is_visible(INDEX_HTML.read_text(encoding="utf-8"))
