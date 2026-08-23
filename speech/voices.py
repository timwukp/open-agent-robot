"""speech/voices.py — which engine speaks a text, and the Art 50(2) marking.

Two jobs that only look separate:

1. **Routing.** `engine_for(text)` decides Cantonese vs Mandarin vs English from
   the text alone. This used to live inside `tts_node._sentences` as two module
   regexes; it moved here so that the routing decision and the marking are made by
   the **same call**. Had the marking re-derived the engine from the text with its
   own copy of the rule, the two copies could disagree and the label would name an
   engine that did not produce the audio — a marking that is confidently wrong is
   worse than no marking, because a downstream consumer has no way to notice.

2. **Marking.** Art 50(2) asks that synthetic audio be marked in a machine-readable
   form. `started_payload()` is the **only** place a `/speech/tts_started` payload is
   built, which is what makes "every branch is marked" checkable: `tts_node` has a
   cached path and a streaming path, and with one creator the question stops being
   "did we remember on both paths" (a 2-case enumeration that a third path in six
   months silently escapes) and becomes "is there exactly one creator" (an assertion
   that a third path cannot escape). See test item T-E2.

**Honest boundary, stated here because this is the file that claims compliance:**
this is a *stream-level* label, not a watermark. It travels with the ROS message and
the WebSocket event; it does not survive a microphone recording of the speaker, and
nothing in this repo can make it survive one, because we never write the audio to
disk — the PCM goes to the Mac's playback ring and is gone. The proportionality
argument (Art 50(2) second sentence: technical feasibility, cost, state of the art)
is that for a robot that does not record, a stream label is the marking that exists
at the boundary where the content is still ours. **The moment a recording feature is
added, this becomes insufficient** and a real audio watermark is required; the
licensing of the candidate libraries has not been checked.

Zero ROS, zero numpy, zero network — regexes, a tuple and a dict builder, so the
whole census runs under pytest on a laptop.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# Any Han character routes to Mandarin; a Cantonese marker character wins over it.
# Kept here rather than in tts_node so the routing rule has one home.
HAS_CJK = re.compile(r"[一-鿿]")
# Cantonese marker characters -> route to the CosyVoice yue engine
HAS_YUE = re.compile(r"[係唔嘅喇咩乜嘢啲嗰噉咗冇睇諗俾攞邊度緊]")


@dataclass(frozen=True)
class Engine:
    """One speech engine, as it is identified to the person and to the log.

    `id` is the machine-readable marking required by Art 50(2) and is treated as
    stable: it appears in logs and in the browser HUD, so changing a string here
    changes recorded history's vocabulary. `lang_code` is Kokoro's `KPipeline`
    argument and is `None` for engines that are not Kokoro — `in_process` is what
    `tts_node` actually branches on, so a fourth engine declares how it is reached
    instead of `tts_node` growing another `if`.
    """
    id: str
    lang_code: str | None
    voice: str
    in_process: bool


# The census. `engine_for` must return a member of this tuple and nothing else,
# which is what lets a test enumerate every engine × every publish path without a
# hand-written list of branches that can fall behind the code.
KOKORO_ZH = Engine(id="kokoro-82m/zh", lang_code="z", voice="zf_xiaoxiao", in_process=True)
KOKORO_EN = Engine(id="kokoro-82m/en", lang_code="a", voice="af_heart", in_process=True)
COSYVOICE_YUE = Engine(id="cosyvoice-300m-sft/yue", lang_code=None,
                       voice="cosyvoice-sft-default", in_process=False)
ENGINES = (KOKORO_ZH, KOKORO_EN, COSYVOICE_YUE)
ENGINES_BY_ID = {e.id: e for e in ENGINES}

# The key the marking lives under, named once so `tts_node` never spells it.
SYNTHETIC_KEY = "synthetic"
# What a /speech/tts_started payload contains. Written by hand on purpose: a test
# compares it against a payload actually built by `started_payload`, and deriving
# it from that payload would make the comparison vacuous. Consumers are
# gateway/ws_gateway.py (forwards to the daemon and the browser) and any audit
# reader, so removing a key here is an interface change.
PAYLOAD_KEYS = frozenset({"id", "dur_s", "t_us", SYNTHETIC_KEY, "engine", "voice"})


def engine_for(text: str) -> Engine:
    """The engine that will speak `text`.

    Order matters and is the whole rule: a Cantonese sentence is also full of Han
    characters, so the Cantonese test must come first or every `yue` utterance
    would be read by the Mandarin voice — which is intelligible enough that the
    bug would survive listening tests and only show up as "why does it sound like
    a Mandarin speaker reading Cantonese words".
    """
    if HAS_YUE.search(text):
        return COSYVOICE_YUE
    return KOKORO_ZH if HAS_CJK.search(text) else KOKORO_EN


def started_payload(*, uid: str, text: str, dur_s: float, t_us: int) -> dict:
    """The one and only `/speech/tts_started` payload.

    `dur_s` is exact for cached clips and `0.0` while streaming (the length is not
    known until the last sentence is synthesized) — the marking does not depend on
    it, which is the point: a consumer can trust `synthetic` even on the path where
    the duration is a placeholder.

    Callers pass `text` rather than an engine so there is no way to report an
    engine other than the one that will speak. `tts_node` must not build this dict
    itself; `tests/test_voices.py` asserts that it does not, by checking that the
    payload's key strings do not appear in `tts_node.py` at all.
    """
    engine = engine_for(text)
    return {"id": uid, "dur_s": dur_s, "t_us": t_us,
            SYNTHETIC_KEY: True, "engine": engine.id, "voice": engine.voice}
