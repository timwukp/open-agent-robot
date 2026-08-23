"""safety/disclosure.py — the EU AI Act Art 50(1) disclosure, as data and one latch.

Art 50(1) applies from 2026-08-02 and asks that a person interacting with an AI
system be informed of that fact. Two design decisions here are the whole point of
the module, and both are about *who* says it:

**The disclosure is not the model's job.** A clause in the system prompt asking the
model to introduce itself is a request, not a boundary: it competes with every other
instruction, it is skipped when the first turn is a game invitation rather than a
greeting, and it disappears entirely if the Bedrock call fails — which is exactly
the moment a person is most likely to be talking to something they do not
understand. So the text lives here, the utterances are a frozen tuple, and the node
that observes the connection publishes them directly. `constitution.md` C1 still
tells the model never to deny being an AI, because that covers a different
question ("are you human?") than this one ("has anyone told me?").

**We do not lean on the "obvious to a reasonably well-informed person" exemption.**
Recital 132 measures obviousness against the *vulnerable* observer, and this robot
is a humanoid with a face, a voice and a camera that a child will walk up to. The
cost of saying it out loud is two cached sentences at boot.

Two utterances rather than one bilingual sentence, because `speech/voices.py`
routes per *text*: a single mixed sentence would be spoken end-to-end by one
engine, giving either English read by a Mandarin voice or the reverse. Which
language should come first is not knowable before the person speaks — the
deterministic path has no utterance to detect a language from, and Art 50(1) wants
the disclosure *before* the interaction, not after enough of it to classify. So both
are spoken, Chinese first (the user's language, and the language of the HUD), and
that ordering is a stated choice rather than an accident of dict order.

Zero ROS, zero boto3, zero network, no clock — it is a tuple, a predicate and a
one-bit latch, so the whole of test item T-E1 that does not need a real node runs
under pytest on a laptop.
"""
from __future__ import annotations

# What the person hears. The requirement is that they learn two things: that this
# is an AI system rather than a person, and (Art 50(2)) that the voice they are
# hearing is generated. Kept to one sentence each because it is spoken before the
# person has asked for anything, and a paragraph read aloud at boot is a thing
# people learn to talk over.
#
# The Chinese sentence deliberately contains no Cantonese marker character, or
# `voices.engine_for` would route it to the CosyVoice microservice — which is a
# separate process that may not be up yet at boot, and a disclosure that depends
# on another service starting is a disclosure that silently does not happen.
# `test_disclosure.py` asserts the routing of both sentences rather than trusting
# this comment.
DISCLOSURE_ZH = "你好,我是 AI 系統,不是真人,我說話的聲音是合成的。"
DISCLOSURE_EN = "Hello. I am an AI system, not a human, and this voice is synthetic."

# Shown on the browser face, where it is static rather than event-driven: an
# element that only appears when a message arrives is absent exactly when the
# gateway is down. See web/robot-view/index.html and test item T-E3.
DISCLOSURE_HUD = "AI 系統 · 合成語音 / AI system · synthetic voice"

# `cache` is set so the clips are synthesized once and replayed on later boots
# from the same process; `id` is what /speech/tts_started reports, so the audit
# trail can distinguish the disclosure from ordinary speech.
UTTERANCES = (
    {"id": "disclosure_zh", "text": DISCLOSURE_ZH, "cache": True},
    {"id": "disclosure_en", "text": DISCLOSURE_EN, "cache": True},
)

# The ids, for anything that needs to recognise a disclosure after the fact.
DISCLOSURE_IDS = frozenset(u["id"] for u in UTTERANCES)


def browser_event() -> dict:
    """The event a browser face renders when the disclosure is spoken.

    `text` (not only `hud`) on purpose: `daemon/main.c` prints events to the
    terminal *unless* they are hud-only refreshes, so putting the disclosure in
    `text` is what makes the C daemon record it as well — the plan's "daemon and
    HUD both log it" with no change to the C side and therefore no rebuild.
    """
    return {"text": f"{DISCLOSURE_ZH} / {DISCLOSURE_EN}", "hud": DISCLOSURE_HUD,
            "disclosure": {"art": "50(1)", "spoken": True,
                           "ids": sorted(DISCLOSURE_IDS)}}


class DisclosureLatch:
    """Fires once per process: the utterances to speak, then nothing.

    Once per *process* rather than once per connection, and that is the
    interesting half. A reconnect is the same person in the same room — the C
    daemon drops and reattaches over an SSH tunnel that is not always up — so
    firing on every `daemon up` would make the robot announce itself repeatedly
    to somebody who already knows, and people route around a robot that nags.
    Firing once per process means a restart re-discloses, which is the safe
    direction: a new process cannot know who is in front of it.

    No clock and no timeout. A latch that expires after N minutes would need a
    clock and would then have two behaviours to test; the cheap correct thing is
    that the process boundary *is* the session boundary here, because the process
    holds no state across restarts anyway.
    """

    __slots__ = ("fired",)

    def __init__(self) -> None:
        self.fired = False

    def utterances(self) -> tuple[dict, ...]:
        """The /speech/say requests to publish, or `()` if already disclosed.

        Idempotent in the sense that matters: calling it twice speaks once. The
        caller does not have to remember to check `fired`, because forgetting
        would be silent and this returns an empty tuple instead.
        """
        if self.fired:
            return ()
        self.fired = True
        return UTTERANCES

    def reset(self) -> None:
        """Only for tests, and named so that a production caller reads as wrong."""
        self.fired = False
