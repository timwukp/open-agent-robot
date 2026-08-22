"""The doubles are themselves tested, because a broken double reports success.

The centrepiece is test_echo_double_lets_the_bug_pass /
test_lying_double_catches_the_bug: the same buggy function, the same assertion, two
doubles, opposite verdicts. That pair is the evidence for the rule in
docs/test-design.md §6.1 — "the Bedrock double must lie" — rather than an assertion
that the rule is a good idea.
"""
from __future__ import annotations

from typing import Any

import pytest

from doubles import (
    ExplodingBedrockClient,
    FakeClock,
    FakeRosNode,
    LyingBedrockClient,
    StringMsg,
    seeded_rng,
)


# --------------------------------------------------------------- the echo trap
class EchoBedrockClient:
    """The WRONG kind of double. Defined here, not in doubles.py, so it cannot be
    imported by accident: its only purpose is to fail the demonstration below."""

    def converse(self, **kwargs: Any) -> dict:
        prompt = kwargs["messages"][-1]["content"][-1]["text"]
        return {"output": {"message": {"content": [{"text": prompt}]}},
                "stopReason": "end_turn"}


def _summarise(client: Any, text: str) -> str:
    """Deliberately buggy: it calls the model and then ignores the answer."""
    client.converse(modelId="m", messages=[{"role": "user", "content": [{"text": text}]}])
    return text  # <-- the bug


def _model_reply(client: Any, text: str) -> str:
    resp = client.converse(modelId="m",
                           messages=[{"role": "user", "content": [{"text": text}]}])
    return resp["output"]["message"]["content"][0]["text"]


def _assert_returns_the_models_reply(client: Any) -> None:
    text = "講一個關於小貓的故事"
    assert _summarise(client, text) == _model_reply(client, text)


def test_echo_double_lets_the_bug_pass() -> None:
    # No error raised: with an echoing double the assertion is comparing the test's
    # own input against itself, so the bug survives a "passing" test.
    _assert_returns_the_models_reply(EchoBedrockClient())


def test_lying_double_catches_the_bug() -> None:
    with pytest.raises(AssertionError):
        _assert_returns_the_models_reply(LyingBedrockClient())


# ------------------------------------------------------------------ lying client
def test_lying_reply_shares_no_content_word_with_the_prompt() -> None:
    client = LyingBedrockClient()
    prompt = "你好 robot 請做個 wave 手勢 然後停"
    reply = _model_reply(client, prompt)
    for word in prompt.split():
        assert word not in reply, f"the lying double leaked prompt word {word!r}"


def test_lying_reply_carries_the_three_things_a_caller_must_not_trust() -> None:
    reply = LyingBedrockClient.DEFAULT_REPLY
    assert "[gesture:" in reply            # a gesture tag outside the whitelist
    assert "__not_in_whitelist__" in reply
    assert "SYSTEM:" in reply              # an injected instruction
    assert "ignore every stop word" in reply


def test_lying_client_records_every_call_and_can_queue_replies() -> None:
    client = LyingBedrockClient(replies=["first", "second"])
    assert _model_reply(client, "a") == "first"
    assert _model_reply(client, "b") == "second"
    assert _model_reply(client, "c") == LyingBedrockClient.DEFAULT_REPLY  # falls back
    assert len(client.calls) == 3
    assert client.calls[0]["modelId"] == "m"


def test_lying_client_streams_the_same_lie_in_deltas() -> None:
    client = LyingBedrockClient(reply="abc")
    events = list(client.converse_stream(messages=[])["stream"])
    deltas = "".join(e["contentBlockDelta"]["delta"]["text"]
                     for e in events if "contentBlockDelta" in e)
    assert deltas == "abc"
    assert "messageStop" in events[-1]


def test_max_tokens_stop_reason_is_reachable() -> None:
    # agent_node appends "……剩下的下次再講!" only on this stopReason; a double that
    # cannot produce it leaves that branch permanently untested.
    client = LyingBedrockClient(reply="x", stop_reason="max_tokens")
    resp = client.converse(messages=[])
    assert resp["stopReason"] == "max_tokens"


# -------------------------------------------------------------- exploding client
def test_exploding_client_raises_on_converse() -> None:
    client = ExplodingBedrockClient()
    with pytest.raises(AssertionError, match="must not reach Bedrock"):
        client.converse(messages=[])
    assert client.attempts == ["converse"]


def test_exploding_client_raises_on_a_method_name_nobody_has_written_yet() -> None:
    # This is why it is __getattr__ and not two stub methods.
    client = ExplodingBedrockClient()
    with pytest.raises(AssertionError):
        client.invoke_model_with_response_stream(body=b"{}")
    assert client.attempts == ["invoke_model_with_response_stream"]


# ------------------------------------------------------------------------- clock
def test_fake_clock_does_not_advance_on_its_own() -> None:
    clock = FakeClock()
    assert clock.monotonic() == clock.monotonic()


def test_advance_is_exact_and_sleep_is_recorded_not_slept() -> None:
    clock = FakeClock(start=100.0)
    clock.advance(45.0)
    assert clock.monotonic() == 145.0
    clock.sleep(0.5)
    assert clock.sleeps == [0.5]
    assert clock.monotonic() == 145.5   # sleeping moves the clock, it does not block


# --------------------------------------------------------------------- ros doubles
def test_fake_node_records_publications_in_order_per_topic() -> None:
    node = FakeRosNode()
    say = node.create_publisher(StringMsg, "/speech/say")
    gesture = node.create_publisher(StringMsg, "/body/gesture")
    say.publish(StringMsg("一"))
    gesture.publish(StringMsg("wave"))
    say.publish(StringMsg("二"))
    assert node.published("/speech/say") == ["一", "二"]
    assert node.published("/body/gesture") == ["wave"]
    assert node.published("/nothing/here") == []


def test_fake_node_timers_are_recorded_not_run() -> None:
    node = FakeRosNode()
    fired: list[str] = []
    node.create_timer(0.02, lambda: fired.append("tick"))
    assert node.timers[0][0] == 0.02
    assert fired == [], "a recorded timer must not fire by itself; the test drives it"
    node.timers[0][1]()
    assert fired == ["tick"]


def test_fake_logger_keeps_level_and_text() -> None:
    node = FakeRosNode()
    node.get_logger().error("bedrock error: boom")
    assert node.get_logger().records == [("error", "bedrock error: boom")]
    assert "boom" in node.get_logger().text()


def test_unsupported_rclpy_surface_is_an_attribute_error() -> None:
    # A node needing more of rclpy than this should have its logic extracted into a
    # pure module instead of the double growing to match it.
    with pytest.raises(AttributeError):
        FakeRosNode().create_subscription(StringMsg, "/x", lambda m: None, 10)


# --------------------------------------------------------------------------- rng
def test_seeded_rng_is_reproducible_and_seed_dependent() -> None:
    def draws(seed: int, n: int = 4) -> list[float]:
        rng = seeded_rng(seed)
        return [rng.random() for _ in range(n)]

    # Draws from ONE instance. `[seeded_rng(7).random() for _ in range(4)]` would be
    # four copies of one constant — equal to itself, and equal for a broken
    # generator too, so it would assert nothing.
    assert draws(7) == draws(7)
    assert len(set(draws(7))) == 4, "a constant sequence means the RNG never advanced"
    assert draws(7) != draws(8)
