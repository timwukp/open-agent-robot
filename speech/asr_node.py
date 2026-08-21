"""asr_node — streaming Mandarin+English ASR (FunASR paraformer-zh-streaming).

Subscribes /robot/ear (UInt8MultiArray, 16 kHz mono s16 PCM, 20 ms chunks).
Publishes  /speech/asr (String, JSON): {"partial": bool, "text": str, "t_us": int}

Half-duplex gate (plan D9): while /speech/tts_active is true (+200 ms tail),
incoming audio is discarded so the robot doesn't transcribe itself.

paraformer streaming operates on 600 ms chunks: chunk_size [0, 10, 5]
(10 * 60 ms = 600 ms) per the model card.
"""
from __future__ import annotations
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, UInt8MultiArray

CHUNK_SIZE = [0, 10, 5]            # 600 ms streaming config (model card)
CHUNK_SAMPLES = 960 * 10           # 600 ms @ 16 kHz
# Tail must outlast the Mac-side playback lag (150 ms prebuffer + pacing
# surplus), or the robot transcribes the end of its own chant and re-triggers
# the game (observed: a new game every ~30 s).
GATE_TAIL_S = 1.0


class AsrNode(Node):
    def __init__(self):
        super().__init__("asr")
        from funasr import AutoModel
        self.get_logger().info("loading paraformer-zh-streaming…")
        self.model = AutoModel(model="paraformer-zh-streaming", disable_update=True)
        self.cache: dict = {}
        self.buf = np.zeros(0, dtype=np.int16)
        self.gate_until = 0.0
        # paraformer streaming yields INCREMENTAL text per 600ms chunk; keyword
        # matching needs the accumulated utterance, finalized after ~1.2s silence
        self.utterance = ""
        self.silent_chunks = 0
        self.pub = self.create_publisher(String, "/speech/asr", 10)
        self.create_subscription(UInt8MultiArray, "/robot/ear", self.on_pcm, 50)
        self.create_subscription(Bool, "/speech/tts_active", self.on_gate, 10)
        self.get_logger().info("asr ready (600 ms chunks, half-duplex gated)")

    def on_gate(self, msg: Bool):
        if msg.data:
            # 60 s fuse, NOT infinity: if tts dies mid-utterance and never
            # publishes False, the robot must not stay deaf forever
            self.gate_until = time.monotonic() + 60.0
            # the robot is responding — the pending utterance has been consumed;
            # keeping it would re-finalize (and re-trigger) it after the gate
            self.utterance = ""
            self.silent_chunks = 0
        else:
            self.gate_until = time.monotonic() + GATE_TAIL_S

    def on_pcm(self, msg: UInt8MultiArray):
        if time.monotonic() < self.gate_until:
            self.buf = np.zeros(0, dtype=np.int16)  # also flush partial chunk
            self.cache = {}
            return
        self.buf = np.concatenate([self.buf, np.frombuffer(bytes(msg.data), dtype=np.int16)])
        while len(self.buf) >= CHUNK_SAMPLES:
            chunk, self.buf = self.buf[:CHUNK_SAMPLES], self.buf[CHUNK_SAMPLES:]
            res = self.model.generate(
                input=chunk.astype(np.float32) / 32768.0,
                cache=self.cache, is_final=False,
                chunk_size=CHUNK_SIZE,
                encoder_chunk_look_back=4, decoder_chunk_look_back=1)
            text = "".join(r.get("text", "") for r in res)
            if text:
                self.utterance += text
                self.silent_chunks = 0
                self.pub.publish(String(data=json.dumps(
                    {"partial": True, "text": self.utterance,
                     "t_us": time.time_ns() // 1000}, ensure_ascii=False)))
            elif self.utterance:
                self.silent_chunks += 1
                if self.silent_chunks >= 2:   # ~1.2 s of silence ends the utterance
                    self.pub.publish(String(data=json.dumps(
                        {"partial": False, "text": self.utterance,
                         "t_us": time.time_ns() // 1000}, ensure_ascii=False)))
                    self.get_logger().info(f"heard: {self.utterance}")
                    self.utterance = ""
                    self.silent_chunks = 0


def main():
    rclpy.init()
    rclpy.spin(AsrNode())


if __name__ == "__main__":
    main()
