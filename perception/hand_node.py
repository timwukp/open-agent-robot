"""hand_node — user hand-gesture recognition (MediaPipe Tasks HandLandmarker).

mediapipe >=0.10.30 wheels ship ONLY the Tasks API (legacy mp.solutions is gone),
so this uses vision.HandLandmarker in VIDEO mode. The .task model is fetched on
first run.

Subscribes /robot/eye/image_raw/compressed (JPEG frames from the MacBook camera).
Publishes  /hand_pose (String, JSON): {"gesture": "rock|paper|scissors|none",
                                       "fingers": int, "t_us": frame-capture µs}

Classification (fork design): count extended fingers — a finger is extended when
its tip is farther from the wrist than its PIP joint.
  0-1 -> rock, 2 -> scissors, >=4 -> paper (3 ambiguous -> none)
"""
from __future__ import annotations
import json
import os
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.expanduser("~/.cache/oar/hand_landmarker.task")
TIP = [4, 8, 12, 16, 20]
PIP = [2, 6, 10, 14, 18]


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandNode(Node):
    def __init__(self):
        super().__init__("hand")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts_ms = 0
        self._nframes = 0
        self._t_rate = None
        self.pub = self.create_publisher(String, "/hand_pose", 10)
        self.pub_hud = self.create_publisher(String, "/game/events_out", 5)
        self._last_hud = 0.0
        self.create_subscription(CompressedImage, "/robot/eye/image_raw/compressed",
                                 self.on_frame, 1)  # depth 1: newest frame wins
        self.get_logger().info("hand gesture node ready (MediaPipe Tasks)")

    @staticmethod
    def classify(landmarks) -> tuple[str, int]:
        """The THUMB is ignored: people naturally leave it sticking out when
        making scissors, which pushed the extended-count to 3 and misclassified
        every scissors as 'none'. Shape is decided by the four fingers only."""
        pts = np.array([(p.x, p.y) for p in landmarks])
        wrist = pts[0]
        ext = []
        for tip, pip in zip(TIP[1:], PIP[1:]):   # index, middle, ring, pinky
            ext.append(np.linalg.norm(pts[tip] - wrist)
                       > np.linalg.norm(pts[pip] - wrist) * 1.15)
        index_f, middle_f, ring_f, pinky_f = ext
        n = int(sum(ext))  # numpy bool sum -> int64 breaks json.dumps
        if index_f and middle_f and not ring_f and not pinky_f:
            return "scissors", n
        if n >= 3:
            return "paper", n
        if n == 0:
            return "rock", n
        return "none", n

    def on_frame(self, msg: CompressedImage):
        import time as _t
        t0 = _t.monotonic()
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        t1 = _t.monotonic()
        t_us = msg.header.stamp.sec * 1_000_000 + msg.header.stamp.nanosec // 1000
        ts_ms = t_us // 1000
        if ts_ms <= self._last_ts_ms:      # VIDEO mode needs monotonic timestamps
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(mp_img, ts_ms)
        t2 = _t.monotonic()
        if self._nframes % 20 == 1:
            self.get_logger().info(
                f"timing: decode {1000*(t1-t0):.0f}ms detect {1000*(t2-t1):.0f}ms")
        gesture, fingers = "none", 0
        if res.hand_landmarks:
            gesture, fingers = self.classify(res.hand_landmarks[0])
        self.pub.publish(String(data=json.dumps(
            {"gesture": gesture, "fingers": fingers, "t_us": t_us})))
        import time as _t
        # live what-the-robot-sees HUD: only on gesture CHANGE (min 0.5 s apart,
        # 10 s keepalive) — a fixed 0.5 s cadence flooded the event stream
        now = _t.monotonic()
        changed = gesture != getattr(self, "_last_gesture", None)
        if (changed and now - self._last_hud > 0.5) or now - self._last_hud > 10:
            self._last_hud = now
            self._last_gesture = gesture
            label = {"rock": "石頭✊", "paper": "布✋", "scissors": "剪刀✌️",
                     "none": "(沒看到手)"}[gesture]
            self.pub_hud.publish(String(data=json.dumps(
                {"hud": f"👁 {label}"}, ensure_ascii=False)))
        self._nframes += 1
        if self._t_rate is None:
            self._t_rate = _t.monotonic()
        elif self._nframes % 20 == 0:
            dt = _t.monotonic() - self._t_rate
            self.get_logger().info(f"hand rate {20/dt:.1f}/s, last={gesture}({fingers})")
            self._t_rate = _t.monotonic()


def main():
    rclpy.init()
    rclpy.spin(HandNode())


if __name__ == "__main__":
    main()
