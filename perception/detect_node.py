"""detect_node — YOLO26 scene awareness (the robot's object vision).

Runs the Day-0 TensorRT engine (yolo26s.engine, 6.2 ms/frame on the T4) on the
camera stream at ~5 fps, aggregates detections over a rolling 1 s window, and
publishes a stable scene summary the agent injects into its <scene> prompt.

Outputs:
  /detections     String JSON  {"objects":[{"name","n","conf"}...], "t_us":…}
  /detections_msg vision_msgs/Detection2DArray  (class_id is a STRING in ROS 2)
  /game/events_out  throttled HUD line  👁 person×1 cup×2
"""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

ENGINE = os.path.expanduser("~/open-agent-robot/yolo26s.engine")
CONF = 0.4
DETECT_INTERVAL = 0.2      # ~5 fps
WINDOW_S = 1.0             # rolling aggregation window
HUD_INTERVAL = 2.0
MAX_CLASSES = 12


class DetectNode(Node):
    def __init__(self):
        super().__init__("detect")
        from ultralytics import YOLO
        self.get_logger().info(f"loading {ENGINE} …")
        self.model = YOLO(ENGINE, task="detect")
        self.names = self.model.names
        self._next_t = 0.0
        self._window: deque[tuple[float, list]] = deque()
        self._last_scene = ""
        self._last_hud = ("", 0.0)
        self.pub_scene = self.create_publisher(String, "/detections", 5)
        self.pub_msg = self.create_publisher(Detection2DArray, "/detections_msg", 5)
        self.pub_hud = self.create_publisher(String, "/game/events_out", 5)
        self.pub_ask = self.create_publisher(String, "/agent/ask", 5)
        self._class_last_seen: dict[str, float] = {}
        self._ambient = {"person", "book"}   # room fixtures: never "news"
        self.create_subscription(CompressedImage, "/robot/eye/image_raw/compressed",
                                 self.on_frame, 1)
        self.get_logger().info("detect node ready (YOLO26s TensorRT)")

    def on_frame(self, msg: CompressedImage):
        now = time.monotonic()
        if now < self._next_t:          # time-gate to ~5 fps, newest frame wins
            return
        self._next_t = now + DETECT_INTERVAL
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        res = self.model.predict(img, conf=CONF, verbose=False)[0]
        dets = []
        for b in res.boxes:
            dets.append((self.names[int(b.cls)], float(b.conf),
                         [float(x) for x in b.xywh[0]]))
        self._window.append((now, dets))
        while self._window and now - self._window[0][0] > WINDOW_S:
            self._window.popleft()
        self.publish_scene(msg, dets)

    def publish_scene(self, msg: CompressedImage, latest_dets):
        # aggregate the window: per class -> max simultaneous count + best conf
        counts: dict[str, int] = defaultdict(int)
        best: dict[str, float] = defaultdict(float)
        for _, dets in self._window:
            frame_counts: dict[str, int] = defaultdict(int)
            for name, conf, _ in dets:
                frame_counts[name] += 1
                best[name] = max(best[name], conf)
            for name, n in frame_counts.items():
                counts[name] = max(counts[name], n)
        objects = sorted(({"name": k, "n": counts[k], "conf": round(best[k], 2)}
                          for k in counts), key=lambda o: -o["conf"])[:MAX_CLASSES]
        scene = json.dumps({"objects": objects,
                            "t_us": time.time_ns() // 1000}, ensure_ascii=False)
        key = json.dumps([(o["name"], o["n"]) for o in objects])
        if key != self._last_scene:      # publish agent scene only on change
            self._last_scene = key
            self.pub_scene.publish(String(data=scene))

        # proactive vision: a class not seen for 60 s just entered the frame
        now_m = time.monotonic()
        for o in objects:
            name = o["name"]
            if (name not in self._ambient and o["conf"] >= 0.55
                    and now_m - self._class_last_seen.get(name, -1e9) > 60):
                self.pub_ask.publish(String(data=json.dumps(
                    {"kind": "vision_event", "new": name})))
            self._class_last_seen[name] = now_m

        # ROS-standard Detection2DArray from the LATEST frame (string class_id)
        arr = Detection2DArray()
        arr.header = msg.header
        for name, conf, (cx, cy, w, h) in latest_dets:
            d = Detection2D()
            d.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = conf
            d.results.append(hyp)
            d.bbox.center.position.x = cx
            d.bbox.center.position.y = cy
            d.bbox.size_x = w
            d.bbox.size_y = h
            arr.detections.append(d)
        self.pub_msg.publish(arr)

        # HUD line: update when the CLASS SET changes (count flicker like
        # book×2<->×3 shouldn't spam), at most every HUD_INTERVAL, 15 s keepalive
        hud = " ".join(f"{o['name']}×{o['n']}" for o in objects[:5]) or "(nothing)"
        class_set = ",".join(sorted(o["name"] for o in objects))
        now = time.monotonic()
        set_changed = class_set != getattr(self, "_last_class_set", None)
        if ((set_changed and now - self._last_hud[1] > HUD_INTERVAL)
                or now - self._last_hud[1] > 15):
            self._last_hud = (hud, now)
            self._last_class_set = class_set
            self.pub_hud.publish(String(data=json.dumps(
                {"hud": f"👁 {hud}"}, ensure_ascii=False)))


def main():
    rclpy.init()
    rclpy.spin(DetectNode())


if __name__ == "__main__":
    main()
