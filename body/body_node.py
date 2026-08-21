"""open-agent-robot body node — the virtual robot's body.

Two scenes:
  gesture  (default): Unitree G1 with hands, FIXED BASE, kinematic puppet mode
           (qpos set directly + mj_forward — deterministic, cannot fall).
  walk     (later phase): dynamic MuJoCo + unitree_rl_gym pre-trained policy.

Outputs an MJPEG stream (multipart/x-mixed-replace) on --port for the browser
(reached through the SSH tunnel), and — when ROS 2 is available — subscribes
/body/gesture (std_msgs/String) and publishes /joint_states.

Standalone smoke test (no ROS):
    python3 body_node.py --xml assets/unitree_g1/scene_with_hands.xml \
        --standalone --demo
"""
from __future__ import annotations
import argparse
import io
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

from gestures import GESTURES, GesturePlayer, chant_pump, resolve_joint_ids
from gestures import HAND_ROCK, HAND_PAPER, HAND_SCISSORS  # noqa: F401 (agent API)

JPEG_QUALITY = 80


def load_fixed_base(xml_path: str, width: int, height: int) -> mujoco.MjModel:
    """Kinematic mode never calls mj_step, so the pelvis freejoint's qpos stays
    at its initial value — the base is effectively fixed without model surgery.
    A wrapper XML raises the offscreen framebuffer to the render size (MuJoCo
    defaults to 640x480); it must live in the scene's directory so relative
    asset paths keep resolving."""
    import os
    d, base = os.path.split(os.path.abspath(xml_path))
    wrapper = os.path.join(d, "_render_wrapper.xml")
    with open(wrapper, "w") as f:
        f.write(f"""<mujoco>
  <include file="{base}"/>
  <visual><global offwidth="{width}" offheight="{height}"/></visual>
</mujoco>""")
    return mujoco.MjModel.from_xml_path(wrapper)


class MjpegStreamer:
    """Serves the latest rendered frame as an MJPEG stream on localhost."""

    def __init__(self, port: int):
        self._frame: bytes | None = None
        self._cond = threading.Condition()
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence per-request spam
                pass

            def do_GET(self):
                if self.path not in ("/", "/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with streamer._cond:
                            streamer._cond.wait(timeout=1.0)
                            frame = streamer._frame
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[body] MJPEG on http://127.0.0.1:{port}/stream")

    def publish(self, jpeg: bytes):
        with self._cond:
            self._frame = jpeg
            self._cond.notify_all()


class Body:
    def __init__(self, xml_path: str, width=1280, height=720, fps=20, port=8080):
        self.model = load_fixed_base(xml_path, width, height)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.player = GesturePlayer()
        self.qpos_addr = resolve_joint_ids(self.model)
        self.fps = fps
        self.streamer = MjpegStreamer(port)
        self._cam = self._make_camera()
        self._lock = threading.Lock()
        try:
            import cv2
            self._cv2 = cv2
        except ImportError as e:
            raise SystemExit("body_node needs opencv-python-headless for JPEG encode") from e

    def _make_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.9]
        cam.distance = 2.4
        cam.azimuth = 180  # face the robot
        cam.elevation = -10
        return cam

    # --- agent-facing API -------------------------------------------------
    def play_gesture(self, name: str) -> bool:
        g = GESTURES.get(name)
        if g is None:
            m = re.match(r"chant:(\d+):([\d.]+):(rock|paper|scissors)", name)
            if not m:
                return False
            hand = {"rock": HAND_ROCK, "paper": HAND_PAPER,
                    "scissors": HAND_SCISSORS}[m.group(3)]
            g = chant_pump(int(m.group(1)), float(m.group(2)), hand)
        with self._lock:
            self.player.play(g)
        return True

    # --- main loop --------------------------------------------------------
    def run(self, on_tick=None):
        dt = 1.0 / self.fps
        next_t = time.monotonic()
        while True:
            with self._lock:
                pose = self.player.step(dt)
            for logical, val in pose.items():
                addr = self.qpos_addr.get(logical)
                if addr is not None:
                    self.data.qpos[addr] = val
            mujoco.mj_forward(self.model, self.data)  # kinematic: no mj_step
            self.renderer.update_scene(self.data, camera=self._cam)
            rgb = self.renderer.render()
            ok, jpg = self._cv2.imencode(
                ".jpg", rgb[:, :, ::-1],
                [int(self._cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                self.streamer.publish(jpg.tobytes())
            if on_tick:
                on_tick(self)
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--standalone", action="store_true", help="run without ROS 2")
    ap.add_argument("--demo", action="store_true",
                    help="cycle rock/paper/scissors/wave forever")
    args = ap.parse_args()

    body = Body(args.xml, fps=args.fps, port=args.port)

    if args.demo:
        def demo_loop():
            seq = ["wave", "rock", "paper", "scissors", "chant:5:0.4:paper", "nod"]
            i = 0
            while True:
                if not body.player.busy:
                    time.sleep(0.8)
                    body.play_gesture(seq[i % len(seq)])
                    print(f"[demo] {seq[i % len(seq)]}")
                    i += 1
                time.sleep(0.1)
        threading.Thread(target=demo_loop, daemon=True).start()

    if args.standalone:
        body.run()
        return

    # --- ROS 2 mode ---
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("body")
    node.create_subscription(String, "/body/gesture",
                             lambda msg: body.play_gesture(msg.data), 10)
    js_pub = node.create_publisher(JointState, "/joint_states", 10)

    def on_tick(b: Body):
        rclpy.spin_once(node, timeout_sec=0)
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = list(b.qpos_addr.keys())
        msg.position = [float(b.data.qpos[a]) for a in b.qpos_addr.values()]
        js_pub.publish(msg)

    body.run(on_tick=on_tick)


if __name__ == "__main__":
    main()
