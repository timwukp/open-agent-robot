"""ws_gateway — bridges the MacBook daemon/browser (WebSocket) to ROS 2 topics.

Wire protocol (binary frames): [1B type][8B capture-ts µs, little-endian][payload]
  0x01 up   mic PCM      16 kHz mono s16, 20 ms chunks   -> /robot/ear
  0x02 up   camera JPEG  640x360 q~70                    -> /robot/eye/image_raw/compressed
  0x03 down TTS PCM      24 kHz mono s16, 20 ms chunks   <- /robot/mouth
  0x10 both JSON event                                   <-> /game/events(_out)
  0x11 both timesync: up [t_mac]; reply [t_mac][t_srv]

Endpoints (localhost only; reached through the SSH tunnel):
  /stream  — the C daemon (binary, bidirectional)
  /events  — browsers (JSON text, read-only fan-out of /game/events)
"""
from __future__ import annotations
import asyncio
import json
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String, UInt8MultiArray
from sensor_msgs.msg import CompressedImage

import websockets

T_MIC, T_JPEG, T_TTS, T_FLUSH, T_EVT, T_SYNC = 0x01, 0x02, 0x03, 0x04, 0x10, 0x11
HDR = struct.Struct("<BQ")  # type, timestamp µs


def now_us() -> int:
    return time.time_ns() // 1000


class Gateway(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__("ws_gateway")
        self.loop = loop
        self.daemon_ws = None          # the single C-daemon connection
        self.browser_ws: set = set()   # event-feed browsers
        self.pub_ear = self.create_publisher(UInt8MultiArray, "/robot/ear", 50)
        self.pub_eye = self.create_publisher(CompressedImage, "/robot/eye/image_raw/compressed", 5)
        self.pub_evt = self.create_publisher(String, "/game/events", 10)
        self.create_subscription(UInt8MultiArray, "/robot/mouth", self.on_mouth, 200)
        self.create_subscription(String, "/game/events_out", self.on_event_out, 10)
        self.pub_interrupt = self.create_publisher(Empty, "/speech/interrupt", 5)
        self.create_subscription(Empty, "/speech/interrupt", self.on_interrupt, 5)

    # ROS -> WS ------------------------------------------------------------
    def on_mouth(self, msg: UInt8MultiArray):
        self._send_daemon(HDR.pack(T_TTS, now_us()) + bytes(msg.data))

    def on_interrupt(self, _msg):
        # flush the Mac-side playback ring so speech stops within one callback
        self._send_daemon(HDR.pack(T_FLUSH, now_us()))

    def on_event_out(self, msg: String):
        self._send_daemon(HDR.pack(T_EVT, now_us()) + msg.data.encode())
        for ws in list(self.browser_ws):
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, msg.data), self.loop)

    def _send_daemon(self, frame: bytes):
        if self.daemon_ws is not None:
            asyncio.run_coroutine_threadsafe(
                self._safe_send(self.daemon_ws, frame), self.loop)

    @staticmethod
    async def _safe_send(ws, data):
        try:
            await ws.send(data)
        except websockets.ConnectionClosed:
            pass

    # WS -> ROS ------------------------------------------------------------
    def handle_frame(self, data: bytes):
        if len(data) < HDR.size:
            return None
        ftype, ts_us = HDR.unpack_from(data)
        payload = data[HDR.size:]
        if ftype == T_MIC:
            m = UInt8MultiArray()
            m.data = payload
            self.pub_ear.publish(m)
        elif ftype == T_JPEG:
            img = CompressedImage()
            img.header.stamp.sec = ts_us // 1_000_000
            img.header.stamp.nanosec = (ts_us % 1_000_000) * 1000
            img.format = "jpeg"
            img.data = payload
            self.pub_eye.publish(img)
        elif ftype == T_EVT:
            text = payload.decode(errors="replace")
            if '"barge"' in text:      # loud voice over playback -> interrupt
                self.pub_interrupt.publish(Empty())
            self.pub_evt.publish(String(data=text))
        elif ftype == T_SYNC:
            return HDR.pack(T_SYNC, ts_us) + struct.pack("<Q", now_us())
        return None


async def main():
    loop = asyncio.get_running_loop()
    rclpy.init()
    gw = Gateway(loop)
    threading.Thread(target=rclpy.spin, args=(gw,), daemon=True).start()

    async def handler(ws):
        path = ws.request.path
        if path == "/events":
            gw.browser_ws.add(ws)
            gw.get_logger().info("browser connected")
            try:
                state = "已連線" if gw.daemon_ws else "等待 daemon"
                await ws.send(json.dumps(
                    {"text": f"瀏覽器已連上 gateway({state})。說「我們玩石頭剪刀布」開始遊戲。",
                     "hud": "就緒 — 說「石頭剪刀布」開始"}, ensure_ascii=False))
                await ws.wait_closed()
            finally:
                gw.browser_ws.discard(ws)
            return
        # daemon connection (last one wins; game pauses on disconnect)
        gw.daemon_ws = ws
        gw.get_logger().info("daemon connected")
        gw.pub_evt.publish(String(data=json.dumps({"type": "daemon", "state": "up"})))
        for b in list(gw.browser_ws):
            asyncio.ensure_future(gw._safe_send(b, json.dumps(
                {"text": "daemon 已連線(眼耳嘴就緒)", "hud": "daemon 已連線"},
                ensure_ascii=False)))
        try:
            async for data in ws:
                if isinstance(data, bytes):
                    reply = gw.handle_frame(data)
                    if reply:
                        await ws.send(reply)
        except websockets.ConnectionClosed:
            pass
        finally:
            if gw.daemon_ws is ws:
                gw.daemon_ws = None
            gw.pub_evt.publish(String(data=json.dumps({"type": "daemon", "state": "down"})))
            gw.get_logger().info("daemon disconnected")

    async with websockets.serve(handler, "127.0.0.1", 8765,
                                ping_interval=2, ping_timeout=6, max_size=2**20):
        gw.get_logger().info("ws_gateway on ws://127.0.0.1:8765 (/stream, /events)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
