"""agent_node — the robot's System-2: Claude Haiku 4.5 via Bedrock Converse.

Never on the game's beat-critical path (plan D2). Handles:
  * /agent/ask {"kind":"banter", ...}   -> one-liner game commentary -> /speech/say
  * /agent/ask {"kind":"chat","text":…} -> conversational reply (scene-aware)
  * general ASR text that is NOT a game trigger -> chat (scene-aware)

Scene awareness: keeps the latest YOLO26 detections (/detections JSON from
perception) and injects them into the system context, so “你看到什麼?” works.

Skills: skills/loader.py decides — with a frozen phrase table, never by asking the
model — whether a skill's instructions belong in this turn. Only the tiny L1 index
(name + one sentence per skill) is resident in the system prompt; the body is
attached to the user turn as data, and the model pulls deeper files through the
read_skill_file tool. See skills/loader.py for why the gate is not an LLM judge.

The system prompt is **not** written here. `agent/context.py` owns it — the base
prompt, the constitution, and every fenced data block — and this node holds exactly
one line of the result. Until that move the prompt was `SYSTEM + l1_index(...)`,
which meant safety/guard.py's constitution and its instruction/data seam were
correct offline and absent from the running robot (T-C5). Keeping the assembly in
one module is what lets a test diff the instruction region byte-for-byte against a
baseline; a node that concatenated its own strings would have to be launched to be
checked, and this suite never launches rclpy.
"""
from __future__ import annotations
import json
import sys
import threading
from pathlib import Path

import boto3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# run_ec2.sh launches this file directly, so sys.path[0] is agent/, not the repo
# root — without this, `from skills import loader` fails only on the robot.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import context  # noqa: E402
from safety import guard  # noqa: E402
from skills import loader  # noqa: E402

# Sonnet 5 via the global cross-region profile (user-directed switch from
# Haiku 4.5 after runaway-game incident: stronger instruction following).
# Beat-critical paths never wait on the LLM regardless (plan D2).
MODEL_ID = "global.anthropic.claude-sonnet-5"
REGION = "ap-southeast-1"

GAME_TRIGGERS = ("石頭剪刀布", "石头剪刀布", "rock paper scissors", "剪刀石頭布")


class AgentNode(Node):
    def __init__(self):
        super().__init__("agent")
        from botocore.config import Config
        self.brt = boto3.client(
            "bedrock-runtime", region_name=REGION,
            config=Config(read_timeout=25, connect_timeout=5,
                          retries={"max_attempts": 2}))
        self.scene = "[]"
        # A malformed skill directory must not cost the robot its voice: log and
        # carry on with no skills, which is exactly the pre-S1 behaviour.
        try:
            self.skills = loader.discover()
        except loader.SkillError as exc:
            self.get_logger().error(f"skills disabled: {exc}")
            self.skills = ()
        # The constitution is NOT wrapped in a try. A malformed skill directory
        # costs the robot a capability, so it degrades; a constitution that will
        # not load means every boundary clause is absent, and a robot that starts
        # anyway is the one state safety/guard.py:load_constitution exists to
        # prevent. Failing here is loud, launch-time, and fixable in one file.
        self.constitution = guard.load_constitution()
        assembly = context.assemble(self.constitution,
                                    skills_index=loader.l1_index(self.skills))
        self.system_text = assembly.prompt
        for finding in assembly.findings:
            self.get_logger().warning(f"context: {finding}")
        self.history: list[dict] = []
        self.busy = threading.Lock()
        self.pub_say = self.create_publisher(String, "/speech/say", 10)
        self.pub_gesture = self.create_publisher(String, "/body/gesture", 10)
        self.pub_game = self.create_publisher(String, "/game/cmd", 10)
        self.create_subscription(String, "/agent/ask", self.on_ask, 10)
        self.create_subscription(String, "/speech/asr", self.on_asr, 10)
        self.create_subscription(String, "/detections", self.on_det, 5)
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import Bool, Empty
        self.latest_jpeg: bytes = b""
        self.create_subscription(CompressedImage, "/robot/eye/image_raw/compressed",
                                 lambda m: setattr(self, "latest_jpeg", bytes(m.data)), 1)
        # barge-in plumbing: know when the robot is talking / a game is running,
        # so a new user utterance can preempt speech (never the game chant)
        self.tts_active = False
        self.game_state = "idle"
        self.pending: str | None = None
        self._vision_cooldown = 0.0
        self.pub_interrupt = self.create_publisher(Empty, "/speech/interrupt", 5)
        self.create_subscription(Bool, "/speech/tts_active",
                                 lambda m: setattr(self, "tts_active", m.data), 10)
        self.create_subscription(String, "/game/events_out", self.on_game_event, 10)
        self._last_asr = ""
        self.get_logger().info(
            f"agent ready ({MODEL_ID}, {REGION}), skills: "
            f"{[s.name for s in self.skills] or 'none'}")

    def on_det(self, msg: String):
        self.scene = msg.data

    def on_game_event(self, msg: String):
        try:
            self.game_state = json.loads(msg.data).get("state", self.game_state)
        except ValueError:
            pass

    def on_asr(self, msg: String):
        d = json.loads(msg.data)
        text = d.get("text", "")
        # general conversation: every FINALIZED utterance goes to the LLM
        # (partials are noisy fragments; game triggers belong to game_node)
        if d.get("partial"):
            return
        t = text.lower()
        if ("剪刀" in t and ("石" in t or "布" in t)) or \
           ("rock" in t and ("paper" in t or "scissor" in t)):
            return  # game_node owns these
        if len(text) < 3 or text == self._last_asr:
            return
        self._last_asr = text
        # the user spoke a full sentence while the robot was talking -> preempt
        # (the game chant is sacred; game_node handles its own stop reflex)
        if self.tts_active and self.game_state == "idle":
            from std_msgs.msg import Empty
            self.pub_interrupt.publish(Empty())
        self.ask_llm(text)

    def on_ask(self, msg: String):
        req = json.loads(msg.data)
        if req.get("kind") == "vision_event":
            import time as _t
            if (self.tts_active or self.game_state != "idle"
                    or _t.monotonic() < self._vision_cooldown):
                return
            self._vision_cooldown = _t.monotonic() + 30.0
            self.ask_llm(f"(鏡頭裡剛出現了 {req.get('new','')} — "
                         f"看一眼圖像,自然地簡短反應一句,像朋友注意到新東西那樣)")
            return
        if req.get("kind") == "banter":
            prompt = (f"剛剛一局石頭剪刀布:使用者出 {req['user']},你出 {req['robot']},"
                      f"結果你{'贏' if req['outcome']=='win' else ('輸' if req['outcome']=='lose' else '平手')}。"
                      f"總比分 機器人{req['score']['robot']}:使用者{req['score']['user']}。"
                      f"用{'中文' if req.get('lang')=='zh' else 'English'}講一句俏皮吐槽。")
        else:
            prompt = req.get("text", "")
        if prompt:
            self.ask_llm(prompt)

    def ask_llm(self, text: str):
        if not self.busy.acquire(blocking=False):
            self.pending = text     # latest question wins; answered when free
            return
        threading.Thread(target=self._converse, args=(text,), daemon=True).start()

    def _converse(self, text: str):
        try:
            # history keeps TEXT only; the live camera frame is attached to the
            # CURRENT turn (true open-vocabulary vision — YOLO's COCO-80 can't
            # see pens/hats, Sonnet 5's eyes can)
            self.history.append({"role": "user", "content": [
                {"text": f"<scene>{self.scene}</scene>\n{text}"}]})
            self.history[:] = self.history[-12:]  # bounded context
            messages = [dict(m) for m in self.history]
            if self.latest_jpeg:
                jpeg = self.latest_jpeg
                try:    # downscale to 512px long edge: same VLM quality, ~1/3 tokens
                    import cv2
                    import numpy as np
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None and max(img.shape[:2]) > 512:
                        s = 512 / max(img.shape[:2])
                        img = cv2.resize(img, None, fx=s, fy=s)
                        ok, enc = cv2.imencode(".jpg", img,
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        if ok:
                            jpeg = enc.tobytes()
                except Exception:
                    pass
                messages[-1] = {"role": "user", "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
                    {"text": f"<scene>{self.scene}</scene>\n{text}"}]}
            # deterministic skill gate: the phrase table decides, then the body
            # rides along in THIS turn only. Deliberately not appended to
            # self.history, so a skill costs tokens on the turns that need it
            # instead of on every turn after.
            hits = loader.match(text, self.skills)
            for skill in hits:
                try:
                    messages[-1]["content"].append({"text": loader.skill_block(skill)})
                except loader.SkillError as exc:
                    self.get_logger().error(f"skill {skill.name} not loaded: {exc}")
            if hits:
                self.get_logger().info(f"skills active: {[s.name for s in hits]}")

            kwargs = {
                "modelId": MODEL_ID,
                "system": [{"text": self.system_text}],
                "messages": messages,
                # stories need room; temperature deprecated on Claude 5
                "inferenceConfig": {"maxTokens": 1000},
            }
            if hits:
                kwargs["toolConfig"] = loader.TOOL_CONFIG

            resp = self.brt.converse(**kwargs)
            # Bounded: a model that keeps asking for files would otherwise hold the
            # conversation open indefinitely. On the last hop the tool is dropped,
            # which forces an answer instead of a truncated turn.
            for hop in range(loader.MAX_TOOL_HOPS):
                if resp.get("stopReason") != "tool_use":
                    break
                out = resp["output"]["message"]
                messages.append(out)
                results = []
                for block in out.get("content", []):
                    use = block.get("toolUse")
                    if not use:
                        continue
                    body, ok = loader.run_tool(use)
                    self.get_logger().info(
                        f"read_skill_file {use.get('input')} -> "
                        f"{'ok' if ok else body[:80]}")
                    results.append({"toolResult": {
                        "toolUseId": use["toolUseId"],
                        "content": [{"text": body}],
                        "status": "success" if ok else "error"}})
                if not results:
                    break
                messages.append({"role": "user", "content": results})
                if hop == loader.MAX_TOOL_HOPS - 1:
                    kwargs.pop("toolConfig", None)
                resp = self.brt.converse(**kwargs)
            # Claude 5 may prepend reasoning blocks — join every text block
            reply = "".join(b.get("text", "")
                            for b in resp["output"]["message"]["content"]).strip()
            if not reply:
                raise ValueError("empty text in response content")
            if resp.get("stopReason") == "max_tokens":
                reply += "……剩下的下次再講!"
            self.history.append({"role": "assistant", "content": [{"text": reply}]})
            self.get_logger().info(f"agent: {reply}")
            # execute embedded action tags, strip them from the spoken text
            import re as _re
            for g in _re.findall(r"\[gesture:(\w+)\]", reply):
                self.pub_gesture.publish(String(data=g))
            if "[game:start]" in reply:
                self.pub_game.publish(String(data="start"))
            spoken = _re.sub(r"\[(?:gesture:\w+|game:start)\]", "", reply).strip()
            if spoken:
                self.pub_say.publish(String(data=json.dumps({"text": spoken}, ensure_ascii=False)))
        except Exception as e:
            self.get_logger().error(f"bedrock error: {e}")
            self.history and self.history.pop()
            self.pub_say.publish(String(data=json.dumps(
                {"text": "唔,我剛剛走神了,再說一次好嗎?", "cache": True},
                ensure_ascii=False)))
        finally:
            self.busy.release()
            if self.pending:
                nxt, self.pending = self.pending, None
                self.ask_llm(nxt)


def main():
    rclpy.init()
    rclpy.spin(AgentNode())


if __name__ == "__main__":
    main()
