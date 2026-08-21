"""game_node — rock-paper-scissors orchestrator (robot-led mode, plan D8).

Flow:
  idle -> (ASR hears 玩/石頭剪刀布/"play"/"rock paper scissors") -> invite
       -> chant (cached TTS clip; T_throw = clip start + duration)
       -> at T_throw-0.55s: /body/gesture chant pump; final hand lands ON the beat
       -> throw: majority-vote /hand_pose samples in [T_throw, T_throw+0.30s]
       -> judge (local 3x3) -> cached result clip -> async Haiku banter
  Beat timing never touches the LLM or ASR (plan D2/D8).

Publishes: /speech/say, /body/gesture, /game/events_out (browser HUD)
Requests banter via /agent/ask ({"kind":"banter", ...}); agent_node replies on
/speech/say itself, so banter latency never blocks the game loop.
"""
from __future__ import annotations
import json
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

BEATS = {"zh": (5, "石頭、剪刀、布!"), "en": (3, "rock, paper, scissors!")}
# NOTE: invite/nohand texts must NOT contain the trigger phrase 石頭剪刀布 —
# any half-duplex gate leak would re-trigger the game off the robot's own voice.
CLIPS_ZH = {"win": "耶!我贏了!", "lose": "哎呀,你贏了!", "tie": "平手!再來!",
            "invite": "好啊!把手舉到臉旁邊對著鏡頭,聽我口令一起出手哦!",
            "nohand": "咦,我沒看到你的手,把手舉高一點,再來一次!",
            "stopped": "好的,先不玩了!想玩再叫我。"}
CLIPS_EN = {"win": "Yes! I win!", "lose": "Oh no, you win!", "tie": "It's a tie!",
            "invite": "Let's play rock paper scissors! Throw on my count!",
            "nohand": "I couldn't see your hand — let's try again!",
            "stopped": "OK, game over! Call me when you want to play again."}
# Deterministic STOP reflex — never routed through the LLM (plan D2 discipline)
STOP_WORDS = ("停", "不玩", "别玩", "別玩", "暫停", "暂停", "结束", "結束",
              "stop", "enough", "quit", "no more")
BEAT_KEYWORDS = ("石頭剪刀布", "石头剪刀布", "rock paper scissors", "剪刀石頭布")
WINS = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}


class GameNode(Node):
    def __init__(self):
        super().__init__("game")
        self.state = "idle"
        self.lang = "zh"
        self.robot_move = "rock"
        self.t_throw = 0.0
        self.cooldown_until = 0.0   # ignore new-game triggers right after a round
        self.hand_samples: list[tuple[float, str]] = []
        self.score = {"robot": 0, "user": 0, "tie": 0}

        self.pub_say = self.create_publisher(String, "/speech/say", 10)
        self.pub_gesture = self.create_publisher(String, "/body/gesture", 10)
        self.pub_evt = self.create_publisher(String, "/game/events_out", 10)
        self.pub_agent = self.create_publisher(String, "/agent/ask", 10)
        self.create_subscription(String, "/speech/asr", self.on_asr, 10)
        self.create_subscription(String, "/hand_pose", self.on_hand, 20)
        self.create_subscription(String, "/speech/tts_started", self.on_tts_started, 10)
        self.create_subscription(String, "/game/cmd", self.on_cmd, 10)
        self.timer = self.create_timer(0.02, self.tick)
        self.get_logger().info("game node ready — say 石頭剪刀布 / 'rock paper scissors'")
        self._warm_clips()

    # --- setup ------------------------------------------------------------
    def _warm_clips(self):
        for clips in (CLIPS_ZH, CLIPS_EN):
            for text in clips.values():
                self.say(text, cache=True, warm=True)
        for _, chant in BEATS.values():
            self.say(chant, cache=True, warm=True)

    def say(self, text: str, cache=False, warm=False, uid=None):
        self.pub_say.publish(String(data=json.dumps(
            {"text": text, "cache": cache, "id": uid or ("warm" if warm else text[:24])},
            ensure_ascii=False)))

    def hud(self, text: str):
        self.pub_evt.publish(String(data=json.dumps(
            {"text": text, "hud": text,
             "score": self.score, "state": self.state}, ensure_ascii=False)))

    # --- inputs -----------------------------------------------------------
    def on_cmd(self, msg: String):
        if msg.data == "start":
            # the cmd path (agent/remote) must respect state AND cooldown too —
            # bypassing it let the LLM chain games back-to-back (runaway loop)
            if self.state == "idle" and time.monotonic() >= self.cooldown_until:
                self.start_game()
        elif msg.data == "stop":
            self.stop_game()

    def stop_game(self):
        # idempotent: ASR partials repeat the utterance, so stop may fire 2-4x
        already_stopped = (self.state == "idle"
                           and time.monotonic() < self.cooldown_until)
        self.state = "idle"
        self.cooldown_until = time.monotonic() + 15.0   # long quiet period
        if already_stopped:
            return
        self.pub_gesture.publish(String(data="idle"))
        clips = CLIPS_ZH if self.lang == "zh" else CLIPS_EN
        self.say(clips["stopped"], cache=True)
        self.hud("已停止 / stopped")

    @staticmethod
    def is_trigger(text: str) -> bool:
        # fuzzy: streaming ASR garbles/repeats tokens (observed 石头剪刀石石剪刀),
        # so exact-phrase matching misses real triggers
        t = text.lower()
        if "rock" in t and ("paper" in t or "scissor" in t):
            return True
        return "剪刀" in t and ("石" in t or "布" in t)

    def on_asr(self, msg: String):
        text = json.loads(msg.data).get("text", "").lower()
        # STOP is a hard reflex: works in ANY state, checked before anything else
        if any(w in text for w in STOP_WORDS):
            self.stop_game()
            return
        if self.state != "idle" or time.monotonic() < self.cooldown_until:
            return
        if self.is_trigger(text):
            self.lang = "zh" if any(c in text for c in "石头頭剪") else "en"
            self.start_game()

    def on_hand(self, msg: String):
        d = json.loads(msg.data)
        self.hand_samples.append((time.monotonic(), d["gesture"]))
        del self.hand_samples[:-40]

    def on_tts_started(self, msg: String):
        d = json.loads(msg.data)
        if d.get("id") == "chant" and self.state == "await_chant":
            # 布 lands at clip end; jitter ring adds ~80 ms playback lag on the Mac
            self.t_throw = time.monotonic() + d["dur_s"] + 0.08
            self.state = "chanting"
            n = BEATS[self.lang][0]
            beat = d["dur_s"] / n
            self.pub_gesture.publish(String(
                data=f"chant:{n}:{beat:.3f}:{self.robot_move}"))
            self.hud(f"口令中… 機器人已選好(保密) / chanting, robot committed")

    # --- game flow ----------------------------------------------------------
    def start_game(self):
        clips = CLIPS_ZH if self.lang == "zh" else CLIPS_EN
        self.robot_move = random.choice(["rock", "paper", "scissors"])  # commit at arm time (D2)
        self.say(clips["invite"], cache=True, uid="invite")
        self.state = "inviting"
        self.t_invite = time.monotonic()
        self.hud("遊戲開始 / game on")

    def tick(self):
        now = time.monotonic()
        if self.state == "inviting" and now - self.t_invite > 3.5:
            self.state = "await_chant"
            self.say(BEATS[self.lang][1], cache=True, uid="chant")
        elif self.state == "chanting" and now >= self.t_throw + 0.65:
            self.resolve()

    def resolve(self):
        self.cooldown_until = time.monotonic() + 3.0
        # generous window: people hold the shape well past the beat
        window = [g for t, g in self.hand_samples
                  if self.t_throw - 0.10 <= t <= self.t_throw + 0.60 and g != "none"]
        clips = CLIPS_ZH if self.lang == "zh" else CLIPS_EN
        if not window:
            self.say(clips["nohand"], cache=True)
            self.state = "idle"
            self.hud("沒看到手 / no hand seen")
            return
        user = max(set(window), key=window.count)
        if user == self.robot_move:
            outcome = "tie"
        elif (self.robot_move, user) in WINS:
            outcome = "win"    # robot wins
        else:
            outcome = "lose"
        self.score["tie" if outcome == "tie" else
                   ("robot" if outcome == "win" else "user")] += 1
        self.say(clips[outcome], cache=True)
        self.hud(f"你:{user} 機器人:{self.robot_move} → {outcome} "
                 f"(bot {self.score['robot']} : {self.score['user']} you)")
        self.pub_agent.publish(String(data=json.dumps(
            {"kind": "banter", "lang": self.lang, "user": user,
             "robot": self.robot_move, "outcome": outcome,
             "score": self.score}, ensure_ascii=False)))
        self.state = "idle"


def main():
    rclpy.init()
    rclpy.spin(GameNode())


if __name__ == "__main__":
    main()
