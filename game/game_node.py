"""game_node — rock-paper-scissors orchestrator (robot-led mode, plan D8).

Flow:
  idle -> (ASR hears 玩/石頭剪刀布/"play"/"rock paper scissors") -> invite
       -> chant (cached TTS clip; T_throw = clip start + duration)
       -> at T_throw-0.55s: /body/gesture chant pump; final hand lands ON the beat
       -> throw: majority-vote /hand_pose samples in [T_throw, T_throw+0.30s]
       -> judge (local 3x3) -> cached result clip -> async Haiku banter
  Beat timing never touches the LLM or ASR (plan D2/D8).

Publishes: /speech/say, /body/gesture, /game/events_out (browser HUD),
           /game/rps_round (one resolved round, for the single writer to persist)
Subscribes additionally to /memory/rps_table (latched) to restore the model.
Requests banter via /agent/ask ({"kind":"banter", ...}); agent_node replies on
/speech/say itself, so banter latency never blocks the game loop.

The opponent model (game/opponent.py) is consulted **at arm time only** — inside
`start_game()`, roughly 3.5 s before the chant begins. Nothing on the beat path
(`on_tts_started`, `tick`) may reach it: that is test item T-H6, and it is asserted
structurally rather than as a wall-clock budget, because a timing assertion that
goes red when the laptop is busy gets a `skip` within two days.

The model is an optimisation on top of a coin, so every one of its failure modes
has to degrade to the coin. `_arm()` and `_learn()` therefore contain their own
exceptions instead of letting them surface in a ROS callback and take the game
loop down with them.
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

# run_ec2.sh launches this file directly, so sys.path[0] is game/, not the repo
# root. Importing as `game.opponent` (rather than plain `opponent`, which would
# work from game/) keeps ONE module identity: the tests import `game.opponent`,
# and two copies of the same file would be two different Opponent classes whose
# isinstance checks and OpponentError types silently disagree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game import opponent as opp   # noqa: E402
from game import rps               # noqa: E402

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
# There is no WINS table here any more. It used to be
# {("rock","scissors"),("scissors","paper"),("paper","rock")}, a second encoding of
# the same 3x3 invariant that game/opponent.py already held as BEATS, with nothing
# forcing the two to agree. Now that the same round both decides what the user
# hears and trains the model, one table is not a tidiness preference: two would let
# the robot announce one result and learn another. See game/rps.judge.

# Latched, depth 1: the accumulated model is state, not an event stream, so a node
# that starts after memory_node still receives the last table instead of waiting
# for a republish. Until memory_node exists (plan S3+) nothing publishes here, and
# the observable consequence is simply that every boot starts from no counts —
# which is the same safe state as a first-ever conversation.
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


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
        # An UNSEEDED rng, unlike the module default. game/opponent.py defaults to
        # random.Random(0) so the offline harness and the tests are reproducible —
        # but on the robot that would make the uniform fallback replay the identical
        # move sequence after every restart, and "uniform" that repeats itself is
        # not unexploitable, it is a pattern anyone who plays twice can read.
        self.opponent = opp.Opponent(rng=random.Random())

        self.pub_say = self.create_publisher(String, "/speech/say", 10)
        self.pub_gesture = self.create_publisher(String, "/body/gesture", 10)
        self.pub_evt = self.create_publisher(String, "/game/events_out", 10)
        self.pub_agent = self.create_publisher(String, "/agent/ask", 10)
        self.pub_round = self.create_publisher(String, "/game/rps_round", 10)
        self.create_subscription(String, "/speech/asr", self.on_asr, 10)
        self.create_subscription(String, "/hand_pose", self.on_hand, 20)
        self.create_subscription(String, "/speech/tts_started", self.on_tts_started, 10)
        self.create_subscription(String, "/game/cmd", self.on_cmd, 10)
        self.create_subscription(String, "/memory/rps_table", self.on_rps_table, LATCHED)
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
    def _arm(self) -> str:
        """Pick this round's move. Commits at arm time (D2), never on the beat.

        The fallback is a plain uniform draw and NOT `self.opponent.uniform_move()`:
        if the model object is the thing that is broken, asking it for the safe
        answer asks the broken object. This is the only place in the node that
        reads a move out of thin air, and it exists so that "the model raised" and
        "no game happens at all" are different events.
        """
        try:
            return self.opponent.choose()
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warning(f"opponent model failed to choose ({exc}) — coin")
            return random.choice(list(opp.MOVES))

    def start_game(self):
        clips = CLIPS_ZH if self.lang == "zh" else CLIPS_EN
        self.robot_move = self._arm()
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
                  if self.t_throw - 0.10 <= t <= self.t_throw + 0.60]
        clips = CLIPS_ZH if self.lang == "zh" else CLIPS_EN
        # `rps.majority` drops the `none` samples itself and breaks ties on a fixed
        # order. The `and g != "none"` filter used to live in the comprehension
        # above and the vote was `max(set(window), …)`, whose tie-break came out of
        # the per-process string hash seed — the same window read as paper under
        # PYTHONHASHSEED=0 and rock under =2.
        user = rps.majority(window)
        if user is None:
            self.say(clips["nohand"], cache=True)
            self.state = "idle"
            self.hud("沒看到手 / no hand seen")
            return
        try:
            _user_outcome, outcome = rps.judge(user, self.robot_move)
        except opp.OpponentError as exc:
            # Unreachable while hand_node's vocabulary stays MOVES + none, which
            # tests/test_game_wiring.py asserts by parsing that node rather than
            # trusting this comment. Handled anyway: the cost of being wrong is one
            # skipped round, and the cost of an uncaught raise here is the node.
            self.get_logger().error(f"unjudgeable round ({exc}) — treated as no hand")
            self.say(clips["nohand"], cache=True)
            self.state = "idle"
            return
        self.score["tie" if outcome == "tie" else
                   ("robot" if outcome == "win" else "user")] += 1
        self.say(clips[outcome], cache=True)
        self.hud(f"你:{user} 機器人:{self.robot_move} → {outcome} "
                 f"(bot {self.score['robot']} : {self.score['user']} you)")
        self.pub_agent.publish(String(data=json.dumps(
            {"kind": "banter", "lang": self.lang, "user": user,
             "robot": self.robot_move, "outcome": outcome,
             "score": self.score}, ensure_ascii=False)))
        self._learn(user)
        self.state = "idle"

    # --- opponent model -----------------------------------------------------
    def _learn(self, user: str) -> None:
        """Fold the resolved round into the model, then hand it to the writer.

        Order matters: the round is announced and the banter requested *before*
        this runs, so a model that throws cannot delay or silence the response the
        person is waiting for.

        This node keeps the table in RAM and never writes to disk — memory_node is
        the single writer (plan S0-3). The round goes out fire-and-forget, so a
        writer that is down, slow or not yet built costs us persistence and never
        a beat.
        """
        try:
            self.opponent.observe(user, self.robot_move)
        except opp.OpponentError as exc:
            self.get_logger().warning(f"model rejected the round ({exc}) — not recorded")
            return
        self.pub_round.publish(String(data=json.dumps(
            {"user": user, "robot": self.robot_move}, ensure_ascii=False)))

    def on_rps_table(self, msg: String) -> None:
        """Adopt an accumulated model published by memory_node (latched).

        Every malformed state is rejected wholesale rather than merged: a
        half-restored table is a model trained on a subset nobody chose, and
        `from_dict` already validates version, orders, count shape and the legality
        of every replayed round. Rejection leaves the current model untouched, so
        the worst case is the counts we already had.
        """
        try:
            restored = opp.Opponent.from_dict(json.loads(msg.data),
                                              rng=self.opponent.rng)
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warning(f"rejected persisted model ({exc}) — keeping RAM")
            return
        self.opponent = restored
        # Per order, not one total: each round increments one counter in EVERY
        # order that had a context, so summing across orders would report roughly
        # 3x the rounds actually played. And order k only ever sees rounds-k of
        # them, so the three numbers legitimately differ.
        obs = {k: sum(sum(c) for c in restored.tables[k].values())
               for k in sorted(restored.orders)}
        self.get_logger().info(
            f"restored opponent model: observations per order {obs}, "
            f"gate {'open' if restored.gate_open() else 'shut'}")


def main():
    rclpy.init()
    rclpy.spin(GameNode())


if __name__ == "__main__":
    main()
