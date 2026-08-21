"""Gesture primitive library for the Unitree G1 (with hands) MuJoCo model.

Poses are dicts of {joint_name: target_qpos_rad}. Joints not named in a pose
hold their current value. Keyframe sequences interpolate with cosine easing.

The exact joint names come from the Menagerie g1_with_hands.xml; run
    python3 gestures.py --list-joints path/to/g1_with_hands.xml
to print them, and adjust JOINTS below if Menagerie renames anything.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field

# Joint names verified against Menagerie g1_with_hands.xml (Dex3-1 hand:
# thumb 3 dof, index 2 dof, middle 2 dof; no head joints — waist stands in).
JOINTS = {
    "shoulder_pitch": "right_shoulder_pitch_joint",
    "shoulder_roll": "right_shoulder_roll_joint",
    "shoulder_yaw": "right_shoulder_yaw_joint",
    "elbow": "right_elbow_joint",
    "wrist_roll": "right_wrist_roll_joint",
    "thumb_0": "right_hand_thumb_0_joint",     # range -1.05..1.05
    "thumb_1": "right_hand_thumb_1_joint",     # range -1.05..0.72
    "thumb_2": "right_hand_thumb_2_joint",     # range -1.75..0
    "index_0": "right_hand_index_0_joint",     # range 0..1.57
    "index_1": "right_hand_index_1_joint",     # range 0..1.75
    "middle_0": "right_hand_middle_0_joint",   # range 0..1.57
    "middle_1": "right_hand_middle_1_joint",   # range 0..1.75
    "waist_yaw": "waist_yaw_joint",
}

# Arm raised in front, elbow bent — the "ready" chant position.
ARM_READY = {
    "shoulder_pitch": -0.9,
    "shoulder_roll": -0.25,
    "elbow": 1.3,
}
# Arm pumped down on each chant beat.
ARM_PUMP = {
    "shoulder_pitch": -0.45,
    "elbow": 1.0,
}

_FINGERS_CURLED = {"index_0": 1.5, "index_1": 1.7, "middle_0": 1.5, "middle_1": 1.7}
_FINGERS_OPEN = {"index_0": 0.0, "index_1": 0.0, "middle_0": 0.0, "middle_1": 0.0}
_THUMB_CURLED = {"thumb_0": 0.9, "thumb_1": -1.0, "thumb_2": -1.6}
_THUMB_OPEN = {"thumb_0": 0.0, "thumb_1": 0.0, "thumb_2": 0.0}

HAND_ROCK = {**_FINGERS_CURLED, **_THUMB_CURLED}      # fist
HAND_PAPER = {**_FINGERS_OPEN, **_THUMB_OPEN}         # open palm
HAND_SCISSORS = {**_FINGERS_OPEN, **_THUMB_CURLED}    # index+middle out, thumb in


@dataclass
class Keyframe:
    pose: dict[str, float]
    duration_s: float  # time to reach this pose from the previous one


@dataclass
class Gesture:
    name: str
    frames: list[Keyframe] = field(default_factory=list)


def _merge(*poses: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in poses:
        out.update(p)
    return out


def chant_pump(n_beats: int, beat_s: float, final_hand: dict[str, float]) -> Gesture:
    """Arm pumps once per chant beat, final beat lands `final_hand` fully formed."""
    g = Gesture(name="chant")
    g.frames.append(Keyframe(_merge(ARM_READY, HAND_ROCK), 0.4))
    for _ in range(n_beats - 1):
        g.frames.append(Keyframe(_merge(ARM_PUMP, HAND_ROCK), beat_s / 2))
        g.frames.append(Keyframe(_merge(ARM_READY, HAND_ROCK), beat_s / 2))
    # final beat: pump down and open into the chosen shape
    g.frames.append(Keyframe(_merge(ARM_PUMP, final_hand), beat_s / 2))
    return g


GESTURES = {
    "rock": Gesture("rock", [Keyframe(_merge(ARM_READY, HAND_ROCK), 0.3),
                             Keyframe(_merge(ARM_PUMP, HAND_ROCK), 0.2)]),
    "paper": Gesture("paper", [Keyframe(_merge(ARM_READY, HAND_ROCK), 0.3),
                               Keyframe(_merge(ARM_PUMP, HAND_PAPER), 0.2)]),
    "scissors": Gesture("scissors", [Keyframe(_merge(ARM_READY, HAND_ROCK), 0.3),
                                     Keyframe(_merge(ARM_PUMP, HAND_SCISSORS), 0.2)]),
    "wave": Gesture("wave", [
        Keyframe(_merge({"shoulder_pitch": -1.6, "shoulder_roll": -0.4, "elbow": 0.6}, HAND_PAPER), 0.5),
        Keyframe({"shoulder_roll": -0.15}, 0.3),
        Keyframe({"shoulder_roll": -0.55}, 0.3),
        Keyframe({"shoulder_roll": -0.15}, 0.3),
        Keyframe({"shoulder_roll": -0.55}, 0.3),
    ]),
    "nod": Gesture("nod", [Keyframe({"waist_yaw": 0.25}, 0.35),
                           Keyframe({"waist_yaw": -0.25}, 0.5),
                           Keyframe({"waist_yaw": 0.0}, 0.35)]),
    "idle": Gesture("idle", [Keyframe({k: 0.0 for k in
                                       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                        "elbow", "waist_yaw")} | HAND_PAPER, 0.8)]),
}


class GesturePlayer:
    """Kinematic pose animator: call step(dt) each render tick; it eases the
    tracked joints toward the active keyframe targets and returns
    {joint_name: qpos}. Deterministic — no dynamics, cannot fall over."""

    def __init__(self, initial: dict[str, float] | None = None):
        self.current: dict[str, float] = dict(initial or {})
        self._queue: list[Keyframe] = []
        self._frame_start: dict[str, float] = {}
        self._frame_t = 0.0

    def play(self, gesture: Gesture, at_time: float | None = None):
        self._queue = list(gesture.frames)
        self._begin_frame()

    def _begin_frame(self):
        self._frame_start = dict(self.current)
        self._frame_t = 0.0

    @property
    def busy(self) -> bool:
        return bool(self._queue)

    def step(self, dt: float) -> dict[str, float]:
        if not self._queue:
            return self.current
        kf = self._queue[0]
        self._frame_t += dt
        a = min(1.0, self._frame_t / max(kf.duration_s, 1e-3))
        ease = 0.5 - 0.5 * math.cos(math.pi * a)  # cosine ease-in-out
        for logical, target in kf.pose.items():
            start = self._frame_start.get(logical, 0.0)
            self.current[logical] = start + (target - start) * ease
        if a >= 1.0:
            self._queue.pop(0)
            self._begin_frame()
        return self.current


def resolve_joint_ids(model) -> dict[str, int]:
    """Map logical joint names -> mujoco qpos addresses; warns on misses."""
    import mujoco
    ids = {}
    for logical, jname in JOINTS.items():
        if jname is None:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            print(f"[gestures] WARN joint not in model: {jname} (logical {logical})")
            continue
        ids[logical] = model.jnt_qposadr[jid]
    return ids


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-joints", metavar="XML")
    args = ap.parse_args()
    if args.list_joints:
        import mujoco
        m = mujoco.MjModel.from_xml_path(args.list_joints)
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"{j:3d} {name}")
        sys.exit(0)
    ap.print_help()
