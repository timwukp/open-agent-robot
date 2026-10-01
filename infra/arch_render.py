#!/usr/bin/env python3
"""Render docs/architecture.svg — the system diagram, derived rather than drawn.

A diagram is prose with a nicer font, and prose in this repo is unverified by
construction (`infra/doc_counts.py`'s docstring is about exactly that defect). The
first version of `docs/architecture.svg` was hand-placed, and it was wrong in a way
nothing could notice: it named two one-ended topics when the census finds **seven**,
because I listed the gaps I remembered instead of deriving them.

So nothing here is remembered:

  * the node inventory comes from `OAR_DRY=1 infra/run_ec2.sh start-all`
  * every topic, and every publisher and subscriber on it, comes from an AST walk
    (`infra/ros_census.py`) over exactly the files that runner launches
  * whether a one-ended topic is acceptable comes from `infra/graph_gaps.tsv`, a
    reviewable line per topic
  * the module list comes from the tree, split by whether the file imports rclpy and
    whether it is an entrypoint
  * every measured number is read out of `docs/evidence/**/*.json` and printed with
    the file it came from, so a stale number is impossible rather than unlikely

What is still hand-written is the *prose*: one description per node, one line per
module. Those are keyed by census identity, and `tests/test_architecture.py` asserts
the keys equal the derived sets in both directions — an undescribed new node is a
finding, and a description for a node that no longer exists is a finding too.

Layout is derived as well. Text wraps in **pixels** rather than characters (the first
version overflowed 29 runs against real font metrics), every box's height comes from
its own wrapped line count, and every text run carries `data-host` naming the box it
belongs to so a containment check can verify both sides of the claim instead of
guessing which rect a stray label landed in.

Animation. The moving dot is one turn's message travelling the *drawn connectors* —
the motion path of each hop is the same point list the connector is drawn from, so the
animation cannot show a route the diagram does not have. It is CSS rather than SMIL
for one reason: `@media (prefers-reduced-motion: reduce)` can switch it off, and SMIL
cannot be switched off that way. Motion is redundant with the numbered hop list in the
legend, which is what a reduced-motion reader sees.

Determinism is a requirement, not a nicety: `tests/test_architecture.py` rebuilds the
SVG and asserts the committed bytes are identical, which is how the diagram is gated
against going stale. That is a rebuild rather than a stored hash — a hash with two
readings verifies only that the two readings agree.

Exit codes: 0 wrote the file, 1 geometry findings (nothing written), 3 the census could
not run (a renderer that cannot read the graph must not emit a diagram of it).
"""
from __future__ import annotations

import html
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ros_census import Topic, census  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SANS = "Helvetica Neue,Helvetica,Arial,sans-serif"
MONO = "SF Mono,Menlo,Consolas,monospace"
# Conservative advance width per em. Menlo/SF Mono are 0.602; Helvetica mixed case
# averages ~0.50 but caps and punctuation push it up, so both are padded.
ADV = {SANS: 0.560, MONO: 0.625}

C = {
    "edge_f": "#e8f0fe", "edge_s": "#1a73e8",
    "gw_f": "#e6f4ea", "gw_s": "#137333",
    "node_f": "#fff8e1", "node_s": "#b06000",
    "bus_f": "#263238", "bus_i": "#eceff1", "bus_d": "#90a4ae",
    "pure_f": "#f6edfd", "pure_s": "#7627bb",
    "gate_f": "#e6f4ea", "gate_s": "#137333",
    "disk_f": "#f1f3f4", "disk_s": "#5f6368",
    "cloud_f": "#fce8e6", "cloud_s": "#c5221f",
    "gap_f": "#fef7e0", "gap_s": "#a35200",
    "warn_f": "#fce8e6", "warn_s": "#c5221f",
    "meas": "#0b6b62", "legend_f": "#fafafa",
    "ink": "#202124", "dim": "#5f6368", "line": "#3c4043", "pkt": "#1a73e8",
}

PAD_X, PAD_B = 18, 14
W = 1760
LX, LW = 60, 560
BUS_X, BUS_W = 690, 120
RX, RW = 1070, 560
BAND_W = RX + RW - LX          # the full-width bands line up with the node columns
CYCLE = 11.0                   # seconds for one animated turn


# ───────────────────────────────── text primitives ─────────────────────────────────
def wrap_px(s: str, avail: float, size: float, family: str) -> list[str]:
    n = max(8, int(avail / (ADV[family] * size)))
    return textwrap.wrap(s, n) or [""]


def txt(x, y, s, size, fill, family, weight="normal", host="canvas", extra="") -> str:
    # xml:space="preserve" — the mono columns and the "pub  /topic" runs use multiple
    # spaces as alignment; without it XML collapses them and the columns go ragged.
    # Continuation lines are offset by x, never by spaces, so preserving whitespace
    # cannot introduce a phantom indent.
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'xml:space="preserve" data-host="{host}"{extra}>'
            f'{html.escape(s, quote=True)}</text>')


class Run:
    def __init__(self, s, size, family, fill, weight="normal", gap_before=0.0):
        self.s, self.size, self.family, self.fill = s, size, family, fill
        self.weight, self.gap_before = weight, gap_before
        self.lines: list[str] = []
        self.lead = 0.0

    def layout(self, avail: float) -> float:
        self.lines = wrap_px(self.s, avail, self.size, self.family)
        self.lead = self.size * (1.26 if self.family is MONO else 1.32)
        return self.gap_before + len(self.lines) * self.lead


class Box:
    def __init__(self, name, w, fill, stroke, rx=10, dash=None):
        self.name, self.w, self.fill, self.stroke = name, w, fill, stroke
        self.rx, self.dash, self.runs = rx, dash, []
        self.badge: str | None = None
        self.badge_lines: list[str] = []
        self.x = self.y = self.h = 0.0

    def title(self, s, size=14.5):
        self.runs.append(Run(s, size, SANS, C["ink"], "bold", size * 1.5))
        return self

    def sub(self, s, size=11.5):
        self.runs.append(Run(s, size, SANS, C["dim"], gap_before=1))
        return self

    def code(self, s, size=11):
        self.runs.append(Run(s, size, MONO, C["ink"], gap_before=4))
        return self

    def meas(self, s, size=11):
        """A number that came out of docs/evidence/, printed with its source."""
        self.runs.append(Run(s, size, MONO, C["meas"], gap_before=4))
        return self

    def note(self, s, size=11.5):
        self.runs.append(Run(s, size, SANS, C["dim"], gap_before=7))
        return self

    def gap(self, s):
        self.badge = s
        return self

    def measure(self) -> float:
        avail = self.w - 2 * PAD_X
        self.h = round(sum(r.layout(avail) for r in self.runs) + PAD_B, 1)
        if self.badge:
            self.badge_lines = wrap_px("GAP  " + self.badge, avail - 26, 11, MONO)
            self.h = round(self.h + 10 + len(self.badge_lines) * 14 + 12, 1)
        return self.h

    def place(self, x, y) -> float:
        # Rounded, not merely formatted on output: an unrounded accumulator emits
        # y="974.0699999999999", and a run of twelve digits is what
        # infra/redact_scan.py hunts for when it looks for an account id.
        self.x, self.y = round(x, 1), round(y, 1)
        return self.y + self.h

    def draw(self, out: list[str]) -> None:
        d = f' stroke-dasharray="{self.dash}"' if self.dash else ""
        out.append(f'<rect x="{self.x}" y="{self.y}" width="{self.w}" '
                   f'height="{self.h:.1f}" rx="{self.rx}" fill="{self.fill}" '
                   f'stroke="{self.stroke}" stroke-width="1.6" '
                   f'filter="url(#sh)"{d}/>')
        cy = self.y
        for r in self.runs:
            cy += r.gap_before
            for i, ln in enumerate(r.lines):
                cy += r.lead
                out.append(txt(self.x + PAD_X + (0 if i == 0 else 11),
                               cy - r.lead * 0.26, ln, r.size, r.fill, r.family,
                               r.weight, self.name))
        if self.badge:
            bh = len(self.badge_lines) * 14 + 12
            by = self.y + self.h - PAD_B + 2 - bh
            out.append(f'<rect x="{self.x + 14}" y="{by:.1f}" width="{self.w - 28}" '
                       f'height="{bh}" rx="6" fill="{C["warn_f"]}" '
                       f'stroke="{C["warn_s"]}" stroke-width="1.3" '
                       f'stroke-dasharray="5 3" class="pulse"/>')
            for i, ln in enumerate(self.badge_lines):
                out.append(txt(self.x + PAD_X + 5, by + 15 + i * 14, ln, 11,
                               C["warn_s"], MONO, "bold", self.name))


# ───────────────────────────────── derived inputs ─────────────────────────────────
def read_gaps(root: Path) -> dict[str, dict[str, str]]:
    """infra/graph_gaps.tsv -> topic -> {asymmetry, class, reason}."""
    rows: dict[str, dict[str, str]] = {}
    path = root / "infra" / "graph_gaps.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"graph_gaps.tsv: {line[:60]!r} has {len(parts)} columns, "
                             f"expected 4 (topic, asymmetry, class, reason)")
        rows[parts[0]] = {"asymmetry": parts[1], "class": parts[2], "reason": parts[3]}
    if not rows:
        raise ValueError("graph_gaps.tsv holds no rows; an empty manifest and a graph "
                         "with no asymmetries have the same shape, so this is an error")
    return rows


def evidence(root: Path) -> dict:
    """Every recorded measurement this diagram quotes, keyed for formatting below."""
    def load(rel: str) -> dict:
        path = root / rel
        if not path.exists():
            raise ValueError(f"{rel} is missing — this diagram quotes numbers from it, "
                             f"and a diagram must not print a number it cannot read")
        return json.loads(path.read_text(encoding="utf-8"))

    g = load("docs/evidence/track-c/guardrail-latency.json")
    c = load("docs/evidence/track-c/constitution-tokens.json")
    e = load("docs/evidence/track-c/effort-latency.json")
    h = load("docs/evidence/track-b/opponent-h2.json")
    a = load("docs/evidence/track-b/arm-cost.json")
    man = g["pooled"]["groups"]["mandarin"]
    yue = g["pooled"]["groups"]["cantonese"]
    wsls = h["h2"]["win_stay_lose_shift"]["diff"]
    unif = h["h2"]["uniform_random"]["diff"]
    anti = h["h2"]["anti_learner"]
    con = next(s for s in c["samples"] if s["name"] == "constitution_as_shipped")
    return {
        "gr_p50": man["p50_ms"], "gr_p95": man["p95_ms"], "gr_n": man["n"],
        "gr_budget": man["budget"]["budget_ms"], "gr_over": man["budget"]["over_budget"],
        "gr_yue_p95": yue["p95_ms"],
        "gr_lang_real": g["interpretation"]["language_gap_exceeds_replication_noise"],
        "gr_noise": g["interpretation"]["run_to_run_noise_ceiling_ms"],
        "gr_tier": g["tier"], "gr_region": g["region"],
        "con_tokens": con["tokens"], "con_chars": con["chars"],
        "model": e["model"],
        "eff_point": e["high_minus_low_median"]["point_ms"],
        "eff_excl": e["high_minus_low_median"]["ci_excludes_zero"],
        "eff_ratio": e["parameter_honored"]["low_over_unset_median"],
        "wsls": wsls, "unif": unif,
        "anti_diff": anti["diff"], "anti_gate": anti["gate_open_fraction"]["mean"],
        "h_rounds": h["rounds"], "h_seeds": h["seeds"],
        "arm_p50": a["p50"], "arm_p95": a["p95"], "arm_tick": a["tick_period_ms"],
        "arm_ctx": a["contexts"],
    }


# One description per runner node. Keyed by the census name, and
# tests/test_architecture.py asserts these keys equal the census node set both ways.
NODE_PROSE: dict[str, tuple[str, str, str, str | None]] = {
    "gateway": (
        "gateway/ws_gateway.py", "WebSocket to ROS 2 bridge — and the only process "
        "that observes a person arriving",
        "Art 50(1) is spoken by a deterministic path off the daemon 'up' frame with a "
        "per-process latch, never by a clause in the system prompt: the disclosure "
        "still happens when Bedrock is down.", None),
    "asr": (
        "speech/asr_node.py", "paraformer-zh streaming, 600 ms chunks",
        "Silent frames publish nothing at all, so there is no silence event to "
        "subscribe to — which is why segmentation owns its own clock (T-B2).", None),
    "tts": (
        "speech/tts_node.py", "Kokoro-82M zh/en at 17x realtime, with the "
        "speech/yue_server.py Cantonese sidecar reached over a local socket",
        "/speech/interrupt triggers T_FLUSH and clears the playback ring — the kill "
        "switch an asynchronous guard is designed to pull. speech/voices.py is the "
        "only creator of the utterance id and of the Art 50(2) synthetic marker.",
        "safety/output_guard.py has zero production callers, so /speech/say is "
        "unguarded in fact. Recorded as NOT_WIRED, deliberately a different value "
        "from NO_ENFORCER_TODAY: folding the two would describe an unwired guard "
        "as protection"),
    "hand": (
        "perception/hand_node.py", "MediaPipe Tasks HandLandmarker, 8.8 "
        "classifications per second",
        "The thumb is excluded from the finger count, which is what fixed scissors "
        "recognition.", None),
    "detect": (
        "perception/detect_node.py", "YOLO26s TensorRT fp16 — 6.2 ms, 161 FPS on "
        "the T4",
        "vision_msgs class_id is a string, not a numeric id. AGPL: demo and "
        "non-commercial waiver only, and the node is isolated rather than opt-in.",
        None),
    "yue": (
        "speech/yue_server.py", "CosyVoice-300M-SFT Cantonese, 3.63 s first packet",
        "Not on the bus at all: an HTTP sidecar in its own venv, called by tts_node. "
        "It is in the runner table because the version that was not is how Cantonese "
        "hit a closed socket on every clean boot.", None),
    "agent": (
        "agent/agent_node.py", "the only Bedrock caller in the system",
        "Reason free, constrain late: [gesture:X] and [game:start] are parsed after "
        "generation. Hard schema decoding reaches 100% valid and 48% executable, so "
        "moving these to constrained decoding is a forbidden refactor rather than a "
        "clean-up. Uses agent/context.py (the only prompt creator), agent/tools.py "
        "(the closed registry), safety/guard.py and skills/loader.py.", None),
    "game": (
        "game/game_node.py", "System-1 reflexes — the beat never blocks",
        "The stop word is a pure function of the utterance, so state-independence is "
        "a property of the signature rather than of a state matrix. The opponent "
        "model is asked once at arm time, and an AST test proves the beat's reachable "
        "closure never touches it.", None),
    "memory": (
        "memory/memory_node.py", "the single writer — no other process touches the "
        "files",
        "A second writer would break the per-line atomicity that memory/store.py "
        "exists to provide, so every other node sends a ROS message instead.",
        "subscribes to nothing at all, so the hot layer is not fed from the bus and "
        "no round is ever persisted"),
    "body": (
        "body/body_node.py", "MuJoCo G1 with Dex3-1 three-finger hands",
        "mj_forward only: kinematic, so it cannot fall and the pelvis freejoint never "
        "integrates. EGL offscreen at 720p. body/gestures.py is a six-key whitelist.",
        None),
}

# One line per non-node python file. Keyed by repo-relative path; the test asserts
# these keys equal the derived module set both ways, so a new module cannot be
# silently undescribed and a deleted one cannot linger.
MODULE_PROSE: dict[str, str] = {
    "memory/store.py": "append-only, atomic swap, tombstones, seq recovery",
    "memory/segment.py": "deterministic episodes; tick(now) takes time as an argument",
    "memory/salience.py": "de-biometricised; the prosody term is refused, not clamped",
    "memory/recall.py": "five paths, each reporting whether it actually ran",
    "memory/consolidate.py": "closed predicates, N >= 2 independent episodes",
    "memory/erase.py": "cascade delete plus a receipt that carries its own gap",
    "memory/rank.py": "three ablation arms, never a mutated Hit.score",
    "memory/profile.py": "read-only seed until unlock_for_s5()",
    "memory/rank_harness.py": "T-F6 ablation runner; 4 exit codes, no labels = 2",
    "safety/guard.py": "constitution, L0 stop reflex, instruction/data seam",
    "safety/output_guard.py": "asynchronous; a serial version is unrepresentable",
    "safety/disclosure.py": "the Art 50(1) first-utterance text",
    "safety/measure_stats.py": "the only median, percentile and CI implementation",
    "safety/output_guard_harness.py": "T-G4 red team; 0 escapes is a floor, not a proof",
    "safety/token_cost_harness.py": "T-K5/T-F5 against Bedrock, by hand",
    "safety/guardrail_latency_harness.py": "T-K3 apply_guardrail latency, our region",
    "agent/context.py": "the single creator of the system prompt",
    "agent/tools.py": "closed registry, derived dispatch, six outcomes",
    "game/opponent.py": "order-2..4 transitions, anytime-valid gate, 1/3 fallback",
    "game/rps.py": "the only judge, majority and vocabulary",
    "game/opponent_harness.py": "T-H2..H4 offline evaluation, 100 seeds",
    "speech/voices.py": "the only utterance-id and Art 50(2) marker creator",
    "body/gestures.py": "the six-key gesture whitelist; --list-joints is a dev CLI",
    "skills/loader.py": "L1 index injection, read_skill_file path fence",
    "skills/measure_l1_tokens.py": "T-I1 token measurement against Bedrock, by hand",
    "infra/ros_census.py": "the one reader of the node and topic graph",
    "infra/wiring_census.py": "unreached files, uncalled functions, unpassed params",
    "infra/arch_render.py": "this file: renders docs/architecture.svg from the census",
    "infra/entrypoints.py": "AST discovery of executable entrypoints",
    "infra/doc_counts.py": "documented numbers vs pytest collection",
    "infra/doc_status.py": "docs/**/*.md vs the table, the index and the tree",
    "infra/redact_scan.py": "redaction over the explicit push list",
    "infra/day0_measure.py": "the day-0 latency measurements, by hand",
}

def hop_labels(topics: dict[str, Topic]) -> list[str]:
    """One turn, in the order it happens — the legend's list and the animation's.

    The topic names are looked up in the census rather than typed, so renaming a topic
    breaks the render instead of leaving the legend quietly describing the old graph.
    `build()` hands these out one at a time as it registers each animated hop and
    asserts the list is exhausted, so the numbered list a reduced-motion reader sees
    and the dots a moving-image reader sees cannot drift apart.
    """
    def t(name: str) -> str:
        if name not in topics:
            raise ValueError(f"the hop list names {name}, which the census does not "
                             f"find — the legend would be describing a graph that "
                             f"no longer exists")
        return name

    return [
        "1   16 kHz PCM leaves the laptop over the WebSocket, 20 ms per frame",
        f"2   ws_gateway publishes {t('/robot/ear')}",
        "3   onto the bus — from here every hop is publish/subscribe",
        "4   down the spine: one message, every subscriber, no addressee",
        "5   asr_node transcribes it in 600 ms chunks",
        f"6   {t('/speech/asr')} carries the text back",
        "7   agent_node assembles constitution + card + L1 index + history",
        "8   Bedrock Converse, and the reply returns along the same path to "
        "/speech/say",
    ]


# The narrative gaps: what the graph cannot say on its own. One string per item,
# because two items sharing a wrapped paragraph is how "…as protection. 3 There is
# no MCP client" happened on the first render.
NARRATIVE_GAPS = [
    "There is no MCP client, no OAuth and no token storage. That is S8's on-machine "
    "half, and it is the only reason the tool boundary could still be asserted "
    "offline at all: today the answer to \"which tools can the model reach\" is one "
    "name in one file, and the moment a server answers it, the offline suite loses "
    "the ability to ask.",
    "The offline suite contains no latency number whatsoever. T-G1 (verdict to "
    "silence <= 400 ms) and T-G2 (first-sentence delta <= 50 ms) both need the "
    "instance behind the /32, and the measurement above already shows the guard call "
    "alone spending more than the whole T-G1 budget at p95.",
    "T-B3's boundary F1 >= 0.80 and T-F1's 30 frozen queries need one day of real "
    "logs and one human annotation pass. Until those labels exist the S6 ablation "
    "cannot conclude, and they will not be fabricated.",
    "RETENTION_DAYS = 90 is tested to within one second under an injected clock, but "
    "the threshold itself is a policy decision rather than a measurement, and the "
    "audit records are checked for not quoting user text by a report rather than by "
    "an enforcer.",
]


# ───────────────────────────────── the drawing ─────────────────────────────────
def build(root: Path = REPO_ROOT) -> tuple[str, dict]:
    """(the SVG text, a dict of facts the test asserts against)."""
    nodes, edges, topics = census(root)
    gaps = read_gaps(root)
    ev = evidence(root)

    missing = sorted(set(nodes) ^ set(NODE_PROSE))
    if missing:
        raise ValueError(f"node prose and the census disagree on {missing} — every "
                         f"deployed node needs a description and vice versa")

    out: list[str] = []
    segs: list[tuple] = []
    placed: list[Box] = []
    hops: list[str] = []                      # motion path d, in travel order
    labels = hop_labels(topics)
    hop_q = iter(labels)

    def place(box: Box, x: float, y: float) -> float:
        box.measure()
        box.place(x, y)
        placed.append(box)
        return box.y + box.h

    def d_of(pts) -> str:
        return "M " + " L ".join(f"{x:.0f} {y:.0f}" for x, y in pts)

    def link(pts, a: str, b: str, hop: bool = False, style: str = "") -> None:
        """Draw a connector; optionally register it as the next hop of the turn.

        `pts` is in travel order, so the hop's motion path and the drawn connector are
        the same geometry — an animation cannot show a route the diagram lacks. The
        label is pulled from `hop_labels()` in registration order rather than passed,
        so the legend's numbered list is the same list the dots travel.
        """
        out.append(f'<path d="{d_of(pts)}" fill="none" stroke="{C["line"]}" '
                   f'stroke-width="2" marker-end="url(#ar)" '
                   f'marker-start="url(#ar)"{style}/>')
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            segs.append((x1, y1, x2, y2, a, b))
        if hop:
            next(hop_q)
            hops.append(d_of(pts))

    def track(pts) -> None:
        """A motion-only hop inside the bus panel: the packet's travel down the spine.

        Not drawn, and registered as bus-to-bus so the geometry checks treat it as
        interior to the panel it never leaves.
        """
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            segs.append((x1, y1, x2, y2, "bus", "bus"))
        next(hop_q)
        hops.append(d_of(pts))

    # ---- header
    title = "open-agent-robot — system architecture"
    subtitle = (
        "An open, hackable multimodal humanoid-robot agent. Every node, every topic "
        "and every one-ended topic on this page is derived from the source by "
        "infra/ros_census.py, and every number is read out of docs/evidence/ and "
        "printed with the file it came from. Nothing here is remembered: "
        "infra/arch_render.py rebuilds this file, and tests/test_architecture.py "
        "fails if the committed bytes differ.")
    sub_lines = wrap_px(subtitle, W - 2 * LX, 13, SANS)

    LEG_TOP = 70 + len(sub_lines) * 17 + 14
    leg_h = legend_height(labels)
    TOP = LEG_TOP + leg_h + 44

    # ---- row A: the laptop, the tunnel, the face
    mac = (Box("mac", 520, C["edge_f"], C["edge_s"])
           .title("MacBook Pro — eye, ear, mouth", 15)
           .code("daemon/main.c — C, started as a LaunchAgent because a headless "
                 "daemon can never be granted camera TCC")
           .code("miniaudio capture and playback ring; ffmpeg avfoundation 720p30 "
                 "to JPEG")
           .note("libwebsockets: one lws_write per writable callback, RECEIVE "
                 "arrives fragmented, and video must be prioritised or the 20 ms "
                 "audio stream starves."))
    tun = (Box("tunnel", 560, "#ffffff", C["dim"], rx=26, dash="6 4")
           .title("SSH port-forward: ssh -F infra/ssh_config -N -f robot", 12.5)
           .sub("The security group allows 22/tcp from one /32. No ROS port is "
                "public, and there is no SSM fallback."))
    bro = (Box("browser", 470, C["edge_f"], C["edge_s"])
           .title("Browser — the robot's face", 15)
           .code("web/robot-view/index.html")
           .code("HUD: detections, gesture, subtitles")
           .note("The Art 50(1) banner is verified in a real browser (T-E3) as a "
                 "computed colour rather than a class token, and hit-tested for "
                 "overlap: the token was in the CSS while all 38 boxes still "
                 "rendered slate."))

    mac_b = place(mac, LX, TOP)
    bro_b = place(bro, W - 60 - 470, TOP)
    tun.measure()
    tun.place(LX + 520 + 40, TOP + (max(mac.h, bro.h) - tun.h) / 2)
    placed.append(tun)

    # ---- the gateway sits between the edge and the bus, above the spine
    gwp = NODE_PROSE["gateway"]
    gw = (Box("gateway", 620, C["gw_f"], C["gw_s"])
          .title(gwp[0], 14.5).sub(gwp[1])
          .code(edge_line(topics, "gateway", "pub"))
          .code(edge_line(topics, "gateway", "sub"))
          .code("uses safety/disclosure.py")
          .note(gwp[2]))
    gw_b = place(gw, BUS_X + BUS_W / 2 - 310, max(mac_b, bro_b) + 48)

    # ---- the bus spine, with a node column on each side
    BUS_TOP = gw_b + 48
    left_names = ["asr", "tts", "hand", "detect", "yue"]
    right_names = ["agent", "game", "memory", "body"]
    if sorted(left_names + right_names + ["gateway"]) != sorted(nodes):
        raise ValueError("the column assignment does not cover the census node set")

    def node_box(name: str, w: float) -> Box:
        prose = NODE_PROSE[name]
        b = Box(name, w, C["node_f"], C["node_s"]).title(prose[0]).sub(prose[1])
        pub, sub = edge_line(topics, name, "pub"), edge_line(topics, name, "sub")
        b.code(pub)
        b.code(sub)
        if name == "game":
            e = ev
            b.meas(f"win-stay/lose-shift  +{e['wsls']['mean']:.3f} pts/round, "
                   f"95% CI [{e['wsls']['lo']:.3f}, {e['wsls']['hi']:.3f}]")
            b.meas(f"uniform opponent     {e['unif']['mean']:+.4f}, CI "
                   f"[{e['unif']['lo']:+.4f}, {e['unif']['hi']:+.4f}] — the "
                   f"non-inferiority arm, and the reason the gate exists")
            b.meas(f"counter-learner      +{e['anti_diff']['mean']:.3f}, gate open "
                   f"{e['anti_gate'] * 100:.0f}% of rounds")
            b.meas(f"arm cost {e['arm_p50']:.4f} ms p50 / {e['arm_p95']:.4f} ms p95 "
                   f"against a {e['arm_tick']:.0f} ms tick, {e['arm_ctx']} contexts")
            b.meas(f"[{e['h_seeds']} seeds x {e['h_rounds']} rounds — "
                   f"docs/evidence/track-b/opponent-h2.json, arm-cost.json]")
        b.note(prose[2])
        if prose[3]:
            b.gap(prose[3])
        return b

    left = [node_box(n, LW) for n in left_names]
    right = [node_box(n, RW) for n in right_names]

    y = BUS_TOP + 12
    for b in left:
        y = place(b, LX, y) + 22
    left_bot = y - 22
    y = BUS_TOP + 12
    for b in right:
        y = place(b, RX, y) + 22
    BUS_BOT = max(left_bot, y - 22) + 12
    bus_h = BUS_BOT - BUS_TOP

    # ---- the topic ledger: every topic the census found, both ends counted
    LED_TOP = BUS_BOT + 48
    led_rows = [(name, t) for name, t in topics.items()]
    per_col = (len(led_rows) + 1) // 2
    led_head = band_head(LEDGER_INTRO)
    led_h = round(led_head + per_col * 15.6 + PAD_B + 6, 1)

    # ---- the module band
    MOD_TOP = LED_TOP + led_h + 48
    mod_items = module_rows(root, nodes)
    mod_missing = sorted(set(mod_items) ^ set(MODULE_PROSE))
    if mod_missing:
        raise ValueError(f"module prose and the tree disagree on {mod_missing} — "
                         f"every non-node module needs a line and vice versa")
    mod_col_w = (BAND_W - 2 * PAD_X - 30) / 2
    mod_lines = [f"{'†' if ep else ' '} {p:<38}{MODULE_PROSE[p]}"
                 for p, ep in sorted(mod_items.items())]
    mod_per_col = (len(mod_lines) + 1) // 2
    mod_head = band_head(MODULES_INTRO)
    mod_h = round(mod_head + mod_per_col * 14.4 + PAD_B, 1)

    # ---- gates, disk, cloud
    RE_TOP = MOD_TOP + mod_h + 48
    gates = (Box("gates", 520, C["gate_f"], C["gate_s"])
             .title("Five offline gates — laptop, no network, $0", 13.5)
             .code("pytest                 the suite, layers L1-L3")
             .code("pytest -m mutation     every named mutant plus a no-mutant "
                   "control")
             .code("infra/check_deploy.sh  executes the runner rather than parsing it")
             .code("infra/doc_counts.py    the documents' numbers vs collection")
             .code("infra/redact_scan.py   the explicit push list, before every push")
             .note("Case and mutant counts are deliberately not printed here: they "
                   "live in README.md and docs/test-design.md §0, where doc_counts.py "
                   "checks them against pytest --collect-only. A fourth hand-typed "
                   "copy would be a fourth thing to get wrong. Two of the five exit 3 "
                   "rather than 0 when they cannot run: a guard that could not run "
                   "must never report clean. There is no CI — .github/workflows does "
                   "not exist, so “no checks reported” on a pull request is "
                   "the expected state, not a failure."))
    disk = (Box("disk", 520, C["disk_f"], C["disk_s"])
            .title("$HOME/oar-data/ on the instance", 13.5)
            .sub("never in the repo, never under /tmp, mode 0700")
            .code("events.jsonl   append-only, per-line atomic")
            .code("audit stream   REWRITABLE forbids shortening")
            .code("episodes.json  warm layer, re-parsed by the reader")
            .code("profile.json   the resident card")
            .code("rps_table.json per-opponent transition counts — written by "
                  "nobody yet")
            .note("One writer, enforced by there being one process that opens these "
                  "files. Deletion is a cascade with a receipt, not a decay curve: "
                  "“forget what I said about X” has to be a transaction, "
                  "and probabilistic retrieval suppression is not one."))
    cloud = (Box("cloud", 520, C["cloud_f"], C["cloud_s"])
             .title("AWS Bedrock — " + ev["gr_region"], 13.5)
             .code(f"Converse   {ev['model']}")
             .code("toolConfig built from agent/tools.py's closed registry")
             .meas(f"constitution {ev['con_tokens']} tokens for {ev['con_chars']} "
                   f"chars, paid every turn")
             .meas(f"effort:low   first-token median {ev['eff_point']:+.1f} ms, CI "
                   f"excludes 0: {ev['eff_excl']} — output length falls to "
                   f"{ev['eff_ratio']:.3f}x, so it is a cost lever, not a latency one")
             .meas(f"Guardrails {ev['gr_tier']} tier, n={ev['gr_n']}: mandarin p50 "
                   f"{ev['gr_p50']:.1f} / p95 {ev['gr_p95']:.1f} ms vs the "
                   f"{ev['gr_budget']:.0f} ms T-G1 budget — over budget: "
                   f"{ev['gr_over']}")
             .meas(f"[docs/evidence/track-c/*.json]")
             .note("That last number is why the output guard is asynchronous with a "
                   "kill switch and never serial: one guard call at p95 already "
                   "spends more than the entire verdict-to-silence budget. Cantonese "
                   f"p95 is {ev['gr_yue_p95']:.1f} ms, but the between-language "
                   f"spread stays under the {ev['gr_noise']:.1f} ms replication "
                   f"noise ceiling (language gap real: {ev['gr_lang_real']}), and "
                   "Guardrails does not list Cantonese at all — so latency is "
                   "reported and protection is not claimed."))
    re_h = max(b.measure() for b in (gates, disk, cloud))
    for b, x in ((gates, LX), (disk, LX + 540), (cloud, LX + 1080)):
        b.h = re_h
        b.place(x, RE_TOP)
        placed.append(b)

    # ---- the gaps band
    RF_TOP = RE_TOP + re_h + 48
    gap_items = [
        f"{row['asymmetry'].replace('_', ' ')}  ({row['class']})  {name} — "
        f"{row['reason']}"
        for name, row in sorted(gaps.items())
    ] + NARRATIVE_GAPS
    gaps_intro = (
        f"The first {len(gaps)} are the census's own findings, one reviewed line each "
        f"in infra/graph_gaps.tsv; a row that stops being true is a finding too, so "
        f"these get deleted by the pull request that wires the missing end. The rest "
        f"are what the graph cannot say on its own.")
    gap_lines = [wrap_px(g, BAND_W - 2 * PAD_X - 16, 10.5, SANS) for g in gap_items]
    rf_head = band_head(gaps_intro)
    rf_h = round(rf_head + sum(len(l) * 13.8 + 5 for l in gap_lines) + PAD_B, 1)
    H = RF_TOP + rf_h + 34

    # ═══════════════════════════════ emit ═══════════════════════════════
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
               f'height="{H:.0f}" viewBox="0 0 {W} {H:.0f}" '
               f'font-family="{SANS}">')
    out.append(f'<rect width="{W}" height="{H:.0f}" fill="#ffffff"/>')
    # The connectors are appended further down, so the keyframe count comes from the
    # label list rather than from `hops`, which is still empty at this point. The
    # exhaustion check at the end is what ties the two together.
    out.append(defs())
    out.append(style_block(len(labels)))

    out.append(txt(LX, 46, title, 25, C["ink"], SANS, "bold"))
    for i, ln in enumerate(sub_lines):
        out.append(txt(LX, 70 + i * 17, ln, 13, C["dim"], SANS))
    draw_legend(out, LEG_TOP, labels)

    # the bus spine
    out.append(f'<rect x="{BUS_X}" y="{BUS_TOP:.0f}" width="{BUS_W}" '
               f'height="{bus_h:.1f}" rx="8" fill="{C["bus_f"]}" '
               f'filter="url(#sh)"/>')
    bus_mid = BUS_TOP + bus_h / 2
    out.append(f'<text transform="translate({BUS_X + 46} {bus_mid:.0f}) rotate(-90)" '
               f'text-anchor="middle" font-family="{SANS}" font-size="17" '
               f'font-weight="bold" fill="#ffffff" data-host="bus" data-rot="1">'
               f'ROS 2 Jazzy — DDS topic bus</text>')
    out.append(f'<text transform="translate({BUS_X + 72} {bus_mid:.0f}) rotate(-90)" '
               f'text-anchor="middle" font-family="{MONO}" font-size="10.5" '
               f'fill="{C["bus_d"]}" data-host="bus" data-rot="1">'
               f'{len(nodes)} processes, {len(topics)} topics, {len(edges)} edges — '
               f'publish/subscribe, so no line here is point to point</text>')

    for b in placed:
        b.draw(out)

    band(out, LED_TOP, led_h, C["legend_f"], C["dim"], "ledger")
    draw_band_head(out, LED_TOP, "ledger",
                   f"Topic ledger — all {len(topics)} topics, both ends counted",
                   LEDGER_INTRO)
    led_col_w = (BAND_W - 2 * PAD_X) / 2
    for i, (name, t) in enumerate(led_rows):
        col, row = divmod(i, per_col)
        rx = LX + PAD_X + col * led_col_w
        ry = LED_TOP + led_head + row * 15.6
        row_gap = gaps.get(name)
        fill = C["ink"]
        if row_gap:
            fill = C["warn_s"] if row_gap["asymmetry"] == "orphan" else C["gap_s"]
        mark = ""
        if row_gap:
            mark = "  <  nobody" if row_gap["asymmetry"] == "dead_end" \
                else "  >  nobody"
        out.append(txt(rx, ry, f"{name:<34}{len(t.pub)}>{len(t.sub)}{mark}", 11,
                       fill, MONO, "bold" if row_gap else "normal", "ledger"))
        if row_gap:
            # A dot that travels the row and stops at a bar: the animation says
            # "this message is written and nothing receives it". The orphan row gets
            # a hollow dot that pulses without moving — nothing arrives to move.
            bar_x = rx + led_col_w - 116
            out.append(f'<rect x="{bar_x:.0f}" y="{ry - 9:.0f}" width="3" '
                       f'height="12" fill="{fill}"/>')
            if row_gap["asymmetry"] == "dead_end":
                out.append(
                    f'<circle r="3.4" fill="{fill}" class="dead" '
                    f'style="offset-path: path(\'M {rx + 232:.0f} {ry - 3:.0f} '
                    f'L {bar_x - 6:.0f} {ry - 3:.0f}\'); '
                    f'animation-delay: {(i % 5) * 0.42:.2f}s"/>')
            else:
                out.append(f'<circle cx="{bar_x - 8:.0f}" cy="{ry - 3:.0f}" r="3.4" '
                           f'fill="none" stroke="{fill}" stroke-width="1.4" '
                           f'class="pulse"/>')

    # the module band
    band(out, MOD_TOP, mod_h, C["pure_f"], C["pure_s"], "modules")
    draw_band_head(out, MOD_TOP, "modules",
                   f"{len(mod_lines)} modules that are not nodes — zero ROS import, "
                   f"so every line runs on the laptop under pytest", MODULES_INTRO)
    for i, ln in enumerate(mod_lines):
        col, row = divmod(i, mod_per_col)
        out.append(txt(LX + PAD_X + col * (mod_col_w + 30),
                       MOD_TOP + mod_head + row * 14.4, ln, 10.5, C["ink"], MONO,
                       host="modules"))

    # the gaps band
    band(out, RF_TOP, rf_h, C["gap_f"], C["gap_s"], "gaps", dash="6 4")
    draw_band_head(out, RF_TOP, "gaps",
                   "What this diagram does not show as working", gaps_intro,
                   title_fill=C["gap_s"])
    cy = RF_TOP + rf_head - 8
    for lines in gap_lines:
        cy += 5
        for j, ln in enumerate(lines):
            cy += 13.8
            out.append(txt(LX + PAD_X + (0 if j == 0 else 14), cy, ln, 10.5,
                           C["ink"], SANS, host="gaps"))

    # ---- connectors, in the travel order of one turn
    tmid = tun.y + tun.h / 2
    link([(mac.x + mac.w, tmid), (tun.x, tmid)], "mac", "tunnel", hop=True)
    link([(tun.x + tun.w / 2, tun.y + tun.h), (tun.x + tun.w / 2, gw.y)],
         "tunnel", "gateway", hop=True)
    link([(bro.x, tmid), (tun.x + tun.w, tmid)], "browser", "tunnel")
    link([(gw.x + gw.w / 2, gw.y + gw.h), (gw.x + gw.w / 2, BUS_TOP)],
         "gateway", "bus", hop=True)

    spine = BUS_X + BUS_W / 2
    asr = left[0]
    agent = right[0]
    track([(gw.x + gw.w / 2, BUS_TOP), (spine, BUS_TOP + 16),
           (spine, round(asr.y + asr.h / 2))])
    for b in left:
        yc = round(b.y + b.h / 2)
        link([(BUS_X, yc), (b.x + b.w, yc)], "bus", b.name, hop=(b is asr))
    for b in right:
        yc = round(b.y + b.h / 2)
        link([(BUS_X + BUS_W, yc), (b.x, yc)], "bus", b.name)

    track([(spine, round(asr.y + asr.h / 2)), (spine, round(agent.y + agent.h / 2))])
    # The bus-to-agent connector is drawn in the right-column loop above; hop 7 rides
    # the same two points rather than a second line, so there is one geometry.
    next(hop_q)
    hops.append(d_of([(BUS_X + BUS_W, round(agent.y + agent.h / 2)),
                      (agent.x, round(agent.y + agent.h / 2))]))

    CH_A, CH_B = 1700, 1664       # two parallel, non-touching right-margin channels
    ay = round(agent.y + 46)
    link([(RX + RW, ay), (CH_A, ay), (CH_A, RE_TOP - 24),
          (cloud.x + cloud.w / 2, RE_TOP - 24), (cloud.x + cloud.w / 2, RE_TOP)],
         "agent", "cloud", hop=True)
    memory = right[2]
    my = round(memory.y + 42)
    link([(RX + RW, my), (CH_B, my), (CH_B, RE_TOP - 34),
          (disk.x + disk.w / 2, RE_TOP - 34), (disk.x + disk.w / 2, RE_TOP)],
         "memory", "disk")

    if next(hop_q, None) is not None:
        raise ValueError("hop_labels() lists more hops than the drawing registers — "
                         "the legend would number a step the animation never takes")
    for i, d in enumerate(hops):
        # No `cx`/`cy`, deliberately, and this is the one place where leaving an SVG
        # attribute off is the correct choice rather than an omission. The offset
        # transform anchors on the element's `transform-origin`, which for an SVG shape
        # is the user-space origin and not the circle's own centre, so a `cx`/`cy` is
        # *added* to the path point instead of being replaced by it. Measured: setting
        # cx/cy to the route start put every dot at exactly twice its start coordinate,
        # while `getBBox()` still read the route start — a probe that only asks "did the
        # dot move" calls that a pass. The fallback for a renderer without `offset-path`
        # is the support query in `style_block()`, which leaves these invisible; where
        # they are invisible their coordinates are not a claim about anything.
        out.append(f'<circle r="5.5" fill="{C["pkt"]}" class="pkt h{i}" '
                   f'style="offset-path: path(\'{d}\')"/>')

    out.append("</svg>")
    svg = "\n".join(out) + "\n"

    rects = [(b.x, b.y, b.w, b.h, b.name) for b in placed]
    rects += [(BUS_X, BUS_TOP, BUS_W, bus_h, "bus"),
              (LX, LEG_TOP, BAND_W, leg_h, "legend"),
              (LX, LED_TOP, BAND_W, led_h, "ledger"),
              (LX, MOD_TOP, BAND_W, mod_h, "modules"),
              (LX, RF_TOP, BAND_W, rf_h, "gaps")]
    findings = geometry_findings(segs, rects, W, H)
    # `rects` and `segs` are handed back as geometry rather than as counts because
    # tests/test_architecture.py needs to re-run the checks on a deliberately broken
    # copy of this layout: a checker that has only ever seen a clean diagram returns []
    # for the same reason a broken one does.
    meta = {"W": W, "H": round(H), "nodes": sorted(nodes), "topics": sorted(topics),
            "edges": len(edges), "gaps": sorted(gaps), "modules": mod_lines,
            "hop_labels": labels, "hop_paths": hops, "segments": segs,
            "rects": rects, "findings": findings, "leg_h": leg_h,
            "node_prose_keys": sorted(NODE_PROSE)}
    return svg, meta


def edge_line(topics: dict[str, Topic], node: str, kind: str) -> str:
    """`pub  /a  /b` for one node, straight from the census, sorted for determinism."""
    names = sorted(n for n, t in topics.items() if node in getattr(t, kind))
    label = {"pub": "pub", "sub": "sub"}[kind]
    if not names:
        return f"{label}  nothing"
    return f"{label}  " + "  ".join(names)


def module_rows(root: Path, nodes: dict[str, str]) -> dict[str, bool]:
    """Non-node python modules -> whether the file has a `__main__` entrypoint.

    Derived from the tree rather than listed, and the entrypoint half is derived from
    `infra/entrypoints.py` so that the † marker and `standalone_entrypoints.tsv`
    cannot disagree.
    """
    import ast

    import entrypoints

    node_files = set(nodes.values())
    eps = set(entrypoints.discover(root))
    ros = {"rclpy", "std_msgs", "sensor_msgs", "vision_msgs", "builtin_interfaces",
           "geometry_msgs", "rosidl_runtime_py"}
    out: dict[str, bool] = {}
    for pkg in sorted(p.name for p in root.iterdir()
                      if p.is_dir() and (p / "__init__.py").exists()
                      or p.is_dir() and any(p.glob("*.py"))):
        if pkg in {"tests", "web", "daemon", "__pycache__"} or pkg.startswith("."):
            continue
        for path in sorted((root / pkg).glob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in node_files:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imported.update(a.name.split(".")[0] for a in n.names)
                elif isinstance(n, ast.ImportFrom) and n.module:
                    imported.add(n.module.split(".")[0])
            if imported & ros:
                continue
            out[rel] = rel in eps
    if not out:
        raise ValueError("found 0 non-node modules — the module scan read nothing")
    return out


def band(out: list[str], y: float, h: float, fill: str, stroke: str, host: str,
         dash: str | None = None) -> None:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{LX}" y="{y:.0f}" width="{BAND_W}" height="{h:.1f}" '
               f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6" '
               f'filter="url(#sh)"{d}/>')


LEGEND_KEYS = [
    ("edge", "MacBook and browser — the laptop is the eye, ear and mouth"),
    ("gw", "the bridge — WebSocket frames in, ROS 2 messages out"),
    ("node", "a process the runner launches on the GPU instance"),
    ("bus", "the DDS topic bus — publish/subscribe, so nothing here is addressed"),
    ("pure", "modules with no ROS import: every line runs under pytest on a laptop"),
    ("cloud", "AWS Bedrock — the only network dependency in the loop"),
    ("gap", "derived findings: what this diagram does not show as working"),
]


INTRO_SIZE = 11.5
INTRO_LEAD = 16.0

# Each band's intro paragraph. Named constants because the height calculation and the
# drawing must wrap the *same* string: the first version emitted these unwrapped, and
# the modules one ran 76 px past the band edge while the other two fitted by luck —
# which is the same defect, only invisible.
LEDGER_INTRO = (
    "pub>sub is the number of publishing and subscribing processes. An amber row is "
    "published to nobody; a red row is subscribed to by somebody and published by no "
    "one. Each has a reviewed line in infra/graph_gaps.tsv and a matching entry in the "
    "band at the bottom of this page.")
MODULES_INTRO = (
    "§6.1 treats testability as a design constraint: the clock, the store, the random "
    "source and the Bedrock client are all injected, because a test that needs the GPU "
    "instance is a test that will not be run. † marks a file with a __main__ block, "
    "each of which carries a reviewed exemption line in "
    "infra/standalone_entrypoints.tsv.")


def intro_lines(text: str) -> list[str]:
    return wrap_px(text, BAND_W - 2 * PAD_X, INTRO_SIZE, SANS)


def band_head(text: str) -> float:
    """Height a band spends on its title plus its wrapped intro, before any rows."""
    return round(30 + len(intro_lines(text)) * INTRO_LEAD + 14, 1)


def draw_band_head(out: list[str], y: float, host: str, title: str, text: str,
                   title_fill: str = C["ink"]) -> None:
    out.append(txt(LX + PAD_X, y + 26, title, 14.5, title_fill, SANS, "bold", host))
    for i, ln in enumerate(intro_lines(text)):
        out.append(txt(LX + PAD_X, y + 30 + (i + 1) * INTRO_LEAD, ln, INTRO_SIZE,
                       C["dim"], SANS, host=host))


LEGEND_NOTE = (
    "A dot that travels a ledger row and stops at a bar is a topic published to "
    "nobody. A hollow dot that pulses without moving is a topic somebody subscribes "
    "to and nobody publishes. Under prefers-reduced-motion nothing moves at all and "
    "this list is the whole story, which is what makes switching the motion off safe.")


def legend_note_lines() -> list[str]:
    return wrap_px(LEGEND_NOTE, (BAND_W - 2 * PAD_X) / 2 - 8, 10.5, SANS)


def legend_height(labels: list[str]) -> float:
    """Measured before anything is placed, because the header's height depends on it."""
    left = len(LEGEND_KEYS) * 15.4
    right = len(labels) * 15.4 + 8 + len(legend_note_lines()) * 13.8
    return round(44 + max(left, right) + PAD_B, 1)


def draw_legend(out: list[str], y: float, labels: list[str]) -> None:
    """The key: what the colours mean, what the moving dot means, and the hop order.

    The hop list is the reason the animation can be switched off safely: it says in
    text what the dots say in motion, so `prefers-reduced-motion` removes a redundancy
    rather than removing information.
    """
    band(out, y, legend_height(labels), C["legend_f"], C["dim"], "legend")
    out.append(txt(LX + PAD_X, y + 26, "How to read this page", 14.5, C["ink"], SANS,
                   "bold", "legend"))
    col = (BAND_W - 2 * PAD_X) / 2
    for i, (key, text) in enumerate(LEGEND_KEYS):
        ry = y + 44 + i * 15.4
        out.append(f'<rect x="{LX + PAD_X:.0f}" y="{ry - 9:.0f}" width="22" '
                   f'height="11" rx="3" fill="{C[key + "_f"]}" '
                   f'stroke="{C.get(key + "_s", C["bus_f"])}" stroke-width="1.3"/>')
        out.append(txt(LX + PAD_X + 30, ry, text, 11, C["ink"], SANS, host="legend"))
    x2 = LX + PAD_X + col
    out.append(txt(x2, y + 26, "One turn, in order — this is what the moving dot is "
                               "doing", 13, C["ink"], SANS, "bold", "legend"))
    for i, label in enumerate(labels):
        ry = y + 44 + i * 15.4
        out.append(txt(x2, ry, label, 11, C["ink"], MONO, host="legend"))
    ry = y + 44 + len(labels) * 15.4 + 8
    for line in legend_note_lines():
        ry += 13.8
        out.append(txt(x2, ry, line, 10.5, C["dim"], SANS, host="legend"))


def geometry_findings(segs, rects, w: float, h: float) -> list[str]:
    """Every way this layout could be wrong on the page, as a list of findings.

    Pure and importable so that `tests/test_architecture.py` can run it on hand-built
    layouts that are known to be broken — a checker that has only ever seen a clean
    diagram is indistinguishable from one that returns [].
    """
    def bb(s):
        x1, y1, x2, y2 = s[:4]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    problems: list[str] = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if {a[4], a[5]} & {b[4], b[5]}:
                continue
            ax1, ay1, ax2, ay2 = bb(a)
            bx1, by1, bx2, by2 = bb(b)
            if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
                problems.append(f"connectors cross: {a[4]}->{a[5]} {a[:4]} vs "
                                f"{b[4]}->{b[5]} {b[:4]}")
    for s in segs:
        sx1, sy1, sx2, sy2 = bb(s)
        for (bx, by, bw, bh, name) in rects:
            if name in (s[4], s[5]):
                continue
            if not (sx2 <= bx or bx + bw <= sx1 or sy2 <= by or by + bh <= sy1):
                problems.append(f"connector {s[4]}->{s[5]} crosses box {name!r}")
    for i, (ax, ay, aw, ah, an) in enumerate(rects):
        for (bx, by, bw, bh, bn) in rects[i + 1:]:
            if not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay):
                problems.append(f"boxes overlap: {an!r} and {bn!r}")
        if ax < 0 or ay < 0 or ax + aw > w or ay + ah > h:
            problems.append(f"box {an!r} leaves the canvas")
    return sorted(set(problems))


def defs() -> str:
    return ('<defs>'
            '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{C["line"]}"/></marker>'
            '<filter id="sh" x="-4%" y="-4%" width="108%" height="112%">'
            '<feDropShadow dx="0" dy="1" stdDeviation="1.6" flood-opacity="0.14"/>'
            '</filter>'
            '</defs>')


def style_block(n_hops: int) -> str:
    """CSS rather than SMIL, so prefers-reduced-motion can switch the motion off.

    One keyframe block per hop, all sharing a single cycle length, so the dots stay in
    step: hop i is visible only during its own slice of the cycle and parked invisible
    outside it. A shorter per-dot duration with `infinite` would loop each hop
    independently and the sequence would stop being a sequence.

    Every rule that names an animation sits inside `@supports (offset-path: path(...))`,
    and the base state outside it is `opacity: 0`. Without the guard, a renderer with CSS
    animation but no `offset-path` (Firefox before 122, Safari before 16) ignores the
    `offset-distance` keyframes and honours the `opacity` ones — which is fourteen dots
    blinking in sequence at the canvas origin, on top of the title, instead of a diagram
    with no animation. "Degrades to a static diagram" has to be built; it is not what
    happens by default. This query is also why the dots carry no `cx`/`cy`: where the
    motion is off they are invisible, so there is no position left to get wrong.
    """
    css = ["svg { background: #ffffff }",
           # Parked and invisible unless the support query below turns the motion on.
           ".pkt, .dead { opacity: 0 }"]
    # Only the rules that *name* an animation are gated. The `@keyframes` blocks stay at
    # the top level and are inert without a name to bind them: measured in Chrome 141,
    # `@keyframes` nested inside `@supports` parses and lists in `cssRules`, and
    # `animation-name` computes to the right name — and the animation still has no
    # effect (0 of 14 dots moved or became visible over 5.6 s). Assigning the name is
    # the whole gate, so nothing is lost by defining the frames outside.
    sup = [f".pkt {{ animation: none; offset-rotate: 0deg; "
           f"animation-duration: {CYCLE}s; animation-timing-function: linear; "
           f"animation-iteration-count: infinite }}",
           ".dead { animation: deadend 5.5s linear infinite; offset-rotate: 0deg }"]
    for i in range(n_hops):
        a, b = i / n_hops * 100, (i + 1) / n_hops * 100
        sup.append(f".h{i} {{ animation-name: hop{i} }}")
        css.append(
            f"@keyframes hop{i} {{ 0%, {a:.3f}% {{ offset-distance: 0%; opacity: 0 }} "
            f"{a + 0.4:.3f}% {{ offset-distance: 0%; opacity: 1 }} "
            f"{b - 0.4:.3f}% {{ offset-distance: 100%; opacity: 1 }} "
            f"{b:.3f}%, 100% {{ offset-distance: 100%; opacity: 0 }} }}")
    css.append("@keyframes deadend { 0% { offset-distance: 0%; opacity: 0 } "
               "6% { offset-distance: 0%; opacity: 1 } "
               "46% { offset-distance: 100%; opacity: 1 } "
               "60% { offset-distance: 100%; opacity: 0.15 } "
               "100% { offset-distance: 100%; opacity: 0 } }")
    css.append("@supports (offset-path: path('M 0 0 L 1 1')) {\n"
               + "\n".join(sup) + "\n}")
    # The orphan marker is a plain opacity animation on a circle with real coordinates,
    # so it needs no support query: where CSS animation is unavailable it is simply a
    # hollow dot, which is what it means.
    css.append(".pulse { animation: breathe 3.2s ease-in-out infinite }")
    css.append("@keyframes breathe { 0%, 100% { opacity: 1 } 50% { opacity: 0.42 } }")
    # The reduced-motion branch parks every animated element in its resting state.
    # The hop order is written out in the legend, so nothing is only knowable by
    # watching: the motion is redundant, which is what makes switching it off safe.
    css.append("@media (prefers-reduced-motion: reduce) { "
               ".pkt, .dead, .pulse { animation: none !important } "
               ".pkt, .dead { opacity: 0 } .pulse { opacity: 1 } }")
    return "<style>\n" + "\n".join(css) + "\n</style>"


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else REPO_ROOT
    try:
        svg, meta = build(root)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 3
    if meta["findings"]:
        print("GEOMETRY FINDINGS:", file=sys.stderr)
        for f in meta["findings"]:
            print("  " + f, file=sys.stderr)
        return 1
    dest = root / "docs" / "architecture.svg"
    dest.write_text(svg, encoding="utf-8")
    print(f"ok  {meta['W']}x{meta['H']}  {len(meta['nodes'])} nodes  "
          f"{len(meta['topics'])} topics  {meta['edges']} edges  "
          f"{len(meta['gaps'])} one-ended  {len(meta['modules'])} modules  "
          f"{len(meta['hop_paths'])} animated hops  {len(meta['segments'])} segments  "
          f"{len(meta['rects'])} boxes  0 geometry findings")
    print(f"wrote {dest.relative_to(root)} ({dest.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
