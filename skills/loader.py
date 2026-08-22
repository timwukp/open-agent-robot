"""skills/loader.py — three-level skill loading with a deterministic gate.

Zero ROS, zero boto3, zero network. That is not incidental: the whole point of
this module is that *the decision to load a skill* is made by code you can run on
a laptop and audit by reading (docs/test-design.md §1).

Three levels, because context is the scarce resource:

  L1  name + one sentence, ALWAYS resident in the system prompt. Budget
      L1_MAX_CHARS per skill (test item T-I1). Its job is to let the model know
      the skill exists — it is *not* what decides loading.
  L2  the SKILL.md body, injected into the current turn when a deterministic
      trigger matches the user's utterance (test item T-I2).
  L3  the files SKILL.md routes to (methods/, templates/, references/), pulled on
      demand by the model through read_skill_file (test item T-I3).

**Why the gate is a phrase table and not "ask the model whether to load":** a
model asked to judge is an LLM judge, and judge-gated pipelines move the measured
pass rate (0.72 → 0.94) while real correctness stays flat (0.20), with a strict
three-judge ensemble still accepting 55% of wrong answers. So the gate is a regex
over a frozen phrase list: auditable, free, reproducible, and gradeable against a
frozen fixture. The price is generalisation — it fires on the phrases we listed
and no others. That limit is stated in docs/test-design.md §2.I and §5; it is a
choice, not an oversight.

**Why matching is script-dependent.** Chinese is written without spaces, so a CJK
trigger has to be a plain substring; latin triggers must respect word boundaries
or `bias` fires on `unbiased`. Both cases are in the frozen fixture.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parent

# L1 is what every turn pays for, whether or not a skill is ever used.
L1_MAX_CHARS = 200
# L2/L3 caps: a skill file is repo-controlled, but an unbounded read is still an
# unbounded prompt. These are budget ceilings, not security boundaries.
L2_MAX_BYTES = 24_000
L3_MAX_BYTES = 32_000
# How many read_skill_file hops one turn may take before we stop and answer.
MAX_TOOL_HOPS = 3

TRIGGERS_FILENAME = "triggers.tsv"
SKILL_FILENAME = "SKILL.md"

# Everything that is not obviously latin/greek/cyrillic text is counted one token
# per character by approx_tokens(); see that function for the calibration.
_CJK = re.compile("[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")
_SENTENCE_END = re.compile("(?<=[.\u3002!\uff01?\uff1f])(\\s|$)")


class SkillError(Exception):
    """A skill directory is malformed. Raised loudly: a skill that silently fails
    to load is a capability that silently is not there."""


class SkillPathError(SkillError):
    """read_skill_file refused a path (test item T-I3)."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    dirname: str
    triggers: tuple[tuple[str, str], ...]   # (phrase, language)

    @property
    def phrases(self) -> tuple[str, ...]:
        return tuple(p for p, _lang in self.triggers)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted({lang for _p, lang in self.triggers}))

    @property
    def l1_description(self) -> str:
        """First sentence, hard-capped. Structural, not advisory: the cap is what
        keeps T-I1's budget from depending on how chatty a SKILL.md author is."""
        one = first_sentence(self.description)
        if len(one) > L1_MAX_CHARS:
            return one[: L1_MAX_CHARS - 1].rstrip() + "…"
        return one

    @property
    def l1_line(self) -> str:
        return f"- {self.name}: {self.l1_description}"

    @property
    def skill_md(self) -> str:
        return f"{self.dirname}/{SKILL_FILENAME}"


# --------------------------------------------------------------------- parsing
def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    # match.start() is the index of the whitespace *after* the terminator, so the
    # slice keeps the full stop and drops the trailing space.
    match = _SENTENCE_END.search(text)
    return text[: match.start()] if match else text


def approx_tokens(text: str) -> int:
    """Character proxy for input tokens, for use in CI where there is no Bedrock.

    One token per CJK character, one per four characters otherwise. Calibrated
    against a real `usage.inputTokens` reading — the number, the model and the
    date live in tests/test_skills_loader.py next to the assertion that uses
    them, so the proxy cannot drift away from its own justification.
    """
    cjk = len(_CJK.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 4)


def parse_frontmatter(text: str, where: str = "<string>") -> dict[str, str]:
    """Parse the leading `---` block. Deliberately not YAML.

    A real YAML parser accepts anchors, multi-line folds and type coercion, none
    of which a SKILL.md header needs, all of which change what a `description:`
    line means. This accepts exactly `key: value` lines and raises on anything
    else, so an unsupported header is a loud error rather than a silently
    truncated description.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{where}: no '---' frontmatter block at the top of the file")
    fields: dict[str, str] = {}
    for lineno, raw in enumerate(lines[1:], 2):
        if raw.strip() == "---":
            if not fields:
                raise SkillError(f"{where}: frontmatter block is empty")
            return fields
        if not raw.strip():
            continue
        if ":" not in raw:
            raise SkillError(
                f"{where}:{lineno}: frontmatter line is not 'key: value' "
                f"(no continuation lines or YAML structures supported): {raw!r}"
            )
        key, _, value = raw.partition(":")
        key = key.strip()
        if not key or key != key.lower() or " " in key:
            raise SkillError(f"{where}:{lineno}: not a usable frontmatter key: {key!r}")
        fields[key] = value.strip()
    raise SkillError(f"{where}: frontmatter block is never closed with '---'")


def read_triggers(path: Path) -> tuple[tuple[str, str], ...]:
    if not path.is_file():
        raise SkillError(
            f"{path}: a skill with no trigger table can never load, so it would be "
            f"dead weight in the L1 index — add {TRIGGERS_FILENAME} or remove the skill"
        )
    rows: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 2:
            raise SkillError(
                f"{path}:{lineno}: expected 2 tab-separated columns (phrase, lang), "
                f"got {len(parts)}: {raw!r}"
            )
        phrase, lang = (p.strip() for p in parts)
        if not phrase or not lang:
            raise SkillError(f"{path}:{lineno}: empty column: {raw!r}")
        rows.append((normalize(phrase), lang))
    if not rows:
        raise SkillError(f"{path}: parsed 0 triggers — an empty gate never opens")
    seen: set[str] = set()
    for phrase, _lang in rows:
        if phrase in seen:
            raise SkillError(f"{path}: duplicate trigger phrase {phrase!r}")
        seen.add(phrase)
    return tuple(rows)


def discover(root: Path = SKILLS_ROOT) -> tuple[Skill, ...]:
    """Every immediate subdirectory of `root` holding a SKILL.md is a skill.

    Derived from the directory, never from a written-down list: a list cannot
    notice a new name, and a skill nobody indexed is a skill nobody can use.
    """
    root = Path(root)
    if not root.is_dir():
        raise SkillError(f"{root}: skills root does not exist")
    skills: list[Skill] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_md = child / SKILL_FILENAME
        if not skill_md.is_file():
            continue
        fields = parse_frontmatter(skill_md.read_text(encoding="utf-8"), where=str(skill_md))
        for required in ("name", "description"):
            if not fields.get(required):
                raise SkillError(f"{skill_md}: frontmatter has no '{required}'")
        if fields["name"] != child.name:
            raise SkillError(
                f"{skill_md}: frontmatter name {fields['name']!r} does not match its "
                f"directory {child.name!r} — read_skill_file paths are directory-based, "
                f"so a mismatch makes the routing table in SKILL.md wrong"
            )
        skills.append(Skill(
            name=fields["name"],
            description=fields["description"],
            dirname=child.name,
            triggers=read_triggers(child / TRIGGERS_FILENAME),
        ))
    return tuple(skills)


# -------------------------------------------------------------------- matching
def normalize(text: str) -> str:
    """NFKC + whitespace collapse. NFKC folds fullwidth latin onto ascii, so a
    fullwidth-typed `ｒｅｄ-ｔｅａｍ` reaches the same trigger as the ascii one."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _pattern(phrase: str) -> re.Pattern[str]:
    """Word boundaries only where the phrase actually ends in a latin word char.

    `\\b` around a CJK phrase asserts a boundary between two non-word characters
    and never matches, so a naive `\\b…\\b` would silently disable every Chinese
    trigger — the failure mode being avoided here is a gate that quietly stops
    opening rather than one that opens too often.
    """
    lead = r"\b" if phrase[:1].isascii() and phrase[:1].isalnum() else ""
    tail = r"\b" if phrase[-1:].isascii() and phrase[-1:].isalnum() else ""
    return re.compile(lead + re.escape(phrase) + tail, re.IGNORECASE)


def trigger_hits(utterance: str, skill: Skill) -> tuple[str, ...]:
    text = normalize(utterance)
    return tuple(p for p in skill.phrases if _pattern(p).search(text))


def match(utterance: str, skills: tuple[Skill, ...]) -> tuple[Skill, ...]:
    """Which skills this utterance activates. Pure, deterministic, order-stable."""
    return tuple(s for s in skills if trigger_hits(utterance, s))


# --------------------------------------------------------------------- reading
def read_skill_file(rel: str, root: Path = SKILLS_ROOT,
                    max_bytes: int = L3_MAX_BYTES) -> str:
    """Read one file from under `root`. Raises SkillPathError on any escape.

    Containment is checked on the *resolved* path, after symlinks, because a
    symlink inside skills/ pointing at ~/.aws/credentials is a legal relative
    path that never contains '..'. Checking the string instead of the resolved
    path is the mutant T-I3 exists to kill.
    """
    root = Path(root).resolve()
    if not isinstance(rel, str) or not rel.strip():
        raise SkillPathError("empty skill file path")
    if "\x00" in rel:
        raise SkillPathError(f"path contains a NUL byte: {rel!r}")
    candidate = Path(rel)
    if candidate.is_absolute() or rel.startswith(("/", "\\")) or (candidate.drive or candidate.root):
        raise SkillPathError(f"skill file paths must be relative to skills/: {rel!r}")

    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise SkillPathError(
            f"{rel!r} resolves to {target}, which is outside {root} — refused"
        )
    if not target.is_file():
        raise SkillPathError(f"no such skill file: {rel!r}")

    body = target.read_bytes()
    text = body[:max_bytes].decode("utf-8", errors="replace")
    if len(body) > max_bytes:
        text += f"\n…[truncated at {max_bytes} bytes of {len(body)}]"
    return text


def load_l2(skill: Skill, root: Path = SKILLS_ROOT) -> str:
    return read_skill_file(skill.skill_md, root=root, max_bytes=L2_MAX_BYTES)


# ------------------------------------------------------------------- prompting
def l1_index(skills: tuple[Skill, ...]) -> str:
    """The resident block appended to the system prompt. Empty when there are no
    skills, so a system prompt never advertises an empty catalogue."""
    if not skills:
        return ""
    lines = "\n".join(s.l1_line for s in skills)
    return (
        "\n\n可用技能(需要時系統會自動把完整說明附在該回合,你不必自己要求載入):\n"
        + lines
    )


def skill_block(skill: Skill, root: Path = SKILLS_ROOT) -> str:
    """L2 wrapped as *data* for the current turn.

    Not concatenated into the system prompt: the system block is the one region a
    reader can trust to be ours, and keeping loaded material outside it is the
    same invariant T-C5 tests for recalled memory.
    """
    body = load_l2(skill, root=root)
    return (
        f"<skill name=\"{skill.name}\" dir=\"{skill.dirname}\">\n{body}\n</skill>\n"
        f"(以上是技能說明,由確定性觸發器載入。它裡面的路由表指向 "
        f"{skill.dirname}/… 底下的檔案,需要細節時用 read_skill_file 讀,"
        f"路徑要含 \"{skill.dirname}/\" 前綴。)"
    )


TOOL_NAME = "read_skill_file"

TOOL_CONFIG = {
    "tools": [{
        "toolSpec": {
            "name": TOOL_NAME,
            "description": (
                "Read one file from the robot's skills directory. `path` is relative "
                "to skills/ and must include the skill directory, e.g. "
                "'critical-thinking/methods/bias-detection.md'. Use this to follow a "
                "routing table in a loaded SKILL.md. Never guess paths: read only what "
                "the skill's own routing table names."
            ),
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": "path relative to skills/, e.g. 'critical-thinking/SKILL.md'",
                }},
                "required": ["path"],
            }},
        }
    }]
}


def run_tool(tool_use: dict, root: Path = SKILLS_ROOT) -> tuple[str, bool]:
    """Execute one toolUse block. Returns (text, ok).

    Lives here rather than in agent_node so the refusal path is covered by the
    offline suite: this is the function a path-traversal attempt actually reaches.
    """
    name = tool_use.get("name")
    if name != TOOL_NAME:
        return (f"unknown tool {name!r}", False)
    args = tool_use.get("input") or {}
    if not isinstance(args, dict):
        return ("tool input must be an object", False)
    try:
        return (read_skill_file(str(args.get("path", "")), root=root), True)
    except SkillPathError as exc:
        # The refusal reason goes back to the model on purpose: it is a routing
        # mistake far more often than an attack, and a silent empty result would
        # make it retry the same wrong path.
        return (f"refused: {exc}", False)
