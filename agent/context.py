"""agent/context.py — the one place a system prompt is assembled.

Zero ROS, zero boto3, zero clock, zero filesystem of its own. Everything here is a
pure function of its arguments, so the prompt the robot actually sends is a thing a
laptop can build and compare byte-for-byte.

**Why this module exists at all, stated as the defect it closes.**
`safety/guard.py` has held the prompt seam since S2, and `tests/test_guard.py`
proved its structural half: `compose_system(base, con, anything)` starts with
`instruction_region(base, con)` for every `anything`. What an AST census of the
module's fourteen public names against the production files then showed is that
thirteen of them had **no caller anywhere outside the tests** — including
`load_constitution`, `instruction_region`, `data_region` and `compose_system` —
and that `agent/agent_node.py` did not import `safety.guard` at all. Its system
prompt was `SYSTEM + loader.l1_index(self.skills)`, a bare concatenation with the
constitution nowhere in it. So the seam was correct and offline: a boundary that
exists in a module nothing launches is a boundary the robot does not have. T-C5's
pass criterion is about *the prompt that is sent*, so the wiring has to come before
any assertion about injection can mean anything.

**Why the region a value lands in is decided by the parameter it arrives through.**
`assemble()` takes the constitution and the skills index positionally/by keyword as
*instruction* material and the card and recall results as *data*. There is no
parameter that lets a caller choose. That is the whole design: a caller holding a
recalled `Hit` has no argument to put it in that reaches the instruction region, so
the failure mode T-C5 names — recalled text becoming a rule — is not prevented by a
review or by an assertion downstream, it is unreachable from the signature.
`instruction_region()` then independently refuses text carrying a fence marker or
the data preamble, so even our own material cannot smuggle a block in.

**Why the block names are a closed tuple.**
`guard.data_region()` derives each fence name as `re.sub(r"\\W", "", name)[:32]`, so
`card` and `c-a-r-d` used to collapse to the same name, as did any two names sharing
their first 32 word characters. Two blocks with one name leave the reader unable to
say where either ends, which is the one property a fence is for. Two fixes were
possible and both landed: `data_region` now *refuses* a collision (the invariant is
created at the producer, not observed at a consumer), and the names in this module
come from `BLOCKS` — already fence-safe, already distinct, and not supplied by any
caller. `_put()` refuses an unlisted name rather than dropping it, because a block
that silently disappears is a prompt missing memory with nothing to say so.

**What a finding is, and why it is returned instead of written into the prompt.**
A degraded turn has to be visible to the audit log, not described to the model. If a
retrieval path did not run, telling the model "the semantic path was unavailable"
spends tokens teaching it about our plumbing and invites it to talk about it;
`findings` goes to `get_logger()` and to `guard.audit_record`. This is also T-F3's
rule in the place that can enforce it: when recall degrades, the turn continues on
the card alone and says so in the log.

**What is deliberately not assembled here.**
The `<scene>` detections block is attached to the *user* turn by `agent_node`, not
to the system prompt, and this change does not move it. It is data either way, but
it is per-turn data that rides with the image it describes, and relocating it would
change deployed behaviour that only the machine can test. The RPS table has no
subscriber in `agent_node` at all, so there is no block for it: a fence name with no
producer is the same empty gesture as a cap on a parameter nothing can set.

**What the card and recall arguments are, today.**
Both default to `None`, and `agent_node` passes neither yet — the `/memory/recall`
client and the `transient_local` card subscription are S5's node half, which needs
the machine. They are not placeholder attributes on a node; they are parameters of a
pure function, exercised by `tests/test_context.py` against a real `Retriever` over
a real store. That is exactly the shape T-C5 asks for (D+A, offline).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

# Same one-module-identity argument as safety/guard.py and game/game_node.py:
# run_ec2.sh launches nodes by file path, so sys.path[0] is the node's directory.
# Without this, `agent.context` imported from the node and from the tests would be
# two modules holding two different BASE_PROMPTs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import profile as _profile   # noqa: E402
from memory import recall as _recall     # noqa: E402
from safety import guard as _guard       # noqa: E402


class ContextError(_guard.GuardError):
    """Raised when a prompt cannot be assembled — never when memory is merely thin.

    A subclass of `GuardError` on purpose: refusing to assemble a prompt and
    refusing to establish a boundary are the same class of failure, and a caller
    that decided to catch one has already decided about the other.
    """


BASE_PROMPT = """你是 open-agent-robot,一個開源虛擬人形機器人(Unitree G1 模擬體)。
你透過使用者 MacBook 的鏡頭看世界、麥克風聽、揚聲器說話,身體顯示在瀏覽器裡。
語言規則:**嚴格跟隨使用者的語言**——普通話問就用普通話(標準書面用字,絕不混粵語字),
英文問就用英文;只有使用者明顯講粵語時才用粵語詞彙(注意:語音是普通話引擎,粵語字會被讀成普通話音,所以粵語回覆盡量精簡)。
回答簡短口語化(閒聊 1-3 句;講故事/教學時可以長,但用短句、適合朗讀)。
你能看到當前鏡頭畫面(隨訊息附上的圖像)——使用者舉起任何東西問「這是什麼」,直接看圖回答。
你可以在回答文字的任意位置嵌入動作標籤,身體會同步執行(標籤不會被讀出來):
[gesture:wave]=揮手 [gesture:nod]=點頭 [gesture:rock]=握拳 [gesture:paper]=張開手掌
[gesture:scissors]=比剪刀 [gesture:idle]=放下手
遊戲規則(嚴格遵守):只有使用者「明確主動要求」玩石頭剪刀布時才嵌入 [game:start];
一次回覆最多一個 [game:start];絕不自己提議連續開局;局間聊天不要慫恿再玩。
使用者說「停/不玩/stop」時遊戲已被系統強制停止,你只需簡短確認,絕不重新開局。
當前鏡頭的即時物體檢測附在 <scene> 標籤(物名×數量×置信度,英文類名);
回答「看到什麼」時把它自然地翻成使用者的語言描述出來,person 通常就是眼前的使用者;
scene 為空就老實說現在沒看清,不要編造。
注意:語音識別會有錯字,按發音猜測使用者的真實意思,隨機應變。"""

# The closed set of data block names. Fence names come from here and nowhere else.
# Each one survives `re.sub(r"\W", "", name)[:32]` unchanged and they are pairwise
# distinct — asserted in tests/test_context.py rather than assumed, because that is
# the property that makes two blocks two blocks.
CARD_BLOCK = "card"
RECALL_BLOCK = "recall"
BLOCKS = (CARD_BLOCK, RECALL_BLOCK)

# The retrieval path whose material belongs in the card block rather than among the
# recalled lines. A card-path hit *is* a card fact — `recall._run_card` renders it
# with `Fact.line()`, the same producer the resident card uses — so labelling it
# `[card]` in the recall block would put the same sentence in the prompt twice under
# two different framings.
CARD_PATH = "card"

# `Result.hits(k)` applies `k` per path, never to the concatenation. Reusing
# recall's own cap keeps one ranker in the tree: a second sort here would be a
# second implementation of "which five", and the two would drift apart silently.
HIT_CAP = _recall.TOP_K

# What `guard._fence_safe` will silently cut. Read, not redefined, so the number the
# finding quotes is the number the fence uses.
BLOCK_MAX_CHARS = _guard.DATA_MAX_CHARS


class Assembly:
    """One assembled prompt, its block names, and the findings for the audit.

    Immutable for the same reason `recall.Hit` is: this object is handed to the
    Converse call and to the audit record, and an edit between those two readers
    would make the log disagree with what was sent, with nothing anywhere to say so.
    """

    __slots__ = ("prompt", "blocks", "findings")

    def __init__(self, prompt: str, blocks: tuple[str, ...],
                 findings: tuple[str, ...]) -> None:
        object.__setattr__(self, "prompt", str(prompt))
        object.__setattr__(self, "blocks", tuple(blocks))
        object.__setattr__(self, "findings", tuple(findings))

    def __setattr__(self, name: str, value: Any) -> None:
        raise ContextError(
            f"an Assembly is immutable ({name}={value!r}): it is read by the Converse "
            "call and by the audit record, and an edit between them would make the "
            "log a description of a prompt that was never sent")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def as_dict(self) -> dict:
        """The audit view. The prompt is *not* in it — see guard.audit_record."""
        return {"blocks": list(self.blocks), "findings": list(self.findings),
                "chars": len(self.prompt)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Assembly({len(self.prompt)} chars, blocks={self.blocks}, "
                f"{len(self.findings)} finding(s))")


def _put(data: dict[str, str], name: str, text: str,
         findings: list[str]) -> None:
    """Add one data block, refusing an unlisted name and reporting a coming cut.

    The name check cannot fire on today's two call sites — that is the point. It
    fires on the third one somebody adds, and it fires at the producer rather than
    letting `data_region` sanitise an unreviewed name into a fence.
    """
    if name not in BLOCKS:
        raise ContextError(
            f"{name!r} is not one of the data blocks {BLOCKS} — add it there, with a "
            "name that survives guard.data_region's sanitiser unchanged, instead of "
            "letting a new fence name into the prompt unreviewed")
    if not text.strip():
        return
    if len(text) > BLOCK_MAX_CHARS:
        findings.append(
            f"block {name!r} is {len(text)} chars and the fence will cut it at "
            f"{BLOCK_MAX_CHARS}; the tail was dropped from the prompt, not from memory")
    data[name] = text


def _degradation_findings(result: _recall.Result) -> list[str]:
    """One finding per path that did not run, named with the status it reported.

    Derived from `recall.paths()` rather than from a list here: a sixth retrieval
    path must arrive already covered, because a written-down set cannot notice a new
    member and a path nobody checks is a path whose silence looks like "no hits".
    """
    out: list[str] = []
    for name in _recall.paths():
        status = result.status_of(name)
        if status != _recall.RAN:
            out.append(
                f"recall path {name!r} did not run ({status}) — this turn answered "
                "from less than the whole memory")
    return out


def _cap_findings(result: _recall.Result, kept: tuple[_recall.Hit, ...],
                  hit_cap: int) -> list[str]:
    """What the per-path cap dropped, per path, or nothing if it dropped nothing.

    Reported rather than trimmed silently, because "recall found nothing about the
    peanut allergy" and "recall found it sixth" are different states and the second
    one is fixable.
    """
    every = result.hits(None)
    if len(every) <= len(kept):
        return []
    per_path: dict[str, int] = {}
    for hit in every:
        per_path[hit.path] = per_path.get(hit.path, 0) + 1
    per_kept: dict[str, int] = {}
    for hit in kept:
        per_kept[hit.path] = per_kept.get(hit.path, 0) + 1
    detail = ", ".join(
        f"{name}:{per_path[name] - per_kept.get(name, 0)}"
        for name in _recall.paths()
        if per_path.get(name, 0) > per_kept.get(name, 0))
    return [f"the {hit_cap}-per-path cap dropped {len(every) - len(kept)} hit(s) "
            f"({detail}) before the prompt was built"]


def _fresh(text: str, seen: set[str]) -> bool:
    """True the first time a text is offered for the prompt; mutates `seen`.

    The card path emits one hit per source episode on purpose — a cascade delete has
    to see every attribution — and the structured and lexical paths routinely find
    the same sentence. That is right for the deletion check and waste in a prompt:
    the same fact twice is paid for twice and reads to the model as emphasis nobody
    intended. Deduplication happens here, on the way into the prompt, and never on
    the `Hit` objects, which the caller still holds with every attribution intact.
    First offer wins, so the winning label is the earliest path in `recall.paths()`
    and the result is order-stable rather than dependent on a set's iteration.
    """
    if text in seen:
        return False
    seen.add(text)
    return True


def hit_line(hit: _recall.Hit) -> str:
    """One recalled item as one line of data, carrying which path found it.

    The path is in the line and the episode id is not: the id is 24 characters of
    provenance the model can do nothing with, while the path is the difference
    between "you told me this" and "the lexical index matched two characters of it".
    Provenance for deletion lives on the `Hit`, which the caller still holds.
    """
    return f"[{hit.path}] {hit.text}"


def assemble(constitution: str, *, skills_index: str = "",
             card: _profile.ProfileCard | None = None,
             recall: _recall.Result | None = None,
             hit_cap: int = HIT_CAP) -> Assembly:
    """Build the system prompt. Instructions from our files, memory as fenced data.

    `constitution` and `skills_index` are instruction material — both come from
    files in this repository, and `guard.instruction_region` refuses either of them
    if it carries a fence marker or the data preamble, so a hostile `SKILL.md`
    cannot open a block. `card` and `recall` are data, and there is no argument
    through which they could become anything else.

    Returns an `Assembly`. `findings` is for the audit log, never for the prompt.
    """
    if not isinstance(constitution, str) or not constitution.strip():
        raise ContextError(
            "refusing to assemble a system prompt with an empty constitution: the "
            "robot would run with every boundary clause silently absent, which is "
            "the state safety.guard.load_constitution exists to make impossible")
    if int(hit_cap) < 1:
        raise ContextError(f"hit_cap={hit_cap!r} would put no memory in the prompt "
                           "while reporting a healthy turn")

    findings: list[str] = []
    data: dict[str, str] = {}
    kept: tuple[_recall.Hit, ...] = ()

    if recall is not None:
        findings.extend(_degradation_findings(recall))
        kept = recall.hits(int(hit_cap))
        findings.extend(_cap_findings(recall, kept, int(hit_cap)))

    # The card block is the resident card plus whatever the card path recalled, in
    # that order and deduplicated — a union, not a choice. The two disagree in
    # normal operation: the resident card arrives from `memory_node` over a
    # `transient_local` topic while the card path reads `profile.json` off the disk,
    # so a fact written after the last publish exists on one side only. "Resident
    # wins" would drop it from the prompt with nothing anywhere saying a fact had
    # been withheld.
    #
    # Building this block *first* is also what keeps card material out of the recall
    # block: it claims those texts, and `_fresh` then refuses them below. That is
    # one mechanism instead of a filter here and a filter there, and it is why the
    # `==` in the loop is a mutation target — flip it and card facts reappear under
    # a `[card]` prefix among the recalled lines.
    seen: set[str] = set()
    card_lines: list[str] = []
    if card is not None:
        card_lines.extend(line for line in card.render().splitlines()
                          if _fresh(line, seen))
    card_lines.extend(hit.text for hit in kept
                      if hit.path == CARD_PATH and _fresh(hit.text, seen))
    _put(data, CARD_BLOCK, "\n".join(card_lines), findings)

    _put(data, RECALL_BLOCK,
         "\n".join(hit_line(hit) for hit in kept if _fresh(hit.text, seen)),
         findings)

    prompt = _guard.compose_system(BASE_PROMPT + str(skills_index),
                                   constitution, data)
    return Assembly(prompt, tuple(n for n in BLOCKS if n in data), tuple(findings))


def audit_payload(assembly: Assembly, extra: Mapping[str, Any] | None = None) -> dict:
    """The composed-turn record `agent_node` hands to the single writer, kind `chat`.

    **This docstring used to say "hands to `guard.audit_record`", and that consumer
    cannot accept it.** `store.check_audit` selects the declared field set by `kind`;
    this record's row is `AUDIT_FIELDS["chat"] == ("blocks", "chars", "findings")`,
    while `guard.audit_record` stamps `kind="audit"` and mandates
    `action`/`decision`/`reason`. Composing the two is refused at the append in both
    directions — leave the stamped kind and these three fields are undeclared, override
    it to `"chat"` and the three mandated ones are. `tests/test_profile_wiring.py`
    asserts both refusals, so the sentence cannot drift back.

    The caller therefore passes `{"kind": "chat"}` in `extra` and publishes the result on
    `/memory/audit`, where memory_node appends it. Two things followed the wrong
    sentence and are corrected in the same PR: the `AUDIT_FIELDS` comment that quoted it
    and the `dormant_seams.tsv` row that quoted it as this seam's caller.
    """
    out = assembly.as_dict()
    if extra:
        out.update(dict(extra))
    return out
