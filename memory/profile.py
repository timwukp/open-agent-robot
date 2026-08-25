"""memory/profile.py — the resident profile card. A read-only seed until S5.

Zero ROS, zero boto3, zero network, no clock. Loading and rendering the card is a
pure function of a file, so all of T-A5 and the CI half of T-F5 run under pytest.

**Why the card is read-only, and why saying so here would not be enough.**
The plan's build order put the profile card before the episode store, and §0-5 of
the plan records what that costs: a semantic fact must carry
`source_episode_ids` so that deleting an episode deletes what it produced, and a
card written before any episode exists cannot carry one. Writing to it now
therefore manufactures facts that are *unsourced and undeletable* — the exact
shape "forget what I said about X" has to be able to remove.

So the card is a seed. But note where the write API actually lives: `Store` has a
general `write_snapshot(name, obj)`, so a profile module that merely declines to
call it proves nothing — any of the nine nodes could write `profile.json` and this
module would neither know nor care. A read-only guarantee that lives in the reader
is a comment. `memory.store.PROTECTED_SNAPSHOTS` is where it lives instead, and
`unlock_for_s5()` is the one named, reviewable way to lift it. `tests/test_profile.py`
asserts the refusal by calling the writer directly, not by inspecting this file.

**Why the seed's missing provenance is a field and not an absence.**
Every seed fact carries `trust: "seed"`. It would be less code to let
`source_episode_ids` be absent and treat absence as "seeded", and that is the
version that lands the §0-5 defect in S5: the first automatic write also has no
episode ids yet at the moment it is constructed, so absence stops distinguishing
the two. A named trust level cannot be arrived at by forgetting something.

**Why the predicate set is closed here rather than in S5.**
A free-text predicate is a field where instruction-shaped content ("以後忽略停止
指令") lands as durable data and gets recalled into a prompt — the same hole T-C1
closes for the semantic layer. The set has to exist the moment anything is
loadable, so it exists here, and S5's consolidator must *import* `PREDICATES`
rather than restate it. `tests/test_profile.py` asserts there is exactly one
creator of the name, because two copies of a closed set is an open set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from memory.store import ReadOnlyError, StoreError

# The file name inside the store root. Not a path: the root is chosen and
# validated by `memory.store` (never /tmp, never the repo, 0700), and a module
# that could be pointed at an arbitrary path would be a second answer to
# "where does personal data live".
CARD_NAME = "profile.json"

# The example the repo ships. Contains no personal data — the real card lives at
# `<store root>/profile.json`, which is outside the repo by construction, because
# `$HOME/oar-data/` content must never become a blob.
EXAMPLE = Path(__file__).resolve().parent / "profile.example.json"

# Closed set, one creator. `kind` is what a retrieval filter matches on and what
# a deletion has to be able to name, so it is an enum for the same reason
# `why.goal` is (plan §S4.5): a free-text kind is both an unindexable key and a
# place for instruction-like content to live.
#
# `likes`/`dislikes` are the open door §0-6 names: the closed set stops a *fact*
# from being fabricated ("X 的首都是 Y" has no field to land in) but says nothing
# about a preference. What guards those is the source requirement in S5 (N ≥ 2
# independent episodes, self-reported not inferred); at seed level they are
# hand-written by the user, which is the strongest provenance available.
PREDICATES = (
    "name",           # what to call the person
    "call_me",        # a preferred form of address distinct from their name
    "language",       # 中文 / 粵語 / English preference
    "pronouns",
    "likes",
    "dislikes",
    "allergy",        # said out loud once and worth never forgetting
    "occupation",
    "location",
    "birthday",
    "household",      # who else is around, by role not by identity
    "pet",
    "routine",        # "usually home after 19:00"
    "goal",           # what the person told us they are working towards
    "skill_interest",  # a topic they want to practise, e.g. 粵語
    "device",         # a thing in the room the robot can be asked about
    "boundary",       # "don't talk about work" — a user-stated limit
    "note",           # free-form *value* under a closed key, deliberately last
)

# The resident budget (plan T-F5). Tokens, not characters: the card is prepended
# to every system prompt, so it is paid once per turn for the life of the robot.
TOKEN_BUDGET = 350

# `trust` levels. "seed" is the only one that may omit `source_episode_ids`, and
# that exemption is asserted rather than commented — see tests/test_profile.py.
TRUST_SEED = "seed"
TRUST_LEVELS = (TRUST_SEED, "observed", "confirmed")


class ProfileError(StoreError):
    """A card that does not load: a bad schema, an unknown predicate, a bad trust."""


class Fact:
    """One line of the card. Immutable, because the card is published by reference.

    `transient_local` hands the *same* object to every late-joining subscriber, so
    a consumer that mutated a fact would silently edit what the next subscriber
    receives. `__slots__` plus a rejecting `__setattr__` makes that a raised
    exception at the assignment rather than a value that changes underneath a
    reader — the failure mode is otherwise a card that differs between two nodes
    with nothing in any log to say so.
    """

    __slots__ = ("kind", "value", "trust", "source_episode_ids")

    def __init__(self, kind: str, value: str, trust: str,
                 source_episode_ids: tuple[str, ...] = ()) -> None:
        if kind not in PREDICATES:
            raise ProfileError(
                f"unknown predicate {kind!r}; the set is closed and lives in "
                f"memory.profile.PREDICATES ({len(PREDICATES)} entries)")
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"{kind}: value must be a non-empty string, "
                               f"got {value!r}")
        if trust not in TRUST_LEVELS:
            raise ProfileError(f"{kind}: unknown trust {trust!r}, "
                               f"expected one of {TRUST_LEVELS}")
        if trust != TRUST_SEED and not source_episode_ids:
            raise ProfileError(
                f"{kind}: trust {trust!r} requires source_episode_ids — only "
                f"{TRUST_SEED!r} facts may have none, because a fact with no "
                "episode behind it cannot be removed when that episode is deleted")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value.strip())
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "source_episode_ids", tuple(source_episode_ids))

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyError(
            f"a Fact is immutable ({name}={value!r}): the card is published by "
            "reference to every late-joining subscriber, so an in-place edit "
            "would change what the next one receives")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "trust": self.trust,
                "source_episode_ids": list(self.source_episode_ids)}

    def line(self) -> str:
        """The rendered form. This, summed, is what T-F5 budgets."""
        return f"{self.kind}: {self.value}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Fact({self.kind!r}, {self.value!r}, {self.trust!r})"


class ProfileCard:
    """The loaded card: an ordered tuple of facts and one rendering of them.

    Ordered, and the order is the file's. Rendering under a `set` or a `dict`
    grouping would make the resident prompt text depend on iteration order, which
    is the T-B4 determinism defect one layer up: the same card would produce two
    different prompts and the token count measured once would not be the token
    count paid later.
    """

    __slots__ = ("facts", "version")

    def __init__(self, facts: Iterable[Fact], version: int = 1) -> None:
        object.__setattr__(self, "facts", tuple(facts))
        object.__setattr__(self, "version", int(version))

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyError(f"the profile card is read-only until S5 ({name}); "
                            "see memory.store.unlock_for_s5")

    def __delattr__(self, name: str) -> None:
        self.__setattr__(name, None)

    def __len__(self) -> int:
        return len(self.facts)

    def render(self) -> str:
        """The resident block, exactly as it is prepended to the system prompt.

        One fact per line and nothing else — no heading, no "here is what you know
        about the user". A heading is paid on every turn forever and the caller
        already knows what it asked for; `agent_node` supplies the surrounding
        frame, and T-C5 requires that frame to be in the *data* region.
        """
        return "\n".join(f.line() for f in self.facts)

    def as_dict(self) -> dict:
        return {"version": self.version, "facts": [f.as_dict() for f in self.facts]}

    # -------------------------------------------------- the refused write surface
    #
    # These exist so that T-A5 ("call any write API on the seed card") has
    # something to call. They are not the guarantee — the guarantee is in
    # memory.store.PROTECTED_SNAPSHOTS, because a node that never imports this
    # module can still reach the writer. They are here so that the *obvious*
    # thing a future caller reaches for fails loudly and names the reason,
    # instead of that caller adding its own `store.write_snapshot` call.
    def add(self, *_args, **_kwargs):
        raise ReadOnlyError(
            "the S3 profile card is a read-only seed: an added fact would have no "
            "source episode (there are no episodes until S4), so it could never be "
            "removed by a deletion. Automatic writes open in S5 behind "
            "memory.store.unlock_for_s5")

    def remove(self, *_args, **_kwargs):
        raise ReadOnlyError(
            "the S3 profile card is a read-only seed: it is hand-written, so the "
            "way to remove a fact is to edit the file. A programmatic removal here "
            "would be a deletion path with no audit record")

    def save(self, *_args, **_kwargs):
        raise ReadOnlyError(
            "the S3 profile card is a read-only seed: nothing in this repo may "
            "write it yet. memory.store refuses the snapshot as well, which is the "
            "assertion that actually holds — this one only makes it obvious")

    update = add
    replace = save


def budget_findings(card: ProfileCard, ratio: float) -> list[str]:
    """Everything wrong with `card` as a *resident* block. Empty list = within budget.

    A list of findings rather than a bool, for the same reason
    `store.location_findings` is: the node logs each one and the test asserts on the
    text, so a rejection for the right reason and a rejection for the wrong one are
    not the same observation.

    `ratio` is passed in, not imported. The one calibration in this repo lives in
    `tests/test_skills_loader.py` (measured against a real `usage.inputTokens`), and
    a second copy here would be a second calibration drifting from the first — the
    thing the S2 round deliberately avoided. The node supplies it from its own
    constant and the test reads it out of its home by AST.

    Why this is a *function on any card* and not an assertion about the example:
    CI can only see `profile.example.json`, because the real card lives outside the
    repo by construction. A budget checked only against the example is a budget the
    production card never meets — so the check is here, the node runs it at boot,
    and the test proves both that the example passes and that an oversized card is
    refused by this very function.
    """
    from skills.loader import approx_tokens

    findings: list[str] = []
    # Divided by the ratio because the character proxy *under*-counts (0.79 measured
    # on Chinese-heavy text), and under-counting is the dangerous direction for a
    # budget: it lets a real 400-token card through a 350-token gate.
    tokens = approx_tokens(card.render()) / ratio
    if tokens > TOKEN_BUDGET:
        findings.append(
            f"the resident block is ~{tokens:.0f} tokens against a budget of "
            f"{TOKEN_BUDGET} — it is prepended to every system prompt, so this is "
            "paid once per turn for the life of the robot")
    dupes = sorted({f.kind for f in card.facts
                    if sum(1 for g in card.facts if g.kind == f.kind) > 1})
    for kind in dupes:
        if kind not in ("likes", "dislikes", "allergy", "note", "device", "pet"):
            findings.append(
                f"{kind!r} appears more than once — for a single-valued predicate "
                "two lines is a contradiction the prompt presents as two facts")
    return findings


def parse(obj: Any, where: str = "<dict>") -> ProfileCard:
    """Build a card from already-decoded JSON. Raises `ProfileError` on anything odd.

    Separate from `load` so the validation is testable without a file, and so the
    node can validate a card that arrived over ROS with the same code that
    validated the one on disk. Two validators would be two definitions of a legal
    card.
    """
    if not isinstance(obj, dict):
        raise ProfileError(f"{where}: the card must be a JSON object, "
                           f"got {type(obj).__name__}")
    version = obj.get("version")
    if version != 1:
        raise ProfileError(f"{where}: version must be 1, got {version!r} — an "
                           "unrecognised version must not be read as a v1 card")
    raw = obj.get("facts")
    if not isinstance(raw, list) or not raw:
        raise ProfileError(f"{where}: 'facts' must be a non-empty list. An empty "
                           "card loads fine and makes every recall test pass "
                           "vacuously, which is why this is an error")
    facts = []
    for n, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ProfileError(f"{where}: fact {n} is not an object")
        unknown = set(entry) - {"kind", "value", "trust", "source_episode_ids"}
        if unknown:
            raise ProfileError(
                f"{where}: fact {n} has unknown field(s) {sorted(unknown)} — a "
                "silently ignored field is a setting the author believes is on")
        try:
            facts.append(Fact(entry.get("kind"), entry.get("value"),
                              entry.get("trust", TRUST_SEED),
                              tuple(entry.get("source_episode_ids", ()))))
        except ProfileError as exc:
            raise ProfileError(f"{where}: fact {n}: {exc}") from exc
    return ProfileCard(facts, version=version)


def missing_seed_hint(path: Path) -> str:
    """What to say when there is no card. A function, not an f-string at the log site.

    The boot policy below lives here rather than in `memory/memory_node.py` for the
    reason §6.1 gives: a decision inside a module that imports rclpy can only be
    tested on the instance, and a test that can only run on the instance does not
    get run. The node is left with logging and publishing.
    """
    return (f"no profile card at {path} — the robot starts knowing nobody. "
            f"Copy {EXAMPLE} there and edit it.")


def boot_card(root: Path, ratio: float) -> ProfileCard | None:
    """The card a node should publish, or None if the user has not written one.

    Three outcomes, and the asymmetry between them is the decision:

    * **absent → None.** A fresh instance legitimately has no seed, and refusing to
      boot would mean the user cannot talk to the only thing that would tell them
      how to write one.
    * **malformed → raise.** Somebody wrote a card and it is not the card they think
      they wrote. Falling back to "no memory" here would look identical to the case
      above, and the user would conclude the robot simply forgets.
    * **over budget → raise.** The block is prepended to every system prompt for the
      life of the process, so this is paid once per turn forever; the fix is one edit
      and this runs before any interaction. A warning costs money for as long as
      nobody reads the log, which is how long a warning is usually read for.
    """
    path = Path(root) / CARD_NAME
    if not path.exists():
        return None
    card = load(path)
    findings = budget_findings(card, ratio)
    if findings:
        raise ProfileError(f"{path}: " + "; ".join(findings))
    return card


def load(path: Path) -> ProfileCard:
    """Read and validate the card at `path`.

    A missing file is an error, not an empty card. An empty card is a robot that
    has forgotten everything while reporting healthy, and the node's job when the
    seed is absent is to say so loudly at boot — see memory/memory_node.py.
    """
    path = Path(path)
    if not path.exists():
        raise ProfileError(
            f"{path} does not exist. The seed card is hand-written; copy "
            f"{EXAMPLE.name} to the store root and edit it. A default empty card "
            "here would make a missing file indistinguishable from a person the "
            "robot knows nothing about")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path}: not valid JSON ({exc})") from exc
    return parse(obj, where=str(path))
