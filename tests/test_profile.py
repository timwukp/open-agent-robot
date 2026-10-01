"""Offline tests for the S3 seed profile card — T-A5, T-F5, and the §0-5 design law.

Three things here are not the literal shape docs/test-design.md §2.A asks for, and
the reasons are the part worth reading.

**T-A5 is asserted against the writer, not against the reader.** The plan says
"call any write API on the seed card → `ReadOnlyError`, file mtime unchanged". Taken
literally that is a test of `memory/profile.py`, which is the one place the property
cannot live: the write API is `Store.write_snapshot(name, obj)`, so a profile object
that merely refuses to offer a mutator constrains only the callers that went through
it. Any of the nine nodes can construct a `Store` and write `profile.json` without
importing this module. So the assertion that carries the claim calls
`Store.write_snapshot("profile.json", …)` directly. The refusing methods on
`ProfileCard` are tested too, but as a usability property — the obvious thing a
future caller reaches for fails loudly — not as the guarantee.

**"mtime unchanged" is not enough on its own.** `os.replace` inside the same
filesystem second on a coarse-granularity mount leaves an mtime that compares equal
while the bytes are new, which is the same shape as a stale `.pyc` being validated
by (length, second) instead of by content. Both are checked: the bytes and the stat.

**The mutant for the check's *position* needs a dead writer, not a dead disk.**
`write_snapshot` refuses a locked name *before* `_raise_if_failed()`, so that a
locked write is reported as a lock and not as a broken disk. Nothing observes that
ordering unless the writer has already failed, so the test sets the store's recorded
failure and asserts the two mutants report two different things: the real module
raises `ReadOnlyError`, the reordered one raises the writer-died `StoreError`. A
test that only checked "it raises" would pass on both, because `ReadOnlyError` is a
`StoreError` — the reordering would survive as a byte-identical verdict.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
PROFILE_PY = REPO_ROOT / "memory" / "profile.py"

from memory import profile as pf  # noqa: E402
from memory import store as st  # noqa: E402
from skills.loader import approx_tokens  # noqa: E402


@pytest.fixture(autouse=True)
def _relock():
    """`_UNLOCKED` is process-global, so an unlock in one test would leak into the next.

    Process-global is the right scope for the real system — one process is one robot,
    and S5 unlocks once at startup — but it makes the lock tests order-dependent
    unless they are undone. Snapshot and restore rather than `clear()`, so a stray
    unlock somewhere else in the suite is not silently absorbed by this fixture.
    """
    before = set(st._UNLOCKED)
    yield
    st._UNLOCKED.clear()
    st._UNLOCKED.update(before)


def make_store(tmp_path: Path, module=st, **kw):
    """Same visible exemption as tests/test_store.py: tmp_path is ephemeral."""
    return module.Store(tmp_path / "data", _allow_ephemeral=True, **kw)


# --------------------------------------------------------------------- mutants
def load_mutant(tmp_path: Path, target: str, *replacements: tuple[str, str]):
    """Import a text-patched copy of memory/profile.py or memory/store.py.

    Zero replacements gives the no-mutant control required by §6.0 rule 2. The
    example card is copied next to the patched profile module because
    `profile.EXAMPLE` is derived from `__file__`; without the copy a mutant would
    fail on a missing file and be scored as killed for the wrong reason.
    """
    dst = tmp_path / f"mut_{target}_{len(replacements)}"
    dst.mkdir(parents=True, exist_ok=True)
    src_path = REPO_ROOT / "memory" / f"{target}.py"
    src = src_path.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in {target}.py: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    out = dst / f"{target}.py"
    out.write_text(src)
    if target == "profile":
        shutil.copy2(pf.EXAMPLE, dst / pf.EXAMPLE.name)
    name = f"{target}_mutant_{tmp_path.name}_{len(replacements)}"
    spec = importlib.util.spec_from_file_location(name, out)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


LOCK_CHECK = ("        if is_locked(name):\n"
              "            raise ReadOnlyError(\n"
              '                f"{name!r} is a read-only seed until S5: a fact written now would "\n'
              '                "have no source episode (episodes arrive in S4) and no deletion "\n'
              '                "could ever reach it. memory.store.unlock_for_s5 is the one way past")\n'
              "        self._raise_if_failed()")
PROTECTED_SET = 'PROTECTED_SNAPSHOTS = frozenset({"profile.json"})'
IS_LOCKED_BODY = "    return name in PROTECTED_SNAPSHOTS and name not in _UNLOCKED"
UNLOCK_GUARD = "    if name not in PROTECTED_SNAPSHOTS:"
SOURCE_REQUIREMENT = "        if trust != TRUST_SEED and not source_episode_ids:"
INFLATE = "    tokens = approx_tokens(card.render()) / ratio"
RENDER_JOIN = '        return "\\n".join(f.line() for f in self.facts)'
EMPTY_FACTS = "    if not isinstance(raw, list) or not raw:"
UNKNOWN_FIELDS = "        unknown = set(entry) - "


# ------------------------------------------------------ the shipped example card
def test_the_example_card_loads_and_is_not_a_stub() -> None:
    """A floor, because every budget and render assertion below passes vacuously on
    a one-fact card. The plan sizes the seed at ~20 facts / ~300 tokens; 15 is the
    floor at which "it fits the budget" is a claim about a real card."""
    card = pf.load(pf.EXAMPLE)
    assert len(card) >= 15, f"the shipped example has only {len(card)} facts"
    assert pf.EXAMPLE.stat().st_size >= 900
    kinds = {f.kind for f in card.facts}
    assert kinds <= set(pf.PREDICATES)
    # The seed exemption applies to *every* fact in the shipped card. One
    # non-seed fact here would mean the repo ships provenance-free trust.
    assert all(f.trust == pf.TRUST_SEED for f in card.facts)
    assert all(f.source_episode_ids == () for f in card.facts)


def test_the_example_card_still_holds_a_placeholder_and_not_a_real_person() -> None:
    """infra/redact_scan.py knows instance ids and IPv4; it cannot know a name.

    The one mechanical signal that this file was never overwritten with a real
    person's card is the placeholder marker in the `name` fact. Without this,
    committing the live card would pass every gate in the repo.
    """
    card = pf.load(pf.EXAMPLE)
    names = [f.value for f in card.facts if f.kind == "name"]
    assert names, "the example lost its `name` fact"
    assert "示例" in names[0], (
        f"the example card's name is {names[0]!r} — if this is a real person's card "
        "it must not be committed; the seed for a real user lives in the store root")


# ------------------------------------------------- T-A5 the writer refuses (D)
def _seed_on_disk(store, body: dict) -> Path:
    """Put a profile.json in the store root without going through the store."""
    path = store.root / pf.CARD_NAME
    path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8")
    return path


def test_write_snapshot_refuses_the_protected_card_and_changes_nothing(tmp_path) -> None:
    """T-A5, asserted where the write API actually lives."""
    store = make_store(tmp_path)
    try:
        seed = _seed_on_disk(store, pf.load(pf.EXAMPLE).as_dict())
        before_bytes = seed.read_bytes()
        before_stat = seed.stat()

        with pytest.raises(st.ReadOnlyError, match="read-only seed until S5"):
            store.write_snapshot(pf.CARD_NAME, {"version": 1, "facts": []})

        store.flush()
        assert seed.read_bytes() == before_bytes, "the card's bytes changed"
        assert seed.stat().st_mtime_ns == before_stat.st_mtime_ns
        assert not list(store.root.glob(f".{pf.CARD_NAME}.*")), "temp debris left behind"
    finally:
        store.close()


def test_an_unprotected_snapshot_still_writes(tmp_path) -> None:
    """The control for the test above. A lock that refused every snapshot would make
    it pass while breaking the RPS table and the semantic layer, which are the other
    two callers of `write_snapshot`."""
    store = make_store(tmp_path)
    try:
        store.write_snapshot("rps_table.json", {"ok": True})
        store.flush()
        assert json.loads((store.root / "rps_table.json").read_text()) == {"ok": True}
    finally:
        store.close()


def test_unlock_for_s5_is_the_way_past_the_lock(tmp_path) -> None:
    store = make_store(tmp_path)
    try:
        assert st.is_locked(pf.CARD_NAME)
        st.unlock_for_s5(pf.CARD_NAME)
        assert not st.is_locked(pf.CARD_NAME)
        store.write_snapshot(pf.CARD_NAME, {"version": 1, "facts": ["written by S5"]})
        store.flush()
        assert "written by S5" in (store.root / pf.CARD_NAME).read_text()
    finally:
        store.close()


def test_unlock_for_s5_refuses_a_name_it_does_not_govern() -> None:
    """The lock is a named set, so unlocking an unprotected name is a caller that
    believes it is protected. Silently succeeding would teach that belief."""
    with pytest.raises(st.StoreError, match="not a protected snapshot"):
        st.unlock_for_s5("events.jsonl")
    assert "events.jsonl" not in st._UNLOCKED


def test_the_lock_is_not_lifted_anywhere_at_import_time() -> None:
    """Importing the tree must not unlock anything. Asserted in a subprocess with a
    clean interpreter, because this session's own tests unlock deliberately."""
    import subprocess
    code = ("import memory.profile, memory.store as s, safety.guard, skills.loader; "
            "print(sorted(s._UNLOCKED))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_no_module_outside_tests_calls_unlock_for_s5() -> None:
    """S5 has not landed, so the sanctioned unlock must have no production caller yet.

    This scan reads its own source on purpose: the call in
    `test_unlock_for_s5_is_the_way_past_the_lock` is the positive control. Without
    it an empty result would be indistinguishable from a detector that has gone
    blind — a scan that finds nothing is only evidence when it can find something.
    """
    scanned, callers = 0, []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            named = (getattr(fn, "id", None) == "unlock_for_s5"
                     or getattr(fn, "attr", None) == "unlock_for_s5")
            if named:
                callers.append(str(path.relative_to(REPO_ROOT)))
    assert scanned >= 20, f"only {scanned} python files scanned — the walk went blind"
    assert callers, "the detector found no call at all, including its own control"
    outside = sorted(c for c in callers if not c.startswith("tests/"))
    assert not outside, (
        f"{outside} call unlock_for_s5, but S5 has not landed: an unlock in "
        "production code opens provenance-free writes to the seed card")


# ------------------------------------------- T-A5 the card's own refusing surface
READ_METHODS = frozenset({"render", "as_dict"})


def test_every_public_card_method_that_is_not_a_reader_refuses() -> None:
    """Derived from the class, not from a written-down list of method names.

    A list of {add, remove, save, update, replace} cannot notice the sixth mutator
    somebody adds next; asking "which public callables are not readers" turns that
    addition red until it either raises or is declared a reader.
    """
    card = pf.load(pf.EXAMPLE)
    public = {n for n in dir(pf.ProfileCard) if not n.startswith("_")
              and callable(getattr(pf.ProfileCard, n))}
    assert READ_METHODS <= public, f"a reader disappeared: {READ_METHODS - public}"
    writers = public - READ_METHODS
    assert writers, "the card has no write surface at all, so this test proves nothing"
    for name in sorted(writers):
        with pytest.raises(pf.ReadOnlyError):
            getattr(card, name)()


def test_the_card_and_its_facts_reject_attribute_assignment() -> None:
    card = pf.load(pf.EXAMPLE)
    with pytest.raises(pf.ReadOnlyError):
        card.facts = ()
    with pytest.raises(pf.ReadOnlyError):
        card.facts[0].value = "somebody else"
    with pytest.raises(pf.ReadOnlyError):
        del card.version


# ------------------------------------------------------------- the §0-5 law (D)
def test_a_non_seed_fact_must_name_its_source_episodes() -> None:
    """The reason the whole card is locked, stated as an assertion.

    Note what is *not* being tested: that a fact with no `source_episode_ids` is a
    seed. Absence cannot mean seed, because S5's first automatic write also has no
    ids at the moment the object is constructed.
    """
    pf.Fact("goal", "練粵語", pf.TRUST_SEED)                      # allowed
    pf.Fact("goal", "練粵語", "observed", ("ep-1",))              # allowed
    for trust in [t for t in pf.TRUST_LEVELS if t != pf.TRUST_SEED]:
        with pytest.raises(pf.ProfileError, match="requires source_episode_ids"):
            pf.Fact("goal", "練粵語", trust)


@pytest.mark.parametrize("kind,value,trust,match", [
    ("capital_of", "Paris", "seed", "unknown predicate"),
    ("name", "", "seed", "non-empty string"),
    ("name", "   ", "seed", "non-empty string"),
    ("name", None, "seed", "non-empty string"),
    ("name", "小明", "trusted", "unknown trust"),
])
def test_the_predicate_and_trust_sets_are_closed(kind, value, trust, match) -> None:
    with pytest.raises(pf.ProfileError, match=match):
        pf.Fact(kind, value, trust)


@pytest.mark.parametrize("body,match", [
    ([], "must be a JSON object"),
    ({"facts": [{"kind": "name", "value": "小明", "trust": "seed"}]}, "version must be 1"),
    ({"version": 2, "facts": [{"kind": "name", "value": "小明", "trust": "seed"}]},
     "version must be 1"),
    ({"version": 1, "facts": []}, "non-empty list"),
    ({"version": 1, "facts": {}}, "non-empty list"),
    ({"version": 1, "facts": ["name: 小明"]}, "not an object"),
    ({"version": 1, "facts": [{"kind": "name", "value": "小明", "trust": "seed",
                               "confidence": 0.9}]}, "unknown field"),
])
def test_a_card_that_is_not_a_card_is_an_error_not_an_empty_card(body, match) -> None:
    with pytest.raises(pf.ProfileError, match=match):
        pf.parse(body)


def test_a_missing_seed_file_is_an_error(tmp_path) -> None:
    with pytest.raises(pf.ProfileError, match="does not exist"):
        pf.load(tmp_path / "profile.json")


def test_invalid_json_names_the_file(tmp_path) -> None:
    bad = tmp_path / "profile.json"
    bad.write_text('{"version": 1, "facts": [', encoding="utf-8")
    with pytest.raises(pf.ProfileError, match="not valid JSON"):
        pf.load(bad)


def test_the_closed_predicate_set_has_exactly_one_creator() -> None:
    """Two copies of a closed set is an open set. S5's consolidator must import the
    name rather than restate it, and this is what makes restating it red."""
    scanned, creators = 0, []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "PREDICATES"
                    for t in node.targets):
                creators.append(str(path.relative_to(REPO_ROOT)))
    assert scanned >= 20, f"only {scanned} python files scanned — the walk went blind"
    assert creators == ["memory/profile.py"], creators


# ---------------------------------------------------- determinism (T-B4 shape)
def test_render_is_the_files_order_and_byte_identical_across_loads() -> None:
    card = pf.ProfileCard([
        pf.Fact("pet", "一隻貓", pf.TRUST_SEED),
        pf.Fact("allergy", "花生", pf.TRUST_SEED),
        pf.Fact("name", "小明", pf.TRUST_SEED),
    ])
    assert card.render() == "pet: 一隻貓\nallergy: 花生\nname: 小明"
    assert pf.load(pf.EXAMPLE).render() == pf.load(pf.EXAMPLE).render()


def test_the_rendered_block_carries_no_heading_or_trust_annotation() -> None:
    """A heading is paid on every turn for the life of the robot, and a rendered
    `trust: seed` would put a token the model can be told to weigh into the prompt."""
    text = pf.load(pf.EXAMPLE).render()
    assert len(text.splitlines()) == len(pf.load(pf.EXAMPLE))
    assert "trust" not in text and "seed" not in text
    assert not text.startswith("#") and "以下" not in text


# --------------------------------------------------------- T-F5 token budget (M)
def _assigned(path: Path, name: str):
    """The value assigned to `name` at module level in `path`, read by AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is gone from {path.name} — the value this test "
                         "measures against has moved")


def _skills_proxy_ratio() -> float:
    """PROXY_RATIO out of tests/test_skills_loader.py by AST — the same reader
    tests/test_token_cost_harness.py uses, and for the same reason: a second copy of
    a calibration measured once against a real `usage.inputTokens` is a second
    calibration, and the one that drifts is the one nobody reads."""
    return float(_assigned(TESTS_DIR / "test_skills_loader.py", "PROXY_RATIO"))


def test_the_shipped_example_fits_the_resident_budget() -> None:
    ratio = _skills_proxy_ratio()
    assert 0 < ratio < 1, f"a proxy that over-counts would make this gate lax: {ratio}"
    card = pf.load(pf.EXAMPLE)
    assert pf.budget_findings(card, ratio) == []
    # Reported, not asserted as a range: the headroom is for the user's real facts,
    # and a tight upper bound here would turn "the user wrote a longer name" red.
    print(f"resident block: {approx_tokens(card.render()) / ratio:.0f} inflated "
          f"tokens of {pf.TOKEN_BUDGET} across {len(card)} facts")


def test_forty_facts_do_not_fit(tmp_path) -> None:
    """The plan's named T-F5 mutant, as a property of the checker rather than a
    change to the code: the budget must actually bind on a plausible over-large card."""
    kinds = list(pf.PREDICATES)
    card = pf.ProfileCard([
        pf.Fact(kinds[i % len(kinds)], f"這是第 {i} 條事實,長度接近真實使用者會寫的句子",
                pf.TRUST_SEED)
        for i in range(40)
    ])
    findings = pf.budget_findings(card, _skills_proxy_ratio())
    assert any("resident block" in f for f in findings), findings


def _card_inside_the_inflation_window(ratio: float):
    """A card whose raw proxy fits the budget but whose inflated count does not.

    Built by search rather than hand-tuned, so it keeps discriminating if the
    calibration moves. Its existence is asserted, because an empty window would make
    the direction mutant below vacuous.
    """
    for n in range(1, 400):
        card = pf.ProfileCard([pf.Fact("note", "記得這件事情很重要", pf.TRUST_SEED)
                               for _ in range(n)])
        raw = approx_tokens(card.render())
        if pf.TOKEN_BUDGET * ratio < raw <= pf.TOKEN_BUDGET:
            return card, raw
    raise AssertionError(
        f"no card lands between {pf.TOKEN_BUDGET * ratio:.0f} and {pf.TOKEN_BUDGET} "
        "proxy tokens, so the inflation direction cannot be tested")


def test_the_proxy_is_inflated_and_not_deflated() -> None:
    """The proxy under-counts (0.79 measured), and under-counting is the dangerous
    direction for a budget: it lets a real 440-token card through a 350-token gate."""
    ratio = _skills_proxy_ratio()
    card, raw = _card_inside_the_inflation_window(ratio)
    assert raw <= pf.TOKEN_BUDGET, raw
    assert pf.budget_findings(card, ratio), (
        f"{raw} proxy tokens is ~{raw / ratio:.0f} real tokens, over the "
        f"{pf.TOKEN_BUDGET} budget, and was accepted")


def test_a_duplicated_single_valued_predicate_is_a_finding() -> None:
    two_names = pf.ProfileCard([pf.Fact("name", "小明", pf.TRUST_SEED),
                                pf.Fact("name", "小華", pf.TRUST_SEED)])
    assert any("more than once" in f for f in pf.budget_findings(two_names, 0.79))
    # The control: a repeatable predicate must not trip it, or the check would
    # forbid the two `likes` lines the seed card is meant to carry.
    two_likes = pf.ProfileCard([pf.Fact("likes", "猜拳", pf.TRUST_SEED),
                                pf.Fact("likes", "講故事", pf.TRUST_SEED)])
    assert pf.budget_findings(two_likes, 0.79) == []


# ------------------------------------------- the boot policy (pure, so it is tested)
MEMORY_NODE_PY = REPO_ROOT / "memory" / "memory_node.py"


def _seed_root(tmp_path: Path, body: dict | None = None) -> Path:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    if body is not None:
        (root / pf.CARD_NAME).write_text(json.dumps(body, ensure_ascii=False),
                                         encoding="utf-8")
    return root


def test_no_seed_at_all_is_not_an_error(tmp_path) -> None:
    """A fresh instance has no card, and refusing to boot would mean the user cannot
    talk to the only thing that would tell them how to write one."""
    root = _seed_root(tmp_path)
    assert pf.boot_card(root, 0.79) is None
    hint = pf.missing_seed_hint(root / pf.CARD_NAME)
    assert str(root) in hint and pf.EXAMPLE.name in hint


def test_a_malformed_seed_is_an_error_and_not_a_silent_none(tmp_path) -> None:
    """The distinction that matters: "nobody wrote a card" and "the card you wrote is
    not the card you think you wrote" must not present identically, or the user
    concludes the robot simply forgets."""
    root = _seed_root(tmp_path, {"version": 1, "facts": [
        {"kind": "capital_of", "value": "Paris", "trust": "seed"}]})
    with pytest.raises(pf.ProfileError, match="unknown predicate"):
        pf.boot_card(root, 0.79)


def test_an_over_budget_seed_is_a_boot_failure(tmp_path) -> None:
    kinds = list(pf.PREDICATES)
    root = _seed_root(tmp_path, {"version": 1, "facts": [
        {"kind": kinds[i % len(kinds)],
         "value": f"這是第 {i} 條事實,長度接近真實使用者會寫的句子", "trust": "seed"}
        for i in range(40)]})
    with pytest.raises(pf.ProfileError, match="resident block"):
        pf.boot_card(root, _skills_proxy_ratio())


def test_the_shipped_example_boots(tmp_path) -> None:
    """The control for the three refusals above: the card the repo ships must pass the
    exact policy the node runs, with the exact ratio the node passes in."""
    root = _seed_root(tmp_path)
    shutil.copy2(pf.EXAMPLE, root / pf.CARD_NAME)
    card = pf.boot_card(root, float(_assigned(MEMORY_NODE_PY, "PROXY_RATIO")))
    assert card is not None and len(card) >= 15


# --------------------------------------------- the node's contract, checked statically
def test_the_node_uses_the_one_calibration() -> None:
    """`memory_node.PROXY_RATIO` is a second copy of a number measured once. A copy is
    acceptable — the node cannot import from tests/ — but an unpinned copy is the
    defect, so this is the pin."""
    assert float(_assigned(MEMORY_NODE_PY, "PROXY_RATIO")) == _skills_proxy_ratio()


def _publisher_qos_names(src: str) -> dict[str, str]:
    """topic string → the name of the QoS argument its `create_publisher` is handed.

    Takes source text rather than a path so a mutant can be checked without writing a
    file, the same reason `ros_census.edges_in` does.
    """
    tree = ast.parse(src)
    consts = {t.id: n.value.value
              for n in ast.walk(tree) if isinstance(n, ast.Assign)
              for t in n.targets
              if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_publisher" and len(node.args) >= 3
                and isinstance(node.args[1], ast.Name)
                and isinstance(node.args[2], ast.Name)):
            topic = consts.get(node.args[1].id)
            if topic is not None:
                names[topic] = node.args[2].id
    return names


def _publisher_qos(path: Path, topic_name: str) -> dict[str, ast.expr]:
    """The keyword *nodes* of the `QoSProfile(...)` the publisher for `topic_name` is handed.

    **Derived through the publisher, because the count broke.** The first version read
    *every* `QoSProfile` call in the file and asserted there was exactly one. That held
    for two rounds and then R1.2 added `TABLE_QOS` — a legitimate second profile, for a
    second latched topic — and this test went red for a reason it was never about
    (`expected one QoSProfile, found 2`). A count is pinned to the file not growing, and
    the cheapest way past that red is `profiles[0]`, which silently stops checking which
    of the two the card is published with.

    So the walk goes the other way: find `create_publisher(..., <TOPIC const>, <name>)`,
    resolve `<TOPIC const>` through the module's own assignment, then read the
    `QoSProfile(...)` assigned to `<name>`. Swapping the publisher onto a volatile
    profile now moves this test, and adding a third topic does not.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    qos_name = _publisher_qos_names(src).get(topic_name)
    assert qos_name, (f"no create_publisher for {topic_name} in {path.name} with a named "
                      "QoS argument — the publisher this test measures has moved")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == qos_name for t in node.targets)):
            assert (isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", None) == "QoSProfile"), (
                f"{qos_name} is not its own QoSProfile(...) call — an alias makes two "
                "topics share one policy object, which is one decision wearing two names")
            return {k.arg: k.value for k in node.value.keywords}
    raise AssertionError(f"{qos_name} is not assigned at module level in {path.name}")


def _qos_dump(path: Path, topic_name: str) -> dict[str, str]:
    """`_publisher_qos` with each value rendered, for the enum-membership assertions."""
    return {arg: ast.dump(node) for arg, node in _publisher_qos(path, topic_name).items()}


def test_the_card_is_published_transient_local_with_depth_one() -> None:
    """Static, and honestly so: whether a late joiner actually receives the sample is
    an rclpy behaviour and belongs to layer L4 on the instance. What CI can hold is
    that the durability was asked for at all — a `QoSProfile(depth=10)` with the
    default VOLATILE durability is the silent version of this defect, and it looks
    exactly like every other publisher in the repo.
    """
    kw = _publisher_qos(MEMORY_NODE_PY, pf_topic_name())
    dumped = {arg: ast.dump(node) for arg, node in kw.items()}
    assert "TRANSIENT_LOCAL" in dumped.get("durability", ""), dumped
    assert "RELIABLE" in dumped.get("reliability", ""), dumped
    assert ast.literal_eval(kw["depth"]) == 1


def test_the_second_latched_topic_did_not_quietly_reuse_the_card_policy() -> None:
    """The positive control for the helper above, and a claim of its own.

    R1.2's `/memory/rps_table` is the second latched topic in this file. Two things
    could have gone wrong and only one of them is about the card: the table could have
    been published `VOLATILE` (its own test, in `tests/test_rps_wiring.py`), or the two
    publishers could have been handed the *same* object, at which case a later change to
    one topic's durability silently changes the other's. So this asserts the two
    publishers name two different profiles, and that both ask for durability — which is
    also what proves the helper above is reading the publisher and not just the first
    `QoSProfile` in the file.
    """
    card = pf_topic_name()
    table = _assigned(MEMORY_NODE_PY, "RPS_TABLE_TOPIC")
    assert card != table
    names = _publisher_qos_names(MEMORY_NODE_PY.read_text(encoding="utf-8"))
    assert names.get(card) and names.get(table), names
    assert names[card] != names[table], (
        f"both latched topics are published with {names[card]}: one object, two topics, "
        "so the next durability change moves a topic nobody was editing")
    for topic in (card, table):
        assert "TRANSIENT_LOCAL" in _qos_dump(MEMORY_NODE_PY, topic).get(
            "durability", ""), topic


def test_the_node_starts_no_timer_that_would_hide_a_volatile_subscriber() -> None:
    """A republish timer makes a subscriber that forgot to request durability appear
    to work — it gets the card on the second tick — and then it has no profile on the
    first turn, which is the turn that matters."""
    src = MEMORY_NODE_PY.read_text()
    assert "create_timer" not in src


def test_the_node_does_not_render_the_block_itself() -> None:
    """One renderer. A pre-rendered string on the wire becomes a second one the moment
    a consumer builds the block from `facts`, and then the measured budget is not the
    budget paid."""
    tree = ast.parse(MEMORY_NODE_PY.read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    # AST, not a substring search: the module docstring names `render()` when it
    # explains that consumers do the rendering, and a grep would read the
    # explanation of the rule as a violation of it.
    assert "render" not in called, called
    assert "as_dict" in called, called


def test_the_node_is_the_only_python_file_that_publishes_the_profile_topic() -> None:
    """Two publishers of a `transient_local` topic is two cards, and which one a late
    joiner receives depends on subscribe timing.

    **The derivation is the finding here, not the result.** This test used to scan every
    production file for the substring `"/memory/profile"` and assert the hit list was
    exactly `["memory/memory_node.py"]`. A substring cannot tell a publisher from a
    subscriber, from a docstring, or from a comment — so R1.1, whose entire content is
    adding the **consumer** this topic was waiting for, turned it red for a reason it was
    never about. And the cheapest way past that red is to add `agent/agent_node.py` to
    the expected list, which would then also excuse a real second publisher there.
    A discovery pattern is a claim: this one claimed to find publishers and found
    mentions.

    It now asks `infra/ros_census.py`, the one reader of the topic graph, which resolves
    each topic argument through the module's own constants — so `TOPIC` in memory_node and
    `PROFILE_TOPIC` in agent_node are both read as the topic they name, and a mention in
    prose is not an edge. The subscriber side is asserted too, because "exactly one
    publisher" and "somebody is listening" are two claims and the pair is what makes the
    latched card arrive at all.
    """
    sys.path.insert(0, str(REPO_ROOT / "infra"))
    from ros_census import census  # noqa: PLC0415
    _nodes, _edges, topics = census(REPO_ROOT)
    assert len(topics) >= 10, (f"only {len(topics)} topics censused — the graph walk "
                               "went blind and every claim below would hold vacuously")
    topic = topics.get(pf_topic_name())
    assert topic is not None, sorted(topics)
    assert topic.pub == ["memory"], topic.pub
    assert topic.sub == ["agent"], topic.sub


def pf_topic_name() -> str:
    """The profile topic, read out of memory_node's own assignment rather than typed.

    A literal here would be a third copy of the name — the node has one, agent_node has
    a pinned copy — and the copy in a test is the one that keeps passing after a rename.
    """
    tree = ast.parse(MEMORY_NODE_PY.read_text(encoding="utf-8"))
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TOPIC" for t in stmt.targets):
            assert isinstance(stmt.value, ast.Constant), ast.dump(stmt.value)
            return stmt.value.value
    raise AssertionError("memory/memory_node.py no longer assigns TOPIC")


# ================================================================ layer L3: mutants
@pytest.mark.mutation
def test_no_mutant_control_still_refuses_the_card(tmp_path) -> None:
    """§6.0 rule 2. A harness that cannot import a clean copy reports every mutant
    as killed, which reads exactly like a well-defended lock."""
    control = load_mutant(tmp_path, "store")
    assert control.is_locked("profile.json")
    store = control.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        with pytest.raises(control.ReadOnlyError):
            store.write_snapshot("profile.json", {"version": 1, "facts": []})
    finally:
        store.close()


@pytest.mark.mutation
def test_removing_the_lock_check_lets_the_card_be_overwritten(tmp_path) -> None:
    """The T-A5 mutant the plan names: "開放寫入 → 紅"."""
    mutant = load_mutant(tmp_path, "store", (LOCK_CHECK, "        self._raise_if_failed()"))
    store = mutant.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        store.write_snapshot("profile.json", {"version": 1, "facts": ["overwritten"]})
        store.flush()
        assert "overwritten" in (store.root / "profile.json").read_text()
    finally:
        store.close()


@pytest.mark.mutation
def test_an_empty_protected_set_lets_the_card_be_overwritten(tmp_path) -> None:
    """A second route to the same failure: keep the check, empty the set. Worth its
    own mutant because `PROTECTED_SNAPSHOTS = frozenset()` is the edit somebody makes
    to unblock a write without touching `write_snapshot`."""
    mutant = load_mutant(tmp_path, "store",
                         (PROTECTED_SET, "PROTECTED_SNAPSHOTS = frozenset()"))
    assert not mutant.is_locked("profile.json")
    store = mutant.Store(tmp_path / "data", _allow_ephemeral=True)
    try:
        store.write_snapshot("profile.json", {"version": 1, "facts": ["overwritten"]})
        store.flush()
        assert "overwritten" in (store.root / "profile.json").read_text()
    finally:
        store.close()


@pytest.mark.mutation
def test_an_inverted_lock_predicate_locks_exactly_the_wrong_names(tmp_path) -> None:
    """`name in _UNLOCKED` instead of `name not in _UNLOCKED`. Both halves are
    checked, because the inverted predicate still returns True for the fresh card
    when `_UNLOCKED` is empty — asserting only "the fresh card is locked" would
    let this one live."""
    mutant = load_mutant(tmp_path, "store", (
        IS_LOCKED_BODY,
        "    return name in PROTECTED_SNAPSHOTS and name in _UNLOCKED"))
    assert not mutant.is_locked("profile.json"), "the mutant is not the mutant"
    mutant.unlock_for_s5("profile.json")
    assert mutant.is_locked("profile.json"), "unlocking must not lock"


@pytest.mark.mutation
def test_an_unguarded_unlock_would_accept_any_name(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "store", (UNLOCK_GUARD, "    if False:"))
    mutant.unlock_for_s5("events.jsonl")
    assert "events.jsonl" in mutant._UNLOCKED


@pytest.mark.mutation
def test_checking_the_lock_after_the_writer_check_misreports_the_reason(tmp_path) -> None:
    """The ordering claim, which nothing observes while the writer is healthy.

    Real module: a locked write is a lock, even on a dead store. Mutant: the same
    call is reported as a disk failure, which is how somebody removes a lock to
    "fix" a disk. `ReadOnlyError` subclasses `StoreError`, so the discrimination has
    to be on the exact type — `pytest.raises(StoreError)` passes on both.
    """
    for module, expect_lock in ((st, True), (load_mutant(
            tmp_path, "store",
            (LOCK_CHECK,
             "        self._raise_if_failed()\n"
             "        if is_locked(name):\n"
             "            raise ReadOnlyError(\n"
             '                f"{name!r} is a read-only seed until S5: a fact written now would "\n'
             '                "have no source episode (episodes arrive in S4) and no deletion "\n'
             '                "could ever reach it. memory.store.unlock_for_s5 is the one way past")')),
            False)):
        root = tmp_path / ("real" if expect_lock else "mut")
        store = module.Store(root, _allow_ephemeral=True)
        try:
            store._failure = RuntimeError("no space left on device")
            with pytest.raises(module.StoreError) as caught:
                store.write_snapshot("profile.json", {"version": 1, "facts": []})
        finally:
            store._failure = None
            store.close()
        is_lock = isinstance(caught.value, module.ReadOnlyError)
        assert is_lock is expect_lock, (
            f"expected a {'lock' if expect_lock else 'disk'} error, got {caught.value!r}")


@pytest.mark.mutation
def test_dropping_the_source_requirement_admits_undeletable_facts(tmp_path) -> None:
    """§0-5, as a mutant: without the check, an `observed` fact with no episode
    behind it constructs cleanly and no deletion can ever reach it."""
    mutant = load_mutant(tmp_path, "profile",
                         (SOURCE_REQUIREMENT, "        if False:"))
    fact = mutant.Fact("goal", "練粵語", "observed")
    assert fact.source_episode_ids == ()
    assert fact.trust == "observed"


@pytest.mark.mutation
def test_multiplying_by_the_ratio_deflates_the_budget(tmp_path) -> None:
    """The direction mutant for T-F5. The card in the window is accepted by the
    mutant and refused by the real module, so the two differ on a real card rather
    than only on an absurd one."""
    ratio = _skills_proxy_ratio()
    card, raw = _card_inside_the_inflation_window(ratio)
    mutant = load_mutant(tmp_path, "profile",
                         (INFLATE, "    tokens = approx_tokens(card.render()) * ratio"))
    mutant_card = mutant.ProfileCard([mutant.Fact(f.kind, f.value, f.trust)
                                      for f in card.facts])
    assert mutant.budget_findings(mutant_card, ratio) == [], "the mutant is not the mutant"
    assert pf.budget_findings(card, ratio), raw


@pytest.mark.mutation
def test_sorting_the_render_changes_the_resident_prompt(tmp_path) -> None:
    """T-B4's shape one layer down: a `sorted()` here makes the block a function of
    the alphabet rather than of the file, so a token count measured once is not the
    count paid later."""
    mutant = load_mutant(tmp_path, "profile", (
        RENDER_JOIN,
        '        return "\\n".join(sorted(f.line() for f in self.facts))'))
    facts = [("pet", "一隻貓"), ("allergy", "花生"), ("name", "小明")]
    real = pf.ProfileCard([pf.Fact(k, v, pf.TRUST_SEED) for k, v in facts])
    mut = mutant.ProfileCard([mutant.Fact(k, v, mutant.TRUST_SEED) for k, v in facts])
    assert real.render() != mut.render()


@pytest.mark.mutation
def test_accepting_an_empty_facts_list_makes_recall_tests_vacuous(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "profile", (
        EMPTY_FACTS, "    if not isinstance(raw, list):"))
    card = mutant.parse({"version": 1, "facts": []})
    assert len(card) == 0 and card.render() == ""


@pytest.mark.mutation
def test_ignoring_unknown_fields_hides_a_setting_the_author_believes_is_on(tmp_path) -> None:
    mutant = load_mutant(tmp_path, "profile", (
        UNKNOWN_FIELDS, "        unknown = set() - "))
    card = mutant.parse({"version": 1, "facts": [
        {"kind": "name", "value": "小明", "trust": "seed", "confidence": 0.1}]})
    assert len(card) == 1, "the mutant did not accept the unknown field"
    assert not hasattr(card.facts[0], "confidence")


@pytest.mark.mutation
def test_a_mutable_fact_would_edit_what_the_next_subscriber_receives(tmp_path) -> None:
    """`transient_local` hands the same object to every late joiner, so a mutable
    fact is a card that differs between two nodes with nothing in any log to say so."""
    mutant = load_mutant(tmp_path, "profile", (
        '        raise ReadOnlyError(\n'
        '            f"a Fact is immutable ({name}={value!r}): the card is published by "\n'
        '            "reference to every late-joining subscriber, so an in-place edit "\n'
        '            "would change what the next one receives")',
        "        object.__setattr__(self, name, value)"))
    fact = mutant.Fact("name", "小明", mutant.TRUST_SEED)
    fact.value = "somebody else"
    assert fact.line() == "name: somebody else"


@pytest.mark.mutation
def test_a_load_that_defaults_a_missing_file_hides_total_amnesia(tmp_path) -> None:
    """A robot that has forgotten everything while reporting healthy."""
    mutant = load_mutant(tmp_path, "profile", (
        "    if not path.exists():\n        raise ProfileError(",
        "    if False:\n        raise ProfileError("))
    with pytest.raises(FileNotFoundError):
        mutant.load(tmp_path / "absent.json")
    # The real module names the fix instead of raising the OS error at the caller.
    with pytest.raises(pf.ProfileError, match="hand-written"):
        pf.load(tmp_path / "absent.json")


@pytest.mark.mutation
def test_a_boot_that_swallows_a_malformed_card_reads_as_forgetting(tmp_path) -> None:
    """The mutant is the tempting one: wrap `load` in try/except and return None, so
    the robot always starts. It makes "your card has a typo" indistinguishable from
    "you never wrote one", and the only symptom is a robot that forgets."""
    mutant = load_mutant(tmp_path, "profile", (
        "    card = load(path)\n",
        "    try:\n"
        "        card = load(path)\n"
        "    except ProfileError:\n"
        "        return None\n"))
    root = _seed_root(tmp_path, {"version": 1, "facts": [
        {"kind": "capital_of", "value": "Paris", "trust": "seed"}]})
    assert mutant.boot_card(root, 0.79) is None
    with pytest.raises(pf.ProfileError):
        pf.boot_card(root, 0.79)


@pytest.mark.mutation
def test_a_boot_that_only_logs_the_budget_finding_publishes_it_anyway(tmp_path) -> None:
    """The warning-instead-of-refusal mutant. It costs input tokens on every turn for
    the life of the process, and the only place it appears is a log line."""
    mutant = load_mutant(tmp_path, "profile", ("    if findings:", "    if False:"))
    kinds = list(pf.PREDICATES)
    root = _seed_root(tmp_path, {"version": 1, "facts": [
        {"kind": kinds[i % len(kinds)],
         "value": f"這是第 {i} 條事實,長度接近真實使用者會寫的句子", "trust": "seed"}
        for i in range(40)]})
    ratio = _skills_proxy_ratio()
    assert len(mutant.boot_card(root, ratio)) == 40
    with pytest.raises(pf.ProfileError, match="resident block"):
        pf.boot_card(root, ratio)


@pytest.mark.mutation
def test_a_second_publisher_of_the_card_is_not_read_as_a_subscriber() -> None:
    """The mutant the substring census could not have had, which is why it is here.

    The check above was rewritten this round because a scan for the text
    `"/memory/profile"` counted mentions and called them publishers. The property that
    replaced it — the census distinguishes `create_publisher` from
    `create_subscription` — is exactly what no mutant could have tested before, because
    the old derivation gave the same answer either way.

    So the mutation is applied to agent_node's edge itself: turn the consumer R1.1 added
    into a second publisher of a latched topic. `edges_in` is the census's own parser and
    takes source text, so the mutant needs no file on disk and the real tree is untouched
    (asserted by the control half below, which reads `kind == "sub"` from the shipped
    source in the same call).
    """
    sys.path.insert(0, str(REPO_ROOT / "infra"))
    from ros_census import edges_in  # noqa: PLC0415
    agent_node = REPO_ROOT / "agent" / "agent_node.py"
    src = agent_node.read_text(encoding="utf-8")
    topic = pf_topic_name()
    anchor = "self.create_subscription(String, PROFILE_TOPIC, self.on_profile,"
    assert anchor in src, f"mutant anchor no longer in {agent_node.name}: {anchor!r}"

    def kinds(source: str) -> set[str]:
        return {e.kind for e in edges_in(source, "agent", "agent/agent_node.py")
                if e.topic == topic}

    assert kinds(src) == {"sub"}, "the shipped agent_node is not a subscriber of the card"
    mutant = src.replace(
        anchor, "self.create_publisher(String, PROFILE_TOPIC, ", 1)
    assert kinds(mutant) == {"pub"}, (
        "the census reports a create_publisher call as something other than a publisher, "
        "so `topic.pub == ['memory']` cannot notice a second card producer")


@pytest.mark.mutation
def test_the_two_latched_publishers_sharing_one_profile_object_is_caught() -> None:
    """The mutant the `len(profiles) == 1` version could not have had.

    That version asserted the file contained exactly one `QoSProfile`, so the way to
    make it pass with two latched topics was to give them **one** profile object — the
    mutation below — and the count would then have been 1 and green. The count agreed
    with the defect. Reading the QoS name *through the publisher* is what makes the two
    cases distinguishable, so this is the assertion that pays for the rewrite.

    Applied to source text, like the census mutant above: nothing on disk moves, and the
    control half re-reads the shipped source in the same call.
    """
    src = MEMORY_NODE_PY.read_text(encoding="utf-8")
    card, table = pf_topic_name(), _assigned(MEMORY_NODE_PY, "RPS_TABLE_TOPIC")
    anchor = "RPS_TABLE_TOPIC, TABLE_QOS)"
    assert anchor in src, f"mutant anchor no longer in {MEMORY_NODE_PY.name}: {anchor!r}"

    shipped = _publisher_qos_names(src)
    assert shipped[card] != shipped[table], "the shipped tree already shares one profile"

    mutant = _publisher_qos_names(src.replace(anchor, "RPS_TABLE_TOPIC, PROFILE_QOS)", 1))
    assert mutant[card] == mutant[table] == shipped[card], mutant
    # And the count the old version read is *unchanged* by the mutation in the direction
    # that matters: patch out the now-unused TABLE_QOS assignment and the file is back to
    # one QoSProfile call, which the old assertion would have called correct.
    trimmed = src.replace(anchor, "RPS_TABLE_TOPIC, PROFILE_QOS)", 1)
    trimmed = trimmed[:trimmed.index("TABLE_QOS = QoSProfile(")] + "TABLE_QOS = None\n"
    assert trimmed.count("QoSProfile(") == 1, trimmed.count("QoSProfile(")


def test_the_store_module_is_unpolluted_after_the_mutants() -> None:
    """The mutants import text-patched copies under their own module names; this
    asserts the real module was not one of the things they patched (the residue
    defect §6.2 T-J4 guards against, checked here rather than trusted)."""
    assert st.PROTECTED_SNAPSHOTS == frozenset({"profile.json"})
    assert st.is_locked("profile.json")
    assert PROFILE_PY.read_text().count("PREDICATES = (") == 1
    assert pf.load(pf.EXAMPLE).render() == pf.load(pf.EXAMPLE).render()
