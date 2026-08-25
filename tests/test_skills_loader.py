"""skills/loader.py — test items T-I1 (L1 budget), T-I2 (gate), T-I3 (traversal).

The three named mutants live at the bottom of each section. They are cheap here
(a text-mutated copy of one pure module, imported in-process) rather than the
script-exec mutants of test_deploy_path.py, so they carry the same `mutation`
marker but a very different cost.
"""
from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skills import loader

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
LOADER_SRC = SKILLS_DIR / "loader.py"
AGENT_NODE = REPO_ROOT / "agent" / "agent_node.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skill_triggers.tsv"

# T-I1's budget, from the plan: what one skill may cost in every single turn.
L1_BUDGET_TOKENS = 150

# ---------------------------------------------------------------------------
# Real measurement, 2026-08-22, `python skills/measure_l1_tokens.py` against
# global.anthropic.claude-sonnet-5 in ap-southeast-1 (the same model and region
# agent/agent_node.py uses — test_model_id_matches_the_measurement keeps that
# true):
#
#   baseline request                     8 input tokens
#   same request + the L1 index        104 input tokens
#   -> the resident L1 block costs      96 tokens for 1 skill
#   approx_tokens() on the same string   76
#
# So the character proxy UNDER-counts by 21% (76 / 96 = 0.79). Under-counting is
# the dangerous direction for a budget check — it would let a real 190-token index
# pass a 150-token gate — so the CI assertion divides the proxy by this ratio
# before comparing. The ratio is Chinese-heavy text on one tokenizer; it is a
# calibration, not a law.
MEASURED_L1_TOKENS = 96
MEASURED_PROXY_TOKENS = 76
PROXY_RATIO = 0.79
# ---------------------------------------------------------------------------


def read_fixture() -> list[tuple[str, str, str]]:
    rows = []
    for lineno, raw in enumerate(FIXTURE.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        assert len(parts) == 3, f"{FIXTURE.name}:{lineno}: expected 3 columns, got {len(parts)}"
        rows.append(tuple(p.strip() for p in parts))
    return rows


FIXTURE_ROWS = read_fixture()


def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-mutated copy of skills/ as a standalone module.

    The copy includes the skill directories, so the mutant's SKILLS_ROOT points at
    its own tree and a mutated loader can never read the real one. Each anchor is
    asserted present: a stale mutant that silently applies nothing would report
    "mutant survived" and blame the assertion under test.
    """
    dst = tmp_path / "skills"
    shutil.copytree(SKILLS_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    target = dst / "loader.py"
    src = target.read_text()
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present in loader.py: {old!r}"
        src = src.replace(old, new)
    target.write_text(src)

    name = f"loader_mutant_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so an
    # unregistered module makes the mutant fail at import for a reason that has
    # nothing to do with the mutation.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# ============================================================ discovery, loudly
def test_the_vendored_skill_is_discovered() -> None:
    skills = loader.discover()
    assert [s.name for s in skills] == ["critical-thinking"]
    skill = skills[0]
    assert skill.dirname == skill.name
    assert len(skill.triggers) >= 20, "a gate with a handful of phrases barely opens"
    assert set(skill.languages) == {"zh", "yue", "en"}, (
        "the robot answers in three languages, so a gate that only fires in one is "
        "a capability that exists for some users and not others"
    )


def test_a_skill_without_a_trigger_table_is_a_loud_error(tmp_path: Path) -> None:
    """The failure mode being prevented: a skill that can never fire, sitting in
    the L1 index costing tokens every turn, with nothing anywhere going red."""
    root = tmp_path / "skills"
    (root / "orphan").mkdir(parents=True)
    (root / "orphan" / "SKILL.md").write_text(
        "---\nname: orphan\ndescription: no gate.\n---\nbody\n")
    with pytest.raises(loader.SkillError, match="triggers.tsv"):
        loader.discover(root)


def test_frontmatter_name_must_match_its_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "thinking").mkdir(parents=True)
    (root / "thinking" / "SKILL.md").write_text(
        "---\nname: critical-thinking\ndescription: mismatched.\n---\n")
    (root / "thinking" / "triggers.tsv").write_text("phrase\tzh\n")
    with pytest.raises(loader.SkillError, match="does not match its directory"):
        loader.discover(root)


def test_frontmatter_rejects_yaml_it_does_not_actually_support(tmp_path: Path) -> None:
    """A folded description would be silently truncated at the first line, and the
    L1 budget would then look comfortable for a description nobody is sending."""
    with pytest.raises(loader.SkillError, match="not 'key: value'"):
        loader.parse_frontmatter(
            "---\ndescription: >\n  folded over\n  two lines\n---\n", where="x")


def test_an_empty_or_duplicated_trigger_table_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.tsv"
    empty.write_text("# only comments\n\n")
    with pytest.raises(loader.SkillError, match="0 triggers"):
        loader.read_triggers(empty)

    dupe = tmp_path / "dupe.tsv"
    dupe.write_text("紅隊\tzh\n紅隊\tzh\n")
    with pytest.raises(loader.SkillError, match="duplicate"):
        loader.read_triggers(dupe)


def test_directories_without_a_skill_md_are_not_skills(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "notaskill").mkdir(parents=True)
    (root / "notaskill" / "README.md").write_text("hi")
    assert loader.discover(root) == ()


# ================================================================ T-I1  budget
def test_l1_index_fits_the_budget() -> None:
    skills = loader.discover()
    index = loader.l1_index(skills)
    inflated = loader.approx_tokens(index) / PROXY_RATIO
    assert inflated <= L1_BUDGET_TOKENS * len(skills), (
        f"L1 index is ~{inflated:.0f} tokens for {len(skills)} skill(s), over the "
        f"{L1_BUDGET_TOKENS}/skill budget"
    )
    for skill in skills:
        line = loader.approx_tokens(skill.l1_line) / PROXY_RATIO
        assert line <= L1_BUDGET_TOKENS, f"{skill.name}: L1 line is ~{line:.0f} tokens"


def test_the_recorded_real_measurement_is_within_budget_and_still_current() -> None:
    """Two claims, derived separately.

    (1) the measured figure is inside the budget — that is the M-level result;
    (2) the string that was measured is still the string we ship. A recorded
        measurement with no liveness check is provenance theatre: the index could
        triple and the comment would still read like evidence. So the proxy is
        recomputed and compared against what it said on measurement day.
    """
    assert MEASURED_L1_TOKENS <= L1_BUDGET_TOKENS
    now = loader.approx_tokens(loader.l1_index(loader.discover()))
    assert now == MEASURED_PROXY_TOKENS, (
        f"the L1 index changed since it was measured (proxy {now} vs "
        f"{MEASURED_PROXY_TOKENS} on 2026-08-22) — re-run "
        f"`python skills/measure_l1_tokens.py` and update the block above"
    )


def test_l1_carries_no_part_of_the_skill_body() -> None:
    """L1 advertises; L2 instructs. If any prose from SKILL.md leaks into the
    resident block, every turn pays for guidance almost no turn uses."""
    skills = loader.discover()
    index = loader.l1_index(skills)
    body = loader.load_l2(skills[0])
    assert len(index) < len(body) / 10
    for line in body.splitlines():
        stripped = line.strip()
        if len(stripped) > 40 and not stripped.startswith(("#", "-", "|")):
            assert stripped not in index


def test_l1_index_is_empty_when_there_are_no_skills() -> None:
    assert loader.l1_index(()) == ""


@pytest.mark.mutation
def test_mutant_putting_the_whole_skill_md_in_l1_blows_the_budget(tmp_path: Path) -> None:
    """T-I1's named mutant: `把 SKILL.md 全文放進 L1`."""
    mutant = load_mutant(
        tmp_path,
        ("        one = first_sentence(self.description)",
         "        one = (Path(__file__).resolve().parent / self.dirname / 'SKILL.md').read_text()"),
        ("        if len(one) > L1_MAX_CHARS:", "        if False:"),
    )
    skills = mutant.discover()
    inflated = mutant.approx_tokens(mutant.l1_index(skills)) / PROXY_RATIO
    assert inflated > L1_BUDGET_TOKENS * len(skills), (
        "the whole SKILL.md in L1 did not breach the budget, so the budget "
        "assertion is not measuring what it claims"
    )


# ================================================================== T-I2  gate
def test_the_frozen_fixture_is_big_enough_and_balanced() -> None:
    """Guards the denominator. A 20/20 over three rows is not 20/20."""
    positives = [r for r in FIXTURE_ROWS if r[1] != "-"]
    negatives = [r for r in FIXTURE_ROWS if r[1] == "-"]
    assert len(FIXTURE_ROWS) >= 20, f"only {len(FIXTURE_ROWS)} frozen utterances"
    assert len(positives) >= 10 and len(negatives) >= 10
    near_misses = [r for r in negatives if "NEAR MISS" in r[2]]
    assert len(near_misses) >= 4, (
        "the negative half is only informative if some of it nearly matches; "
        "unrelated small talk proves almost nothing about over-firing"
    )


@pytest.mark.parametrize("utterance,expect,note", FIXTURE_ROWS)
def test_the_deterministic_gate_classifies_every_frozen_utterance(
        utterance: str, expect: str, note: str) -> None:
    """T-I2: 20/20 on the frozen set. Honest scope: these 20, not the language."""
    skills = loader.discover()
    hit = loader.match(utterance, skills)
    got = hit[0].name if hit else "-"
    assert got == expect, f"{utterance!r} ({note}): expected {expect}, got {got}"


def test_word_boundaries_apply_to_latin_and_not_to_chinese() -> None:
    skills = loader.discover()
    # latin: a trigger inside a longer word must not fire …
    assert loader.match("an unbiased estimator", skills) == ()
    # … while the same trigger as a word does
    assert loader.match("what bias is that", skills) != ()
    # chinese has no spaces, so mid-sentence substrings must fire
    assert loader.match("那你幫我分析論證好不好", skills) != ()


def test_fullwidth_input_reaches_the_same_trigger() -> None:
    skills = loader.discover()
    assert loader.match("ｒｅｄ－ｔｅａｍ this plan", skills) != ()


def test_matching_is_pure_and_order_stable() -> None:
    skills = loader.discover()
    first = [s.name for s in loader.match("幫我紅隊這個計劃", skills)]
    for _ in range(5):
        assert [s.name for s in loader.match("幫我紅隊這個計劃", skills)] == first


def test_the_loader_never_imports_an_llm_client() -> None:
    """T-I2's second mutant, as a static check: `改成問 LLM 要不要載` violates
    rule 1 (no LLM judge), and the cheapest place to make that unrepresentable is
    the import list of the module that owns the decision."""
    banned = {"boto3", "botocore", "requests", "urllib", "urllib3", "http",
              "socket", "httpx", "rclpy"}
    imported: set[str] = set()
    for node in ast.walk(ast.parse(LOADER_SRC.read_text())):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & banned, f"skills/loader.py imports {sorted(imported & banned)}"


def test_importing_the_loader_pulls_in_no_client_at_runtime() -> None:
    """The import list can be clean while a deferred `import boto3` sits inside a
    function, so this asks the interpreter instead of the source."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import skills.loader; "
         "bad=[m for m in ('boto3','botocore','rclpy') if m in sys.modules]; "
         "print(bad); raise SystemExit(1 if bad else 0)" % str(REPO_ROOT)],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"importing skills.loader loaded {proc.stdout.strip()}"


@pytest.mark.mutation
def test_mutant_loading_every_skill_always_fails_the_negative_rows(tmp_path: Path) -> None:
    """T-I2's first named mutant: `改成常駐載入`."""
    mutant = load_mutant(
        tmp_path,
        ("    return tuple(s for s in skills if trigger_hits(utterance, s))",
         "    return tuple(skills)"),
    )
    skills = mutant.discover()
    missed = [r[0] for r in FIXTURE_ROWS if r[1] == "-" and mutant.match(r[0], skills)]
    assert len(missed) >= 10, (
        "an always-resident loader still passed the negative rows, so the gate test "
        "is not testing the gate"
    )


@pytest.mark.mutation
def test_mutant_asking_bedrock_whether_to_load_trips_the_static_check(tmp_path: Path) -> None:
    """The other half of T-I2's second mutant: prove the static check has teeth."""
    dst = tmp_path / "skills"
    shutil.copytree(SKILLS_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    target = dst / "loader.py"
    target.write_text("import boto3\n" + target.read_text())
    imported: set[str] = set()
    for node in ast.walk(ast.parse(target.read_text())):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
    assert "boto3" in imported


# ============================================================= T-I3  traversal
LEGIT_READS = [
    "critical-thinking/SKILL.md",
    "critical-thinking/methods/bias-detection.md",
    "critical-thinking/templates/fallacy-reference.md",
    "critical-thinking/references/nietzsche.md",
]

ESCAPE_ATTEMPTS = [
    "../pytest.ini",
    "../../etc/passwd",
    "critical-thinking/../../README.md",
    "critical-thinking/methods/../../../LICENSE",
    "./../agent/agent_node.py",
    "/etc/passwd",
    "//etc/passwd",
    "\\etc\\passwd",
    "critical-thinking/SKILL.md/../../../pytest.ini",
    "..",
    ".",
    "",
    "   ",
    "critical-thinking",                      # a directory, not a file
    "critical-thinking/nope.md",              # inside, but absent
]


@pytest.mark.parametrize("rel", LEGIT_READS)
def test_legitimate_skill_files_are_readable(rel: str) -> None:
    """The positive control for T-I3. Without it, `raise` on every input passes the
    whole traversal suite — 0 escapes and 0 capability look identical."""
    body = loader.read_skill_file(rel)
    assert len(body) > 200 and "\n" in body


@pytest.mark.parametrize("rel", ESCAPE_ATTEMPTS)
def test_no_path_escapes_the_skills_root(rel: str) -> None:
    with pytest.raises(loader.SkillPathError):
        loader.read_skill_file(rel)


def test_a_nul_byte_is_refused_before_it_reaches_the_filesystem() -> None:
    with pytest.raises(loader.SkillPathError, match="NUL"):
        loader.read_skill_file("critical-thinking/SKILL.md\x00.txt")


def test_a_symlink_out_of_the_tree_is_refused(tmp_path: Path) -> None:
    """The case a string check cannot see: a legal relative path, no '..' anywhere,
    pointing at a file outside the tree."""
    secret = tmp_path / "credentials"
    secret.write_text("aws_secret_access_key = not-a-real-key\n")
    root = tmp_path / "skills"
    (root / "critical-thinking").mkdir(parents=True)
    (root / "critical-thinking" / "SKILL.md").write_text("---\nname: x\n---\n")
    (root / "critical-thinking" / "leak.md").symlink_to(secret)

    with pytest.raises(loader.SkillPathError, match="outside"):
        loader.read_skill_file("critical-thinking/leak.md", root=root)


def test_a_symlink_inside_the_tree_is_still_allowed(tmp_path: Path) -> None:
    """Containment, not link-phobia: refusing every symlink would pass the escape
    tests for the wrong reason."""
    root = tmp_path / "skills"
    (root / "s").mkdir(parents=True)
    real = root / "s" / "real.md"
    real.write_text("x" * 300 + "\n")
    (root / "s" / "alias.md").symlink_to(real)
    assert len(loader.read_skill_file("s/alias.md", root=root)) > 200


def test_an_oversized_file_is_truncated_not_silently_halved(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "s").mkdir(parents=True)
    (root / "s" / "big.md").write_text("y" * 5000)
    out = loader.read_skill_file("s/big.md", root=root, max_bytes=1000)
    assert out.startswith("y" * 1000)
    assert "truncated at 1000 bytes of 5000" in out


@pytest.mark.mutation
def test_mutant_string_check_instead_of_resolution_lets_the_symlink_out(tmp_path: Path) -> None:
    """T-I3's named mutant: `移除路徑正規化`.

    The mutant is the plausible-looking version — reject '..', accept the rest —
    which is exactly what a reviewer might write. It must leak the file the real
    implementation refuses, otherwise the symlink test proves nothing about
    resolution.
    """
    mutant = load_mutant(
        tmp_path,
        ("    target = (root / candidate).resolve()", "    target = root / candidate"),
        ("    if target != root and root not in target.parents:",
         "    if '..' in rel:"),
    )
    secret = tmp_path / "credentials"
    secret.write_text("aws_secret_access_key = not-a-real-key\n")
    root = Path(mutant.__file__).resolve().parent
    (root / "critical-thinking" / "leak.md").symlink_to(secret)

    leaked = mutant.read_skill_file("critical-thinking/leak.md", root=root)
    assert "not-a-real-key" in leaked, (
        "the mutant did not leak, so the escape suite may be passing for an "
        "unrelated reason"
    )
    with pytest.raises(loader.SkillPathError):
        loader.read_skill_file("critical-thinking/leak.md", root=root)


# ================================================== the tool the model can call
def test_the_loader_no_longer_owns_a_tool_surface() -> None:
    """S8 moved the name, the Converse spec, the hop bound and the dispatch into
    `agent/tools.py`; what stays here is `read_skill_file` — the executor and the
    path fence of T-I3, tested above.

    The split is not tidying. While this module owned `TOOL_CONFIG`, "which tools
    can the model reach" was answered by the module that knows about skills, so a
    second tool would have been appended next to the first one and the whole tool
    surface would have kept the name of one subsystem. The invocation contract and
    its six outcomes now live in `tests/test_tools.py` (T-L1..T-L5).
    """
    for gone in ("TOOL_CONFIG", "TOOL_NAME", "run_tool", "MAX_TOOL_HOPS"):
        assert not hasattr(loader, gone), (
            f"loader.{gone} is back; agent/tools.py is supposed to be the only "
            f"creator of the tool surface")


# ====================================================== the wiring in agent_node
def test_agent_node_gates_on_the_loader_and_not_on_the_model() -> None:
    src = AGENT_NODE.read_text()
    assert "loader.match(" in src, "agent_node must ask the deterministic gate"
    assert "loader.l1_index(" in src, "the L1 index must reach the system prompt"
    assert "tools.MAX_TOOL_HOPS" in src, "the tool loop must be bounded"


def test_the_skill_body_never_enters_the_system_block() -> None:
    """Same invariant as T-C5 for recalled memory: loaded material is data in the
    turn, never instructions in the system block. Checked textually because
    importing agent_node needs rclpy."""
    for lineno, line in enumerate(AGENT_NODE.read_text().splitlines(), 1):
        if "system=" in line:
            assert "skill_block" not in line and "load_l2" not in line, (
                f"agent_node.py:{lineno} puts skill text in the system block")


def test_model_id_matches_the_measurement_script() -> None:
    """T-I1's real figure is tokenizer-specific, so it stops being evidence the day
    the node and the measurement script disagree about the model."""
    def model_id(path: Path) -> str:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "MODEL_ID" for t in node.targets):
                return ast.literal_eval(node.value)
        raise AssertionError(f"no MODEL_ID assignment in {path}")

    assert model_id(AGENT_NODE) == model_id(SKILLS_DIR / "measure_l1_tokens.py")
