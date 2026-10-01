"""The numbers the documents quote, checked against pytest collection (T-J8).

This module exists because of a defect, not a principle. On 2026-08-23 `docs/test-design.md`
carried a §0 table row reading ``tests/test_meta.py | 25`` and, four lines below it, a
sentence saying that file's offline import scan "went from 17 to 23". Both numbers describe
the same file; they cannot both be right; the suite was fully green. Five more figures
(the suite total and the mutant total) are repeated across `README.md` and
`README.zh-TW.md`, so every PR that adds a test invalidates hand-typed numbers in three
documents at once.

The tests below are therefore about a specific failure mode: **a stale document reporting
clean**. So the audit is exercised on deliberately damaged copies of the real documents,
one damage per case, and each check gets a mutant that proves the check is what noticed.

Note the two cases that are not "a number drifted": a document whose sentence was reworded
away, and a document where the sentence appears twice. Both must be findings. A checker
that treats a missing sentence as "nothing to check" lets a rewrite switch off its own
gate one line at a time, which is the same shape as a scan that reads zero files and exits
0 (§0 rule 5).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DOC_COUNTS_PY = REPO_ROOT / "infra" / "doc_counts.py"

sys.path.insert(0, str(REPO_ROOT / "infra"))
import doc_counts  # noqa: E402


def load_mutant(tmp_path: Path, *replacements: tuple[str, str]):
    """Import a text-patched copy of infra/doc_counts.py.

    Zero replacements is the no-mutant control (§0 rule 2). `audit()` is pure, so the copy
    needs nothing beside it — the copy's own REPO_ROOT points at tmp_path and is unused.
    """
    target = tmp_path / "doc_counts.py"
    src = DOC_COUNTS_PY.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in src, f"mutant anchor no longer present: {old!r}"
        assert src.count(old) == 1, f"mutant anchor is not unique: {old!r}"
        src = src.replace(old, new)
    target.write_text(src, encoding="utf-8")
    name = f"doc_counts_mutant_{tmp_path.name}_{len(replacements)}"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


MISSING_SENTENCE = ('        if got is None:\n'
                    '            findings.append(f"{doc}: cannot check the {what} — {why}")')
DUPLICATE_SENTENCE = ('    if len(found) > 1:\n'
                      '        return None, f"the sentence appears {len(found)} times, '
                      'so no single number is quoted"')
ROW_DRIFT = ('        elif documented[path] != tested[path]:')
ROW_UNION = ('    for path in sorted(set(documented) | set(tested)):')
SCAN_INVARIANT = ('        if module_count != scan_cases:')
SEAM_ROWS_JOIN = ('    for doc, pattern, expected, what in checks + seam_checks:')
ROW_L3_DRIFT = ('        elif int(said[0]) != observed_l3:')
ROW_L3_UNNUMBERED = ('            if observed_l3:')
PROSE_CASES_LOOP = ('    for path, said in prose_cases:')
PROSE_CASES_RAN = ('    if not prose_cases:')
WRAP_NORMALISE = ('    found = re.findall(pattern, re.sub(r"\\s+", " ", text), re.S)')


# --------------------------------------------------------------- observed reality
@pytest.fixture(scope="module")
def observed() -> dict:
    """One collection pass, shared: three nested pytest runs is enough to pay for once."""
    per_file = doc_counts.collect_counts()
    assert per_file, "pytest --collect-only produced no per-module lines"
    mutation = doc_counts.collect_counts("mutation")
    assert mutation, "pytest -m mutation --collect-only produced no per-module lines"
    functions = doc_counts.test_function_counts(REPO_ROOT)
    assert functions, "the AST walk found no test functions under tests/"
    seams = doc_counts.seam_row_count(
        (REPO_ROOT / doc_counts.SEAMS_TSV).read_text(encoding="utf-8"))
    assert seams > 10, "the dormancy registry read as almost empty"
    return {
        "per_file": per_file,
        "mutation_per_file": mutation,
        "mutation_total": sum(mutation.values()),
        "module_count": len(list((REPO_ROOT / "tests").glob("*.py"))),
        "scan_cases": doc_counts.scan_case_count(),
        "test_functions": functions,
        "seam_rows": seams,
    }


@pytest.fixture(scope="module")
def docs() -> dict[str, str]:
    return {name: (REPO_ROOT / name).read_text(encoding="utf-8") for name in doc_counts.DOCS}


def kw(observed: dict, **override) -> dict:
    """The observed reality as `audit()` keyword arguments.

    One place, because the mutant cases call `audit()` on a patched copy of the module and a
    positional call list goes stale the moment the signature grows — which it did in R0b-2.
    """
    kwargs = {"per_file": observed["per_file"],
              "mutation_per_file": observed["mutation_per_file"],
              "module_count": observed["module_count"],
              "scan_cases": observed["scan_cases"],
              "test_functions": observed["test_functions"],
              "seam_rows": observed["seam_rows"]}
    kwargs.update(override)
    return kwargs


def run(docs: dict[str, str], observed: dict, **override) -> list[str]:
    return doc_counts.audit(docs, **kw(observed, **override))


def bumped_mutants(observed: dict, delta: int) -> dict[str, int]:
    """The mutation total moved without touching any module whose figure a document quotes.

    Deliberately a module that does not exist: bumping `tests/test_store.py` would move the
    mutation total *and* the store figure §4.5 quotes, and the case could not then say which
    check had noticed.
    """
    return dict(observed["mutation_per_file"], **{"tests/test_not_a_module.py": delta})


def test_the_documents_as_shipped_agree_with_collection(docs, observed) -> None:
    """The live gate. If this is red, a number in a document is stale — fix the document,
    not this test."""
    assert run(docs, observed) == []


def test_the_scan_is_one_case_per_module(observed) -> None:
    """The relation the §0 paragraph claims: the offline import scan is parameterised over
    `tests/*.py`, so it cannot fall behind a new module the way a written list would."""
    assert observed["scan_cases"] == observed["module_count"] > 10


# ------------------------------------------------------- one damaged copy per check
SUITE, MUTANTS = "suite", "mutants"
TOOLS_CASES, TOOLS_FUNCS, STORE_MUTANTS = "tools_cases", "tools_funcs", "store_mutants"
SEAM_ROWS = "seam_rows"
QUOTED = [
    # R1.2 reworded this one from "真正跑起來的是" to "被收集到的是": the gate computes
    # `--collect-only`, and the verification run was 1668 passed plus one host-conditional
    # skip. Both readers of the sentence had to move together, which is the whole reason this
    # list is a mirror of `doc_counts.checks` and not a second source of truth.
    ("docs/test-design.md", "今天被收集到的是 {} 個案例", SUITE, "suite total"),
    ("docs/test-design.md", "全套 {} 個案例", MUTANTS, "mutation total"),
    ("docs/test-design.md", "選到 **{} 個案例**", MUTANTS, "mutation total (T-J4)"),
    ("README.md", "Offline suite, {} cases", SUITE, "suite total"),
    ("README.md", "the mutants alone ({} cases on", MUTANTS, "mutation total"),
    ("README.zh-TW.md", "離線測試 {} 個案例", SUITE, "suite total"),
    ("README.zh-TW.md", "量到 {} 個案例", MUTANTS, "mutation total"),
    # `pytest.ini` is not a document under docs/, and that is the point: the `mutation`
    # marker's help text quotes the whole-layer figures so they sit beside the marker they
    # describe, and they were two rounds stale (1512 cases, 304 mutants) while every gate was
    # green. Both patterns tolerate a line wrap, because the ini file's help text is reflowed
    # by hand and rule 5 would otherwise turn a re-wrap into a finding about nothing.
    ("pytest.ini", "the {}-case suite", SUITE, "suite total"),
    ("pytest.ini", "the {} mutants were", MUTANTS, "mutation total"),
    # §3's own wall-clock paragraphs, found by grepping for the number rather than for an
    # already-gated sentence. One states the budget overrun as a multiple and the other states
    # that the two duration sets have different denominators; both quantities are only readable
    # if the denominator is right, so these are the last two places a count should be typed.
    ("docs/test-design.md", "今天是 {} 個案例", SUITE, "suite total (§3 budget overrun)"),
    ("docs/test-design.md", "對**整個層**({} 個案例", SUITE, "suite total (§3 denominator)"),
    ("docs/test-design.md", "個案例、其中 {} 個\n突變體)", MUTANTS,
     "mutation total (§3 denominator)"),
    # R0b-2: the three documents the widened `DOCS` tuple brought in. Each is here because
    # the tuple alone checks nothing — a path with no pattern quoting a number out of it is
    # coverage in name only, which `test_every_document_in_the_tuple_is_read` states as a
    # difference rather than as a comment.
    ("docs/design.md", "({} collected cases from", TOOLS_CASES, "test_tools case count"),
    ("docs/design.md", "collected cases from {} test functions", TOOLS_FUNCS,
     "test_tools function count"),
    ("docs/design.zh-TW.md", "收集到 {} 個案例", TOOLS_CASES, "test_tools case count"),
    ("docs/design.zh-TW.md", "來自 {} 個測試函式", TOOLS_FUNCS, "test_tools function count"),
    ("docs/plan.md", "({} cases per `docs/test-design.md`", TOOLS_CASES,
     "test_tools case count"),
    ("docs/design.md", "**{} mutation cases** in `tests/test_store.py`", STORE_MUTANTS,
     "test_store mutation case count"),
    ("docs/design.zh-TW.md", "`tests/test_store.py` 的 **{} 個突變案例", STORE_MUTANTS,
     "test_store mutation case count"),
    ("docs/plan.md", "the {} mutation cases of\n`tests/test_store.py`", STORE_MUTANTS,
     "test_store mutation case count"),
    # The one figure here that is not a pytest number, and the reason it was added: adding
    # `infra/doc_status.py` to the dormancy registry in this same PR moved this count in four
    # documents at once, and every gate stayed green. `docs/design.md` §1 tells the reader to
    # quote this number *instead of* the dormancy percentage, which makes it the last one
    # that should have been hand-typed.
    ("docs/design.md", "reports **{}** today", SEAM_ROWS, "dormant seam count"),
    ("docs/design.zh-TW.md", "今天報 **{}**", SEAM_ROWS, "dormant seam count"),
    ("docs/plan.md", "`infra/dormant_seams.tsv` ({} rows)", SEAM_ROWS, "dormant seam count"),
    ("docs/test-design.md", "「今天這 {} 列是對的」", SEAM_ROWS, "dormant seam count"),
    # Found by grepping for the number rather than for the sentence: two of these were still
    # 42 while the four above were 43, inside the PR that added the seam check. Each phrasing
    # needs its own row because a pattern is a claim about one sentence.
    ("docs/test-design.md", "的 {} 列雙向對帳", SEAM_ROWS, "dormant seam count (§0 row)"),
    ("docs/test-design.md", "的 {} 列對稱差", SEAM_ROWS, "dormant seam count (T-N1)"),
    ("docs/design.md", "carries all {} with a named reason", SEAM_ROWS,
     "dormant seam count (the registry's own row count)"),
    ("docs/design.zh-TW.md", "{} 列全部帶具名理由", SEAM_ROWS,
     "dormant seam count (the registry's own row count)"),
    ("docs/plan.md", "reports **{}** seams reachable", SEAM_ROWS,
     "dormant seam count (the census figure)"),
    ("docs/plan.md", "carries {}\nrows with named reasons", SEAM_ROWS,
     "dormant seam count (the registry's own row count)"),
]


def real_value(kind: str, observed: dict) -> int:
    return {
        SUITE: sum(observed["per_file"].values()),
        MUTANTS: observed["mutation_total"],
        TOOLS_CASES: observed["per_file"]["tests/test_tools.py"],
        TOOLS_FUNCS: observed["test_functions"]["tests/test_tools.py"],
        STORE_MUTANTS: observed["mutation_per_file"]["tests/test_store.py"],
        SEAM_ROWS: observed["seam_rows"],
    }[kind]


def bump(docs: dict[str, str], doc: str, template: str, kind: str,
         observed: dict) -> dict[str, str]:
    """Replace one quoted figure with a wrong one, leaving everything else alone.

    The phrase is matched with each space allowed to be any run of whitespace, for the
    same reason `doc_counts.quoted` collapses whitespace before matching: a markdown
    paragraph's line breaks are not part of its prose. A literal `str.replace` here made
    this case go red the day a paragraph in `design.md` was re-wrapped — and it went red
    with `the case is vacuous`, which reads as "this assertion checks nothing" rather
    than "the phrase moved to the next line". Two readers of one format have to agree on
    what the format is, or the one that is only in the test drifts.
    """
    real = real_value(kind, observed)
    old = template.format(real)
    pattern = r"\s+".join(re.escape(part) for part in old.split(" "))
    hits = re.findall(pattern, docs[doc])
    assert len(hits) == 1, f"the case is vacuous: {old!r} appears " \
                           f"{len(hits)} times in {doc}"
    patched = dict(docs)
    patched[doc] = re.sub(pattern, template.format(real + 7).replace("\\", "\\\\"),
                          docs[doc], count=1)
    return patched


def test_every_document_in_the_tuple_is_read(docs, observed) -> None:
    """`DOCS` is a claim that each listed file is checked; blanking one must be noticed.

    R0b-2 widened the tuple, and a widened tuple with no pattern quoting a number out of the
    new file reads as coverage while checking nothing. Derived rather than written down: every
    path is emptied in turn and the audit has to say something about *that* path.
    """
    for doc in doc_counts.DOCS:
        findings = run(dict(docs, **{doc: ""}), observed)
        assert any(f.startswith(f"{doc}:") for f in findings), (
            f"{doc} is in DOCS but emptying it produces no finding, so nothing reads it:\n"
            + "\n".join(findings)
        )


def with_total(docs: dict[str, str], observed: dict, delta: int) -> dict[str, str]:
    """The three documents' suite totals rewritten by `delta`, and nothing else.

    Needed because "a new module has no row" cannot be tested at a constant total: adding
    a module adds cases. Rather than let two checks fire and pick one out of the list, move
    the documents to where they would be if someone had updated only the headline — then
    the missing row is the single remaining disagreement.
    """
    patched = dict(docs)
    total = sum(observed["per_file"].values())
    for doc, template, kind, _ in QUOTED:
        if kind != SUITE:
            continue
        old = template.format(total)
        assert patched[doc].count(old) == 1, f"{doc}: {old!r} is not the only copy"
        patched[doc] = patched[doc].replace(old, template.format(total + delta))
    return patched

@pytest.mark.parametrize("doc,template,kind,what", QUOTED,
                         ids=[f"{d}:{w}" for d, _, _, w in QUOTED])
def test_a_drifted_headline_is_a_finding(docs, observed, doc, template, kind, what) -> None:
    findings = run(bump(docs, doc, template, kind, observed), observed)
    assert len(findings) == 1, findings
    assert findings[0].startswith(f"{doc}: {what} says"), findings[0]


def test_the_sentinel_tracks_the_mutation_total_rather_than_being_its_own_number(
        docs, observed) -> None:
    """`T-J7 保證沒有第 N 個` is one past the mutant count, so a new mutant moves both
    numbers. A sentinel that only moved when someone remembered it is decoration."""
    findings = run(docs, observed, mutation_per_file=bumped_mutants(observed, 1))
    quoted = [f for f in findings if "sentinel" in f]
    assert len(quoted) == 1, findings
    assert f"observed {observed['mutation_total'] + 2}" in quoted[0]


def moved(observed: dict, delta: dict[str, int]) -> dict:
    """Per-file counts perturbed **at a constant total**.

    Learned from this test failing: bumping one module by 3 also bumps the suite total by
    3, so the case produced four findings and could not tell which check had noticed. The
    row checks and the headline checks have to be exercisable one at a time, which means
    moving cases between modules rather than adding them.
    """
    per_file = dict(observed["per_file"])
    for path, d in delta.items():
        per_file[path] = per_file.get(path, 0) + d
    assert sum(per_file.values()) == sum(observed["per_file"].values()), \
        "this helper must not move the suite total, or the case tests two things at once"
    return per_file


def test_a_row_whose_count_drifted_is_a_finding(docs, observed) -> None:
    per_file = moved(observed, {"tests/test_opponent.py": 3, "tests/test_doubles.py": -3})
    findings = run(docs, observed, per_file=per_file)
    assert len(findings) == 2, findings
    assert any("tests/test_opponent.py row says 74, pytest collects 77" in f
               for f in findings), findings
    assert any("tests/test_doubles.py row says 16, pytest collects 13" in f
               for f in findings), findings


def test_a_module_with_no_row_in_the_table_is_a_finding(docs, observed) -> None:
    """The direction a hand-written table cannot notice: a brand new test file whose cases
    are real but which appears in no count (§0 rule 5 — unlisted is uncounted)."""
    per_file = dict(observed["per_file"], **{"tests/test_brand_new_thing.py": 12})
    findings = run(with_total(docs, observed, 12), observed, per_file=per_file)
    assert len(findings) == 1, findings
    assert "tests/test_brand_new_thing.py collects 12 cases but has no row" in findings[0]


def test_a_row_for_a_module_that_collects_nothing_is_a_finding(docs, observed) -> None:
    patched = dict(docs)
    patched["docs/test-design.md"] = docs["docs/test-design.md"].replace(
        "| `tests/test_opponent.py` | 74 |",
        "| `tests/test_deleted.py` | 9 |\n| `tests/test_opponent.py` | 74 |")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert "lists tests/test_deleted.py, which collects nothing" in findings[0]


def test_a_reworded_sentence_is_a_finding_rather_than_a_pass(docs, observed) -> None:
    """The rule-5 half. Deleting the sentence a check reads must not exempt the check."""
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace("Offline suite, ", "Offline tests, ")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert "cannot check the suite total" in findings[0]
    assert "no longer in the document" in findings[0]


def test_a_sentence_appearing_twice_is_a_finding(docs, observed) -> None:
    """Two copies of the claim is worse than none: the checker would read one and the
    reader the other, which is exactly how the 25-vs-23 pair survived."""
    total = sum(observed["per_file"].values())
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace(
        f"Offline suite, {total} cases",
        f"Offline suite, {total} cases (elsewhere: Offline suite, {total - 9} cases)")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert "appears 2 times" in findings[0]


def test_rewrapping_a_paragraph_is_not_a_reworded_sentence(docs, observed) -> None:
    """The other half of rule 5, and the half R1.1 needed: a markdown line break is not prose.

    `README.md`'s sentence is re-wrapped at the space before the number — every word
    untouched, one space become a newline. Before R1.1 this reported `cannot check the suite
    total`, which is a true statement about the pattern and a false one about the document,
    and the cheapest way out of it is to un-wrap the paragraph instead of fixing the pattern.
    """
    total = sum(observed["per_file"].values())
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace(
        f"Offline suite, {total} cases", f"Offline suite,\n{total} cases")
    assert patched["README.md"] != docs["README.md"], "the wrap target is no longer present"
    assert run(patched, observed) == [], "a re-wrapped line is being read as a reword"


@pytest.mark.mutation
def test_mutant_reading_the_raw_text_calls_a_line_wrap_a_reword(
        docs, observed, tmp_path: Path) -> None:
    """Removing the whitespace collapse turns line wrapping into phantom rewords.

    Two demonstrations, because the second one alone would be a synthetic case. The
    first is that the mutant is red on **the tree as shipped**: `docs/design.md` really
    does wrap "carries all 41 with a named reason" across a line, so the mutant reports a
    reword that never happened. The second adds one more by re-wrapping `README.md`, which
    shows the finding count tracks the wraps rather than being a single fixed complaint.
    """
    mutant = load_mutant(tmp_path,
                         (WRAP_NORMALISE, "    found = re.findall(pattern, text, re.S)"))
    as_shipped = mutant.audit(dict(docs), **kw(observed))
    assert len(as_shipped) == 1, as_shipped
    assert "cannot check the dormant seam count" in as_shipped[0]
    assert "no longer in the document" in as_shipped[0]

    total = sum(observed["per_file"].values())
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace(
        f"Offline suite, {total} cases", f"Offline suite,\n{total} cases")
    assert patched["README.md"] != docs["README.md"], "the wrap target is no longer present"
    rewrapped = mutant.audit(patched, **kw(observed))
    assert len(rewrapped) == 2, rewrapped
    assert any("cannot check the suite total" in f for f in rewrapped), rewrapped


def test_the_seam_count_is_the_registry_row_count_and_not_its_line_count(observed) -> None:
    """Derived from the file that states the claim, with the two things that are not rows.

    The registry's header is fifty-odd lines of comment explaining the three granularities,
    so a line count would report a number three times the truth — and it would still move in
    the right direction whenever a row was added, which is how a wrong derivation survives.
    That the registry's row count equals what the census measures is `infra/wiring_census.py`'s
    own two-directional check, not this one's.
    """
    synthetic = "# a comment\n\n# another\nmemory/recall.py\tunreached\tpending_stage\tR1.4\n"
    assert doc_counts.seam_row_count(synthetic) == 1
    real = (REPO_ROOT / doc_counts.SEAMS_TSV).read_text(encoding="utf-8")
    assert observed["seam_rows"] == doc_counts.seam_row_count(real) < len(real.splitlines())


def test_the_scan_falling_behind_the_directory_is_a_finding(docs, observed) -> None:
    findings = run(docs, observed, scan_cases=observed["scan_cases"] - 1)
    assert any("no longer one-per-module" in f for f in findings), findings


# ------------------------------------------- the row's second number: its L3 total
#
# Every §0 row states two quantities — how many cases the module collects and how many of
# them the mutation layer selects — and only the first had a reader. The three that were
# wrong when the check was written are the shape of these cases: one row 13-vs-12, and two
# rows that stated a breakdown of named mutants (17 + 1, 8 + 2) without a total, so
# collection had nothing to disagree with.
L3_PATH = "tests/test_opponent.py"


def row_line(design: str, path: str) -> str:
    """The one §0 row for `path`, so a case can patch it without re-typing its prose."""
    lines = [l for l in design.splitlines() if l.startswith(f"| `{path}` |")]
    assert len(lines) == 1, f"{path} has {len(lines)} rows in the §0 table"
    return lines[0]


def patch_row(docs: dict[str, str], path: str, old: str, new: str) -> dict[str, str]:
    """Rewrite one row of the §0 table, asserting the edit lands inside that row only."""
    design = docs["docs/test-design.md"]
    line = row_line(design, path)
    assert line.count(old) == 1, f"the case is vacuous: {old!r} is not unique in {path}'s row"
    return dict(docs, **{"docs/test-design.md": design.replace(line, line.replace(old, new))})


def l3_phrase(observed: dict, path: str = L3_PATH) -> str:
    """The row's own L3 sentence, built from the observed count rather than remembered."""
    return f"L3 選到 {observed['mutation_per_file'][path]} 個"


def test_a_drifted_l3_total_in_a_row_is_a_finding(docs, observed) -> None:
    real = observed["mutation_per_file"][L3_PATH]
    patched = patch_row(docs, L3_PATH, l3_phrase(observed), f"L3 選到 {real + 5} 個")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert f"{L3_PATH} L3 total says {real + 5}, pytest -m mutation collects {real}" \
        in findings[0], findings[0]


def test_a_row_stating_no_l3_total_while_the_layer_collects_is_a_finding(docs, observed) -> None:
    """The two rows that were wrong this way stated their named mutants and stopped. A
    breakdown is not a total: 17 + 1 is only 18 to a reader who adds it up."""
    patched = patch_row(docs, L3_PATH, l3_phrase(observed), "L3 有好幾個")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert f"{L3_PATH} states no L3 total" in findings[0], findings[0]
    assert "unnumbered layer is an uncounted one" in findings[0]


def test_a_row_claiming_an_l3_total_for_an_empty_layer_is_a_finding(docs, observed) -> None:
    """The other direction, and the one a deleted mutant leaves behind: the row still
    advertises a mutation layer for a module that now has none, and it reads as coverage."""
    design = docs["docs/test-design.md"]
    empty = sorted(m.group(1) for m in doc_counts.TABLE_ROW.finditer(design)
                   if not observed["mutation_per_file"].get(m.group(1)))
    assert empty, "no §0 row describes a module without a mutation layer, so this case is moot"
    line = row_line(design, empty[0])
    assert line.endswith("|")
    patched = dict(docs, **{"docs/test-design.md":
                            design.replace(line, line[:-1] + "(L3 選到 4 個) |")})
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert f"{empty[0]} says L3 selects 4 cases, but the mutation layer collects nothing" \
        in findings[0], findings[0]


def test_two_l3_totals_in_one_row_is_a_finding(docs, observed) -> None:
    """Rule 5 again, per row: with two totals in one cell the checker reads one and the
    reader the other, which is the 25-vs-23 pair with a narrower blast radius."""
    phrase = l3_phrase(observed)
    patched = patch_row(docs, L3_PATH, phrase, f"{phrase}、{phrase}")
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert f"{L3_PATH} states 2 L3 totals" in findings[0], findings[0]


# ------------------------------------ the same counts where they appear in running prose
def prose_site(design: str) -> tuple[str, str, int]:
    """One `` `tests/test_x.py` N 個案例 `` site: its text, its module, its number.

    Derived from the document rather than named, because the point of the check is that a
    paragraph written next month is read the day it is written. The first site whose text is
    unique is used, so the patch cannot land in two places at once.
    """
    for m in doc_counts.PROSE_CASES.finditer(design):
        if design.count(m.group(0)) == 1:
            return m.group(0), m.group(1), int(m.group(2))
    raise AssertionError("no prose site quotes a module's case count exactly once")


def test_prose_quoting_a_stale_module_count_is_a_finding(docs, observed) -> None:
    design = docs["docs/test-design.md"]
    text, path, said = prose_site(design)
    patched = dict(docs, **{"docs/test-design.md":
                            design.replace(text, text.replace(f"{said} 個案例",
                                                              f"{said + 8} 個案例"))})
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert f"prose says {path} has {said + 8} cases, pytest collects {said}" in findings[0], \
        findings[0]


def test_prose_quoting_a_module_that_collects_nothing_is_a_finding(docs, observed) -> None:
    """A renamed test file leaves its old name in a paragraph, and the number stays right."""
    design = docs["docs/test-design.md"]
    text, path, _ = prose_site(design)
    patched = dict(docs, **{"docs/test-design.md":
                            design.replace(text, text.replace(path, "tests/test_gone.py"))})
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert "for tests/test_gone.py, which collects nothing" in findings[0], findings[0]


# ------------------------------------------------------------------- the mutants
@pytest.mark.mutation
def test_mutant_skipping_an_unreadable_sentence_lets_a_rewrite_switch_off_its_own_gate(
        docs, observed, tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (MISSING_SENTENCE,
                                    '        if got is None:\n            continue'))
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace("Offline suite, ", "Offline tests, ")
    assert mutant.audit(patched, **kw(observed)) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_reading_the_first_of_two_copies_misses_the_stale_one(
        docs, observed, tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (DUPLICATE_SENTENCE, "    pass"))
    total = sum(observed["per_file"].values())
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace(
        f"Offline suite, {total} cases",
        f"Offline suite, {total} cases (elsewhere: Offline suite, {total - 9} cases)")
    assert mutant.audit(patched, **kw(observed)) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_without_the_row_comparison_accepts_any_number_in_the_table(
        docs, observed, tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (ROW_DRIFT, '        elif False:'))
    per_file = moved(observed, {"tests/test_opponent.py": 3, "tests/test_doubles.py": -3})
    assert mutant.audit(docs, **kw(observed, per_file=per_file)) == []
    assert run(docs, observed, per_file=per_file) != []


@pytest.mark.mutation
def test_mutant_iterating_the_table_cannot_notice_a_name_it_does_not_contain(
        docs, observed, tmp_path: Path) -> None:
    """A written-down list is blind to a new name by construction — the loop has to run
    over the union of what the table says and what pytest found, and this mutant is the
    version that runs over the table alone."""
    mutant = load_mutant(tmp_path, (ROW_UNION, '    for path in sorted(documented):'))
    per_file = dict(observed["per_file"], **{"tests/test_brand_new_thing.py": 12})
    patched = with_total(docs, observed, 12)
    assert mutant.audit(patched, **kw(observed, per_file=per_file)) == []
    assert run(patched, observed, per_file=per_file) != []


@pytest.mark.mutation
def test_mutant_dropping_the_scan_invariant_only_checks_what_the_document_says(
        docs, observed, tmp_path: Path) -> None:
    """The document could agree with a scan that had silently stopped covering a module —
    the invariant is between the directory and the scan, not between the doc and itself."""
    mutant = load_mutant(tmp_path, (SCAN_INVARIANT, '        if False:'))
    assert mutant.audit(docs, **kw(observed, scan_cases=observed["scan_cases"] - 1)) != []
    theirs = mutant.audit(docs, **kw(observed, scan_cases=observed["scan_cases"] - 1))
    assert not any("one-per-module" in f for f in theirs), theirs


@pytest.mark.mutation
def test_mutant_dropping_the_seam_check_accepts_a_registry_that_grew(
        docs, observed, tmp_path: Path) -> None:
    """The failure this check was written for, replayed: one row added, four documents stale.

    The mutant is the file as it stood before this PR — the seam rows built but never joined
    to the loop. That shape, rather than a falsified comparison, is what actually happened:
    the numbers were in four documents and nothing walked them.
    """
    mutant = load_mutant(
        tmp_path,
        (SEAM_ROWS_JOIN, '    for doc, pattern, expected, what in checks:'))
    grown = observed["seam_rows"] + 1
    assert mutant.audit(docs, **kw(observed, seam_rows=grown)) == []
    assert run(docs, observed, seam_rows=grown) != []


@pytest.mark.mutation
def test_mutant_without_the_l3_comparison_accepts_any_number_in_the_row(
        docs, observed, tmp_path: Path) -> None:
    """The file as it stood before this PR, for the second number: the rows were parsed and
    the L3 figure in each of them was never compared to anything."""
    mutant = load_mutant(tmp_path, (ROW_L3_DRIFT, '        elif False:'))
    real = observed["mutation_per_file"][L3_PATH]
    patched = patch_row(docs, L3_PATH, l3_phrase(observed), f"L3 選到 {real + 5} 個")
    assert mutant.audit(patched, **kw(observed)) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_only_checking_rows_that_state_a_total_cannot_notice_a_missing_one(
        docs, observed, tmp_path: Path) -> None:
    """The half that found two of the three real defects. A checker that only compares the
    totals it *finds* is switched off by deleting the sentence, so the two rows carrying a
    breakdown and no total would have gone on reading as covered."""
    mutant = load_mutant(tmp_path, (ROW_L3_UNNUMBERED, '            if False:'))
    patched = patch_row(docs, L3_PATH, l3_phrase(observed), "L3 有好幾個")
    assert mutant.audit(patched, **kw(observed)) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_reading_only_the_table_leaves_the_prose_unchecked(
        docs, observed, tmp_path: Path) -> None:
    """A module's count is quoted in the table *and* in the paragraphs that discuss it, and
    the table check reads only the table — which is how the original 25-vs-23 pair survived
    four lines apart."""
    mutant = load_mutant(tmp_path, (PROSE_CASES_LOOP, '    for path, said in []:'))
    design = docs["docs/test-design.md"]
    text, _, said = prose_site(design)
    patched = dict(docs, **{"docs/test-design.md": design.replace(
        text, text.replace(f"{said} 個案例", f"{said + 8} 個案例"))})
    assert mutant.audit(patched, **kw(observed)) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_without_the_ran_flag_reports_clean_when_the_phrasing_moved(
        docs, observed, tmp_path: Path) -> None:
    """Zero prose sites is not zero disagreements. Rewording every one of them switches this
    scan off, and without the flag the switch-off prints the same line as a pass."""
    mutant = load_mutant(tmp_path, (PROSE_CASES_RAN, '    if False:'))
    design = docs["docs/test-design.md"]
    reworded = doc_counts.PROSE_CASES.sub(
        lambda m: f"`{m.group(1)}` 收集 {m.group(2)} 個", design)
    assert not doc_counts.PROSE_CASES.search(reworded), "the rewording left a site behind"
    patched = dict(docs, **{"docs/test-design.md": reworded})
    assert mutant.audit(patched, **kw(observed)) == []
    findings = run(patched, observed)
    assert len(findings) == 1, findings
    assert "read nothing; a scan that reads nothing must not report clean" in findings[0]


@pytest.mark.mutation
def test_the_no_mutant_control_reports_the_same_findings(docs, observed, tmp_path: Path) -> None:
    control = load_mutant(tmp_path)
    assert control.audit(docs, **kw(observed)) == run(docs, observed)
    per_file = moved(observed, {"tests/test_opponent.py": 3, "tests/test_doubles.py": -3})
    assert control.audit(docs, **kw(observed, per_file=per_file)) == \
        run(docs, observed, per_file=per_file)


def test_this_checker_is_registered_as_a_standalone_entrypoint() -> None:
    """`infra/check_deploy.sh` treats every `__main__` as something the runner must launch;
    a hand-run guard belongs in the exemption list with its reason, not in a comment."""
    tsv = (REPO_ROOT / "infra" / "standalone_entrypoints.tsv").read_text(encoding="utf-8")
    rows = [l.split("\t") for l in tsv.splitlines() if l and not l.startswith("#")]
    listed = {r[0]: r[1] for r in rows if len(r) == 2}
    assert "infra/doc_counts.py" in listed
    assert re.search(r"\bexit\b", listed["infra/doc_counts.py"]), \
        "the reason must say what running it does, including that it has a third exit code"
