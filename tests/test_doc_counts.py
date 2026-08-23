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


# --------------------------------------------------------------- observed reality
@pytest.fixture(scope="module")
def observed() -> dict:
    """One collection pass, shared: three nested pytest runs is enough to pay for once."""
    per_file = doc_counts.collect_counts()
    assert per_file, "pytest --collect-only produced no per-module lines"
    mutation = doc_counts.collect_counts("mutation")
    assert mutation, "pytest -m mutation --collect-only produced no per-module lines"
    return {
        "per_file": per_file,
        "mutation_total": sum(mutation.values()),
        "module_count": len(list((REPO_ROOT / "tests").glob("*.py"))),
        "scan_cases": doc_counts.scan_case_count(),
    }


@pytest.fixture(scope="module")
def docs() -> dict[str, str]:
    return {name: (REPO_ROOT / name).read_text(encoding="utf-8") for name in doc_counts.DOCS}


def run(docs: dict[str, str], observed: dict, **override) -> list[str]:
    kwargs = {"per_file": observed["per_file"], "mutation_total": observed["mutation_total"],
              "module_count": observed["module_count"], "scan_cases": observed["scan_cases"]}
    kwargs.update(override)
    return doc_counts.audit(docs, **kwargs)


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
QUOTED = [
    ("docs/test-design.md", "今天真正跑起來的是 {} 個案例", SUITE, "suite total"),
    ("docs/test-design.md", "全套 {} 個案例", MUTANTS, "mutation total"),
    ("docs/test-design.md", "選到 **{} 個案例**", MUTANTS, "mutation total (T-J4)"),
    ("README.md", "Offline suite, {} cases", SUITE, "suite total"),
    ("README.md", "the mutants alone ({} cases on", MUTANTS, "mutation total"),
    ("README.zh-TW.md", "離線測試 {} 個案例", SUITE, "suite total"),
    ("README.zh-TW.md", "量到 {} 個案例", MUTANTS, "mutation total"),
]


def bump(docs: dict[str, str], doc: str, template: str, kind: str,
         observed: dict) -> dict[str, str]:
    """Replace one quoted figure with a wrong one, leaving everything else alone."""
    real = (observed["mutation_total"] if kind == MUTANTS
            else sum(observed["per_file"].values()))
    old = template.format(real)
    assert docs[doc].count(old) == 1, f"the case is vacuous: {old!r} appears " \
                                     f"{docs[doc].count(old)} times in {doc}"
    patched = dict(docs)
    patched[doc] = docs[doc].replace(old, template.format(real + 7))
    return patched


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
    findings = run(docs, observed, mutation_total=observed["mutation_total"] + 1)
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


def test_the_scan_falling_behind_the_directory_is_a_finding(docs, observed) -> None:
    findings = run(docs, observed, scan_cases=observed["scan_cases"] - 1)
    assert any("no longer one-per-module" in f for f in findings), findings


# ------------------------------------------------------------------- the mutants
@pytest.mark.mutation
def test_mutant_skipping_an_unreadable_sentence_lets_a_rewrite_switch_off_its_own_gate(
        docs, observed, tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (MISSING_SENTENCE,
                                    '        if got is None:\n            continue'))
    patched = dict(docs)
    patched["README.md"] = docs["README.md"].replace("Offline suite, ", "Offline tests, ")
    assert mutant.audit(patched, observed["per_file"], observed["mutation_total"],
                        observed["module_count"], observed["scan_cases"]) == []
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
    assert mutant.audit(patched, observed["per_file"], observed["mutation_total"],
                        observed["module_count"], observed["scan_cases"]) == []
    assert run(patched, observed) != []


@pytest.mark.mutation
def test_mutant_without_the_row_comparison_accepts_any_number_in_the_table(
        docs, observed, tmp_path: Path) -> None:
    mutant = load_mutant(tmp_path, (ROW_DRIFT, '        elif False:'))
    per_file = moved(observed, {"tests/test_opponent.py": 3, "tests/test_doubles.py": -3})
    assert mutant.audit(docs, per_file, observed["mutation_total"],
                        observed["module_count"], observed["scan_cases"]) == []
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
    assert mutant.audit(patched, per_file, observed["mutation_total"],
                        observed["module_count"], observed["scan_cases"]) == []
    assert run(patched, observed, per_file=per_file) != []


@pytest.mark.mutation
def test_mutant_dropping_the_scan_invariant_only_checks_what_the_document_says(
        docs, observed, tmp_path: Path) -> None:
    """The document could agree with a scan that had silently stopped covering a module —
    the invariant is between the directory and the scan, not between the doc and itself."""
    mutant = load_mutant(tmp_path, (SCAN_INVARIANT, '        if False:'))
    assert mutant.audit(docs, observed["per_file"], observed["mutation_total"],
                        observed["module_count"], observed["scan_cases"] - 1) != []
    theirs = mutant.audit(docs, observed["per_file"], observed["mutation_total"],
                          observed["module_count"], observed["scan_cases"] - 1)
    assert not any("one-per-module" in f for f in theirs), theirs


@pytest.mark.mutation
def test_the_no_mutant_control_reports_the_same_findings(docs, observed, tmp_path: Path) -> None:
    control = load_mutant(tmp_path)
    assert control.audit(docs, observed["per_file"], observed["mutation_total"],
                         observed["module_count"], observed["scan_cases"]) == run(docs, observed)
    per_file = moved(observed, {"tests/test_opponent.py": 3, "tests/test_doubles.py": -3})
    assert control.audit(docs, per_file, observed["mutation_total"],
                         observed["module_count"], observed["scan_cases"]) == \
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
