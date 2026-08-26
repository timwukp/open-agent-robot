#!/usr/bin/env python3
"""Cross-check the case counts written into the documents against pytest collection.

Why this exists, from a defect in this repo rather than from a principle: on 2026-08-23
the §0 table in `docs/test-design.md` said ``tests/test_meta.py | 25`` while the paragraph
directly beneath it said the offline import scan "went from 17 to 23" — two numbers about
the same file that could not both be true, sitting four lines apart, and nothing anywhere
went red. Three more numbers (435 cases, 87 mutants) are repeated across `README.md` and
`README.zh-TW.md`, so every PR that adds a test silently invalidates five hand-typed
figures in three files.

A number in prose is unverified by construction. This script derives each one from
`pytest --collect-only` and reports the disagreements, which turns "remember to update the
README" into a gate.

Two rules from `docs/test-design.md` §0 shape the design:

  rule 5  a scan that reads nothing must not report clean — a regex that fails to match
          its sentence is a **finding**, not a silent pass, because a reworded document
          would otherwise switch its own checks off one at a time.
  rule 6  the numbers live in the collection output; the documents merely quote them.

Exit codes:  0 = the documents agree, 1 = findings, 3 = could not check (no collection
output parsed, or a document missing). 3 rather than 1 because "the checker could not
run" and "the documents are wrong" call for different actions.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DESIGN = "docs/test-design.md"
README_EN = "README.md"
README_ZH = "README.zh-TW.md"
DESIGN_EN = "docs/design.md"
DESIGN_ZH = "docs/design.zh-TW.md"
PLAN = "docs/plan.md"
PYTEST_INI = "pytest.ini"

# Widened in R0b-2. `docs/design.md` said so itself: its own I8 row quoted a collected-case
# count while this tuple could not see the file, so the number was hand-checked inside a
# document arguing that hand-checked numbers are the failure mode. The Chinese counterpart
# is here for the reason the READMEs ship in one PR — two readers of one number each stay
# green while the other side moves.
#
# `pytest.ini` is in the tuple for the sharpest version of that: the `mutation` marker's help
# text carries the whole-layer figures so they cannot drift from the marker they describe,
# and on the day this widening was written those figures still said 1512 cases and 304
# mutants. The prose explaining why a duration must be re-measured per round was itself
# quoting a suite that no longer existed. It is not a Markdown document, which is the point —
# membership here is "a file that quotes one of these numbers", not "a file under docs/".
DOCS = (TEST_DESIGN, README_EN, README_ZH, DESIGN_EN, DESIGN_ZH, PLAN, PYTEST_INI)

# The two test modules whose figures the prose quotes by name.
TOOLS_TESTS = "tests/test_tools.py"
STORE_TESTS = "tests/test_store.py"

# Not a pytest number at all, and here because it went stale the same day the widening
# landed: adding `infra/doc_status.py` to the dormancy registry moved a figure quoted in
# four documents, and nothing noticed. `infra/wiring_census.py` already gates the census
# against this file in both directions, so counting its rows is the cheap derivation — the
# expensive one would re-run the census to learn a number the registry already states.
SEAMS_TSV = "infra/dormant_seams.tsv"

# `tests/test_x.py: 22` — printed per module by tests/conftest.py under --collect-only.
COLLECT_LINE = re.compile(r"^(tests/[A-Za-z0-9_]+\.py):\s*(\d+)\s*$", re.M)
# `| `tests/test_opponent.py` | 74 | T-H1..T-H5(L3 選到 7 個…) |` — the §0 table. Group 3 is
# the prose cell, captured by the same pattern rather than by a second one: two regexes over
# one row would each stay green while the other moved, and the row's own two numbers (its
# collected count and its L3 total) then have a single reader.
TABLE_ROW = re.compile(
    r"^\|\s*`(tests/test_[A-Za-z0-9_]+\.py)`\s*\|\s*(\d+)\s*\|(.*)$", re.M)
# `L3 選到 7 個` / `L3 選到 **17 個**`, the mutation-layer total a row states. Deliberately does
# not match `L3 多選到 2 個`, which is an *incremental* figure two rows carry for one T-ID
# inside a file that also has others — a different quantity, so it gets no free ride here.
ROW_L3 = re.compile(r"L3 選到 \*{0,2}(\d+)")
# `` `tests/test_context.py` 43 個案例 `` / `` `tests/test_architecture.py` 的 43 個案例 `` —
# a module's collected count quoted in running prose, outside the table.
PROSE_CASES = re.compile(r"`(tests/test_[A-Za-z0-9_]+\.py)`\s*(?:的\s*)?(\d+) 個案例")


def collect_counts(marker: str | None = None) -> dict[str, int]:
    """Per-module case counts straight out of pytest, as the only source of truth."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    if marker:
        cmd += ["-m", marker]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {m.group(1): int(m.group(2)) for m in COLLECT_LINE.finditer(proc.stdout)}


def quoted(text: str, pattern: str) -> tuple[int | None, str]:
    """The single number a sentence quotes, or None plus why it could not be read.

    A pattern that matches nothing, or matches twice, returns None — see rule 5 above.
    Rewording the sentence must break the check loudly rather than exempt it.

    Every run of whitespace collapses to one space before matching, because a markdown
    paragraph's line breaks are not part of its prose. R1.1 found this the way it is always
    found: rewrapping a paragraph — not touching a word of it — moved a newline into the
    middle of "carries all 41 with a named reason" and the check reported `cannot check`.
    That is the loud failure this function is built to give, so nothing was hidden; but the
    reword it reported never happened, and a gate that cries wolf at line wrapping teaches
    the next reader to re-wrap the number instead of the pattern. Four patterns already
    spelled a space `\\s+` for exactly this reason, which made wrap-tolerance a property of
    whoever remembered to type it. It is a property of the reader now.
    """
    found = re.findall(pattern, re.sub(r"\s+", " ", text), re.S)
    if not found:
        return None, "the sentence this check reads is no longer in the document"
    if len(found) > 1:
        return None, f"the sentence appears {len(found)} times, so no single number is quoted"
    return int(found[0]), ""


def test_function_counts(root: Path) -> dict[str, int]:
    """How many `test_*` functions each module defines, by AST.

    A different quantity from the collected count and quoted as such: `tests/test_tools.py`
    is 38 functions and 62 cases, and the gap is parametrisation. Two numbers in one
    sentence need two derivations, or the sentence is half checked.
    """
    counts: dict[str, int] = {}
    for path in sorted((root / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        counts[f"tests/{path.name}"] = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_"))
    return counts


def seam_row_count(text: str) -> int:
    """Data rows in `infra/dormant_seams.tsv` — the number the design documents quote."""
    return sum(1 for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith("#"))


def audit(docs: dict[str, str], per_file: dict[str, int],
          mutation_per_file: dict[str, int], module_count: int, scan_cases: int,
          test_functions: dict[str, int], seam_rows: int) -> list[str]:
    """Every documented number against its observed value. Pure, so it can be mutated."""
    findings: list[str] = []
    design = docs[TEST_DESIGN]
    total = sum(per_file.values())
    mutation_total = sum(mutation_per_file.values())

    # --- the per-file table, both directions: a row that drifted AND a module with no row
    rows = list(TABLE_ROW.finditer(design))
    documented = {m.group(1): int(m.group(2)) for m in rows}
    row_prose = {m.group(1): m.group(3) for m in rows}
    tested = {k: v for k, v in per_file.items() if Path(k).name.startswith("test_")}
    for path in sorted(set(documented) | set(tested)):
        if path not in documented:
            findings.append(f"{TEST_DESIGN}: {path} collects {tested[path]} cases but has no "
                            "row in the §0 table — an unlisted module is an uncounted one")
        elif path not in tested:
            findings.append(f"{TEST_DESIGN}: the §0 table lists {path}, which collects nothing")
        elif documented[path] != tested[path]:
            findings.append(f"{TEST_DESIGN}: {path} row says {documented[path]}, "
                            f"pytest collects {tested[path]}")

    # --- the second number in every row: the mutation-layer total its prose states. Nothing
    #     read these until R0b-2, and 33 rows carried one each: the row for
    #     `tests/test_consolidate.py` said `L3 選到 13 個:12 個具名突變體 + 無突變對照` while the
    #     layer collected 12 (11 named mutants plus the control), and two rows whose modules do
    #     have a mutation layer stated a *breakdown* of named mutants with no total at all, so
    #     there was no number for collection to disagree with. Unnumbered is uncounted.
    #
    #     Both directions, because they fail differently. A row quoting a total for a layer that
    #     collects nothing is what a deleted mutant leaves behind, and it reads as coverage.
    for path, prose in sorted(row_prose.items()):
        observed_l3 = mutation_per_file.get(path, 0)
        said = ROW_L3.findall(prose)
        if len(said) > 1:
            findings.append(f"{TEST_DESIGN}: {path} states {len(said)} L3 totals, so no single "
                            "number is checked")
        elif not said:
            if observed_l3:
                findings.append(
                    f"{TEST_DESIGN}: {path} states no L3 total, but the mutation layer collects "
                    f"{observed_l3} cases there — an unnumbered layer is an uncounted one")
        elif not observed_l3:
            findings.append(f"{TEST_DESIGN}: {path} says L3 selects {said[0]} cases, but the "
                            "mutation layer collects nothing there")
        elif int(said[0]) != observed_l3:
            findings.append(f"{TEST_DESIGN}: {path} L3 total says {said[0]}, "
                            f"pytest -m mutation collects {observed_l3}")

    # --- the same per-module counts where they appear in running prose rather than in the
    #     table. Seven sites today and the number seven is deliberately not written down: the
    #     loop reads whatever the document quotes, so a paragraph written next month is checked
    #     the day it is written, which a list of sentences could not be. The `ran` flag is the
    #     weakest form of rule 5 that fits — zero sites means the phrasing moved, never that the
    #     document stopped quoting counts.
    prose_cases = PROSE_CASES.findall(design)
    if not prose_cases:
        findings.append(f"{TEST_DESIGN}: no prose quotes a module's case count, so this scan "
                        "read nothing; a scan that reads nothing must not report clean")
    for path, said in prose_cases:
        if path not in per_file:
            findings.append(f"{TEST_DESIGN}: prose quotes {said} cases for {path}, which "
                            "collects nothing")
        elif int(said) != per_file[path]:
            findings.append(f"{TEST_DESIGN}: prose says {path} has {said} cases, "
                            f"pytest collects {per_file[path]}")

    # --- every number a document quotes, in one loop. Three loops was three copies of the
    #     "cannot check" branch, and a branch that excuses a rewrite is worth having once.
    tools_cases = per_file.get(TOOLS_TESTS, 0)
    tools_funcs = test_functions.get(TOOLS_TESTS, 0)
    store_mutants = mutation_per_file.get(STORE_TESTS, 0)
    checks = (
        # the headline totals, quoted in three documents.
        # R1.2 reworded the first one. `total` is `pytest --collect-only`, and the sentence
        # said "what actually runs today", which is a different quantity: the R1.2 verification
        # run was 1668 passed and one skipped — `tests/test_meta.py` needs an `oar-daemon`
        # *process*, not the binary. Numerically the two agreed, so nothing here went red; the
        # label was wrong about what this line computes, and the skipped case is the one that
        # separates them. This gate cannot derive a skip count (it never executes the suite),
        # so the sentence now says `收集到` and the pass/skip pair is prose beside the seconds.
        (TEST_DESIGN, r"今天被收集到的是 (\d+) 個案例", total, "suite total"),
        (TEST_DESIGN, r"全套 (\d+) 個案例", mutation_total, "mutation total"),
        (TEST_DESIGN, r"選到 \*\*(\d+) 個案例\*\*", mutation_total, "mutation total (T-J4)"),
        (TEST_DESIGN, r"沒有第 (\d+) 個躲在標記外面", mutation_total + 1,
         "the sentinel one past the mutation total"),
        (README_EN, r"Offline suite, (\d+) cases", total, "suite total"),
        (README_EN, r"the mutants alone \((\d+) cases on", mutation_total, "mutation total"),
        (README_ZH, r"離線測試 (\d+) 個案例", total, "suite total"),
        (README_ZH, r"量到 (\d+) 個案例", mutation_total, "mutation total"),
        (PYTEST_INI, r"the (\d+)-case\s+suite", total, "suite total"),
        (PYTEST_INI, r"the (\d+) mutants\s+were", mutation_total, "mutation total"),
        # §3's own two sentences about the wall clock. Found the same way as the seam sites —
        # by grepping for the number rather than for a sentence already gated — and they are
        # the sharpest case for gating prose: one of them is the paragraph that states the
        # budget overrun as a multiple, so a stale case count there misreports the ratio, and
        # the other is the paragraph explaining that the two duration sets have *different
        # denominators*. A denominator quoted from memory is not a denominator.
        (TEST_DESIGN, r"今天是 (\d+) 個案例", total, "suite total (§3 budget overrun)"),
        (TEST_DESIGN, r"對\*\*整個層\*\*\((\d+) 個案例", total, "suite total (§3 denominator)"),
        # Anchored on `整個層` because the same sentence quotes a *subset*'s figures four lines
        # later ("145 個案例、其中 18 個突變體"), and those are a 2026-08-22 measurement of
        # something else. An unanchored pattern would read two numbers and report neither.
        (TEST_DESIGN, r"整個層\*\*\(\d+ 個案例、其中 (\d+) 個\s*突變體\)", mutation_total,
         "mutation total (§3 denominator)"),
        # the figures the design documents and the plan quote by module name. These are the
        # numbers R0b-2 widened `DOCS` for, and the widening is only real because of them:
        # adding a path to the tuple makes a file *readable*, while nothing is checked until
        # a pattern quotes a number out of it.
        (DESIGN_EN, r"\((\d+) collected cases from", tools_cases, "test_tools case count"),
        (DESIGN_EN, r"collected cases from (\d+) test functions", tools_funcs,
         "test_tools function count"),
        (DESIGN_ZH, r"收集到 (\d+) 個案例", tools_cases, "test_tools case count"),
        (DESIGN_ZH, r"來自 (\d+) 個測試函式", tools_funcs, "test_tools function count"),
        (PLAN, r"\((\d+) cases per `docs/test-design\.md`", tools_cases,
         "test_tools case count"),
        (DESIGN_EN, r"\*\*(\d+) mutation cases\*\* in `tests/test_store\.py`", store_mutants,
         "test_store mutation case count"),
        (DESIGN_ZH, r"`tests/test_store\.py` 的 \*\*(\d+) 個突變案例", store_mutants,
         "test_store mutation case count"),
        (PLAN, r"the (\d+) mutation cases of\s+`tests/test_store\.py`", store_mutants,
         "test_store mutation case count"),
    )
    # the seam count, which §1 of `docs/design.md` explicitly asks readers to quote
    # *instead of* the dormancy percentage. A number a document recommends is the last
    # one that should be hand-typed — and this one went stale in four documents at once
    # the day a by-hand row joined the registry.
    seam_checks = (
        (DESIGN_EN, r"reports \*\*(\d+)\*\* today", seam_rows, "dormant seam count"),
        (DESIGN_ZH, r"今天報 \*\*(\d+)\*\*", seam_rows, "dormant seam count"),
        (PLAN, r"`infra/dormant_seams\.tsv` \((\d+) rows\)", seam_rows, "dormant seam count"),
        (TEST_DESIGN, r"「今天這 (\d+) 列是對的」", seam_rows, "dormant seam count"),
        # Six more sites, found by grepping for the *number* instead of for the sentence
        # already gated: two of them were still saying 42 while the four above said 43, in the
        # same PR that added the check. A remembered phrasing finds the sentence you wrote
        # down; only the number finds the ones you did not.
        (TEST_DESIGN, r"的 (\d+) 列雙向對帳", seam_rows, "dormant seam count (§0 row)"),
        (TEST_DESIGN, r"的 (\d+) 列對稱差", seam_rows, "dormant seam count (T-N1)"),
        (DESIGN_EN, r"carries all (\d+) with a named reason", seam_rows,
         "dormant seam count (the registry's own row count)"),
        (DESIGN_ZH, r"(\d+) 列全部帶具名理由", seam_rows,
         "dormant seam count (the registry's own row count)"),
        (PLAN, r"reports \*\*(\d+)\*\* seams reachable", seam_rows,
         "dormant seam count (the census figure)"),
        (PLAN, r"carries (\d+)\s+rows with named reasons", seam_rows,
         "dormant seam count (the registry's own row count)"),
    )
    for doc, pattern, expected, what in checks + seam_checks:
        got, why = quoted(docs[doc], pattern)
        if got is None:
            findings.append(f"{doc}: cannot check the {what} — {why}")
        elif got != expected:
            findings.append(f"{doc}: {what} says {got}, observed {expected}")

    if tools_funcs and tools_cases < tools_funcs:
        findings.append(
            f"tests/: {TOOLS_TESTS} defines {tools_funcs} test functions but collects "
            f"{tools_cases} cases — collection cannot be below the function count, so one "
            "of the two derivations is reading the wrong thing")

    # --- the claim that the offline import scan is derived from the directory, not listed
    pair = re.findall(r"\*\*(\d+) 個模組\s*恰好對應 (\d+) 個案例\*\*", design, re.S)
    if len(pair) != 1:
        findings.append(f"{TEST_DESIGN}: cannot check the module/scan claim — the sentence "
                        f"matched {len(pair)} times")
    else:
        said_modules, said_cases = (int(v) for v in pair[0])
        if (said_modules, said_cases) != (module_count, scan_cases):
            findings.append(
                f"{TEST_DESIGN}: the §0 paragraph claims {said_modules} modules and "
                f"{said_cases} scan cases; observed {module_count} and {scan_cases}")
        if module_count != scan_cases:
            findings.append(
                f"tests/: {module_count} modules in tests/*.py but the offline import scan "
                f"collects {scan_cases} cases — the scan is no longer one-per-module")

    return findings


def main() -> int:
    docs: dict[str, str] = {}
    for name in DOCS:
        path = REPO_ROOT / name
        if not path.exists():
            print(f"error: {name} is missing — cannot check its numbers", file=sys.stderr)
            return 3
        docs[name] = path.read_text(encoding="utf-8")

    per_file = collect_counts()
    if not per_file:
        print("error: parsed 0 modules out of pytest --collect-only; a scan that reads "
              "nothing must not report clean", file=sys.stderr)
        return 3
    mutation_per_file = collect_counts("mutation")
    if not mutation_per_file:
        print("error: parsed 0 mutation cases; layer L3 cannot be empty", file=sys.stderr)
        return 3
    functions = test_function_counts(REPO_ROOT)
    if not functions:
        print("error: found 0 test functions under tests/; a scan that reads nothing must "
              "not report clean", file=sys.stderr)
        return 3

    seams = REPO_ROOT / SEAMS_TSV
    if not seams.exists():
        print(f"error: {SEAMS_TSV} is missing — cannot derive the seam count", file=sys.stderr)
        return 3
    seam_rows = seam_row_count(seams.read_text(encoding="utf-8"))
    if not seam_rows:
        print(f"error: {SEAMS_TSV} has 0 data rows; a registry that reads empty must not "
              "report clean", file=sys.stderr)
        return 3

    modules = sorted(p.name for p in (REPO_ROOT / "tests").glob("*.py"))
    scan_cases = scan_case_count()
    findings = audit(docs, per_file, mutation_per_file, len(modules), scan_cases, functions,
                     seam_rows)
    for line in findings:
        print(f"finding  {line}")
    print(f"{sum(per_file.values())} cases, {sum(mutation_per_file.values())} mutants, "
          f"{len(findings)} findings")
    return 1 if findings else 0


def scan_case_count() -> int:
    """How many cases the directory-derived offline import scan actually collects."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_meta.py", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-k", "no_test_module_imports"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    found = re.findall(r"^tests/test_meta\.py:\s*(\d+)\s*$", proc.stdout, re.M)
    return int(found[0]) if len(found) == 1 else 0


if __name__ == "__main__":
    sys.exit(main())
