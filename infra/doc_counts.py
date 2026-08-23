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

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DESIGN = "docs/test-design.md"
README_EN = "README.md"
README_ZH = "README.zh-TW.md"
DOCS = (TEST_DESIGN, README_EN, README_ZH)

# `tests/test_x.py: 22` — printed per module by tests/conftest.py under --collect-only.
COLLECT_LINE = re.compile(r"^(tests/[A-Za-z0-9_]+\.py):\s*(\d+)\s*$", re.M)
# `| `tests/test_opponent.py` | 74 | ...` — the §0 table.
TABLE_ROW = re.compile(r"^\|\s*`(tests/test_[A-Za-z0-9_]+\.py)`\s*\|\s*(\d+)\s*\|", re.M)


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
    """
    found = re.findall(pattern, text, re.S)
    if not found:
        return None, "the sentence this check reads is no longer in the document"
    if len(found) > 1:
        return None, f"the sentence appears {len(found)} times, so no single number is quoted"
    return int(found[0]), ""


def audit(docs: dict[str, str], per_file: dict[str, int], mutation_total: int,
          module_count: int, scan_cases: int) -> list[str]:
    """Every documented number against its observed value. Pure, so it can be mutated."""
    findings: list[str] = []
    design = docs[TEST_DESIGN]
    total = sum(per_file.values())

    # --- the per-file table, both directions: a row that drifted AND a module with no row
    documented = {m.group(1): int(m.group(2)) for m in TABLE_ROW.finditer(design)}
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

    # --- the headline totals, quoted in three documents
    for doc, pattern, expected, what in (
        (TEST_DESIGN, r"今天真正跑起來的是 (\d+) 個案例", total, "suite total"),
        (TEST_DESIGN, r"全套 (\d+) 個案例", mutation_total, "mutation total"),
        (TEST_DESIGN, r"選到 \*\*(\d+) 個案例\*\*", mutation_total, "mutation total (T-J4)"),
        (TEST_DESIGN, r"沒有第 (\d+) 個躲在標記外面", mutation_total + 1,
         "the sentinel one past the mutation total"),
        (README_EN, r"Offline suite, (\d+) cases", total, "suite total"),
        (README_EN, r"the mutants alone \((\d+) cases on", mutation_total, "mutation total"),
        (README_ZH, r"離線測試 (\d+) 個案例", total, "suite total"),
        (README_ZH, r"量到 (\d+) 個案例", mutation_total, "mutation total"),
    ):
        got, why = quoted(docs[doc], pattern)
        if got is None:
            findings.append(f"{doc}: cannot check the {what} — {why}")
        elif got != expected:
            findings.append(f"{doc}: {what} says {got}, observed {expected}")

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
    mutation_total = sum(collect_counts("mutation").values())
    if not mutation_total:
        print("error: parsed 0 mutation cases; layer L3 cannot be empty", file=sys.stderr)
        return 3

    modules = sorted(p.name for p in (REPO_ROOT / "tests").glob("*.py"))
    scan_cases = scan_case_count()
    findings = audit(docs, per_file, mutation_total, len(modules), scan_cases)
    for line in findings:
        print(f"finding  {line}")
    print(f"{sum(per_file.values())} cases, {mutation_total} mutants, "
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
