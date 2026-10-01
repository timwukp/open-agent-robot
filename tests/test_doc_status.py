"""T-O1 document-status table + T-O2 mutation liveness for infra/doc_status.py.

The table is a claim about which documents may be cited as the design, and it was written
because four of the six research rounds in `docs/` carried no supersession marker of any
kind while every guard in this repo was green. So the claim needs falsifying the same way
`tests/test_wiring_census.py` falsifies the seam census: an unmutated control first, then a
named mutant per check with the specific finding it must provoke, then the guard's own
failure modes as exit 3.

Four of the tests below are load-bearing beyond killing a mutant, and they are why this
file is longer than the guard:

  `test_a_non_recursive_walk_loses_the_evidence_directory` states the recursion as a
  difference. `docs/evidence/` holds two of the seventeen documents. Swap `rglob` for
  `glob` and the walk still reports a clean-looking count — of fifteen — so the recursion
  is not decoration and the number it feeds is not arbitrary.

  `test_a_single_value_relation_loses_round_4s_second_overturner` states the `;` the same
  way. Round 4 has two independent overturners: the critique replaced its §0 build order
  and round 5 re-answered its memory pick. Parse only the first term and the index may
  drop the second one silently — which is the flattening the whole relation vocabulary
  exists to prevent.

  `test_the_ran_flag_names_the_cause_instead_of_a_symptom` is what the flag actually buys,
  which is **not** what the guard's docstring first claimed. An emptied `docs/` is refused
  twice over — the flag catches it, and so does the missing-prose-document check, because
  every prose document this guard reads lives under `docs/` itself. So the flag is not what
  makes an empty tree non-clean. What it is load-bearing for is a walk pointed at the wrong
  directory: with the flag the guard exits 3 saying it read nothing, and with the flag
  deleted the same typo exits 1 with a finding against *every row in the table*, which
  points the reader at deleting the registry. Detection was never the hole; misdirection was.

  `test_the_round_4_attribution_is_checked_against_the_table` is a real positive control,
  not an invented one: `docs/design.md` shipped in PR #29 crediting round 7 with
  overturning round 4, and round 7 never names round 4 except to cite it approvingly. Two
  mutants sit on that sentence because it can fail in two ways, and only one of them was
  the way it did fail: rewriting the sentence altogether trips rule 5 (a check whose
  subject is gone is a finding, not a pass), while keeping its shape and swapping the
  document it names trips the comparison against the table. A guard with only the first
  check would accept a wrong attribution as long as it was phrased right.

The trimmed tree these tests run on holds only what the guard reads — its own source, the
TSV, and every `docs/**/*.md`. That omission is not assumed harmless:
`test_the_trimmed_tree_agrees_with_the_real_one` runs both and requires the same verdict,
so if the guard ever grows a reader of something else, the control turns red rather than a
mutant turning quietly green.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import patch

GUARD = "infra/doc_status.py"
TABLE = "infra/doc_status.tsv"
INDEX = "docs/INDEX.md"
PLAN = "docs/plan.md"
DESIGN_EN = "docs/design.md"
DESIGN_ZH = "docs/design.zh-TW.md"

REPO_ROOT = Path(__file__).resolve().parent.parent

# The documents the relation vocabulary exists for. Named once so a rename cannot leave
# these tests asserting about a supersession that no longer exists while still passing.
ROUND4 = "docs/research-round4-agent-brain-2026-08-21.md"
CRITIQUE = "docs/design-critique-memory-safety-2026-08-22.md"
ROUND5 = "docs/research-round5-memory-safety-2026-08-22.md"
ROUND7 = "docs/research-round7-rl-and-two-tier-2026-08-22.md"
ROUND6 = "docs/research-round6-simreal-joint-mapping-2026-08-22.md"

# The sentence PR #29 shipped wrong, kept verbatim as the mutant for the check that
# would have caught it.
WRONG_ATTRIBUTION = "which round 7 overturns in part."
RIGHT_ATTRIBUTION = (
    f"whose §0 build order\n`{CRITIQUE}` §4 replaces outright and whose memory\n"
    f"pick `{ROUND5}` re-answers in its own header."
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A trimmed copy holding exactly what the guard reads, and nothing else."""
    dst = tmp_path / "tree"
    (dst / "infra").mkdir(parents=True)
    for rel in (GUARD, TABLE):
        shutil.copy2(REPO_ROOT / rel, dst / rel)
    for src in (REPO_ROOT / "docs").rglob("*.md"):
        out = dst / src.relative_to(REPO_ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    return dst


def run_status(root: Path) -> subprocess.CompletedProcess:
    """Run the copy's own guard, so a mutated guard is the one under test."""
    return subprocess.run([sys.executable, str(root / GUARD)], capture_output=True, text=True)


def report(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


def summary(proc: subprocess.CompletedProcess) -> tuple[int, int]:
    """`(documents censused, findings)` out of the guard's own report.

    Read from the report rather than inferred from the exit code: a guard that exits 1
    while reporting 0 findings is itself a failure, and the two must be graded apart.
    """
    for line in reversed(report(proc).splitlines()):
        if "documents censused" in line and "findings" in line:
            parts = line.split()
            return int(parts[0]), int(parts[-2])
    raise AssertionError("the guard printed no summary line, so it cannot be graded:\n"
                         + report(proc))


def findings(proc: subprocess.CompletedProcess) -> list[str]:
    return [ln for ln in report(proc).splitlines() if ln.startswith("finding")]


# --------------------------------------------------------------------------- controls

def test_the_real_tree_agrees() -> None:
    """T-O1. The table, the index and `docs/` agree as committed."""
    proc = run_status(REPO_ROOT)
    censused, found = summary(proc)
    assert proc.returncode == 0, report(proc)
    assert found == 0, report(proc)
    assert censused == len(list((REPO_ROOT / "docs").rglob("*.md")))


def test_the_trimmed_tree_agrees_with_the_real_one(tree: Path) -> None:
    """The no-mutant control, and the proof that the fixture's omissions changed nothing."""
    real, copied = run_status(REPO_ROOT), run_status(tree)
    assert (real.returncode, summary(real)) == (copied.returncode, summary(copied)), (
        "the trimmed tree and the real tree disagree, so the fixture is no longer a "
        f"faithful copy of what the guard reads:\nreal:\n{report(real)}\n"
        f"copy:\n{report(copied)}"
    )


def test_every_document_under_docs_has_a_row(tree: Path) -> None:
    """The membership claim itself, derived from the tree rather than from the table."""
    table = (tree / TABLE).read_text(encoding="utf-8")
    listed = {ln.split("\t")[0] for ln in table.splitlines()
              if ln.strip() and not ln.lstrip().startswith("#")}
    on_disk = {p.relative_to(tree).as_posix() for p in (tree / "docs").rglob("*.md")}
    assert listed == on_disk, (
        "the table and the tree name different documents; the guard checks this too, and "
        "this test states it once more without the guard's own parser in the loop"
    )


# ------------------------------------------------------- the holes, stated as differences

def test_a_non_recursive_walk_loses_the_evidence_directory(tree: Path) -> None:
    """Recursion is load-bearing: two of seventeen documents live one level down."""
    clean = run_status(tree)
    assert (clean.returncode, summary(clean)[1]) == (0, 0), report(clean)

    patch(tree, GUARD, 'docs.rglob("*.md")', 'docs.glob("*.md")')
    shallow = run_status(tree)
    censused, found = summary(shallow)
    assert censused == 15, (
        "a non-recursive walk must lose exactly the two documents under docs/evidence/; "
        f"it censused {censused}:\n{report(shallow)}"
    )
    assert found >= 1 and shallow.returncode == 1, (
        "and it must not report clean while quietly exempting a whole directory:\n"
        + report(shallow)
    )


def test_a_single_value_relation_loses_round_4s_second_overturner(tree: Path) -> None:
    """The `;` is load-bearing: round 4 has two independent overturners, not one."""
    patch(tree, INDEX, f"`overturned_in_part_by:{CRITIQUE}`;`overturned_in_part_by:{ROUND5}`",
          f"`overturned_in_part_by:{CRITIQUE}`")
    caught = run_status(tree)
    assert caught.returncode == 1 and any(ROUND4 in f for f in findings(caught)), (
        "dropping one of round 4's two overturners from the index must be a finding:\n"
        + report(caught)
    )

    patch(tree, GUARD,
          'return [t.strip() for t in relation.split(";") if t.strip()]',
          'return [relation.split(";")[0].strip()]')
    flattened = run_status(tree)
    assert not any(ROUND4 in f for f in findings(flattened)), (
        "with relations parsed as a single value the guard must go quiet about the "
        "dropped overturner — if it still reports it, something other than the `;` split "
        "is finding it and that instrument is not what closed the hole:\n"
        + report(flattened)
    )


def test_an_empty_walk_is_refused_twice_over(tree: Path) -> None:
    """A walk that reads nothing must not report agreement — and here two checks say so.

    Stated as a pair on purpose. The ran flag refuses the empty walk, and the
    missing-prose-document check would refuse the same tree even without it, because every
    document whose prose this guard quotes lives under `docs/`. Writing that down is the
    point: a redundancy nobody measured is usually a redundancy that is not there.
    """
    for md in (tree / "docs").rglob("*.md"):
        md.unlink()
    guarded = run_status(tree)
    assert guarded.returncode == 3 and "found 0 documents" in report(guarded), report(guarded)

    patch(tree, GUARD, "    if not on_disk:", "    if False:")
    without_flag = run_status(tree)
    assert without_flag.returncode == 3 and INDEX in report(without_flag), (
        "with the flag removed the second check must still refuse the empty tree:\n"
        + report(without_flag)
    )


def test_the_ran_flag_names_the_cause_instead_of_a_symptom(tree: Path) -> None:
    """The flag's real job: a walk pointed at the wrong directory must not read as findings.

    `DOCS_DIR` mistyped is the realistic way the walk goes silent while every file it should
    have read is still on disk. With the flag the guard says it read nothing. Without it the
    guard reports a finding against **every row in the table** — a report whose obvious fix
    is deleting the registry this whole gate exists to keep.
    """
    patch(tree, GUARD, 'DOCS_DIR = "docs"', 'DOCS_DIR = "doc"')
    guarded = run_status(tree)
    assert guarded.returncode == 3 and "found 0 documents" in report(guarded), (
        "a walk that read nothing must say so, whatever the reason it read nothing:\n"
        + report(guarded)
    )

    patch(tree, GUARD, "    if not on_disk:", "    if False:")
    misdirected = run_status(tree)
    censused, found = summary(misdirected)
    rows = sum(1 for ln in (tree / TABLE).read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#"))
    assert censused == 0 and misdirected.returncode == 1, report(misdirected)
    assert found >= rows, (
        f"without the flag a misdirected walk must indict all {rows} rows — that is the "
        f"misdirection the flag exists to prevent; it reported {found}:\n"
        + report(misdirected)
    )


def test_the_reason_floor_is_where_it_says_it_is(tree: Path) -> None:
    """59 characters is a finding and 60 is not, so the floor is measured, not implied."""
    original = (tree / TABLE).read_text(encoding="utf-8")
    for length, wanted in ((59, 1), (60, 0)):
        rows = []
        for line in original.splitlines(keepends=True):
            if line.startswith("docs/day0-measurements.md\t"):
                path, status, relation, _ = line.rstrip("\n").split("\t")
                line = f"{path}\t{status}\t{relation}\t{'x' * length}\n"
            rows.append(line)
        (tree / TABLE).write_text("".join(rows), encoding="utf-8")
        proc = run_status(tree)
        assert summary(proc)[1] == wanted, (
            f"a {length}-character reason must be {'a finding' if wanted else 'accepted'}:\n"
            + report(proc)
        )


# --------------------------------------------------------------------------- exit code 3

@pytest.mark.parametrize("removed", [TABLE, INDEX, PLAN, DESIGN_EN, DESIGN_ZH])
def test_a_missing_document_this_guard_reads_is_exit_3(tree: Path, removed: str) -> None:
    """Cannot-check is not clean, and it is not a finding either."""
    (tree / removed).unlink()
    proc = run_status(tree)
    assert proc.returncode == 3, (
        f"with {removed} gone the guard must report that it cannot check rather than "
        f"dropping the checks that read it:\n{report(proc)}"
    )
    assert removed in report(proc)


# ------------------------------------------------------------------ the named mutants

def _wrong_round4_attribution(root: Path) -> None:
    patch(root, DESIGN_EN, RIGHT_ATTRIBUTION, WRONG_ATTRIBUTION)


def _reason_becomes_a_shrug(root: Path) -> None:
    rows = []
    for line in (root / TABLE).read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("docs/day0-measurements.md\t"):
            path, status, relation, _ = line.rstrip("\n").split("\t")
            line = f"{path}\t{status}\t{relation}\tunwired\n"
        rows.append(line)
    (root / TABLE).write_text("".join(rows), encoding="utf-8")


def _duplicate_row(root: Path) -> None:
    patch(root, TABLE, "docs/DEMO.md\tcurrent",
          "docs/DEMO.md\tcurrent\t-\ta second row for one path, which is what a table "
          "with no duplicate check silently accepts\ndocs/DEMO.md\tcurrent")


MUTANTS: dict[str, tuple] = {
    # --- the counted sentences, one per document that quotes one
    "index_document_total": (
        lambda r: patch(r, INDEX, "**17 documents**", "**18 documents**"),
        "document total says 18"),
    "plan_document_total": (
        lambda r: patch(r, PLAN, "**17 files**", "**18 files**"),
        "document total says 18"),
    "plan_top_level_count": (
        lambda r: patch(r, PLAN, "15 at the top level plus", "14 at the top level plus"),
        "top-level count says 14"),
    "design_en_size_claim": (
        lambda r: patch(r, DESIGN_EN, "**191 KiB together**", "**190 KiB together**"),
        "size claim says ['six', 'one', '190']"),
    "design_zh_size_claim": (
        lambda r: patch(r, DESIGN_ZH, "**合計 191 KiB**", "**合計 190 KiB**"),
        "size claim says ['六', '一', '190']"),
    "index_size_claim": (
        lambda r: patch(r, INDEX, "**191 KiB together**", "**192 KiB together**"),
        "size claim says 192 KiB"),
    "design_en_research_count": (
        lambda r: patch(r, DESIGN_EN, "six research documents and one",
                        "seven research documents and one"),
        "size claim says ['seven', 'one', '191']"),
    "design_zh_research_count": (
        lambda r: patch(r, DESIGN_ZH, "裡六份研究文件", "裡七份研究文件"),
        "size claim says ['七', '一', '191']"),
    # --- the attribution PR #29 got wrong, in both of its failure modes
    "round4_sentence_rewritten_away": (
        _wrong_round4_attribution,
        "cannot check the round 4 attribution"),
    "round4_names_the_wrong_document": (
        lambda r: patch(r, DESIGN_EN, f"`{CRITIQUE}` §4 replaces outright",
                        f"`{ROUND7}` §4 replaces outright"),
        "round 4 attribution names"),
    "round4_zh_names_the_wrong_document": (
        lambda r: patch(r, DESIGN_ZH, f"build order 被 `{CRITIQUE}` §4",
                        f"build order 被 `{ROUND7}` §4"),
        "round 4 attribution names"),
    # --- the table against itself
    "status_outside_the_closed_set": (
        lambda r: patch(r, TABLE, "docs/design.md\tcurrent", "docs/design.md\tlive"),
        "outside the closed set"),
    "row_deleted_for_a_document_on_disk": (
        lambda r: patch(r, TABLE, "docs/DEMO.md\tcurrent", "#docs/DEMO.md\tcurrent"),
        "cannot join docs/ silently"),
    "row_kept_for_a_document_removed": (
        lambda r: (r / "docs" / "DEMO.md").unlink(),
        "names a file that does not exist"),
    "reason_short_enough_to_be_a_shrug": (_reason_becomes_a_shrug, "the floor is 60"),
    "relation_verb_outside_the_vocabulary": (
        lambda r: patch(r, TABLE, f"overturned_in_part_by:{ROUND7}", f"overturned_by:{ROUND7}"),
        "which is not <verb>:<path>"),
    "relation_target_that_does_not_exist": (
        lambda r: patch(r, TABLE, f"overturned_in_part_by:{ROUND7}",
                        "overturned_in_part_by:docs/research-round8.md"),
        "which has no row here"),
    "superseded_without_a_replacement": (
        lambda r: patch(r, TABLE, "docs/DEMO.md\tcurrent\t-", "docs/DEMO.md\tsuperseded\t-"),
        "names no superseded_by target"),
    "two_rows_for_one_path": (_duplicate_row, "has two rows"),
    # --- the index against the table
    "index_relation_dropped": (
        lambda r: patch(r, INDEX, f"`overturned_in_part_by:{ROUND7}`", "`-`"),
        f"{ROUND6} names []"),
    "index_status_disagrees": (
        lambda r: patch(r, INDEX, "| [`docs/DEMO.md`](DEMO.md) | current",
                        "| [`docs/DEMO.md`](DEMO.md) | evidence"),
        "is evidence here and current in"),
    "index_row_removed": (
        lambda r: patch(r, INDEX, "| [`docs/workflow.md`](workflow.md)",
                        "x | [`docs/workflow.md`](workflow.md)"),
        "has a row in infra/doc_status.tsv but none here"),
    "index_link_dangles": (
        lambda r: patch(r, INDEX, "](workflow.md)", "](workflows.md)"),
        "does not resolve"),
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_named_mutant_is_killed(name: str, tree: Path) -> None:
    mutate, expected = MUTANTS[name]
    mutate(tree)
    proc = run_status(tree)
    assert proc.returncode != 0, f"mutant {name!r} survived the guard:\n{report(proc)}"
    if proc.returncode == 1:
        assert summary(proc)[1] >= 1, (
            f"mutant {name!r} made the guard exit 1 while reporting 0 findings — the exit "
            f"code and the report disagree:\n{report(proc)}"
        )
    assert any(expected in f for f in findings(proc)), (
        f"mutant {name!r} was killed, but not by the check it targets (expected a finding "
        f"mentioning {expected!r}):\n" + "\n".join(findings(proc))
    )


@pytest.mark.mutation
def test_the_round_4_attribution_is_checked_against_the_table(tree: Path) -> None:
    """The positive control for the check that would have caught PR #29's own error.

    Round 7 cites round 4 once, approvingly, and never overturns it. The claim it did was
    prose, and prose about which document supersedes which is exactly as unverified as any
    other prose. What this test adds over the mutant of the same name is the *content* of
    the finding: a mismatch has two sides, and a report naming only one of them leaves the
    reader guessing whether to edit the document or the table.
    """
    patch(tree, DESIGN_EN, f"`{CRITIQUE}` §4 replaces outright",
          f"`{ROUND7}` §4 replaces outright")
    proc = run_status(tree)
    named = [f for f in findings(proc) if "round 4 attribution names" in f]
    assert named, "the guard must read the attribution sentence:\n" + report(proc)
    assert ROUND7 in named[0] and CRITIQUE in named[0] and ROUND5 in named[0], (
        "the finding must show both what the prose claims and what the table says, or a "
        f"reader cannot tell which side to fix:\n{named[0]}"
    )
