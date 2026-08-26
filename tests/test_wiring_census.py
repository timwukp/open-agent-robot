"""T-N1 wiring census + T-N2 mutation liveness + T-N3 launch bootstrap for
infra/wiring_census.py.

The census is a claim about which subsystems the running robot can reach, and it was
written because every existing guard in this repo was green while four of the five
promises ran only under pytest. So the claim needs falsifying the same way
`tests/test_deploy_path.py` falsifies `check_deploy.sh`: an unmutated control first,
then a named mutant per check with the specific finding it must provoke, then the
census's own failure modes as exit 3.

Three of the tests below are load-bearing beyond killing a mutant, and they are the
reason this file is longer than the census:

  `test_reachability_alone_loses_the_assemble_seam` states the hole differentially.
  Same tree, two instruments: with `unpassed` the seam `agent/context.py::assemble
  (recall=)` is reported; without it the census goes quiet about a file it still
  calls reached. If reachability alone found it, the second instrument would be
  decorative and this file would not be worth its length.

  `test_the_bare_sibling_import_is_resolved` is a real positive control, not an
  invented one: `body/gestures.py` reported dormant in the first version of the
  census, because `body/body_node.py` reaches it with `from gestures import …` and
  every other node in the tree uses `from body import gestures`. Remove that one
  resolution and a wired module reads as dead.

  `test_no_other_instrument_notices_a_node_that_cannot_start` is the false *reached*,
  which is the one direction this tool exists to make impossible. Remove
  `memory/memory_node.py`'s sys.path bootstrap — the state that file was in until
  R1.1b, in which the single writer raised ModuleNotFoundError before `rclpy.init()`
  on every boot — and reachability still resolves the import, the function instrument
  still sees the calls, and the manifest still agrees. Drop `bootstrap_findings` and
  the census reports that tree **green**, so the test asserts both halves.

The two boundaries the census documents are tested on a synthetic tree rather than on
this repo, and that is a measured statement about coverage, not a preference. Both
`_`-prefixed keyword-only parameters on disk (`erase.delete(_after_step=)` and
`store.Store.__init__(_allow_ephemeral=)`) sit in a file the census reports as
unreached or on a method it skips, so neither skip is exercised by the real tree at
all. A boundary nothing reaches is a boundary nobody has checked.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import patch

CENSUS = "infra/wiring_census.py"
MANIFEST = "infra/dormant_seams.tsv"
STANDALONE = "infra/standalone_entrypoints.tsv"
RUNNER = "infra/run_ec2.sh"

# The seam the census exists for. Named once so a rename cannot leave the tests
# asserting about a seam that no longer exists while still passing.
RECALL_SEAM = "agent/context.py::assemble(recall=)"
HIT_CAP_SEAM = "agent/context.py::assemble(hit_cap=)"


def run_census(root: Path, *, script_root: Path | None = None) -> subprocess.CompletedProcess:
    """Run the census over `root`, using the copy's own source unless told otherwise."""
    script = (script_root or root) / "infra" / "wiring_census.py"
    return subprocess.run([sys.executable, str(script), str(root)],
                          capture_output=True, text=True)


def _summary(proc: subprocess.CompletedProcess) -> list[str]:
    for line in reversed((proc.stdout + proc.stderr).splitlines()):
        if "seams censused" in line and "findings" in line:
            return line.split()
    raise AssertionError(
        "the census printed no summary line, so it cannot be graded:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def census_findings(proc: subprocess.CompletedProcess) -> int:
    """The finding count out of the census's own report, not inferred from the exit
    code — a census that exits 1 while reporting 0 findings is itself a failure."""
    return int(_summary(proc)[-2])


def census_files(proc: subprocess.CompletedProcess) -> int:
    return int(_summary(proc)[0])


def failures(proc: subprocess.CompletedProcess) -> str:
    report = proc.stdout + proc.stderr
    return "\n".join(l for l in report.splitlines() if l.startswith(("FAIL", "ERROR")))


# --------------------------------------------------------------------------- control


def test_control_the_real_tree_is_green(repo_root: Path) -> None:
    proc = run_census(repo_root)
    assert census_findings(proc) == 0, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_control_an_unmutated_copy_is_green(repo_copy: Path) -> None:
    """Proves the copy — and conftest's docs/daemon omission — changes no verdict.

    Without it, a mutant that reds because copying broke something scores as the
    census working.
    """
    proc = run_census(repo_copy)
    assert census_findings(proc) == 0, proc.stdout + proc.stderr
    assert proc.returncode == 0


def test_the_census_reads_a_useful_number_of_files(repo_root: Path) -> None:
    # A floor on files read, deliberately not on seams found: every R1 PR is supposed
    # to make the seam count smaller, so a floor there would turn each wiring step red
    # for succeeding. The files-read count only shrinks if discovery degrades.
    assert census_files(run_census(repo_root)) >= 35


def _rows(root: Path) -> list[list[str]]:
    return [l.split("\t") for l in (root / MANIFEST).read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def test_the_manifest_is_well_formed(repo_root: Path) -> None:
    rows = _rows(repo_root)
    assert rows, "the manifest declares no rows"
    assert all(len(r) == 4 for r in rows)
    assert len({r[0] for r in rows}) == len(rows)


def test_the_duplicated_tag_rules_are_the_declared_defects(repo_root: Path) -> None:
    """The `defect` class holds what is wrong rather than merely unwired, and this test
    pins the two rows to the second implementation that makes them wrong.

    Not a source-text assertion for its own sake: if someone deletes the inline regex in
    agent_node and calls the guard, this test reds and the manifest rows must go in the
    same PR. If someone deletes the guard functions instead, the census reds. Either way
    the duplication cannot be resolved silently in one direction only.
    """
    defects = {r[0] for r in _rows(repo_root) if r[2] == "defect"}
    assert defects == {"safety/guard.py::gesture_findings()",
                       "safety/guard.py::spoken_text()"}, defects

    node = (repo_root / "agent" / "agent_node.py").read_text()
    assert r're.findall(r"\[gesture:(\w+)\]"' in node.replace("_re.", "re."), (
        "agent_node no longer scans gesture tags inline. If it now calls "
        "safety/guard.gesture_findings, both defect rows are stale and belong in that PR"
    )
    assert r're.sub(r"\[(?:gesture:\w+|game:start)\]"' in node.replace("_re.", "re."), (
        "agent_node no longer strips action tags inline — see above for spoken_text"
    )
    # The consequence, asserted where it is actually contained: an unknown label is
    # dropped by a dict lookup in a file with zero tests, not refused by the guard.
    body = (repo_root / "body" / "body_node.py").read_text()
    assert "GESTURES.get(name)" in body


# --------------------------------------------------------------------------- mutants
#
# Split in two on purpose. Manifest mutants ask "does the census read its registry
# strictly?"; source mutants ask "is each instrument load-bearing?" A file that only
# had the first kind would pass with a census that reported the same seams whatever
# the tree looked like.


def _row(root: Path, seam: str) -> str:
    for line in (root / MANIFEST).read_text().splitlines():
        if line.startswith(seam + "\t"):
            return line
    raise AssertionError(f"no manifest row for {seam!r} to mutate")


def _mut_row_deleted(root: Path) -> None:
    """The censused seam nobody reviewed: a whole dormant module with no row."""
    patch(root, MANIFEST, _row(root, "memory/erase.py") + "\n", "")


def _mut_stale_row_for_a_reached_module(root: Path) -> None:
    """A row outliving its asymmetry. This is the row R1.4 must delete, and if a stale
    one reads as clean then the next seam arriving under this name is pre-excused."""
    with (root / MANIFEST).open("a") as fh:
        fh.write("memory/recall.py\tunreached\tpending_stage\tR1.4 has already landed, "
                 "so this row describes a module the robot now reaches on every turn\n")


def _mut_class_outside_the_closed_set(root: Path) -> None:
    old = _row(root, "memory/erase.py")
    patch(root, MANIFEST, old, old.replace("\tpending_stage\t", "\twired\t"))


def _mut_asymmetry_contradicts_the_census(root: Path) -> None:
    old = _row(root, "memory/erase.py")
    patch(root, MANIFEST, old, old.replace("\tunreached\t", "\tunpassed\t"))


def _mut_reason_is_a_shrug(root: Path) -> None:
    old = _row(root, "memory/erase.py")
    patch(root, MANIFEST, old, "memory/erase.py\tunreached\tpending_stage\tnot wired yet")


def _mut_pending_stage_without_a_stage_id(root: Path) -> None:
    """The reason still reads like an argument; it just stops naming the stage that
    would delete it, which is the half a machine can check."""
    old = _row(root, "memory/erase.py")
    patch(root, MANIFEST, old, old.replace("R1.5 wires it.", "A later stage wires it."))


def _mut_by_hand_entrypoint_names_another_tool(root: Path) -> None:
    """The defect the two-shape rule exists for: an entrypoint excused by a true
    statement about a different file. Under a rule that accepts any path with a row in
    standalone_entrypoints.tsv, this reads fine."""
    old = _row(root, "infra/arch_render.py")
    patch(root, MANIFEST, old,
          old.replace("infra/arch_render.py has its own row", "memory/rank_harness.py has a row", 1))


def _mut_by_hand_library_names_nothing(root: Path) -> None:
    """The other shape: a library of a by-hand tool whose reason names no tool at all,
    so nothing ties it to something anyone runs."""
    old = _row(root, "infra/ros_census.py")
    patch(root, MANIFEST, old, old.replace("infra/arch_render.py", "a rendering script"))


def _mut_test_only_names_a_missing_test(root: Path) -> None:
    old = _row(root, HIT_CAP_SEAM)
    patch(root, MANIFEST, old, old.replace("tests/test_context.py", "tests/test_ghost.py"))


def _mut_test_only_names_no_test(root: Path) -> None:
    old = _row(root, HIT_CAP_SEAM)
    patch(root, MANIFEST, old,
          old.replace("Driven by tests/test_context.py, which passes",
                      "Driven by the suite, which passes"))


def _mut_aliases_drop_the_bare_sibling(root: Path) -> None:
    """The first version of this census had exactly this hole, and it reported a wired
    module as dormant. body/body_node.py is launched as a file, so sys.path[0] is
    body/ and it is the one node that imports a sibling by bare name."""
    patch(root, CENSUS,
          "                record(local, (here / f\"{parts[0]}.py\").as_posix())\n"
          "        elif isinstance(node, ast.Import):",
          "        elif isinstance(node, ast.Import):")


def _mut_reach_is_not_transitive(root: Path) -> None:
    """Reachability that stops at the launched process's direct callees. Every module
    behind one hop — the whole memory layer under memory_node — reads as dormant."""
    patch(root, CENSUS, "                frontier.append(target)\n", "                pass\n")


def _mut_tests_count_as_callers(root: Path) -> None:
    """The one exclusion that carries the whole argument. With tests/ counted, every
    dormant function has a caller and every dormant parameter has a supplier, so the
    census reports a wired tree — precisely the false green this file exists to remove.
    """
    patch(root, CENSUS,
          'SKIP_DIR_PARTS = ("tests", "__pycache__"',
          'SKIP_DIR_PARTS = ("__pycache__"')


def _mut_uncalled_row_deleted(root: Path) -> None:
    """The dormant function nobody reviewed. `stamp` is the one whose docstring says
    "the only way to attach one", so a missing row here is a silent salience of zero."""
    patch(root, MANIFEST, _row(root, "memory/salience.py::stamp()") + "\n", "")


def _mut_uncalled_instrument_removed(root: Path) -> None:
    """Drop the function-granularity instrument and 19 reviewed rows go stale at once —
    including the four in safety/guard.py, in a file the census still calls reached."""
    patch(root, CENSUS,
          '                seams.append((f"{rel}::{node.name}()", "uncalled"))\n',
          "                pass\n")


def _mut_defect_without_a_stage_id(root: Path) -> None:
    """A `defect` row that names no stage is a parking space, and the two rows in that
    class are the ones most likely to be left there."""
    old = _row(root, "safety/guard.py::spoken_text()")
    patch(root, MANIFEST, old, old.replace("R1.6 fixes it", "Someone should fix it"))


def _mut_uncalled_is_only_about_its_own_file(root: Path) -> None:
    """`uncalled` must mean "the robot never calls it", not "this file does not".
    Scoped to one file, every helper called from a sibling module reports as dormant."""
    patch(root, CENSUS,
          "    called_anywhere = {name for found in call_of.values() for name, _o, _k in found}",
          "    called_anywhere = {name for name, _o, _k in call_of.get('memory/store.py', [])}")


# ------------------------------------------------- the launch bootstrap (a third kind)
#
# These mutate a *launched node's own source* rather than the manifest or the census.
# That is the only place the defect can live: the census resolved `from memory import
# profile` statically and reported the seams behind it as reached, while the process it
# claimed to describe raised ModuleNotFoundError before `rclpy.init()`. So the tree is
# what has to be broken to prove the check reads it.

WRITER = "memory/memory_node.py"
BOOTSTRAP_LINE = "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))"


def writer_package_imports(root: Path) -> list[str]:
    """The single writer's own `from memory import ...` lines, read from the tree.

    **Derived, because the written-down version broke.** This used to be a two-line
    constant holding `profile` and `store`, and the mutant below anchored on
    `BOOTSTRAP_LINE + "\\n\\n" + <that constant>`. R1.2 legitimately added a third import
    (`rps_state`), which moved the anchor, and the failure the suite reported was *"mutant
    anchor no longer present"* — the harness saying it could not run, not the guard saying
    the tree was clean. That is the right direction to fail in, and it is still a mutant
    that stopped mutating: a list of names cannot notice a new name, and this one was
    pinned to the writer not growing an import.
    """
    lines = [ln for ln in (root / WRITER).read_text().splitlines()
             if ln.startswith("from memory import")]
    assert lines, f"no `from memory import` line in {WRITER} — the mutant has no anchor"
    return lines


def _mut_bootstrap_removed(root: Path) -> None:
    """The state this file was actually in until R1.1b: the single writer, launched as a
    file by run_ec2.sh, with no `memory` package inside `memory/` to import. Every
    offline test still passed, because pytest puts the repo root on sys.path first."""
    patch(root, WRITER, BOOTSTRAP_LINE + "\n", "")


def _mut_bootstrap_below_the_import(root: Path) -> None:
    """Present but useless. An import runs when it is read, so a bootstrap underneath it
    fixes sys.path for a process that already died. A check that only asked "is the line
    there" would call this tree clean, and the line is there.

    Moved below the **last** package import rather than the first, so the mutant stays a
    real one however many the writer grows: with any import left above the bootstrap, the
    process still dies where it always did."""
    last = writer_package_imports(root)[-1]
    patch(root, WRITER, BOOTSTRAP_LINE + "\n", "")
    patch(root, WRITER, last, last + "\n" + BOOTSTRAP_LINE)


def _mut_bootstrap_at_the_wrong_depth(root: Path) -> None:
    """One `.parent` short — the shape of the line is right and it points at `memory/`,
    which is the directory that was already on sys.path. This is what makes the depth a
    derived quantity rather than a constant: the census computes it from how far down
    the launch table says the file lives."""
    patch(root, WRITER, BOOTSTRAP_LINE,
          "sys.path.insert(0, str(Path(__file__).resolve().parent))")


# name -> (mutation, the finding it must provoke). The second element matters as much
# as the first: a mutant killed by some other check is still a surviving mutant for
# the check it was written to test, and "exit != 0" cannot tell those apart.
MUTANTS = {
    "row_deleted": (_mut_row_deleted, "memory/erase.py is unreached"),
    "stale_row_for_a_reached_module": (_mut_stale_row_for_a_reached_module,
                                       "which is no longer unreached"),
    "class_outside_the_closed_set": (_mut_class_outside_the_closed_set, "declares class"),
    "asymmetry_contradicts_the_census": (_mut_asymmetry_contradicts_the_census,
                                         "declared unpassed but censuses as unreached"),
    "reason_is_a_shrug": (_mut_reason_is_a_shrug, "-character reason"),
    "pending_stage_without_a_stage_id": (_mut_pending_stage_without_a_stage_id,
                                         "names no stage id"),
    "by_hand_entrypoint_names_another_tool": (_mut_by_hand_entrypoint_names_another_tool,
                                              "does not name itself"),
    "by_hand_library_names_nothing": (_mut_by_hand_library_names_nothing,
                                      "names no path that has a row"),
    "test_only_names_a_missing_test": (_mut_test_only_names_a_missing_test,
                                       "which does not exist"),
    "test_only_names_no_test": (_mut_test_only_names_no_test, "names no tests/ file"),
    "aliases_drop_the_bare_sibling": (_mut_aliases_drop_the_bare_sibling,
                                      "body/gestures.py is unreached"),
    "reach_is_not_transitive": (_mut_reach_is_not_transitive, "is unreached"),
    "tests_count_as_callers": (_mut_tests_count_as_callers,
                               f"{RECALL_SEAM} which is no longer unpassed"),
    "uncalled_row_deleted": (_mut_uncalled_row_deleted,
                             "memory/salience.py::stamp() is uncalled"),
    "uncalled_instrument_removed": (_mut_uncalled_instrument_removed,
                                    "which is no longer uncalled"),
    "defect_without_a_stage_id": (_mut_defect_without_a_stage_id,
                                  "is class defect but its reason names no stage id"),
    "uncalled_is_only_about_its_own_file": (_mut_uncalled_is_only_about_its_own_file,
                                            "is uncalled"),
    "bootstrap_removed": (_mut_bootstrap_removed,
                          f"{WRITER} is launched by {RUNNER} and imports 'memory' at line"),
    "bootstrap_below_the_import": (_mut_bootstrap_below_the_import,
                                   "a bootstrap below it is a bootstrap that never happens"),
    "bootstrap_at_the_wrong_depth": (
        _mut_bootstrap_at_the_wrong_depth,
        "no sys.path.insert of Path(__file__).resolve().parent.parent precedes it"),
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_named_mutant_is_killed(name: str, repo_copy: Path) -> None:
    mutate, expected = MUTANTS[name]
    mutate(repo_copy)
    proc = run_census(repo_copy)
    report = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"mutant {name!r} survived the census:\n{report}"
    if proc.returncode == 1:
        assert census_findings(proc) >= 1, (
            f"mutant {name!r} made the census exit 1 while reporting 0 findings — the "
            f"exit code and the report disagree:\n{report}"
        )
    assert expected in failures(proc), (
        f"mutant {name!r} was killed, but not by the check it targets (expected a "
        f"finding mentioning {expected!r}):\n{failures(proc)}"
    )


# ------------------------------------------------------ the hole, stated differentially


def _drop_the_unpassed_instrument(root: Path) -> None:
    """Not a MUTANTS entry: this one is *supposed* to lose the finding, which is the
    hole itself. Used by the differential test, where losing it is the assertion."""
    patch(root, CENSUS, "    for rel in sorted(reached & set(trees)):", "    for rel in []:")


@pytest.mark.mutation
def test_reachability_alone_loses_the_assemble_seam(repo_copy: Path) -> None:
    """The 26.6% that no guard could see, as a difference rather than as a story.

    `agent/context.py` is reached on every turn — agent_node calls `assemble` in its
    constructor — so no reachability instrument can ever report it. What is dead is
    what sits behind `if recall is not None`, and the parameter is the only place that
    shows. Asserting both directions is what makes the second instrument load-bearing.
    """
    found = run_census(repo_copy)
    assert found.returncode == 0 and census_findings(found) == 0

    _drop_the_unpassed_instrument(repo_copy)
    lost = run_census(repo_copy)
    report = lost.stdout + lost.stderr
    assert f"{RECALL_SEAM} which is no longer unpassed" in report, (
        "with the parameter instrument removed the census must go quiet about this "
        "seam — if it still reports it, something other than `unpassed` is finding it "
        "and that instrument is not what closed the hole:\n" + report
    )
    # And the file itself is never reported dormant, in either configuration. This is
    # the claim reachability cannot make, spelled out rather than implied.
    assert "agent/context.py\tunreached" not in report
    assert f"FAIL {RECALL_SEAM} is unpassed" not in report


GUARD_SEAMS = ("safety/guard.py::clamp_reply()", "safety/guard.py::spoken_text()",
               "safety/guard.py::gesture_findings()", "safety/guard.py::forget_request()")


@pytest.mark.mutation
def test_the_file_instrument_cannot_see_the_dormant_guard_functions(repo_copy: Path) -> None:
    """Why a third instrument, stated as a difference rather than as a preference.

    `safety/guard.py` is imported and called by `agent_node` on every turn — for
    `load_constitution` — so it is reached, and no file-granularity instrument can ever
    report it. Four functions inside that live file have no caller in the robot,
    including the deletion recogniser the plan already knew about and the two whose rule
    `agent_node` re-implements inline. Remove the function instrument and all four go
    quiet in a file the census still, correctly, calls reached.
    """
    found = run_census(repo_copy)
    assert found.returncode == 0 and census_findings(found) == 0

    _mut_uncalled_instrument_removed(repo_copy)
    lost = run_census(repo_copy)
    report = lost.stdout + lost.stderr
    assert "safety/guard.py\tunreached" not in report, (
        "the census now calls safety/guard.py unreachable, which would be wrong — "
        "agent_node calls load_constitution on every turn"
    )
    for seam in GUARD_SEAMS:
        assert f"{seam} which is no longer uncalled" in report, (
            f"{seam} did not go stale when the function instrument was removed, so "
            f"something else is reporting it and this instrument is not what found it"
        )


def _drop_the_bootstrap_check(root: Path) -> None:
    """Not a MUTANTS entry, for the same reason `_drop_the_unpassed_instrument` is not:
    removing this check makes the *real* tree greener, not redder, so exit-code mutation
    cannot see it. The loss has to be stated as a difference."""
    patch(root, CENSUS, "    out: list[str] = bootstrap_findings(root)",
          "    out: list[str] = []")


@pytest.mark.mutation
def test_no_other_instrument_notices_a_node_that_cannot_start(repo_copy: Path) -> None:
    """The false *reached*, as a difference. This is the direction the census exists to
    make impossible, and every other instrument in the file is blind to it.

    With `memory/memory_node.py`'s bootstrap removed the tree is broken in the strongest
    way this tool reports on — the single writer cannot import its own package, so it
    dies before `rclpy.init()` — and yet reachability still resolves the import, the
    function instrument still sees the calls, and the manifest still describes seams as
    wired. Remove `bootstrap_findings` and the census goes *green* on that tree. So the
    assertion is in two parts: red with the check, and green without it, because a red
    that some other instrument would also have produced proves nothing about this one.
    """
    _mut_bootstrap_removed(repo_copy)
    found = run_census(repo_copy)
    assert found.returncode == 1, found.stdout + found.stderr
    assert f"{WRITER} is launched by {RUNNER}" in failures(found)

    _drop_the_bootstrap_check(repo_copy)
    lost = run_census(repo_copy)
    assert lost.returncode == 0, (
        "some other instrument reported the broken tree, so this differential does not "
        "isolate the bootstrap check:\n" + failures(lost)
    )
    assert census_findings(lost) == 0, failures(lost)
    # And the census keeps calling the dead node's seams reached, which is the sentence
    # worth having on the record: the manifest and the census agreed about a process
    # that could not start.
    report = lost.stdout + lost.stderr
    assert f"{WRITER}\tunreached" not in report
    assert "memory/profile.py\tunreached" not in report


def test_a_bootstrap_check_that_examines_nothing_exits_3_not_0(repo_copy: Path) -> None:
    """Zero is not a pass. `bootstrap_findings` returning `[]` has two causes — every
    launched file bootstraps correctly, or the check found no package-qualified import to
    check — and they print identically. Emptying the package set stands in for the real
    ways that happens (the launch table stops naming nodes, or the import style changes),
    and the census must refuse to grade rather than report clean."""
    patch(repo_copy, CENSUS, "    packages = repo_packages(root)", "    packages = set()")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "examined nothing and reported" in failures(proc), failures(proc)


# --------------------------------------------------- the documented boundaries, tested
#
# On a synthetic tree, because neither boundary is reachable in this repo — see the
# module docstring. A synthetic tree is the weaker instrument, so what it is standing
# in for is named rather than left as a gap someone has to notice.

SYNTH_ROW = ("app/lib.py::f(real=)\tunpassed\tpending_stage\tR9.9 is the synthetic "
             "stage this fixture uses; the row exists so the manifest is non-empty "
             "and the seam is reviewed rather than unrowed\n")


# The synthetic tree's `app/main.py` is launched as a file by its synthetic runner, so it
# needs the same sys.path bootstrap a real node needs — `bootstrap_findings` reports it
# otherwise, and reports it *correctly*: without this line the fixture describes a node
# that raises ModuleNotFoundError. Written here rather than exempted, because a check
# taught to skip synthetic trees would be a check with a hole shaped like a test fixture.
SYNTH_BOOTSTRAP = ("import sys\nfrom pathlib import Path\n"
                   "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n")


def _synthetic(root: Path, lib: str, rows: str = SYNTH_ROW,
               main: str = "from app import lib\nlib.f()\n") -> Path:
    (root / "infra").mkdir(parents=True, exist_ok=True)
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "infra" / "run_ec2.sh").write_text("gateway|python3 app/main.py\n")
    (root / "infra" / "dormant_seams.tsv").write_text("# synthetic\n" + rows)
    (root / "infra" / "standalone_entrypoints.tsv").write_text("# synthetic, no rows\n")
    (root / "app" / "main.py").write_text(SYNTH_BOOTSTRAP + main)
    (root / "app" / "lib.py").write_text(lib)
    return root


def test_a_declared_test_hook_is_not_a_seam(tmp_path: Path, repo_root: Path) -> None:
    """Boundary 3. One run asserts both halves: `real=` must be censused (or its row
    reads stale) and `_hook=` must not be (or it reds as an unrowed seam). A parameter
    that announces itself as a hook is not a finding; one that does not, is."""
    root = _synthetic(tmp_path / "t", "def f(*, _hook=None, real=None):\n    return _hook, real\n")
    proc = run_census(root, script_root=repo_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert census_findings(proc) == 0, failures(proc)
    assert "_hook" not in proc.stdout + proc.stderr


def test_the_underscore_skip_is_the_reason_and_not_an_accident(tmp_path: Path,
                                                              repo_root: Path) -> None:
    """The discriminating half of the test above: rename the hook and it is censused.
    Without this, a census that simply missed both parameters would pass."""
    root = _synthetic(tmp_path / "t", "def f(*, hook=None, real=None):\n    return hook, real\n")
    proc = run_census(root, script_root=repo_root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "app/lib.py::f(hook=) is unpassed" in failures(proc)


def test_the_method_name_collapse_is_a_loss_the_census_declares(tmp_path: Path,
                                                               repo_root: Path) -> None:
    """Boundary 2, pinned as the loss it is. `A.cfg` and `B.cfg` are one name to a name
    matcher, so a call passing `clock=` to either would mark both supplied. Rather than
    report a seam it cannot attribute, the census skips methods — and `B.cfg(clock=)`
    going unreported here is that cost, asserted so it cannot silently grow."""
    root = _synthetic(
        tmp_path / "t",
        "class A:\n    def cfg(self, *, clock=None):\n        return clock\n"
        "class B:\n    def cfg(self, *, clock=None):\n        return clock\n"
        "def f(*, real=None):\n    return real\n",
        main="from app import lib\nlib.f()\nlib.A().cfg(clock=1)\n",
    )
    proc = run_census(root, script_root=repo_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cfg" not in proc.stdout + proc.stderr


@pytest.mark.mutation
def test_flipping_module_level_only_exits_3_not_0(repo_copy: Path) -> None:
    """The boundary refuses to be crossed quietly: a method-level census under name
    matching is unsound, so the census refuses to run rather than reporting seams it
    cannot attribute."""
    patch(repo_copy, CENSUS, "MODULE_LEVEL_ONLY = True", "MODULE_LEVEL_ONLY = False")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "MODULE_LEVEL_ONLY" in proc.stderr


# ------------------------------------------------------ the census's own failure modes


def test_missing_manifest_exits_3_not_0(repo_copy: Path) -> None:
    """Without the manifest every seam below is unreviewed, which must not read as
    clean — and with 23 seams and no rows, exit 1 would be almost as wrong as exit 0."""
    (repo_copy / MANIFEST).unlink()
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "is missing" in proc.stderr


def test_empty_manifest_exits_3_not_0(repo_copy: Path) -> None:
    """An empty manifest and a fully wired tree are different states. The day R1
    finishes, the manifest holds the by_hand rows and not nothing."""
    (repo_copy / MANIFEST).write_text("# every row removed\n")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "declares no rows" in proc.stderr


def test_missing_standalone_manifest_exits_3_not_0(repo_copy: Path) -> None:
    """The by_hand class is checked against another registry. With that registry gone
    the check cannot run, and 17 rows would silently become unverified prose."""
    (repo_copy / STANDALONE).unlink()
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr


def test_a_malformed_row_exits_3_not_0(repo_copy: Path) -> None:
    with (repo_copy / MANIFEST).open("a") as fh:
        fh.write("memory/somewhere.py\tunreached\tpending_stage\n")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "tab-separated" in proc.stderr


def test_a_second_row_for_one_seam_exits_3_not_0(repo_copy: Path) -> None:
    """Two reasons for one seam means one of them is unread, and which one the census
    happens to keep is an implementation detail nobody should have to know."""
    with (repo_copy / MANIFEST).open("a") as fh:
        fh.write(_row(repo_copy, "memory/erase.py").replace(
            "R1.5 wires it.", "R1.5 wires it, and this duplicate row says so twice.") + "\n")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "second row" in proc.stderr


def test_an_empty_launch_inventory_exits_3_not_0(repo_copy: Path) -> None:
    """Zero launched processes makes every file in the tree unreachable, so the census
    would report a fully dormant repo — the loudest possible wrong answer, and still
    an answer. It must refuse instead."""
    (repo_copy / RUNNER).write_text("#!/usr/bin/env bash\necho no node table here\n")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "launches 0 python files" in proc.stderr


def test_an_unparseable_file_exits_3_not_0(repo_copy: Path) -> None:
    """A file the census cannot parse must not have its seams default to reached."""
    (repo_copy / "memory" / "store.py").write_text("def broken(:\n")
    proc = run_census(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "does not parse" in proc.stderr


def test_reading_zero_files_exits_3_not_0(tmp_path: Path, repo_root: Path) -> None:
    """A census that read nothing prints the same empty finding list as a clean tree
    unless it refuses to. This is the `ran` flag, as an exit code."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    proc = run_census(empty, script_root=repo_root)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "read 0 python files" in proc.stderr


# ---------------------------------------------------------------- the fixed defect
#
# Stated in behaviour, not in source: what is actually broken today is that the robot
# cannot recall anything, and the manifest row is the receipt for knowing it.


def test_the_dormant_promises_each_have_a_named_stage(repo_root: Path) -> None:
    """Every promise the running robot does not keep names the stage that lands it.

    This is the assertion that turns §1.1's table from an analysis someone did once
    into a fact the suite maintains. When R1 finishes, these rows are gone and this
    test's set comprehension is empty — which is why it asserts a property of whatever
    rows exist rather than a fixed list of five.
    """
    import re

    pending = {}
    for line in (repo_root / MANIFEST).read_text().splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        seam, _asym, cls, reason = line.split("\t")
        if cls == "pending_stage":
            pending[seam] = re.findall(r"\bR[0-9]+\.[0-9]+\b", reason)

    for seam, stages in pending.items():
        assert stages, f"{seam} is pending on no named stage"
    # The recall seam is the one this whole file was written for, so it is asserted by
    # name: 1757 production lines behind one omitted argument. When R1.4 lands, this
    # row is deleted and this assertion is the one that must be deleted with it — a
    # deliberate coupling, so the wiring PR cannot forget the manifest.
    assert RECALL_SEAM in pending, (
        f"{RECALL_SEAM} has no pending_stage row. If R1.4 has landed, this assertion "
        f"and the manifest row are both stale and belong in that same PR"
    )
