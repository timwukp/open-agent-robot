"""T-J1 deploy path + T-J4 mutation liveness for infra/check_deploy.sh.

A guard is a claim about the tree, and an unfalsified claim is worth nothing. So
every check in check_deploy.sh has a named mutant here that must turn it red, plus
an unmutated control that must stay green. If a mutant survives, the guard is
decorative; if the control fails, the harness is broken and every red below is
uninterpretable — which is why the control is asserted first, not last.

The mutants are the real defect history of this repo, not invented ones:
`yue_not_wired` is the bug that shipped, and `unwired_new_entrypoint` is the same
bug arriving again in a file nobody has written yet.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from conftest import guard_checks, guard_findings, patch, run_guard

# --------------------------------------------------------------------------- control


def test_control_the_real_tree_is_green(repo_root: Path) -> None:
    proc = run_guard(repo_root)
    assert guard_findings(proc) == 0, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_control_an_unmutated_copy_is_green(repo_copy: Path) -> None:
    """Proves the copy itself (and conftest's docs/daemon omission) changes no verdict.

    Without this, a mutant that turns red because copying broke something would be
    scored as the guard working.
    """
    proc = run_guard(repo_copy)
    assert guard_findings(proc) == 0, proc.stdout + proc.stderr
    assert proc.returncode == 0


def test_the_guard_actually_checks_a_useful_number_of_things(repo_root: Path) -> None:
    # A guard that silently degrades to two checks still exits 0. The floor is a
    # floor, not the current count: it must not need editing on every new node.
    assert guard_checks(run_guard(repo_root)) >= 20


# --------------------------------------------------------------------------- mutants

RUNNER = "infra/run_ec2.sh"
SETUP = "infra/setup-ec2.sh"
PATHS_TSV = "infra/generated_paths.tsv"
STANDALONE_TSV = "infra/standalone_entrypoints.tsv"


def _mut_yue_not_wired(root: Path) -> None:
    """The bug that shipped: the Cantonese engine has no line in the node table."""
    patch(root, RUNNER, "yue|$HOME/venv-yue/bin/python speech/yue_server.py\n", "")


def _mut_second_hand_kept_list(root: Path) -> None:
    """The mechanism behind the bug: a literal list that can drift from the table."""
    patch(root, RUNNER, 'ALL=$(printf \'%s\\n\' "$NODES" | awk -F\'|\' \'NF{printf "%s ", $1}\')',
          'ALL="gateway body asr tts hand detect agent game"')


def _mut_status_reports_only_core(root: Path) -> None:
    """Started set and reported set drift: a node dies and status never says so."""
    patch(root, RUNNER, "  status)\n    for n in $ALL; do", "  status)\n    for n in $CORE; do")


def _mut_unknown_node_is_silent(root: Path) -> None:
    """A typo in a deploy script becomes a successful no-op."""
    patch(root, RUNNER,
          '  if ! cmd=$(cmd_for "$n"); then\n    echo "$n: unknown node (known: $ALL)" >&2; return 1\n  fi',
          '  cmd=$(cmd_for "$n") || cmd=""')


def _mut_asset_path_has_no_producer(root: Path) -> None:
    """A fetched path is renamed, so the manifest row no longer backs what runs."""
    patch(root, RUNNER, "body/assets/unitree_g1/scene_with_hands.xml",
          "body/assets/unitree_g1/scene_renamed.xml")


def _mut_producer_never_invoked(root: Path) -> None:
    """fetch_assets.sh goes back to being an orphan: real file, nothing calls it."""
    patch(root, SETUP, '  "$(dirname "$0")/../body/fetch_assets.sh"\n', "")


def _mut_producer_deleted(root: Path) -> None:
    (root / "body" / "fetch_assets.sh").unlink()


def _mut_producer_not_executable(root: Path) -> None:
    p = root / "body" / "fetch_assets.sh"
    p.chmod(p.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _mut_stage_removed_from_dispatch(root: Path) -> None:
    """The venv install exists as a shell function but no human can reach it."""
    patch(root, SETUP, "  yue)    stage_yue ;;\n", "")


def _mut_manifest_evidence_is_bogus(root: Path) -> None:
    """The row stops proving anything about its producer."""
    patch(root, PATHS_TSV, "venv-yue/bin\t", "totally-unrelated\t")


def _mut_stale_manifest_row(root: Path) -> None:
    """An excuse left behind for a path nothing executes any more."""
    with (root / PATHS_TSV).open("a") as fh:
        fh.write("$HOME/venv-gone/bin/python\tinfra/setup-ec2.sh\tvenv/bin\tinfra/setup-ec2.sh:stack\n")


def _mut_unwired_new_entrypoint(root: Path) -> None:
    """The original defect class, arriving in a brand-new file: a complete,
    runnable node that nothing in the repo ever starts."""
    (root / "perception" / "depth_node.py").write_text(
        '"""A finished node that nobody launches."""\n\n\n'
        'def main():\n    pass\n\n\n'
        'if __name__ == "__main__":\n    main()\n'
    )


def _mut_launched_node_declared_standalone(root: Path) -> None:
    """A contradictory exemption: claimed standalone while the runner launches it."""
    with (root / STANDALONE_TSV).open("a") as fh:
        fh.write("speech/tts_node.py\tclaiming an exemption it does not have\n")


def _mut_stale_standalone_exemption(root: Path) -> None:
    """An exemption for a file that no longer exists silently covers its successor."""
    with (root / STANDALONE_TSV).open("a") as fh:
        fh.write("infra/deleted_tool.py\tan exemption outliving its file\n")


# name -> (mutation, the finding it must provoke). The second element matters as
# much as the first: a mutant killed by some *other* check is still a surviving
# mutant for the check it was written to test, and "exit != 0" alone cannot tell
# those apart.
MUTANTS = {
    "yue_not_wired": (_mut_yue_not_wired, "speech/yue_server.py"),
    "second_hand_kept_list": (_mut_second_hand_kept_list, "literal ALL="),
    "status_reports_only_core": (_mut_status_reports_only_core, "NOT reported by status"),
    "unknown_node_is_silent": (_mut_unknown_node_is_silent, "unknown node exited 0"),
    "asset_path_has_no_producer": (_mut_asset_path_has_no_producer,
                                   "neither in the tree nor declared"),
    "producer_never_invoked": (_mut_producer_never_invoked, "never calls fetch_assets.sh"),
    "producer_deleted": (_mut_producer_deleted, "producer does not exist"),
    "producer_not_executable": (_mut_producer_not_executable, "producer is not executable"),
    "stage_removed_from_dispatch": (_mut_stage_removed_from_dispatch, "has no 'yue)' stage"),
    "manifest_evidence_is_bogus": (_mut_manifest_evidence_is_bogus, "is not part of the path"),
    "stale_manifest_row": (_mut_stale_manifest_row, "venv-gone"),
    "unwired_new_entrypoint": (_mut_unwired_new_entrypoint, "perception/depth_node.py"),
    "launched_node_declared_standalone": (_mut_launched_node_declared_standalone,
                                          "contradictory declaration"),
    "stale_standalone_exemption": (_mut_stale_standalone_exemption, "stale exemption"),
}


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_named_mutant_is_killed(name: str, repo_copy: Path) -> None:
    mutate, expected = MUTANTS[name]
    mutate(repo_copy)
    proc = run_guard(repo_copy)
    report = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"mutant {name!r} survived the guard:\n{report}"
    if proc.returncode == 1:
        assert guard_findings(proc) >= 1, (
            f"mutant {name!r} made the guard exit 1 while reporting 0 findings — "
            f"the exit code and the report disagree:\n{report}"
        )
    failures = "\n".join(l for l in report.splitlines() if l.startswith(("FAIL", "ERROR")))
    assert expected in failures, (
        f"mutant {name!r} was killed, but not by the check it targets "
        f"(expected a finding mentioning {expected!r}):\n{failures}"
    )


# ------------------------------------------------------- the guard's own failure mode


def test_guard_cannot_run_exits_3_not_0(repo_copy: Path) -> None:
    """A guard that cannot run must not report clean (exit 3, never 0)."""
    runner = repo_copy / RUNNER
    runner.chmod(runner.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    proc = run_guard(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "ERROR" in proc.stderr


def test_missing_manifest_exits_3_not_0(repo_copy: Path) -> None:
    (repo_copy / PATHS_TSV).unlink()
    proc = run_guard(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr


def test_empty_inventory_exits_3_not_0(repo_copy: Path) -> None:
    """Zero nodes scanned is an error, not a pass — a guard that finds nothing to
    check has the same output as a clean tree unless it says so."""
    patch(repo_copy, RUNNER, "NODES=$(cat <<EOF\n", "NODES=$(cat <<EOF\nSKIP\nEOF\n)\nUNUSED=$(cat <<EOF\n")
    proc = run_guard(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr


def test_no_python_entrypoints_exits_3_not_0(repo_copy: Path, monkeypatch) -> None:
    """The scanned-file floor: `grep -rl` reading nothing must not look clean."""
    for py in repo_copy.rglob("*.py"):
        if "tests/" not in str(py.relative_to(repo_copy)):
            py.write_text("# emptied\n")
    proc = run_guard(repo_copy)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "0 python entrypoints" in proc.stderr


# ------------------------------------------------------------------ the fixed defect


def test_the_cantonese_engine_is_wired(repo_root: Path) -> None:
    """The regression test for the shipped bug, stated in behaviour not in source:
    speech/tts_node.py routes Cantonese text to 127.0.0.1:8901 with no fallback, so
    `yue` missing from the runner is a hard TTS failure on any Cantonese reply."""
    import subprocess

    env = {**os.environ, "OAR_DRY": "1", "OAR_RUN": "/nonexistent"}
    out = subprocess.run([str(repo_root / RUNNER), "start-all"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "yue: would run" in out
    assert "speech/yue_server.py" in out
    status = subprocess.run([str(repo_root / RUNNER), "status"],
                            capture_output=True, text=True, env=env, check=True).stdout
    assert "yue:" in status
