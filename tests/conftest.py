"""Shared fixtures and the collection floor (test item T-J2).

The floor exists because a suite that collects nothing exits successfully in most
harnesses, and "0 tests ran" then reads exactly like "everything passed". pytest
already exits 5 when the *whole* session collects nothing; what it does not catch
is a single test module that silently contributes zero items — a module-level
`skip`, an `if False:` guard, or a rename that stops matching `test_*`. So the
floor is per-module and the module list is *derived from the directory*, never
written down: a list cannot notice a new name.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Pure modules under test import as `memory.store`, `game.opponent`, … with no
# package installation step; ROS nodes are never imported by the suite at all.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail the run if any test module contributed zero collected items."""
    expected = {p.name for p in TESTS_DIR.glob("test_*.py")}
    if not expected:
        raise pytest.UsageError(
            f"no test modules found in {TESTS_DIR} — an empty suite is an error, not a pass"
        )
    # `-k` / `-m` deselect on purpose, so the floor cannot distinguish an
    # intentionally empty module from a silently empty one; it stands down rather
    # than becoming an obstacle people learn to route around.
    if session.config.option.keyword or session.config.option.markexpr:
        return
    # Narrow to explicitly named files so `pytest tests/test_one.py` is not judged
    # against modules the invocation never asked for.
    named = {Path(a.split("::")[0]).name
             for a in session.config.invocation_params.args
             if a.split("::")[0].endswith(".py")}
    if named:
        expected &= named
    seen = {Path(str(item.fspath)).name for item in session.items}
    silent = sorted(expected - seen)
    if silent:
        raise pytest.UsageError(
            "these test modules collected 0 items, so their assertions never ran: "
            + ", ".join(silent)
        )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A throwaway copy of the tree, for tests that mutate deploy scripts.

    `docs/` and `daemon/` are omitted only because they are 11 MB and hold nothing
    the deploy guard reads (no `.py`, no runner, no producer). The unmutated-control
    test in test_deploy_path.py is what proves that omission changed no verdict: if
    it ever does, the control turns red rather than a mutant turning quietly green.
    """
    dst = tmp_path / "tree"
    shutil.copytree(
        REPO_ROOT,
        dst,
        ignore=shutil.ignore_patterns("docs", "daemon", "__pycache__", ".pytest_cache", ".git"),
    )
    return dst


def run_guard(root: Path) -> subprocess.CompletedProcess:
    """Run infra/check_deploy.sh inside `root` and return the completed process."""
    return subprocess.run(
        [str(root / "infra" / "check_deploy.sh")],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


def guard_findings(proc: subprocess.CompletedProcess) -> int:
    """Read the finding count out of the guard's own summary line.

    The number is parsed from the report rather than inferred from the exit code,
    so a guard that exits 1 while reporting 0 findings is itself a failure.
    """
    for line in reversed((proc.stdout + proc.stderr).splitlines()):
        if "findings" in line and "checks passed" in line:
            return int(line.split()[3])
    raise AssertionError(
        "the guard printed no summary line, so it cannot be graded:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def guard_checks(proc: subprocess.CompletedProcess) -> int:
    for line in reversed((proc.stdout + proc.stderr).splitlines()):
        if "findings" in line and "checks passed" in line:
            return int(line.split()[0])
    raise AssertionError("the guard printed no summary line")


def patch(root: Path, rel: str, old: str, new: str, count: int = 1) -> None:
    """Textually mutate a file in a copied tree, asserting the anchor still exists.

    Without the anchor assertion a stale mutant becomes a no-op, the guard finds
    nothing, and the test reports "mutant survived" — blaming the guard for a
    broken mutant. The two failures need different messages.
    """
    p = root / rel
    src = p.read_text()
    assert old in src, f"mutant anchor no longer present in {rel}: {old!r}"
    p.write_text(src.replace(old, new, count))
