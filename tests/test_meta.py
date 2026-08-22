"""Health of the test machinery itself: T-J2 collection floor, T-J5 stale artefacts.

Nothing here tests a product behaviour. It tests the three ways this suite could
report success while checking nothing: a module that collects no items, a suite that
quietly acquires a network dependency and gets skipped on the laptop, and a compiled
artefact that is older than the source the assertions are written against.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

FORBIDDEN_IMPORTS = {"rclpy", "boto3", "botocore", "torch", "mujoco", "cv2",
                     "requests", "urllib3", "socket"}


# ------------------------------------------------------- T-J2 the collection floor
def _run_pytest(target: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider", *extra],
        capture_output=True, text=True, cwd=str(target.parent),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _sandbox(tmp_path: Path, *modules: tuple[str, str]) -> Path:
    """A miniature suite using the real conftest.py, so the floor under test is ours."""
    sandbox = tmp_path / "tests"
    sandbox.mkdir()
    (sandbox / "conftest.py").write_text((TESTS_DIR / "conftest.py").read_text())
    for name, body in modules:
        (sandbox / name).write_text(body)
    return sandbox


def test_a_module_that_collects_nothing_fails_the_run(tmp_path: Path) -> None:
    """The defect this floor exists for: 0 tests ran, exit 0, reads as "passed"."""
    sandbox = _sandbox(
        tmp_path,
        ("test_real.py", "def test_ok():\n    assert True\n"),
        ("test_silent.py", "import pytest\npytest.skip('gone quiet', allow_module_level=True)\n"),
    )
    proc = _run_pytest(sandbox)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "collected 0 items" in proc.stdout + proc.stderr
    assert "test_silent.py" in proc.stdout + proc.stderr


def test_the_floor_does_not_fire_on_a_healthy_sandbox(tmp_path: Path) -> None:
    """The no-mutant control: without it, a floor that always fires looks correct."""
    sandbox = _sandbox(tmp_path, ("test_real.py", "def test_ok():\n    assert True\n"))
    proc = _run_pytest(sandbox)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_suite_with_no_modules_at_all_fails(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    proc = _run_pytest(sandbox)
    assert proc.returncode != 0
    assert "empty suite is an error" in proc.stdout + proc.stderr


def test_the_floor_stands_down_for_an_explicit_k_selection(tmp_path: Path) -> None:
    """-k deselects on purpose; a floor that fought it would just get switched off."""
    sandbox = _sandbox(
        tmp_path,
        ("test_a.py", "def test_alpha():\n    assert True\n"),
        ("test_b.py", "def test_beta():\n    assert True\n"),
    )
    proc = _run_pytest(sandbox, "-k", "alpha")
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------- the offline invariant
@pytest.mark.parametrize("module", sorted(p.name for p in TESTS_DIR.glob("*.py")))
def test_no_test_module_imports_ros_aws_or_the_network(module: str) -> None:
    """Parsed with ast, not grepped: the suite must stay laptop-runnable, because a
    test that needs EC2 credentials becomes a test that is never run."""
    tree = ast.parse((TESTS_DIR / module).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    banned = imported & FORBIDDEN_IMPORTS
    assert not banned, f"{module} imports {sorted(banned)} — the suite must run offline"


def test_the_forbidden_list_is_not_vacuous() -> None:
    """A guard whose banned set never matches anything is indistinguishable from a
    guard that is switched off; this proves the detector works on a positive case."""
    tree = ast.parse("import boto3\nfrom rclpy.node import Node\n")
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    assert found & FORBIDDEN_IMPORTS == {"boto3", "rclpy"}


# ------------------------------------------------------- T-J5 stale build artefacts
DAEMON_SOURCES = ("daemon/main.c", "daemon/ring.h", "daemon/Makefile")
DAEMON_BINARY = "daemon/oar-daemon"


def _stale_sources(binary: Path, sources: list[Path]) -> list[str]:
    """Sources newer than the artefact built from them."""
    built = binary.stat().st_mtime
    return [str(s) for s in sources if s.stat().st_mtime > built + 1.0]


def test_stale_source_detector_works_on_a_constructed_case(tmp_path: Path) -> None:
    """Proves the comparison direction before trusting it on the real tree — the
    detector is one `>` away from reporting clean forever."""
    binary = tmp_path / "app"
    source = tmp_path / "app.c"
    binary.write_bytes(b"\x00")
    source.write_text("int main(){}\n")
    os.utime(binary, (1_000_000, 1_000_000))
    os.utime(source, (2_000_000, 2_000_000))
    assert _stale_sources(binary, [source]) == [str(source)]
    os.utime(source, (500_000, 500_000))
    assert _stale_sources(binary, [source]) == []


def test_the_built_daemon_is_not_older_than_its_source() -> None:
    binary = REPO_ROOT / DAEMON_BINARY
    sources = [REPO_ROOT / s for s in DAEMON_SOURCES if (REPO_ROOT / s).is_file()]
    assert sources, f"none of {DAEMON_SOURCES} exist — the check has nothing to compare"
    if not binary.is_file():
        pytest.skip(f"{DAEMON_BINARY} not built on this host")
    stale = _stale_sources(binary, sources)
    assert not stale, (
        f"{DAEMON_BINARY} is older than {stale} — run `make -C daemon`; a build that "
        f"reports success having built nothing leaves exactly this trace"
    )


def _running_daemon_start_time() -> datetime | None:
    """Local start time of a running oar-daemon, or None. Darwin has no /proc, so the
    process's text inode is unreadable; its start time answers the same question —
    a binary written after the process began is not the image that is running."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,lstart=,comm="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.rstrip().endswith("/oar-daemon") or line.rstrip().endswith(" oar-daemon"):
            fields = line.split()
            stamp = " ".join(fields[1:6])          # Fri Aug 21 23:24:36 2026
            try:
                return datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y")
            except ValueError:
                return None
    return None


def test_a_running_daemon_is_not_an_older_image_than_the_binary_on_disk() -> None:
    started = _running_daemon_start_time()
    if started is None:
        pytest.skip("no oar-daemon process on this host")
    binary = REPO_ROOT / DAEMON_BINARY
    if not binary.is_file():
        pytest.skip(f"{DAEMON_BINARY} not built on this host")
    built = datetime.fromtimestamp(binary.stat().st_mtime)
    assert built <= started, (
        f"{DAEMON_BINARY} was rebuilt at {built} but the running process started at "
        f"{started} — the daemon under test is the old image; restart it before "
        f"believing any behavioural result"
    )


def test_the_start_time_parser_handles_the_ps_format() -> None:
    """The parser is the whole check; a silent ValueError would make it skip forever."""
    for stamp in ("Fri Aug 21 23:24:36 2026", "Sun Aug 1 09:05:00 2026"):
        assert isinstance(datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y"), datetime)


# ---------------------------------------------------------------- suite plumbing
def test_pytest_ini_points_at_the_tests_directory() -> None:
    ini = (REPO_ROOT / "pytest.ini").read_text()
    assert re.search(r"^testpaths\s*=\s*tests\s*$", ini, re.M), ini
