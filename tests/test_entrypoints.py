"""T-J1 support: infra/entrypoints.py, the discovery half of check 8.

Check 8 of infra/check_deploy.sh answers "is every runnable entrypoint launched by
something?". Its old discovery, `grep -rl '^if __name__ =='`, had a hole that the
guard's own report made invisible: `.push/push.py` did its work at import time with
no guard at all, so it was never a candidate, and the guard printed a clean tree
while flagging the guarded sibling next to it.

Both directions are pinned here, because a discovery step has two ways to fail and
they need different tests:

  * too narrow — an unguarded import-time script is not found (the shipped hole)
  * too wide — the `sys.path` bootstrap that every node in this repo uses gets
    counted as work, and check 8 starts demanding a deploy path for library modules

The second is the reason `is_sys_path_bootstrap` exists, and it is the one judgement
call in the file, so the mutants below make it load-bearing in both directions.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "infra") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "infra"))

import entrypoints as ep  # noqa: E402

# --------------------------------------------------------------------- classification

GUARDED = '"""doc."""\nimport sys\n\n\ndef main():\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n'
BOOTSTRAP_ONLY = '"""doc."""\nimport sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parent))\n\nWORDS = ("a",)\n\n\ndef f():\n    pass\n'
BARE_SCRIPT = '"""doc."""\nimport sys\n\nREPO = "x"\nprint(REPO)\n'


def test_a_main_guard_is_an_entrypoint() -> None:
    assert ep.classify(GUARDED) == (True, [])


def test_a_module_whose_only_top_level_call_is_the_sys_path_bootstrap_is_not() -> None:
    """The over-wide failure mode. safety/guard.py is exactly this shape: a pure
    library with a sys.path bootstrap, and it must never need a deploy path."""
    assert ep.classify(BOOTSTRAP_ONLY) == (False, [])


def test_an_unguarded_script_is_an_entrypoint_and_this_is_the_shipped_hole() -> None:
    """`.push/push.py` created branches on import and the grep could not see it."""
    guard, work = ep.classify(BARE_SCRIPT)
    assert guard is False
    assert work == [5], work          # the print, not the docstring and not the assign


def test_the_bootstrap_exclusion_is_sys_path_and_not_sys_in_general() -> None:
    """A module that calls sys.exit or writes to stdout on import does work.

    If the exclusion ever widens to "any call whose target is sys", these two stop
    being entrypoints and a real script hides behind the bootstrap idiom.
    """
    for line in ("sys.exit(1)", "sys.stdout.write('x')",
                 "os.path.insert(0, 'x')", "path.insert(0, 'x')"):
        src = f"import sys\n{line}\n"
        assert ep.classify(src)[1] == [2], f"{line!r} should count as import-time work"


def test_known_limit_a_top_level_assignment_is_declarative_even_if_it_calls() -> None:
    """The honest boundary of this classifier, asserted so it is a decision not a gap.

    `X = f()` does run `f` on import, so `app = Flask(__name__)` would slip through.
    Assignments are nonetheless treated as declaration because this tree has 28
    top-level `re.compile(...)` / `Path(...).resolve()` constants and every one of
    them would otherwise demand a deploy path. If an assignment-shaped entrypoint
    ever appears here, this test is the line to delete — not the one to route around.
    """
    assert ep.classify("import re\nX = re.compile('a')\n") == (False, [])


def test_the_bootstrap_exclusion_covers_the_three_forms_this_repo_uses() -> None:
    for call in ("sys.path.insert(0, 'x')", "sys.path.append('x')", "sys.path.extend(['x'])"):
        assert ep.classify(f"import sys\n{call}\n")[1] == [], call


@pytest.mark.parametrize("stmt", ["for x in (1,):\n    pass", "while False:\n    pass",
                                  "with open('f') as fh:\n    pass", "assert True",
                                  "raise SystemExit(0)", "try:\n    pass\nexcept Exception:\n    pass",
                                  "if os.environ.get('X'):\n    pass"])
def test_control_flow_at_module_level_is_work(stmt: str) -> None:
    assert ep.classify(f"import os\n{stmt}\n")[1] == [2], stmt


@pytest.mark.parametrize("stmt", ['"""doc."""', "import os", "from os import path",
                                  "X = 1", "X: int = 1", "def f():\n    pass",
                                  "class C:\n    pass"])
def test_declarations_are_not_work(stmt: str) -> None:
    assert ep.classify(f"{stmt}\n") == (False, []), stmt


# ------------------------------------------------------------------------- discovery


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return root


def test_discover_finds_both_shapes_and_skips_libraries(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a/guarded.py": GUARDED, "b/bare.py": BARE_SCRIPT,
                            "c/lib.py": BOOTSTRAP_ONLY})
    assert ep.discover(root) == ["a/guarded.py", "b/bare.py"]


def test_discover_skips_the_suite_itself(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"a.py": GUARDED, "tests/test_x.py": BARE_SCRIPT,
                            "tests/conftest.py": BARE_SCRIPT})
    assert ep.discover(root) == ["a.py"]


def test_a_scan_that_reads_zero_files_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    """Zero files read and a tree of pure libraries produce the same empty list, so
    the first has to be an error. Otherwise a mistyped root reports clean."""
    (tmp_path / "readme.md").write_text("no python here\n")
    with pytest.raises(ValueError, match="read 0 python files"):
        ep.discover(tmp_path)


def test_an_unparseable_file_raises_rather_than_being_classified_harmless(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"ok.py": GUARDED})
    (root / "broken.py").write_text("def f(\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not parse"):
        ep.discover(root)


def test_the_cli_exits_3_when_it_cannot_classify(tmp_path: Path, capsys) -> None:
    (tmp_path / "readme.md").write_text("x\n")
    assert ep.main(["entrypoints.py", str(tmp_path)]) == 3
    assert "ERROR" in capsys.readouterr().err


# ------------------------------------------------------------------ on the real tree


def test_the_real_tree_lists_the_pusher_that_the_old_grep_could_not_see(repo_root: Path) -> None:
    found = ep.discover(repo_root)
    assert ".push/push_branch.py" in found
    assert "infra/entrypoints.py" in found
    assert "safety/guard.py" not in found, (
        "guard.py's only top-level call is the sys.path bootstrap; counting it would "
        "make check 8 demand a deploy path for a library")
    assert len(found) > 10, found


# --------------------------------------------------------------------------- mutants


@pytest.mark.mutation
def test_mutant_discovery_that_only_looks_for_main_guards(repo_root: Path,
                                                          monkeypatch) -> None:
    """The shipped hole, reintroduced: no import-time analysis at all."""
    monkeypatch.setattr(ep, "import_time_work", lambda tree: [])
    found = ep.discover(repo_root)
    assert ".push/push_branch.py" in found, "this file has a guard, so it survives"
    src = "import sys\nsys.exit(1)\n"
    assert ep.classify(src)[1] == [], "the mutant is not in effect"
    # and the property the real discovery has is gone:
    assert ep.classify(BARE_SCRIPT) == (False, []), (
        "with import-time analysis removed, an unguarded script looks like a library "
        "— which is exactly how .push/push.py stayed invisible")


@pytest.mark.mutation
def test_mutant_the_bootstrap_exclusion_removed_makes_libraries_entrypoints(
        repo_root: Path, monkeypatch) -> None:
    """The over-wide direction: without the exclusion, real library modules in this
    tree start demanding a deploy path."""
    monkeypatch.setattr(ep, "is_sys_path_bootstrap", lambda node: False)
    found = ep.discover(repo_root)
    assert "safety/guard.py" in found, (
        "guard.py must be the file that turns up, or this mutant is not reaching the "
        "classifier at all")


@pytest.mark.mutation
def test_control_unmutated_discovery_disagrees_with_both_mutants(repo_root: Path) -> None:
    """Without this, a monkeypatch that silently failed to apply would score as the
    mutant being killed."""
    found = ep.discover(repo_root)
    assert "safety/guard.py" not in found
    assert ep.classify(BARE_SCRIPT)[1] == [5]
