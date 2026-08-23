#!/usr/bin/env python3
"""Derive the set of python entrypoints in the tree, for check 8 of check_deploy.sh.

Check 8 asks "is every runnable entrypoint launched by something?". It used to answer
the first half with `grep -rl '^if __name__ =='`, and that pattern has a hole big
enough to hide the worst case: **a script with no `__main__` guard at all runs its
work at import time, so it is more of an entrypoint than a guarded file, and the grep
cannot see it.** `.push/push.py` was exactly that — a top-level script that created
branches and pushed blobs the moment it was imported — and check 8 called the tree
clean while flagging its guarded sibling.

So an entrypoint is defined here as: a module that either (a) has a top-level
`if __name__ == "__main__"` block, or (b) does work at import time. "Work" is any
top-level statement that is not an import, a def/class, an assignment, or a
docstring — with one exclusion, `sys.path` mutation, which is this repo's idiom for
letting a node import its sibling packages and is not what a module *does*.

That exclusion is the one judgement call in this file, so it is asserted rather than
trusted: tests/test_entrypoints.py pins the exclusion to `sys.path` specifically and
fails if it ever widens to "any call on sys" or "any expression statement".

Exit codes follow the guard convention: 0 with the list on stdout, 3 if the scan
could not be trusted (zero files read, or a file that would not parse — an
unparseable file cannot be classified, and an unclassified file must not be silently
treated as "not an entrypoint").
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Top-level statement kinds that are declaration, not work.
DECLARATIVE = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.ClassDef, ast.Assign, ast.AnnAssign)

SKIP_DIR_PARTS = ("tests", "__pycache__", "venv", "venv-yue", "node_modules",
                  ".git", "site-packages")


def is_sys_path_bootstrap(node: ast.stmt) -> bool:
    """True for `sys.path.insert(...)` / `sys.path.append(...)` and nothing else.

    Narrow on purpose. `sys.exit(...)`, `sys.stdout.write(...)` and every other call
    on `sys` are work, and a module doing them at import time is an entrypoint.
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if not isinstance(func, ast.Attribute) or func.attr not in ("insert", "append", "extend"):
        return False
    target = func.value                                   # expect `sys.path`
    return (isinstance(target, ast.Attribute) and target.attr == "path"
            and isinstance(target.value, ast.Name) and target.value.id == "sys")


def is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and any(
        isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(node.test))


def is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def import_time_work(tree: ast.Module) -> list[int]:
    """Line numbers of top-level statements that do work when the module is imported."""
    lines = []
    for node in tree.body:
        if isinstance(node, DECLARATIVE) or is_docstring(node) or is_main_guard(node):
            continue
        if is_sys_path_bootstrap(node):
            continue
        lines.append(node.lineno)
    return lines


def classify(source: str) -> tuple[bool, list[int]]:
    """(has a __main__ guard, line numbers of import-time work)."""
    tree = ast.parse(source)
    return any(is_main_guard(n) for n in tree.body), import_time_work(tree)


def candidates(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIR_PARTS for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def discover(root: Path) -> list[str]:
    """Relative paths of every entrypoint under `root`, sorted.

    Raises ValueError if the scan read no files, or if any candidate would not parse:
    a scan that read nothing has the same empty output as a clean tree, and a file
    that cannot be classified must not default to "not an entrypoint".
    """
    files = candidates(root)
    if not files:
        raise ValueError(f"read 0 python files under {root} — a scan that reads "
                         f"nothing is an error, not an empty result")
    found = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            guard, work = classify(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise ValueError(f"{rel} does not parse ({exc}) — cannot classify it, and "
                             f"an unclassified file must not be assumed harmless") from exc
        if guard or work:
            found.append(rel)
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent)
    try:
        for rel in discover(root.resolve()):
            print(rel)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
