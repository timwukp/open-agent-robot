"""The one derivation of the `/speech/say` request shape (used by T-E1 and T-D4).

Two tests need to know which keys a `/speech/say` message may carry:
`tests/test_disclosure.py` checks that every disclosure utterance uses only keys a
reader actually reads, and `tests/test_guard.py` checks that the key set contains no
amplitude S2's volume cap would have to guard. Giving each of them its own AST walk is
the two-readers-one-format defect at the level of the test suite: each census pins its
own answer, so when the format moves, one of them can stay green.

So there is one census here, and both import it.

**Why the reader set is derived rather than listed.** It used to be "the keys
`speech/tts_node.py` reads", and that was true until S7 moved the `id` default into
`speech/voices.py:request_id()` to give the uninterruptible-id namespace a single
creator. The census then derived `{"cache", "text"}` — `id` had not stopped being
accepted, it had stopped being *visible to the census*, which is the failure mode a
written-down file list cannot report. What sees the payload is `tts_node.py` plus
whatever repo-local module `tts_node.py` hands it to, so that is what is walked:
transitively, from the node's own imports.

Over-inclusion is the safe direction here. A local module that happens to read some
unrelated `req`/`d` dict adds a key, and the exact-equality assertion in
`test_guard.py` turns red and demands a human look — whereas under-inclusion silently
shrinks the set every subset assertion is checked against.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAY_ENTRY = REPO_ROOT / "speech" / "tts_node.py"

# The names a request dict is bound to at its readers. Both are needed: `speak(req)`
# in the node, `request_id(req)` in voices, and the `d`-shaped handlers that predate
# them. A key read off a differently-named local would be missed, which is why the
# equality assertion rather than this list is the gate.
REQUEST_VARS = ("req", "d")


def _local_imports(path: Path, root: Path) -> set[Path]:
    """Repo-local modules imported by `path`, resolved to files that exist."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            candidates = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for dotted in candidates:
            candidate = root / Path(*dotted.split("."))
            for target in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                if target.is_file():
                    found.add(target.resolve())
    return found


def say_payload_readers(entry: Path | None = None,
                        root: Path | None = None) -> list[Path]:
    """Every file that can see a `/speech/say` payload, derived from the entry node.

    Transitive, because a two-hop delegation is not different in kind from the
    one-hop delegation that broke the previous census. `entry`/`root` are parameters
    so a test can run this over a constructed two-file tree: a derivation that returns
    two files today is not yet evidence that it *follows* a delegation.
    """
    entry = (entry or SAY_ENTRY).resolve()
    root = (root or REPO_ROOT).resolve()
    seen, frontier = {entry}, [entry]
    while frontier:
        for nxt in _local_imports(frontier.pop(), root):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return sorted(seen)


def request_keys_by_reader(entry: Path | None = None,
                           root: Path | None = None) -> dict[Path, set[str]]:
    """Per-file census, so a test can say *which* reader stopped contributing."""
    out: dict[Path, set[str]] = {}
    for path in say_payload_readers(entry, root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keys: set[str] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in REQUEST_VARS
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in REQUEST_VARS and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
        out[path] = keys
    return out


def say_payload_keys(entry: Path | None = None,
                     root: Path | None = None) -> set[str]:
    """The accepted `/speech/say` key set: the union over every reader."""
    return set().union(*request_keys_by_reader(entry, root).values())
