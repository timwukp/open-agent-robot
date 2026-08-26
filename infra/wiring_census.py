#!/usr/bin/env python3
"""Derive every subsystem seam in this tree that only `pytest` reaches.

`infra/graph_gaps.tsv` does this for ROS topics: a published topic nobody subscribes
to, a subscribed topic nobody publishes. This file does the same thing one layer down,
for python call seams, because that is where the larger gap turned out to be — and no
guard in the repo could see it.

**The defect this file exists to make unrepresentable.** `agent/context.py:assemble`
takes `card` and `recall`; `agent/agent_node.py` passes neither. So the retrieval and
profile subsystems run under `pytest` and nowhere else. Nothing was red:
`check_deploy.sh` was green because every entrypoint is launched, `graph_gaps.tsv` was
green because the profile topic already has a row of its own there, and the suite was
green because the tests call `assemble(card=…)` themselves. A promise can be fully
implemented, fully tested, and unreachable, and this repo had no instrument that could
say so.

**Three instruments at three granularities, because one cannot do it.** They are
complementary, and the complementarity is the design rather than redundancy — each was
added because the one before it was measurably blind to a real seam in this tree:

  unreached  (file) no call path from any process `infra/run_ec2.sh` launches reaches
             this file. Answers "the robot cannot run this at all". Catches whole
             modules: `memory/erase.py`, `safety/output_guard.py`, `memory/segment.py`.
  uncalled   (function) a public module-level function in a file that IS reached, which
             no production file calls at all. Catches the case a file-level instrument
             cannot see: `safety/guard.py` is loaded by `agent_node` on every turn for
             `load_constitution`, and `clamp_reply`, `gesture_findings`, `spoken_text`
             and `forget_request` in that same live file have no caller in the robot.
  unpassed   (parameter) a keyword parameter of a module-level function that IS called
             across files, which nothing outside `tests/` ever supplies. Catches the
             narrowest case: `agent/context.py:assemble` runs on every turn, and every
             call into recall inside it sits behind `if recall is not None`, so the
             file is live, the function is live, and the subsystem is dead.

Only the outermost asymmetry is reported: an unpassed parameter of an uncalled function
inside an unreached file gets one row, not three, because two rows for one cause means
one of them goes stale silently.

**A fourth boundary, from the gap between `uncalled` and `unpassed`.** A function called
only from its own file — a helper under that file's own `main()` — is neither. It has a
caller, so it is not `uncalled`; it has no cross-file caller, so its parameters are not
checked. That gap is deliberate: without it, every private-by-convention helper in a
by-hand tool reports its options as seams, which is how the first version of this census
produced 213 rows. A 213-row registry is not a registry, it is a second copy of the tree.

**Why the unit is a seam and not a module.** An earlier version of this census asked
"does a node import this module?" and got the answer wrong in both directions.
`body/gestures.py` looked dormant because `body_node.py` reaches it with a bare
sibling `from gestures import …` — it is launched as a file, so `sys.path[0]` is
`body/` rather than the repo root, and it is the one node in the tree that does not use
the `sys.path.insert` + `from body import gestures` idiom. And `memory/recall.py`
looked wired because `agent/context.py` imports it — for a type annotation and one
module attribute read (`HIT_CAP = _recall.TOP_K`). **Import is not reach.**

Whether a given asymmetry is *acceptable* is a judgement, so it is a reviewable row in
`infra/dormant_seams.tsv` with a stated reason — the same shape as the three manifests
already in `infra/`, and for the same reason: an excuse living inside the guard is
where the next unreviewed exemption gets added.

**Three honest boundaries, each asserted rather than trusted.**

1. Reachability is name-matched and ignores branch conditions, so it **over**-reports
   reach: a file reachable only through a branch nothing takes still counts as
   reached. That is the exact hole `unpassed` covers, and
   `tests/test_wiring_census.py` states it differentially — remove the `unpassed`
   instrument and the `assemble` seam disappears from the census.
2. `unpassed` is restricted to module-level `def`s (`MODULE_LEVEL_ONLY`). With methods
   included, every `__init__(…, clock=)` in the tree reports all eight ROS nodes as
   callers, because `Node.__init__` and `Store.__init__` are one name to a name
   matcher. The restriction loses real coverage, so it is named and pinned: a test
   constructs the collapsing case and fails if the census starts reporting it.
3. Parameters whose name begins with `_` are skipped: a seam that announces itself as a
   hook is not a finding. The tree's two such parameters are the crash-injection hook on
   `memory/erase.py`'s deletion entry point and the ephemeral-root escape on `Store`'s
   constructor — named here by description rather than spelled out, because
   `tests/test_erase.py` asserts that hook is referenced from `tests/` only and would
   read this very sentence as a production call site. `tests/test_wiring_census.py`
   covers the skip on a synthetic tree instead.

Exit codes follow the convention in `infra/redact_scan.py`, `infra/doc_counts.py` and
`infra/entrypoints.py`:

    0   census and manifest agree
    1   findings, one per line on stderr, prefixed FAIL
    3   the census could not be trusted — zero files read, a file that would not parse,
        a missing manifest, or an empty launch inventory. A census that read nothing
        prints the same empty finding list as a clean tree unless it refuses to.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MANIFEST = Path("infra") / "dormant_seams.tsv"
RUNNER = Path("infra") / "run_ec2.sh"
STANDALONE = Path("infra") / "standalone_entrypoints.tsv"

# `tests/` is the one that carries the argument: a call from a test is exactly what
# this census must not count as reach.
SKIP_DIR_PARTS = ("tests", "__pycache__", "venv", "venv-yue", "node_modules",
                  ".git", "site-packages", "docs", "daemon")

# See boundary 2. Named so the restriction is greppable and testable.
MODULE_LEVEL_ONLY = True

ASYMMETRIES = ("unreached", "uncalled", "unpassed")

# The row's `reason` must name something that exists elsewhere in the tree, and for
# four of the five classes a machine checks that naming. A reason only a reader can
# verify is a reason that stops being verified.
CLASSES = {
    "pending_stage":       "the caller belongs to a stage that has not landed. The "
                           "reason names a stage id in STAGE_RE's shape, and the row "
                           "is deleted in the PR that wires it",
    "by_hand":             "reached only from a by-hand tool or harness. If the seam "
                           "has its own row in infra/standalone_entrypoints.tsv the "
                           "reason must name itself; otherwise it is a library of such "
                           "a tool and must name that tool's path",
    # There is deliberately no `separate_process` class. `speech/yue_server.py` is the
    # case that would want one — a whole process nothing calls into — but the runner
    # launches it, so reachability starts *at* it and it is never censused. A class
    # that cannot occur is dead vocabulary, and dead vocabulary is where a wrong row
    # goes to look right.
    "test_only_by_design": "an injection or tuning seam that exists for tests to "
                           "drive. The reason must name a tests/ file that exists",
    "defect":              "not merely unwired but wrong as written — most often a "
                           "second implementation of a rule the dormant function owns. "
                           "The reason must name the stage that fixes it, same as "
                           "pending_stage, so this class cannot become a parking lot",
}

# Both classes that describe future work must name the stage that closes them. The
# difference the class carries is diagnosis, not schedule: `pending_stage` says nothing
# is wrong and nobody connected it, `defect` says there are two implementations of one
# rule and the live one is the weaker.
NEEDS_STAGE = ("pending_stage", "defect")

# The shape of a stage id, and the **only** place in this repo that spells it.
#
# The optional letter suffix is not cosmetic and it is not speculative. `R1.1b` is a
# stage that has already shipped, and until R1.3 no row's reason happened to name a
# suffixed stage — so the vocabulary the project uses for its own stages and the
# vocabulary this gate accepts had already disagreed, invisibly, for two rounds. The
# suffix carries the one thing renumbering would lose: *the same concern, split because
# it grew too large to review in one PR*. Renumbering R1.3's consolidation half to a
# free `R<n>.<m>` would file it next to unrelated work and drop that.
#
# Still strict at both ends: `\b` after the optional letter refuses `R1.3bx`, so this
# admits one suffix letter and not an arbitrary word.
#
# **What this does not check, stated because the old failure message claimed it did.**
# The message used to read "as used in docs/plan.md", and nothing here reads
# `docs/plan.md` — this is a shape test, so a reason may name `R9.9` and pass. Two
# consequences follow and neither is checked today: a stage id that exists in no plan,
# and a row still naming a stage that has already *landed* (the census only notices the
# row when the module becomes reached, which is a different event). Both are membership
# questions against one authority, which is the R3 T-ID gate's family, and plan §6 item
# 2 now carries them. Until then the enforcement is review, and saying so is cheaper
# than a message that describes a check nobody wrote.
STAGE_RE = re.compile(r"\bR[0-9]+\.[0-9]+[a-z]?\b")
REASON_FLOOR = 60          # a shorter reason is a shrug; see graph_gaps.tsv's rows

# The shape of the one line that makes a launched file able to import its own repo, as
# every launched node that needs it already spells it:
#
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#
# Matched by shape and **not evaluated**, and the limit is stated because it is where this
# check stops: a bootstrap that computes the wrong directory has the right shape. What it
# does catch is the absence of one, which is the defect that was on disk — and absence is
# the failure that produces no traceback anybody reads, because the node dies at import
# and `run_ec2.sh status` reports it as `down` exactly like a node nobody started.
#
# The `.parent` count is derived from how deep the launched file sits rather than fixed at
# two, so a node added at `pkg/sub/node.py` needs three and this check knows it. Every
# launched file today is one directory down.
BOOTSTRAP_CALL = "sys.path.insert"
BOOTSTRAP_ROOT = "Path(__file__).resolve()"


class CensusError(Exception):
    """The census cannot be trusted. Exit 3, never 0."""


# --------------------------------------------------------------------------- reading


def production_files(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIR_PARTS for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def parse(path: Path, rel: str) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise CensusError(f"{rel} does not parse ({exc}) — an unparseable file cannot "
                          f"be censused, and its seams must not default to reached")


def launched(root: Path) -> list[str]:
    """Repo-relative python files that `infra/run_ec2.sh`'s node table launches.

    Read from the here-doc's command column rather than from a glob, because
    `gateway/ws_gateway.py` is not named `*_node.py` and a glob would silently drop
    the process that speaks the Article 50(1) disclosure. This is the same table
    `check_deploy.sh` treats as the single source of truth.
    """
    text = (root / RUNNER).read_text(encoding="utf-8")
    found = []
    for line in text.splitlines():
        if "|" not in line or line.lstrip().startswith("#"):
            continue
        cmd = line.split("|", 1)[1]
        for token in cmd.split():
            if token.endswith(".py") and (root / token).exists():
                found.append(token)
    if not found:
        raise CensusError(f"{RUNNER} launches 0 python files — an empty launch "
                          f"inventory makes every file below unreachable and would "
                          f"report the whole tree as dormant")
    return sorted(set(found))


def aliases(root: Path, rel: str, tree: ast.Module) -> dict[str, str]:
    """Local name -> the repo-relative module file it came from.

    Resolves the two import forms this tree actually uses: `from <pkg> import <mod>`
    after a `sys.path` bootstrap, and the bare sibling `from <mod> import <name>` that
    `body/body_node.py` uses because it is launched as a file.
    """
    here = Path(rel).parent
    out: dict[str, str] = {}

    def record(local: str, module: str) -> None:
        if (root / module).exists():
            out[local] = module

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            parts = node.module.split(".")
            for alias in node.names:
                local = alias.asname or alias.name
                # from pkg import mod
                record(local, f"{parts[0]}/{alias.name}.py")
                # from pkg.mod import name   /   from mod import name (sibling)
                if len(parts) >= 2:
                    record(local, f"{parts[0]}/{parts[1]}.py")
                record(local, (here / f"{parts[0]}.py").as_posix())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                local = alias.asname or alias.name
                if len(parts) >= 2:
                    record(local, f"{parts[0]}/{parts[1]}.py")
                record(local, (here / f"{parts[0]}.py").as_posix())
    return out


def calls(tree: ast.Module) -> list[tuple[str, str | None, set[str]]]:
    """(called name, the object it was called on or None, keyword names passed)."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        kw = {k.arg for k in node.keywords if k.arg}
        if isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            out.append((func.attr, owner, kw))
        elif isinstance(func, ast.Name):
            out.append((func.id, None, kw))
    return out


# --------------------------------------------------------------------------- census


def census(root: Path) -> list[tuple[str, str]]:
    """Every seam in `root` that only tests reach, as (seam, asymmetry) pairs, sorted.

    `seam` is a repo-relative path for `unreached` and `path::func(param=)` for
    `unpassed`, so every row names something a reader can go and look at.
    """
    files = production_files(root)
    if not files:
        raise CensusError(f"read 0 python files under {root} — a census that reads "
                          f"nothing is an error, not an empty result")

    trees = {p.relative_to(root).as_posix(): parse(p, p.relative_to(root).as_posix())
             for p in files}
    alias_of = {rel: aliases(root, rel, tree) for rel, tree in trees.items()}
    call_of = {rel: calls(tree) for rel, tree in trees.items()}

    # edges: file -> the module files it calls into
    edges: dict[str, set[str]] = {}
    for rel, found in call_of.items():
        al = alias_of[rel]
        targets = set()
        for name, owner, _kw in found:
            if owner and owner in al:
                targets.add(al[owner])          # mod.fn(...)
            elif name in al:
                targets.add(al[name])           # fn(...) imported from mod
        edges[rel] = targets - {rel}

    reached = set(launched(root))
    frontier = list(reached)
    while frontier:
        for target in edges.get(frontier.pop(), ()):
            if target not in reached:
                reached.add(target)
                frontier.append(target)

    # Every name called anywhere in production, which is what makes `uncalled` mean
    # "the robot never calls it" rather than "this one file does not".
    called_anywhere = {name for found in call_of.values() for name, _o, _k in found}

    seams: list[tuple[str, str]] = []
    for rel in sorted(trees):
        if rel not in reached:
            seams.append((rel, "unreached"))
    for rel in sorted(reached & set(trees)):
        if not MODULE_LEVEL_ONLY:                      # pragma: no cover - see boundary 2
            raise CensusError("method-level unpassed census is unsound under name "
                              "matching; see MODULE_LEVEL_ONLY")
        passed: dict[str, set[str]] = {}
        for other, found in call_of.items():
            if other == rel:
                continue
            for name, _owner, kw in found:
                passed.setdefault(name, set()).update(kw)
        for node in trees[rel].body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            if node.name not in called_anywhere:
                seams.append((f"{rel}::{node.name}()", "uncalled"))
                continue                   # outermost only: no parameter rows below it
            if not any(name == node.name for other in call_of if other != rel
                       for name, _o, _k in call_of[other]):
                continue                   # own-file caller only; see boundary 4
            for arg in node.args.kwonlyargs:
                if arg.arg.startswith("_"):
                    continue               # a declared test hook; see boundary 3
                if arg.arg not in passed.get(node.name, set()):
                    seams.append((f"{rel}::{node.name}({arg.arg}=)", "unpassed"))
    return sorted(seams)


# ------------------------------------------------------------------------- manifest


def read_manifest(root: Path) -> dict[str, tuple[str, str, str]]:
    """seam -> (asymmetry, class, reason)."""
    path = root / MANIFEST
    if not path.exists():
        raise CensusError(f"{MANIFEST} is missing — without it every seam below is "
                          f"unreviewed, which must not read as clean")
    rows: dict[str, tuple[str, str, str]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise CensusError(f"{MANIFEST}:{lineno} has {len(parts)} tab-separated "
                              f"fields, expected 4 (seam/asymmetry/class/reason)")
        seam, asym, cls, reason = (p.strip() for p in parts)
        if seam in rows:
            raise CensusError(f"{MANIFEST}:{lineno} is a second row for {seam!r} — two "
                              f"reasons for one seam means one of them is unread")
        rows[seam] = (asym, cls, reason)
    if not rows:
        raise CensusError(f"{MANIFEST} declares no rows — an empty manifest and a "
                          f"fully wired tree are different states")
    return rows


def standalone_paths(root: Path) -> set[str]:
    path = root / STANDALONE
    if not path.exists():
        raise CensusError(f"{STANDALONE} is missing, so class 'by_hand' cannot be "
                          f"cross-checked and its rows would be unverified prose")
    return {line.split("\t", 1)[0].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


# --------------------------------------------------------------------------- verdict


def repo_packages(root: Path) -> set[str]:
    """Top-level directories a `from <pkg> import ...` in this tree could name."""
    return {p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and p.name not in SKIP_DIR_PARTS and any(p.glob("*.py"))}


def bootstrap_findings(root: Path) -> list[str]:
    """Launched files whose package-qualified imports the launcher cannot resolve.

    This closes an assumption `aliases` above states in its own docstring and never
    checked: "the two import forms this tree actually uses: `from <pkg> import <mod>`
    **after a `sys.path` bootstrap**". The census resolves that form statically, so a
    launched file missing the bootstrap has its seams reported `wired` while the process
    cannot start — a false *reached*, which is the one direction this tool exists to make
    impossible. `memory/memory_node.py` was in exactly that state: `run_ec2.sh` launches
    it as a file, so `sys.path[0]` is `memory/`, there is no `memory` package inside
    `memory/`, and `from memory import profile` raised `ModuleNotFoundError` from every
    working directory. The single writer had never run, and every offline test passed
    because pytest puts the repo root on `sys.path` before importing anything.

    Derived on both sides. The launched set comes from the runner's own table (`launched`,
    the same table `check_deploy.sh` treats as the source of truth) and the package set
    from the directories on disk, so a new node and a new top-level package are both
    picked up without anyone adding a name here.
    """
    packages = repo_packages(root)
    out: list[str] = []
    examined = 0
    for rel in launched(root):
        path = root / rel
        tree = parse(path, rel)
        needed = sorted((n.lineno, n.module) for n in ast.walk(tree)
                        if isinstance(n, ast.ImportFrom) and n.module and not n.level
                        and n.module.split(".")[0] in packages)
        if not needed:
            continue
        examined += 1
        line, module = needed[0]
        # A file at `pkg/node.py` is one directory down, so the root is `.parent.parent`.
        wanted = BOOTSTRAP_ROOT + ".parent" * len(Path(rel).parts)
        boots = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and ast.unparse(n.func) == BOOTSTRAP_CALL
                 and wanted in ast.unparse(n)]
        if not boots:
            out.append(
                f"FAIL {rel} is launched by {RUNNER} and imports {module!r} at line "
                f"{line}, but no {BOOTSTRAP_CALL} of {wanted} precedes it — sys.path[0] "
                f"is {Path(rel).parent}/ when the runner launches a file, so this node "
                f"raises ModuleNotFoundError before rclpy.init() and dies looking exactly "
                f"like a node nobody started. The census resolves this import statically "
                f"and would report the seams behind it as reached")
        elif min(boots) > line:
            out.append(
                f"FAIL {rel} bootstraps sys.path at line {min(boots)} but already needs "
                f"{module!r} at line {line} — an import runs when it is read, so a "
                f"bootstrap below it is a bootstrap that never happens")
    if not examined:
        raise CensusError(
            f"0 of the {len(launched(root))} files {RUNNER} launches use a "
            f"package-qualified import, so this check examined nothing and reported "
            f"clean — which is the same output as a tree where every bootstrap is "
            f"present. Either the launch table stopped naming the nodes or the import "
            f"style changed, and both mean this check needs rewriting rather than "
            f"passing")
    return out


def findings(root: Path) -> list[str]:
    """Every disagreement between the census and the manifest, in both directions —

    plus every launched file whose imports the census resolved and the launcher cannot.
    That last family is not a manifest disagreement, and it is here rather than in a tool
    of its own because it is the same claim under the same exit code: this census says
    which seams the running robot reaches, and a node that cannot import is a node that
    reaches none of them.
    """
    seams = dict(census(root))
    rows = read_manifest(root)
    hands = standalone_paths(root)
    out: list[str] = bootstrap_findings(root)

    for seam, asym in sorted(seams.items()):
        if seam not in rows:
            out.append(f"FAIL {seam} is {asym} — only tests reach it, and no row in "
                       f"{MANIFEST} says why that is acceptable")

    for seam, (asym, cls, reason) in sorted(rows.items()):
        if seam not in seams:
            out.append(f"FAIL {MANIFEST} has a row for {seam} which is no longer "
                       f"{asym} — a stale row silently excuses the next seam that "
                       f"arrives with this name, and it is the row that must be "
                       f"deleted in the PR that wires it")
            continue
        if asym not in ASYMMETRIES:
            out.append(f"FAIL {seam} declares asymmetry {asym!r}, not one of "
                       f"{sorted(ASYMMETRIES)}")
        elif asym != seams[seam]:
            out.append(f"FAIL {seam} is declared {asym} but censuses as {seams[seam]}")
        if cls not in CLASSES:
            out.append(f"FAIL {seam} declares class {cls!r}, not one of "
                       f"{sorted(CLASSES)}")
        if len(reason) < REASON_FLOOR:
            out.append(f"FAIL {seam} has a {len(reason)}-character reason — under "
                       f"{REASON_FLOOR} is a shrug, and the manifest is where the "
                       f"argument goes")

        if cls == "by_hand":
            # Two shapes, and which one applies is derived rather than declared: the
            # seam is itself a by-hand entrypoint, or it is a library of one. Naming
            # *some* path in STANDALONE would accept the second reason on the first
            # kind of row — `arch_render.py` excused by naming `rank_harness.py` reads
            # fine and is a lie. So the tool that reaches it must be the seam itself
            # exactly when the seam is the entrypoint.
            if seam in hands:
                if seam not in reason:
                    out.append(f"FAIL {seam} is class by_hand and has its own row in "
                               f"{STANDALONE}, but its reason does not name itself — a "
                               f"reason naming some other tool would excuse this seam "
                               f"with a claim about a different file")
            elif not any(p in reason for p in hands):
                out.append(f"FAIL {seam} is class by_hand and has no row in "
                           f"{STANDALONE}, so it is a library of a by-hand tool, but "
                           f"its reason names no path that has a row there — nothing "
                           f"ties it to a tool anyone runs")
        elif cls == "test_only_by_design":
            named = [t.rstrip(".,)") for t in reason.split() if t.startswith("tests/")]
            missing = [t for t in named if not (root / t).exists()]
            if not named:
                out.append(f"FAIL {seam} is class test_only_by_design but its reason "
                           f"names no tests/ file, so the seam has no declared driver")
            elif missing:
                out.append(f"FAIL {seam} names {missing}, which does not exist")
        elif cls in NEEDS_STAGE and not STAGE_RE.search(reason):
            out.append(f"FAIL {seam} is class {cls} but its reason names no stage id "
                       f"matching {STAGE_RE.pattern} — a row describing future work "
                       f"with no stage is a row that stays. This is a shape test and "
                       f"nothing here reads docs/plan.md; see STAGE_RE")
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent)
    root = root.resolve()
    try:
        files = production_files(root)
        seams = census(root)
        rows = read_manifest(root)
        bad = findings(root)
    except CensusError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 3
    for line in bad:
        print(line, file=sys.stderr)
    kinds = {a: sum(1 for _s, x in seams if x == a) for a in ASYMMETRIES}
    # The files-read count leads the line because it is the number a floor can be put
    # on. The seam count must not have one: every R1 PR is supposed to make it smaller,
    # so a floor on seams would turn each wiring step red for succeeding.
    print(f"{len(files)} python files read, {len(seams)} seams censused "
          f"({kinds['unreached']} unreached, {kinds['uncalled']} uncalled, "
          f"{kinds['unpassed']} unpassed), {len(rows)} manifest rows, "
          f"{len(bad)} findings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
