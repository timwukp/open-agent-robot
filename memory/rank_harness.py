"""memory/rank_harness.py — the T-F6 ablation runner, and the exit code that refuses.

Pure and offline: no ROS, no boto3, no network, no GPU, $0. It reads a store root and
a query list, runs all three arms of `memory/rank.py` over every query, writes the
evidence JSON, and returns an exit code that says which of four states the run is in.

    python3 memory/rank_harness.py --root ~/oar-data            # -> exit 3 today
    python3 memory/rank_harness.py --root <root> --queries <md> --now 1767225600

**Why four exit codes and not a boolean.**
T-F6 says 「三組數字齊備才允許下結論;缺任一組 → harness 非零退出」. A boolean would
merge the three ways of not concluding, and two of them are the states this project is
actually in:

    0  three arms ran, and a conclusion is supportable. **Unreachable today**, and it
       must stay unreachable until T-F1's labels exist — see below.
    1  the comparison is malformed: an arm missing, `reorder` dropping something, an
       arm losing a hit without reporting it. A defect in this code.
    2  the comparison is well-formed and still cannot conclude anything, because no
       relevance labels exist. **This is the expected result today.**
    3  it could not run at all: no store, no query file, zero queries, or zero hits
       across every query.

Code 3 is the one the guard rules of this repo are about: a scan that read nothing must
not report clean (§6.0 rule 5). Zero queries and zero hits are both counted and both
turn into a refusal, because "three arms agreed perfectly" is exactly what an empty run
looks like from the outside.

**Why the query file being absent is a refusal rather than a default set.**
T-F1's 30 queries have to be frozen *before* the retriever is tuned, and they need one
day of real logs plus one human annotation pass. A harness that fell back to a handful
of queries written today would produce a JSON indistinguishable in shape from the real
one, and the number in it would be about the fixture's author. So the missing file is
the loud, legible record that S6's conclusion is blocked on data collection.

**Why the clock is a flag.**
Decay is a function of elapsed time, so a run without a recorded clock is not
reproducible: the same store gives different orders tomorrow. `--now` defaults to the
wall clock *here in the harness* — `memory/rank.py` never reads a clock (§6.1) — and
the value used is written into the JSON, so a run can be replayed exactly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import rank  # noqa: E402
from memory import recall  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "s6"
QUERIES_DOC = REPO_ROOT / "docs" / "eval-queries.md"

# One query per line, as a markdown list item, with the id T-F1 freezes it under:
#   - `q07` 我對什麼過敏
# The id is part of the format because a regression line has to be able to name which
# query moved, and a positional index renumbers itself the first time a query is added.
QUERY_LINE = re.compile(r"^\s*[-*]\s+`(?P<id>[A-Za-z0-9_-]+)`\s+(?P<text>\S.*?)\s*$")

CANNOT_RUN = 3
MALFORMED = 1
NO_CONCLUSION = 2
CONCLUDED = 0


def parse_queries(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """The `(id, query)` pairs in `text`, and everything wrong with the file.

    A findings list rather than a raise, so a file with one malformed line reports that
    line instead of failing in a way that reads as "the file is missing". Duplicate ids
    are a finding for the reason the id exists at all: two rows under one id make the
    regression line unable to say which query moved.
    """
    pairs: list[tuple[str, str]] = []
    findings: list[str] = []
    seen: dict[str, int] = {}
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#") or line.lstrip().startswith(">"):
            continue
        m = QUERY_LINE.match(line)
        if not m:
            continue
        qid, query = m.group("id"), m.group("text")
        if qid in seen:
            findings.append(f"{QUERIES_DOC.name}:{n}: duplicate query id {qid!r} "
                            f"(first seen on line {seen[qid]})")
            continue
        seen[qid] = n
        pairs.append((qid, query))
    return pairs, findings


def run(root: Path, queries: list[tuple[str, str]], *, now: float) -> dict:
    """Every query through every arm. The returned dict is the evidence JSON verbatim."""
    retriever = recall.Retriever(root)
    reports = {}
    for qid, query in queries:
        result = retriever.query(query)
        ev = rank.evidence(result, root, now=now)
        report = rank.ablation(result, ev)
        # `include_identical=False`: the collision check is aggregated in `verdict()`
        # from `identical_arms`, because its severity depends on how many queries it
        # happened on and a per-query finding cannot know that.
        report["structural_findings"] = rank.ablation_findings(
            report, include_identical=False)
        report["blockers"] = rank.conclusion_blockers(report)
        for name in rank.arms():
            report["arms"][name]["kept_findings"] = rank.kept_findings(
                result, rank.rank(name, result, ev))
        reports[qid] = report
    return {
        "now": float(now),
        "root_hits": sum(len(r["baseline"][p]) for r in reports.values()
                         for p in r["baseline"]),
        "queries": len(reports),
        "arms": list(rank.arms()),
        "policy": {"half_life_s": rank.HALF_LIFE_S,
                   "decay_floor": rank.DECAY_FLOOR,
                   "prune_min": rank.PRUNE_MIN,
                   "policy_findings": rank.policy_findings()},
        "reports": reports,
    }


def verdict(run_data: dict) -> tuple[int, list[str]]:
    """The exit code for a completed run, and the lines explaining it.

    Aggregation matters here and is the reason this is a function rather than a few
    `if`s in `main`. A pair of arms coinciding on *one* query is noise; the same pair
    coinciding on *every* query means the two policies are one policy, which is the
    「A/B 只有 A」 failure wearing three names. So the collision is only structural when
    it is total.
    """
    lines: list[str] = []
    structural = [f"{qid}: {f}" for qid, r in sorted(run_data["reports"].items())
                  for f in r["structural_findings"]]
    for qid, r in sorted(run_data["reports"].items()):
        for name, arm in sorted(r["arms"].items()):
            structural += [f"{qid}/{name}: {f}" for f in arm["kept_findings"]]
    structural += [f"policy: {f}" for f in run_data["policy"]["policy_findings"]]

    total = run_data["queries"]
    collisions: dict[tuple[str, ...], int] = {}
    for r in run_data["reports"].values():
        for group in r["identical_arms"]:
            collisions[tuple(group)] = collisions.get(tuple(group), 0) + 1
    for group, hits in sorted(collisions.items()):
        line = (f"arms {list(group)} produced identical orders and drops on {hits} of "
                f"{total} query/queries")
        if hits == total:
            structural.append(line + " — that is the same policy under two names")
        else:
            lines.append("note     " + line)

    if structural:
        return MALFORMED, lines + [f"finding  {s}" for s in structural]
    blockers = sorted({b for r in run_data["reports"].values() for b in r["blockers"]})
    if blockers:
        return NO_CONCLUSION, lines + [f"blocked  {b}" for b in blockers]
    return CONCLUDED, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path,
                    help="the store root to query (never a path inside the repo)")
    ap.add_argument("--queries", type=Path, default=QUERIES_DOC,
                    help=f"T-F1's frozen query list (default {QUERIES_DOC.name})")
    ap.add_argument("--now", type=float, default=None,
                    help="the clock reading decay is computed against; defaults to the "
                         "wall clock and is recorded in the JSON either way")
    ap.add_argument("--out", type=Path, default=EVIDENCE / "rank-ablation.json")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"CANNOT RUN: store root {args.root} does not exist", file=sys.stderr)
        return CANNOT_RUN
    if not args.queries.is_file():
        print(f"CANNOT RUN: {args.queries} does not exist. T-F1's 30 queries have to "
              f"be frozen before the retriever is tuned, and they need one day of real "
              f"logs plus one human annotation pass — a default query set written "
              f"today would measure its own author", file=sys.stderr)
        return CANNOT_RUN
    queries, findings = parse_queries(args.queries.read_text(encoding="utf-8"))
    for line in findings:
        print(f"finding  {line}", file=sys.stderr)
    if findings:
        return MALFORMED
    if not queries:
        print(f"CANNOT RUN: parsed 0 queries out of {args.queries}; a run over nothing "
              f"must not report three agreeing arms", file=sys.stderr)
        return CANNOT_RUN

    now = time.time() if args.now is None else args.now
    data = run(args.root, queries, now=now)
    if data["root_hits"] == 0:
        print(f"CANNOT RUN: {len(queries)} query/queries returned 0 hits in total. "
              f"Three arms over an empty answer are identical for a reason that has "
              f"nothing to do with the policies", file=sys.stderr)
        return CANNOT_RUN

    code, lines = verdict(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"{data['queries']} queries, {len(data['arms'])} arms, "
          f"{data['root_hits']} hits, exit {code}")
    print(f"wrote {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
