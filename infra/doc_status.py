#!/usr/bin/env python3
"""Check every document under `docs/` against `infra/doc_status.tsv` and `docs/INDEX.md`.

Why this exists, from a measurement rather than a principle: on 2026-08-25 `docs/` held
six research rounds and one design critique — 191 KiB together — and **four of the six
carried no supersession marker of any kind**. One of the four was round 4, whose §0 build
order `docs/design-critique-memory-safety-2026-08-22.md` §4 replaces outright and whose
memory pick `docs/research-round5-memory-safety-2026-08-22.md` re-answers in its own
header. A reader who opened round 4 first could not learn either fact from the tree.

The rounds are kept verbatim on purpose — they are dated measurements and the
measurements are still good — so the retraction cannot live inside them. It has to live
in a table, and a table nobody checks is a comment.

The attribution above is the case for the gate rather than an illustration of it. An
earlier draft of `docs/design.md` said round 4 was "overturned in part by round 7";
deriving the citation graph showed round 7 never names round 4 except to *cite* it
approvingly. Prose about which document overturns which is exactly as unverified as any
other prose, so this guard reads that sentence too and requires the documents it names to
be the ones the table names.

What is checked, and each direction matters:

  * every `docs/**/*.md` on disk has a row — **a new document cannot join `docs/`
    silently**, which is the failure this file was written for;
  * every row's file exists — a stale row would quietly cover the next document arriving
    with the same name;
  * `status` is in a closed set, and `relation` terms are in a closed vocabulary whose
    target must exist, have its own row, and itself be `current` or `evidence` (a
    retraction pointing at a superseded document is a dead end with a footnote);
  * `docs/INDEX.md` and the TSV agree on every path, status and relation, in both
    directions, because two readers of one set of facts each stay green while the other
    side moves;
  * every relative link in the INDEX table resolves on disk;
  * the counted sentences in `docs/INDEX.md`, `docs/plan.md`, `docs/design.md` and
    `docs/design.zh-TW.md` match what the walk measured.

The walk is **recursive**. That is not cosmetic: `docs/evidence/` holds two of the
seventeen documents, and a non-recursive glob would have exempted that directory while
reporting clean.

**What this guard does not check, stated in place rather than left to be discovered.** It
cannot check whether a document contains an in-file supersession marker, so
`docs/design.md`'s dated claim that four of six rounds carry none is hand-measured, not
gated. Deciding that a `⚠️` paragraph is or is not a supersession marker is a reading, and
this repo does not put readings behind gates. What is gated instead is the table that
makes the missing markers unnecessary.

Exit codes:  0 = the table, the index and the tree agree, 1 = findings, 3 = could not
check (a document this guard reads is missing, or the walk found no documents at all).
3 rather than 1 because "the checker could not run" and "the tree is wrong" call for
different actions — and a walk that reads nothing must never report clean.

**What the ran flag on that walk actually buys, which is not what it looks like.** An
emptied `docs/` is refused twice here: by the flag, and by the missing-prose-document check
below it, because every document this guard quotes lives under `docs/` itself. So detection
was never the hole. The flag is load-bearing for a walk pointed at the *wrong* directory —
one mistyped `DOCS_DIR` and every file is still on disk while the census reads nothing.
Without the flag that exits 1 with a finding against every row in the table, and the fix it
appears to ask for is deleting the registry. `tests/test_doc_status.py` states both halves,
because a redundancy nobody measured is usually a redundancy that is not there.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = "docs"
TABLE = "infra/doc_status.tsv"
INDEX = "docs/INDEX.md"
PLAN = "docs/plan.md"
DESIGN_EN = "docs/design.md"
DESIGN_ZH = "docs/design.zh-TW.md"

# Documents whose prose this guard quotes. Missing any of them is "cannot check", not
# "clean": a guard that silently drops a check when its subject moves is worse than none.
PROSE_DOCS = (INDEX, PLAN, DESIGN_EN, DESIGN_ZH)

STATUSES = ("current", "evidence", "superseded")
RELATIONS = ("superseded_by", "overturned_in_part_by")
REASON_FLOOR = 60

RESEARCH_PREFIX = "docs/research-"
CRITIQUE_PREFIX = "docs/design-critique-"

# `| [`docs/design.md`](design.md) | current | `-` | why … |` — the INDEX table.
INDEX_ROW = re.compile(
    r"^\|\s*\[`(docs/[^`]+\.md)`\]\(([^)]+)\)\s*\|\s*([A-Za-z_]+)\s*\|\s*([^|]*?)\s*\|",
    re.M)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "十一": 11, "十二": 12,
}


def census(root: Path) -> list[str]:
    """Every markdown document under `docs/`, recursively, as repo-relative posix paths."""
    docs = root / DOCS_DIR
    return sorted(p.relative_to(root).as_posix() for p in docs.rglob("*.md"))


def parse_table(text: str) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """The TSV's data rows, plus findings for lines that are not four tab-separated fields."""
    rows: list[tuple[str, str, str, str]] = []
    findings: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 4:
            findings.append(f"{TABLE}:{lineno}: {len(fields)} tab-separated fields, "
                            "expected path/status/relation/reason")
            continue
        rows.append(tuple(f.strip() for f in fields))  # type: ignore[arg-type]
    return rows, findings


def split_relation(relation: str) -> list[str]:
    """`-` is no relation; anything else is one or more `;`-joined terms."""
    if relation == "-":
        return []
    return [t.strip() for t in relation.split(";") if t.strip()]


def relation_targets(relation: str) -> list[str]:
    """Just the paths a relation names, in order, ignoring which verb named them."""
    targets = []
    for term in split_relation(relation):
        _, _, target = term.partition(":")
        if target:
            targets.append(target)
    return targets


def one_match(text: str, pattern: str, what: str, where: str) -> tuple[tuple[str, ...] | None, str]:
    """The single tuple a sentence yields, or None plus why it could not be read.

    A pattern matching nothing, or matching twice, is a **finding** rather than a silent
    pass: a reworded document would otherwise switch its own checks off one at a time.
    Same rule as `infra/doc_counts.py`, and it is `docs/test-design.md` §0 rule 5.
    """
    found = re.findall(pattern, text)
    if not found:
        return None, f"{where}: cannot check the {what} — the sentence it reads is gone"
    if len(found) > 1:
        return None, (f"{where}: cannot check the {what} — the sentence appears "
                      f"{len(found)} times, so no single value is quoted")
    hit = found[0]
    return (hit if isinstance(hit, tuple) else (hit,)), ""


def word_to_int(token: str) -> int | None:
    """A digit run or a spelled-out number in either language; None if neither."""
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def check_rows(rows: list[tuple[str, str, str, str]], on_disk: set[str]) -> list[str]:
    """The table against itself and against the tree, in both directions."""
    findings: list[str] = []
    by_path: dict[str, tuple[str, str, str, str]] = {}
    for path, status, relation, reason in rows:
        if path in by_path:
            findings.append(f"{TABLE}: {path} has two rows — one of them will go stale "
                            "silently, which is the whole failure this table prevents")
            continue
        by_path[path] = (path, status, relation, reason)

    for path, status, relation, reason in by_path.values():
        if status not in STATUSES:
            findings.append(f"{TABLE}: {path} has status {status!r}, which is outside the "
                            f"closed set {'/'.join(STATUSES)}")
        if len(reason) < REASON_FLOOR:
            findings.append(f"{TABLE}: {path}'s reason is {len(reason)} characters; the "
                            f"floor is {REASON_FLOOR}, because a reason short enough to "
                            "be a shrug is not a reason")
        terms = split_relation(relation)
        if relation != "-" and not terms:
            findings.append(f"{TABLE}: {path}'s relation is neither `-` nor a term")
        for term in terms:
            verb, sep, target = term.partition(":")
            if verb not in RELATIONS or not sep or not target:
                findings.append(
                    f"{TABLE}: {path} names relation {term!r}, which is not "
                    f"<verb>:<path> with verb in {'/'.join(RELATIONS)}")
                continue
            if target == path:
                findings.append(f"{TABLE}: {path} names itself as its own {verb}")
            elif target not in by_path:
                findings.append(f"{TABLE}: {path} is {verb} {target}, which has no row "
                                "here — a retraction pointing outside the table cannot "
                                "be followed")
            elif by_path[target][1] not in ("current", "evidence"):
                findings.append(f"{TABLE}: {path} is {verb} {target}, which is itself "
                                f"{by_path[target][1]} — that is a dead end with a footnote")
            if target not in on_disk:
                findings.append(f"{TABLE}: {path} is {verb} {target}, which is not in the tree")
        verbs = {t.partition(':')[0] for t in terms}
        if status == "superseded" and "superseded_by" not in verbs:
            findings.append(f"{TABLE}: {path} is superseded but names no superseded_by "
                            "target; a replacement nobody wrote down is a deletion")
        if status != "superseded" and "superseded_by" in verbs:
            findings.append(f"{TABLE}: {path} names a superseded_by target while its own "
                            f"status is {status}; wholly replaced and still current cannot "
                            "both be true")

    for path in sorted(on_disk - set(by_path)):
        findings.append(f"{TABLE}: {path} is in the tree with no row — a document cannot "
                        "join docs/ silently")
    for path in sorted(set(by_path) - on_disk):
        findings.append(f"{TABLE}: the row for {path} names a file that does not exist; a "
                        "stale row would cover the next document with that name")
    return findings


def check_index(index_text: str, rows: list[tuple[str, str, str, str]],
                root: Path) -> list[str]:
    """`docs/INDEX.md` and the TSV against each other, plus the index's own links."""
    findings: list[str] = []
    table = {path: (status, tuple(relation_targets(relation)))
             for path, status, relation, _ in rows}
    seen: set[str] = set()
    for match in INDEX_ROW.finditer(index_text):
        path, link, status, relation_cell = match.groups()
        seen.add(path)
        # The index writes relations as backticked terms; strip the ticks to compare.
        cell = relation_cell.replace("`", "").strip()
        targets = tuple(relation_targets(cell))
        if path not in table:
            findings.append(f"{INDEX}: the row for {path} has no row in {TABLE}")
            continue
        want_status, want_targets = table[path]
        if status != want_status:
            findings.append(f"{INDEX}: {path} is {status} here and {want_status} in "
                            f"{TABLE}")
        if targets != want_targets:
            findings.append(
                f"{INDEX}: {path} names {list(targets)} as what overturns it; {TABLE} "
                f"names {list(want_targets)}")
        if not (root / DOCS_DIR / link).exists():
            findings.append(f"{INDEX}: {path}'s link target {link!r} does not resolve; a "
                            "dangling link in the index is the one place it cannot be")
    for path in sorted(set(table) - seen):
        findings.append(f"{INDEX}: {path} has a row in {TABLE} but none here — the index "
                        "a human reads is the one that would go stale unnoticed")
    return findings


def check_prose(docs: dict[str, str], counted: dict[str, int],
                round4_targets: tuple[str, ...]) -> list[str]:
    """Every sentence that quotes a number this walk measured, or a document it named."""
    findings: list[str] = []
    total, top_level, nested = counted["total"], counted["top_level"], counted["nested"]
    research, critique, kib = counted["research"], counted["critique"], counted["kib"]

    # --- the document totals, quoted in the index and in the plan
    for doc, pattern, expected, what in (
        (INDEX, r"\*\*(\d+) documents\*\*", total, "document total"),
        (PLAN, r"\*\*(\d+) files\*\*", total, "document total"),
        (PLAN, r"(\d+) at the top level plus", top_level, "top-level count"),
    ):
        got, why = one_match(docs[doc], pattern, what, doc)
        if got is None:
            findings.append(why)
        elif word_to_int(got[0]) != expected:
            findings.append(f"{doc}: the {what} says {got[0]}, the walk measured {expected}")

    if nested != total - top_level:
        findings.append(f"{DOCS_DIR}/: {nested} nested documents against {total} total and "
                        f"{top_level} top level, which cannot all be true")

    # --- the size claim, which is three facts in one sentence: how many research rounds,
    #     how many critiques, and how big they are together. Each is derived by prefix, so
    #     a seventh round moves the count and the sentence has to be rewritten with it.
    for doc, pattern, what in (
        (DESIGN_EN,
         r"(\w+) research documents and (\w+) design critique inside `docs/` — "
         r"\*\*(\d+) KiB together\*\*", "research/critique size claim"),
        (DESIGN_ZH,
         r"`docs/` 裡(.)份研究文件與(.)份設計審視 —— \*\*合計 (\d+) KiB\*\*",
         "research/critique size claim"),
    ):
        got, why = one_match(docs[doc], pattern, what, doc)
        if got is None:
            findings.append(why)
            continue
        said = [word_to_int(g) for g in got]
        if said != [research, critique, kib]:
            findings.append(
                f"{doc}: the {what} says {list(got)}; the walk measured "
                f"{research} research documents, {critique} critique and {kib} KiB")

    got, why = one_match(docs[INDEX], r"\*\*(\d+) KiB together\*\*", "size claim", INDEX)
    if got is None:
        findings.append(why)
    elif int(got[0]) != kib:
        findings.append(f"{INDEX}: the size claim says {got[0]} KiB, the walk measured {kib}")

    # --- who overturns round 4. This check exists because the prose was wrong: an earlier
    #     draft credited round 7, which never names round 4 except to cite it approvingly.
    for doc, pattern in (
        (DESIGN_EN, r"(?s)one of the four is round 4,(.*?)in its own header\."),
        (DESIGN_ZH, r"(?s)四份裡的一份是第四輪:(.*?)重新回答。"),
    ):
        got, why = one_match(docs[doc], pattern, "round 4 attribution", doc)
        if got is None:
            findings.append(why)
            continue
        named = tuple(dict.fromkeys(re.findall(r"`(docs/[^`]+\.md)`", got[0])))
        if named != round4_targets:
            findings.append(
                f"{doc}: the round 4 attribution names {list(named)}; {TABLE} says round 4 "
                f"is overturned in part by {list(round4_targets)}")
    return findings


def audit(root: Path, docs: dict[str, str], on_disk: list[str],
          sizes: dict[str, int]) -> list[str]:
    """Every check, pure apart from the link probe, so it can be mutated."""
    rows, findings = parse_table(docs[TABLE])
    findings += check_rows(rows, set(on_disk))
    findings += check_index(docs[INDEX], rows, root)

    research = [p for p in on_disk if p.startswith(RESEARCH_PREFIX)]
    critique = [p for p in on_disk if p.startswith(CRITIQUE_PREFIX)]
    counted = {
        "total": len(on_disk),
        "top_level": sum(1 for p in on_disk if p.count("/") == 1),
        "nested": sum(1 for p in on_disk if p.count("/") > 1),
        "research": len(research),
        "critique": len(critique),
        "kib": round(sum(sizes[p] for p in research + critique) / 1024),
    }
    round4 = next((r for r in rows if "round4" in r[0]), None)
    findings += check_prose(docs, counted,
                            tuple(relation_targets(round4[2])) if round4 else ())
    return findings


def main() -> int:
    on_disk = census(REPO_ROOT)
    if not on_disk:
        print(f"error: walked {DOCS_DIR}/ and found 0 documents; a walk that reads "
              "nothing must not report clean", file=sys.stderr)
        return 3

    docs: dict[str, str] = {}
    for name in (TABLE, *PROSE_DOCS):
        path = REPO_ROOT / name
        if not path.exists():
            print(f"error: {name} is missing — cannot check the document statuses",
                  file=sys.stderr)
            return 3
        docs[name] = path.read_text(encoding="utf-8")

    sizes = {p: (REPO_ROOT / p).stat().st_size for p in on_disk}
    findings = audit(REPO_ROOT, docs, on_disk, sizes)
    for line in findings:
        print(f"finding  {line}")
    rows, _ = parse_table(docs[TABLE])
    print(f"{len(on_disk)} documents censused, {len(rows)} rows, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
