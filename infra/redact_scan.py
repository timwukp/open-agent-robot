#!/usr/bin/env python3
"""Redaction scan over an explicit file list, run before every push.

    python3 infra/redact_scan.py .push/files.txt

The list format is the one `.push/push.py` already reads: `<mode> <path>` per
line, so the scan reads *exactly* the bytes that are about to become blobs. That
is deliberate and it is the whole design: a scan over `git ls-files`, or over the
working tree, or over `git diff --name-only`, answers a different question than
"what am I publishing right now".

Three exit codes, same convention as `infra/check_deploy.sh`:

    0  clean
    1  findings printed, one per line
    3  the scan could not run

Exit 3 is not decoration. Every unexpected failure is mapped onto it because a
traceback also exits 1 — so without it, "the scanner crashed" and "the scanner
found a secret" are the same observation. Measured while building this: the
positive control below reported a kill for all seven pattern classes, and the
no-mutant control reported a finding, while the scanner was in fact crashing on a
file-list path it could not open. Every signal looked right and none of them was.

An empty list is exit 3 too: a scan that reads zero files must not report clean.

This module describes its patterns rather than embedding realistic examples,
because a repo-wide scan reads its own source. The realistic strings live only in
`tests/test_redact_scan.py`, which builds them at runtime and writes them under
`tmp_path` — nothing matchable is ever committed.
"""
from __future__ import annotations

import pathlib
import re
import sys

CLEAN, FINDINGS, CANNOT_RUN = 0, 1, 3

# Loopback and the documentation resolver are allowed: they appear all over the
# runbook and none of them identifies this account. Everything else numeric with
# four dotted octets is treated as an address until proven otherwise.
_ALLOWED_IPS = ("127.0.0.1", "0.0.0.0", "1.1.1.1", "255.255.255.255")

HOME_DIR_NAME = pathlib.Path.home().name


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    """The pattern list, built at call time so the home-directory name is never a
    literal in this file."""
    allowed = "|".join(re.escape(ip) for ip in _ALLOWED_IPS)
    return [
        ("instance id", re.compile(r"\bi-[0-9a-f]{8,17}\b")),
        ("12-digit acct", re.compile(r"(?<![\d.])\d{12}(?![\d.])")),
        ("ARN", re.compile(r"\barn:aws[a-z-]*:")),
        # One-or-more, not zero-or-more: with `*` a bare mention of the
        # extension matches, and the first thing that flagged was this file
        # and its own test. A key *path* is account-specific; the four
        # characters of a file extension are not.
        ("pem path", re.compile(r"[\w./~-]+\.pem\b")),
        ("EC2 DNS", re.compile(
            r"\bec2-[\d-]+\.[a-z0-9-]+\.compute\.amazonaws\.com\b")),
        ("public IPv4", re.compile(
            rf"(?<![\d.])(?!{allowed})(?:\d{{1,3}}\.){{3}}\d{{1,3}}(?![\d.])")),
        # The operator's own home path first, then any absolute home path: the
        # second is wider and catches a path belonging to somebody else's machine
        # that a copied command line dragged in.
        ("own home path", re.compile(
            r"/(?:Users|home)/" + re.escape(HOME_DIR_NAME) + r"\b")),
        ("abs home path", re.compile(r"/(?:Users|home)/[A-Za-z][\w.-]*/")),
    ]


def read_list(list_path: str) -> list[str]:
    """Paths from a `<mode> <path>` file list. Raises OSError if unreadable.

    Blank lines and `#` comments are skipped, because `.push/push_branch.py`
    skips them and the two readers must agree about what the manifest *is*. They
    did not: this scanner used to split `# T-C5 …` into a mode and a path and then
    report `CANNOT RUN: listed path is not a file`, so a manifest that pushed
    cleanly could not be scanned at all. That failure was loud rather than a false
    clean, which is why it survived — but a format only one of its two readers
    implements is not a format. `tests/test_redact_scan.py` now asserts the two
    parsers return the same paths for the same text, so the next divergence is red
    whichever side moves.
    """
    out = []
    with open(list_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"not a '<mode> <path>' line: {line!r}")
            out.append(parts[1].strip())
    return out


def scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    """(path, line number, pattern name, matched text) for each hit.

    First match per line per pattern — reporting every overlap on one line turns a
    single leaked ARN into a wall of output and buries the second finding.
    """
    hits = []
    pats = patterns()
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in pats:
            m = rx.search(line)
            if m:
                hits.append((path, lineno, name, m.group(0)))
    return hits


def main(argv: list[str]) -> int:
    list_path = argv[1] if len(argv) > 1 else ".push/files.txt"
    try:
        files = read_list(list_path)
    except (OSError, ValueError) as exc:
        print(f"CANNOT RUN: {exc}")
        return CANNOT_RUN
    if not files:
        print(f"CANNOT RUN: {list_path} lists no files")
        return CANNOT_RUN

    findings: list[tuple[str, int, str, str]] = []
    read = 0
    try:
        for f in files:
            p = pathlib.Path(f)
            if not p.is_file():
                print(f"CANNOT RUN: listed path is not a file: {f}")
                return CANNOT_RUN
            findings += scan_text(f, p.read_text(errors="replace"))
            read += 1
    except Exception as exc:                                   # noqa: BLE001
        print(f"CANNOT RUN: {type(exc).__name__}: {exc}")
        return CANNOT_RUN

    if read != len(files):
        print(f"CANNOT RUN: read {read} of {len(files)} listed files")
        return CANNOT_RUN

    for path, lineno, name, hit in findings:
        print(f"FINDING {path}:{lineno}  {name}  {hit!r}")
    print(f"scanned {read} files, {len(findings)} findings")
    return FINDINGS if findings else CLEAN


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
