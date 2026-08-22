"""Frozen-fixture integrity (test item T-J3).

Replay tests (30 eval queries, hand-annotated episode boundaries, the red-team set)
are only as trustworthy as the files they read. Three ways a fixture silently stops
being the thing it was annotated as:

  1. it gets truncated — a size floor catches that;
  2. it gets edited in place at the same length — only a content hash catches that,
     which is the whole reason the hash is here and not just a byte count;
  3. it gets *added* without being annotated, and a replay threshold quietly starts
     grading against unlabelled data — the unlisted-file rule catches that.

Manifest lives at tests/fixtures.tsv, i.e. OUTSIDE tests/fixtures/, so the
unlisted-file rule needs no exemption for the manifest itself. Regenerate rows with
`python tests/fixture_manifest.py --freeze`, which prints and never writes: a
verifier that can rewrite its own expectations turns a corrupted fixture into a
new baseline.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
MANIFEST_PATH = TESTS_DIR / "fixtures.tsv"
COLUMNS = ("path", "min_bytes", "min_lines", "sha256")


class ManifestError(Exception):
    """The manifest itself is unreadable — distinct from a fixture being wrong."""


@dataclass(frozen=True)
class Row:
    path: str
    min_bytes: int
    min_lines: int
    sha256: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(manifest_path: Path = MANIFEST_PATH) -> list[Row]:
    if not manifest_path.is_file():
        raise ManifestError(f"no fixture manifest at {manifest_path}")
    rows: list[Row] = []
    for lineno, raw in enumerate(manifest_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != len(COLUMNS):
            raise ManifestError(
                f"{manifest_path}:{lineno}: expected {len(COLUMNS)} tab-separated "
                f"columns {COLUMNS}, got {len(parts)}: {raw!r}"
            )
        path, min_bytes, min_lines, digest = (p.strip() for p in parts)
        try:
            rows.append(Row(path, int(min_bytes), int(min_lines), digest))
        except ValueError as exc:
            raise ManifestError(f"{manifest_path}:{lineno}: {exc}") from exc
        if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise ManifestError(f"{manifest_path}:{lineno}: not a sha256 hex digest: {digest!r}")
    seen: set[str] = set()
    for row in rows:
        if row.path in seen:
            raise ManifestError(f"{manifest_path}: duplicate row for {row.path!r}")
        seen.add(row.path)
    return rows


def verify(fixtures_dir: Path = FIXTURES_DIR,
           manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Return a list of problems; empty means the frozen set is intact."""
    problems: list[str] = []
    rows = read_manifest(manifest_path)

    if not fixtures_dir.is_dir():
        if rows:
            problems.append(f"manifest lists {len(rows)} fixtures but {fixtures_dir} does not exist")
        return problems

    for row in rows:
        target = fixtures_dir / row.path
        if not target.is_file():
            problems.append(f"{row.path}: listed in the manifest but missing from {fixtures_dir}")
            continue
        size = target.stat().st_size
        if size < row.min_bytes:
            problems.append(f"{row.path}: {size} bytes, below the floor of {row.min_bytes} "
                            f"(truncated fixtures make a replay test pass on almost no data)")
        lines = len(target.read_bytes().splitlines())
        if lines < row.min_lines:
            problems.append(f"{row.path}: {lines} lines, below the floor of {row.min_lines}")
        actual = sha256_of(target)
        if actual != row.sha256:
            problems.append(f"{row.path}: content changed (sha256 {actual[:12]}… != "
                            f"frozen {row.sha256[:12]}…) — re-annotate or revert, do not re-freeze")

    listed = {row.path for row in rows}
    present = {str(p.relative_to(fixtures_dir)) for p in fixtures_dir.rglob("*") if p.is_file()}
    for extra in sorted(present - listed):
        problems.append(f"{extra}: present in {fixtures_dir.name}/ but not in the manifest — "
                        f"unannotated data must not reach a replay threshold")
    return problems


def _freeze(fixtures_dir: Path = FIXTURES_DIR) -> str:
    out = []
    for p in sorted(fixtures_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(fixtures_dir))
            body = p.read_bytes()
            out.append(f"{rel}\t{len(body)}\t{len(body.splitlines())}\t{sha256_of(p)}")
    return "\n".join(out)


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        print("# paste into tests/fixtures.tsv after reviewing the diff:")
        print(_freeze())
        raise SystemExit(0)
    found = verify()
    for problem in found:
        print(f"FAIL  {problem}", file=sys.stderr)
    print(f"{len(read_manifest())} fixtures checked, {len(found)} problems")
    raise SystemExit(1 if found else 0)
