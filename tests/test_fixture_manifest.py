"""T-J3 — the fixture verifier is exercised against a temp dir, not only the repo.

The repo's own fixture set holds one file today, so a test that merely asserted
"the real set verifies" would be nearly vacuous and would stay nearly vacuous right
up to the moment stage S4 lands 30 annotated queries behind it. Every rule is
therefore proved by construction on a temp directory: build a valid set, break it
one way at a time, and require the matching problem — including the mutation that
neither floor can see (an in-place edit of identical length).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import fixture_manifest as fm


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Path, Path]:
    """A valid two-file fixture set plus its manifest."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "queries.jsonl").write_text("".join(f'{{"q": {i}}}\n' for i in range(30)))
    (fixtures / "nested" / "boundaries").mkdir(parents=True)
    (fixtures / "nested" / "boundaries" / "day1.tsv").write_text("12\tsilence\n40\tscene\n")

    manifest = tmp_path / "fixtures.tsv"
    manifest.write_text(
        "# a comment line and a blank line must both be ignored\n"
        "\n"
        + fm._freeze(fixtures) + "\n"
    )
    return fixtures, manifest


def test_a_valid_frozen_set_has_no_problems(frozen) -> None:
    fixtures, manifest = frozen
    assert fm.verify(fixtures, manifest) == []
    assert len(fm.read_manifest(manifest)) == 2  # nested files are included


def test_truncation_is_caught_by_the_size_floor(frozen) -> None:
    fixtures, manifest = frozen
    (fixtures / "queries.jsonl").write_text('{"q": 0}\n')
    problems = fm.verify(fixtures, manifest)
    assert any("below the floor" in p and "queries.jsonl" in p for p in problems)


def test_one_long_line_is_caught_by_the_line_floor(frozen) -> None:
    """Same total size, one line: the byte floor alone would pass this."""
    fixtures, manifest = frozen
    original = (fixtures / "queries.jsonl").read_bytes()
    (fixtures / "queries.jsonl").write_bytes(original.replace(b"\n", b" ") + b"\n")
    problems = fm.verify(fixtures, manifest)
    assert any("lines, below the floor" in p for p in problems)


def test_same_length_edit_is_caught_only_by_the_hash(frozen) -> None:
    """The mutation both floors are blind to, and the reason a hash is in the row."""
    fixtures, manifest = frozen
    target = fixtures / "nested" / "boundaries" / "day1.tsv"
    before = target.read_bytes()
    target.write_bytes(before.replace(b"12\t", b"13\t"))
    assert len(target.read_bytes()) == len(before), "the point of this test is equal length"
    problems = fm.verify(fixtures, manifest)
    assert any("content changed" in p for p in problems)
    assert any("do not re-freeze" in p for p in problems)


def test_a_deleted_fixture_is_caught(frozen) -> None:
    fixtures, manifest = frozen
    (fixtures / "queries.jsonl").unlink()
    assert any("missing from" in p for p in fm.verify(fixtures, manifest))


def test_an_unlisted_file_is_caught(frozen) -> None:
    fixtures, manifest = frozen
    (fixtures / "extra_annotations.jsonl").write_text('{"q": 99}\n')
    problems = fm.verify(fixtures, manifest)
    assert any("not in the manifest" in p and "extra_annotations" in p for p in problems)


def test_an_unlisted_nested_file_is_caught(frozen) -> None:
    """rglob, not glob: a subdirectory is exactly where unannotated data hides."""
    fixtures, manifest = frozen
    (fixtures / "nested" / "boundaries" / "day2.tsv").write_text("1\tx\n")
    assert any("day2.tsv" in p for p in fm.verify(fixtures, manifest))


def test_an_empty_manifest_over_a_populated_dir_is_all_problems(frozen) -> None:
    fixtures, manifest = frozen
    manifest.write_text("# every row deleted\n")
    problems = fm.verify(fixtures, manifest)
    assert len(problems) == 2
    assert all("not in the manifest" in p for p in problems)


def test_a_manifest_over_a_missing_directory_is_a_problem(tmp_path: Path) -> None:
    manifest = tmp_path / "fixtures.tsv"
    manifest.write_text("q.jsonl\t1\t1\t" + "0" * 64 + "\n")
    problems = fm.verify(tmp_path / "does_not_exist", manifest)
    assert len(problems) == 1 and "does not exist" in problems[0]


# ---------------------------------------------- the manifest's own failure modes
@pytest.mark.parametrize(
    "bad_line, expect",
    [
        ("only\ttwo\n", "tab-separated"),
        ("q.jsonl\tnot_a_number\t1\t" + "a" * 64 + "\n", "invalid literal"),
        ("q.jsonl\t1\t1\tdeadbeef\n", "not a sha256"),
        ("q.jsonl\t1\t1\t" + "z" * 64 + "\n", "not a sha256"),
    ],
)
def test_a_malformed_manifest_raises_rather_than_verifying_nothing(
    tmp_path: Path, bad_line: str, expect: str
) -> None:
    """A manifest that fails to parse must not silently become zero rows to check."""
    manifest = tmp_path / "fixtures.tsv"
    manifest.write_text(bad_line)
    with pytest.raises(fm.ManifestError, match=expect):
        fm.read_manifest(manifest)


def test_duplicate_rows_are_rejected(tmp_path: Path) -> None:
    """Two rows for one path let the looser one shadow the stricter one."""
    manifest = tmp_path / "fixtures.tsv"
    digest = "a" * 64
    manifest.write_text(f"q.jsonl\t100\t10\t{digest}\nq.jsonl\t1\t1\t{digest}\n")
    with pytest.raises(fm.ManifestError, match="duplicate row"):
        fm.read_manifest(manifest)


def test_a_missing_manifest_raises_rather_than_passing(tmp_path: Path) -> None:
    with pytest.raises(fm.ManifestError, match="no fixture manifest"):
        fm.read_manifest(tmp_path / "absent.tsv")


def test_freeze_prints_and_never_writes(frozen) -> None:
    fixtures, manifest = frozen
    before = manifest.read_bytes()
    (fixtures / "queries.jsonl").write_text("corrupted\n")
    fm._freeze(fixtures)
    assert manifest.read_bytes() == before, (
        "--freeze must not be able to launder a corrupted fixture into a new baseline"
    )


# -------------------------------------------------------------- the real fixture set
def test_the_repo_fixture_set_is_intact() -> None:
    problems = fm.verify()
    assert problems == [], "\n".join(problems)


def test_the_repo_manifest_is_not_empty() -> None:
    """Guards the vacuous state this suite would otherwise sit in until S4."""
    assert len(fm.read_manifest()) >= 1
