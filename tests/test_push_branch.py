"""The push mechanism itself: .push/push_branch.py.

Pushing is the one operation in this repo with no undo and no local witness, and its
discipline had been living in two throwaway scripts and in memory. Both of those
scripts have now been deleted in favour of one argument-taking pusher, so the
discipline is asserted here instead of remembered:

  * the blob sha is derived three ways (git's own `hash-object`, the `blob <len>\\0`
    framing in the pusher, and the sha the API returns) and all three must agree
  * a 201 is not evidence — the blob is read back and compared byte-for-byte
  * `gh api -f field=@path` posts the literal string "@path" and returns 201, so no
    request body may ever be built that way
  * a zero-file or suspiciously short push is refused before anything is uploaded
  * the mode in the list is checked against the working tree's executable bit and
    against the base tree's mode, because the pusher's own tree check only ever
    compared the mode it was *asked* for

The network is not reachable from the suite, so the API is replaced by a double —
and the double **lies**. An echoing double would make every verification branch
unreachable and every assertion below vacuous: it would return whatever sha it was
given, so a pusher with no sha comparison at all would pass. Each test here hands
`upload()` a specific lie and requires it to abort.
"""
from __future__ import annotations

import ast
import base64
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PUSHER = REPO_ROOT / ".push" / "push_branch.py"


def _load():
    spec = importlib.util.spec_from_file_location("push_branch", PUSHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load()


# --------------------------------------------------------------- the sha, three ways

PAYLOADS = [b"hello", b"", b"\n", b"\x00\x01\x02", "停不玩 stop\n".encode(),
            b"a" * 4096, bytes(range(256))]


@pytest.mark.parametrize("data", PAYLOADS, ids=lambda d: f"{len(d)}B")
def test_the_framing_agrees_with_git_itself(data: bytes) -> None:
    """Two independent derivations, so a bug in the framing cannot agree with itself."""
    assert pb.framed_blob_sha(data) == pb.git_blob_sha(data)


def test_a_frozen_sha_pins_the_framing_without_needing_git() -> None:
    # `printf hello | git hash-object -t blob --stdin`
    assert pb.framed_blob_sha(b"hello") == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"
    assert pb.framed_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_a_plain_sha1_of_the_content_is_not_a_git_blob_sha() -> None:
    """The mutant this frozen value exists to kill: dropping git's object framing.

    Without the `blob <len>\\0` header the digest is still 40 hex chars and still
    stable, so a pusher using it would compare two numbers that always disagree with
    the API — or, worse, agree if the comparison were also dropped.
    """
    import hashlib
    assert hashlib.sha1(b"hello").hexdigest() != pb.framed_blob_sha(b"hello")


# ------------------------------------------------------------------ the file listing


def _listing(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "files.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    rows = pb.parse_files(_listing(tmp_path, "# a comment\n\n100644 a.py\n  \n100755 b/c.sh\n"))
    assert rows == [("100644", "a.py"), ("100755", "b/c.sh")]


def test_a_row_without_a_mode_is_refused_not_guessed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="expected '<mode> <path>'"):
        pb.parse_files(_listing(tmp_path, "a.py\n"))


def test_a_non_numeric_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is not a file mode"):
        pb.parse_files(_listing(tmp_path, "blob a.py\n"))


def test_the_push_list_lists_itself_and_the_pusher(repo_root: Path) -> None:
    """Caught by checking the remote, not the local tree, and it had already shipped.

    The first push of `.push/push_branch.py` landed without `.push/files.txt`, so the
    pushed tree contained a test module (this one) that reads a file the tree does not
    have — a clone would have failed on a missing fixture rather than on a real defect.
    A local check cannot notice that by looking at local files, because locally they
    are all there. What it CAN notice is the invariant behind it: the push list and the
    pusher must both be in the list, or the mechanism is not reviewable in the tree it
    produces.
    """
    rows = {p for _m, p in pb.parse_files(repo_root / ".push" / "files.txt")}
    for required in (".push/files.txt", ".push/push_branch.py"):
        assert required in rows, (
            f"{required} is not in .push/files.txt, so it never reaches the pushed "
            f"tree — and this test module reads it")


def test_the_real_listing_parses_and_every_path_exists(repo_root: Path) -> None:
    rows = pb.parse_files(repo_root / ".push" / "files.txt")
    assert len(rows) > 1
    missing = [p for _m, p in rows if not (repo_root / p).is_file()]
    assert missing == [], f"files.txt names paths that are not in the tree: {missing}"
    empty = [p for _m, p in rows if not (repo_root / p).read_bytes()]
    assert empty == [], f"files.txt names empty files: {empty}"


# ------------------------------------------------------ refusals, before any upload

CLI = ["--base", "main", "--branch", "topic", "--message-file"]


def _args(tmp_path: Path, listing: str, msg: str = "a message\n", extra=()) -> list[str]:
    (tmp_path / "msg.txt").write_text(msg, encoding="utf-8")
    return [*CLI, str(tmp_path / "msg.txt"), "--files", str(_listing(tmp_path, listing)),
            "--root", str(tmp_path), *extra]


def test_a_zero_file_push_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="must not report success"):
        pb.main(_args(tmp_path, "# only a comment\n"))


def test_a_push_below_the_floor_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="floor is 5"):
        pb.main(_args(tmp_path, "100644 a.py\n", extra=("--min-files", "5")))


def test_a_duplicated_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="lists the same path twice"):
        pb.main(_args(tmp_path, "100644 a.py\n100755 a.py\n"))


def test_an_empty_commit_message_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is empty"):
        pb.main(_args(tmp_path, "100644 a.py\n", msg="   \n"))


def test_committing_a_branch_onto_itself_is_refused(tmp_path: Path) -> None:
    (tmp_path / "msg.txt").write_text("m\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="both 'main'"):
        pb.main(["--base", "main", "--branch", "main",
                 "--message-file", str(tmp_path / "msg.txt")])


def test_a_listed_path_that_is_not_in_the_tree_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is not a file"):
        pb.main(_args(tmp_path, "100644 gone.py\n"))


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    with pytest.raises(SystemExit, match="zero-byte blob"):
        pb.main(_args(tmp_path, "100644 a.py\n"))


# ------------------------------------------------------------------- the mode is a claim
#
# The pusher already asserted that every entry in the created tree carries the mode it
# requested. That check cannot fail on a wrong request, and a wrong request is the easy
# mistake: every other line of `.push/files.txt` says 100644, so typing 100644 for
# `infra/run_ec2.sh` is the natural thing to do — and it silently strips the executable
# bit from the node runner, with the blob shas agreeing, the tree entry matching the
# request, and the read-back matching the local bytes. It surfaces as `permission
# denied` in a fresh clone, on the deploy path. Caught while assembling the S3 push.


def test_a_mode_that_disagrees_with_the_working_tree_is_found(tmp_path: Path) -> None:
    """The offline half: this is the one that runs in CI, with no network at all."""
    exe, plain = tmp_path / "run.sh", tmp_path / "mod.py"
    exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    plain.write_text("x = 1\n"); plain.chmod(0o644)
    rows = [("100644", "run.sh"), ("100644", "mod.py")]
    assert pb.exec_bit_mismatches(rows, tmp_path) == [("run.sh", "100644", "100755")]
    # ...and the opposite direction, which would mark a plain file executable
    assert pb.exec_bit_mismatches([("100755", "mod.py")], tmp_path) == [
        ("mod.py", "100755", "100644")]
    assert pb.exec_bit_mismatches([("100755", "run.sh"), ("100644", "mod.py")],
                                  tmp_path) == []


def test_only_the_executable_bit_is_compared_not_the_whole_mode(tmp_path: Path) -> None:
    """A group-writable file is still `100644` to git, so it must not be a finding.

    Comparing full modes would make the guard fail on a difference git cannot even
    record, which is how a guard gets switched off.
    """
    p = tmp_path / "mod.py"
    p.write_text("x = 1\n"); p.chmod(0o664)
    assert pb.exec_bit_mismatches([("100644", "mod.py")], tmp_path) == []


def test_a_listed_path_that_is_absent_is_left_to_the_upload(tmp_path: Path) -> None:
    """Two findings for one cause is how the real one gets scrolled past."""
    assert pb.exec_bit_mismatches([("100644", "gone.py")], tmp_path) == []


def test_the_real_push_list_agrees_with_the_working_tree(repo_root: Path) -> None:
    """The live assertion: this is what went red on `infra/run_ec2.sh` at 100644."""
    rows = pb.parse_files(repo_root / ".push" / "files.txt")
    assert pb.exec_bit_mismatches(rows, repo_root) == []


def test_a_mode_that_regresses_against_the_base_tree_is_found() -> None:
    """The online half. A pure function over a mode map, so it needs no double."""
    base = {"infra/run_ec2.sh": "100755", "README.md": "100644"}
    rows = [("100644", "infra/run_ec2.sh"), ("100644", "README.md")]
    assert pb.mode_regressions(rows, base) == [
        ("infra/run_ec2.sh", "100755", "100644")]


def test_a_new_path_has_no_mode_to_regress_against() -> None:
    """A file absent from the base is new; there is no prior value to compare to, and
    inventing one from the filesystem is what `exec_bit_mismatches` is for."""
    assert pb.mode_regressions([("100755", "memory/profile.py")], {}) == []


def test_a_named_mode_change_is_allowed_and_an_unnamed_one_is_not() -> None:
    base = {"a.sh": "100644", "b.sh": "100644"}
    rows = [("100755", "a.sh"), ("100755", "b.sh")]
    assert pb.mode_regressions(rows, base, allowed=("a.sh",)) == [
        ("b.sh", "100644", "100755")]


def test_the_exec_bit_guard_refuses_the_push_before_any_upload(tmp_path: Path,
                                                              monkeypatch) -> None:
    """Refusal, not just detection: the function returning a list proves nothing about
    `main` acting on it. The double serves nothing, so any API call would fail loudly."""
    exe = tmp_path / "run.sh"
    exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    monkeypatch.setattr(pb, "api", Api())
    with pytest.raises(SystemExit, match="executable bit"):
        pb.main(_args(tmp_path, "100644 run.sh\n"))


# --------------------------------------------------------------- the double that lies


class Api:
    """A stand-in for `gh api` that can be told to lie in one specific way.

    Honest by default so the control below has something to be a control for; every
    other test flips exactly one lie, because a double that lies about everything at
    once cannot say which comparison caught it.
    """

    def __init__(self, *, wrong_sha: bool = False, wrong_content: bool = False) -> None:
        self.wrong_sha = wrong_sha
        self.wrong_content = wrong_content
        self.calls: list[tuple[str, bool]] = []
        self.blobs: dict[str, bytes] = {}

    def __call__(self, path, body=None, method=None, repo=None):
        self.calls.append((path, body is not None))
        if path == "git/blobs":
            data = base64.b64decode(body["content"])
            sha = pb.framed_blob_sha(data)
            if self.wrong_sha:
                sha = sha[:-1] + ("0" if sha[-1] != "0" else "1")
            self.blobs[sha] = data
            return {"sha": sha}
        if path.startswith("git/blobs/"):
            data = self.blobs[path.split("/")[-1]]
            if self.wrong_content:
                data = data + b"tampered"
            return {"content": base64.b64encode(data).decode()}
        raise AssertionError(f"the double was asked for {path!r}, which it does not serve")


def _one_file(tmp_path: Path) -> list[tuple[str, str]]:
    (tmp_path / "a.py").write_text('print("x")\n', encoding="utf-8")
    return [("100644", "a.py")]


def test_control_an_honest_api_uploads_and_verifies(tmp_path: Path, monkeypatch) -> None:
    api = Api()
    monkeypatch.setattr(pb, "api", api)
    items = pb.upload(_one_file(tmp_path), tmp_path, "o/r")
    assert [i["path"] for i in items] == ["a.py"]
    assert items[0]["sha"] == pb.framed_blob_sha(b'print("x")\n')


def test_a_sha_the_api_invented_aborts_the_push(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pb, "api", Api(wrong_sha=True))
    with pytest.raises(SystemExit, match="api sha"):
        pb.upload(_one_file(tmp_path), tmp_path, "o/r")


def test_a_matching_sha_with_different_bytes_aborts_the_push(tmp_path: Path,
                                                            monkeypatch) -> None:
    """The read-back is the point: the sha matched, and the stored bytes still differ."""
    monkeypatch.setattr(pb, "api", Api(wrong_content=True))
    with pytest.raises(SystemExit, match="read-back differs"):
        pb.upload(_one_file(tmp_path), tmp_path, "o/r")


def test_every_upload_reads_its_blob_back(tmp_path: Path, monkeypatch) -> None:
    api = Api()
    monkeypatch.setattr(pb, "api", api)
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    pb.upload([("100644", "a.py"), ("100644", "b.py")],
              _one_file(tmp_path) and tmp_path, "o/r")
    reads = [p for p, _ in api.calls if p.startswith("git/blobs/")]
    assert len(reads) == 2, api.calls


# ------------------------------------------------------------- structural properties

SOURCE = PUSHER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_no_request_body_is_ever_built_as_a_file_reference() -> None:
    """`gh api -f field=@path` posts the literal string "@path" with HTTP 201.

    Checked on the argv literals rather than by grepping the whole file, so the
    paragraph in the docstring that explains the defect cannot satisfy the test.
    """
    literals = [n.value for n in ast.walk(TREE)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.startswith("-")]
    assert "-f" not in literals and "--field" not in literals, literals
    assert "--input" in literals


def test_the_pusher_shells_out_to_gh_and_git_and_nothing_else() -> None:
    programs = set()
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run" and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.List) and arg.elts and isinstance(arg.elts[0], ast.Constant):
                programs.add(arg.elts[0].value)
    assert programs == {"gh", "git"}, programs


def test_the_pusher_never_invokes_git_push() -> None:
    """The standing rule of this repo, asserted on the source rather than trusted."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant):
            argv = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            assert not (argv[:1] == ["git"] and "push" in argv), argv
    assert "hash-object" in SOURCE


def test_upload_derives_the_sha_both_ways_before_trusting_the_api() -> None:
    fn = next(n for n in ast.walk(TREE)
              if isinstance(n, ast.FunctionDef) and n.name == "upload")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"framed_blob_sha", "git_blob_sha"} <= called, called


def test_the_two_superseded_pushers_are_gone(repo_root: Path) -> None:
    """They were replaced, not left beside the new one: a second pusher with stale
    hard-coded branch names is how a push lands on the wrong base."""
    for stale in ("push.py", "push_track_c.py"):
        assert not (repo_root / ".push" / stale).exists(), stale


# --------------------------------------------------------------------------- mutants


@pytest.mark.mutation
def test_mutant_the_sha_comparison_removed_and_the_read_back_does_not_cover_it(
        tmp_path: Path, monkeypatch) -> None:
    """Identity and content are two claims, and neither check implies the other.

    This mutant was written expecting the read-back to catch it, and it does not.
    The read-back fetches `git/blobs/{sha}` using **the sha the server just handed
    us**, so if that sha is not the git object id of our bytes, the read-back happily
    confirms the wrong object and the push proceeds. The tree entry then points at a
    blob no future clone resolves to our content. Only comparing the returned sha
    against a locally derived one closes that, which is why both checks are here and
    why removing either one leaves a live failure mode.
    """
    src = SOURCE.replace(
        'if remote != framed:\n            sys.exit(f"{path}: api sha {remote} != local {framed}")',
        "if False:\n            pass")
    assert src != SOURCE, "mutant anchor no longer present in .push/push_branch.py"
    mut = _compile(src, tmp_path)
    monkeypatch.setattr(mut, "api", Api(wrong_sha=True))
    (tmp_path / "a.py").write_text('print("x")\n', encoding="utf-8")
    items = mut.upload([("100644", "a.py")], tmp_path, "o/r")
    assert items[0]["sha"] != mut.framed_blob_sha(b'print("x")\n'), (
        "without the sha comparison an invented object id is written into the tree, "
        "and the read-back cannot notice because it looks the blob up by that id")


@pytest.mark.mutation
def test_mutant_the_read_back_removed(tmp_path: Path, monkeypatch) -> None:
    src = SOURCE.replace(
        'if back != data:\n            sys.exit(f"{path}: read-back differs '
        '({len(back)} vs {len(data)} bytes)")',
        "if False:\n            pass")
    assert src != SOURCE, "mutant anchor no longer present in .push/push_branch.py"
    mut = _compile(src, tmp_path)
    monkeypatch.setattr(mut, "api", Api(wrong_content=True))
    (tmp_path / "a.py").write_text('print("x")\n', encoding="utf-8")
    items = mut.upload([("100644", "a.py")], tmp_path, "o/r")
    assert items, "without the read-back, tampered bytes are accepted silently"


@pytest.mark.mutation
def test_mutant_the_zero_file_refusal_removed(tmp_path: Path) -> None:
    src = SOURCE.replace('    if not rows:\n', '    if False:\n')
    assert src != SOURCE, "mutant anchor no longer present in .push/push_branch.py"
    mut = _compile(src, tmp_path)
    # With the refusal gone the empty list flows on to the message read and the
    # upload loop, and a push of nothing would report success.
    with pytest.raises(SystemExit) as exc:
        mut.main(_args(tmp_path, "# nothing\n"))
    assert "must not report success" not in str(exc.value)


@pytest.mark.mutation
def test_mutant_the_exec_bit_refusal_removed(tmp_path: Path, monkeypatch) -> None:
    """Without it, a 100644 row for a 0755 file walks straight past every local check.

    The double serves only blobs, so reaching it at all is the evidence: the mutant got
    as far as resolving the base ref while carrying a mode that would strip an
    executable bit, which is exactly how the original defect would have shipped.
    """
    src = SOURCE.replace("    if wrong_bit:\n", "    if False:\n")
    assert src != SOURCE, "mutant anchor no longer present in .push/push_branch.py"
    mut = _compile(src, tmp_path)
    exe = tmp_path / "run.sh"
    exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    monkeypatch.setattr(mut, "api", Api())
    with pytest.raises(Exception) as exc:
        mut.main(_args(tmp_path, "100644 run.sh\n"))
    assert "executable bit" not in str(exc.value), (
        "the mutant still refused, so this test is not measuring the guard")
    assert "git/ref/heads" in str(exc.value), (
        "expected the mutant to reach the base-ref lookup carrying the wrong mode")


def _compile(src: str, tmp_path: Path):
    target = tmp_path / "push_branch_mutant.py"
    target.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"pbm_{tmp_path.name}", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.mutation
def test_control_the_unmutated_pusher_still_rejects_all_four(tmp_path: Path,
                                                             monkeypatch) -> None:
    """If this control fails the mutants above are uninterpretable."""
    (tmp_path / "a.py").write_text('print("x")\n', encoding="utf-8")
    monkeypatch.setattr(pb, "api", Api(wrong_sha=True))
    with pytest.raises(SystemExit, match="api sha"):
        pb.upload([("100644", "a.py")], tmp_path, "o/r")
    monkeypatch.setattr(pb, "api", Api(wrong_content=True))
    with pytest.raises(SystemExit, match="read-back differs"):
        pb.upload([("100644", "a.py")], tmp_path, "o/r")
    with pytest.raises(SystemExit, match="must not report success"):
        pb.main(_args(tmp_path, "# nothing\n"))
    exe = tmp_path / "run.sh"
    exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    with pytest.raises(SystemExit, match="executable bit"):
        pb.main(_args(tmp_path, "100644 run.sh\n"))


def test_git_is_available_so_the_two_way_derivation_is_not_silently_one_way() -> None:
    """`git_blob_sha` shells out; if git vanished it would sys.exit, and the
    agreement test above would be reporting on a tool that never ran."""
    assert subprocess.run(["git", "--version"], capture_output=True).returncode == 0
