#!/usr/bin/env python3
"""Push a file list to GitHub as a branch, through the Git Data API only.

This repo has a standing rule: **never `git push`.** Every commit is assembled from
blobs through the Git Data API so that each object can be verified before anything
becomes reachable. Until now that rule was implemented by two throwaway scripts with
branch names hard-coded at the top, so every push re-derived the verification
discipline from memory — and one of them (`.push/push.py`) ran its work at import
time, which made it invisible to check 8 of infra/check_deploy.sh. This file replaces
both, takes the branch pair as arguments, and is covered by tests/test_push_branch.py.

Each verification step below exists because of a specific past defect:

  * `gh api -f field=@path` posts the LITERAL string "@path" and returns HTTP 201.
    So every request body here is real JSON handed to `--input -`. A 201 is not
    evidence of anything.
  * The blob sha is derived THREE ways and all three must agree: git's own
    `hash-object`, the `blob <len>\\0<bytes>` framing computed here, and the sha the
    API returns. Two independent local derivations mean a bug in the framing cannot
    quietly agree with itself.
  * The blob is then read BACK from the API and compared byte-for-byte, because a sha
    match only proves the server understood the request, not that it stored the bytes.
  * A truncated tree also returns 201, so `truncated` is checked and the tree is
    listed recursively with every path AND mode asserted present.
  * That mode assertion compares against the mode this script was ASKED for, so it
    cannot notice a wrong request. Writing `100644` for a file the base tree has at
    `100755` strips its executable bit with every single check green — caught live
    while assembling the S3 push, on `infra/run_ec2.sh`, i.e. on the deploy path. So
    each row's mode is compared to the base tree's mode before anything is uploaded,
    and a deliberate change has to name its path with `--allow-mode-change`.
  * After the ref moves, every file is fetched from the pushed branch and compared to
    the local bytes — a needed file has existed locally and never reached the remote
    before, and no local check can fail on that.
  * A zero-file push is refused, and so is a push below an explicit `--min-files`
    floor: a short file list looks exactly like a successful small push.

Usage:
  python3 .push/push_branch.py --base <branch> --branch <new> --message-file <path>
                              [--files .push/files.txt] [--min-files N]
                              [--allow-mode-change PATH]... [--dry-run]

`--dry-run` performs every local check, the base-tree mode comparison, and every
per-blob check, then stops before the tree, commit, and ref exist. It reads the base
ref, because a check that only runs on the real push is a check the dry run is lying
about. Blobs it uploads are unreachable objects, so a dry run leaves no branch and no
commit behind.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = "timwukp/open-agent-robot"
ROOT = Path(__file__).resolve().parent.parent


def api(path: str, body: dict | None = None, method: str | None = None,
        repo: str = REPO) -> dict:
    """One `gh api` call. Bodies are always real JSON on stdin, never `-f x=@path`."""
    cmd = ["gh", "api", f"repos/{repo}/{path}"]
    if method:
        cmd += ["--method", method]
    if body is not None:
        cmd += ["--input", "-"]
    out = subprocess.run(cmd, input=json.dumps(body) if body is not None else None,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"api {path} failed rc={out.returncode}\n{out.stderr}\n{out.stdout}")
    return json.loads(out.stdout) if out.stdout.strip() else {}


def ref_exists(branch: str, repo: str = REPO) -> bool:
    return subprocess.run(["gh", "api", f"repos/{repo}/git/ref/heads/{branch}"],
                          capture_output=True).returncode == 0


def framed_blob_sha(data: bytes) -> str:
    """git's object id for a blob, derived here rather than shelled out."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    """git's own answer, as a second independent derivation. Needs no repository."""
    out = subprocess.run(["git", "hash-object", "-t", "blob", "--stdin"],
                         input=data, capture_output=True)
    if out.returncode != 0:
        sys.exit(f"git hash-object failed: {out.stderr.decode()}")
    return out.stdout.decode().strip()


def parse_files(listing: Path) -> list[tuple[str, str]]:
    """Rows of `<mode> <path>`, comments and blank lines skipped."""
    rows: list[tuple[str, str]] = []
    for lineno, line in enumerate(listing.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            sys.exit(f"{listing}:{lineno}: expected '<mode> <path>', got {line!r}")
        mode, path = parts[0], parts[1].strip()
        if not mode.isdigit():
            sys.exit(f"{listing}:{lineno}: {mode!r} is not a file mode")
        rows.append((mode, path))
    return rows


def upload(rows: list[tuple[str, str]], root: Path, repo: str) -> list[dict]:
    items = []
    for mode, path in rows:
        local_path = root / path
        if not local_path.is_file():
            sys.exit(f"{path} is not a file in {root}")
        data = local_path.read_bytes()
        if not data:
            sys.exit(f"{path} is empty — refusing to push a zero-byte blob")
        framed, from_git = framed_blob_sha(data), git_blob_sha(data)
        if framed != from_git:
            sys.exit(f"{path}: framed sha {framed} != git hash-object {from_git}")
        remote = api("git/blobs", {"content": base64.b64encode(data).decode(),
                                   "encoding": "base64"}, repo=repo)["sha"]
        if remote != framed:
            sys.exit(f"{path}: api sha {remote} != local {framed}")
        back = base64.b64decode(api(f"git/blobs/{remote}", repo=repo)["content"])
        if back != data:
            sys.exit(f"{path}: read-back differs ({len(back)} vs {len(data)} bytes)")
        print(f"  ok {framed[:12]} {len(data):>7} B  {path}")
        items.append({"path": path, "mode": mode, "type": "blob", "sha": remote})
    return items


def exec_bit_mismatches(rows: list[tuple[str, str]],
                        root: Path) -> list[tuple[str, str, str]]:
    """(path, mode requested, mode on disk) where the executable bit disagrees.

    The offline half of the mode guard, and the half that runs in CI. `100644` for a
    file that is `0755` on disk is the typo that strips a runner's executable bit, and
    it is detectable without asking GitHub anything: the local file is right there.

    Only the executable bit is compared, not the whole mode. Group and other bits vary
    with the operator's umask and git does not record them, so comparing them would make
    the guard fail on a difference git cannot express. `chmod +x`, by contrast, is never
    something a umask does by accident.
    """
    out = []
    for mode, path in rows:
        p = root / path
        if not p.is_file():
            continue                      # `upload` refuses this, with a better message
        on_disk = "100755" if p.stat().st_mode & 0o111 else "100644"
        if (mode == "100755") != (on_disk == "100755"):
            out.append((path, mode, on_disk))
    return out


def tree_modes(tree_sha: str, repo: str = REPO) -> dict[str, str]:
    """path -> mode for every blob in `tree_sha`, recursively.

    Refuses a truncated listing for the same reason the post-commit check does: a
    truncated tree is a *shorter* answer to "what modes does the base have", and a
    missing path reads as "this path is new", which is precisely the case the caller
    below treats as having nothing to compare against.
    """
    listed = api(f"git/trees/{tree_sha}?recursive=1", repo=repo)
    if listed.get("truncated"):
        sys.exit(f"base tree {tree_sha[:12]} listing truncated — cannot tell whether "
                 f"this push changes a file mode")
    return {e["path"]: e["mode"] for e in listed["tree"] if e["type"] == "blob"}


def mode_regressions(rows: list[tuple[str, str]], base_modes: dict[str, str],
                     allowed: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """(path, mode on base, mode requested) for each silent mode change.

    Why this exists: the post-commit check below asserts that each entry in the created
    tree has the mode **this script asked for**, which cannot notice that the mode asked
    for is wrong. Writing `100644 infra/run_ec2.sh` in the file list — the natural thing
    to type, since every other line says 100644 — strips the executable bit from the
    node runner, and every check in this script passes: the blob shas agree, the tree
    entry matches the request, the read-back matches the local bytes. The failure
    surfaces later as `permission denied` in a fresh clone, i.e. on the deploy path,
    which is where this repo has been bitten before.

    A path absent from `base_modes` is new and has nothing to regress against. The mode
    of a *new* file is still unchecked by this function, deliberately: there is no prior
    value to compare to, and inventing one from the local filesystem would make the
    guard depend on the operator's umask.
    """
    return [(path, base_modes[path], mode) for mode, path in rows
            if path in base_modes and base_modes[path] != mode and path not in allowed]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="branch this commit's parent is on")
    ap.add_argument("--branch", required=True, help="branch to create or fast-forward")
    ap.add_argument("--message-file", required=True, type=Path)
    ap.add_argument("--files", type=Path, default=ROOT / ".push" / "files.txt")
    ap.add_argument("--min-files", type=int, default=1,
                    help="refuse a push with fewer files than this; a short list looks "
                         "exactly like a small successful push")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--allow-mode-change", action="append", default=[], metavar="PATH",
                    help="permit this path's mode to differ from the base tree's; "
                         "naming the path is the point, so that a deliberate chmod is a "
                         "line in the command a reviewer sees")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.base == args.branch:
        sys.exit(f"--base and --branch are both {args.base!r}: that would commit a "
                 f"branch onto itself and lose the review boundary")

    rows = parse_files(args.files)
    if not rows:
        sys.exit(f"{args.files} listed no files — a zero-file push must not report success")
    if len(rows) < args.min_files:
        sys.exit(f"{len(rows)} files listed, floor is {args.min_files} — refusing a "
                 f"suspiciously short push")
    paths = [p for _m, p in rows]
    if len(set(paths)) != len(paths):
        dupes = sorted({p for p in paths if paths.count(p) > 1})
        sys.exit(f"{args.files} lists the same path twice: {dupes}")

    message = args.message_file.read_text(encoding="utf-8")
    if not message.strip():
        sys.exit(f"{args.message_file} is empty")

    wrong_bit = exec_bit_mismatches(rows, args.root.resolve())
    if wrong_bit:
        detail = "; ".join(f"{p}: list says {want}, disk says {have}"
                           for p, want, have in wrong_bit)
        sys.exit(f"{args.files} disagrees with the working tree on the executable bit: "
                 f"{detail}. This is the check that needs no network — fix the list, or "
                 f"chmod the file if the list is what you meant")

    print(f"{len(rows)} files -> {args.repo} {args.base} .. {args.branch}"
          f"{' (DRY RUN)' if args.dry_run else ''}")

    # Resolved before the upload, not after, so that a mode regression costs nothing and
    # a dry run exercises the same refusal as a real push. These are reads: a dry run
    # still creates no tree, no commit and no ref.
    base = api(f"git/ref/heads/{args.base}", repo=args.repo)["object"]["sha"]
    base_tree = api(f"git/commits/{base}", repo=args.repo)["tree"]["sha"]
    changed = mode_regressions(rows, tree_modes(base_tree, args.repo),
                               tuple(args.allow_mode_change))
    if changed:
        detail = "; ".join(f"{p}: {was} on {args.base} -> {now} requested"
                           for p, was, now in changed)
        sys.exit(f"{args.files} changes {len(changed)} file mode(s) without saying so: "
                 f"{detail}. Every other check in this script would pass — it verifies "
                 f"the mode it was ASKED for. Fix the list, or pass "
                 f"--allow-mode-change for each path on purpose")

    tree_items = upload(rows, args.root.resolve(), args.repo)

    if args.dry_run:
        print(f"dry run: {len(tree_items)} blobs verified; no tree, commit, or ref created")
        return 0

    tree = api("git/trees", {"base_tree": base_tree, "tree": tree_items}, repo=args.repo)
    if tree.get("truncated"):
        sys.exit("tree came back TRUNCATED — refusing to commit it")

    listed = api(f"git/trees/{tree['sha']}?recursive=1", repo=args.repo)
    if listed.get("truncated"):
        sys.exit("recursive tree listing truncated — cannot verify what was committed")
    present = {e["path"]: e for e in listed["tree"]}
    for mode, path in rows:
        entry = present.get(path)
        if entry is None:
            sys.exit(f"{path} is NOT in the created tree")
        if entry["mode"] != mode:
            sys.exit(f"{path}: tree mode {entry['mode']} != requested {mode}")
    print(f"tree {tree['sha'][:12]}: all {len(rows)} paths present "
          f"({len(present)} entries total)")

    commit = api("git/commits", {"message": message, "tree": tree["sha"],
                                 "parents": [base]}, repo=args.repo)["sha"]
    if ref_exists(args.branch, args.repo):
        api(f"git/refs/heads/{args.branch}", {"sha": commit, "force": True},
            method="PATCH", repo=args.repo)
    else:
        api("git/refs", {"ref": f"refs/heads/{args.branch}", "sha": commit},
            repo=args.repo)
    head = api(f"git/ref/heads/{args.branch}", repo=args.repo)["object"]["sha"]
    if head != commit:
        sys.exit(f"ref is {head}, expected {commit}")
    print(f"branch {args.branch} -> {commit}")

    for _mode, path in rows:
        got = api(f"contents/{path}?ref={args.branch}", repo=args.repo)
        if base64.b64decode(got["content"]) != (args.root / path).read_bytes():
            sys.exit(f"{path}: the pushed tree differs from the local tree")
    print(f"verified {len(rows)} files in the pushed tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
