"""Tests for infra/redact_scan.py — the guard that authorises a push.

Shape of this file: every pattern class gets a **named mutant** (a probe file
carrying a realistic secret of that class, which the scan must find), plus a
**no-mutant control** (a probe carrying nothing, which must come back clean), plus
two **cannot-run controls** (exit 3). Without the no-mutant control, "8 kills" is
consistent with a scanner that flags every file unconditionally; without the exit-3
controls, "clean" is consistent with a scanner that never ran.

The realistic strings are built here, at runtime, and written under `tmp_path`.
They are never committed — this file's own source has to survive the scan it tests.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCANNER = REPO_ROOT / "infra" / "redact_scan.py"

CLEAN, FINDINGS, CANNOT_RUN = 0, 1, 3


def run(list_path: pathlib.Path | str) -> subprocess.CompletedProcess[str]:
    """Run the scanner as a subprocess, i.e. through its real exit codes.

    Importing `main()` and reading the return value would test the function; the
    thing that authorises a push is the process's exit status, and that is what a
    caller in a shell pipeline sees.
    """
    return subprocess.run([sys.executable, str(SCANNER), str(list_path)],
                          capture_output=True, text=True, cwd=REPO_ROOT)


def write_probe(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    probe = tmp_path / "probe.txt"
    probe.write_text(body + "\n")
    listing = tmp_path / "files.txt"
    listing.write_text(f"100644 {probe}\n")
    return listing


# Built at runtime so no realistic secret is a literal in a committed file. Each
# value is assembled from pieces that are individually unmatchable.
def _secrets() -> dict[str, str]:
    octets = ".".join(["13", "250", "1", "2"])
    twelve = "".join(str(d) for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2])
    home = pathlib.Path.home().name
    return {
        "instance id": "OAR_INSTANCE_ID=" + "i-0" + "abcdef1234567890",
        "12-digit acct": f"the account is {twelve} exactly",
        "ARN": "arn" + ":aws:bedrock:ap-southeast-1:acct:foundation-model/x",
        "pem path": "ssh -i ~/keys/oar-robot" + ".pem" + " robot",
        "EC2 DNS": "ec2-" + octets.replace(".", "-")
                   + ".ap-southeast-1.compute" + ".amazonaws.com",
        "public IPv4": f"HostName {octets}",
        "own home path": "/Users/" + home + "/Downloads/open-agent-robot",
        "abs home path": "/home/" + "someone-else" + "/checkout/file.txt",
    }


@pytest.mark.mutation
@pytest.mark.parametrize("name", sorted(_secrets()))
def test_each_pattern_class_is_caught(name: str, tmp_path: pathlib.Path) -> None:
    """Named mutant per class: a probe carrying one real secret of that class.

    Asserting the *class name* appears in the output, not merely that the exit code
    is 1, is what makes these eight cases eight assertions instead of one repeated
    eight times — a single over-broad pattern would satisfy `rc == 1` everywhere
    while reporting the wrong reason.
    """
    listing = write_probe(tmp_path, _secrets()[name])
    r = run(listing)
    assert r.returncode == FINDINGS, f"rc={r.returncode}\n{r.stdout}{r.stderr}"
    assert name in r.stdout, f"caught something, but not as {name!r}:\n{r.stdout}"


def test_no_mutant_control_is_clean(tmp_path: pathlib.Path) -> None:
    """Without this, a scanner that flags everything passes all eight above."""
    listing = write_probe(
        tmp_path, "nothing account-specific here, just prose and a path docs/x.md")
    r = run(listing)
    assert r.returncode == CLEAN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"
    assert "0 findings" in r.stdout


def test_the_operators_own_home_path_is_a_distinct_class(
        tmp_path: pathlib.Path) -> None:
    """`own home path` must not be reported as the wider `abs home path`.

    The two overlap by construction and the narrow one is listed first for exactly
    this reason. The distinction earns its keep in a diff review: "your home
    directory leaked" and "somebody else's checkout path leaked" get fixed
    differently.
    """
    home = pathlib.Path.home().name
    listing = write_probe(tmp_path, "/Users/" + home + "/Downloads/x")
    r = run(listing)
    assert r.returncode == FINDINGS
    assert "own home path" in r.stdout


def test_a_bare_extension_mention_is_not_a_finding(
        tmp_path: pathlib.Path) -> None:
    """A bare extension mention is not account-specific; a key *path* is.

    Regression test for two live findings, one of which was caused by the fix for
    the other. First the pattern allowed an empty prefix, so it flagged the
    scanner and this file for merely naming the extension. Then this docstring
    quoted a plausible key path as an illustration — and that quotation was itself
    a finding, because the scan cannot tell an example from a leak. Both times the
    guard was in the list of things being pushed, which is the normal case, not an
    edge case: describe the pattern, never reproduce it. The matchable strings in
    this file are all assembled at runtime in `_secrets()`.
    """
    listing = write_probe(
        tmp_path, "the key path is written into a .pem file by provision.sh")
    r = run(listing)
    assert r.returncode == CLEAN, r.stdout


@pytest.mark.parametrize("ip", ["127.0.0.1", "0.0.0.0"])
def test_loopback_is_not_a_finding(ip: str, tmp_path: pathlib.Path) -> None:
    """The runbook is full of `127.0.0.1:8080`. If those were findings the scan
    would be muted within a day, and a muted guard is worse than none."""
    listing = write_probe(tmp_path, f"curl -s http://{ip}:8080/stream")
    r = run(listing)
    assert r.returncode == CLEAN, r.stdout


def test_a_missing_file_list_cannot_run(tmp_path: pathlib.Path) -> None:
    """Exit 3, not 1. A traceback also exits 1, so without this mapping a crashed
    scanner is indistinguishable from a scanner that found a secret — which is
    precisely how this scanner was briefly, invisibly broken."""
    r = run(tmp_path / "no-such-list.txt")
    assert r.returncode == CANNOT_RUN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"
    assert "CANNOT RUN" in r.stdout


def test_an_empty_file_list_cannot_run(tmp_path: pathlib.Path) -> None:
    """A scan that reads zero files must not report clean."""
    listing = tmp_path / "files.txt"
    listing.write_text("\n\n")
    r = run(listing)
    assert r.returncode == CANNOT_RUN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"


def test_a_listed_path_that_does_not_exist_cannot_run(
        tmp_path: pathlib.Path) -> None:
    """The failure this catches is the one that has actually happened here: a file
    that exists locally but was never pushed, or a path typo in the list. Either
    way the guard has not seen the bytes it claims to have cleared."""
    listing = tmp_path / "files.txt"
    listing.write_text(f"100644 {tmp_path / 'ghost.py'}\n")
    r = run(listing)
    assert r.returncode == CANNOT_RUN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"


def test_a_malformed_list_line_cannot_run(tmp_path: pathlib.Path) -> None:
    """`<mode> <path>` or nothing. A bare path silently becomes a mode with no
    path, and the scan then reads a file list of length zero and reports clean."""
    listing = tmp_path / "files.txt"
    listing.write_text("README.md\n")
    r = run(listing)
    assert r.returncode == CANNOT_RUN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"


def test_the_real_push_list_is_clean_if_one_exists() -> None:
    """The live check, when there is a push staged. Skipped rather than failed
    when `.push/files.txt` is absent: this file is a permanent part of the suite
    and a push list is not."""
    listing = REPO_ROOT / ".push" / "files.txt"
    if not listing.is_file():
        pytest.skip("no push staged")
    r = run(listing)
    assert r.returncode == CLEAN, f"rc={r.returncode}\n{r.stdout}{r.stderr}"


def test_the_scanner_is_a_declared_standalone_entrypoint() -> None:
    """It has `if __name__ ==` and no runner starts it, so check 8 of
    check_deploy.sh requires it to be declared. Asserted here as well, because the
    declaration is the thing that stops the deploy guard from going red on a file
    that was never meant to be a node."""
    tsv = (REPO_ROOT / "infra" / "standalone_entrypoints.tsv").read_text()
    assert any(line.split("\t")[0] == "infra/redact_scan.py"
               for line in tsv.splitlines() if line and not line.startswith("#")), \
        "infra/redact_scan.py is not declared in infra/standalone_entrypoints.tsv"
