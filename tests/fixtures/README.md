# Frozen test fixtures

Every file in this directory is **frozen**: it was annotated once, by hand, and the
annotation is only valid for the exact bytes that were annotated. `tests/fixtures.tsv`
records a size floor, a line floor and a sha256 for each one, and
`tests/test_fixture_manifest.py` fails if any of the three stops matching.

Rules, in the order they matter:

1. **Adding a file here without adding a row to `tests/fixtures.tsv` is a failure.**
   An unannotated file that reaches a replay threshold is grading against unlabelled
   data while reporting a number.
2. **A hash mismatch means re-annotate or revert — never re-freeze.** `--freeze`
   prints rows to stdout and refuses to write for exactly this reason.
3. **Nothing here may contain personal data.** Conversation fixtures are synthetic or
   redacted before they land; `$HOME/oar-data/` never enters the repo.
4. **Thresholds derived from these files are regression lines, not truth.** The
   30-query recall set and the boundary-F1 set are frozen *before* the retriever and
   segmenter are written, so the first run records a number instead of gating on one
   (docs/test-design.md §6.0 grade R).

Empty of data fixtures until stage S4 — this file is registered in the manifest so
the machinery is live from the first commit rather than vacuously green.
