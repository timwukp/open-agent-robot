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
   prints rows to stdout and refuses to write for exactly this reason. The single
   exception is this file, which carries no annotation: it is registered so the
   machinery is live rather than vacuously green, and a document describing the set
   has to be allowed to describe the set. Every other row is a label, and re-freezing
   a label is how a corrupted fixture becomes the new baseline.
3. **Nothing here may contain personal data.** Conversation fixtures are synthetic or
   redacted before they land; `$HOME/oar-data/` never enters the repo.
4. **Thresholds derived from these files are regression lines, not truth.** The
   30-query recall set and the boundary-F1 set are frozen *before* the retriever and
   segmenter are written, so the first run records a number instead of gating on one
   (docs/test-design.md §6.0 grade R).

## What is here now

| file | stage | what it is | read by |
|---|---|---|---|
| `skill_triggers.tsv` | S1 | 20 frozen utterances, 10 that must load the skill and 10 that must not (T-I2) | `tests/test_loader.py` |
| `why_goal_rejects.jsonl` | S4.5 | the frozen `why.goal` rejection set — 31 rows, 9 classes, **0 may be accepted** (T-C1) | `tests/test_why.py` |

`why_goal_rejects.jsonl` is jsonl rather than tsv because one of its rows contains a
literal newline: an embedded newline in a retrieval key would forge a second record
if the value were ever written unescaped, so the row that tests for it cannot live in
a line-delimited-by-tabs format at all. Both adversarial sets are **lower bounds and
only ever grow** (docs/test-design.md §6.0 rule 3).

Still to arrive: the 30 hand-annotated evaluation queries and the episode-boundary
labels (both S4, blocked on a real conversation log and one human annotation pass),
and the S7 red-team set.
