# skills/ — three-level skill loading

A skill is a directory of markdown. `loader.py` makes it reachable from the robot's
conversation without paying for it on every turn.

## Why three levels

The full `critical-thinking/SKILL.md` is 5,672 bytes ≈ 1.4k input tokens. Resident in
the system prompt, that is 1.4k tokens on *every* utterance — including "你今天過得怎麼樣"
— to serve the small fraction of turns that argue about evidence. So it is split:

| Level | What | Where it lives | When it is paid for |
|---|---|---|---|
| **L1** | name + first sentence of `description` | appended to the system prompt | every turn |
| **L2** | the `SKILL.md` body | attached to the current **user** turn | turns whose text matches a trigger |
| **L3** | `methods/`, `templates/`, `references/` | fetched by the model via `read_skill_file` | when the SKILL.md routing table sends it there |

Measured on 2026-08-22 against `global.anthropic.claude-sonnet-5` in ap-southeast-1
(`python skills/measure_l1_tokens.py`): the L1 block for one skill costs **96 real input
tokens**, against a budget of 150/skill (test item T-I1). The CI check uses a character
proxy calibrated against that reading — it under-counts by 21%, so the assertion inflates
it before comparing. Re-run the script whenever the index changes;
`tests/test_skills_loader.py` fails if the string moves away from the measured one.

## What decides that a skill is active

A frozen phrase table (`<skill>/triggers.tsv`) plus a regex. **Not the model.**

Asking an LLM "should I load this skill?" is an LLM judge, and judge-gated pipelines
move the *measured* pass rate 0.72 → 0.94 while real correctness stays flat at 0.20; a
strict three-judge ensemble still accepts 55% of wrong answers (docs/test-design.md §0
rule 1). A phrase table is free, auditable, reproducible, and gradeable against a frozen
fixture — `tests/fixtures/skill_triggers.tsv` holds 20 utterances, 10 that must load and
10 that must not, and the suite requires 20/20.

The honest cost: **it fires on the phrases we listed and no others.** A user who asks for
the same thing in words nobody wrote down gets an ordinary answer. That is a stated limit,
not an oversight; the fix is adding rows, and each new row must arrive with a near-miss
row that must *not* fire.

Matching is script-dependent, which is subtler than it looks. Chinese is written without
spaces, so a CJK trigger has to match as a plain substring. Latin triggers need word
boundaries or `bias` fires inside `unbiased`. And `\b` around a CJK phrase asserts a
boundary between two non-word characters and therefore *never matches* — a naive
`\b…\b` would silently disable every Chinese trigger, a gate that quietly stops opening.
Both directions are in the fixture.

## Adding a skill

```
skills/<name>/
  SKILL.md      ---\nname: <name>\ndescription: …\n---  then the body
  triggers.tsv  phrase <TAB> lang, one per line
  …             anything else the routing table in SKILL.md names
```

Rules the loader enforces loudly, because a skill that silently fails to load is a
capability that silently is not there:

1. `name:` in the frontmatter must equal the directory name — `read_skill_file` paths are
   directory-based, so a mismatch makes the routing table wrong.
2. Frontmatter is `key: value` lines only. Not YAML: a real parser accepts folds and
   anchors, and a folded `description:` would be silently truncated at its first line
   while the token budget still looked comfortable.
3. `triggers.tsv` must exist and be non-empty. A skill with no gate can never load, so it
   would sit in the L1 index costing tokens forever with nothing going red.
4. Discovery scans the directory. There is no list of skill names anywhere — a list cannot
   notice a new name.

Then: add fixture rows for the new triggers (positive **and** near-miss), and re-run
`python skills/measure_l1_tokens.py` to refresh the recorded token figure.

## read_skill_file is confined to this directory

L3 reads go through `loader.read_skill_file`, which resolves the path and refuses anything
landing outside `skills/`. Containment is checked **after** symlink resolution, because a
symlink inside `skills/` pointing at `~/.aws/credentials` is a legal relative path
containing no `..` — a string check on the path passes it. That exact mutant is in the
suite (T-I3), alongside ~15 traversal payloads and positive controls, since a
`read_skill_file` that raised on everything would pass a traversal suite perfectly.

Refusals are returned to the model as a tool result with the reason, not raised: a wrong
path is a routing mistake far more often than an attack, and a silent empty result just
makes it retry the same wrong path. The tool loop is bounded at `MAX_TOOL_HOPS = 3`.

## Loaded skill text is data, not instruction

L2 goes into the current user turn wrapped in `<skill name=… dir=…>`. It is never
concatenated into the `system` block — that block is the one region a reader can trust to
be ours, and keeping loaded material outside it is the same invariant the memory layer is
held to (T-C5).

## Vendored skills and licences

| Directory | Origin | Licence |
|---|---|---|
| `critical-thinking/` | copied unmodified from the author's own Claude Code skill | MIT (see `critical-thinking/LICENSE`) |

`triggers.tsv` is the only file **added** to a vendored skill; everything else is the
upstream original, so an upstream update is a clean re-copy. The `yue` rows in
`critical-thinking/triggers.tsv` were written by a non-native Cantonese speaker and are
marked as such in that file.

`loader.py` imports no boto3, no rclpy, no network — enforced statically and at runtime by
the test suite. That is what makes the whole gate auditable by reading it on a laptop.
