#!/usr/bin/env bash
# open-agent-robot — deploy-path guard (test item T-J1).
#
# A node that exists in the tree but is unreachable from the runner is dead in
# production while looking alive in the source: `start-all` never starts it,
# `status` never reports it, and nothing fails loudly. That is how the Cantonese
# engine (`yue`) sat unstarted while speech/tts_node.py routed Cantonese text to
# a closed socket with no fallback.
#
# The guard verifies the runner's inventory BY RUNNING IT (OAR_DRY=1), not by
# parsing it — a parser that stops matching after a refactor reports clean, and a
# guard that cannot run must never report clean. Hence three exit codes:
#
#   exit 0  every check passed
#   exit 1  at least one finding (all findings are printed, not just the first)
#   exit 3  the guard could not run (unreadable runner, empty inventory, bad parse)
#
# Deliberately no `set -e`: we want the whole report in one pass.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="${OAR_RUNNER:-$ROOT/infra/run_ec2.sh}"
MANIFEST="${OAR_MANIFEST:-$ROOT/infra/generated_paths.tsv}"
STANDALONE="${OAR_STANDALONE:-$ROOT/infra/standalone_entrypoints.tsv}"

fail=0
findings=0
checks=0
note() { printf 'FAIL  %s\n' "$*" >&2; findings=$((findings + 1)); fail=1; }
ok()   { printf 'ok    %s\n' "$*"; checks=$((checks + 1)); }
die()  { printf 'ERROR %s\n' "$*" >&2; exit 3; }

[ -r "$RUNNER" ] || die "cannot read runner: $RUNNER"
[ -x "$RUNNER" ] || die "runner is not executable: $RUNNER"

# A throwaway RUN dir: the runner must never consult real pidfiles here, and a
# dry run must not create this directory at all (checked below).
DRYRUN_DIR="${TMPDIR:-/tmp}/oar-check.$$"

dry() { OAR_DRY=1 OAR_RUN="$DRYRUN_DIR" "$RUNNER" "$@" 2>&1; }
names() { sed -n 's/^\([a-z0-9_][a-z0-9_]*\):.*/\1/p'; }   # "gateway: would run …" -> gateway

in_list() { # in_list <needle> <space separated haystack>
  local needle=$1 hay=$2 item
  for item in $hay; do [ "$item" = "$needle" ] && return 0; done
  return 1
}

# ---- run the runner ---------------------------------------------------------
START_OUT=$(dry start-all)       || die "OAR_DRY=1 $RUNNER start-all exited non-zero:"$'\n'"$START_OUT"
STATUS_OUT=$(dry status)         || die "OAR_DRY=1 $RUNNER status exited non-zero:"$'\n'"$STATUS_OUT"
CORE_OUT=$(dry start)            || die "OAR_DRY=1 $RUNNER start exited non-zero:"$'\n'"$CORE_OUT"

STARTED=$(printf '%s\n' "$START_OUT"  | names | sort | tr '\n' ' ')
REPORTED=$(printf '%s\n' "$STATUS_OUT" | names | sort | tr '\n' ' ')
CORE=$(printf '%s\n' "$CORE_OUT"       | names | sort | tr '\n' ' ')

n_started=$(printf '%s\n' $STARTED | grep -c '[a-z]' || true)
[ "$n_started" -gt 0 ] || die "start-all named 0 nodes — empty inventory is a broken guard, not a pass"
[ "$(printf '%s\n' $CORE | grep -c '[a-z]' || true)" -gt 0 ] || die "bare 'start' named 0 nodes"

# ---- check 1: started set == reported set (behavioural T-J1) ----------------
for n in $STARTED; do
  if in_list "$n" "$REPORTED"; then
    ok "node '$n' is both started by start-all and reported by status"
  else
    note "node '$n' is started by start-all but NOT reported by status → it can die unnoticed"
  fi
done
for n in $REPORTED; do
  in_list "$n" "$STARTED" || \
    note "node '$n' is reported by status but NOT started by start-all → status shows a node nothing launches"
done
[ "$STARTED" = "$REPORTED" ] && ok "inventory sets are identical ($n_started nodes)"

# ---- check 2: the inventory is derived, not a second hand-kept list ---------
# The original defect was two lists drifting apart. Deriving one from the other
# makes it unrepresentable; this check fails if someone re-introduces a literal.
if grep -qE '^[[:space:]]*ALL="[a-z]' "$RUNNER"; then
  note "$RUNNER declares a literal ALL=\"...\" node list → it can drift from the command table again"
elif grep -q 'ALL=.*NODES' "$RUNNER"; then
  ok "ALL is derived from the node table (no second list to drift)"
else
  note "cannot tell how ALL is built in $RUNNER → re-read it; the guard's assumption no longer holds"
fi

# ---- check 3: the default 'start' set is inside the inventory ---------------
for c in $CORE; do
  if in_list "$c" "$STARTED"; then
    ok "default-start node '$c' is in the inventory"
  else
    note "bare 'start' launches '$c', which start-all/status do not know about"
  fi
done

# ---- check 4: an unknown node fails loudly, not cryptically ----------------
UNKNOWN_OUT=$(dry start __no_such_node__); UNKNOWN_RC=$?
if [ "$UNKNOWN_RC" -eq 0 ]; then
  note "asking the runner to start an unknown node exited 0 → a typo in a deploy script would look like success"
elif printf '%s' "$UNKNOWN_OUT" | grep -q 'unknown node'; then
  ok "unknown node is rejected with a named error (rc=$UNKNOWN_RC)"
else
  note "unknown node failed (rc=$UNKNOWN_RC) but without a readable message: $UNKNOWN_OUT"
fi

# ---- check 5: dry mode has no side effects ---------------------------------
if [ -e "$DRYRUN_DIR" ]; then
  note "a dry run created $DRYRUN_DIR → OAR_DRY is not side-effect free, so this guard mutates the host"
  rm -rf "$DRYRUN_DIR"
else
  ok "dry run created no state directory"
fi

# ---- check 6: every executed path exists, or has a real producer -----------
# Refs come from what the runner would actually exec, with $HOME re-symbolised so
# they can be compared against the manifest.
REFS=$(printf '%s\n' "$START_OUT" | sed "s#$HOME#\$HOME#g" | tr ' ' '\n' \
       | grep -E '\.(py|xml)$|/bin/python$' | sort -u)
n_refs=$(printf '%s\n' "$REFS" | grep -c . || true)
[ "$n_refs" -gt 0 ] || die "extracted 0 executable paths from the runner's dry output"

[ -r "$MANIFEST" ] || die "cannot read generated-path manifest: $MANIFEST"
# Strip comments/blank lines once; a manifest that parses to nothing is a bug
# only if some path actually needs it, so the floor is checked lazily below.
ROWS=$(grep -v '^[[:space:]]*#' "$MANIFEST" | grep -v '^[[:space:]]*$')

manifest_row() { printf '%s\n' "$ROWS" | awk -F'\t' -v p="$1" '$1==p{print; f=1} END{exit !f}'; }

for f in $REFS; do
  if [ -e "$ROOT/$f" ]; then
    ok "executed path is in the tree: $f"
    continue
  fi
  row=$(manifest_row "$f") || { note "runner executes '$f' but it is neither in the tree nor declared in $(basename "$MANIFEST")"; continue; }
  producer=$(printf '%s' "$row" | cut -f2)
  evidence=$(printf '%s' "$row" | cut -f3)
  invoked=$(printf '%s' "$row" | cut -f4)
  inv_file=${invoked%%:*}
  inv_stage=${invoked#*:}
  why="$f (producer $producer, via $invoked)"

  bad=0
  [ -n "$producer" ] && [ -n "$evidence" ] && [ -n "$invoked" ] || { note "manifest row for '$f' has an empty column"; continue; }
  [ -f "$ROOT/$producer" ] || { note "$why: producer does not exist"; bad=1; }
  [ -x "$ROOT/$producer" ] || { note "$why: producer is not executable"; bad=1; }
  case "$f" in
    *"$evidence"*) ;;
    *) note "$why: declared evidence '$evidence' is not part of the path → the row proves nothing"; bad=1 ;;
  esac
  if [ -f "$ROOT/$producer" ] && ! grep -qF -- "$evidence" "$ROOT/$producer"; then
    note "$why: producer never mentions '$evidence' → it does not look like what creates this path"; bad=1
  fi
  if [ ! -f "$ROOT/$inv_file" ]; then
    note "$why: invoking script '$inv_file' does not exist"; bad=1
  else
    grep -qE "^[[:space:]]*$inv_stage\)" "$ROOT/$inv_file" || {
      note "$why: '$inv_file' has no '$inv_stage)' stage → nothing a human can run reaches the producer"; bad=1; }
    if [ "$producer" != "$inv_file" ] && ! grep -qF -- "$(basename "$producer")" "$ROOT/$inv_file"; then
      note "$why: '$inv_file' never calls $(basename "$producer") → the producer is an orphan"; bad=1
    fi
  fi
  [ "$bad" -eq 0 ] && ok "executed path has a live producer: $why"
done

# ---- check 7: no stale manifest rows --------------------------------------
# A row for a path nothing executes would silently excuse a future missing file.
# A here-string, not a pipe: a `while read` in a pipeline runs in a subshell and
# every finding it counts is discarded when that subshell exits.
while IFS=$'\t' read -r p _rest; do
  [ -n "${p:-}" ] || continue
  if printf '%s\n' "$REFS" | grep -qxF -- "$p"; then
    ok "manifest row is live: $p"
  else
    note "manifest declares '$p' but the runner never executes it → a stale excuse for a future missing file"
  fi
done <<< "$ROWS"

# ---- check 8: every entrypoint in the tree is launched, or declared standalone
# The other direction of "no deploy path = no component", and the one that catches
# the defect at its source: speech/yue_server.py was a finished entrypoint that
# nothing started. Discovered from the tree, never from a remembered node list.
ENTRYPOINTS=$(cd "$ROOT" && grep -rl '^if __name__ ==' --include='*.py' . \
              | sed 's#^\./##' | grep -v '^tests/' | sort)
n_entry=$(printf '%s\n' "$ENTRYPOINTS" | grep -c . || true)
[ "$n_entry" -gt 0 ] || die "found 0 python entrypoints in $ROOT — the scan read nothing, which is an error, not a pass"

[ -r "$STANDALONE" ] || die "cannot read standalone-entrypoint manifest: $STANDALONE"
EXEMPT=$(grep -v '^[[:space:]]*#' "$STANDALONE" | grep -v '^[[:space:]]*$' | cut -f1 | sort)

for e in $ENTRYPOINTS; do
  launched=no; printf '%s\n' "$REFS" | grep -qxF -- "$e" && launched=yes
  exempt=no;   printf '%s\n' "$EXEMPT" | grep -qxF -- "$e" && exempt=yes
  case "$launched/$exempt" in
    yes/no) ok "entrypoint is launched by the runner: $e" ;;
    no/yes) ok "entrypoint is declared standalone: $e" ;;
    no/no)  note "'$e' is a runnable entrypoint that nothing launches and nothing declares standalone → dead in production while looking alive in the source" ;;
    yes/yes) note "'$e' is declared standalone in $(basename "$STANDALONE") but the runner does launch it → contradictory declaration" ;;
  esac
done
for x in $EXEMPT; do
  [ -f "$ROOT/$x" ] || \
    note "$(basename "$STANDALONE") exempts '$x', which does not exist → a stale exemption silently covers whatever takes its place"
done

printf -- '----\n%d checks passed, %d findings\n' "$checks" "$findings"
exit "$fail"
