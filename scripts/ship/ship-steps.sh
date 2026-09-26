#!/bin/bash
# mrp-paper's steps for the host ship queue (mellowmarshall/devtools#92). The
# queue runs origin/main's copy of this file, never a branch's, so a branch
# cannot change how it ships. .tree-guard.conf names each step. Each step
# runs in the job's worktree.
#
#   ship-steps.sh body BODYFILE     the body names an issue (Closes #N)
#   ship-steps.sh ci                the repo's check (see run_ci)
#   ship-steps.sh merge PR          squash-merge the tested head
set -u

REPO_SLUG=mellowmarshall/mrp-paper

# Parse every tracked Python file in memory: no bytecode, no __pycache__.
py_syntax() { # PATHSPEC...
  git ls-files -z -- "$@" | xargs -0 -r python3 -c '
import sys
bad = 0
for f in sys.argv[1:]:
    try:
        compile(open(f, encoding="utf-8").read(), f, "exec")
    except SyntaxError as e:
        print(f"syntax error: {e}"); bad = 1
print(f"python syntax: {len(sys.argv) - 1} file(s) checked")
sys.exit(bad)'
}

# mrp-paper has no test suite (pyproject names a tests/ directory that does
# not exist), and its runtime needs torch and model weights. The CI is the
# cheapest real check that exists: every tracked Python file parses.
run_ci() {
  py_syntax '*.py' && echo "ci passed"
}

# Exit 0 merged; 75 the base moved under the tested head (one more round);
# 76 GitHub could not be read (run once more); anything else refuses, with the
# line in $SHIP_QUEUE_REASON_FILE.
run_merge() { # PR
  local out rc=0
  out=$(gh pr merge "$1" -R "$REPO_SLUG" --squash --match-head-commit "${SHIP_QUEUE_TIP:?the queue sets the tip}" 2>&1) || rc=$?
  echo "$out"
  [ "$rc" != 0 ] || return 0
  case "$out" in
    *"not mergeable"*|*"is not up to date"*|*"merge conflict"*|*"Base branch was modified"*) return 75 ;;
    *"timeout"*|*"TLS"*|*"connection"*|*"502"*|*"503"*) return 76 ;;
  esac
  [ -z "${SHIP_QUEUE_REASON_FILE:-}" ] || printf '%s\n' "$(printf '%s' "$out" | head -1)" > "$SHIP_QUEUE_REASON_FILE"
  return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    body) grep -qE '[Cc]loses #[0-9]+|[Rr]efs #[0-9]+' "$2" 2>/dev/null ||
        { echo "names no issue: write 'Closes #N' before submitting"; exit 1; } ;;
    ci) run_ci ;;
    merge) run_merge "$2" ;;
    *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
fi
