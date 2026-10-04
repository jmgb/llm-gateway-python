#!/usr/bin/env bash
# Local CI: the same checks as .github/workflows/ci.yml (Python 3.11, 3.13, 3.14),
# for when GitHub Actions has no minutes left. The workflow calls this script
# too, so local and remote runs cannot drift apart.
#
#   bash scripts/ci-local.sh [all|3.11|3.13|3.14]
#   bash scripts/ci-local.sh changed [base]   # only if the diff against base (origin/main) needs it
#   bash scripts/ci-local.sh pre-push         # paths on stdin (used by .githooks/pre-push)
#
# Every step runs even when an earlier one fails, and a summary closes the run;
# the exit code is 1 if any blocking step failed. One log per step lands in
# $CI_LOCAL_LOG_DIR. Each interpreter gets its own .venv-ci-<version>, so the
# development environment is never touched. `advisory` records a WARN without
# changing the exit code (no step uses it today).
#
# Keep in step: scripts/release.py::_checks() runs a similar list against the
# development environment before a release; a check changed here changes there.
#
# Not reproduced: tests/live (real APIs, keys and money) and `uv publish` /
# `gh release` (credentials, scripts/release.py only).
set -uo pipefail
mode="all"
LOG_DIR="${CI_LOCAL_LOG_DIR:-${TMPDIR:-/tmp}/ci-local-llm-gateway-python}"
summary=()
failed=0

_run_logged() { # <name> <cmd> -> rc; leaves the log path in $log
    local name="$1" cmd="$2" rc
    mkdir -p "$LOG_DIR"
    log="$LOG_DIR/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_-' '_').log"
    printf '▶ %s\n' "$name"
    if [[ -n "${CI:-}" ]]; then
        # In Actions the full output belongs in the job log.
        bash -c "$cmd" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        bash -c "$cmd" >"$log" 2>&1
        rc=$?
    fi
    return "$rc"
}

# step <name> <bash command>: run it, keep its log, carry on if it fails.
step() {
    local name="$1" start=$SECONDS rc log
    _run_logged "$name" "$2"; rc=$?
    if (( rc == 0 )); then
        summary+=("ok    $(( SECONDS - start ))s  $name")
    else
        summary+=("FAIL  $(( SECONDS - start ))s  $name  → $log")
        failed=1
        [[ -n "${CI:-}" ]] || tail -n 40 "$log"
    fi
    return "$rc"
}

# advisory <name> <bash command>: like step, but a failure is a WARN and keeps the exit code.
advisory() {
    local name="$1" start=$SECONDS rc log
    _run_logged "$name" "$2"; rc=$?
    if (( rc == 0 )); then
        summary+=("ok    $(( SECONDS - start ))s  $name (advisory)")
    else
        summary+=("WARN  $(( SECONDS - start ))s  $name (advisory)  → $log")
    fi
    return 0
}

finish() {
    echo
    echo "== ci-local ($mode) · logs in $LOG_DIR"
    (( ${#summary[@]} )) && printf '%s\n' "${summary[@]}"
    return "$failed"
}

# Sourced (by the hook tests): define the functions and stop.
(return 0 2>/dev/null) && return 0

cd "$(dirname "$0")/.."
run=0
versions=(3.11 3.13 3.14)

# Prose alone (Markdown, docs/, the licence, build output) runs nothing;
# anything else runs everything. Two files look like prose and are not:
# docs/pricing.md is read by a contract test, and .gitignore decides what
# hatchling leaves out of the sdist, which the artifact audit must then see.
select_from_paths() {
    local path
    while IFS= read -r path; do
        case "$path" in
            "") ;;
            docs/pricing.md|.gitignore) run=1 ;;
            *.md|docs/*|LICENSE|dist/*) ;;
            *) run=1 ;;
        esac
    done
}

mode="${1:-all}"
case "$mode" in
    all) run=1 ;;
    3.11|3.13|3.14) run=1; versions=("$mode") ;;
    pre-push) select_from_paths ;;
    changed)
        base="${2:-origin/main}"
        merge_base=$(git merge-base "$base" HEAD) || { echo "ci-local: cannot find $base (git fetch?)" >&2; exit 2; }
        # Captured first: a failing git inside a process substitution would
        # read as an empty diff, and an empty diff passes as prose-only.
        paths=$({ git diff --name-only "$merge_base" && git ls-files --others --exclude-standard; }) \
            || { echo "ci-local: cannot list the changes against $base" >&2; exit 2; }
        select_from_paths <<< "$paths" ;;
    *) echo "usage: bash scripts/ci-local.sh [all|3.11|3.13|3.14|changed [base]|pre-push]" >&2; exit 2 ;;
esac

if (( ! run )); then
    echo "ci-local: prose-only change, nothing to check"
    exit 0
fi

pycheck='import os, sys; assert sys.version_info[:2] == tuple(map(int, os.environ["UV_PYTHON"].split(".")))'
check_version() {
    local v="$1" p="py$1"
    export UV_PYTHON="$v" UV_PROJECT_ENVIRONMENT=".venv-ci-$v"
    step "$p: uv sync" "uv sync --locked && uv run --no-sync python -c '$pycheck'" || return 0
    step "$p: ruff check" 'uv run --no-sync ruff check .'
    step "$p: ruff format" 'uv run --no-sync ruff format --check .'
    step "$p: mypy" 'uv run --no-sync mypy'
    step "$p: pytest" 'uv run --no-sync pytest'
    step "$p: build" 'uv build'
    step "$p: audit dist" 'uv run --no-sync python scripts/audit_dist.py'
}

for v in "${versions[@]}"; do check_version "$v"; done
step "uv.lock up to date" 'uv lock --check --offline'
# An unpacked sdist has no .githooks (the artifact audit refuses dotdirs), and
# there are no hooks to test in it either.
if [[ -d .githooks ]]; then
    step "githooks: tests" 'bash .githooks/test-pre-push.sh && bash .githooks/test-env-isolation.sh && bash .githooks/test-ci-selection.sh'
fi
finish
