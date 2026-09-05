#!/usr/bin/env bash
# The hook covers both supported CI interpreters, without altering the dev venv.
set -euo pipefail
cd "$(dirname "$0")/.."
check_version() (
    export UV_PYTHON="$1"
    export UV_PROJECT_ENVIRONMENT=".venv-ci-$1"
    uv sync --locked
    uv run --no-sync python -c 'import os, sys; assert sys.version_info[:2] == tuple(map(int, os.environ["UV_PYTHON"].split(".")))'
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .
    uv run --no-sync mypy
    uv run --no-sync pytest
    uv build
    uv run --no-sync python scripts/audit_dist.py
)
case "${1:-all}" in
    all|pre-push) check_version 3.11; check_version 3.13 ;;
    3.11|3.13) check_version "$1" ;;
    *) echo "usage: bash scripts/ci-local.sh [all|3.11|3.13]" >&2; exit 2 ;;
esac
bash .githooks/test-pre-push.sh
bash .githooks/test-env-isolation.sh
