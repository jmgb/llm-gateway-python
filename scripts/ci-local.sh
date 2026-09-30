#!/usr/bin/env bash
# CI local: los mismos checks que .github/workflows/ci.yml (matriz 3.11 y 3.13),
# para cuando GitHub Actions no tiene minutos. El workflow también llama a este
# script, así que local y remoto no pueden divergir.
#
#   bash scripts/ci-local.sh [all|3.11|3.13]
#   bash scripts/ci-local.sh changed [base]   # solo si el diff vs base (origin/main) toca código
#   bash scripts/ci-local.sh pre-push         # rutas por stdin (lo usa .githooks/pre-push)
#
# Ejecuta todos los pasos aunque alguno falle y termina con un resumen; el exit
# code es 1 si falló alguno bloqueante. Un log por paso en $CI_LOCAL_LOG_DIR.
# Cada versión usa su propio venv (.venv-ci-<versión>) para no tocar el de desarrollo.
# `advisory` registra WARN sin cambiar el exit code (hoy ningún paso lo usa).
#
# Sincronía: scripts/release.py::_checks() ejecuta una lista parecida sobre el
# venv de desarrollo antes de publicar; si cambias un check aquí, cámbialo allí.
#
# No se reproduce: tests/live (API reales, necesitan claves y dinero), y
# `uv publish`/`gh release` (credenciales, solo scripts/release.py).
set -uo pipefail
mode="all"
LOG_DIR="${CI_LOCAL_LOG_DIR:-${TMPDIR:-/tmp}/ci-local-llm-gateway-python}"
summary=()
failed=0

_run_logged() { # <nombre> <cmd> -> rc; deja el log en $log
    local name="$1" cmd="$2" rc
    mkdir -p "$LOG_DIR"
    log="$LOG_DIR/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_-' '_').log"
    printf '▶ %s\n' "$name"
    if [[ -n "${CI:-}" ]]; then
        # En Actions la salida completa va al log del job.
        bash -c "$cmd" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        bash -c "$cmd" >"$log" 2>&1
        rc=$?
    fi
    return "$rc"
}

# step <nombre> <comando bash>: ejecuta, guarda log y sigue aunque falle.
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

# advisory <nombre> <comando bash>: como step, pero un fallo es WARN y no cambia el exit code.
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
    echo "== ci-local ($mode) · logs en $LOG_DIR"
    (( ${#summary[@]} )) && printf '%s\n' "${summary[@]}"
    return "$failed"
}

# Cargado con `source` (tests): solo las funciones.
(return 0 2>/dev/null) && return 0

cd "$(dirname "$0")/.."
run=0
versions=(3.11 3.13)

# Solo docs (markdown, docs/, licencia, artefactos) no lanza nada; lo demás sí.
select_from_paths() {
    local path
    while IFS= read -r path; do
        case "$path" in
            "") ;;
            *.md|docs/*|LICENSE|.gitignore|dist/*) ;;
            *) run=1 ;;
        esac
    done
}

mode="${1:-all}"
case "$mode" in
    all) run=1 ;;
    3.11|3.13) run=1; versions=("$mode") ;;
    pre-push) select_from_paths ;;
    changed)
        base="${2:-origin/main}"
        merge_base=$(git merge-base "$base" HEAD) || { echo "ci-local: no encuentro $base (¿git fetch?)" >&2; exit 2; }
        select_from_paths < <({ git diff --name-only "$merge_base"; git ls-files --others --exclude-standard; } | sort -u) ;;
    *) echo "usage: bash scripts/ci-local.sh [all|3.11|3.13|changed [base]|pre-push]" >&2; exit 2 ;;
esac

if (( ! run )); then
    echo "ci-local: solo docs, nada que validar"
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
step "uv.lock al día" 'uv lock --check --offline'
step "githooks: tests" 'bash .githooks/test-pre-push.sh && bash .githooks/test-env-isolation.sh && bash .githooks/test-ci-selection.sh'
finish
