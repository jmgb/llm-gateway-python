#!/usr/bin/env bash
# Selección de scripts/ci-local.sh con stubs de uv (sin red ni venvs).
set -euo pipefail
while IFS= read -r variable; do
    unset "$variable"
done < <(git rev-parse --local-env-vars)
script="$(cd "$(dirname "$0")/.." && pwd)/scripts/ci-local.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/repo"/{scripts,src,docs,.githooks}
repo="$tmp/repo"
cp "$script" "$repo/scripts/ci-local.sh"
for t in test-pre-push test-env-isolation test-ci-selection; do echo 'exit 0' > "$repo/.githooks/$t.sh"; done
cat > "$tmp/bin/uv" <<'STUB'
#!/usr/bin/env bash
echo "uv $*" >> "$CHECK_CALLS"
[[ -n "${FAIL_MATCH:-}" && "$*" == *"$FAIL_MATCH"* ]] && exit 1
exit 0
STUB
chmod +x "$tmp/bin/uv"
export PATH="$tmp/bin:$PATH" CHECK_CALLS="$tmp/calls" CI_LOCAL_LOG_DIR="$tmp/logs"
ci() { bash "$repo/scripts/ci-local.sh" "$@"; }

# docs-only -> nada
: > "$CHECK_CALLS"
ci pre-push <<< $'README.md\ndocs/a.md' > /dev/null
[[ ! -s "$CHECK_CALLS" ]] || { echo 'FAIL: docs-only ejecuta checks'; exit 1; }
# código -> pytest en las dos versiones
ci pre-push <<< 'src/pkg/x.py' > /dev/null
[[ "$(grep -c 'run --no-sync pytest' "$CHECK_CALLS")" == 2 ]] || { echo 'FAIL: código no ejecuta pytest en 3.11 y 3.13'; exit 1; }
# un fallo no corta el resto y da exit 1
: > "$CHECK_CALLS"
if FAIL_MATCH='ruff check' ci 3.11 > /dev/null; then echo 'FAIL: exit 0 con ruff fallando'; exit 1; fi
grep -q 'mypy' "$CHECK_CALLS" && grep -q 'run --no-sync pytest' "$CHECK_CALLS" || { echo 'FAIL: el fallo cortó los pasos siguientes'; exit 1; }
# advisory fallido -> exit 0 con WARN
out=$(bash -c "source '$repo/scripts/ci-local.sh'; advisory prueba false; finish" | tail -n 2)
grep -q WARN <<< "$out" || { echo 'FAIL: advisory sin WARN'; exit 1; }
bash -c "source '$repo/scripts/ci-local.sh'; advisory prueba false; finish" > /dev/null || { echo 'FAIL: advisory cambia el exit code'; exit 1; }
# changed: diff vs origin/main
(cd "$repo" && git init -q && git add -A && git -c user.email=ci@local -c user.name=ci commit -qm base \
    && git update-ref refs/remotes/origin/main HEAD && echo x > docs/new.md)
: > "$CHECK_CALLS"
(cd "$repo" && ci changed) > /dev/null
[[ ! -s "$CHECK_CALLS" ]] || { echo 'FAIL: changed con docs ejecuta checks'; exit 1; }
echo x > "$repo/src/new.py"
(cd "$repo" && ci changed) > /dev/null
grep -q 'run --no-sync pytest' "$CHECK_CALLS" || { echo 'FAIL: changed ignora el código'; exit 1; }
echo 'PASS: docs-only no ejecuta nada; los fallos no cortan y dan exit 1; advisory no rompe; changed sigue el diff'
