#!/usr/bin/env bash
# Path selection in scripts/ci-local.sh, against a stubbed uv (no network, no venvs).
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

# prose only -> nothing
: > "$CHECK_CALLS"
ci pre-push <<< $'README.md\ndocs/a.md' > /dev/null
[[ ! -s "$CHECK_CALLS" ]] || { echo 'FAIL: a prose-only push runs checks'; exit 1; }
# files that look like prose but are read by a test or by the build -> checks
for path in docs/pricing.md .gitignore; do
    : > "$CHECK_CALLS"
    ci pre-push <<< "$path" > /dev/null
    grep -q 'run --no-sync pytest' "$CHECK_CALLS" || { echo "FAIL: $path skips the checks"; exit 1; }
done
# code -> pytest on every interpreter
: > "$CHECK_CALLS"
ci pre-push <<< 'src/pkg/x.py' > /dev/null
[[ "$(grep -c 'run --no-sync pytest' "$CHECK_CALLS")" == 3 ]] || { echo 'FAIL: code does not run pytest on 3.11, 3.13 and 3.14'; exit 1; }
# an unpacked sdist has no .githooks -> the hook tests are skipped, not failed
mv "$repo/.githooks" "$tmp/githooks"
out=$(ci 3.11) || { echo 'FAIL: a tree without .githooks fails'; exit 1; }
! grep -q githooks <<< "$out" || { echo 'FAIL: a tree without .githooks runs the hook tests'; exit 1; }
mv "$tmp/githooks" "$repo/.githooks"
# one failure does not stop the rest, and exits 1
: > "$CHECK_CALLS"
if FAIL_MATCH='ruff check' ci 3.11 > /dev/null; then echo 'FAIL: exit 0 while ruff fails'; exit 1; fi
grep -q 'mypy' "$CHECK_CALLS" && grep -q 'run --no-sync pytest' "$CHECK_CALLS" || { echo 'FAIL: one failure stopped the later steps'; exit 1; }
# a failed advisory -> exit 0 with a WARN
out=$(bash -c "source '$repo/scripts/ci-local.sh'; advisory probe false; finish" | tail -n 2)
grep -q WARN <<< "$out" || { echo 'FAIL: advisory without a WARN'; exit 1; }
bash -c "source '$repo/scripts/ci-local.sh'; advisory probe false; finish" > /dev/null || { echo 'FAIL: advisory changes the exit code'; exit 1; }
# changed: the diff against origin/main
(cd "$repo" && git init -q && git add -A && git -c user.email=ci@local -c user.name=ci commit -qm base \
    && git update-ref refs/remotes/origin/main HEAD && echo x > docs/new.md)
: > "$CHECK_CALLS"
(cd "$repo" && ci changed) > /dev/null
[[ ! -s "$CHECK_CALLS" ]] || { echo 'FAIL: changed runs checks for prose'; exit 1; }
echo x > "$repo/src/new.py"
(cd "$repo" && ci changed) > /dev/null
grep -q 'run --no-sync pytest' "$CHECK_CALLS" || { echo 'FAIL: changed ignores code'; exit 1; }
echo 'PASS: prose runs nothing, test-read files run everything; no .githooks skips the hook tests; failures do not stop the run and exit 1; advisory does not fail it; changed follows the diff'
