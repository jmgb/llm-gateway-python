#!/usr/bin/env bash
set -euo pipefail
# Git hooks may export an absolute GIT_DIR/GIT_WORK_TREE. A cd alone is not isolation.
while IFS= read -r variable; do
    unset "$variable"
done < <(git rev-parse --local-env-vars)
test_script="$(cd "$(dirname "$0")" && pwd)/test-pre-push.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp"
git -C "$tmp" config user.name Test
git -C "$tmp" config user.email test@example.com
echo preserve > "$tmp/user.txt"
git -C "$tmp" add user.txt
git -C "$tmp" commit -qm original
before=$(git -C "$tmp" rev-parse HEAD)
GIT_DIR="$tmp/.git" GIT_WORK_TREE="$tmp" bash "$test_script" > "$tmp/.git/test-output" 2>&1 || { echo 'FAIL: inherited Git environment breaks self-test'; exit 1; }
[[ "$(git -C "$tmp" rev-parse HEAD)" == "$before" ]] || { echo 'FAIL: decoy repo HEAD changed'; exit 1; }
[[ -z "$(git -C "$tmp" status --porcelain)" ]] || { echo 'FAIL: decoy repo files changed'; exit 1; }
[[ "$(git -C "$tmp" config user.name)" == Test ]] || exit 1
echo 'PASS: inherited Git environment leaves outer repository untouched'
