#!/usr/bin/env bash
# Exercise the installed hook with a real disposable repository, without CI tools.
set -euo pipefail
hook="$(cd "$(dirname "$0")" && pwd)/pre-push"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp"
cd "$tmp"
git config user.name Test
git config user.email test@example.com
mkdir scripts
cat > scripts/ci-local.sh <<'GATE'
#!/usr/bin/env bash
cat > "$CALLS"
[[ ! -f "$FAIL_GATE" ]]
GATE
git add .
git commit -qm initial
sha=$(git rev-parse HEAD)
zero=0000000000000000000000000000000000000000
export CALLS="$tmp/../hook-calls-$$" FAIL_GATE="$tmp/../hook-fail-$$"
trap 'rm -rf "$tmp"; rm -f "$CALLS" "$FAIL_GATE"' EXIT
push_line="refs/heads/main $sha refs/heads/main $zero"
# A deletion and an empty push must not run validation.
bash "$hook" <<< "refs/heads/main $zero refs/heads/main $sha"
test ! -f "$CALLS"
bash "$hook" < /dev/null
test ! -f "$CALLS"
# A new branch validates its complete tree.
bash "$hook" <<< "$push_line"
grep -q scripts/ci-local.sh "$CALLS"
# Validation cannot certify uncommitted fixes or a different commit.
echo dirty > untracked.txt
if bash "$hook" <<< "$push_line"; then echo 'FAIL: dirty tree accepted'; exit 1; fi
rm untracked.txt
echo change > tracked.txt
git add tracked.txt
git commit -qm second
if bash "$hook" <<< "$push_line"; then echo 'FAIL: wrong SHA accepted'; exit 1; fi
sha=$(git rev-parse HEAD)
touch "$FAIL_GATE"
if bash "$hook" <<< "refs/heads/main $sha refs/heads/main $zero"; then
    echo 'FAIL: failed checks accepted'; exit 1
fi
rm "$FAIL_GATE"
# A concurrent edit during checks invalidates the result.
echo 'touch raced.txt' >> scripts/ci-local.sh
git add scripts/ci-local.sh
git commit -qm race
sha=$(git rev-parse HEAD)
if bash "$hook" <<< "refs/heads/main $sha refs/heads/main $zero"; then
    echo 'FAIL: concurrent edit accepted'; exit 1
fi
echo 'PASS: exact commit, clean tree, failed gate and concurrent edits'
