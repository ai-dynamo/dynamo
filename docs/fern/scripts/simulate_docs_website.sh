#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# simulate_docs_website.sh — local regression harness for the fern-docs.yml
# sync + release-version composition.
#
# The workflow's real jobs only fire on main pushes and tag cuts, so changes
# to the composition (rsync scopes, nav path transforms, the shared-Reference
# machinery) are otherwise unvalidatable before merge. This script replays
# both jobs against a scratch checkout of the local docs-website branch and
# asserts the invariants:
#
#   1. fern check on the composed tree reports 0 errors.
#   2. The generated versions/<TAG>.yml keeps the Reference General variant
#      on ../pages-dev/ (shared, always-current) while the Kubernetes API and
#      Components variants point at the frozen ../pages-<TAG>/ snapshot.
#   3. The pages-<TAG> snapshot drops exactly the shared reference files and
#      keeps the versioned ones (runtime-config, observability).
#   4. No React .tsx leaks into pages-dev/components/ (doc pages only).
#   5. Pre-rework version files gain no shared-reference pointers.
#   6. Round two: a page added to the General variant on a later main push
#      propagates into the already-cut version's nav.
#
# Usage: ./scripts/simulate_docs_website.sh [TAG]
#   TAG defaults to v9.9.9 (must not exist on docs-website yet).
#
# Requires: git, rsync, perl, yq v4, python3 >= 3.10 (or python3.13), and the
# fern CLI for check 1 (skipped with a warning if unavailable).
set -euo pipefail

TAG="${1:-v9.9.9}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/docs/fern"

PY="$(command -v python3.13 || command -v python3)"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: python3 >= 3.10 required (convert_callouts.py uses 3.10 syntax)"; exit 1
fi
command -v yq >/dev/null || { echo "ERROR: yq (v4) required"; exit 1; }

WT="$(mktemp -d)/docs-checkout"
cleanup() { git -C "$REPO_ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || true; }
trap cleanup EXIT
git -C "$REPO_ROOT" worktree add --quiet --detach "$WT" docs-website

fail=0
note() { printf '%-64s %s\n' "$1" "$2"; }
assert() { # assert <label> <ok|FAIL>
  note "$1" "$2"; [ "$2" = "ok" ] || fail=1
}

REF_SEL='.navigation[] | select(.tab == "reference") | .variants[] | select(.title == "General")'

echo "=== SYNC JOB (replayed) ==="
rm -rf "$WT/fern/pages-dev"; mkdir -p "$WT/fern/pages-dev"
rsync -a \
  --exclude='digest' --exclude='index.yml' --exclude='fern.config.json' \
  --exclude='docs.yml' --exclude='components' --exclude='main.css' \
  --exclude='products' --exclude='welcome.mdx' --exclude='convert_callouts.py' \
  --exclude='.gitignore' --exclude='dev.sh' --exclude='watch.sh' \
  "$SRC/" "$WT/fern/pages-dev/"
rsync -a --include='*/' --include='*.md' --include='*.mdx' --exclude='*' --prune-empty-dirs \
  "$SRC/components/" "$WT/fern/pages-dev/components/"
rsync -a --include='*/' --include='backends/*/deploy/**' --exclude='*' --prune-empty-dirs \
  "$REPO_ROOT/examples/" "$WT/examples/"

cp "$SRC/index.yml" "$WT/fern/versions/dev.yml"
cp "$SRC/fern.config.json" "$WT/fern/fern.config.json"
[ -f "$SRC/README.md" ] && cp "$SRC/README.md" "$WT/fern/README.md" || true
[ -f "$SRC/convert_callouts.py" ] && cp "$SRC/convert_callouts.py" "$WT/fern/convert_callouts.py" || true
rm -rf "$WT/fern/components"; cp -r "$SRC/components" "$WT/fern/components"
rm -rf "$WT/fern/products"; cp -r "$SRC/products" "$WT/fern/products"
cp "$SRC/welcome.mdx" "$WT/fern/welcome.mdx"
[ -d "$SRC/assets" ] && cp -r "$SRC/assets/." "$WT/fern/assets/" || true
yq -i '(.. | select(has("path")).path) |= sub("^../", "../pages-dev/")' "$WT/fern/products/community.yml"
if [ -d "$SRC/digest" ]; then
  rm -rf "$WT/fern/digest"; cp -r "$SRC/digest" "$WT/fern/digest"
  perl -pi -e 's|(path: \.\./digest/.*)\.md$|$1.mdx|' "$WT"/fern/versions/v*.yml
fi
[ -f "$SRC/main.css" ] && cp "$SRC/main.css" "$WT/fern/main.css" || true
[ -f "$SRC/custom.js" ] && cp "$SRC/custom.js" "$WT/fern/custom.js" || true

yq -i '(.. | select(has("path")).path) |= sub("^digest/", "../digest/")' "$WT/fern/versions/dev.yml"
yq -i '(.. | select(has("path")).path) |= sub("^([a-zA-Z])", "../pages-dev/${1}")' "$WT/fern/versions/dev.yml"

propagate_shared_reference() {
  yq "[$REF_SEL][0]" "$WT/fern/versions/dev.yml" > "$WT/.ref_general.yml"
  for vfile in "$WT"/fern/versions/v*.yml; do
    [ -e "$vfile" ] || continue
    if [ "$(yq "[$REF_SEL] | length" "$vfile")" != "0" ]; then
      REF_BLOCK="$WT/.ref_general.yml" \
        yq -i "($REF_SEL) = load(strenv(REF_BLOCK))" "$vfile"
    fi
  done
}
propagate_shared_reference

"$PY" "$WT/fern/convert_callouts.py" --dir "$WT/fern/pages-dev" >/dev/null

cd "$WT/fern"
yq '[.products[] | select(.display-name == "Docs" or .display-name == "Dynamo")][0] | {"path": .path, "versions": .versions}' \
  docs.yml > "$WT/.preserved_docs.yml"
cp "$SRC/docs.yml" docs.yml
PRESERVED="$WT/.preserved_docs.yml" \
  yq -i '( .products[] | select(.display-name == "Docs") ) *= load(strenv(PRESERVED))' docs.yml

echo "=== RELEASE-VERSION JOB (replayed, tag $TAG) ==="
cd "$WT"
[ -d "fern/pages-$TAG" ] && { echo "ERROR: pages-$TAG already exists on docs-website"; exit 1; }
[ -f "fern/versions/$TAG.yml" ] && { echo "ERROR: versions/$TAG.yml already exists"; exit 1; }

cp -r fern/pages-dev "fern/pages-$TAG"
yq "$REF_SEL | .. | select(has(\"path\")) | .path" fern/versions/dev.yml \
  | sed 's|^\.\./pages-dev/||' | while read -r relpath; do
    [ -n "$relpath" ] && rm -f "fern/pages-$TAG/$relpath"
  done
find "fern/pages-$TAG/reference" -type d -empty -delete 2>/dev/null || true

find "fern/pages-$TAG" \( -name "*.md" -o -name "*.mdx" \) -print0 | xargs -0 perl -pi -e \
  "s|github.com/ai-dynamo/dynamo/tree/main|github.com/ai-dynamo/dynamo/tree/$TAG|g; s|github.com/ai-dynamo/dynamo/blob/main|github.com/ai-dynamo/dynamo/blob/$TAG|g"
"$PY" fern/convert_callouts.py --dir "fern/pages-$TAG" >/dev/null

VERSION_FILE="fern/versions/$TAG.yml"
cp fern/versions/dev.yml "$VERSION_FILE"
perl -pi -e "s|path: \.\./pages-dev/|path: ../pages-$TAG/|g" "$VERSION_FILE"
yq -i "($REF_SEL | .. | select(has(\"path\")).path) |= sub(\"\.\./pages-$TAG/\", \"../pages-dev/\")" "$VERSION_FILE"
perl -pi -e "s|href: /dynamo/dev/|href: /dynamo/$TAG/|g" "$VERSION_FILE"

DOCS_IDX=$(yq '.products | to_entries | map(select(.value.display-name == "Docs")) | .[0].key' fern/docs.yml)
DEV_IDX=$(yq ".products[$DOCS_IDX].versions | to_entries | map(select(.value.display-name == \"dev\")) | .[0].key" fern/docs.yml)
INSERT_IDX=$((DEV_IDX + 1))
yq -i "
  .products[$DOCS_IDX].versions |= (
    .[:$INSERT_IDX] +
    [{\"display-name\": \"$TAG\", \"path\": \"./versions/$TAG.yml\", \"slug\": \"$TAG\", \"availability\": \"stable\"}] +
    .[$INSERT_IDX:]
  )
" fern/docs.yml
yq -i ".products[$DOCS_IDX].path = \"./versions/$TAG.yml\"" fern/docs.yml
yq -i ".products[$DOCS_IDX].versions[0].path = \"./versions/$TAG.yml\"" fern/docs.yml
yq -i ".products[$DOCS_IDX].versions[0].display-name = \"Latest ($TAG)\"" fern/docs.yml

echo "=== ASSERTIONS ==="
shared=$(grep -c "path: \.\./pages-dev/reference/" "$VERSION_FILE" || true)
frozen=$(grep -c "path: \.\./pages-$TAG/reference/runtime-config-reference.mdx" "$VERSION_FILE" || true)
[ "$shared" -ge 10 ] && s1=ok || s1=FAIL
assert "2. $TAG.yml: General variant shared ($shared pages-dev refs)" "$s1"
[ "$frozen" -eq 1 ] && s2=ok || s2=FAIL
assert "2. $TAG.yml: Components variant frozen (runtime-config)" "$s2"

[ ! -e "fern/pages-$TAG/reference/compatibility.mdx" ] && \
  [ -e "fern/pages-$TAG/reference/runtime-config-reference.mdx" ] && \
  [ -d "fern/pages-$TAG/reference/observability" ] && s3=ok || s3=FAIL
assert "3. snapshot drops shared files, keeps versioned reference/" "$s3"

[ "$(find fern/pages-dev/components -name '*.tsx' | wc -l | tr -d ' ')" = "0" ] && s4=ok || s4=FAIL
assert "4. no .tsx in pages-dev/components/" "$s4"

prework=$(git diff -U0 -- 'fern/versions/v*.yml' 2>/dev/null \
  | grep -E '^[+-][^+-]' | grep -c "pages-dev/reference" || true)
[ "$prework" -eq 0 ] && s5=ok || s5=FAIL
assert "5. pre-rework versions gain no shared-reference pointers" "$s5"

# Round two: a later main push adds a page to the shared reference; the
# already-cut version's nav must pick it up via propagation.
FAKE='{"page": "Sim Test Page", "path": "../pages-dev/reference/sim-test-page.mdx", "slug": "sim-test-page"}'
FAKE="$FAKE" yq -i "($REF_SEL | .layout[] | select(has(\"section\")) | select(.section == \"Releases\") | .contents) += [env(FAKE)]" \
  "$WT/fern/versions/dev.yml"
propagate_shared_reference
grep -q "sim-test-page" "$VERSION_FILE" && s6=ok || s6=FAIL
assert "6. round-two propagation reaches the cut version's nav" "$s6"
# Undo before fern check (the fake page has no backing file).
perl -ni -e 'print unless /sim-test-page|Sim Test Page/' "$WT/fern/versions/dev.yml" "$VERSION_FILE"

if command -v fern >/dev/null 2>&1; then
  errors=$(cd "$WT/fern" && fern check 2>&1 | grep -oE "Found [0-9]+ error" | grep -oE "[0-9]+" || echo "?")
  [ "$errors" = "0" ] && s7=ok || s7=FAIL
  assert "1. fern check on composed tree ($errors errors)" "$s7"
else
  note "1. fern check" "SKIPPED (fern CLI not installed)"
fi

echo ""
if [ "$fail" -eq 0 ]; then echo "ALL ASSERTIONS PASSED"; else echo "ASSERTIONS FAILED"; exit 1; fi
