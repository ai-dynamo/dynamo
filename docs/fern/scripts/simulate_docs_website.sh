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
#   2. The generated versions/<TAG>.yml keeps the shared Reference group
#      (reference/general/) on ../pages-dev/ (always-current) while Kubernetes
#      API, Components, Backends, Observability and NIXL Connect point at the
#      frozen ../pages-<TAG>/ snapshot.
#   3. The pages-<TAG> snapshot drops exactly the shared reference files and
#      keeps the versioned ones (runtime-config, observability).
#   4. No React .tsx leaks into pages-dev/components/ (doc pages only).
#   5. Pre-rework version files gain no shared-reference pointers.
#   6. Round two: a page added to the General variant on a later main push
#      propagates into the already-cut version's nav.
#   7. The callout converter is published under fern/scripts/ and the legacy
#      root-level copy is removed.
#   8. Translation links resolve from authored relative paths to dev URLs in
#      pages-dev and tag-pinned URLs in pages-<TAG>, with no /dev/ leakage.
#   9. dev.yml still exposes a shared Reference group at all. Checked first,
#      because 2, 3 and 6 are vacuous without it — and because a selector that
#      quietly stops matching is exactly how this broke: #12410 flattened the
#      Reference tab and the old label-keyed selector no-opped for six days.
#  10. The composed default version starts with the shared Home tab so the
#      bare site URL renders Home instead of the release snapshot's first page.
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
DOCS_WEBSITE_REF="${DOCS_WEBSITE_REF:-upstream/docs-website}"
if ! git -C "$REPO_ROOT" rev-parse --verify "$DOCS_WEBSITE_REF" >/dev/null 2>&1; then
  DOCS_WEBSITE_REF=docs-website
fi
git -C "$REPO_ROOT" worktree add --quiet --detach "$WT" "$DOCS_WEBSITE_REF"

fail=0
note() { printf '%-64s %s\n' "$1" "$2"; }
assert() { # assert <label> <ok|FAIL>
  note "$1" "$2"; [ "$2" = "ok" ] || fail=1
}

# The shared Reference group is identified by CONTENT PATH, not by a nav label.
# It used to key on `variants[] | select(.title == "General")`; #12410
# deliberately flattened the Reference tab into folded sections with no variant
# selector, and the label-keyed selector then matched nothing without failing
# anything. Directory layout is the durable signal — IA work renames labels.
#
# GEN_SEL selects the reference tab's top-level entries whose every descendant
# path is under reference/general/. Never put it on the left of a `yq -i`
# assignment against a file that may not match: yq auto-creates missing keys in
# lvalue position, which is how the old selector injected `variants: []` and
# invalidated the whole navigation. Reads only; writes go through the
# layout-scoped `|=` in propagate_shared_reference.
GEN_SEL='.navigation[] | select(.tab == "reference") | .layout[] | select([.. | select(has("path")) | .path] | all_c(test("/reference/general/")))'

echo "=== SYNC JOB (replayed) ==="
rm -rf "$WT/fern/pages-dev"; mkdir -p "$WT/fern/pages-dev"
rsync -a --exclude='/home/index.mdx' "$SRC/pages/" "$WT/fern/pages-dev/"
"$PY" "$SRC/scripts/rewrite_snapshot_paths.py" "$WT/fern/pages-dev"
rsync -a --include='*/' --include='backends/*/deploy/**' --exclude='*' --prune-empty-dirs \
  "$REPO_ROOT/examples/" "$WT/examples/"

cp "$SRC/index.yml" "$WT/fern/versions/dev.yml"
cp "$SRC/fern.config.json" "$WT/fern/fern.config.json"
[ -f "$SRC/pages/community/contributing/documentation/building-and-publishing.md" ] && cp "$SRC/pages/community/contributing/documentation/building-and-publishing.md" "$WT/fern/README.md" || true
rm -f "$WT/fern/convert_callouts.py"
mkdir -p "$WT/fern/scripts"
[ -f "$SRC/scripts/convert_callouts.py" ] && cp "$SRC/scripts/convert_callouts.py" "$WT/fern/scripts/convert_callouts.py" || true
rm -rf "$WT/fern/components"; cp -r "$SRC/components" "$WT/fern/components"
rm -rf "$WT/fern/products"
cp "$SRC/pages/home/index.mdx" "$WT/fern/index.mdx"
perl -pi -e 's|\.\./\.\./assets/|./assets/|g' "$WT/fern/index.mdx"
[ -d "$SRC/assets" ] && cp -r "$SRC/assets/." "$WT/fern/assets/" || true
if [ -d "$SRC/pages/blog/_assets" ]; then
  mkdir -p "$WT/fern/digest"; cp -r "$SRC/pages/blog/_assets/." "$WT/fern/digest/"
  perl -pi -e 's|(path: \.\./digest/.*)\.md$|$1.mdx|' "$WT"/fern/versions/v*.yml
fi
[ -f "$SRC/main.css" ] && cp "$SRC/main.css" "$WT/fern/main.css" || true
[ -f "$SRC/custom.js" ] && cp "$SRC/custom.js" "$WT/fern/custom.js" || true

if [ -d "$SRC/translations" ]; then
  for d in "$WT"/fern/translations/*/pages-dev; do
    [ -d "$d" ] || continue
    lang=$(basename "$(dirname "$d")")
    [ -d "$SRC/translations/$lang/pages" ] || rm -rf "$d"
  done
  for lang_dir in "$SRC"/translations/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/pages" ]; then
      rm -rf "$WT/fern/translations/$lang/pages-dev"
      mkdir -p "$WT/fern/translations/$lang"
      cp -r "$lang_dir/pages" "$WT/fern/translations/$lang/pages-dev"
    fi
  done
fi

yq -i '(.. | select(has("path")).path) |= sub("^digest/", "../digest/")' "$WT/fern/versions/dev.yml"
yq -i '(.. | select(has("path")).path) |= sub("^pages/home/index\.mdx$", "../index.mdx")' "$WT/fern/versions/dev.yml"
yq -i '(.. | select(has("path")).path) |= sub("^pages/", "../pages-dev/")' "$WT/fern/versions/dev.yml"

propagate_shared_reference() {
  yq "[$GEN_SEL]" "$WT/fern/versions/dev.yml" > "$WT/.ref_general.yml"
  for vfile in "$WT"/fern/versions/v*.yml; do
    [ -e "$vfile" ] || continue
    # Only rewrite files that already carry shared entries. Versions cut before
    # the tab restructure have no reference tab at all (v1.3.0.yml is
    # `tab: docs`), so this stays inert on every currently-released version.
    if [ "$(yq "[$GEN_SEL] | length" "$vfile")" != "0" ]; then
      ENTRIES="$WT/.ref_general.yml" \
        yq -i "(.navigation[] | select(.tab == \"reference\") | .layout) |=
               (load(strenv(ENTRIES)) + [.[] | select([.. | select(has(\"path\")) | .path] | any_c(test(\"/reference/general/\")) | not)])" "$vfile"
    fi
  done
}
propagate_shared_reference

"$PY" "$WT/fern/scripts/convert_callouts.py" --dir "$WT/fern/pages-dev" >/dev/null
if [ -d "$WT/fern/translations" ]; then
  "$PY" "$WT/fern/scripts/convert_callouts.py" --dir "$WT/fern/translations" >/dev/null
  "$PY" "$SRC/scripts/resolve_translation_links.py" \
    --nav "$SRC/index.yml" --translations-root "$WT/fern/translations" \
    --site-root /dynamo --version-slug dev --pages-dir pages-dev >/dev/null
fi

cd "$WT/fern"
yq '. as $doc | ([$doc.products[]? | select(.display-name == "Docs" or .display-name == "Dynamo")][0].versions // $doc.versions)' \
  docs.yml > "$WT/.preserved_versions.yml"
cp "$SRC/docs.yml" docs.yml
yq -i '."landing-page".path = "./index.mdx"' docs.yml
PRESERVED="$WT/.preserved_versions.yml" \
  yq -i '.versions = load(strenv(PRESERVED))' docs.yml
"$SRC/scripts/ensure_default_version_home.sh" "$WT/fern"

echo "=== RELEASE-VERSION JOB (replayed, tag $TAG) ==="
cd "$WT"
[ -d "fern/pages-$TAG" ] && { echo "ERROR: pages-$TAG already exists on docs-website"; exit 1; }
[ -f "fern/versions/$TAG.yml" ] && { echo "ERROR: versions/$TAG.yml already exists"; exit 1; }

cp -r fern/pages-dev "fern/pages-$TAG"
# The shared group is a directory, so the drop is a plain path operation — no
# nav traversal, and no yq assignment against a file it only means to read.
rm -rf "fern/pages-$TAG/reference/general"
find "fern/pages-$TAG/reference" -type d -empty -delete 2>/dev/null || true

if [ -d "$SRC/translations" ]; then
  for lang_dir in "$SRC"/translations/*/; do
    lang=$(basename "$lang_dir")
    if [ -d "$lang_dir/pages" ]; then
      rm -rf "fern/translations/$lang/pages-$TAG"
      mkdir -p "fern/translations/$lang/pages-$TAG"
      rsync -a "$lang_dir/pages/" "fern/translations/$lang/pages-$TAG/"
    fi
  done
fi

find "fern/pages-$TAG" \( -name "*.md" -o -name "*.mdx" \) -print0 | xargs -0 perl -pi -e \
  "s|github.com/ai-dynamo/dynamo/tree/main|github.com/ai-dynamo/dynamo/tree/$TAG|g; s|github.com/ai-dynamo/dynamo/blob/main|github.com/ai-dynamo/dynamo/blob/$TAG|g"
"$PY" fern/scripts/convert_callouts.py --dir "fern/pages-$TAG" >/dev/null
for d in fern/translations/*/"pages-$TAG"; do
  [ -d "$d" ] && "$PY" "$SRC/scripts/convert_callouts.py" --dir "$d" >/dev/null
done
if compgen -G "fern/translations/*/pages-$TAG" > /dev/null; then
  "$PY" "$SRC/scripts/resolve_translation_links.py" \
    --nav "$SRC/index.yml" --translations-root "fern/translations" \
    --site-root /dynamo --version-slug "$TAG" --pages-dir "pages-$TAG" \
    --github-ref "$TAG" >/dev/null
fi

VERSION_FILE="fern/versions/$TAG.yml"
cp fern/versions/dev.yml "$VERSION_FILE"
perl -pi -e "s|path: \.\./pages-dev/|path: ../pages-$TAG/|g" "$VERSION_FILE"
# Text substitution, not a yq assignment: a yq assignment whose left-hand side
# traverses a missing key auto-creates it, which is how the old `.variants[]`
# form injected `variants: []` and invalidated the navigation.
perl -pi -e "s|path: \.\./pages-$TAG/reference/general/|path: ../pages-dev/reference/general/|g" "$VERSION_FILE"
perl -pi -e "s|href: /dynamo/dev/|href: /dynamo/$TAG/|g" "$VERSION_FILE"

DEV_IDX=$(yq '.versions | to_entries | map(select(.value.display-name == "dev")) | .[0].key' fern/docs.yml)
INSERT_IDX=$((DEV_IDX + 1))
TAG="$TAG" INSERT_IDX="$INSERT_IDX" yq -i '
  .versions |= (
    .[:env(INSERT_IDX)] +
    [{"display-name": env(TAG), "path": ("./versions/" + env(TAG) + ".yml"), "slug": env(TAG), "availability": "stable"}] +
    .[env(INSERT_IDX):]
  )
' fern/docs.yml
yq -i ".versions[0].path = \"./versions/$TAG.yml\"" fern/docs.yml
yq -i ".versions[0].display-name = \"Latest ($TAG)\"" fern/docs.yml

echo "=== ASSERTIONS ==="
default_nav=$(yq -r '.versions[0].path' fern/docs.yml)
default_nav="fern/${default_nav#./}"
default_first_tab=$(yq -r '.navigation[0].tab // ""' "$default_nav")
[ "$default_first_tab" = "home" ] && s11=ok || s11=FAIL
assert "10. default version starts with the shared Home tab" "$s11"

# 9 first: everything below assumes the shared group is findable at all. This is
# the tripwire the old label-keyed selector lacked — it no-opped silently for six
# days after #12410 renamed the nav out from under it.
general=$(yq "[$GEN_SEL] | length" "$WT/fern/versions/dev.yml")
[ "$general" -gt 0 ] && s10=ok || s10=FAIL
assert "9. dev.yml exposes a shared Reference group ($general entries)" "$s10"

shared=$(grep -c "path: \.\./pages-dev/reference/general/" "$VERSION_FILE" || true)
frozen=$(grep -c "path: \.\./pages-$TAG/reference/components/runtime-configuration.mdx" "$VERSION_FILE" || true)
[ "$shared" -ge 10 ] && s1=ok || s1=FAIL
assert "2. $TAG.yml: reference/general shared ($shared pages-dev refs)" "$s1"
[ "$frozen" -eq 1 ] && s2=ok || s2=FAIL
assert "2. $TAG.yml: Components section frozen (runtime configuration)" "$s2"

[ ! -d "fern/pages-$TAG/reference/general" ] && \
  [ -e "fern/pages-$TAG/reference/components/runtime-configuration.mdx" ] && \
  [ -d "fern/pages-$TAG/reference/observability" ] && s3=ok || s3=FAIL
assert "3. snapshot drops shared files, keeps versioned reference/" "$s3"

# The find target must exist, or find errors to stderr, wc counts 0 and the
# assertion reports ok on a failure — which it did once pages-dev/components/
# stopped existing.
if [ -d fern/pages-dev/components ]; then
  [ "$(find fern/pages-dev/components -name '*.tsx' | wc -l | tr -d ' ')" = "0" ] && s4=ok || s4=FAIL
  assert "4. no .tsx in pages-dev/components/" "$s4"
else
  note "4. no .tsx in pages-dev/components/" "n/a (no components/ page dir)"
fi

prework=$(git diff -U0 -- 'fern/versions/v*.yml' 2>/dev/null \
  | grep -E '^[+-][^+-]' | grep -c "pages-dev/reference" || true)
[ "$prework" -eq 0 ] && s5=ok || s5=FAIL
assert "5. pre-rework versions gain no shared-reference pointers" "$s5"

# Round two: a later main push adds a page to the shared reference; the
# already-cut version's nav must pick it up via propagation. Append to the
# reference tab's layout rather than descending into a named section, so the
# test does not re-couple itself to a nav label.
FAKE='{"page": "Sim Test Page", "path": "../pages-dev/reference/general/sim-test-page.mdx", "slug": "sim-test-page"}'
FAKE="$FAKE" yq -i '(.navigation[] | select(.tab == "reference") | .layout) += [env(FAKE)]' \
  "$WT/fern/versions/dev.yml"
propagate_shared_reference
grep -q "sim-test-page" "$VERSION_FILE" && s6=ok || s6=FAIL
assert "6. round-two propagation reaches the cut version's nav" "$s6"
# Undo before fern check (the fake page has no backing file).
perl -ni -e 'print unless /sim-test-page|Sim Test Page/' "$WT/fern/versions/dev.yml" "$VERSION_FILE"

[ -f "$WT/fern/scripts/convert_callouts.py" ] && \
  [ ! -e "$WT/fern/convert_callouts.py" ] && s8=ok || s8=FAIL
assert "7. converter moved to fern/scripts/ with no stale root copy" "$s8"

read -r dev_relative dev_versioned tag_relative tag_dev_leaks tag_versioned < <(
  "$PY" - "$WT/fern/translations" "$TAG" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
tag = sys.argv[2]
link = re.compile(r"!?\[[^]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")

def counts(trees, version: str):
    relative = versioned = dev = 0
    for tree in trees:
        if not tree.exists():
            continue
        for page in tree.rglob("*"):
            if page.suffix not in {".md", ".mdx"}:
                continue
            text = page.read_text(encoding="utf-8")
            for target in link.findall(text):
                if target.endswith((".md", ".mdx")) and not target.startswith(
                    ("http://", "https://", "mailto:", "/")
                ):
                    relative += 1
            dev += len(re.findall(r"/dynamo/(?:[^/]+/)?dev/", text))
            versioned += len(
                re.findall(rf"/dynamo/(?:[^/]+/)?{re.escape(version)}/", text)
            )
    return relative, dev, versioned


dev_rel, _, dev_urls = counts(root.glob("*/pages-dev"), "dev")
tag_rel, tag_dev, tag_urls = counts(root.glob(f"*/pages-{tag}"), tag)
print(dev_rel, dev_urls, tag_rel, tag_dev, tag_urls)
PY
)
printf '%s\n' \
  "translation link counts: dev_relative=$dev_relative dev_urls=$dev_versioned" \
  "tag_relative=$tag_relative tag_dev_leaks=$tag_dev_leaks tag_urls=$tag_versioned"
[ "$dev_relative" -eq 0 ] && [ "$dev_versioned" -gt 0 ] && \
  [ "$tag_relative" -eq 0 ] && [ "$tag_dev_leaks" -eq 0 ] && \
  [ "$tag_versioned" -gt 0 ] && s9=ok || s9=FAIL
assert "8. translated links resolve to dev and tag-pinned site URLs" "$s9"

if command -v fern >/dev/null 2>&1; then
  fern_out=$(cd "$WT/fern" && fern check 2>&1 || true)
  errors=$(printf '%s\n' "$fern_out" | grep -oE "Found [0-9]+ error" | tail -1 | grep -oE "[0-9]+" || true)
  errors=${errors:-unknown}
  [ "$errors" = "0" ] && s7=ok || s7=FAIL
  assert "1. fern check on composed tree ($errors errors)" "$s7"
  if [ "$s7" = "FAIL" ]; then
    printf '%s\n' "$fern_out"
  fi
else
  note "1. fern check" "SKIPPED (fern CLI not installed)"
fi

echo ""
if [ "$fail" -eq 0 ]; then echo "ALL ASSERTIONS PASSED"; else echo "ASSERTIONS FAILED"; exit 1; fi
