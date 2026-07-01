#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Local dry-run of the release-version job in .github/workflows/fern-docs.yml.
#
# The script builds and validates a versioned snapshot in temporary worktrees.
# It stops before the workflow's commit, push, and publish steps.
#
# Usage:
#   fern/release_dryrun.sh [TAG]          # default: v1.2.1
#   fern/release_dryrun.sh v1.2.1
#   KEEP=1 fern/release_dryrun.sh v1.2.1  # keep worktrees
#
# Requires: git, fern, yq, jq, rsync, and Python 3.10 or newer. Set PYTHON to
# choose a non-default interpreter, for example PYTHON=.venv/bin/python.

set -euo pipefail

TAG="${1:-v1.2.1}"
DOCS_WEBSITE_REF="${DOCS_WEBSITE_REF:-origin/docs-website}"
PYTHON="${PYTHON:-python3}"

if ! echo "$TAG" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "error: invalid tag '$TAG' (must be vX.Y.Z, for example v1.2.1)" >&2
  exit 2
fi

for tool in git fern yq jq rsync; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    hint=""
    [ "$tool" = "yq" ] && hint=" (install with: brew install yq)"
    echo "error: required tool '$tool' not found$hint" >&2
    exit 2
  fi
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: Python interpreter '$PYTHON' not found" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "error: $PYTHON must be Python 3.10 or newer (set PYTHON to another interpreter)" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "error: tag '$TAG' not found locally. Try: git fetch --tags" >&2
  exit 2
fi

echo "Fetching origin/docs-website ..."
git fetch origin docs-website >/dev/null 2>&1 || \
  echo "warning: could not fetch docs-website; using the local ref if present" >&2
git rev-parse -q --verify "$DOCS_WEBSITE_REF" >/dev/null || {
  echo "error: ref '$DOCS_WEBSITE_REF' not found" >&2
  exit 2
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fern-release-dryrun.XXXXXX")"
SOURCE_CHECKOUT="$WORK/source-checkout"
DOCS_CHECKOUT="$WORK/docs-checkout"

cleanup() {
  if [ "${KEEP:-0}" = "1" ]; then
    echo "KEEP=1 set; leaving worktrees at $WORK"
    return
  fi
  git worktree remove --force "$SOURCE_CHECKOUT" 2>/dev/null || true
  git worktree remove --force "$DOCS_CHECKOUT" 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true
  git worktree prune 2>/dev/null || true
}
trap cleanup EXIT

echo "Creating worktrees under $WORK ..."
git worktree add --detach "$SOURCE_CHECKOUT" "$TAG" >/dev/null
git worktree add --detach "$DOCS_CHECKOUT" "$DOCS_WEBSITE_REF" >/dev/null

if [ -d "$DOCS_CHECKOUT/fern/pages-$TAG" ] || [ -f "$DOCS_CHECKOUT/fern/versions/$TAG.yml" ]; then
  echo "note: $TAG already exists on docs-website; the workflow requires force_rebuild=true."
  echo "      The dry-run rebuilds it for validation without publishing."
fi

echo "Building fern/pages-$TAG/ from source @ $TAG docs/ ..."
rm -rf "$DOCS_CHECKOUT/fern/pages-$TAG"
mkdir -p "$DOCS_CHECKOUT/fern/pages-$TAG"
rsync -a \
  --exclude='digest' \
  --exclude='index.yml' \
  "$SOURCE_CHECKOUT/docs/" "$DOCS_CHECKOUT/fern/pages-$TAG/"

echo "Pinning tree/main and blob/main links to $TAG ..."
find "$DOCS_CHECKOUT/fern/pages-$TAG" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
  if grep -q "github.com/ai-dynamo/dynamo/tree/main" "$file"; then
    sed -i.bak "s|github.com/ai-dynamo/dynamo/tree/main|github.com/ai-dynamo/dynamo/tree/$TAG|g" "$file"
  fi
done
find "$DOCS_CHECKOUT/fern/pages-$TAG" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
  if grep -q "github.com/ai-dynamo/dynamo/blob/main" "$file"; then
    sed -i.bak "s|github.com/ai-dynamo/dynamo/blob/main|github.com/ai-dynamo/dynamo/blob/$TAG|g" "$file"
  fi
done
find "$DOCS_CHECKOUT/fern/pages-$TAG" -name "*.bak" -delete

echo "Converting callouts with the tag's convert_callouts.py ..."
"$PYTHON" "$SOURCE_CHECKOUT/fern/convert_callouts.py" --dir "$DOCS_CHECKOUT/fern/pages-$TAG" >/dev/null

echo "Building fern/versions/$TAG.yml from the tag's index.yml ..."
VERSION_FILE="$DOCS_CHECKOUT/fern/versions/$TAG.yml"
cp "$SOURCE_CHECKOUT/docs/index.yml" "$VERSION_FILE"
yq -i '(.. | select(has("path")).path) |= sub("^digest/", "../digest/")' "$VERSION_FILE"
yq -i '(.. | select(has("path")).path) |= sub("^([a-zA-Z])", "../pages-'"$TAG"'/${1}")' "$VERSION_FILE"

DOCS_FILE="$DOCS_CHECKOUT/fern/docs.yml"
if yq -e ".products[0].versions[] | select(.\"display-name\" == \"$TAG\")" "$DOCS_FILE" >/dev/null 2>&1; then
  echo "docs.yml already lists $TAG; leaving the version list unchanged."
else
  echo "Inserting $TAG into docs.yml ..."
  DEV_IDX=$(yq '.products[0].versions | to_entries | map(select(.value."display-name" == "dev")) | .[0].key' "$DOCS_FILE")
  INSERT_IDX=$((DEV_IDX + 1))
  yq -i "
    .products[0].versions |= (
      .[:$INSERT_IDX] +
      [{\"display-name\": \"$TAG\", \"path\": \"./versions/$TAG.yml\", \"slug\": \"$TAG\", \"availability\": \"stable\"}] +
      .[$INSERT_IDX:]
    )
  " "$DOCS_FILE"
  yq -i ".products[0].path = \"./versions/$TAG.yml\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].path = \"./versions/$TAG.yml\"" "$DOCS_FILE"
  yq -i ".products[0].versions[0].\"display-name\" = \"Latest ($TAG)\"" "$DOCS_FILE"
  yq -i '.products[0].versions[0].slug = "latest"' "$DOCS_FILE"
  yq -i '.products[0].versions[0].availability = "stable"' "$DOCS_FILE"
fi

WANT_FERN="$(jq -r '.version' "$DOCS_CHECKOUT/fern/fern.config.json")"
HAVE_FERN="$(
  "$PYTHON" -c \
    'import json, pathlib, sys; package = pathlib.Path(sys.argv[1]).resolve().parent / "package.json"; print(json.loads(package.read_text())["version"])' \
    "$(command -v fern)" 2>/dev/null || echo "unknown"
)"
if [ "$WANT_FERN" != "$HAVE_FERN" ]; then
  echo "note: local fern-api is $HAVE_FERN but docs-website pins $WANT_FERN."
  echo "      Install the pinned version with: npm install -g fern-api@$WANT_FERN"
fi

echo "Verifying the snapshot file inventory ..."
diff -u \
  <(cd "$SOURCE_CHECKOUT/docs" && find . -type f ! -path './digest/*' ! -name 'index.yml' -print | sort) \
  <(cd "$DOCS_CHECKOUT/fern/pages-$TAG" && find . -type f -print | sort)

echo "Verifying version navigation targets ..."
while IFS= read -r path; do
  case "$path" in
    ../pages-$TAG/*|../digest/*)
      if [ ! -e "$DOCS_CHECKOUT/fern/versions/$path" ]; then
        echo "error: version navigation target does not exist: $path" >&2
        exit 1
      fi
      ;;
  esac
done < <(yq -r '(.. | select(has("path")).path)' "$VERSION_FILE")

echo "Running Fern configuration validation in $DOCS_CHECKOUT ..."
cd "$DOCS_CHECKOUT"
fern check

echo "DRY RUN OK: $TAG was built from the tag and passed release validation."
echo "Skipped: git push origin docs-website; fern generate --docs."
