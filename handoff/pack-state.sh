#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${DYNAMO_HANDOFF_OUT:-$ROOT/handoff}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="$(mktemp -d -t dynamo-prod-k8s-bundle.XXXXXX)"
trap 'rm -rf "$BUNDLE_DIR"' EXIT

mkdir -p "$BUNDLE_DIR/docs" "$BUNDLE_DIR/inventory"

cd "$ROOT"

cat >"$BUNDLE_DIR/STATE.txt" <<EOF
Dynamo production Kubernetes handoff bundle
Created: $(date -Iseconds)
Source: $(hostname) ($(uname -m))

Repository:
  root: $ROOT
  head: $(git rev-parse HEAD)
  branch: $(git branch --show-current)
  status: $(git status --short | wc -l | tr -d ' ') changed paths

Remotes:
$(git remote -v)

Latest refs:
  origin/main: $(git ls-remote https://github.com/ai-blaise/dynamo-prod-k8s.git refs/heads/main | awk '{print $1}')
  upstream/main: $(git ls-remote https://github.com/ai-dynamo/dynamo.git refs/heads/main | awk '{print $1}')
EOF

cp -a handoff/MANIFEST.md "$BUNDLE_DIR/MANIFEST.md"
cp -a AGENTS.md "$BUNDLE_DIR/docs/AGENTS.md"
cp -a deploy/production/README.md "$BUNDLE_DIR/docs/deploy-production-README.md"
cp -a deploy/production/docs/smg-integration.md "$BUNDLE_DIR/docs/smg-integration.md"
cp -a docs/components/frontend/Tokenizer.md "$BUNDLE_DIR/docs/frontend-Tokenizer.md"
cp -a docs/features/tokenizer/README.md "$BUNDLE_DIR/docs/feature-tokenizer-README.md"

find deploy/production -maxdepth 3 -type f | sort >"$BUNDLE_DIR/inventory/deploy-production-files.txt"
find deploy/production/addons -maxdepth 2 -name README.md | sort >"$BUNDLE_DIR/inventory/addon-readmes.txt"

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/dynamo-prod-k8s-state-bundle-$STAMP.tar.gz"
tar czf "$OUT" -C "$BUNDLE_DIR" .

echo "Bundle: $OUT"
du -h "$OUT" | awk '{print "Size: " $1}'
