#!/usr/bin/env bash

set -euo pipefail

BUNDLE="${1:-}"
DEST="${DYNAMO_HANDOFF_DEST:-$HOME/Documents/Codex/dynamo-prod-k8s}"
ORIGIN_URL="${DYNAMO_HANDOFF_ORIGIN:-git@github.com:ai-blaise/dynamo-prod-k8s.git}"
UPSTREAM_URL="${DYNAMO_HANDOFF_UPSTREAM:-git@github.com:ai-dynamo/dynamo.git}"

echo "==Preflight checks=="
for cmd in git curl python3; do
  command -v "$cmd" >/dev/null || {
    echo "ERROR: missing required command: $cmd" >&2
    exit 1
  }
done

if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || echo "WARNING: gh is installed but not authenticated"
else
  echo "WARNING: gh is not installed"
fi

if command -v gcloud >/dev/null 2>&1; then
  gcloud config get-value project >/dev/null 2>&1 || echo "WARNING: gcloud has no default project"
else
  echo "WARNING: gcloud is not installed"
fi

if [ -n "$BUNDLE" ]; then
  echo
  echo "==Reading bundle metadata=="
  WORK="$(mktemp -d -t dynamo-prod-k8s-restore.XXXXXX)"
  trap 'rm -rf "$WORK"' EXIT
  tar xzf "$BUNDLE" -C "$WORK"
  cat "$WORK/STATE.txt" 2>/dev/null || true
fi

echo
echo "==Cloning or updating repo=="
mkdir -p "$(dirname "$DEST")"
if [ -d "$DEST/.git" ]; then
  cd "$DEST"
  git fetch origin main
else
  git clone "$ORIGIN_URL" "$DEST"
  cd "$DEST"
fi

git checkout main
git pull --ff-only origin main

if git remote get-url upstream >/dev/null 2>&1; then
  git remote set-url upstream "$UPSTREAM_URL"
else
  git remote add upstream "$UPSTREAM_URL"
fi
git fetch upstream main

echo
echo "==Repository state=="
echo "origin/main:   $(git rev-parse origin/main)"
echo "upstream/main: $(git rev-parse upstream/main)"
echo "HEAD:          $(git rev-parse HEAD)"

echo
echo "==Fork contract checks=="
test -d deploy/production
test -f deploy/production/docs/smg-integration.md
test -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
grep -q 'default="fastokens"' components/src/dynamo/frontend/frontend_args.py
grep -q 'DYN_TOKENIZER' components/src/dynamo/frontend/main.py
grep -q 'FastOkenS is the default' handoff/MANIFEST.md
echo "deploy/production files: $(find deploy/production -type f | wc -l | tr -d ' ')"
echo "addon docs: $(find deploy/production/addons -maxdepth 2 -name README.md | wc -l | tr -d ' ')"

cat <<'POST'

==================================================
HANDOFF READY
==================================================
Read these before changing code:
  AGENTS.md
  handoff/MANIFEST.md
  deploy/production/README.md
  deploy/production/docs/smg-integration.md
  docs/components/frontend/Tokenizer.md
  docs/features/tokenizer/README.md

Before merge work:
  git ls-remote https://github.com/ai-blaise/dynamo-prod-k8s.git refs/heads/main
  git ls-remote https://github.com/ai-dynamo/dynamo.git refs/heads/main

Deployment and production verification target:
  gcloud compute ssh instance-20260415-161450 --zone asia-south1-b

Core verification:
  PYTHONPATH=components/src python -m pytest \
    tests/deploy/test_deepseek_reap_manifest_contract.py \
    tests/deploy/test_smg_boundary_contract.py
  kubectl kustomize deploy/production/gitops >/tmp/dynamo-prod-k8s-kustomize.yaml
  cargo fmt --check --package dynamo-tokenizers --package dynamo-llm --package dynamo-kv-router
  cargo test -p dynamo-tokenizers fastokens
POST
