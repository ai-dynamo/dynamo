#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# check_reference.sh — one-command gate for the Reference pages.
#
# Run after every release bump (see the PER-RELEASE BUMP CHECKLIST in
# components/releases.data.ts) and before opening a docs PR:
#
#   ./scripts/check_reference.sh
#
# Checks, in order:
#   1. Agent twins are fresh (gen_llms_tables.py --check) — also proves
#      releases.data.ts still satisfies the generator's parser contract.
#   2. custom.js parses (node --check).
#   3. No stale reference/(support|feature)-matrix repo links outside the
#      allowed remnants.
#   4. Fern broken-links contains zero errors inside reference/ pages
#      (skipped with a warning if the fern CLI is unavailable).
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

echo "== 1/4 agent twins fresh =="
python3 scripts/gen_llms_tables.py --check || fail=1

echo "== 2/4 custom.js parses =="
node --check custom.js || fail=1

echo "== 3/4 no stale matrix links =="
stale=$(grep -rnE "reference/(support|feature)-matrix" --include="*.md" --include="*.mdx" . \
  | grep -vE "documentation-style-guide|^\./README\.md" || true)
if [[ -n "$stale" ]]; then
  echo "$stale"
  echo "stale support-matrix/feature-matrix links found"
  fail=1
else
  echo "clean"
fi

echo "== 4/4 fern broken-links (reference/ scope) =="
if command -v fern >/dev/null 2>&1; then
  out=$(fern docs broken-links 2>&1 || true)
  scoped=$(echo "$out" | grep -cE "fix here: reference/" || true)
  total=$(echo "$out" | grep -c "\[error\]" || true)
  echo "total site errors: ${total} (pre-existing baseline elsewhere); in reference/: ${scoped}"
  if [[ "${scoped}" != "0" ]]; then
    echo "$out" | grep -B2 "fix here: reference/" | head -30
    fail=1
  fi
else
  echo "WARNING: fern CLI not found — broken-links check skipped"
fi

if [[ "$fail" != "0" ]]; then
  echo "CHECK FAILED"
  exit 1
fi
echo "ALL CHECKS PASSED"
