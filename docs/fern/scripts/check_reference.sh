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
#   4. Every absolute /dynamo/dev/reference/... href in components, the data
#      module, generated assets, and reference pages resolves to a URL the
#      index.yml Reference General variant actually publishes. Catches nav
#      restructures (e.g. pages moving under a new section slug) that
#      fern broken-links cannot see because the hrefs live in TSX/JSON.
#   5. Fern broken-links contains zero errors inside reference/ pages
#      (skipped with a warning if the fern CLI is unavailable).
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

echo "== 1/5 agent twins fresh =="
python3 scripts/gen_llms_tables.py --check || fail=1

echo "== 2/5 custom.js parses =="
node --check custom.js || fail=1

echo "== 3/5 no stale matrix links =="
stale=$(grep -rnE "reference/(support|feature)-matrix" --include="*.md" --include="*.mdx" . \
  | grep -vE "documentation-style-guide|^\./README\.md" || true)
if [[ -n "$stale" ]]; then
  echo "$stale"
  echo "stale support-matrix/feature-matrix links found"
  fail=1
else
  echo "clean"
fi

echo "== 4/5 absolute reference hrefs match the nav =="
python3 - <<'PY' || fail=1
"""Validate /dynamo/dev/reference/... hrefs against index.yml-derived URLs."""
import pathlib
import re
import sys

import yaml

nav = yaml.safe_load(pathlib.Path("index.yml").read_text())


def kebab(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def collect(items, prefix, urls):
    for item in items:
        if "page" in item:
            slug = item.get("slug") or kebab(item["page"])
            urls.add(f"{prefix}/{slug}")
        elif "section" in item:
            slug = item.get("slug") or kebab(item["section"])
            collect(item.get("contents", []), f"{prefix}/{slug}", urls)


general = next(
    variant
    for tab in nav["navigation"]
    if tab.get("tab") == "reference"
    for variant in tab.get("variants", [])
    if variant.get("title") == "General"
)
valid: set[str] = set()
collect(general["layout"], "/dynamo/dev/reference", valid)

href_re = re.compile(r"/dynamo/dev/reference/[a-z0-9/-]+")
sources = [
    *pathlib.Path("components").glob("*.tsx"),
    pathlib.Path("components/releases.data.ts"),
    pathlib.Path("scripts/gen_llms_tables.py"),
    *pathlib.Path("reference").rglob("*.mdx"),
    pathlib.Path("assets/releases.json"),
    pathlib.Path("assets/releases-atom.xml"),
]
bad = []
for source in sources:
    if not source.exists():
        continue
    for lineno, line in enumerate(source.read_text().splitlines(), 1):
        for href in href_re.findall(line):
            if href.rstrip("/") not in valid:
                bad.append(f"{source}:{lineno}: {href}")

if bad:
    print("\n".join(bad))
    print(f"{len(bad)} absolute reference href(s) do not match any published URL")
    sys.exit(1)
print(f"clean ({len(valid)} published reference URLs)")
PY

echo "== 5/5 fern broken-links (reference/ scope) =="
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
