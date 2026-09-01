#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prove gen_llms_tables.py is reproducible: two runs under deliberately hostile
# environments must produce byte-identical output, that output must match the
# committed bytes, and it must be LF-only.
#
# Lived in docs-link-check.yml until it was moved here. It has nothing to do
# with link checking, and as a script it also runs locally, which the inline
# workflow step never could.
#
# Usage: docs/fern/scripts/check_llms_tables_determinism.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

python3 docs/fern/scripts/gen_llms_tables.py --check

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/first" "$tmp_dir/second"
cp -R docs "$tmp_dir/first/docs"
cp -R docs "$tmp_dir/second/docs"

outputs=(
  docs/fern/pages/reference/general/compatibility.mdx
  docs/fern/pages/reference/general/release-artifacts.mdx
  docs/fern/pages/reference/general/model-early-access-builds.mdx
  docs/fern/pages/reference/general/releases-machine-readable.mdx
  docs/fern/pages/reference/general/releases/release-history.mdx
  docs/fern/assets/releases.json
  docs/fern/assets/releases-atom.xml
)

# Force both runs to actually WRITE every output: drop the whole-file assets and
# CRLF-corrupt the spliced pages. A copied-through current tree would skip every
# write on the unchanged fast-path, leaving cmp comparing the originals with
# themselves and proving nothing about how the generator writes.
#
# The CRLF step goes through python3 rather than `sed -i`: GNU sed takes no
# argument after -i and BSD sed requires one, so the inline `sed -i 's/$/\r/'`
# this script came from fails on macOS. This script is meant to run locally.
for tree in first second; do
  rm -f "$tmp_dir/$tree/docs/fern/assets/releases.json" \
        "$tmp_dir/$tree/docs/fern/assets/releases-atom.xml"
  # One interpreter for the whole tree rather than one per file.
  python3 - "$tmp_dir/$tree" "${outputs[@]}" <<'CRLF_PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for rel in sys.argv[2:]:
    if not rel.endswith(".mdx"):
        continue
    path = root / rel
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    path.write_bytes(data)
CRLF_PY
done

TZ=UTC LC_ALL=C PYTHONHASHSEED=1 SOURCE_DATE_EPOCH=0 \
  python3 "$tmp_dir/first/docs/fern/scripts/gen_llms_tables.py"
TZ=Pacific/Honolulu LC_ALL=C.UTF-8 PYTHONHASHSEED=2 SOURCE_DATE_EPOCH=4102444800 \
  python3 "$tmp_dir/second/docs/fern/scripts/gen_llms_tables.py"

failed=0
for output in "${outputs[@]}"; do
  if ! cmp -s "$tmp_dir/first/$output" "$tmp_dir/second/$output"; then
    echo "::error file=$output::Generator output is not byte-identical across runs"
    diff -u "$tmp_dir/first/$output" "$tmp_dir/second/$output" || true
    failed=1
  fi
  # Committed-bytes comparison only where the output is committed (the two
  # assets stop being tracked once publish-time generation lands; temp-vs-temp
  # and the CR scan still cover them).
  if [ -f "$output" ] && ! cmp -s "$tmp_dir/first/$output" "$output"; then
    echo "::error file=$output::Regenerated output does not match the committed bytes"
    failed=1
  fi
  if LC_ALL=C grep -q $'\r' "$tmp_dir/first/$output"; then
    echo "::error file=$output::Generator output contains CR bytes; outputs must be LF-only"
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
echo "gen_llms_tables.py output is deterministic, current, and LF-only."
