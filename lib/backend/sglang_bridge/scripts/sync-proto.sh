#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Refresh the vendored `sglang.runtime.v1` proto from a SGLang checkout.
#
# Vendored output lives at `proto/sglang/runtime/v1/sglang.proto` and is
# committed to the tree so `cargo build` works without network access.
# Run this script when picking up upstream proto changes.
#
# Default source: ishandhanani/sglang (the dynamo team's working fork that
# carries the `DisaggregatedParams` + `SubscribeKvEvents` additions on top
# of upstream). Once those land in `sgl-project/sglang`, point this at
# upstream and pin a tag.
#
# Usage:
#   ./scripts/sync-proto.sh                  # default branch
#   ./scripts/sync-proto.sh <ref>            # any branch, tag, or SHA
#   SGLANG_REPO=https://... ./scripts/sync-proto.sh
#
# After running, inspect the diff, build the crate, and commit:
#   git diff -- lib/backend/sglang_bridge/proto/
#   cargo build -p dynamo-sglang-bridge
#   git add lib/backend/sglang_bridge/proto/ && git commit

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CRATE_ROOT="$(readlink -f "$SCRIPT_DIR/..")"
PROTO_DST="$CRATE_ROOT/proto/sglang/runtime/v1/sglang.proto"

SGLANG_REPO="${SGLANG_REPO:-https://github.com/ishandhanani/sglang.git}"
SGLANG_REF="${1:-idhanani/alexnails-on-main}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Bypass any user-global `insteadOf` rewrites (e.g. https→ssh) that would
# otherwise turn a plain `https://github.com/...` URL into an SSH clone
# requiring keys we may not have in CI.
export GIT_CONFIG_GLOBAL=/dev/null

echo "Cloning $SGLANG_REPO @ $SGLANG_REF ..."
if ! git clone --depth=1 --branch "$SGLANG_REF" "$SGLANG_REPO" "$TMP/sglang" 2>/dev/null; then
    # `--branch` only accepts branches and tags; fall back to a full clone +
    # checkout so SHA refs work too.
    git clone "$SGLANG_REPO" "$TMP/sglang"
    git -C "$TMP/sglang" checkout "$SGLANG_REF"
fi

SRC="$TMP/sglang/proto/sglang/runtime/v1/sglang.proto"
if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found in $SGLANG_REPO @ $SGLANG_REF" >&2
    exit 1
fi

SHA=$(git -C "$TMP/sglang" rev-parse HEAD)
SHORT_SHA="${SHA:0:12}"

mkdir -p "$(dirname "$PROTO_DST")"
{
    echo "// SPDX-FileCopyrightText: Copyright (c) 2025-2026 SGLang contributors."
    echo "// SPDX-License-Identifier: Apache-2.0"
    echo "//"
    echo "// Vendored from $SGLANG_REPO"
    echo "// Commit:        $SHA"
    echo "// Refresh with:  lib/backend/sglang_bridge/scripts/sync-proto.sh [ref]"
    echo
    # Drop the upstream SPDX header (first two lines + blank) since we
    # already stamped one above; keep everything else verbatim.
    awk 'NR <= 3 && /^(\/\/|$)/ { next } { print }' "$SRC"
} > "$PROTO_DST"

echo "Wrote $PROTO_DST"
echo "Source: $SGLANG_REPO @ $SHORT_SHA"
