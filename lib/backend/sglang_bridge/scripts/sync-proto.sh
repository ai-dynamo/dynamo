#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch `sglang.runtime.v1` proto into proto/sglang/runtime/v1/. Invoked
# automatically by build.rs when the proto is missing; can also be run
# manually to pick up upstream proto changes.
#
# Default source carries our pending `DisaggregatedParams` /
# `SubscribeKvEvents` additions. Switch to sgl-project/sglang + a pinned
# tag once those land.
#
#   ./scripts/sync-proto.sh [<ref>]          # default: idhanani/alexnails-on-main
#   SGLANG_REPO=https://... ./scripts/sync-proto.sh <ref>

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CRATE_ROOT="$(readlink -f "$SCRIPT_DIR/..")"
PROTO_DST="$CRATE_ROOT/proto/sglang/runtime/v1/sglang.proto"

SGLANG_REPO="${SGLANG_REPO:-https://github.com/ishandhanani/sglang.git}"
SGLANG_REF="${1:-idhanani/alexnails-on-main}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Skip any user-global `insteadOf` rewrites (e.g. https→ssh).
export GIT_CONFIG_GLOBAL=/dev/null

echo "Cloning $SGLANG_REPO @ $SGLANG_REF ..."
if ! git clone --depth=1 --branch "$SGLANG_REF" "$SGLANG_REPO" "$TMP/sglang" 2>/dev/null; then
    # --branch rejects SHA refs; fall back to a full clone.
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
    # Drop upstream's SPDX header (first 3 lines if comment/blank) since
    # we stamp one above.
    awk 'NR <= 3 && /^(\/\/|$)/ { next } { print }' "$SRC"
} > "$PROTO_DST"

echo "Wrote $PROTO_DST"
echo "Source: $SGLANG_REPO @ $SHORT_SHA"
