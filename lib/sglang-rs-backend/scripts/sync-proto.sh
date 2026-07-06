#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch Shepherd Model Gateway's SGLang scheduler proto. build.rs runs this
# when the generated proto files are missing. Re-run manually to pick up
# upstream changes — the resolved commit SHA is captured in the generated proto
# header so the version is always traceable from the files.
#
#   ./scripts/sync-proto.sh [<ref>]          # default: v1.5.0
#   SMG_REPO=https://... ./scripts/sync-proto.sh <ref>

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CRATE_ROOT="$(readlink -f "$SCRIPT_DIR/..")"
PROTO_DIR="$CRATE_ROOT/proto"

SMG_REPO="${SMG_REPO:-https://github.com/lightseekorg/smg.git}"
SMG_REF="${1:-v1.5.0}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Skip any user-global `insteadOf` rewrites (e.g. https→ssh).
export GIT_CONFIG_GLOBAL=/dev/null

echo "Cloning $SMG_REPO @ $SMG_REF ..."
if ! git clone --depth=1 --branch "$SMG_REF" "$SMG_REPO" "$TMP/smg" 2>/dev/null; then
    # --branch rejects SHA refs; fall back to a full clone.
    git clone "$SMG_REPO" "$TMP/smg"
    git -C "$TMP/smg" checkout "$SMG_REF"
fi

SRC_DIR="$TMP/smg/crates/grpc_client/proto"
SHA=$(git -C "$TMP/smg" rev-parse HEAD)
SHORT_SHA="${SHA:0:12}"

for name in common.proto sglang_scheduler.proto; do
    SRC="$SRC_DIR/$name"
    DST="$PROTO_DIR/$name"
    if [ ! -f "$SRC" ]; then
        echo "ERROR: $SRC not found in $SMG_REPO @ $SMG_REF" >&2
        exit 1
    fi
    {
        echo "// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Shepherd Model Gateway contributors."
        echo "// SPDX-License-Identifier: Apache-2.0"
        echo "//"
        echo "// Generated from $SMG_REPO"
        echo "// Commit:        $SHA"
        echo "// Refresh with:  lib/sglang-rs-backend/scripts/sync-proto.sh [ref]"
        echo
        awk 'NR <= 3 && /^(\/\/|$)/ { next } { print }' "$SRC"
    } > "$DST"
    echo "Wrote $DST"
done

echo "Source: $SMG_REPO @ $SHORT_SHA"
