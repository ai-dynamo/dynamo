#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EXPERIMENT ONLY (ledger D7): unpack the pinned aisimulate-core crate, verify it against
# Cargo.lock, apply the SGLang-cadence patch, and print the cargo argument that builds against
# the patched copy. The default build keeps the published crate (legacy mocker cadence).
#
#   lib/e2e-indexer-tools/sglang-cadence-mocker/prepare.sh <out-dir>
#   cargo bench --no-run ... "$(prepare.sh <out-dir>)"   # prints --config=patch...
#
# cargo rewrites Cargo.lock for a patched build; restore it afterwards
# (git checkout -- Cargo.lock) and keep the patched build in its own CARGO_TARGET_DIR.
set -euo pipefail

OUT=${1:?usage: prepare.sh <out-dir>}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PATCH=$(cd "$(dirname "$0")" && pwd)/aisimulate-core-sglang-cadence.patch
CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}

lock_field() {
  awk -v key="$1" '
    $0 == "name = \"aisimulate-core\"" { found = 1; next }
    found && $1 == key { gsub(/"/, "", $3); print $3; exit }
    found && /^$/ { exit }' "$ROOT/Cargo.lock"
}
VERSION=$(lock_field version)
CHECKSUM=$(lock_field checksum)
[ -n "$VERSION" ] && [ -n "$CHECKSUM" ] || { echo "aisimulate-core not found in Cargo.lock" >&2; exit 1; }
[ "$VERSION" = "0.13.0-dev.202609300000000061" ] ||
  { echo "the patch targets aisimulate-core 0.13.0-dev.202609300000000061, Cargo.lock pins $VERSION" >&2; exit 1; }

find_crate() {
  find "$CARGO_HOME/registry/cache" -name "aisimulate-core-$VERSION.crate" 2>/dev/null | head -1
}
CRATE=$(find_crate)
if [ -z "$CRATE" ]; then
  cargo fetch --manifest-path "$ROOT/Cargo.toml" --locked >&2
  CRATE=$(find_crate)
fi
[ -n "$CRATE" ] || { echo "aisimulate-core-$VERSION.crate is not in $CARGO_HOME/registry/cache" >&2; exit 1; }
[ "$(sha256sum "$CRATE" | cut -d' ' -f1)" = "$CHECKSUM" ] ||
  { echo "$CRATE does not match the Cargo.lock checksum" >&2; exit 1; }

DEST="$OUT/aisimulate-core-$VERSION"
rm -rf "$DEST"
mkdir -p "$OUT"
tar -xzf "$CRATE" -C "$OUT"
# A ceiling keeps git apply from resolving paths against an enclosing repository.
(cd "$DEST" && GIT_CEILING_DIRECTORIES=$(cd "$OUT" && pwd) git apply --whitespace=nowarn "$PATCH")
echo "patched $DEST" >&2
echo "--config=patch.crates-io.aisimulate-core.path=\"$(cd "$DEST" && pwd)\""
