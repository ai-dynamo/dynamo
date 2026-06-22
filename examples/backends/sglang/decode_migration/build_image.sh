#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-dynamo-sglang-decode-migration:dev}
DYNAMO_ROOT=${DYNAMO_ROOT:-$(cd "$HERE/../../../.." && pwd)}
BUILD_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

command -v maturin >/dev/null || {
    echo "maturin is required to build the matching Dynamo runtime wheel" >&2
    exit 1
}
command -v protoc >/dev/null || {
    echo "protoc is required to build the matching Dynamo runtime wheel" >&2
    exit 1
}

maturin build \
    --release \
    --manifest-path "$DYNAMO_ROOT/lib/bindings/python/Cargo.toml" \
    --out "$BUILD_DIR"
cp "$HERE/Dockerfile" "$BUILD_DIR/Dockerfile"
docker build -t "$IMAGE" -f "$BUILD_DIR/Dockerfile" "$BUILD_DIR"
