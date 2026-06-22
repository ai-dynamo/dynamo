#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-dynamo-sglang-decode-migration:dev}
DYNAMO_ROOT=${DYNAMO_ROOT:-$(cd "$HERE/../../../.." && pwd)}
SGLANG_ROOT=${SGLANG_ROOT:-$(cd "$DYNAMO_ROOT/../sglang" && pwd)}
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

DYNAMO_COMMIT=$(git -C "$DYNAMO_ROOT" rev-parse HEAD)
SGLANG_COMMIT=$(git -C "$SGLANG_ROOT" rev-parse HEAD)

if [[ -n $(git -C "$DYNAMO_ROOT" status --porcelain --untracked-files=no) ]]; then
    echo "Dynamo has tracked changes; commit them before building" >&2
    exit 1
fi
if [[ -n $(git -C "$SGLANG_ROOT" status --porcelain --untracked-files=no) ]]; then
    echo "SGLang has tracked changes; commit them before building" >&2
    exit 1
fi

maturin build \
    --release \
    --manifest-path "$DYNAMO_ROOT/lib/bindings/python/Cargo.toml" \
    --out "$BUILD_DIR"
mkdir -p "$BUILD_DIR/dynamo" "$BUILD_DIR/sglang"
git -C "$DYNAMO_ROOT" archive "$DYNAMO_COMMIT" components/src | \
    tar -x -C "$BUILD_DIR/dynamo"
git -C "$SGLANG_ROOT" archive "$SGLANG_COMMIT" python | \
    tar -x -C "$BUILD_DIR/sglang"
cp "$HERE/Dockerfile" "$BUILD_DIR/Dockerfile"
docker build \
    --build-arg "DYNAMO_COMMIT=$DYNAMO_COMMIT" \
    --build-arg "SGLANG_COMMIT=$SGLANG_COMMIT" \
    -t "$IMAGE" \
    -f "$BUILD_DIR/Dockerfile" \
    "$BUILD_DIR"

echo "Built $IMAGE with Dynamo $DYNAMO_COMMIT and SGLang $SGLANG_COMMIT"
