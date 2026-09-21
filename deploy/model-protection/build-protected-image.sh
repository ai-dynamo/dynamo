#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-dynamo:protected-vllm-base}"
IMAGE_TAG="${IMAGE_TAG:-dynamo-vllm-protected:local}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 127; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 127; }
if command -v maturin >/dev/null 2>&1; then
  MATURIN=(maturin)
elif command -v uv >/dev/null 2>&1; then
  # Keep the build reproducible without requiring a global maturin install.
  MATURIN=(uv run --no-project --with 'maturin[patchelf]' maturin)
else
  echo "maturin is required (install with: uv tool install 'maturin[patchelf]')" >&2
  exit 127
fi

# bindgen/clang may not inherit GCC's standard include directory (notably when
# uv creates an isolated CPython build environment). Make stdbool.h and the
# other libc headers explicit instead of relying on the caller's shell.
if [[ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]]; then
  GCC_INCLUDE="$(gcc -print-file-name=include 2>/dev/null || true)"
  BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include -I/usr/include/x86_64-linux-gnu"
  [[ -d "$GCC_INCLUDE" ]] && BINDGEN_EXTRA_CLANG_ARGS+=" -I$GCC_INCLUDE"
  export BINDGEN_EXTRA_CLANG_ARGS
fi

cd "$ROOT_DIR"
python3 container/render.py --framework vllm --target runtime --output-short-filename
docker build -t "$BASE_IMAGE" -f container/rendered.Dockerfile .

WHEEL_DIR="$ROOT_DIR/lib/bindings/python/target/wheels"
(cd "$ROOT_DIR/lib/bindings/python" && "${MATURIN[@]}" build --release --features model-protection-tpm2)
mapfile -t WHEELS < <(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'ai_dynamo_runtime-*.whl' -printf '%T@ %p\n' | sort -nr | sed 's/^[^ ]* //')
[[ ${#WHEELS[@]} -gt 0 ]] || { echo "TPM-enabled wheel was not produced" >&2; exit 1; }

BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-protected-build.XXXXXX")"
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
cp "$ROOT_DIR/deploy/model-protection/Dockerfile" "$BUILD_CONTEXT/Dockerfile"
cp "${WHEELS[0]}" "$BUILD_CONTEXT/"

docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE_TAG" "$BUILD_CONTEXT"
echo "protected image: $IMAGE_TAG"
