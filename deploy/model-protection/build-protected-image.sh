#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-dynamo:protected-vllm-base}"
IMAGE_TAG="${IMAGE_TAG:-dynamo-vllm-protected:local}"
VLLM_PROTECTED_VERSION="${VLLM_PROTECTED_VERSION:-0.30.0}"
SKIP_BASE_BUILD="${SKIP_BASE_BUILD:-false}"
[[ "$VLLM_PROTECTED_VERSION" == "0.30.0" ]] || { echo 'Protected runtime requires vLLM 0.30.0' >&2; exit 2; }
[[ "$SKIP_BASE_BUILD" =~ ^(true|false)$ ]] || { echo 'SKIP_BASE_BUILD must be true or false' >&2; exit 2; }

# Avoid HTTP/2 stream resets while fetching crates in Docker and on the host.
export CARGO_HTTP_MULTIPLEXING="${CARGO_HTTP_MULTIPLEXING:-false}"
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-5}"
export CARGO_HTTP_TIMEOUT="${CARGO_HTTP_TIMEOUT:-60}"
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
[[ "$CARGO_HTTP_MULTIPLEXING" =~ ^(true|false)$ ]] || { echo 'CARGO_HTTP_MULTIPLEXING must be true or false' >&2; exit 2; }
[[ "$CARGO_NET_RETRY" =~ ^[0-9]+$ ]] || { echo 'CARGO_NET_RETRY must be a non-negative integer' >&2; exit 2; }
[[ "$CARGO_HTTP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'CARGO_HTTP_TIMEOUT must be a positive number of seconds' >&2; exit 2; }

command -v docker >/dev/null || { echo "docker is required" >&2; exit 127; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 127; }
if ! python3 -c 'import yaml, jinja2' >/dev/null 2>&1; then
  echo 'Dockerfile rendering requires: python3 -m pip install PyYAML Jinja2' >&2
  exit 2
fi
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
  if [[ -d "$GCC_INCLUDE" ]]; then
    BINDGEN_EXTRA_CLANG_ARGS+=" -I$GCC_INCLUDE"
  fi
  export BINDGEN_EXTRA_CLANG_ARGS
fi

cd "$ROOT_DIR"
python3 container/render.py --framework vllm --target runtime --output-short-filename
if [[ "$SKIP_BASE_BUILD" == "false" ]]; then
  docker build --build-arg CARGO_HTTP_MULTIPLEXING="$CARGO_HTTP_MULTIPLEXING" \
    --build-arg CARGO_NET_RETRY="$CARGO_NET_RETRY" \
    --build-arg CARGO_HTTP_TIMEOUT="$CARGO_HTTP_TIMEOUT" \
    --build-arg RUNTIME_IMAGE_TAG="v${VLLM_PROTECTED_VERSION}-ubuntu2404" \
    --build-arg VLLM_OMNI_REF="v${VLLM_PROTECTED_VERSION}rc1" \
    -t "$BASE_IMAGE" -f container/rendered.Dockerfile .
fi
# Check before the expensive wheel build, including when reusing a local base.
docker run --rm --network none --read-only --entrypoint python3 \
  -e EXPECTED_VLLM_VERSION="$VLLM_PROTECTED_VERSION" "$BASE_IMAGE" \
  -c 'import os, sys; from importlib.metadata import version; from dynamo.vllm import protection_bootstrap; actual = version("vllm"); expected = os.environ["EXPECTED_VLLM_VERSION"]; loader = getattr(protection_bootstrap, "SUPPORTED_VLLM_VERSION", None); print(f"protected base: vLLM={actual}, expected={expected}, loader={loader}", flush=True); sys.exit(0 if actual == expected and loader == expected else "Base image vLLM/loader version mismatch; rebuild the base with SKIP_BASE_BUILD=false")'

BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-protected-build.XXXXXX")"
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
WHEEL_DIR="$BUILD_CONTEXT/wheels"
mkdir -m 0700 "$WHEEL_DIR"
(cd "$ROOT_DIR/lib/bindings/python" && "${MATURIN[@]}" build --locked --release \
  --features model-protection-tpm2 --out "$WHEEL_DIR")
shopt -s nullglob
WHEELS=("$WHEEL_DIR"/ai_dynamo_runtime-*.whl)
[[ ${#WHEELS[@]} -eq 1 ]] || { echo "Expected exactly one freshly built TPM-enabled wheel" >&2; exit 1; }
cp "$ROOT_DIR/deploy/model-protection/Dockerfile" "$BUILD_CONTEXT/Dockerfile"
cp "$ROOT_DIR/deploy/model-protection/check-runtime.py" "$BUILD_CONTEXT/"
cp "${WHEELS[0]}" "$BUILD_CONTEXT/"

docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_PROTECTED_VERSION="$VLLM_PROTECTED_VERSION" \
  --build-arg DYNAMO_SOURCE_REVISION="$(git rev-parse HEAD)" \
  --build-arg DYNAMO_SOURCE_DIRTY="$(test -z "$(git status --porcelain)" && echo false || echo true)" \
  -t "$IMAGE_TAG" "$BUILD_CONTEXT"
echo "protected image: $IMAGE_TAG"
