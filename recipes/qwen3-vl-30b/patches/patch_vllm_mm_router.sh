#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  cat <<'EOF'
Build a patched image for multimodal-aware router support (vLLM PR #33304).

Usage:
  ./patch_vllm_mm_router.sh --base-image <image> --output-image <image>

Options:
  --base-image   Base runtime image to patch (required)
  --output-image Output image tag to build (required)
  -h, --help     Show this help

Example:
  ./patch_vllm_mm_router.sh \
    --base-image nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0 \
    --output-image my-registry/vllm-runtime:0.8.0-mm-router-patched
EOF
}

BASE_IMAGE=""
OUTPUT_IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image)
      BASE_IMAGE="${2:-}"
      shift 2
      ;;
    --output-image)
      OUTPUT_IMAGE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BASE_IMAGE}" || -z "${OUTPUT_IMAGE}" ]]; then
  echo "--base-image and --output-image are required."
  usage
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required."
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "Preparing patch diff..."
curl -sL "https://github.com/vllm-project/vllm/pull/33304.diff" | python3 -c '
import sys
chunks = sys.stdin.read().split("diff --git ")
filtered = [c for c in chunks if c.startswith("a/vllm/")]
print("".join("diff --git " + c for c in filtered), end="")
' > "${WORKDIR}/vllm_pr33304_vllm_only.diff"

cat > "${WORKDIR}/Dockerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*

COPY vllm_pr33304_vllm_only.diff /tmp/vllm_pr33304_vllm_only.diff

RUN set -eux; \
  SITE_PACKAGES_ROOT="$(python3 - <<'PY' \
import pathlib, vllm \
print(pathlib.Path(vllm.__file__).resolve().parent.parent) \
PY \
  )"; \
  cd "${SITE_PACKAGES_ROOT}"; \
  patch --dry-run -p1 < /tmp/vllm_pr33304_vllm_only.diff; \
  patch -p1 < /tmp/vllm_pr33304_vllm_only.diff; \
  rm -f /tmp/vllm_pr33304_vllm_only.diff
EOF

echo "Building patched image: ${OUTPUT_IMAGE}"
docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${OUTPUT_IMAGE}" "${WORKDIR}"

echo "Done. Use image: ${OUTPUT_IMAGE}"
