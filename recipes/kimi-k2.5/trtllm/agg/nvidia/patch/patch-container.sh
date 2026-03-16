#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <docker-image>"
    echo "  Patches modeling_deepseekv3.py with KimiK25ForConditionalGeneration class."
    echo "  Outputs: <docker-image>-patched"
    exit 1
fi

SRC_IMAGE="$1"
DST_IMAGE="${SRC_IMAGE}-patched"
TARGET_FILE="/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_deepseekv3.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/kimi.patch"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cp "$PATCH_FILE" "$TMPDIR/kimi.patch"

cat > "$TMPDIR/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TARGET_FILE

USER root

COPY kimi.patch /opt/kimi.patch

RUN if grep -q 'KimiK25ForConditionalGeneration' "${TARGET_FILE}"; then \
        echo "Patch already applied, skipping."; \
    else \
        if ! head -50 "${TARGET_FILE}" | grep -q '^import copy'; then \
            sed -i '1s/^/import copy\n/' "${TARGET_FILE}"; \
        fi && \
        echo "" >> "${TARGET_FILE}" && \
        cat /opt/kimi.patch >> "${TARGET_FILE}"; \
    fi && \
    rm -f /opt/kimi.patch

USER 1000
DOCKERFILE

echo "Building patched image: ${DST_IMAGE}"
docker build \
    --build-arg BASE_IMAGE="$SRC_IMAGE" \
    --build-arg TARGET_FILE="$TARGET_FILE" \
    -t "$DST_IMAGE" \
    "$TMPDIR"

echo "Done. Patched image: ${DST_IMAGE}"
