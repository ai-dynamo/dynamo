#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <docker-image>"
    echo "  Patches modeling_deepseekv3.py with KimiK25ForConditionalGeneration class"
    echo "  and applies attention-dp patch for KVBM support."
    echo "  Outputs: <docker-image>-patched"
    exit 1
fi

SRC_IMAGE="$1"
DST_IMAGE="${SRC_IMAGE}-patched"
TARGET_FILE="/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/models/modeling_deepseekv3.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMI_PATCH="${SCRIPT_DIR}/kimi.patch"
ATTENTION_DP_PATCH="${SCRIPT_DIR}/attention-dp.patch"

if [[ ! -f "$KIMI_PATCH" ]]; then
    echo "ERROR: Patch file not found: $KIMI_PATCH"
    exit 1
fi

if [[ ! -f "$ATTENTION_DP_PATCH" ]]; then
    echo "ERROR: Patch file not found: $ATTENTION_DP_PATCH"
    exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cp "$KIMI_PATCH" "$TMPDIR/kimi.patch"
cp "$ATTENTION_DP_PATCH" "$TMPDIR/attention-dp.patch"

cat > "$TMPDIR/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TARGET_FILE

USER root

COPY kimi.patch /opt/kimi.patch
COPY attention-dp.patch /opt/attention-dp.patch

# Apply kimi.patch: append KimiK25ForConditionalGeneration class to modeling_deepseekv3.py
RUN if grep -q 'KimiK25ForConditionalGeneration' "${TARGET_FILE}"; then \
        echo "Kimi patch already applied, skipping."; \
    else \
        if ! head -50 "${TARGET_FILE}" | grep -q '^import copy'; then \
            sed -i '1s/^/import copy\n/' "${TARGET_FILE}"; \
        fi && \
        echo "" >> "${TARGET_FILE}" && \
        cat /opt/kimi.patch >> "${TARGET_FILE}"; \
    fi && \
    rm -f /opt/kimi.patch

# Apply attention-dp.patch: enable attention_dp to work with KVBM
RUN cd /opt/dynamo/venv/lib/python3.12/site-packages && \
    git apply /opt/attention-dp.patch && \
    rm -f /opt/attention-dp.patch

USER 1000
DOCKERFILE

echo "Building patched image: ${DST_IMAGE}"
docker build \
    --build-arg BASE_IMAGE="$SRC_IMAGE" \
    --build-arg TARGET_FILE="$TARGET_FILE" \
    -t "$DST_IMAGE" \
    "$TMPDIR"

echo "Done. Patched image: ${DST_IMAGE}"
