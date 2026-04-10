#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.1}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${WORKSPACE_DIR}/container/deps/fastvideo/requirements.fastvideo.txt}"
INSTALL_LOCAL_DYNAMO="${INSTALL_LOCAL_DYNAMO:-false}"
INSTALL_TORCH_FROM_INDEX="${INSTALL_TORCH_FROM_INDEX:-true}"

CUDA_MAJOR="${CUDA_VERSION%%.*}"
# PyTorch uses a single CUDA 13 wheel family (`cu130`) for CUDA 13.x runtimes.
if [ "${CUDA_MAJOR}" = "13" ]; then
    TORCH_BACKEND="cu130"
else
    TORCH_BACKEND="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
fi
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_BACKEND}"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

if [ "${INSTALL_TORCH_FROM_INDEX}" = "true" ]; then
    uv pip install \
        --index-strategy unsafe-best-match \
        --index-url "${TORCH_INDEX_URL}" \
        --torch-backend "${TORCH_BACKEND}" \
        torch torchvision
fi

# Follow-up for parity with the more mature backend installers: keep the
# curated NGC PyTorch stack pinned during this install so transitive FastVideo
# dependencies do not replace it with a different torch wheel family.
# FastVideo's pyproject uses uv package sources for torch that prefer CUDA 12.8
# wheels. Ignore uv package sources during this install so resolution follows
# the CUDA backend selected for the Dynamo image.
uv pip install \
    --no-sources \
    --index-strategy unsafe-best-match \
    --torch-backend "${TORCH_BACKEND}" \
    --requirement "${REQUIREMENTS_FILE}"

if [ "${INSTALL_LOCAL_DYNAMO}" = "true" ]; then
    uv pip install "${WORKSPACE_DIR}/lib/bindings/python"
    uv pip install "${WORKSPACE_DIR}"
fi
