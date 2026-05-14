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

# FastVideo's pyproject uses uv package sources for torch. Ignore sources here
# so the Dynamo image's CUDA backend selection remains authoritative.
uv pip install \
    --no-sources \
    --index-strategy unsafe-best-match \
    --torch-backend "${TORCH_BACKEND}" \
    --requirement "${REQUIREMENTS_FILE}"

if [ "${INSTALL_LOCAL_DYNAMO}" = "true" ]; then
    uv pip install "${WORKSPACE_DIR}/lib/bindings/python"
    uv pip install "${WORKSPACE_DIR}"
fi
