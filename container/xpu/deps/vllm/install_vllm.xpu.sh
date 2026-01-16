#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script is used to install vLLM and its dependencies
# If installing vLLM from a release tag, we will use pip to manage the install
# Otherwise, we will use git to checkout the vLLM source code and build it from source.
# The dependencies are installed in the following order:
# 1. vLLM
# 2. LMCache

set -euo pipefail

VLLM_REF="v0.10.2"

# Basic Configurations
ARCH=$(uname -m)
MAX_JOBS=16
INSTALLATION_DIR=/tmp

# VLLM and Dependency Configurations
# LMCache version - 0.3.9+ like 0.3.10 required for vLLM 0.11.2 compatibility
LMCACHE_REF="0.3.7"

# These flags are applicable when installing vLLM from source code
EDITABLE=true
VLLM_GIT_URL="https://github.com/vllm-project/vllm.git"

while [[ $# -gt 0 ]]; do
    case $1 in
        --editable)
            EDITABLE=true
            shift
            ;;
        --no-editable)
            EDITABLE=false
            shift
            ;;
        --vllm-ref)
            VLLM_REF="$2"
            shift 2
            ;;
        --vllm-git-url)
            VLLM_GIT_URL="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --installation-dir)
            INSTALLATION_DIR="$2"
            shift 2
            ;;
        --lmcache-ref)
            LMCACHE_REF="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--editable|--no-editable] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--installation-dir DIR]"
            echo "Options:"
            echo "  --editable        Install vllm in editable mode (default)"
            echo "  --no-editable     Install vllm in non-editable mode"
            echo "  --vllm-ref REF    Git reference to checkout (default: ${VLLM_REF})"
            echo "  --max-jobs NUM    Maximum number of parallel jobs (default: ${MAX_JOBS})"
            echo "  --arch ARCH       Architecture (amd64|arm64, default: auto-detect)"
            echo "  --installation-dir DIR  Directory to install vllm (default: ${INSTALLATION_DIR})"
            echo "  --lmcache-ref REF   LMCache version (default: ${LMCACHE_REF})"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert x86_64 to amd64 for consistency with Docker ARG
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

export MAX_JOBS=$MAX_JOBS

echo "=== Installing prerequisites ==="

echo "\n=== Configuration Summary ==="
echo "  VLLM_REF=$VLLM_REF | EDITABLE=$EDITABLE | ARCH=$ARCH | LMCACHE_REF=$LMCACHE_REF"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  INSTALLATION_DIR=$INSTALLATION_DIR | VLLM_GIT_URL=$VLLM_GIT_URL"

echo "\n=== Installing LMCache ==="
if [ "$ARCH" = "amd64" ]; then
    # LMCache origin & license:
    #   Origin: PyPI package 'lmcache' (https://pypi.org/project/lmcache/)
    #   Source: https://github.com/LMCache/LMCache
    #   License: Apache-2.0 (per project metadata). Ensure Apache-2.0 notice retained on redistribution.
    # Install LMCache BEFORE vLLM so vLLM's dependencies take precedence
    uv pip install lmcache==${LMCACHE_REF}
    echo "✓ LMCache ${LMCACHE_REF} installed"
fi

echo "\n=== Cloning vLLM repository ==="
# We need to clone to install dependencies
cd $INSTALLATION_DIR
git clone $VLLM_GIT_URL vllm
cd vllm
git checkout $VLLM_REF

echo "\n=== Applying custom modifications ==="
git apply --ignore-whitespace /tmp/vllm-xpu.patch

echo "\n=== Installing vLLM"
uv pip install -r requirements/xpu.txt --index-strategy unsafe-best-match

if [ "$EDITABLE" = "true" ]; then
    uv pip install --verbose --no-build-isolation -e .
else
    uv pip install --verbose --no-build-isolation .
fi

echo "✓ vLLM installation completed"

echo "\n✅ All installations completed successfully!"


