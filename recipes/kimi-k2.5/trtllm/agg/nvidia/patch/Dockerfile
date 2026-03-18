# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Patches TensorRT-LLM with KimiK25ForConditionalGeneration support.
# Upstream tracking PR: https://github.com/NVIDIA/TensorRT-LLM/pull/11816
#
# Usage:
#   docker build --build-arg BASE_IMAGE=<image> -t <image>-patched .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

COPY kimi.patch /tmp/kimi.patch

# Apply upstream diff — idempotent, fails if target file has diverged
RUN SITE_PKGS=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    TARGET="$SITE_PKGS/tensorrt_llm/_torch/models/modeling_deepseekv3.py" && \
    cd "$SITE_PKGS" && \
    if patch -p1 --forward --fuzz=0 --dry-run < /tmp/kimi.patch > /dev/null 2>&1; then \
        patch -p1 --forward --fuzz=0 < /tmp/kimi.patch; \
    elif patch -p1 --reverse --fuzz=0 --dry-run < /tmp/kimi.patch > /dev/null 2>&1; then \
        echo "Patch already applied, skipping."; \
    else \
        echo "ERROR: Patch failed — the target file may have changed upstream." >&2; \
        echo "Try updating kimi.patch from https://github.com/NVIDIA/TensorRT-LLM/pull/11816" >&2; \
        exit 1; \
    fi && \
    rm -f /tmp/kimi.patch

# Smoke test
RUN SITE_PKGS=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    grep -q '@register_auto_model("KimiK25ForConditionalGeneration")' \
        "$SITE_PKGS/tensorrt_llm/_torch/models/modeling_deepseekv3.py" || \
    { echo "ERROR: KimiK25ForConditionalGeneration not registered after patching" >&2; exit 1; }

USER dynamo
