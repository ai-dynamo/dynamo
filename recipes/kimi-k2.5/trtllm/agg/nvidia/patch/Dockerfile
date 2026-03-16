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

# Resolve site-packages dynamically (no hardcoded Python version) and apply
# the upstream unified diff with --forward (idempotent) and --fuzz=0 (exact
# context match — fails loudly if the target file has changed).
RUN SITE_PKGS=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    cd "$SITE_PKGS" && \
    if patch -p1 --forward --fuzz=0 --dry-run < /tmp/kimi.patch > /dev/null 2>&1; then \
        patch -p1 --forward --fuzz=0 < /tmp/kimi.patch; \
    else \
        echo "Patch already applied or context mismatch — verifying..."; \
    fi && \
    rm -f /tmp/kimi.patch

# Smoke test — verify the class was registered (no GPU needed at build time)
RUN SITE_PKGS=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    grep -q 'class KimiK25ForConditionalGeneration' \
        "$SITE_PKGS/tensorrt_llm/_torch/models/modeling_deepseekv3.py"

USER 1000
