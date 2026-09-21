# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=protected-model-requires-base-image-arg
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libtss2-esys-3.0.2-0t64 \
        libtss2-tcti-device0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY ai_dynamo_runtime-*.whl /tmp/
RUN uv pip install --system --no-deps --reinstall /tmp/ai_dynamo_runtime-*.whl \
    && rm /tmp/ai_dynamo_runtime-*.whl \
    && python3 -c "import vllm; from dynamo._core import model_protection_tpm_enabled; assert model_protection_tpm_enabled; assert vllm.__version__.startswith('0.29.'), vllm.__version__"

USER 1000
