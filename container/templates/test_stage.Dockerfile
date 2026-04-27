{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/test_stage.Dockerfile ===
FROM {{ target }} AS {{ target }}_test
USER root
RUN --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    if [ -n "$VIRTUAL_ENV" ]; then \
        uv pip install \
            --requirement /tmp/requirements.test.txt; \
    else \
        pip install --break-system-packages \
            --requirement /tmp/requirements.test.txt; \
    fi

USER dynamo
COPY --chmod=664 --chown=dynamo:0 pyproject.toml /workspace/pyproject.toml
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/
