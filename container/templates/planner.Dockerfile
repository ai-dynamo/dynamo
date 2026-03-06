{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/planner.Dockerfile ===
##############################################
########## Planner / Profiler image ##########
##############################################
# Extends the dynamo runtime with planner- and profiler-specific
# Python dependencies that are not needed by the base runtime,
# frontend, or backend framework images.

FROM runtime AS planner

# Install planner/profiler-specific Python dependencies
RUN --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install --requirement /tmp/requirements.planner.txt
