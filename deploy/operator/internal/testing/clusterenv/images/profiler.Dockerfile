# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.3.0
FROM ${BASE_IMAGE}

COPY --chmod=775 --chown=1000:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=1000:0 components/src/dynamo/planner /workspace/components/src/dynamo/planner
COPY --chmod=775 --chown=1000:0 components/src/dynamo/profiler /workspace/components/src/dynamo/profiler
COPY --chmod=664 --chown=1000:0 deploy/__init__.py /workspace/deploy/__init__.py
COPY --chmod=775 --chown=1000:0 deploy/utils /workspace/deploy/utils
