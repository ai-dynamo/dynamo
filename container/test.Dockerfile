# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CI-only test layer on top of a runtime image.
# Adds test dependencies from requirements.test.txt.
# This image is NOT pushed to ACR or used in production — it is built
# once per CI run, pushed to ECR, and pulled by test runners.
#
# Build context: container/deps/
# Usage: docker buildx build --build-arg BASE_IMAGE=<runtime> -f container/test.Dockerfile container/deps/

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

# Install test dependencies using bind mount (no extra layer for the file).
# uv (vllm/trtllm/dynamo): installs into venv via VIRTUAL_ENV env var
# pip (sglang): installs to system site-packages
RUN --mount=type=bind,source=requirements.test.txt,target=/tmp/requirements.test.txt \
    if command -v uv >/dev/null 2>&1; then \
        uv pip install -r /tmp/requirements.test.txt; \
    else \
        pip install --break-system-packages -r /tmp/requirements.test.txt; \
    fi

USER dynamo
