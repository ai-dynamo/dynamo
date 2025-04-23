# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG DEBIAN_FRONTEND=noninteractive

FROM ubuntu:24.04 AS version

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -qq -y git && \
    git config --global --add safe.directory /workspace

RUN --mount=type=bind,source=./container/deps/,target=/workspace \
    uv venv -p 3.12 /venv && \
    source /venv/bin/activate && \
    uv pip install dunamai==1.23.1

RUN --mount=type=bind,source=.,target=/workspace \
    source /venv/bin/activate && \
    python ./container/version.py > /version.txt

CMD ["cat", "/version.txt"]
