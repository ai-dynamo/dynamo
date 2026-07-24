# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hello-world Dynamo custom engine on the published Dynamo base image.
# The base carries the Dynamo runtime + ai-dynamo Python package; we
# layer the engine wheel on top. No GPU, no model weights.
ARG BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-frontend
ARG BASE_IMAGE_TAG=1.3.0

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

WORKDIR /opt/hello-engine

COPY pyproject.toml README.md ./
COPY src ./src

# The base image manages its venv with uv (no pip installed).
RUN uv pip install --python /opt/dynamo/venv/bin/python3 --no-deps . \
    && uv pip install --python /opt/dynamo/venv/bin/python3 tokenizers \
    && python3 -c "import hello_engine"

ENTRYPOINT ["python3", "-m", "hello_engine.main"]
