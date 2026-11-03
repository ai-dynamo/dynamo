# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image for k8s.yaml: the Dynamo runtime and the two example scripts.
# DYNAMO_RUNTIME is a pip requirement, or the path of a wheel in this directory.
FROM python:3.12-slim
ARG DYNAMO_RUNTIME=ai-dynamo-runtime
COPY . /example/
WORKDIR /example
RUN pip install --no-cache-dir --pre --extra-index-url https://pypi.nvidia.com/ "${DYNAMO_RUNTIME}"
