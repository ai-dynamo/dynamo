# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image for k8s.yaml: the Dynamo runtime wheel and the two example scripts.
# DYNAMO_RUNTIME selects the wheel; the default is the latest nightly.
FROM python:3.12-slim
ARG DYNAMO_RUNTIME=ai-dynamo-runtime
RUN pip install --no-cache-dir --pre --extra-index-url https://pypi.nvidia.com/ "${DYNAMO_RUNTIME}"
COPY decode.py prefill.py /example/
