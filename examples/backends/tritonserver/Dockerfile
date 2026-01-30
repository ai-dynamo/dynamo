# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG DYNAMO_BASE_IMAGE="dynamo:latest-none"
ARG TRITON_SERVER_IMAGE="nvcr.io/nvidia/tritonserver:25.01-py3"

FROM ${TRITON_SERVER_IMAGE} AS triton_source

FROM ${DYNAMO_BASE_IMAGE} AS dynamo_base

COPY --from=triton_source /opt/tritonserver /opt/tritonserver
COPY --from=triton_source /usr/local/dcgm /usr/local/dcgm
COPY --from=triton_source /lib/x86_64-linux-gnu/libdcgm*.so* /lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/tritonserver/lib:/opt/tritonserver/backends:/usr/local/dcgm/lib64
ENV PATH=/opt/tritonserver/bin:$PATH

COPY --chown=dynamo: src/ /workspace/src/
COPY --chown=dynamo: model_repo/ /workspace/model_repo/
COPY --chown=dynamo: launch/ /workspace/launch/

WORKDIR /workspace
USER dynamo

RUN uv pip install --no-cache-dir tritonclient[grpc]
RUN uv pip install /opt/tritonserver/python/triton*.whl
