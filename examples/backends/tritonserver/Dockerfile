# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG DYNAMO_BASE_IMAGE="dynamo-base:latest"
ARG TRITON_SERVER_IMAGE="nvcr.io/nvidia/tritonserver:25.10-py3"

FROM ${TRITON_SERVER_IMAGE} AS triton_source
# Ensure DCGM directory exists so COPY never fails when the source image omits it
RUN mkdir -p /usr/local/dcgm

FROM ${DYNAMO_BASE_IMAGE} AS dynamo_base

COPY --from=triton_source /opt/tritonserver /opt/tritonserver
COPY --from=triton_source /usr/local/dcgm /usr/local/dcgm
USER root
RUN --mount=type=bind,from=triton_source,source=/usr/lib/x86_64-linux-gnu,target=/triton_usr_lib \
    find /triton_usr_lib -name "libdcgm*.so*" -exec cp -a {} /lib/x86_64-linux-gnu/ \;
ENV BACKEND_DIR=/opt/tritonserver/backends
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/tritonserver/lib:/opt/tritonserver/backends:/usr/local/dcgm/lib64
ENV PATH=/opt/tritonserver/bin:$PATH

COPY --chown=dynamo: src/ /workspace/src/
COPY --chown=dynamo: model_repo/ /workspace/model_repo/
COPY --chown=dynamo: launch/ /workspace/launch/

WORKDIR /workspace
USER dynamo

RUN uv pip install --no-cache-dir tritonclient[grpc]
RUN uv pip install /opt/tritonserver/python/triton*.whl
# Validate CUDA ABI compatibility at build time
RUN python3 -c "import tritonserver"
