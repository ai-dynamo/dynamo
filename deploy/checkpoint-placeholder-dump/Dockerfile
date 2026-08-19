# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dump-capable snapshot placeholder: framework runtime (BASE_IMAGE) plus the
# binaries the live Dynamo operator still injects on multi-GPU checkpoint jobs:
#   cuda-checkpoint --launch-job … /usr/local/bin/snapshot-cuda-vmm-launch …
#
# The thin deploy/checkpoint-placeholder image ships neither binary. This
# overlay is a compose-branch workaround until the operator wrap and the thin
# placeholder contract agree.
#
#   docker build \
#     --build-arg BASE_IMAGE=<framework-runtime-image> \
#     --build-arg DOCKER_PROXY=<optional dockerhub mirror prefix> \
#     -t placeholder-dump:latest .

ARG BASE_IMAGE=placeholder-requires-base-image-arg
ARG DOCKER_PROXY=

FROM ${DOCKER_PROXY}library/ubuntu:24.04 AS cuda-ckpt
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint

FROM ${BASE_IMAGE} AS vmm
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY cuda-vmm-interpose /src/cuda-vmm-interpose
RUN make -C /src/cuda-vmm-interpose BUILD_DIR=/cuda-vmm-build CUDA_HOME=/usr/local/cuda

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64" >&2; exit 1; \
    fi \
    && mkdir -p /checkpoints /usr/local/lib/snapshot

COPY --from=cuda-ckpt /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/cuda-checkpoint
COPY --from=vmm /cuda-vmm-build/snapshot-cuda-vmm /usr/local/bin/snapshot-cuda-vmm
COPY --from=vmm /cuda-vmm-build/libsnapshot_cuda_vmm.so /usr/local/lib/snapshot/libsnapshot_cuda_vmm.so
COPY cuda-vmm-interpose/launch.sh /usr/local/bin/snapshot-cuda-vmm-launch
RUN chmod 0755 /usr/local/bin/cuda-checkpoint \
      /usr/local/bin/snapshot-cuda-vmm \
      /usr/local/bin/snapshot-cuda-vmm-launch \
    && chmod 0644 /usr/local/lib/snapshot/libsnapshot_cuda_vmm.so
