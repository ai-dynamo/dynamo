# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:c3e66f09092d354ba2ad17703a5fb4b59c7bb7d113ddf7eb9b3939bd96fa18e8
FROM ${BASE_IMAGE}

USER root

COPY --chmod=644 lib/gpu_memory_service/cli/snapshot/loader.py \
    /usr/local/lib/python3.12/dist-packages/gpu_memory_service/cli/snapshot/loader.py
COPY --chmod=644 lib/gpu_memory_service/snapshot/storage_client.py \
    /usr/local/lib/python3.12/dist-packages/gpu_memory_service/snapshot/storage_client.py
COPY --chmod=644 lib/gpu_memory_service/snapshot/transfer.py \
    /usr/local/lib/python3.12/dist-packages/gpu_memory_service/snapshot/transfer.py
COPY --chmod=644 lib/gpu_memory_service/snapshot/backends/nixl_staging.py \
    /usr/local/lib/python3.12/dist-packages/gpu_memory_service/snapshot/backends/nixl_staging.py

USER dynamo
