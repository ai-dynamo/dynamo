#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress - no need to specify DYN_REQUEST_PLANE, it will auto-detect
python -m dynamo.frontend --http-port=8000 &

# run worker - DYN_REQUEST_PLANE=http needed for worker to register with HTTP transport in etcd
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_REQUEST_PLANE=http DYN_HTTP_RPC_HOST=0.0.0.0 DYN_HTTP_RPC_PORT=8080  DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none
