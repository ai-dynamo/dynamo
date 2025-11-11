#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# switch between request plane modes
export DYN_REQUEST_PLANE=tcp

# Or alternatively, use Nats/HTTP request plane
# export DYN_REQUEST_PLANE=http
# export DYN_REQUEST_PLANE=nats

# Frontend
python -m dynamo.frontend --http-port=8000 &

DYN_SYSTEM_ENABLED=true \
DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none &
