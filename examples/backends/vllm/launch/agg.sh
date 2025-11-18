#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
# DYN_HTTP_PORT env var is read by dynamo.frontend (defaults to 8000 if not set)
python -m dynamo.frontend &

# run worker
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none
