#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with KVBM enabled
# NOTE: remove --enforce-eager for production use
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector kvbm --enforce-eager
