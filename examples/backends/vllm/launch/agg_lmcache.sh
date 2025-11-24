#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run worker with LMCache enabled
DYN_SYSTEM_PORT=8081 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector lmcache
