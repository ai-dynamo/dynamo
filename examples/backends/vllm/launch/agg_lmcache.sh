#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let LMCache or Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run worker with LMCache enabled (without PROMETHEUS_MULTIPROC_DIR set externally)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector lmcache
