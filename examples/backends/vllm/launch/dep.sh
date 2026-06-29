#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Common configuration
MODEL="Qwen/Qwen3-30B-A3B"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

print_launch_banner "Launching Data Parallel / Expert Parallelism (4 GPUs)" "$MODEL" "$HTTP_PORT"


# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend --router-mode kv &

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
# Chose Qwen3-30B because its a small MOE that can fit on smaller GPUs (L40S for example)
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
python3 -m dynamo.vllm \
--model Qwen/Qwen3-30B-A3B \
--data-parallel-hybrid-lb \
--data-parallel-size 4 \
--data-parallel-size-local 4 \
--data-parallel-start-rank 0 \
--enable-expert-parallel \
--enforce-eager \
--kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:20080\",\"enable_kv_cache_events\":true}" &

echo "All workers starting. (press Ctrl+C to stop)..."
# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
