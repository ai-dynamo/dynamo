#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Common configuration
MODEL="Qwen/Qwen3-0.6B"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Parse command line arguments BEFORE installing the kill-process-group
# EXIT trap — `--help` and unknown-option early exits would otherwise
# kill the caller's process group before any worker is even launched.
USE_UNIFIED=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --unified)
            # Run the workers via the unified entry point
            # (`python -m dynamo.vllm.unified_main`) so disagg goes through
            # the new dynamo.common.backend / dynamo_backend_common path
            # instead of the legacy main.py / WorkerFactory dispatch.
            USE_UNIFIED=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --unified            Use the unified backend entry point"
            echo "                       (python -m dynamo.vllm.unified_main)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

if [ "$USE_UNIFIED" = true ]; then
    WORKER_MODULE="dynamo.vllm.unified_main"
else
    WORKER_MODULE="dynamo.vllm"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving (2 GPUs)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of relying on vLLM defaults
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m "$WORKER_MODULE" \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m "$WORKER_MODULE" \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
