#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving with the sample (echo) backend — GPU-free smoke test.
#
# Spawns dynamo.frontend plus one prefill worker and one decode worker, both
# backed by the sample engine. Prefill workers register as ModelType::Prefill
# (the unified Rust Worker overrides registration based on
# WorkerConfig.disaggregation_mode); the frontend's PrefillRouter forwards
# the synthetic disaggregated_params handle from prefill to decode.
#
# GPUs: 0 (CPU-only, useful for CI smoke tests of the unified disagg path).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL_NAME="${MODEL_NAME:-sample-model}"

# Parse arguments BEFORE installing the kill-process-group EXIT trap.
# `--help` and unknown-option exits would otherwise kill the caller.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-name <name>  Specify model name (default: $MODEL_NAME)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Any additional options are passed through to both sample workers."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sample Disaggregated Serving (CPU-only)" "$MODEL_NAME" "$HTTP_PORT"

# run frontend
python3 -m dynamo.frontend &

# Per-worker DYN_SYSTEM_PORT so parallel CI runs don't collide on the metrics
# port. Mirrors examples/backends/vllm/launch/disagg.sh.
# run prefill worker
# Distinct --component name keeps the two workers visible separately in
# discovery, mirroring the per-role components in vLLM/SGLang/TRT-LLM.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --component sample-prefill \
  --disaggregation-mode prefill \
  "${EXTRA_ARGS[@]}" &

# run decode worker
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.common.backend.sample_main \
  --model-name "$MODEL_NAME" \
  --component sample-decode \
  --disaggregation-mode decode \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
