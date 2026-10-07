#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only image serving through the real Dynamo frontend and vLLM sidecars.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# No GPU memory setup is needed: both engine roles use CPU-only mocks.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../../examples/common/launch_utils.sh"

MODE="${1:-aggregated}"
case "$MODE" in
    aggregated|disagg) ;;
    -h|--help)
        echo "Usage: $0 [aggregated|disagg]"
        echo "MODEL selects the model metadata (default: Qwen/Qwen2.5-VL-3B-Instruct)."
        echo "DYN_HTTP_PORT, VLLM_GRPC_PORT1/2, and DYN_SYSTEM_PORT1/2 override ports."
        echo "DYN_NAMESPACE and DYN_FILE_KV isolate discovery on this host."
        exit 0
        ;;
    *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
if (( $# > 1 )); then
    echo "Expected one serving mode; use --help for usage." >&2
    exit 1
fi

trap dynamo_exit_trap EXIT

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DYN_NAMESPACE="${DYN_NAMESPACE:-mm-mock-$$}"
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_FILE_KV="${DYN_FILE_KV:-$PWD/.dynamo-mm-mock}"
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Image data is opaque to the mock; its token-only cache must not imply image reuse.
ENGINE_ARGS='{"engine":{"enable_prefix_caching":false,"speedup_ratio":0.0,"max_model_len":8192,"num_gpu_blocks":4096,"max_num_seqs":64,"max_num_batched_tokens":8192}}'

print_launch_banner --multimodal "vLLM image mock: $MODE (CPU only)" "$MODEL" "$DYN_HTTP_PORT"

# The sidecar advertises URL passthrough. No frontend media-decoder override is needed.
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-0}" \
    python3 -m dynamo.frontend --router-mode round-robin --namespace "$DYN_NAMESPACE" &

launch_worker() {
    local role="$1" grpc_port="$2" system_port="$3"
    dynamo-vllm-mocker-server \
        --listen "127.0.0.1:$grpc_port" --model "$MODEL" \
        --disaggregation-mode "$role" --extra-engine-args "$ENGINE_ARGS" &
    DYN_SYSTEM_PORT="$system_port" dynamo-vllm-sidecar \
        --grpc-endpoint "127.0.0.1:$grpc_port" --disaggregation-mode "$role" &
}

if [[ "$MODE" == aggregated ]]; then
    launch_worker aggregated "${VLLM_GRPC_PORT1:-50051}" "${DYN_SYSTEM_PORT1:-0}"
else
    launch_worker decode "${VLLM_GRPC_PORT1:-50051}" "${DYN_SYSTEM_PORT1:-0}"
    launch_worker prefill "${VLLM_GRPC_PORT2:-50052}" "${DYN_SYSTEM_PORT2:-0}"
fi

wait_any_exit
