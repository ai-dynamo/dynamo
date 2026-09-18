#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving through two TensorRT-LLM OpenEngine gRPC servers (2 GPUs).
#
# Run this where `TRTLLM_PYTHON` has TensorRT-LLM installed --
# `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000` or newer, the
# first releases carrying the OpenEngine servicer. The bindings it needs are not
# in that image; the pip step below adds them.
#
# KV cache moves prefill -> decode engine-to-engine over the cache transceiver,
# not through the sidecars. Dynamo carries only the handoff: the prefill worker
# returns a `KvSessionRef` and the decode worker replays it.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# Resolved relative to this script, not via $DYNAMO_HOME: some runtime images
# bake DYNAMO_HOME to a minimal install path with no examples/ directory, which
# would silently override this and break sourcing.
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/common.sh"    # trtllm_ensure_openengine_bindings, trtllm_resolve_context_length

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model-path)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for $1"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model|--model-path <name>] [TensorRT-LLM engine options...]"
            echo
            echo "Additional options are passed to both TensorRT-LLM engines."
            echo
            echo "Environment overrides:"
            echo "  MODEL                     Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  TRTLLM_PYTHON             Python with TensorRT-LLM installed (default: python3)"
            echo "  TRTLLM_PREFILL_GPU        Prefill GPU assignment (default: 0)"
            echo "  TRTLLM_DECODE_GPU         Decode GPU assignment (default: 1)"
            echo "  DYN_HTTP_PORT             Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1          Prefill sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2          Decode sidecar system port (default: 8082)"
            echo "  TRTLLM_PREFILL_GRPC_PORT  Prefill TensorRT-LLM gRPC port (default: 50051)"
            echo "  TRTLLM_DECODE_GRPC_PORT   Decode TensorRT-LLM gRPC port (default: 50052)"
            echo "  TRTLLM_CACHE_TRANSCEIVER_BACKEND  KV transfer backend (default: NIXL)"
            echo "  TRTLLM_CONTEXT_LENGTH     Model context length, applied to both engines and both"
            echo "                            sidecars (default: 4096; unset when --max_seq_len is given)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

TRTLLM_EXTRA_CONFIG=""
trtllm_exit_trap() {
    local rc=$?
    if [[ -n "$TRTLLM_EXTRA_CONFIG" ]]; then
        rm -f -- "$TRTLLM_EXTRA_CONFIG"
    fi
    echo "Cleaning up..."
    dynamo_reap_and_exit "$rc"
}
trap trtllm_exit_trap EXIT

TRTLLM_PYTHON="${TRTLLM_PYTHON:-python3}"
TRTLLM_HOST="127.0.0.1"
TRTLLM_PREFILL_GRPC_PORT="${TRTLLM_PREFILL_GRPC_PORT:-50051}"
TRTLLM_DECODE_GRPC_PORT="${TRTLLM_DECODE_GRPC_PORT:-50052}"
TRTLLM_PREFILL_GPU="${TRTLLM_PREFILL_GPU:-0}"
TRTLLM_DECODE_GPU="${TRTLLM_DECODE_GPU:-1}"
TRTLLM_CACHE_TRANSCEIVER_BACKEND="${TRTLLM_CACHE_TRANSCEIVER_BACKEND:-NIXL}"

# `--extra_llm_api_options` is last-wins, not additive, so a forwarded copy
# would drop the transceiver and leave the prefill worker producing a handoff
# no decode worker can consume -- as an opaque engine-side transfer error, not
# a launcher one. Refuse it rather than silently losing the setting.
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --extra_llm_api_options|--extra_llm_api_options=*)
            echo "Cannot forward --extra_llm_api_options: this launcher needs it for" >&2
            echo "cache_transceiver_config. Merge your settings into that file, or set" >&2
            echo "TRTLLM_CACHE_TRANSCEIVER_BACKEND and run the engines yourself." >&2
            exit 1
            ;;
    esac
done

trtllm_resolve_context_length "${EXTRA_ARGS[@]}"

trtllm_ensure_openengine_bindings "$TRTLLM_PYTHON"

# Both engines need a cache transceiver or the handoff has nothing to move the
# KV cache over. NIXL picks its own underlying transport (UCX where there is no
# RDMA fabric) and is the path Dynamo uses elsewhere for disaggregation.
TRTLLM_EXTRA_CONFIG=$(mktemp "${TMPDIR:-/tmp}/dynamo-trtllm-sidecar.XXXXXX.yaml")
build_trtllm_override_args_with_mem \
    --merge-with-json "{\"cache_transceiver_config\": {\"backend\": \"${TRTLLM_CACHE_TRANSCEIVER_BACKEND}\"}}" \
    > "$TRTLLM_EXTRA_CONFIG"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"

print_launch_banner "Launching TensorRT-LLM OpenEngine-gRPC Sidecar (Disaggregated, 2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Prefill:     GPU ${TRTLLM_PREFILL_GPU}, gRPC ${TRTLLM_HOST}:${TRTLLM_PREFILL_GRPC_PORT}" \
    "Decode:      GPU ${TRTLLM_DECODE_GPU}, gRPC ${TRTLLM_HOST}:${TRTLLM_DECODE_GRPC_PORT}" \
    "KV transfer: ${TRTLLM_CACHE_TRANSCEIVER_BACKEND} cache transceiver" \
    "Context length:    ${TRTLLM_CONTEXT_LENGTH:-from engine report}"

python3 -m dynamo.frontend &

# TensorRT-LLM's OpenEngine gRPC listener is unauthenticated; keep it on
# loopback. The cache transceiver binds its own port on a routable address, so
# the engines still reach each other.
CUDA_VISIBLE_DEVICES="$TRTLLM_PREFILL_GPU" \
"$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc \
    --grpc-protocol openengine \
    --host "$TRTLLM_HOST" \
    --port "$TRTLLM_PREFILL_GRPC_PORT" \
    "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
    --extra_llm_api_options "$TRTLLM_EXTRA_CONFIG" \
    "${EXTRA_ARGS[@]}" &

CUDA_VISIBLE_DEVICES="$TRTLLM_DECODE_GPU" \
"$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc \
    --grpc-protocol openengine \
    --host "$TRTLLM_HOST" \
    --port "$TRTLLM_DECODE_GRPC_PORT" \
    "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
    --extra_llm_api_options "$TRTLLM_EXTRA_CONFIG" \
    "${EXTRA_ARGS[@]}" &

# `--disaggregation-mode prefill` also registers this worker under the `prefill`
# component, which is what the frontend's prefill router targets.
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-trtllm-sidecar \
    --disaggregation-mode prefill \
    --grpc-endpoint "${TRTLLM_HOST}:${TRTLLM_PREFILL_GRPC_PORT}" \
    --model-path "$MODEL" \
    "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-trtllm-sidecar \
    --disaggregation-mode decode \
    --grpc-endpoint "${TRTLLM_HOST}:${TRTLLM_DECODE_GRPC_PORT}" \
    --model-path "$MODEL" \
    "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &

wait_any_exit
