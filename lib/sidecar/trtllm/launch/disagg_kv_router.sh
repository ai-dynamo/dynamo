#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated KV-aware serving (2 prefill, 2 decode GPUs).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

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
            echo "Additional options are passed to all four TensorRT-LLM engines."
            echo
            echo "Environment overrides:"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            echo "  MODEL                     Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  TRTLLM_PYTHON             Python with TensorRT-LLM installed (default: python3)"
            echo "  TRTLLM_WORKER{1..4}_GPU            GPU per worker (default: inherited devices, otherwise 0..3)"
            echo "  TRTLLM_WORKER{1..4}_GRPC_PORT      gRPC ports (defaults: 30001, 30011, 30021, 30031)"
            echo "  TRTLLM_WORKER{1..4}_KV_EVENT_PORT  ZMQ ports (defaults: 5557..5560)"
            echo "  DYN_SYSTEM_PORT{1..4}             Sidecar ports (defaults: 8081..8084)"
            echo "  TRTLLM_CACHE_TRANSCEIVER_BACKEND  KV transfer backend (default: NIXL)"
            echo "  TRTLLM_CONTEXT_LENGTH             Context length (default: 4096)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=, read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES-0,1,2,3}"
for worker in 1 2 3 4; do
    gpu_var="TRTLLM_WORKER${worker}_GPU"
    if [[ -z "${!gpu_var:-}" ]]; then
        gpu="${visible_gpus[worker - 1]:-}"
        if [[ -z "$gpu" ]]; then
            echo "Missing GPU for worker ${worker}: extend CUDA_VISIBLE_DEVICES or set ${gpu_var}." >&2
            exit 1
        fi
        printf -v "$gpu_var" '%s' "$gpu"
    fi
done

resolve_sidecar trtllm SIDECAR_CMD

TRTLLM_CONFIG_FILES=()
trtllm_exit_trap() {
    local rc=$?
    for config in "${TRTLLM_CONFIG_FILES[@]}"; do
        rm -f -- "$config"
    done
    echo "Cleaning up..."
    dynamo_reap_and_exit "$rc"
}
trap trtllm_exit_trap EXIT

TRTLLM_PYTHON="${TRTLLM_PYTHON:-python3}"
TRTLLM_CACHE_TRANSCEIVER_BACKEND="${TRTLLM_CACHE_TRANSCEIVER_BACKEND:-NIXL}"
# Reject config overrides that would replace required engine settings.
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --extra_llm_api_options|--extra_llm_api_options=*|--config|--config=*)
            echo "Cannot forward ${arg%%=*}: this launcher supplies per-worker engine configs." >&2
            exit 1
            ;;
    esac
done

trtllm_resolve_context_length "${EXTRA_ARGS[@]}"
trtllm_ensure_openengine_bindings "$TRTLLM_PYTHON"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "TensorRT-LLM OpenEngine KV-aware sidecars (2 prefill, 2 decode)" "$MODEL" "$HTTP_PORT" \
    "KV transfer: ${TRTLLM_CACHE_TRANSCEIVER_BACKEND} cache transceiver"
python3 -m dynamo.frontend --router-mode kv &

for worker in 1 2 3 4; do
    role=prefill
    if (( worker > 2 )); then role=decode; fi
    gpu_var="TRTLLM_WORKER${worker}_GPU"
    grpc_var="TRTLLM_WORKER${worker}_GRPC_PORT"
    kv_var="TRTLLM_WORKER${worker}_KV_EVENT_PORT"
    system_var="DYN_SYSTEM_PORT${worker}"
    gpu="${!gpu_var}"
    grpc_port=$(dyn_port DYN_ENGINE_GRPC_PORT "$worker" "${!grpc_var:-$((30001 + 10 * (worker - 1)))}")
    kv_port=$(dyn_port DYN_VLLM_KV_EVENT_PORT "$worker" "${!kv_var:-$((5556 + worker))}")
    system_port="${!system_var:-$((8080 + worker))}"
    config=$(mktemp "${TMPDIR:-/tmp}/dynamo-trtllm-disagg-kv.XXXXXX.yaml")
    TRTLLM_CONFIG_FILES+=("$config")
    build_trtllm_override_args_with_mem \
        --merge-with-json "$(cat <<EOF
{
  "cache_transceiver_config": {
    "backend": "${TRTLLM_CACHE_TRANSCEIVER_BACKEND}"
  },
  "guided_decoding_backend": "xgrammar",
  "kv_cache_config": {
    "use_kv_cache_manager_v2": true,
    "enable_block_reuse": true,
    "kv_cache_event_hash_algo": "v1_block_key",
    "kv_events_config": {
      "enable_kv_cache_events": true,
      "endpoint": "tcp://127.0.0.1:${kv_port}"
    }
  }
}
EOF
)" \
        > "$config"
    echo "${role} ${worker}: GPU ${gpu}, gRPC 127.0.0.1:${grpc_port}, KV events tcp://127.0.0.1:${kv_port}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
        --grpc --grpc-protocol openengine --openengine-enable-load-metrics \
        --host 127.0.0.1 --port "$grpc_port" \
        "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
        --extra_llm_api_options "$config" "${EXTRA_ARGS[@]}" &
    DYN_SYSTEM_PORT="$system_port" "${SIDECAR_CMD[@]}" \
        --disaggregation-mode "$role" --grpc-endpoint "127.0.0.1:${grpc_port}" \
        --model-path "$MODEL" "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &
done

wait_any_exit
