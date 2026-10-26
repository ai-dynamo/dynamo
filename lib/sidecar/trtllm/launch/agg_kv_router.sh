#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated KV-aware serving (2 GPUs).

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
            echo "Additional options are passed to both TensorRT-LLM engines."
            echo
            echo "Environment overrides:"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  TRTLLM_PYTHON           Python with TensorRT-LLM installed (default: python3)"
            echo "  TRTLLM_WORKER1_GPU      First GPU (default: inherited device 1, otherwise 0)"
            echo "  TRTLLM_WORKER2_GPU      Second GPU (default: inherited device 2, otherwise 1)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1        First sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2        Second sidecar system port (default: 8082)"
            echo "  TRTLLM_WORKER1_GRPC_PORT     First gRPC port (default: 30001)"
            echo "  TRTLLM_WORKER2_GRPC_PORT     Second gRPC port (default: 30011)"
            echo "  TRTLLM_WORKER1_KV_EVENT_PORT First ZMQ port (default: 5557)"
            echo "  TRTLLM_WORKER2_KV_EVENT_PORT Second ZMQ port (default: 5558)"
            echo "  TRTLLM_CONTEXT_LENGTH   Context length for both engines and sidecars (default: 4096)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=, read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES-0,1}"
for worker in 1 2; do
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
TRTLLM_WORKER1_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 1 "${TRTLLM_WORKER1_GRPC_PORT:-30001}")
TRTLLM_WORKER2_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 2 "${TRTLLM_WORKER2_GRPC_PORT:-30011}")
TRTLLM_WORKER1_KV_EVENT_PORT=$(dyn_port DYN_VLLM_KV_EVENT_PORT 1 "${TRTLLM_WORKER1_KV_EVENT_PORT:-5557}")
TRTLLM_WORKER2_KV_EVENT_PORT=$(dyn_port DYN_VLLM_KV_EVENT_PORT 2 "${TRTLLM_WORKER2_KV_EVENT_PORT:-5558}")

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

for worker in 1 2; do
    port_var="TRTLLM_WORKER${worker}_KV_EVENT_PORT"
    config=$(mktemp "${TMPDIR:-/tmp}/dynamo-trtllm-agg-kv.XXXXXX.yaml")
    TRTLLM_CONFIG_FILES+=("$config")
    build_trtllm_override_args_with_mem \
        --merge-with-json "$(cat <<EOF
{
  "guided_decoding_backend": "xgrammar",
  "kv_cache_config": {
    "use_kv_cache_manager_v2": true,
    "enable_block_reuse": true,
    "kv_cache_event_hash_algo": "v1_block_key",
    "kv_events_config": {
      "enable_kv_cache_events": true,
      "endpoint": "tcp://127.0.0.1:${!port_var}"
    }
  }
}
EOF
)" \
        > "$config"
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "TensorRT-LLM OpenEngine KV-aware sidecars (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Worker 1: GPU ${TRTLLM_WORKER1_GPU}, gRPC ${TRTLLM_WORKER1_GRPC_PORT}, KV ${TRTLLM_WORKER1_KV_EVENT_PORT}" \
    "Worker 2: GPU ${TRTLLM_WORKER2_GPU}, gRPC ${TRTLLM_WORKER2_GRPC_PORT}, KV ${TRTLLM_WORKER2_KV_EVENT_PORT}"

python3 -m dynamo.frontend --router-mode kv &
for worker in 1 2; do
    gpu_var="TRTLLM_WORKER${worker}_GPU"
    grpc_var="TRTLLM_WORKER${worker}_GRPC_PORT"
    system_var="DYN_SYSTEM_PORT${worker}"
    system_port="${!system_var:-$((8080 + worker))}"
    CUDA_VISIBLE_DEVICES="${!gpu_var}" \
    "$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
        --grpc --grpc-protocol openengine --openengine-enable-load-metrics \
        --host 127.0.0.1 --port "${!grpc_var}" \
        "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
        --extra_llm_api_options "${TRTLLM_CONFIG_FILES[worker - 1]}" "${EXTRA_ARGS[@]}" &
    DYN_SYSTEM_PORT="$system_port" "${SIDECAR_CMD[@]}" \
        --grpc-endpoint "127.0.0.1:${!grpc_var}" --model-path "$MODEL" \
        "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &
done

wait_any_exit
