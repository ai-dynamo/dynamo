#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving through TensorRT-LLM's OpenEngine gRPC server (1 GPU).
#
# Run with `TRTLLM_PYTHON` from `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202610290007`.
# The bindings it needs are installed by the helper below.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# Resolved relative to this script, not via $DYNAMO_HOME: some runtime images
# (e.g. vllm_runtime.Dockerfile) bake DYNAMO_HOME to a minimal install path
# with no examples/ directory, which would silently override this and break
# sourcing. Matches examples/backends/trtllm/launch/agg.sh's own approach.
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/common.sh"    # OpenEngine bindings, context length, config merge

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
            echo "Additional options are passed to the TensorRT-LLM engine."
            echo
            echo "Environment overrides:"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  TRTLLM_PYTHON           Python with TensorRT-LLM installed (default: python3)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment (default: 0)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  TRTLLM_GRPC_PORT        TensorRT-LLM gRPC port (default: 30001)"
            echo "  AGG_ENGINE_ARGS         Base TensorRT-LLM YAML/JSON options file"
            echo "  TRTLLM_CONTEXT_LENGTH   Model context length, applied to both the engine and the"
            echo "                          sidecar (default: 4096; unset when --max_seq_len is given)"
            echo "  TRTLLM_DISABLE_GUIDED_DECODING=1"
            echo "                          Disable xgrammar for models that do not support it"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

resolve_sidecar trtllm SIDECAR_CMD

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
TRTLLM_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 1 "${TRTLLM_GRPC_PORT:-30001}")
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# `--extra_llm_api_options` (alias `--config`) is last-wins, not additive. The
# launcher reserves it for required and profiler overrides; callers can supply
# base options through AGG_ENGINE_ARGS, which are merged below.
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --extra_llm_api_options|--extra_llm_api_options=*|--config|--config=*)
            echo "Cannot forward ${arg%%=*}: use AGG_ENGINE_ARGS instead." >&2
            exit 1
            ;;
    esac
done

trtllm_resolve_context_length "${EXTRA_ARGS[@]}"

trtllm_ensure_openengine_bindings "$TRTLLM_PYTHON"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
# Dynamo sends a required or named tool_choice as a JSON schema, which
# TensorRT-LLM enforces only with a guided-decoding backend.
TRTLLM_EXTRA_CONFIG=$(mktemp "${TMPDIR:-/tmp}/dynamo-trtllm-sidecar.XXXXXX.yaml")
TRTLLM_REQUIRED_OVERRIDES='{"guided_decoding_backend": "xgrammar"}'
if [[ "${TRTLLM_DISABLE_GUIDED_DECODING:-0}" == "1" ]]; then
    TRTLLM_REQUIRED_OVERRIDES='{"guided_decoding_backend": null}'
fi
TRTLLM_OVERRIDES=$(build_trtllm_override_args_with_mem \
    --merge-with-json "$TRTLLM_REQUIRED_OVERRIDES")
trtllm_write_merged_config \
    "${AGG_ENGINE_ARGS:-}" \
    "$TRTLLM_OVERRIDES" \
    "$TRTLLM_EXTRA_CONFIG" \
    "$TRTLLM_PYTHON"

print_launch_banner "Launching TensorRT-LLM OpenEngine-gRPC Sidecar (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "TensorRT-LLM gRPC: 127.0.0.1:${TRTLLM_GRPC_PORT}" \
    "Context length:    ${TRTLLM_CONTEXT_LENGTH:-from engine report}"

python3 -m dynamo.frontend &

# TensorRT-LLM's OpenEngine gRPC listener is unauthenticated; keep it on loopback.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc \
    --grpc-protocol openengine \
    --host 127.0.0.1 \
    --port "$TRTLLM_GRPC_PORT" \
    "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
    --extra_llm_api_options "$TRTLLM_EXTRA_CONFIG" \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "127.0.0.1:${TRTLLM_GRPC_PORT}" \
    --model-path "$MODEL" \
    "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &

wait_any_exit
