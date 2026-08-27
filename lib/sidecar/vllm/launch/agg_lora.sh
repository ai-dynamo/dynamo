#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with LoRA through vLLM's native gRPC server (1 GPU).
#
# Requires a vLLM build containing vllm-project/vllm#52840, which adds the
# LoadLora, UnloadLora, and ListLoras control RPCs and per-request lora_name.
#
# For s3:// sources, start MinIO first with examples/backends/vllm/launch/lora/setup_minio.sh.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"   # build_vllm_gpu_mem_args
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [vLLM engine options...]"
            echo
            echo "Additional options are passed to the managed vLLM engine."
            echo
            echo "Environment overrides:"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  LORA_NAME               Adapter name used in the printed examples"
            echo "  MAX_LORAS               GPU-resident adapter capacity (default: 4)"
            echo "  MAX_LORA_RANK           Maximum adapter rank (default: 64)"
            echo "  DYN_LORA_PATH           S3 download root (default: /tmp/dynamo_loras_minio)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment (default: 0)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  VLLM_RS_HTTP_PORT       vLLM HTTP port (default: 8100)"
            echo "  VLLM_GRPC_PORT          vLLM gRPC port (default: 50051)"
            echo "  MAX_MODEL_LEN           Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS     Maximum concurrent sequences (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES  KV cache cap when not profiling (default: 1119388000)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"
VLLM_GRPC_PORT="${VLLM_GRPC_PORT:-50051}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# S3/MinIO credentials for s3:// adapter sources.
export AWS_ENDPOINT="${AWS_ENDPOINT:-http://localhost:9000}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ALLOW_HTTP="${AWS_ALLOW_HTTP:-true}"

# The sidecar exposes the lifecycle routes only when this is truthy, vLLM
# advertises supports_lora, and max_loras > 0.
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH="${DYN_LORA_PATH:-/tmp/dynamo_loras_minio}"
mkdir -p "$DYN_LORA_PATH"

# vLLM resolves adapters by absolute path, so it must accept the roots the
# sidecar downloads into. Colocated here; separate containers must mount these
# at identical absolute paths.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES="${VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES:-${DYN_LORA_PATH}:${HF_CACHE}}"

DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
print_launch_banner --no-curl "Launching vLLM Native-gRPC Sidecar + LoRA (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "vLLM HTTP:  http://127.0.0.1:${VLLM_RS_HTTP_PORT}" \
    "vLLM gRPC:  127.0.0.1:${VLLM_GRPC_PORT}"
echo ""
echo "Once running, test with:"
echo ""
echo "  # Load an adapter (file://, hf://, or s3://). The returned lora_id is assigned by vLLM."
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"hf://${LORA_NAME}\"}}' | jq ."
echo ""
echo "  # List loaded adapters"
echo "  curl -s http://localhost:${SYSTEM_PORT}/v1/loras | jq ."
echo ""
echo "  # Adapter inference"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}], \"max_tokens\": 300, \"temperature\": 0.0}' | jq ."
echo ""
echo "  # Base-model inference, for comparison"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}], \"max_tokens\": 300, \"temperature\": 0.0}' | jq ."
echo ""
echo "  # Unload"
echo "  curl -X DELETE http://localhost:${SYSTEM_PORT}/v1/loras/${LORA_NAME}"
echo ""
echo "=========================================="

python -m dynamo.frontend &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_RS_HTTP_PORT" \
    --grpc-port "$VLLM_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --enable-lora \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="$SYSTEM_PORT" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_GRPC_PORT}" &

wait_any_exit
