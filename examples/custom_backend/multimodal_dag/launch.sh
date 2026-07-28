#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the minimal two-GPU custom vision DAG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../common/launch_utils.sh"

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PUBLIC_MODEL="multimodal-dag"
ENCODER_GPU="${DYN_ENCODER_GPU:-0}"
VLLM_GPU="${DYN_VLLM_GPU:-1}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${DYN_MAX_NUM_SEQS:-1}"
SYSTEM_PORT_BASE="${DYN_SYSTEM_PORT_BASE:-8081}"
READINESS_TIMEOUT="${DYN_READINESS_TIMEOUT:-900}"
PYTHON_BIN="${DYN_PYTHON_BIN:-python}"
RUNTIME_DIR="$(mktemp -d -t dynamo-multimodal-dag.XXXXXX)"
DISCOVERY_DIR="$RUNTIME_DIR/discovery"
LOG_DIR="$RUNTIME_DIR/logs"

mkdir -p "$DISCOVERY_DIR" "$LOG_DIR"
trap dynamo_exit_trap EXIT

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="$DISCOVERY_DIR"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export DYN_HTTP_PORT="$HTTP_PORT"

read -r -a VLLM_GPU_MEM_ARGS <<< "$(build_vllm_gpu_mem_args)"
if [[ ${#VLLM_GPU_MEM_ARGS[@]} -eq 0 ]]; then
    VLLM_GPU_MEM_ARGS=(--gpu-memory-utilization "${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.8}")
fi

print_launch_banner --no-curl \
    "Custom Vision DAG — two-GPU PoC" \
    "$MODEL" \
    "$HTTP_PORT" \
    "Public model: $PUBLIC_MODEL" \
    "Encoder GPU:  $ENCODER_GPU" \
    "vLLM GPU:     $VLLM_GPU" \
    "Logs:         $LOG_DIR"

cd "$REPO_ROOT"

echo "[1/5] Starting frontend"
DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE)) \
    "$PYTHON_BIN" -m dynamo.frontend \
    >"$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "[2/5] Starting custom vision encoder on GPU $ENCODER_GPU"
CUDA_VISIBLE_DEVICES="$ENCODER_GPU" \
DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + 1)) \
    "$PYTHON_BIN" -m \
    examples.custom_backend.multimodal_dag.vision_encoder_worker \
    --model "$MODEL" \
    >"$LOG_DIR/vision_encoder.log" 2>&1 &
ENCODER_PID=$!

echo "[3/5] Starting dummy classifier"
DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + 2)) \
    "$PYTHON_BIN" -m \
    examples.custom_backend.multimodal_dag.classifier_worker \
    >"$LOG_DIR/classifier.log" 2>&1 &
CLASSIFIER_PID=$!

echo "[4/5] Starting vLLM decoder on GPU $VLLM_GPU"
CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + 3)) \
    "$PYTHON_BIN" -m dynamo.vllm \
    --model "$MODEL" \
    --endpoint multimodal_dag.vllm.generate \
    --endpoint-types internal \
    --dtype bfloat16 \
    --enable-multimodal \
    --enable-mm-embeds \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${VLLM_GPU_MEM_ARGS[@]}" \
    "$@" \
    >"$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

echo "[5/5] Starting orchestrator"
DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + 4)) \
    "$PYTHON_BIN" -m \
    examples.custom_backend.multimodal_dag.orchestrator_worker \
    --model "$MODEL" \
    >"$LOG_DIR/orchestrator.log" 2>&1 &
ORCHESTRATOR_PID=$!

PIDS=(
    "$FRONTEND_PID"
    "$ENCODER_PID"
    "$CLASSIFIER_PID"
    "$VLLM_PID"
    "$ORCHESTRATOR_PID"
)

model_is_ready() {
    curl -sf --max-time 2 "http://localhost:$HTTP_PORT/v1/models" |
        "$PYTHON_BIN" -c \
            'import json, sys; data=json.load(sys.stdin); expected=sys.argv[1]; raise SystemExit(0 if any(item.get("id") == expected for item in data.get("data", [])) else 1)' \
            "$PUBLIC_MODEL"
}

echo "Waiting up to ${READINESS_TIMEOUT}s for public model $PUBLIC_MODEL"
READY_START=$SECONDS
while (( SECONDS - READY_START < READINESS_TIMEOUT )); do
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "A component exited during startup. Recent logs:" >&2
            tail -n 40 "$LOG_DIR"/*.log >&2 || true
            exit 1
        fi
    done
    if model_is_ready; then
        echo "Public model is ready: http://localhost:$HTTP_PORT/v1/models"
        echo "Run: $PYTHON_BIN -m examples.custom_backend.multimodal_dag.client"
        wait_any_exit
    fi
    sleep 1
done

echo "Timed out waiting for $PUBLIC_MODEL. Recent logs:" >&2
tail -n 40 "$LOG_DIR"/*.log >&2 || true
exit 1
