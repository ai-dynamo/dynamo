#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lightseek-powered exact MM-aware routing.
#
# Architecture:
#   HTTP client
#     -> Rust frontend
#          - resolves the image-placeholder token id via lightseek's
#            per-model ModelProcessorSpec (Qwen3-VL, Qwen2.5-VL, Qwen2-VL,
#            LLaVA-NeXT, LLaVA-1.5, Phi-3-vision, Llama-4, Kimi-K2.5);
#            each spec reads the appropriate config.json field
#          - lightseek `calculate_num_tokens(W,H)` per image (header-only fetch)
#          - expands placeholder -> N copies in routing_token_ids
#          - hashes URL (xxh3) -> u64 -> 64-char hex -> extra_args["mm_hashes"]
#          - builds block_mm_infos and feeds the KV router
#     -> N x vLLM worker
#          - publishes KV events via zmq (one port per worker)
#          - vLLM's mm_uuid is the frontend's hex string -> events match
#            the router's routing-side block hashes -> bit-exact cache hits
#
# Build prerequisite (run once inside the dynamo dev container):
#   cd /workspace/lib/bindings/python && maturin develop --release --features lightseek-mm
#
# Run inside the dynamo container that already has /workspace mounted.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"
# shellcheck source=../../../common/gpu_utils.sh
source "${SCRIPT_DIR}/../../../common/gpu_utils.sh"
# shellcheck source=../../../common/launch_utils.sh
source "${SCRIPT_DIR}/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-lightseek-poc}"
HTTP_PORT="${HTTP_PORT:-${DYN_HTTP_PORT:-8000}}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# --single-gpu / SINGLE_GPU: Packs all workers onto GPU 0 for functional
# testing on machines with a single GPU.  Reduces performance by sharing
# GPU memory between workers; production deployments should leave it false.
SINGLE_GPU="${SINGLE_GPU:-false}"
NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
VLLM_SYSTEM_PORT_BASE="${VLLM_SYSTEM_PORT_BASE:-18081}"
KV_EVENTS_PORT_BASE="${KV_EVENTS_PORT_BASE:-5557}"
DYN_LOG_VAL="${DYN_LOG:-info,lightseek_mm=debug,dynamo_kv_router::scheduling=debug}"

# Pass-through extra args for `python -m dynamo.vllm`.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --single-gpu) SINGLE_GPU="true"; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--model NAME] [--num-workers N] [--single-gpu] [EXTRA_VLLM_ARGS...]

Env vars:
  MODEL                       (default Qwen/Qwen3-VL-2B-Instruct)
  NUM_WORKERS                 (default 2)
  GPU_MEMORY_UTILIZATION      (default 0.20 — sized for 2 workers on 1 GPU)
  HTTP_PORT                   (default 8000)
  BLOCK_SIZE                  (default 16)
  MAX_MODEL_LEN               (default 4096)
  KV_EVENTS_PORT_BASE         (default 5557 — worker i uses port BASE + (i-1))
  DYN_LOG                     (default info + lightseek_mm + scheduling debug)

Routing test (run after the script reports "All services are ready"):
  # Same image twice -> 2nd request should pin to same worker (high overlap_blocks)
  IMG_A=http://images.cocodataset.org/val2017/000000039769.jpg
  IMG_B=http://images.cocodataset.org/val2017/000000000139.jpg
  for url in "\$IMG_A" "\$IMG_A" "\$IMG_B"; do
    curl -s http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"'"\${MODEL}"'","messages":[{"role":"user","content":[
            {"type":"text","text":"Describe this image."},
            {"type":"image_url","image_url":{"url":"'\$url'"}}]}],"max_tokens":4}' \\
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage'])"
  done

Watch the frontend log for 'Selected worker' + 'Formula' lines. Same-image
requests should route to the same worker_id and report ~26 effective cached
blocks; the different-image request should route to a fresh worker (or the
same one with low overlap).
EOF
            exit 0
            ;;
        *) PASSTHRU_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Lightseek MM Exact Routing Launch ==="
echo "MODEL=${MODEL}"
echo "NUM_WORKERS=${NUM_WORKERS}, BLOCK_SIZE=${BLOCK_SIZE}, GPU_MEM_FRAC=${GPU_MEMORY_UTILIZATION}"
echo "HTTP_PORT=${HTTP_PORT}, NAMESPACE=${NAMESPACE}"

# Clear the trap inside the handler so `kill 0` (which SIGTERMs the script
# itself) doesn't re-enter this trap and loop forever.
trap 'trap - EXIT INT TERM; echo; kill 0' EXIT INT TERM

wait_ready() {
    local url="$1" name="$2" timeout_s="${3:-900}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for ${name} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
            echo "${name} is ready"
            return 0
        fi
        sleep 1
    done
    return 1
}

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
    "DYN_MM_ALLOW_INTERNAL=1"
)

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

for i in $(seq 1 "${NUM_WORKERS}"); do
    WORKER_PORT=$((VLLM_SYSTEM_PORT_BASE + (i - 1) * 2))
    KV_EVENTS_PORT=$((KV_EVENTS_PORT_BASE + (i - 1)))
    if [[ "${SINGLE_GPU}" == "true" ]]; then GPU_ID=0; else GPU_ID=$((i - 1)); fi

    KV_EVENTS_CONFIG="{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\"}"

    echo "--- launching vLLM worker $i (GPU=${GPU_ID}, system_port=${WORKER_PORT}, kv_events=tcp://*:${KV_EVENTS_PORT}) ---"
    env "${COMMON_ENV[@]}" \
        "DYN_SYSTEM_PORT=${WORKER_PORT}" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    python -m dynamo.vllm \
        --model "${MODEL}" \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --max-model-len "${MAX_MODEL_LEN}" \
        --kv-events-config "${KV_EVENTS_CONFIG}" \
        ${GPU_MEM_ARGS} ${VLLM_EXTRA_ARGS} "${PASSTHRU_ARGS[@]}" &
    wait_ready "http://127.0.0.1:${WORKER_PORT}/health" "vLLM backend $i"
done

echo "=== Starting frontend (KV router, lightseek MM exact routing) ==="
env "${COMMON_ENV[@]}" \
    "DYN_LOG=${DYN_LOG_VAL}" \
python -m dynamo.frontend \
    --http-port "${HTTP_PORT}" \
    --router-mode kv \
    --kv-cache-block-size "${BLOCK_SIZE}" &

echo "Waiting for frontend to accept requests ..."
DEADLINE=$((SECONDS + 300))
while (( SECONDS < DEADLINE )); do
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        2>/dev/null || echo "000")
    [[ "$HTTP_CODE" == "200" ]] && { echo "Frontend ready"; break; }
    sleep 2
done

echo
echo "=== All services are ready ==="
echo "Frontend:        http://127.0.0.1:${HTTP_PORT}"
for i in $(seq 1 "${NUM_WORKERS}"); do
    echo "Worker $i health: http://127.0.0.1:$((VLLM_SYSTEM_PORT_BASE + (i - 1) * 2))/health"
    echo "Worker $i kv-events: tcp://*:$((KV_EVENTS_PORT_BASE + (i - 1)))"
done
echo
echo "Architecture: Rust frontend + lightseek -> ${NUM_WORKERS}x vLLM workers"
echo "  - mm_hashes forwarded as multi_modal_uuids -> bit-exact match with vLLM KV events"
echo "  - Image dims via header-only HTTP fetch (Range: bytes=0-65535)"
echo "  - No PyO3, no GIL, no Python deps in the routing path"
echo
echo "Routing test (recommended):"
echo "  IMG_A=http://images.cocodataset.org/val2017/000000039769.jpg"
echo "  IMG_B=http://images.cocodataset.org/val2017/000000000139.jpg"
echo "  for url in \"\$IMG_A\" \"\$IMG_A\" \"\$IMG_B\"; do"
echo "    curl -s http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":["
echo "            {\"type\":\"text\",\"text\":\"Describe this image.\"},"
echo "            {\"type\":\"image_url\",\"image_url\":{\"url\":\"'\$url'\"}}]}],\"max_tokens\":4}' \\"
echo "      | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['usage'])\""
echo "  done"
echo
echo "Watch frontend logs for 'Selected worker: ... worker_id=...' and"
echo "'Formula for worker_id=X dp_rank=0 with N effective cached blocks'."
echo
echo "Press Ctrl+C to stop all services"

wait_any_exit
