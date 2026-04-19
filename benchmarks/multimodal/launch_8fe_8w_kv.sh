#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch 8 frontend replicas (KV routing) + 8 vllm workers on 8 GPUs.
# A round-robin LB proxy on port 9000 distributes traffic across all frontends.
#
# Run in a SEPARATE detached MCP window (not in the same shell as aiperf):
#   run(command="cd /workspace && bash benchmarks/multimodal/launch_8fe_8w_kv.sh",
#       name="topo", detach=true)
#
# Then run aiperf in another window. See bench_8fe_kv_multiturn.sh for the
# multi-turn pinassistant workload, or run aiperf manually:
#   aiperf profile --model "..." --url "http://localhost:9000" --endpoint-type chat ...

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

export PYTHONHASHSEED=0

MODEL="${DYN_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
NUM_FE="${DYN_NUM_FE:-8}"
NUM_WORKERS="${DYN_NUM_WORKERS:-8}"
LB_PORT="${DYN_LB_PORT:-9000}"
FE_BASE_PORT="${DYN_FE_BASE_PORT:-8000}"
WORKER_SYS_BASE="${DYN_WORKER_SYS_BASE:-8080}"
FE_SYS_BASE="${DYN_FE_SYS_BASE:-8090}"
ZMQ_BASE="${DYN_ZMQ_BASE:-20080}"

echo "=== 8-FE KV Router Benchmark ==="
echo "Model:    $MODEL"
echo "Frontend: $NUM_FE replicas (ports ${FE_BASE_PORT}-$((FE_BASE_PORT + NUM_FE - 1)))"
echo "Workers:  $NUM_WORKERS (GPUs 0-$((NUM_WORKERS - 1)))"
echo "LB proxy: port $LB_PORT"
echo ""

# --- Infrastructure ---
echo "[infra] Starting etcd + nats..."
etcd --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --data-dir /tmp/etcd &

nats-server -js -m 8222 -p 4222 -sd /tmp/nats &
sleep 3

# --- Frontends ---
echo "[frontend] Starting $NUM_FE frontends with KV routing..."
for i in $(seq 0 $((NUM_FE - 1))); do
  RESET_FLAG=""
  [[ $i -eq 0 ]] && RESET_FLAG="--router-reset-states"

  DYN_SYSTEM_PORT=$((FE_SYS_BASE + i)) \
  DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD=1.0 \
  python -m dynamo.frontend \
    --router-mode kv \
    $RESET_FLAG \
    --http-port $((FE_BASE_PORT + i)) &

  echo "  FE $i -> http://localhost:$((FE_BASE_PORT + i)) (sys=$((FE_SYS_BASE + i)))"
done

# --- Workers ---
echo "[workers] Starting $NUM_WORKERS vllm workers..."
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  DYN_SYSTEM_PORT=$((WORKER_SYS_BASE + i)) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600 + i)) \
  CUDA_VISIBLE_DEVICES=$i \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enable-multimodal \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((ZMQ_BASE + i))\",\"enable_kv_cache_events\":true}" &

  echo "  Worker $i -> GPU $i (sys=$((WORKER_SYS_BASE + i)), zmq=$((ZMQ_BASE + i)))"
done

# --- LB Proxy ---
BACKENDS=$(seq -s, $FE_BASE_PORT $((FE_BASE_PORT + NUM_FE - 1)))
echo "[lb] Starting round-robin proxy on :$LB_PORT -> [$BACKENDS]"
python benchmarks/multimodal/lb_proxy.py \
  --listen-port "$LB_PORT" \
  --backends "$BACKENDS" &

echo ""
echo "=== All processes launched. Waiting... ==="
echo "  LB endpoint: http://localhost:$LB_PORT/v1/chat/completions"
echo "  Check models: curl http://localhost:$FE_BASE_PORT/v1/models"
wait
