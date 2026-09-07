#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SGLang Elastic EP Scale-Up Regression Test (scale-up only)
#
# Scale-UP only: SGLang supports elastic-EP scale-up (attach a joining group,
# integrate its GPUs, redistribute experts) but NOT scale-down. The scale
# sequence is therefore monotonic:
#
#   Baseline ep=4  ->  ep=6  ->  ep=8
#
# For each scale step the script:
#   1. Launches a SGLang JOINING GROUP inside the worker pod on the idle GPUs
#      (--elastic-ep-join-mode scale --elastic-ep-join-rank-offset <current_ep>).
#      NOTE: the Dynamo operator does not launch joiners yet, so the script does
#      it manually via `kubectl exec` (tracked as a follow-up).
#   2. POSTs /engine/control/scale_elastic_ep {"new_ep_size": <target>} to the
#      leader system port (9090).
#   3. Polls /engine/control/is_scaling_elastic_ep until scale_phase reaches
#      "serving_expanded".
#   4. Snapshots GPU memory and runs a live inference to confirm serving.
#
# Usage:
#   ./run_sglang_elastic_ep_scale_up_test.sh [NAMESPACE] [DEPLOYMENT_NAME]
#
# Defaults:
#   NAMESPACE       = default
#   DEPLOYMENT_NAME = sglang-elastic-ep
#
# Prerequisites (see sglang_elastic_ep.yaml header):
#   - kubectl configured and pointing at the right cluster
#   - Deployment already applied (sglang_elastic_ep.yaml), 8 GPUs on one node
#   - SGLang >= 0.5.16 in the image; RDMA/IB available; --mooncake-ib-device set

set -uo pipefail

NS="${1:-default}"
DEPLOYMENT_NAME="${2:-sglang-elastic-ep}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
MODEL_PATH="$MODEL"  # joiner must serve the same model as the primary

# Joiner launch knobs (match sglang_elastic_ep.yaml).
DIST_INIT_ADDR="127.0.0.1:24555"
IB_DEVICE="${SGLANG_MOONCAKE_IB_DEVICE:-mlx5_0}"

echo "Namespace:  $NS"
echo "Deployment: $DEPLOYMENT_NAME"
echo "Model:      $MODEL"
echo ""

# ── Pod lookup helpers ────────────────────────────────────────────────────────
worker_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=SGLangDecodeWorker,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
    --field-selector=status.phase=Running \
    --sort-by='.metadata.name' \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

frontend_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=Frontend,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

INITIAL_WORKER=$(worker_pod)
if [ -z "$INITIAL_WORKER" ]; then
  echo "ERROR: no running SGLangDecodeWorker pod found in namespace $NS" >&2
  exit 1
fi
echo "Leader/primary pod: $INITIAL_WORKER"

# ── Wait for the leader to be Ready ───────────────────────────────────────────
echo "=== Waiting for leader pod to be Ready ==="
if ! kubectl wait pod/"$(worker_pod)" -n "$NS" --for=condition=Ready --timeout=1200s; then
  echo "ERROR: leader pod did not become Ready within 1200s" >&2
  exit 1
fi
echo "Ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Port-forwards ─────────────────────────────────────────────────────────────
# 9090 = leader system server (engine control routes). 8000 = frontend OpenAI API.
pkill -f "port-forward.*8001:9090" 2>/dev/null || true
pkill -f "port-forward.*8002:8000" 2>/dev/null || true
sleep 2

kubectl port-forward pod/"$(worker_pod)" 8001:9090 -n "$NS" &
PF_ENGINE=$!
kubectl port-forward pod/"$(frontend_pod)" 8002:8000 -n "$NS" &
PF_FRONTEND=$!
echo "Port-forwards: engine=$PF_ENGINE frontend=$PF_FRONTEND"
# Reap the port-forwards on ANY exit (including the exit 1 error paths below).
cleanup_port_forwards() { kill "$PF_ENGINE" "$PF_FRONTEND" 2>/dev/null || true; }
trap cleanup_port_forwards EXIT
sleep 5

# ── Wait for inference endpoint ───────────────────────────────────────────────
echo "=== Waiting for inference endpoint ==="
ENDPOINT_READY=0
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8002/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "Endpoint ready (checked after ~$((i * 5))s)"
    ENDPOINT_READY=1
    break
  fi
  sleep 5
done
if [ "$ENDPOINT_READY" != "1" ]; then
  echo "ERROR: inference endpoint never became ready after 300s" >&2
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
snapshot() {
  local label="$1"
  local pod
  pod=$(worker_pod)
  echo ""
  echo "--- nvidia-smi ($label) pod=$pod ---"
  kubectl exec "$pod" -n "$NS" -- \
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1
}

infer() {
  local label="$1"
  local tmp code
  echo ""
  echo "--- inference ($label) ---"
  tmp=$(mktemp)
  code=$(curl -s -m 30 -o "$tmp" -w "%{http_code}" http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2=\",\"max_tokens\":5,\"temperature\":0}")
  echo "response ($code): $(cat "$tmp")"
  if [ "$code" != "200" ] || ! python3 -c "import json,sys; d=json.load(open('$tmp')); assert isinstance(d['choices'][0]['text'], str)" 2>/dev/null; then
    echo "ERROR: inference ($label) failed (HTTP $code or invalid completion JSON)" >&2
    rm -f "$tmp"
    exit 1
  fi
  rm -f "$tmp"
}

# launch_joiner starts a SGLang joining group inside the worker pod on the idle
# GPUs, so its ranks are up before the scale request. rank_offset = current EP
# size; join_gpus = number of GPUs the joiner contributes.
launch_joiner() {
  local rank_offset="$1"
  local join_gpus="$2"
  local pod
  pod=$(worker_pod)
  local gpu_end=$((rank_offset + join_gpus))
  local visible
  visible=$(seq -s, "$rank_offset" $((gpu_end - 1)))
  echo ""
  echo "--- launching joining group: rank_offset=$rank_offset gpus=$visible ---"
  # Detached joiner: shares --dist-init-addr with the primary. Its own HTTP port
  # is never used for serving; the primary keeps serving throughout.
  kubectl exec "$pod" -n "$NS" -- bash -lc "
    CUDA_VISIBLE_DEVICES=$visible nohup sglang serve \
      --model-path $MODEL_PATH \
      --trust-remote-code \
      --tp $join_gpus --dp $join_gpus \
      --enable-dp-attention --enable-dp-lm-head \
      --moe-a2a-backend nixl --deepep-mode low_latency \
      --elastic-ep-backend mooncake --mooncake-ib-device $IB_DEVICE \
      --enable-eplb --ep-num-redundant-experts 24 \
      --elastic-ep-initial-size 4 --max-ep-size 8 \
      --elastic-ep-join-mode scale --elastic-ep-join-rank-offset $rank_offset \
      --dist-init-addr $DIST_INIT_ADDR \
      --host 127.0.0.1 --port $((30000 + rank_offset)) \
      > /tmp/sglang_joiner_${rank_offset}.log 2>&1 &
    echo joiner-launched
  " 2>&1
  # Give the joiner time to boot and register its GPUs with the primary.
  sleep 30
}

# scale_up attaches a joiner then drives the leader's scale_elastic_ep route.
scale_up() {
  local from_ep="$1"
  local to_ep="$2"
  local join_gpus=$((to_ep - from_ep))
  echo ""
  echo "=========================================="
  echo "SCALE UP ep=$from_ep -> ep=$to_ep at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  leader pod: $(worker_pod)"
  echo "=========================================="

  launch_joiner "$from_ep" "$join_gpus"

  echo "--- request: POST /engine/control/scale_elastic_ep {\"new_ep_size\": $to_ep} ---"
  RESP=$(curl -s -X POST http://localhost:8001/engine/control/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d "{\"new_ep_size\": $to_ep}" \
    --max-time 700)
  echo "--- response ---"
  echo "$RESP"
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  if [ "$STATUS" != "ok" ]; then
    echo "ERROR: scale_elastic_ep returned status='$STATUS' (expected 'ok')" >&2
    exit 1
  fi

  # Poll until SGLang reports the scale finished (scale_phase == serving_expanded).
  echo "--- polling /engine/control/is_scaling_elastic_ep ---"
  local deadline=$((SECONDS + 300))
  local scale_confirmed=0
  while [ $SECONDS -lt $deadline ]; do
    STATE=$(curl -s -m 10 http://localhost:8001/engine/control/is_scaling_elastic_ep)
    PHASE=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('scale_phase',''))" 2>/dev/null)
    SCALING=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_scaling_elastic_ep',''))" 2>/dev/null)
    if [ "$SCALING" = "False" ] && [ "$PHASE" = "serving_expanded" ]; then
      echo "scale complete: $STATE"
      scale_confirmed=1
      break
    fi
    sleep 3
  done
  if [ "$scale_confirmed" != "1" ]; then
    echo "ERROR: scale-up to ep=$to_ep did not reach serving_expanded within 300s (last state: $STATE)" >&2
    exit 1
  fi

  snapshot "after ep=$to_ep"
  infer "ep=$to_ep"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "BASELINE ep=4 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
snapshot "baseline ep=4"
infer "ep=4"

# ── Scale-up steps (monotonic; scale-down is unsupported by SGLang) ────────────
scale_up 4 6   # attach a 2-GPU joiner: ep=4 -> ep=6
scale_up 6 8   # attach a 2-GPU joiner: ep=6 -> ep=8 (full node)

echo ""
echo "=== ALL SCALE-UP STEPS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
# Port-forwards are reaped by the EXIT trap.
