#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Elastic EP Scaling Regression Test
#
# Runs a full 6-step scale sequence on a live elastic EP deployment:
#   Baseline (dp=2) → dp=3 → dp=4 → dp=3 → dp=2 → dp=4 → dp=2
#
# After each step captures:
#   - Full API request + response
#   - nvidia-smi GPU memory for all 4 GPUs
#   - ps aux PIDs for Ray actors (DPMoEEngineCoreActor, RayWorkerWrapper)
#   - Live inference result with latency
#
# Usage:
#   ./run_elastic_ep_scale_test.sh [NAMESPACE] [DEPLOYMENT_NAME]
#
# Defaults:
#   NAMESPACE       = default
#   DEPLOYMENT_NAME = vllm-elastic-ep-demo
#
# Prerequisites:
#   - kubectl configured and pointing at the right cluster
#   - Deployment already applied (see moe_elastic_ep_demo.yaml)
#   - Ports 8001 and 8002 free on localhost


set -uo pipefail

NS="${1:-default}"
DEPLOYMENT_NAME="${2:-vllm-elastic-ep-demo}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"

echo "Namespace:  $NS"
echo "Model:      $MODEL"
echo ""

# ── Pod lookup helpers ────────────────────────────────────────────────────────
# Always re-resolved from the cluster so the script handles pod restarts
# and works regardless of the randomly-generated pod name suffix.

worker_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=VllmDecodeWorker" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

frontend_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=Frontend" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

# Verify at least one worker pod exists before proceeding
INITIAL_POD=$(worker_pod)
if [ -z "$INITIAL_POD" ]; then
  echo "ERROR: no running VllmDecodeWorker pod found in namespace $NS" >&2
  exit 1
fi
echo "Worker pod (at start): $INITIAL_POD"

# ── Wait for pod ready ────────────────────────────────────────────────────────
echo "=== Waiting for worker pod to be Ready ==="
kubectl wait pod/"$(worker_pod)" -n "$NS" --for=condition=Ready --timeout=900s
echo "Ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Port-forwards ─────────────────────────────────────────────────────────────
pkill -f "port-forward.*8001:9090" 2>/dev/null || true
pkill -f "port-forward.*8002:8000" 2>/dev/null || true
sleep 2

kubectl port-forward pod/"$(worker_pod)" 8001:9090 -n "$NS" &
PF_ENGINE=$!
kubectl port-forward pod/"$(frontend_pod)" 8002:8000 -n "$NS" &
PF_FRONTEND=$!
echo "Port-forwards: engine=$PF_ENGINE frontend=$PF_FRONTEND"
sleep 5

# ── Wait for inference endpoint ───────────────────────────────────────────────
echo "=== Waiting for inference endpoint ==="
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8002/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "Endpoint ready (checked after ~$((i * 5))s)"
    break
  fi
  sleep 5
done

# ── Helpers ───────────────────────────────────────────────────────────────────
snapshot() {
  local label="$1"
  local pod
  pod=$(worker_pod)
  echo ""
  echo "--- nvidia-smi ($label) ---"
  kubectl exec "$pod" -n "$NS" -- \
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1
  echo "--- Ray processes ($label) ---"
  kubectl exec "$pod" -n "$NS" -- ps aux 2>&1 \
    | awk '/DPMoEEngineCoreActor|RayWorkerWrapper/{printf "PID=%-8s CMD=%s\n", $2, $11}'
}

infer() {
  local label="$1"
  local pod
  pod=$(worker_pod)
  echo ""
  echo "--- inference ($label) ---"
  # Patch CRD if event_channels became null after scale (known Rust serde bug,
  # fixed in lib/runtime/src/discovery/metadata.rs)
  EC=$(kubectl get dynamoworkermetadata "$pod" -n "$NS" \
    -o jsonpath='{.spec.data.event_channels}' 2>/dev/null)
  if [ "$EC" = "null" ] || [ -z "$EC" ]; then
    kubectl patch dynamoworkermetadata "$pod" -n "$NS" \
      --type=merge -p '{"spec":{"data":{"event_channels":{}}}}' 2>/dev/null
    echo "(patched event_channels: null → {} — workaround for discovery 404)"
    sleep 3
  fi
  RESP=$(curl -s -m 30 http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2=\",\"max_tokens\":5,\"temperature\":0}")
  echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('text:', repr(d['choices'][0]['text'].strip()), '  time_ms:', d['nvext']['timing']['total_time_ms'])
" 2>/dev/null || echo "response: $RESP"
}

scale() {
  local from_dp="$1"
  local to_dp="$2"
  local timeout="${3:-700}"
  echo ""
  echo "=========================================="
  echo "SCALE dp=$from_dp → dp=$to_dp at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  worker pod: $(worker_pod)"
  echo "=========================================="
  echo "--- request: POST /engine/scale_elastic_ep {\"new_data_parallel_size\": $to_dp} ---"
  RESP=$(curl -s -X POST http://localhost:8001/engine/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d "{\"new_data_parallel_size\": $to_dp}" \
    --max-time "$timeout")
  echo "--- response ---"
  echo "$RESP"
  snapshot "after dp=$to_dp"
  infer "dp=$to_dp"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "BASELINE dp=2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
snapshot "baseline dp=2"
infer "dp=2"

# ── 6 scale steps ─────────────────────────────────────────────────────────────
scale 2 3 700   # step 1: dp=2 → dp=3
scale 3 4 700   # step 2: dp=3 → dp=4
scale 4 3 300   # step 3: dp=4 → dp=3
scale 3 2 300   # step 4: dp=3 → dp=2
scale 2 4 700   # step 5: dp=2 → dp=4  (scale-up after scale-down — known regression)
scale 4 2 300   # step 6: dp=4 → dp=2

echo ""
echo "=== ALL STEPS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
kill $PF_ENGINE $PF_FRONTEND 2>/dev/null || true
