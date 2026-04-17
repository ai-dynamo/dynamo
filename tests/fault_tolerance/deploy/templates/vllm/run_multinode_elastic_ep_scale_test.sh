#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-Node Elastic EP Scale Test — 3 GPU/node (6 GPUs total)
#
# Warm-standby topology:
#   - 2 nodes × 3 GPUs each = 6 GPUs total
#   - Baseline dp=2: 2 leader GPUs active, 4 GPUs idle (1 leader + 3 worker)
#
# Note: --enable-elastic-ep requires --enable-eplb, and --enable-eplb requires
# dp>=2 at startup. Scale DOWN to dp=1 crashes vLLM's _eplb_reshuffle_before_scale_down
# (vLLM bug) — dp=1 is intentionally excluded from this test sequence.
#
# Scale sequence:
#   dp=2 → dp=3 → dp=4 → dp=5 → dp=6 → dp=5 → dp=4 → dp=3 → dp=2 → dp=4 → dp=6 → dp=2
#
# Node placement per step:
#   dp=2: leader GPU 0+1                              ← baseline
#   dp=3: leader GPU 0+1+2                            ← all leader GPUs used
#   dp=4: leader GPU 0+1+2, worker GPU 0              ← first cross-node actor
#   dp=5: leader GPU 0+1+2, worker GPU 0+1
#   dp=6: leader GPU 0+1+2, worker GPU 0+1+2          ← full capacity
#
# nvidia-smi memory usage is captured from BOTH pods after every scale step so
# we can observe which node's GPUs become active as ranks are added or removed.
#
# Usage:
#   ./run_elastic_ep_scale_test_multinode_3gpu.sh [NAMESPACE] [DEPLOYMENT_NAME]
#
# Defaults:
#   NAMESPACE       = tzulingk-multinode-elastic
#   DEPLOYMENT_NAME = ep-mn

set -uo pipefail

NS="${1:-tzulingk-multinode-elastic}"
DEPLOYMENT_NAME="${2:-ep-mn}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"

echo "Namespace:  $NS"
echo "Deployment: $DEPLOYMENT_NAME"
echo "Model:      $MODEL"
echo ""

# ── Pod lookup helpers ────────────────────────────────────────────────────────

# All running VllmDecodeWorker pods (both nodes)
all_worker_pods() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=VllmDecodeWorker" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null
}

# Leader pod (lowest-sorted name = rank-0)
# Only the leader exposes port 9090 (scale API)
head_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=VllmDecodeWorker" \
    --field-selector=status.phase=Running \
    --sort-by='.metadata.name' \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

frontend_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=Frontend" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

# Verify pods are present
INITIAL_HEAD=$(head_pod)
if [ -z "$INITIAL_HEAD" ]; then
  echo "ERROR: no running VllmDecodeWorker pod found in namespace $NS" >&2
  exit 1
fi
echo "Leader pod (at start): $INITIAL_HEAD"
echo "All worker pods:       $(all_worker_pods)"

# ── Wait for leader ready ─────────────────────────────────────────────────────
echo ""
echo "=== Waiting for leader pod to be Ready ==="
kubectl wait pod/"$(head_pod)" -n "$NS" --for=condition=Ready --timeout=900s
echo "Ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Wait for inference endpoint ───────────────────────────────────────────────
# Use kubectl exec for all API calls — kubectl port-forward tunnels through the
# Teleport/AKS API proxy and die after a single connection drop, never
# recovering.  kubectl exec opens a fresh connection per call.
echo "=== Waiting for inference endpoint ==="
for i in $(seq 1 60); do
  fpod=$(frontend_pod)
  CODE=$(kubectl exec "$fpod" -n "$NS" -- \
    curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8000/v1/models 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "Endpoint ready (checked after ~$((i * 5))s)"
    break
  fi
  sleep 5
done
if [ "$CODE" != "200" ]; then
  echo "ERROR: inference endpoint never became ready (last HTTP code: ${CODE:-none})" >&2
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# Captures nvidia-smi from BOTH pods so we can see which node holds active GPUs
snapshot() {
  local label="$1"
  local pods
  pods=$(all_worker_pods)
  echo ""
  for pod in $pods; do
    node=$(kubectl get pod "$pod" -n "$NS" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    echo "--- nvidia-smi ($label) pod=$pod node=$node ---"
    kubectl exec "$pod" -n "$NS" -- \
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv 2>&1
    echo "--- Ray actors ($label) pod=$pod ---"
    kubectl exec "$pod" -n "$NS" -- ps aux 2>&1 \
      | awk '/DPMoEEngineCoreActor|RayWorkerWrapper/{printf "PID=%-8s CMD=%s\n", $2, $11}'
  done
}

infer() {
  local label="$1"
  echo ""
  echo "--- inference ($label) ---"
  local fpod
  fpod=$(frontend_pod)
  if [ -z "$fpod" ]; then
    echo "  (no frontend pod found — skipping)"
    return
  fi
  # Use kubectl exec so we curl from inside the pod — avoids relying on a
  # persistent port-forward which dies after the first connection on AKS.
  RESP=$(kubectl exec "$fpod" -n "$NS" -- \
    curl -s -m 60 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2=\",\"max_tokens\":16,\"temperature\":0}" \
    2>&1)
  echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['text'].strip()
timing = d.get('nvext', {}).get('timing', {}).get('total_time_ms', 'n/a')
print('text:', repr(text), '  time_ms:', timing)
" 2>/dev/null || echo "raw response: $RESP"
}

scale() {
  local from_dp="$1"
  local to_dp="$2"
  local timeout="${3:-700}"
  echo ""
  echo "=========================================="
  echo "SCALE dp=$from_dp → dp=$to_dp at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  leader pod: $(head_pod)"
  echo "  all pods:   $(all_worker_pods)"
  echo "=========================================="
  local lpod
  lpod=$(head_pod)
  RESP=$(kubectl exec "$lpod" -n "$NS" -- \
    curl -s -X POST http://localhost:9090/engine/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d "{\"new_data_parallel_size\": $to_dp}" \
    --max-time "$timeout" \
    2>&1)
  echo "--- scale response ---"
  echo "$RESP"
  SCALE_STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  SCALE_DP=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_data_parallel_size',''))" 2>/dev/null)
  if [ "$SCALE_STATUS" != "ok" ] || [ "$SCALE_DP" != "$to_dp" ]; then
    echo "ERROR: scale to dp=$to_dp failed: $RESP" >&2
    exit 1
  fi
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

# ── Scale sequence (11 steps) — dp=1 excluded (vLLM _eplb_reshuffle bug) ─────
scale 2 3 700    # step 1:  dp=2 → dp=3  (within-leader: 3rd GPU on leader)
scale 3 4 700    # step 2:  dp=3 → dp=4  (cross-node:   first actor on worker)
scale 4 5 700    # step 3:  dp=4 → dp=5  (cross-node:   2nd actor on worker)
scale 5 6 700    # step 4:  dp=5 → dp=6  (cross-node:   worker fully active)
scale 6 5 300    # step 5:  dp=6 → dp=5  (scale down:   remove 1 from worker)
scale 5 4 300    # step 6:  dp=5 → dp=4  (scale down)
scale 4 3 300    # step 7:  dp=4 → dp=3  (scale down:   worker back to idle)
scale 3 2 300    # step 8:  dp=3 → dp=2  (scale down:   back to baseline)
scale 2 4 700    # step 9:  dp=2 → dp=4  (jump up:      skip dp=3)
scale 4 6 700    # step 10: dp=4 → dp=6  (jump to full capacity)
scale 6 2 300    # step 11: dp=6 → dp=2  (jump down:    back to baseline)

echo ""
echo "=== ALL STEPS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
