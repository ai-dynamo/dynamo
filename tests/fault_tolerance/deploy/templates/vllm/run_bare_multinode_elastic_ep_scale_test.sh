#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bare vLLM Multi-Node Elastic EP Scale Test
#
# Companion script for bare_multinode_elastic_ep.yaml.
# Tests cross-node elastic EP scaling using vLLM's native API — no Dynamo involved.
#
# Warm-standby topology at baseline:
#   vllm-ep-leader (Node 1): 2 GPUs active (dp ranks 0, 1)
#   vllm-ep-worker (Node 2): 2 GPUs idle in Ray cluster, claimed on scale-up
#
# Scale sequence:
#   Baseline dp=2 → dp=3 → dp=4 → dp=3 → dp=2 → dp=4 → dp=2
#
# Key differences from dynamo scripts:
#   - Single port-forward on 8000 (vLLM serves both inference AND scale API)
#   - Scale API path: POST /scale_elastic_ep  (no /engine/ prefix, no port 9090)
#   - Pods addressed by name, not label selector
#   - No frontend pod, no dynamoworkermetadata patch
#   - Inference response parsed as plain OpenAI (no nvext.timing field)
#
# Usage:
#   ./run_bare_multinode_elastic_ep_scale_test.sh [NAMESPACE]
#
# Defaults:
#   NAMESPACE = tzulingk-multinode-elastic
#
# Prerequisites:
#   - kubectl configured and pointing at the right cluster
#   - Deployment already applied: kubectl apply -f bare_multinode_elastic_ep.yaml -n <NS>
#   - Port 8001 free on localhost


set -uo pipefail

NS="${1:-tzulingk-multinode-elastic}"
LEADER_POD="vllm-ep-leader"
WORKER_POD="vllm-ep-worker"
MODEL="deepseek-ai/DeepSeek-V2-Lite"

echo "Namespace:   $NS"
echo "Leader pod:  $LEADER_POD"
echo "Worker pod:  $WORKER_POD"
echo "Model:       $MODEL"
echo ""

# ── Verify pods exist ─────────────────────────────────────────────────────────
for pod in "$LEADER_POD" "$WORKER_POD"; do
  phase=$(kubectl get pod "$pod" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ -z "$phase" ]; then
    echo "ERROR: pod $pod not found in namespace $NS" >&2
    exit 1
  fi
  echo "Pod $pod: phase=$phase"
done
echo ""

# ── Wait for leader pod ready ─────────────────────────────────────────────────
echo "=== Waiting for leader pod to be Ready ==="
kubectl wait pod/"$LEADER_POD" -n "$NS" --for=condition=Ready --timeout=900s
echo "Ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Port-forward (auto-restarting) ────────────────────────────────────────────
# Port 8000 on the leader serves both inference (/v1/completions) and the elastic
# EP scale API (/scale_elastic_ep). A single port-forward covers both.
# vLLM is not on port 8000 until the model finishes loading, so the port-forward
# will fail and restart repeatedly until vLLM is up — that is expected.
pkill -f "port-forward.*8001:8000" 2>/dev/null || true
sleep 1

(while true; do
  kubectl port-forward pod/"$LEADER_POD" 8001:8000 -n "$NS" 2>&1
  sleep 2
done) &
PF=$!
echo "Port-forward: auto-restarting loop pid=$PF  localhost:8001 → $LEADER_POD:8000"
sleep 3

# ── Wait for inference endpoint ───────────────────────────────────────────────
echo "=== Waiting for inference endpoint ==="
ENDPOINT_READY=0
for i in $(seq 1 120); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8001/health 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "Endpoint ready (checked after ~$((i * 5))s)"
    ENDPOINT_READY=1
    break
  fi
  sleep 5
done
if [ "$ENDPOINT_READY" = "0" ]; then
  echo "ERROR: inference endpoint never became ready after 600s" >&2
  kill $PF 2>/dev/null
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# Blocks until the worker node appears in the Ray cluster (2 active nodes).
# The worker pod runs `ray start --address=...` only after vLLM health=200.
# On first pod start, Python bytecode compilation can silently delay this by
# up to 10 minutes — scaling before the worker is in Ray guarantees failure.
wait_worker_in_ray() {
  local timeout="${1:-900}"
  local interval=15
  local elapsed=0
  echo ""
  echo "=== Waiting for worker node to join Ray cluster (need 2 active nodes) ==="
  while [ "$elapsed" -lt "$timeout" ]; do
    STATUS=$(kubectl exec pod/"$LEADER_POD" -n "$NS" -- ray status 2>/dev/null)
    NODES=$(echo "$STATUS" | awk '/^Active:/{p=1;next} /^Pending:/{p=0} p && /node_/{c++} END{print c+0}')
    if [ "${NODES:-0}" -ge 2 ]; then
      echo "Worker joined Ray ($NODES active nodes) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "$STATUS" | awk '/Resources/,/Pending Demands/' | head -6
      return 0
    fi
    echo "  ${NODES:-0}/2 nodes in Ray, retrying in ${interval}s... (${elapsed}s elapsed)"
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done
  echo "ERROR: worker never joined Ray after ${timeout}s" >&2
  exit 1
}

# Captures nvidia-smi from both pods so we can see which node's GPUs are active.
snapshot() {
  local label="$1"
  echo ""
  for pod in "$LEADER_POD" "$WORKER_POD"; do
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
  RESP=$(curl -fsS -m 30 http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2=\",\"max_tokens\":5,\"temperature\":0}") || {
    echo "ERROR: inference request failed" >&2
    return 1
  }
  # Plain OpenAI response — no nvext field in bare vLLM
  echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['text'].strip()
tokens = d.get('usage', {})
print('text:', repr(text), '  usage:', tokens)
" 2>/dev/null || {
    echo "ERROR: invalid inference response: $RESP" >&2
    return 1
  }
}

# Calls vLLM's native scale endpoint on port 8000 (same port as inference).
# NOTE: bare vLLM uses /scale_elastic_ep directly, NOT /engine/scale_elastic_ep.
scale() {
  local from_dp="$1"
  local to_dp="$2"
  local timeout="${3:-700}"
  echo ""
  echo "=========================================="
  echo "SCALE dp=$from_dp → dp=$to_dp at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=========================================="
  echo "--- POST /scale_elastic_ep {\"new_data_parallel_size\": $to_dp} ---"
  RESP=$(curl -fsS -X POST http://localhost:8001/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d "{\"new_data_parallel_size\": $to_dp}" \
    --max-time "$timeout") || {
    echo "ERROR: scale_elastic_ep request failed" >&2
    return 1
  }
  echo "--- response ---"
  echo "$RESP"
  SCALE_STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  if [ "$SCALE_STATUS" != "ok" ]; then
    echo "ERROR: scale_elastic_ep to dp=$to_dp failed: $RESP" >&2
    exit 1
  fi
  snapshot "after dp=$to_dp"
  infer "dp=$to_dp"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
# Expected: leader pod shows 2 GPUs active, worker pod shows 2 GPUs idle (low memory.used)
echo ""
echo "=========================================="
echo "BASELINE dp=2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
snapshot "baseline dp=2"
infer "dp=2" || exit 1

# ── Wait for worker in Ray before any scale step ──────────────────────────────
# Do this after baseline so the baseline snapshot/inference runs while the
# worker's `ray start` is still warming up in the background.
wait_worker_in_ray 900

# ── 6 scale steps ─────────────────────────────────────────────────────────────
scale 2 3 700 || exit 1   # step 1: dp=2 → dp=3  (Ray places 1 actor on worker node)
scale 3 4 700 || exit 1   # step 2: dp=3 → dp=4  (Ray places 1 more actor on worker node)
scale 4 3 300 || exit 1   # step 3: dp=4 → dp=3  (removes highest rank from worker node)
scale 3 2 300 || exit 1   # step 4: dp=3 → dp=2  (worker node back to idle)
scale 2 4 700 || exit 1   # step 5: dp=2 → dp=4  (both worker node GPUs claimed)
scale 4 2 300 || exit 1   # step 6: dp=4 → dp=2  (worker node back to warm standby)

echo ""
echo "=== ALL STEPS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
kill $PF 2>/dev/null || true
