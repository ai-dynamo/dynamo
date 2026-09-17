#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Elastic EP Scaling Regression Test
#
# CHANGING THE WIDTH TAKES TWO STEPS, in opposite order for grow and shrink. Kubernetes owns
# whether a pod exists; vLLM owns whether its GPU carries a rank. Neither alone changes the
# serving width, and nothing in the operator calls scale_elastic_ep yet (Phase 6/7), so both
# steps are driven from here:
#
#   grow    scale the follower DCD up -> wait for the new pods to join Ray
#           -> POST /scale_elastic_ep {"new_data_parallel_size": N}
#
#   shrink  POST /scale_elastic_ep {"new_data_parallel_size": N}   # drain the ranks FIRST
#           -> scale the follower DCD down
#
# The drain has to come first on the way down. Deleting a follower pod that still holds a
# live rank leaves the engine committed to a data-parallel size whose members are gone --
# DYN-3838 records the leader surviving at restart=0 with inference stopped, and DYN-2660
# records the orphaned placement group then blocking every later scale-up.
#
# Sequence: baseline dp=2 -> dp=4 -> dp=2
#
# dp=3 is deliberately absent. EPLB requires the expert count to divide evenly across ranks
# and DeepSeek-V2-Lite has 64 experts, so dp=3 fails with "EPLB currently only supports even
# distribution of experts across ranks". The previous version of this script stepped through
# dp=3 twice and could not have passed.
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
#   - Deployment already applied (see moe_elastic_ep_demo.yaml): one pod per rank, so
#     reaching dp=4 needs 4 nodes with a free GPU each
#   - Ports 8001 and 8002 free on localhost

set -uo pipefail

NS="${1:-default}"
DEPLOYMENT_NAME="${2:-vllm-elastic-ep-demo}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"

echo "Namespace:  $NS"
echo "Model:      $MODEL"
echo ""

# ── Pod and DCD lookup ────────────────────────────────────────────────────────
# Re-resolved from the cluster every time, so this survives pod restarts and the random
# name suffixes. The leader and the follower are DIFFERENT components now: the follower's
# component label is "<leader>-flw", so a plain component selector finds only the leader.
#
# EVERY selector is scoped by dynamo-graph-deployment-name. The component label alone is not
# unique in a shared namespace -- "Frontend" matched this deployment AND an unrelated one on
# the test cluster, and `.items[0]` would then port-forward to a stranger's frontend and send
# inference there. Scoping is what makes these safe to run alongside other work.
SEL="nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME"

leader_pod() {
  kubectl get pods -n "$NS" \
    -l "$SEL,nvidia.com/dynamo-component=worker" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

follower_pods() {
  kubectl get pods -n "$NS" \
    -l "$SEL,nvidia.com/dynamo-component=worker-flw" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null
}

frontend_pod() {
  kubectl get pods -n "$NS" \
    -l "$SEL,nvidia.com/dynamo-component=Frontend" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

# The synthesized follower DCD. Its name carries the leader's worker hash, so it is resolved
# rather than assumed -- and scoped, so a second elastic-EP deployment in the same namespace
# is not what gets scaled.
follower_dcd() {
  kubectl get dynamocomponentdeployment -n "$NS" -l "$SEL" -o name 2>/dev/null \
    | sed 's|.*/||' | grep -- "-flw$" | head -1
}

INITIAL_POD=$(leader_pod)
if [ -z "$INITIAL_POD" ]; then
  echo "ERROR: no running leader pod found in namespace $NS" >&2
  exit 1
fi
FLW_DCD=$(follower_dcd)
if [ -z "$FLW_DCD" ]; then
  echo "ERROR: no synthesized follower DCD found; is this an elastic-EP deployment?" >&2
  exit 1
fi
echo "Leader pod:   $INITIAL_POD"
echo "Follower DCD: $FLW_DCD"

echo "=== Waiting for leader pod to be Ready ==="
# The leader blocks on its width gate until every declared rank has joined Ray, so this also
# proves the followers came up.
kubectl wait pod/"$(leader_pod)" -n "$NS" --for=condition=Ready --timeout=900s
echo "Ready at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Port-forwards ─────────────────────────────────────────────────────────────
pkill -f "port-forward.*8001:9090" 2>/dev/null || true
pkill -f "port-forward.*8002:8000" 2>/dev/null || true
sleep 2

kubectl port-forward pod/"$(leader_pod)" 8001:9090 -n "$NS" &
PF_ENGINE=$!
kubectl port-forward pod/"$(frontend_pod)" 8002:8000 -n "$NS" &
PF_FRONTEND=$!
echo "Port-forwards: engine=$PF_ENGINE frontend=$PF_FRONTEND"
sleep 5

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
# Snapshot every pod, not just one. Ranks live in separate pods now, so a single-pod
# nvidia-smi would show one GPU and miss the placement entirely.
snapshot() {
  local label="$1" p
  echo ""
  echo "--- GPU + Ray actors ($label) ---"
  for p in $(leader_pod) $(follower_pods); do
    echo "  [$p]"
    kubectl exec "$p" -n "$NS" -- \
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 \
      | sed 's/^/    gpu  /'
    kubectl exec "$p" -n "$NS" -- ps aux 2>&1 \
      | awk '/DPMoEEngineCoreActor|RayWorkerWrapper/{printf "    ray  PID=%-8s %s\n", $2, $11}'
  done
}

infer() {
  local label="$1" pod
  pod=$(leader_pod)
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

# mark_drained_pods_for_deletion -- tell Kubernetes WHICH followers are safe to remove.
#
# Draining with scale_elastic_ep decides how many ranks the engine uses; it does not decide
# which pods keep them. Scaling the Deployment down afterwards picks victims by Kubernetes'
# own ordering, which knows nothing about ranks -- and on this cluster it deleted a follower
# that was still holding one, taking the engine down for ~2.5 minutes while it restarted.
#
# controller.kubernetes.io/pod-deletion-cost is the supported way to express the preference:
# a ReplicaSet removes the lowest-cost pods first. A drained follower has released its GPU
# memory, so nvidia-smi distinguishes the two -- near-zero used means no rank.
#
# This is a test-harness workaround for a real gap. Durably, the follower needs per-pod
# identity so the operator can retire a NAMED pod rather than a count (grove#793).
mark_drained_pods_for_deletion() {
  local p used
  echo "--- marking drained followers as preferred deletion targets ---"
  for p in $(follower_pods); do
    used=$(kubectl exec "$p" -n "$NS" -- \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' \r' | head -1)
    if [ -z "$used" ]; then
      echo "    $p: could not read GPU memory, leaving cost unset"
      continue
    fi
    if [ "$used" -lt 1024 ]; then
      kubectl annotate pod "$p" -n "$NS" --overwrite \
        controller.kubernetes.io/pod-deletion-cost=-100 >/dev/null 2>&1
      echo "    $p: ${used}MiB -> drained, cost=-100 (delete this one first)"
    else
      kubectl annotate pod "$p" -n "$NS" --overwrite \
        controller.kubernetes.io/pod-deletion-cost=100 >/dev/null 2>&1
      echo "    $p: ${used}MiB -> holds a rank, cost=100 (keep)"
    fi
  done
}

# set_followers <n> -- change the POD count and wait for it to settle.
set_followers() {
  local want="$1" i got
  echo "--- kubectl patch dcd $FLW_DCD replicas=$want ---"
  kubectl patch dynamocomponentdeployment "$FLW_DCD" -n "$NS" \
    --type=merge -p "{\"spec\":{\"replicas\":$want}}" >/dev/null 2>&1
  for i in $(seq 1 60); do
    sleep 10
    got=$(follower_pods | grep -c . | tr -d ' ')
    [ "$got" = "$want" ] && { echo "    follower pods: $got (settled after ~$((i * 10))s)"; return 0; }
  done
  echo "    WARNING: wanted $want follower pods, have $(follower_pods | grep -c . | tr -d ' ') after 10m"
  echo "    (are there enough nodes with a free GPU? each rank needs its own pod)"
  return 1
}

# set_engine_width <n> -- change the RANK count on the live engine.
set_engine_width() {
  local to_dp="$1" timeout="${2:-700}"
  echo "--- POST /engine/control/scale_elastic_ep {\"new_data_parallel_size\": $to_dp} ---"
  curl -s -X POST http://localhost:8001/engine/control/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d "{\"new_data_parallel_size\": $to_dp}" \
    --max-time "$timeout"
  echo ""
}

grow() {
  local to_dp="$1"
  echo ""
  echo "=========================================="
  echo "GROW to dp=$to_dp at $(date -u +%Y-%m-%dT%H:%M:%SZ)   (pods first, then ranks)"
  echo "=========================================="
  set_followers "$((to_dp - 1))" || return 1
  set_engine_width "$to_dp" 700
  snapshot "after grow to dp=$to_dp"
  infer "dp=$to_dp"
}

shrink() {
  local to_dp="$1"
  echo ""
  echo "=========================================="
  echo "SHRINK to dp=$to_dp at $(date -u +%Y-%m-%dT%H:%M:%SZ)   (ranks first, then pods)"
  echo "=========================================="
  set_engine_width "$to_dp" 300
  snapshot "after draining to dp=$to_dp"
  # Between the drain and the scale-down, name the pods that are safe to lose. Without this
  # the Deployment picks its own victims and can delete a follower that still holds a rank.
  mark_drained_pods_for_deletion
  set_followers "$((to_dp - 1))"
  snapshot "after releasing pods at dp=$to_dp"
  infer "dp=$to_dp"
}

# ── Baseline ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "BASELINE dp=2 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
snapshot "baseline dp=2"
infer "dp=2"

# ── Scale sequence ────────────────────────────────────────────────────────────
grow 4
shrink 2

echo ""
echo "=== ALL STEPS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
kill $PF_ENGINE $PF_FRONTEND 2>/dev/null || true
