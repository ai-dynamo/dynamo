#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# K8s Failover E2E Validation Script
# Tests per-container K8s discovery with GPU failover
#
# Prerequisites:
#   - kubectl configured for the target cluster
#   - Dynamo platform deployed with K8s discovery (no etcd)
#   - HF token secret created in the namespace
#   - DGD deployed with failover.enabled: true
#
# Usage:
#   ./test_k8s_failover.sh [NAMESPACE] [DGD_NAME]

set -euo pipefail

NAMESPACE="${1:-failover-k8s-disc}"
DGD_NAME="${2:-vllm-agg-failover}"
WORKER_LABEL="nvidia.com/dynamo-component=VllmDecodeWorker"
FRONTEND_SVC="${DGD_NAME}-frontend"
PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

cleanup() {
    pkill -f "port-forward.*${FRONTEND_SVC}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=============================================="
echo "  K8s Failover E2E Test"
echo "=============================================="
echo "Namespace: ${NAMESPACE}"
echo "DGD: ${DGD_NAME}"
echo ""

# --- Phase 1: Verify pods are running ---
echo "=== Phase 1: Pod Status ==="

WORKER=$(kubectl get pods -n "$NAMESPACE" -l "$WORKER_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$WORKER" ]; then
    echo "ERROR: No worker pod found with label ${WORKER_LABEL}"
    exit 1
fi
echo "Worker pod: ${WORKER}"

# Check container count
CONTAINER_COUNT=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{range .status.containerStatuses[*]}{.name}{"\n"}{end}' | wc -l)
if [ "$CONTAINER_COUNT" -eq 2 ]; then
    pass "Worker has 2 engine containers"
else
    fail "Expected 2 engine containers, got ${CONTAINER_COUNT}"
fi

# Check at least one engine is ready
ENGINE0_READY=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[?(@.name=="engine-0")].ready}')
ENGINE1_READY=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[?(@.name=="engine-1")].ready}')

if [ "$ENGINE0_READY" = "true" ] || [ "$ENGINE1_READY" = "true" ]; then
    pass "At least one engine is ready"
    if [ "$ENGINE0_READY" = "true" ]; then
        ACTIVE_ENGINE="engine-0"
        STANDBY_ENGINE="engine-1"
    else
        ACTIVE_ENGINE="engine-1"
        STANDBY_ENGINE="engine-0"
    fi
    echo "  Active: ${ACTIVE_ENGINE}, Standby: ${STANDBY_ENGINE}"
else
    fail "No engine is ready"
    exit 1
fi

# --- Phase 2: Discovery CRs ---
echo ""
echo "=== Phase 2: Discovery CRs ==="

ACTIVE_CR="${WORKER}-${ACTIVE_ENGINE}"
STANDBY_CR="${WORKER}-${STANDBY_ENGINE}"

# Active engine should have generate endpoint
ACTIVE_HAS_GENERATE=$(kubectl get dynamoworkermetadata "$ACTIVE_CR" -n "$NAMESPACE" -o jsonpath='{.spec.data}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if any('generate' in k for k in d.get('endpoints',{})) else 'no')" 2>/dev/null || echo "no")

if [ "$ACTIVE_HAS_GENERATE" = "yes" ]; then
    pass "Active engine CR has generate endpoint"
else
    fail "Active engine CR missing generate endpoint"
fi

# Standby engine should NOT have generate endpoint
STANDBY_HAS_GENERATE=$(kubectl get dynamoworkermetadata "$STANDBY_CR" -n "$NAMESPACE" -o jsonpath='{.spec.data}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if any('generate' in k for k in d.get('endpoints',{})) else 'no')" 2>/dev/null || echo "no")

if [ "$STANDBY_HAS_GENERATE" = "no" ]; then
    pass "Standby engine CR does NOT have generate endpoint"
else
    fail "Standby engine CR has generate endpoint (should not)"
fi

# --- Phase 3: Pre-failover inference ---
echo ""
echo "=== Phase 3: Pre-Failover Inference ==="

kubectl port-forward "svc/${FRONTEND_SVC}" 8000:8000 -n "$NAMESPACE" > /dev/null 2>&1 &
PF_PID=$!
sleep 5

RESPONSE=$(curl -s --max-time 30 http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16}' 2>/dev/null)

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['text']" 2>/dev/null; then
    TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:60])")
    pass "Pre-failover inference: ${TEXT}"
else
    fail "Pre-failover inference failed: ${RESPONSE}"
fi

# --- Phase 4: Failover ---
echo ""
echo "=== Phase 4: Failover ==="

echo "Killing ${ACTIVE_ENGINE}..."
KILL_TIME=$(date +%s%3N)
kubectl exec "$WORKER" -n "$NAMESPACE" -c "$ACTIVE_ENGINE" -- kill 1 2>/dev/null

echo "Waiting for ${STANDBY_ENGINE} to wake and register..."
for i in $(seq 1 30); do
    sleep 2
    NEW_HAS_GENERATE=$(kubectl get dynamoworkermetadata "${WORKER}-${STANDBY_ENGINE}" -n "$NAMESPACE" -o jsonpath='{.spec.data}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if any('generate' in k for k in d.get('endpoints',{})) else 'no')" 2>/dev/null || echo "no")
    if [ "$NEW_HAS_GENERATE" = "yes" ]; then
        REGISTER_TIME=$(date +%s%3N)
        FAILOVER_MS=$((REGISTER_TIME - KILL_TIME))
        pass "Standby engine registered generate endpoint (${FAILOVER_MS}ms)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "Standby engine did not register within 60s"
    fi
done

# --- Phase 5: Post-failover inference ---
echo ""
echo "=== Phase 5: Post-Failover Inference ==="

# Wait for frontend to discover new engine
sleep 10

RESPONSE=$(curl -s --max-time 30 http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"After the failover","max_tokens":16}' 2>/dev/null)

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['text']" 2>/dev/null; then
    TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:60])")
    pass "Post-failover inference: ${TEXT}"
else
    fail "Post-failover inference failed: ${RESPONSE}"
fi

# --- Phase 6: Stale CR cleanup ---
echo ""
echo "=== Phase 6: Stale CR Cleanup ==="

sleep 30  # Wait for killed engine to restart

ACTIVE_CR_EXISTS=$(kubectl get dynamoworkermetadata "$ACTIVE_CR" -n "$NAMESPACE" 2>/dev/null && echo "yes" || echo "no")
if [ "$ACTIVE_CR_EXISTS" = "no" ]; then
    pass "Killed engine's stale CR was cleaned up"
else
    # Check if it still has generate endpoint
    STALE_GENERATE=$(kubectl get dynamoworkermetadata "$ACTIVE_CR" -n "$NAMESPACE" -o jsonpath='{.spec.data}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if any('generate' in k for k in d.get('endpoints',{})) else 'no')" 2>/dev/null || echo "no")
    if [ "$STALE_GENERATE" = "no" ]; then
        pass "Killed engine's CR exists but has no generate endpoint (acceptable)"
    else
        fail "Killed engine's CR still has generate endpoint (stale)"
    fi
fi

# --- Summary ---
echo ""
echo "=============================================="
echo "  Results: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "=============================================="

kill $PF_PID 2>/dev/null || true
exit $FAIL_COUNT
