#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# K8s Non-Failover (Aggregated) DGD Validation Script
# Tests that per-container K8s discovery works correctly for standard single-engine deployments.
#
# Prerequisites:
#   - kubectl configured for the target cluster
#   - Dynamo platform deployed with K8s discovery
#   - HF token secret created in the namespace
#   - GPU resources available on at least one node
#
# Usage:
#   ./test_k8s_agg.sh [NAMESPACE] [IMAGE_TAG]
#
# Examples:
#   ./test_k8s_agg.sh failover-k8s-disc failover-m6-discovery-0a5eace8c-vllm-runtime

set -euo pipefail

NAMESPACE="${1:-failover-k8s-disc}"
IMAGE_TAG="${2:-}"
REGISTRY="dynamoci.azurecr.io/ai-dynamo/dynamo"
DGD_NAME="vllm-agg-test"
WORKER_LABEL="nvidia.com/dynamo-graph-deployment-name=${DGD_NAME},nvidia.com/dynamo-component-type=worker"
FRONTEND_SVC="${DGD_NAME}-frontend"
PASS_COUNT=0
FAIL_COUNT=0
MODEL="Qwen/Qwen3-0.6B"

if [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 [NAMESPACE] IMAGE_TAG"
    echo "  IMAGE_TAG is required (e.g., failover-m6-discovery-0a5eace8c-vllm-runtime)"
    exit 1
fi

IMAGE="${REGISTRY}:${IMAGE_TAG}"

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

cleanup() {
    pkill -f "port-forward.*${FRONTEND_SVC}" 2>/dev/null || true
    kubectl delete dgd "$DGD_NAME" -n "$NAMESPACE" 2>/dev/null || true
}
trap cleanup EXIT

echo "=============================================="
echo "  K8s Non-Failover (Agg) E2E Test"
echo "=============================================="
echo "Namespace: ${NAMESPACE}"
echo "Image: ${IMAGE}"
echo ""

# --- Phase 1: Deploy ---
echo "=== Phase 1: Deploy DGD ==="

cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: ${DGD_NAME}
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: ${IMAGE}
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: ${IMAGE}
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "${MODEL}"]
EOF

echo "Waiting for worker pod to be ready (up to 5 min)..."
for i in $(seq 1 30); do
    WORKER=$(kubectl get pods -n "$NAMESPACE" -l "$WORKER_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$WORKER" ]; then
        READY=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
        if [ "$READY" = "true" ]; then
            echo "Worker ready after $((i * 10))s"
            break
        fi
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Worker not ready within 5 min"
        kubectl get pods -n "$NAMESPACE" -l "$WORKER_LABEL" 2>&1
        exit 1
    fi
    sleep 10
done

pass "DGD deployed and worker ready"

# --- Phase 2: Pod structure ---
echo ""
echo "=== Phase 2: Pod Structure ==="

CONTAINER_COUNT=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{range .status.containerStatuses[*]}{.name}{"\n"}{end}' | wc -l)
CONTAINER_NAME=$(kubectl get pod "$WORKER" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].name}')

if [ "$CONTAINER_COUNT" -eq 1 ]; then
    pass "Single container pod (non-failover)"
else
    fail "Expected 1 container, got ${CONTAINER_COUNT}"
fi

echo "  Container name: ${CONTAINER_NAME}"

# --- Phase 3: Discovery CR ---
echo ""
echo "=== Phase 3: Discovery CR ==="

CR_NAME="${WORKER}-${CONTAINER_NAME}"
CR_EXISTS=$(kubectl get dynamoworkermetadata "$CR_NAME" -n "$NAMESPACE" > /dev/null 2>&1 && echo "yes" || echo "no")

if [ "$CR_EXISTS" = "yes" ]; then
    pass "Per-container CR exists: ${CR_NAME}"
else
    fail "Per-container CR not found: ${CR_NAME}"
fi

HAS_GENERATE=$(kubectl get dynamoworkermetadata "$CR_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.data}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if any('generate' in k for k in d.get('endpoints',{})) else 'no')" 2>/dev/null || echo "no")

if [ "$HAS_GENERATE" = "yes" ]; then
    pass "CR has generate endpoint"
else
    fail "CR missing generate endpoint"
fi

# --- Phase 4: Inference ---
echo ""
echo "=== Phase 4: Inference ==="

kubectl port-forward "svc/${FRONTEND_SVC}" 8000:8000 -n "$NAMESPACE" > /dev/null 2>&1 &
PF_PID=$!
sleep 5

RESPONSE=$(curl -s --max-time 30 http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}" 2>/dev/null)

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['text']" 2>/dev/null; then
    TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'][:60])")
    pass "Inference: ${TEXT}"
else
    fail "Inference failed: ${RESPONSE}"
fi

kill $PF_PID 2>/dev/null || true

# --- Summary ---
echo ""
echo "=============================================="
echo "  Results: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "=============================================="

exit $FAIL_COUNT
