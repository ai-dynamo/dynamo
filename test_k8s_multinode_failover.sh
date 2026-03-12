#!/bin/bash
# K8s Multinode Failover Test
#
# Tests multinode (TP=2, 2 nodes) GPU failover with per-container K8s discovery.
# Each pod has 2 engine containers (engine-0, engine-1) + GMS sidecar.
# Engine-0 across pods forms one TP group, engine-1 forms another.
# The leader pod's flock decides which group is active.
#
# Usage: ./test_k8s_multinode_failover.sh [NAMESPACE] [DGD_NAME]

set -e

NAMESPACE="${1:-multinode-failover}"
DGD_NAME="${2:-vllm-mn-fo}"

pass_count=0
fail_count=0
pass() { pass_count=$((pass_count + 1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count + 1)); echo "  FAIL: $1"; }

echo "=============================================="
echo "  K8s Multinode Failover Test"
echo "=============================================="
echo "Namespace: $NAMESPACE"
echo "DGD: $DGD_NAME"
echo ""

# ============================================================
# Phase 1: Pod Structure Validation
# ============================================================
echo "=== Phase 1: Pod Structure ==="

# Wait for pods to be running
echo "Waiting for worker pods..."
for i in $(seq 1 300); do
    RUNNING=$(kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component=Worker --no-headers 2>/dev/null | grep -c "Running" || echo 0)
    if [ "$RUNNING" -ge 2 ]; then
        echo "Both worker pods running (${i}s)"
        break
    fi
    if [ "$i" -eq 300 ]; then
        echo "TIMEOUT waiting for pods"
        kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component=Worker
        exit 1
    fi
    sleep 2
done

# Get pod names (leader has -ldr suffix, worker has -wkr suffix)
LEADER_POD=$(kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component=Worker --no-headers 2>/dev/null | grep "ldr" | awk '{print $1}' | head -n 1)
WORKER_POD=$(kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component=Worker --no-headers 2>/dev/null | grep "wkr" | awk '{print $1}' | head -n 1)

if [ -z "$LEADER_POD" ] || [ -z "$WORKER_POD" ]; then
    echo "ERROR: Could not find leader and worker pods"
    kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component=Worker
    exit 1
fi

echo "Leader pod: $LEADER_POD"
echo "Worker pod: $WORKER_POD"

# Check container count (expect 2 engine containers per pod)
LEADER_CONTAINERS=$(kubectl get pod "$LEADER_POD" -n "$NAMESPACE" -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}' | wc -l)
WORKER_CONTAINERS=$(kubectl get pod "$WORKER_POD" -n "$NAMESPACE" -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}' | wc -l)

echo "Leader containers: $LEADER_CONTAINERS"
kubectl get pod "$LEADER_POD" -n "$NAMESPACE" -o jsonpath='{range .spec.containers[*]}{.name}{" "}{end}'
echo ""
echo "Worker containers: $WORKER_CONTAINERS"
kubectl get pod "$WORKER_POD" -n "$NAMESPACE" -o jsonpath='{range .spec.containers[*]}{.name}{" "}{end}'
echo ""

if [ "$LEADER_CONTAINERS" -ge 2 ]; then
    pass "Leader pod has $LEADER_CONTAINERS containers (engine-0 + engine-1)"
else
    fail "Leader pod has $LEADER_CONTAINERS containers (expected >= 2)"
fi

if [ "$WORKER_CONTAINERS" -ge 2 ]; then
    pass "Worker pod has $WORKER_CONTAINERS containers (engine-0 + engine-1)"
else
    fail "Worker pod has $WORKER_CONTAINERS containers (expected >= 2)"
fi

# ============================================================
# Phase 2: Engine Initialization
# ============================================================
echo ""
echo "=== Phase 2: Engine Initialization ==="

# Wait for one engine to register (become active)
echo "Waiting for active engine to register..."
for i in $(seq 1 300); do
    # Check leader engine-0
    if kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c engine-0 2>/dev/null | grep -q "Registered endpoint 'generate'"; then
        ACTIVE_ENGINE="engine-0"
        SHADOW_ENGINE="engine-1"
        echo "Engine-0 group is active (${i}s)"
        break
    fi
    # Check leader engine-1
    if kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c engine-1 2>/dev/null | grep -q "Registered endpoint 'generate'"; then
        ACTIVE_ENGINE="engine-1"
        SHADOW_ENGINE="engine-0"
        echo "Engine-1 group is active (${i}s)"
        break
    fi
    if [ "$i" -eq 300 ]; then
        echo "TIMEOUT: No engine registered"
        echo "=== Leader engine-0 (last 20) ==="
        kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c engine-0 --tail=20 2>/dev/null
        echo "=== Leader engine-1 (last 20) ==="
        kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c engine-1 --tail=20 2>/dev/null
        echo "=== Worker engine-0 (last 20) ==="
        kubectl logs "$WORKER_POD" -n "$NAMESPACE" -c engine-0 --tail=20 2>/dev/null
        exit 1
    fi
    sleep 2
done

pass "Active engine: $ACTIVE_ENGINE"

# Check shadow is NOT registered
if kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c "$SHADOW_ENGINE" 2>/dev/null | grep -q "Registered endpoint 'generate'"; then
    fail "Shadow engine also registered (both active!)"
else
    pass "Shadow engine ($SHADOW_ENGINE) not registered (sleeping)"
fi

# ============================================================
# Phase 3: Pre-failover Inference
# ============================================================
echo ""
echo "=== Phase 3: Pre-failover Inference ==="

# Port-forward frontend
FRONTEND_POD=$(kubectl get pods -n "$NAMESPACE" -l nvidia.com/dynamo-component-type=frontend --no-headers | awk '{print $1}' | head -n 1)
if [ -z "$FRONTEND_POD" ]; then
    echo "ERROR: No frontend pod found"
    exit 1
fi

kubectl port-forward "$FRONTEND_POD" 8000:8000 -n "$NAMESPACE" &
PF_PID=$!
sleep 5

RESP=$(curl -s -m 30 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}")

if echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null; then
    pass "Pre-failover inference works"
else
    fail "Pre-failover inference failed: $RESP"
fi

# ============================================================
# Phase 4: Failover
# Kill active engine on BOTH pods to simulate coordinated failure.
# ============================================================
echo ""
echo "=== Phase 4: Failover ==="

echo "Killing $ACTIVE_ENGINE on leader pod..."
kubectl exec "$LEADER_POD" -n "$NAMESPACE" -c "$ACTIVE_ENGINE" -- kill 1 2>/dev/null || true

echo "Killing $ACTIVE_ENGINE on worker pod..."
kubectl exec "$WORKER_POD" -n "$NAMESPACE" -c "$ACTIVE_ENGINE" -- kill 1 2>/dev/null || true

echo "Waiting for shadow engine to wake and register..."
for i in $(seq 1 180); do
    if kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c "$SHADOW_ENGINE" 2>/dev/null | grep -q "Registered endpoint 'generate'"; then
        echo "Shadow engine registered! (${i}s)"
        break
    fi
    if [ "$i" -eq 180 ]; then
        echo "TIMEOUT: Shadow did not wake"
        echo "=== Shadow leader (last 30) ==="
        kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c "$SHADOW_ENGINE" --tail=30 2>/dev/null
        echo "=== Shadow worker (last 30) ==="
        kubectl logs "$WORKER_POD" -n "$NAMESPACE" -c "$SHADOW_ENGINE" --tail=30 2>/dev/null
    fi
    sleep 2
done

if kubectl logs "$LEADER_POD" -n "$NAMESPACE" -c "$SHADOW_ENGINE" 2>/dev/null | grep -q "Registered endpoint 'generate'"; then
    pass "Shadow engine woke and registered after failover"
else
    fail "Shadow engine did not register after failover"
fi

# ============================================================
# Phase 5: Post-failover Inference
# ============================================================
echo ""
echo "=== Phase 5: Post-failover Inference ==="

# Wait for discovery propagation
sleep 10

RESP=$(curl -s -m 30 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}")

if echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null; then
    pass "Post-failover inference works"
else
    fail "Post-failover inference failed: $RESP"
fi

# ============================================================
# Cleanup
# ============================================================
kill $PF_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Results: $pass_count passed, $fail_count failed"
echo "=============================================="
[ "$fail_count" -gt 0 ] && exit 1
exit 0
