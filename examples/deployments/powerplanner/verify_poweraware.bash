#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Comprehensive verification script for power-aware deployment
# Tests all components, RBAC, Prometheus metrics, and planner functionality

# Don't exit on error - we want to run all tests
set +e

# Dynamically determine the repository root (parent of examples/deployments/powerplanner)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PATH=${DEV_REPO}/bin_bin:$PATH
export MINIKUBE_HOME=${DEV_REPO}/minikube_home
NAMESPACE=dynamo-system

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED_TESTS=0
PASSED_TESTS=0

print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED_TESTS++))
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED_TESTS++))
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     POWER-AWARE DEPLOYMENT VERIFICATION SUITE               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Test 1: Check Minikube
print_header "TEST 1: Minikube Status"
if minikube status &>/dev/null; then
    pass "Minikube is running"
else
    fail "Minikube is not running"
fi

# Test 2: Check Namespace
print_header "TEST 2: Namespace Status"
if kubectl get namespace ${NAMESPACE} &>/dev/null; then
    pass "Namespace ${NAMESPACE} exists"
else
    fail "Namespace ${NAMESPACE} not found"
fi

# Test 3: Check All Pods
print_header "TEST 3: Pod Status"
TOTAL_PODS=$(kubectl get pods -n ${NAMESPACE} --no-headers 2>/dev/null | wc -l)
RUNNING_PODS=$(kubectl get pods -n ${NAMESPACE} --no-headers 2>/dev/null | grep Running | wc -l)
info "Total Pods: ${TOTAL_PODS}"
info "Running Pods: ${RUNNING_PODS}"

if [ "$TOTAL_PODS" -ge 8 ]; then
    pass "Expected minimum 8 pods found"
else
    fail "Expected minimum 8 pods, found ${TOTAL_PODS}"
fi

if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ]; then
    pass "All pods are running"
else
    warn "${RUNNING_PODS}/${TOTAL_PODS} pods running"
fi

kubectl get pods -n ${NAMESPACE}

# Test 4: Check PodMonitors
print_header "TEST 4: PodMonitor Configuration"
PODMONITOR_COUNT=$(kubectl get podmonitor -n ${NAMESPACE} --no-headers 2>/dev/null | wc -l)
if [ "$PODMONITOR_COUNT" -ge 3 ]; then
    pass "PodMonitors exist (found ${PODMONITOR_COUNT})"
else
    fail "Expected 3 PodMonitors, found ${PODMONITOR_COUNT}"
fi

# Check for correct relabeling in dynamo-worker PodMonitor
if kubectl get podmonitor dynamo-worker -n ${NAMESPACE} -o yaml | grep -q "dynamo_namespace"; then
    pass "Worker PodMonitor has dynamo_namespace relabeling"
else
    fail "Worker PodMonitor missing dynamo_namespace relabeling"
fi

# Test 5: Check RBAC Permissions
print_header "TEST 5: Planner RBAC Permissions"
if kubectl get clusterrole dynamo-platform-dynamo-operator-planner -o yaml | grep -q "pods"; then
    pass "Planner ClusterRole has pod permissions"
else
    fail "Planner ClusterRole missing pod permissions"
fi

# Test 6: Check ConfigMap
print_header "TEST 6: Profiling Data ConfigMap"
if kubectl get configmap planner-profile-data -n ${NAMESPACE} &>/dev/null; then
    pass "Profiling data ConfigMap exists"
else
    fail "Profiling data ConfigMap not found"
fi

# Test 7: Check Power Limits
print_header "TEST 7: Power Limit Annotations"
PLANNER_POD=$(kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-component=Planner -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$PLANNER_POD" ]; then
    fail "Planner pod not found"
else
    pass "Planner pod found: ${PLANNER_POD}"

    # Check power limits on workers
    PREFILL_POWER=$(kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-sub-component-type=prefill -o jsonpath='{.items[0].metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}' 2>/dev/null)
    DECODE_POWER=$(kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-sub-component-type=decode -o jsonpath='{.items[0].metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}' 2>/dev/null)

    if [ "$PREFILL_POWER" == "250" ]; then
        pass "Prefill worker has power limit: ${PREFILL_POWER}W"
    else
        warn "Prefill worker power limit: ${PREFILL_POWER:-none}"
    fi

    if [ "$DECODE_POWER" == "250" ]; then
        pass "Decode worker has power limit: ${DECODE_POWER}W"
    else
        warn "Decode worker power limit: ${DECODE_POWER:-none}"
    fi
fi

# Test 8: Check Prometheus Connection
print_header "TEST 8: Prometheus Connectivity"
if kubectl exec -n ${NAMESPACE} ${PLANNER_POD} -- curl -s http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090/-/healthy &>/dev/null; then
    pass "Planner can reach Prometheus"
else
    fail "Planner cannot reach Prometheus"
fi

# Test 9: Check Prometheus Metrics with Labels
print_header "TEST 9: Prometheus Metrics and Labels"
METRIC_CHECK=$(kubectl exec -n ${NAMESPACE} ${PLANNER_POD} -- python3 -c "
from prometheus_api_client import PrometheusConnect
prom = PrometheusConnect(url='http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090', disable_ssl=True)
result = prom.custom_query(query='vllm:time_to_first_token_seconds_sum')
if result:
    for r in result:
        ns = r['metric'].get('dynamo_namespace', 'MISSING')
        model = r['metric'].get('model_name', 'MISSING')
        if ns != 'MISSING' and model != 'MISSING':
            print('OK')
            exit(0)
print('MISSING')
" 2>/dev/null)

if [ "$METRIC_CHECK" == "OK" ]; then
    pass "Prometheus metrics have correct labels (model_name, dynamo_namespace)"
else
    warn "Prometheus metrics missing required labels"
fi

# Test 10: Check Planner Logs
print_header "TEST 10: Planner Functionality"
# Check entire log for model detection (message appears early in logs)
if kubectl logs -n ${NAMESPACE} ${PLANNER_POD} | grep -q "Detected model name"; then
    pass "Planner detected model name"
else
    fail "Planner did not detect model name"
fi

if kubectl logs -n ${NAMESPACE} ${PLANNER_POD} --tail=50 | grep -q "Applied power limits"; then
    pass "Planner is applying power limits"
else
    warn "Planner has not applied power limits yet"
fi

# Test 11: Send Test Traffic
print_header "TEST 11: End-to-End Traffic Test"
info "Starting port-forward..."
pkill -9 -f "port-forward.*8000" 2>/dev/null || true
sleep 2
kubectl port-forward svc/vllm-disagg-frontend 8000:8000 -n ${NAMESPACE} > /tmp/verify_pf.log 2>&1 &
PF_PID=$!
sleep 5

info "Sending test request..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5}' 2>&1)

if echo "$RESPONSE" | grep -q "choices"; then
    pass "Frontend responded successfully"
    TOKENS=$(echo "$RESPONSE" | python3 -c "import json, sys; r=json.load(sys.stdin); print(r['usage']['total_tokens'])" 2>/dev/null || echo "0")
    info "Response tokens: ${TOKENS}"
else
    fail "Frontend did not respond correctly"
fi

# Test 12: Check Planner Observes Metrics
print_header "TEST 12: Planner Metric Observation"
info "Sending continuous traffic for 40 seconds..."
for i in {1..15}; do
    curl -s -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}], \"max_tokens\": 10}" > /dev/null &
    sleep 2
done

sleep 10
info "Waiting for planner observation cycle..."
sleep 5

# Check if planner observed non-zero metrics
OBSERVED_METRICS=$(kubectl logs -n ${NAMESPACE} ${PLANNER_POD} --tail=50 | grep "Observed num_req" | tail -1)
if echo "$OBSERVED_METRICS" | grep -q "Observed num_req: [1-9]"; then
    pass "Planner observed non-zero metrics"
    info "$OBSERVED_METRICS"
else
    warn "Planner showing zero metrics (may need more traffic/time)"
    info "$OBSERVED_METRICS"
fi

# Test 13: Verify GPU Power Limits Are Actually Set
print_header "TEST 13: GPU Power Limit Enforcement (Hardware)"
info "Checking nvidia-smi for actual GPU power limits..."

# Get GPUs with workloads
GPUS_WITH_WORKLOADS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)

if [ -z "$GPUS_WITH_WORKLOADS" ]; then
    warn "No GPU processes found - cannot verify power limits"
else
    # Check power limits on all GPUs
    ALL_LIMITS=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits)

    LIMITS_ENFORCED=0
    TOTAL_WORKLOAD_GPUS=0

    while IFS=, read -r GPU_IDX LIMIT; do
        GPU_IDX=$(echo $GPU_IDX | tr -d ' ')
        LIMIT=$(echo $LIMIT | tr -d ' ')

        # Check if this GPU has a workload
        GPU_UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | grep "^${GPU_IDX}," | cut -d',' -f2 | tr -d ' ')

        if echo "$GPUS_WITH_WORKLOADS" | grep -q "$GPU_UUID"; then
            ((TOTAL_WORKLOAD_GPUS++))
            info "GPU ${GPU_IDX}: ${LIMIT}W (has workload)"

            # Check if limit is 250W (allowing for small variance)
            if [ "${LIMIT%.*}" -ge 240 ] && [ "${LIMIT%.*}" -le 260 ]; then
                ((LIMITS_ENFORCED++))
            fi
        else
            # GPU without workload should remain at default (700W for H200)
            info "GPU ${GPU_IDX}: ${LIMIT}W (no workload)"
        fi
    done <<< "$ALL_LIMITS"

    if [ $TOTAL_WORKLOAD_GPUS -gt 0 ] && [ $LIMITS_ENFORCED -eq $TOTAL_WORKLOAD_GPUS ]; then
        pass "GPU power limits enforced on hardware ($LIMITS_ENFORCED/$TOTAL_WORKLOAD_GPUS GPUs at ~250W)"
    elif [ $TOTAL_WORKLOAD_GPUS -gt 0 ]; then
        fail "GPU power limits NOT fully enforced ($LIMITS_ENFORCED/$TOTAL_WORKLOAD_GPUS GPUs at correct limit)"
        info "Expected: 250W on GPUs with workloads, got mixed results"
        info "This may indicate the Power Agent needs to be restarted or /host/proc mount is not working"
    else
        warn "Could not verify - no GPUs with workloads found"
    fi
fi

# Cleanup
kill $PF_PID 2>/dev/null || true

# Final Summary
print_header "VERIFICATION SUMMARY"
echo ""
echo -e "${GREEN}Passed Tests: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed Tests: ${FAILED_TESTS}${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   ALL TESTS PASSED ✓                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              SOME TESTS FAILED OR WARNED                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    exit 1
fi

