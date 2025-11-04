#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run SLO comparison test

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source SLO configuration from deploy.sh
# Extract SLO values using grep/sed (strip comments before removing spaces)
LOW_TTFT=$(grep "^export LOW_TTFT=" "${SCRIPT_DIR}/deploy.sh" | cut -d= -f2 | cut -d'#' -f1 | tr -d ' ')
LOW_ITL=$(grep "^export LOW_ITL=" "${SCRIPT_DIR}/deploy.sh" | cut -d= -f2 | cut -d'#' -f1 | tr -d ' ')
HIGH_TTFT=$(grep "^export HIGH_TTFT=" "${SCRIPT_DIR}/deploy.sh" | cut -d= -f2 | cut -d'#' -f1 | tr -d ' ')
HIGH_ITL=$(grep "^export HIGH_ITL=" "${SCRIPT_DIR}/deploy.sh" | cut -d= -f2 | cut -d'#' -f1 | tr -d ' ')

# Build goodput strings
STREAM1_GOODPUT="time_to_first_token:${LOW_TTFT} inter_token_latency:${LOW_ITL}"
STREAM2_GOODPUT="time_to_first_token:${HIGH_TTFT} inter_token_latency:${HIGH_ITL}"

# Default values
NAMESPACE="${NAMESPACE:-default}"
SAVE_RESULTS="${SAVE_RESULTS:-true}"
TEST_SCENARIO="${TEST_SCENARIO:-both}"  # Options: "1", "2", or "both"
KEEP_ALIVE_SECONDS="${KEEP_ALIVE_SECONDS:-7200}"  # Keep pod alive after test (default: 2 hours)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TEST_POD_NAME="dynamo-slo-comparison-test"

echo -e "${GREEN}==================================================================================${NC}"
echo -e "${GREEN}SLO Comparison Test${NC}"
echo -e "${GREEN}==================================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Namespace: $NAMESPACE"
echo "  Test Pod: $TEST_POD_NAME"
echo "  Test Scenario: $TEST_SCENARIO"
echo "  Keep Alive: $((KEEP_ALIVE_SECONDS / 60)) minutes after test"
echo ""

# Validate TEST_SCENARIO
if [[ "$TEST_SCENARIO" != "1" && "$TEST_SCENARIO" != "2" && "$TEST_SCENARIO" != "both" ]]; then
    echo -e "${RED}ERROR: TEST_SCENARIO must be '1', '2', or 'both', got '${TEST_SCENARIO}'${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up test resources...${NC}"
    kubectl delete pod "$TEST_POD_NAME" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete configmap "$TEST_POD_NAME-script" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete rolebinding "$TEST_POD_NAME-binding" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete role "$TEST_POD_NAME-role" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete serviceaccount "$TEST_POD_NAME-sa" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
}

# Only cleanup on exit if user wants
if [[ "${CLEANUP_ON_EXIT}" == "true" ]]; then
    trap cleanup EXIT
fi

# Step 1: Create RBAC for test pod
echo -e "${GREEN}Step 1: Creating RBAC for test pod${NC}"

cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: $TEST_POD_NAME-sa
  labels:
    app: slo-comparison-test
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: $TEST_POD_NAME-role
  labels:
    app: slo-comparison-test
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["nvidia.com"]
  resources: ["dynamographdeployments"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: $TEST_POD_NAME-binding
  labels:
    app: slo-comparison-test
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: $TEST_POD_NAME-role
subjects:
- kind: ServiceAccount
  name: $TEST_POD_NAME-sa
  namespace: $NAMESPACE
EOF

echo -e "${GREEN}✓ RBAC created${NC}"
echo ""

# Step 2: Create ConfigMap with test script
echo -e "${GREEN}Step 2: Creating ConfigMap with test script${NC}"

if [[ ! -f "test_slo_comparison.py" ]]; then
    echo -e "${RED}Error: test_slo_comparison.py not found${NC}"
    exit 1
fi

kubectl delete configmap "$TEST_POD_NAME-script" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true

kubectl create configmap "$TEST_POD_NAME-script" \
    --from-file=test_slo_comparison.py=test_slo_comparison.py \
    -n "$NAMESPACE"

echo -e "${GREEN}✓ ConfigMap created${NC}"
echo ""

# Step 3: Create and run test pod
echo -e "${GREEN}Step 3: Creating test pod${NC}"

kubectl delete pod "$TEST_POD_NAME" -n "$NAMESPACE" --ignore-not-found --wait=true 2>/dev/null || true

cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: $TEST_POD_NAME
  labels:
    app: slo-comparison-test
spec:
  serviceAccountName: $TEST_POD_NAME-sa
  restartPolicy: Never
  containers:
  - name: test
    image: python:3.12-slim
    command:
      - /bin/bash
      - -c
      - |
        set -e
        
        echo "Installing dependencies..."
        apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
        
        # Install Python packages
        pip install --no-cache-dir aiperf kubernetes httpx
        
        # Create test directory and copy script
        mkdir -p /test
        cp /test-script/test_slo_comparison.py /test/
        cd /test
        
        echo "Starting SLO comparison test..."
        echo "  Test scenario: ${TEST_SCENARIO}"
        echo "  Stream 1 goodput: ${STREAM1_GOODPUT}"
        echo "  Stream 2 goodput: ${STREAM2_GOODPUT}"
        python3 test_slo_comparison.py \
          --namespace $NAMESPACE \
          --single-deployment llama-single \
          --deployment-a llama-deployment-a \
          --deployment-b llama-deployment-b \
          --scheduler-url http://dynamo-scheduler.$NAMESPACE.svc.cluster.local \
          --stream1-goodput "${STREAM1_GOODPUT}" \
          --stream2-goodput "${STREAM2_GOODPUT}" \
          --scenario ${TEST_SCENARIO} \
          --save-results
        
        TEST_EXIT_CODE=\$?
        
        echo ""
        echo "Test completed with exit code: \$TEST_EXIT_CODE"
        
        if [ -d test_results ]; then
          echo "Test results saved to /test/test_results"
          ls -lh test_results/
        fi
        
        # Keep container running for result retrieval
        KEEP_ALIVE_MIN=\$((${KEEP_ALIVE_SECONDS} / 60))
        echo ""
        echo "======================================================================"
        echo "Test pod will stay alive for \${KEEP_ALIVE_MIN} minutes for result retrieval"
        echo "======================================================================"
        echo "To copy results:"
        echo "  kubectl cp $NAMESPACE/$TEST_POD_NAME:/test/test_results ./test_results"
        echo ""
        echo "To view results:"
        echo "  kubectl exec $TEST_POD_NAME -n $NAMESPACE -- cat /test/test_results/*/slo_comparison_results.json"
        echo ""
        echo "To delete pod immediately:"
        echo "  kubectl delete pod $TEST_POD_NAME -n $NAMESPACE"
        echo "======================================================================"
        echo ""
        
        # Sleep to allow result retrieval
        sleep ${KEEP_ALIVE_SECONDS}
        
        exit \$TEST_EXIT_CODE
    env:
    - name: TEST_SCENARIO
      value: "${TEST_SCENARIO}"
    - name: STREAM1_GOODPUT
      value: "${STREAM1_GOODPUT}"
    - name: STREAM2_GOODPUT
      value: "${STREAM2_GOODPUT}"
    volumeMounts:
    - name: test-script
      mountPath: /test-script
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"
      limits:
        memory: "8Gi"
        cpu: "4"
  volumes:
  - name: test-script
    configMap:
      name: $TEST_POD_NAME-script
EOF

echo -e "${GREEN}✓ Test pod created${NC}"
echo ""

# Step 4: Wait for pod and stream logs
echo -e "${GREEN}Step 4: Running test (streaming logs)${NC}"
echo -e "${GREEN}==================================================================================${NC}"
echo ""

echo "Waiting for test pod to start..."
kubectl wait --for=condition=Ready pod/"$TEST_POD_NAME" -n "$NAMESPACE" --timeout=300s || {
    echo -e "${YELLOW}Pod didn't become ready, checking status...${NC}"
    kubectl get pod "$TEST_POD_NAME" -n "$NAMESPACE"
}

echo ""
echo -e "${GREEN}Test output:${NC}"
echo "---"
kubectl logs -f "$TEST_POD_NAME" -n "$NAMESPACE" 2>&1 || true

# Step 5: Retrieve test results (pod stays alive)
echo ""
echo "---"
echo -e "${GREEN}Step 5: Retrieving test results${NC}"
echo "Note: Pod will remain running for $((KEEP_ALIVE_SECONDS / 60)) minutes"
echo ""

# Wait a moment for files to be written
sleep 5

POD_STATUS=$(kubectl get pod "$TEST_POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
echo "Pod Status: $POD_STATUS"

if [[ "$POD_STATUS" == "Running" ]]; then
    # Copy results while pod is alive
    RESULT_FILES=$(kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- \
        bash -c "ls /test/test_results/*/slo_comparison_results.json 2>/dev/null || echo ''" || echo "")
    
    if [[ -n "$RESULT_FILES" ]]; then
        echo "Found result files:"
        echo "$RESULT_FILES"
        
        LATEST_RESULT=$(echo "$RESULT_FILES" | tail -1)
        RESULT_FILENAME="slo_comparison_results_$(date +%s).json"
        
        echo "Copying results to local directory..."
        kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- \
            cat "$LATEST_RESULT" > "$RESULT_FILENAME" 2>/dev/null || true
        
        if [[ -f "$RESULT_FILENAME" ]]; then
            echo -e "${GREEN}✓ Results saved to: $RESULT_FILENAME${NC}"
        fi
    else
        echo "No result files found in pod yet"
    fi
    
    echo ""
    echo -e "${GREEN}==================================================================================${NC}"
    echo -e "${GREEN}TEST RUNNER COMPLETE${NC}"
    echo -e "${GREEN}==================================================================================${NC}"
    echo ""
    echo "The test pod is still running and will auto-delete in $((KEEP_ALIVE_SECONDS / 60)) minutes"
    echo ""
    echo "Commands:"
    echo "  View logs:        kubectl logs $TEST_POD_NAME -n $NAMESPACE"
    echo "  Copy results:     kubectl cp $NAMESPACE/$TEST_POD_NAME:/test/test_results ./test_results"
    echo "  Delete pod now:   kubectl delete pod $TEST_POD_NAME -n $NAMESPACE"
    echo ""
    echo "Cleanup all resources:"
    echo "  kubectl delete pod $TEST_POD_NAME -n $NAMESPACE"
    echo "  kubectl delete configmap $TEST_POD_NAME-script -n $NAMESPACE"
    echo "  kubectl delete rolebinding $TEST_POD_NAME-binding -n $NAMESPACE"
    echo "  kubectl delete role $TEST_POD_NAME-role -n $NAMESPACE"
    echo "  kubectl delete serviceaccount $TEST_POD_NAME-sa -n $NAMESPACE"
    echo ""
else
    echo -e "${YELLOW}Warning: Pod status is $POD_STATUS (expected Running)${NC}"
    echo "You may need to wait longer or check pod status manually"
    echo "  kubectl get pod $TEST_POD_NAME -n $NAMESPACE"
    echo "  kubectl logs $TEST_POD_NAME -n $NAMESPACE"
fi

