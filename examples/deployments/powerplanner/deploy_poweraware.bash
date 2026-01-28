#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Deploy and test power-aware autoscaling features
# This script does everything in one go:
# 1. Builds Power Agent image
# 2. Builds custom planner image with power-aware code
# 3. Deploys Power Agent DaemonSet
# 4. Deploys vllm-disagg with power-aware planner

set -e

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

echo "========================================"
echo "Deploy Power-Aware Autoscaling"
echo "========================================"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking Prerequisites${NC}"
echo "----------------------"

if ! minikube status &>/dev/null; then
    echo -e "${RED}✗ Minikube is not running${NC}"
    echo "Please start minikube first:"
    echo "  cd ${DEV_REPO}/examples/deployments/powerplanner"
    echo "  bash deploy_poweraware_baseinfra.bash 1"
    exit 1
fi
echo -e "${GREEN}✓ Minikube is running${NC}"

if ! kubectl get namespace ${NAMESPACE} &>/dev/null; then
    echo -e "${RED}✗ Namespace ${NAMESPACE} not found${NC}"
    echo "Please deploy Dynamo platform first"
    exit 1
fi
echo -e "${GREEN}✓ Namespace ${NAMESPACE} exists${NC}"

if ! kubectl get configmap planner-profile-data -n ${NAMESPACE} &>/dev/null; then
    echo -e "${YELLOW}⚠ ConfigMap planner-profile-data not found${NC}"
    echo "Running profiling to generate profile data..."
    echo ""

    # Apply profiling DynamoGraphDeploymentRequest
    kubectl apply -f ${SCRIPT_DIR}/profile_sla_aic_dgdr.yaml -n ${NAMESPACE}

    # Wait for profiling pod to appear
    echo "Waiting for profiling pod to start..."
    sleep 5

    # Wait for ConfigMap to be created (check every 5 seconds, timeout after 10 minutes)
    echo "Waiting for profiling to complete (this may take 5-10 minutes)..."
    for i in {1..120}; do
        if kubectl get configmap planner-profile-data -n ${NAMESPACE} &>/dev/null; then
            echo -e "${GREEN}✓ Profiling data ConfigMap created${NC}"

            # Clean up profiling deployment
            echo "Cleaning up profiling resources..."
            kubectl delete dynamographdeploymentrequest sla-aic -n ${NAMESPACE} &>/dev/null || true
            sleep 2
            break
        fi

        # Show progress every 12 iterations (60 seconds)
        if [ $((i % 12)) -eq 0 ]; then
            ELAPSED=$((i*5))
            echo "  Still profiling... (${ELAPSED} seconds elapsed)"
            kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep profile-sla-aic | head -1 || echo "  (profiling pod status unknown)"
        fi

        sleep 5
    done

    # Final check
    if ! kubectl get configmap planner-profile-data -n ${NAMESPACE} &>/dev/null; then
        echo -e "${RED}✗ Profiling timed out or failed${NC}"
        PROFILE_POD=$(kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep profile-sla-aic | awk '{print $1}' | head -1)
        if [ -n "${PROFILE_POD}" ]; then
            echo "Check profiling pod logs:"
            echo "  kubectl logs -n ${NAMESPACE} ${PROFILE_POD}"
        fi
        echo "Check DynamoGraphDeploymentRequest:"
        echo "  kubectl describe dynamographdeploymentrequest sla-aic -n ${NAMESPACE}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Profiling data ConfigMap exists${NC}"
fi

echo ""

# Step 1: Build Power Agent image
echo -e "${BLUE}Step 1: Building Power Agent Image${NC}"
echo "------------------------------------"

cd ${DEV_REPO}/components/power_agent

echo "Building Power Agent image..."
docker build -t dynamo/power-agent:v1.0.0 . > /tmp/power-agent-build.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to build Power Agent image${NC}"
    tail -20 /tmp/power-agent-build.log
    exit 1
fi
echo -e "${GREEN}✓ Power Agent image built${NC}"

# Check if image is already in Minikube
if minikube image ls | grep -q "dynamo/power-agent.*v1.0.0"; then
    echo -e "${YELLOW}⚠ Image already in Minikube, skipping load${NC}"
else
    echo "Loading image into Minikube..."
    minikube image load dynamo/power-agent:v1.0.0
    echo -e "${GREEN}✓ Image loaded into Minikube${NC}"
fi

echo ""

# Step 2: Build custom planner image with power-aware code
echo -e "${BLUE}Step 2: Building Custom Planner Image${NC}"
echo "---------------------------------------"

cd ${DEV_REPO}

# Create temporary Dockerfile
cat > /tmp/Dockerfile.planner-custom <<'EOF'
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post2

# Copy updated planner code with power-aware features
COPY components/src/dynamo/planner /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/planner

# Verify the new arguments are available
RUN python3 -m dynamo.planner.planner_sla --help | grep -q "enable-power-awareness" && \
    echo "✓ Power-aware arguments detected" || \
    (echo "✗ Power-aware arguments not found" && exit 1)
EOF

docker build -f /tmp/Dockerfile.planner-custom -t dynamo/planner-power-aware:dev . > /tmp/planner-build.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to build custom planner image${NC}"
    tail -30 /tmp/planner-build.log
    exit 1
fi

echo -e "${GREEN}✓ Custom planner image built${NC}"

# Check if image is already in Minikube
if minikube image ls | grep -q "dynamo/planner-power-aware.*dev"; then
    echo -e "${YELLOW}⚠ Image already in Minikube, skipping load${NC}"
else
    echo "Loading image into Minikube..."
    minikube image load dynamo/planner-power-aware:dev
    echo -e "${GREEN}✓ Image loaded into Minikube${NC}"
fi

echo ""

# Step 3: Deploy Power Agent DaemonSet
echo -e "${BLUE}Step 3: Deploying Power Agent${NC}"
echo "-------------------------------"

kubectl apply -f ${DEV_REPO}/deploy/power_agent/daemonset.yaml

# Patch for Minikube compatibility
echo "Patching for Minikube (removing GPU node selector)..."
kubectl patch daemonset power-agent -n ${NAMESPACE} --type=json \
  -p='[{"op": "remove", "path": "/spec/template/spec/nodeSelector"}]' 2>/dev/null || true

kubectl patch daemonset power-agent -n ${NAMESPACE} --type=json \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/imagePullPolicy", "value": "Never"}]' 2>/dev/null || true

echo -e "${GREEN}✓ Power Agent deployed${NC}"
echo -e "${YELLOW}⚠ Note: Power Agent will crash in Minikube without real GPUs (expected)${NC}"

echo ""

# Step 3.5: Patch Planner ClusterRole and PodMonitor
echo -e "${BLUE}Step 3.5: Updating Planner RBAC and Prometheus Configuration${NC}"
echo "------------------------------------------------------------"

# Apply ClusterRole patch to allow planner to list and patch pods
kubectl apply -f ${DEV_REPO}/examples/deployments/powerplanner/planner-clusterrole-patch.yaml
echo -e "${GREEN}✓ Planner ClusterRole updated with pod permissions${NC}"

# Apply PodMonitor with correct relabeling rules for dynamo_namespace
kubectl apply -f ${DEV_REPO}/examples/deployments/powerplanner/dynamo-worker-podmonitor.yaml
echo -e "${GREEN}✓ Worker PodMonitor updated with label relabeling${NC}"

echo ""

# Step 4: Deploy vllm-disagg with power-aware planner
echo -e "${BLUE}Step 4: Deploying vllm-disagg with Power-Aware Planner${NC}"
echo "--------------------------------------------------------"

# Create deployment with power-aware planner
cat > /tmp/vllm-disagg-power-aware.yaml <<EOF
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-disagg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post2

    Planner:
      dynamoNamespace: vllm-disagg
      componentType: planner
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: dynamo/planner-power-aware:dev
          imagePullPolicy: Never
          command:
            - python3
            - -m
            - dynamo.planner.planner_sla
          args:
            - --environment=kubernetes
            - --backend=vllm
            - --model-name=Qwen/Qwen3-0.6B
            - --adjustment-interval=30
            - --max-gpu-budget=8
            - --min-endpoint=1
            - --decode-engine-num-gpu=1
            - --prefill-engine-num-gpu=1
            - --ttft=1000
            - --itl=50
            - --enable-power-awareness
            - --total-gpu-power-limit=1000
            - --prefill-engine-gpu-power-limit=250
            - --decode-engine-gpu-power-limit=250
            - --profile-results-dir=/workspace/profiling_results
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          volumeMounts:
            - name: planner-profile-data
              mountPath: /workspace/profiling_results
              readOnly: true
        volumes:
          - name: planner-profile-data
            configMap:
              name: planner-profile-data

    VllmDecodeWorker:
      dynamoNamespace: vllm-disagg
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post2
          workingDir: /workspace/examples/backends/vllm
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --is-decode-worker

    VllmPrefillWorker:
      dynamoNamespace: vllm-disagg
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0.post2
          workingDir: /workspace/examples/backends/vllm
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --is-prefill-worker
EOF

kubectl apply -f /tmp/vllm-disagg-power-aware.yaml -n ${NAMESPACE}

echo -e "${GREEN}✓ Deployment applied${NC}"

echo ""
echo "Waiting for planner pod to start..."
sleep 15

# Step 5: Verify deployment
echo ""
echo -e "${BLUE}Step 5: Verifying Deployment${NC}"
echo "-----------------------------"

PLANNER_POD=$(kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-component=Planner 2>/dev/null | grep Running | awk 'NR==1 {print $1}')

if [ -z "${PLANNER_POD}" ]; then
    echo -e "${YELLOW}⚠ Planner pod not running yet${NC}"
    echo ""
    echo "Check status:"
    kubectl get pods -n ${NAMESPACE} | grep planner
else
    echo -e "${GREEN}✓ Planner pod running: ${PLANNER_POD}${NC}"

    echo ""
    echo "Checking planner logs..."
    kubectl logs -n ${NAMESPACE} ${PLANNER_POD} --tail=30 2>&1 | head -20
fi

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo -e "${GREEN}✓ Power Agent deployed${NC} (will crash without real GPUs - expected)"
echo -e "${GREEN}✓ Custom planner image built and deployed${NC}"
echo -e "${GREEN}✓ Power-aware autoscaling enabled${NC}"
echo -e "${GREEN}✓ Profiling data mounted from ConfigMap${NC}"
echo ""
echo "Configuration:"
echo "  - Total GPU power budget: 1000W"
echo "  - Prefill GPU power limit: 250W"
echo "  - Decode GPU power limit: 250W"
echo "  - Profile data: planner-profile-data ConfigMap"
echo ""
echo "Monitor planner logs:"
echo "  kubectl logs -f -n ${NAMESPACE} ${PLANNER_POD:-<planner-pod>}"
echo ""
echo "Check pod power limit annotations:"
echo "  kubectl get pods -n ${NAMESPACE} -o custom-columns=NAME:.metadata.name,POWER-LIMIT:.metadata.annotations.dynamo\\\\.nvidia\\\\.com/gpu-power-limit"
echo ""
echo "Send test traffic:"
echo "  kubectl port-forward svc/vllm-disagg-frontend 8000:8000 -n ${NAMESPACE} &"
echo "  curl -X POST http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "Use monitor script for real-time view:"
echo "  bash ${DEV_REPO}/examples/deployments/powerplanner/monitor_poweraware.bash"
echo ""


