#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy either scenario 1 or scenario 2 for SLO comparison testing

set -e

# ============================================================================
# SLO CONFIGURATION - Single source of truth for all SLO targets
# ============================================================================
# Stricter SLO targets (Stream 1, Deployment A, SLA_MODE=min)
export LOW_TTFT=100      # milliseconds
export LOW_ITL=5.0       # milliseconds

# Relaxed SLO targets (Stream 2, Deployment B, SLA_MODE=max)
export HIGH_TTFT=150     # milliseconds
export HIGH_ITL=10.0      # milliseconds

# Format for aiperf goodput arguments
export STREAM1_GOODPUT="time_to_first_token:${LOW_TTFT} inter_token_latency:${LOW_ITL}"
export STREAM2_GOODPUT="time_to_first_token:${HIGH_TTFT} inter_token_latency:${HIGH_ITL}"
# ============================================================================

NAMESPACE="${NAMESPACE:-default}"
SCENARIO="${SCENARIO:-1}"
SLA_MODE="${SLA_MODE:-min}"  # For Scenario 1: "min" (stricter) or "max" (relaxed)

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================"
echo "Deploying SLO Test Scenario $SCENARIO"
echo "======================================${NC}"
echo "Namespace: ${NAMESPACE}"
echo ""

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}ERROR: kubectl not found${NC}"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}ERROR: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

# Check for HuggingFace secret
if ! kubectl get secret hf-token-secret -n ${NAMESPACE} &> /dev/null; then
    echo -e "${YELLOW}WARNING: hf-token-secret not found in namespace ${NAMESPACE}${NC}"
    echo "Create it with:"
    echo "  kubectl create secret generic hf-token-secret \\"
    echo "    --from-literal=token=your_huggingface_token \\"
    echo "    -n ${NAMESPACE}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Deploy based on scenario
if [[ "$SCENARIO" == "1" ]]; then
    echo -e "${GREEN}Deploying Scenario 1: Single DynamoGraphDeployment${NC}"
    echo "This will create one deployment with local planner"
    echo "SLA Mode: ${SLA_MODE} (min=stricter TTFT:100/ITL:3.0, max=relaxed TTFT:150/ITL:4.0)"
    echo ""
    
    if [[ ! -f "deployment_scenario1.yaml" ]]; then
        echo -e "${RED}ERROR: deployment_scenario1.yaml not found${NC}"
        exit 1
    fi
    
    # Validate SLA_MODE
    if [[ "$SLA_MODE" != "min" && "$SLA_MODE" != "max" ]]; then
        echo -e "${RED}ERROR: SLA_MODE must be 'min' or 'max', got '${SLA_MODE}'${NC}"
        exit 1
    fi
    
    # Determine which SLA values to use based on SLA_MODE
    if [[ "$SLA_MODE" == "min" ]]; then
        PLANNER_TTFT=${LOW_TTFT}
        PLANNER_ITL=${LOW_ITL}
        echo "SLA Mode: min (stricter) - TTFT=${PLANNER_TTFT}ms, ITL=${PLANNER_ITL}ms"
    else
        PLANNER_TTFT=${HIGH_TTFT}
        PLANNER_ITL=${HIGH_ITL}
        echo "SLA Mode: max (relaxed) - TTFT=${PLANNER_TTFT}ms, ITL=${PLANNER_ITL}ms"
    fi
    
    # Apply deployment with SLO value substitution
    echo "Applying deployment..."
    
    # Substitute placeholders and apply
    # This ensures all values are set correctly BEFORE the CRD is created
    sed -e "s/__PLANNER_TTFT__/${PLANNER_TTFT}/g" \
        -e "s/__PLANNER_ITL__/${PLANNER_ITL}/g" \
        deployment_scenario1.yaml | \
        kubectl apply -f - -n ${NAMESPACE}
    
    echo ""
    echo "Waiting for deployment to be ready..."
    kubectl wait --for=jsonpath='{.status.state}'=Ready \
        dynamographdeployment/llama-single \
        -n ${NAMESPACE} \
        --timeout=1200s || true
    
    echo ""
    echo -e "${GREEN}Scenario 1 deployed successfully${NC}"
    echo "SLA Mode: ${SLA_MODE}"
    echo "Frontend accessible at: llama-single-frontend.${NAMESPACE}.svc.cluster.local:8000"
    
elif [[ "$SCENARIO" == "2" ]]; then
    echo -e "${GREEN}Deploying Scenario 2: Global Planner/Scheduler with Two Deployments${NC}"
    echo "This will create scheduler, global planner, and two deployments"
    echo "  Deployment A: TTFT=${LOW_TTFT}ms, ITL=${LOW_ITL}ms"
    echo "  Deployment B: TTFT=${HIGH_TTFT}ms, ITL=${HIGH_ITL}ms"
    echo ""
    
    if [[ ! -f "deployment_scenario2.yaml" ]]; then
        echo -e "${RED}ERROR: deployment_scenario2.yaml not found${NC}"
        exit 1
    fi
    
    # Substitute SLO values before applying
    sed -e "s/__LOW_TTFT__/${LOW_TTFT}/g" \
        -e "s/__LOW_ITL__/${LOW_ITL}/g" \
        -e "s/__HIGH_TTFT__/${HIGH_TTFT}/g" \
        -e "s/__HIGH_ITL__/${HIGH_ITL}/g" \
        deployment_scenario2.yaml | \
        kubectl apply -f - -n ${NAMESPACE}
    
    echo ""
    echo "Waiting for scheduler..."
    kubectl wait --for=condition=available --timeout=300s \
        deployment/dynamo-scheduler -n ${NAMESPACE} || true
    
    echo "Waiting for global planner..."
    kubectl wait --for=condition=available --timeout=300s \
        deployment/dynamo-global-planner -n ${NAMESPACE} || true
    
    echo "Waiting for deployments to be ready..."
    kubectl wait --for=jsonpath='{.status.state}'=Ready \
        dynamographdeployment/llama-deployment-a \
        dynamographdeployment/llama-deployment-b \
        -n ${NAMESPACE} \
        --timeout=1200s || true
    
    echo ""
    echo -e "${GREEN}Scenario 2 deployed successfully${NC}"
    echo "Scheduler accessible at: dynamo-scheduler.${NAMESPACE}.svc.cluster.local"
    
else
    echo -e "${RED}ERROR: Invalid scenario '$SCENARIO'. Must be 1 or 2${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================"
echo "Deployment Status"
echo "======================================${NC}"
kubectl get dynamographdeployment -n ${NAMESPACE}
echo ""
kubectl get pods -n ${NAMESPACE}
echo ""

echo -e "${YELLOW}To run the test:${NC}"
echo "  ./run_test.sh"
echo ""

