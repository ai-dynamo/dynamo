#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run SLA planner scaling end-to-end test
# This script:
# 1. Deploys the disaggregated planner if not already running
# 2. Sets up port forwarding to localhost:8000
# 3. Waits for the deployment to be ready
# 4. Runs the hardcoded scaling test (12 req/s -> 24 req/s)
# 5. Cleans up

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="$SCRIPT_DIR/disagg_planner.yaml"
TEST_FILE="$SCRIPT_DIR/../test_scaling_e2e.py"
FRONTEND_PORT=8000
LOCAL_PORT=8000
DEPLOYMENT_NAME="vllm-disagg-planner"
SAVE_RESULTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please ensure it is installed and in your PATH."
        exit 1
    fi

    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python not found. Please install Python."
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi

    # Check for aiperf
    if ! command -v aiperf &> /dev/null; then
        log_error "aiperf not found. This tool is required for load generation."
        log_error "Please install the required dependencies by following the instructions in tests/planner/README.md"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Check if deployment already exists and is running
check_existing_deployment() {
    log_info "Checking for existing deployment..."

    # Check for the DynamoGraphDeployment custom resource
    if kubectl get dynamographdeployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_info "DynamoGraphDeployment $DEPLOYMENT_NAME already exists - skipping redeployment"

        # Check if the DynamoGraphDeployment is ready
        local status
        status=$(kubectl get dynamographdeployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}')
        if [ "$status" = "successful" ]; then
            # Check if frontend pod is running
            # Note: operator automatically prefixes k8s namespace to dynamo-namespace
            if kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=${NAMESPACE}-vllm-disagg-planner" --field-selector=status.phase=Running | grep -q .; then
                log_success "Existing deployment is ready"
                return 0
            else
                log_warning "Existing deployment pods are not ready, will redeploy"
                return 1
            fi
        else
            log_warning "Existing deployment is not ready (status: $status), will redeploy"
            return 1
        fi
    else
        log_info "No existing deployment found"
        return 1
    fi
}

# Deploy the planner
deploy_planner() {
    log_info "Deploying SLA planner..."

    if [ ! -f "$YAML_FILE" ]; then
        log_error "Deployment file $YAML_FILE not found"
        exit 1
    fi

    # Apply the deployment
    if kubectl apply -f "$YAML_FILE" -n "$NAMESPACE"; then
        log_success "Deployment applied successfully"
    else
        log_error "Failed to apply deployment"
        exit 1
    fi

    log_info "Waiting for DynamoGraphDeployment to be processed..."
    if kubectl wait --for=condition=Ready dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s; then
        log_success "DynamoGraphDeployment is ready"
    else
        log_error "DynamoGraphDeployment failed to become ready within timeout"
        exit 1
    fi

    log_info "Waiting for pods to be running (this may take several minutes for image pulls)..."

    log_info "Waiting for frontend pod..."
    # Note: operator automatically prefixes k8s namespace to dynamo-namespace
    if kubectl wait --for=condition=Ready pod -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=${NAMESPACE}-vllm-disagg-planner" -n "$NAMESPACE" --timeout=900s; then
        log_success "Frontend pod is ready"
    else
        log_error "Frontend pod failed to become ready within timeout"
        exit 1
    fi

    log_info "Waiting for all pods to be running..."
    sleep 30
}

setup_port_forward() {
    log_info "Setting up port forwarding..."

    # Kill any existing port forward on the same port
    if lsof -ti:$LOCAL_PORT &> /dev/null; then
        log_warning "Port $LOCAL_PORT is already in use, attempting to free it..."
        kill "$(lsof -ti:$LOCAL_PORT)" 2>/dev/null || true
        sleep 2
    fi

    local frontend_service="vllm-disagg-planner-frontend"

    if ! kubectl get service "$frontend_service" -n "$NAMESPACE" &> /dev/null; then
        log_error "Frontend service '$frontend_service' not found"
        return 1
    fi

    log_info "Port forwarding to service: $frontend_service"
    kubectl port-forward service/"$frontend_service" "$LOCAL_PORT:$FRONTEND_PORT" -n "$NAMESPACE" >/dev/null 2>&1 &
    PORT_FORWARD_PID=$!

    log_info "Waiting for port forwarding to be established..."
    for i in {1..30}; do
        if curl -s http://localhost:$LOCAL_PORT/health &> /dev/null; then
            log_success "Port forwarding established and service is healthy"
            return 0
        fi
        sleep 2
    done

    log_error "Failed to establish port forwarding or service is not healthy"
    return 1
}

cleanup_port_forward() {
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        log_info "Cleaning up port forwarding..."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        wait $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

cleanup_deployment() {
    log_info "Cleaning up deployment..."
    kubectl delete -f "$YAML_FILE" -n "$NAMESPACE" --ignore-not-found

    log_info "Waiting for cleanup to complete..."
    kubectl wait --for=delete dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=120s || true

    log_info "Cleanup complete"
}

run_test() {
    log_info "Running scaling test (graduated 8->18 req/s)..."

    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi

    local test_args="--namespace $NAMESPACE"
    if [ "$SAVE_RESULTS" = true ]; then
        test_args="$test_args --save-results"
        log_info "Results will be saved to tests/planner/e2e_scaling_results"
    fi

    if $python_cmd "$TEST_FILE" $test_args; then
        log_success "Scaling test PASSED"
        return 0
    else
        log_error "Scaling test FAILED"
        return 1
    fi
}

main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --save-results)
                SAVE_RESULTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--namespace NS] [--save-results]"
                echo ""
                echo "Run SLA planner scaling test (graduated 8->15->25 req/s prefill scaling)"
                echo ""
                echo "Options:"
                echo "  --namespace NS    Kubernetes namespace (default: default)"
                echo "  --save-results    Save results to tests/planner/e2e_scaling_results instead of /tmp"
                echo "  --help            Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    log_info "SLA Planner Scaling Test"
    log_info "Namespace: $NAMESPACE"
    log_info "Scenario: Graduated 8->18 req/s (1P1D -> 2P1D prefill scaling, ISL=4000/OSL=150)"

    check_prerequisites

    trap cleanup_port_forward EXIT

    # Check if we need to deploy
    local deployed_by_us=false
    if ! check_existing_deployment; then
        deploy_planner
        deployed_by_us=true
    fi

    if ! setup_port_forward; then
        log_error "Failed to setup port forwarding"
        exit 1
    fi

    local test_result=0
    if ! run_test; then
        test_result=1
    fi

    # Only cleanup deployment if we deployed it
    if [ "$deployed_by_us" = true ]; then
        cleanup_deployment
    fi

    if [ $test_result -eq 0 ]; then
        log_success "Test completed successfully!"
    else
        log_error "Test failed!"
    fi

    exit $test_result
}

main "$@"