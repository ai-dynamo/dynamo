#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run SLA planner scaling end-to-end test
# This script:
# 1. Deploys the disaggregated planner if not already running
# 2. Sets up port forwarding to localhost:8000
# 3. Waits for the deployment to be ready
# 4. Runs the hardcoded scaling test (10 req/s -> 20 req/s)
# 5. Cleans up

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
YAML_FILE="disagg_planner.yaml"
FRONTEND_PORT=8000
LOCAL_PORT=8000
DEPLOYMENT_NAME="vllm-disagg-planner"

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
        log_error "kubectl not found. Please install kubectl."
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

    if [ ! -f "test_scaling_e2e.py" ]; then
        log_error "test_scaling_e2e.py not found. Make sure you're in the tests/planner directory."
        exit 1
    fi

    # Check for genai-perf
    if ! command -v genai-perf &> /dev/null; then
        log_warning "genai-perf not found. This tool is required for load generation."
        echo -n "Would you like us to install it for you? (y/n): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            log_info "Installing genai-perf and perf_analyzer..."
            if pip install nvidia-ml-py3 genai-perf tritonclient[all]; then
                log_success "genai-perf and perf_analyzer installed successfully"
            else
                log_error "Failed to install genai-perf. Please install it manually: pip install nvidia-ml-py3 genai-perf tritonclient[all]"
                exit 1
            fi
        else
            log_error "genai-perf is required for the scaling test. Please install it: pip install nvidia-ml-py3 genai-perf tritonclient[all]"
            exit 1
        fi
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
        local status=$(kubectl get dynamographdeployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}')
        if [ "$status" = "successful" ]; then
            # Check if frontend pod is running (main indicator)
            if kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=vllm-disagg-planner" --field-selector=status.phase=Running | grep -q .; then
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

    # Wait for DynamoGraphDeployment to be processed
    log_info "Waiting for DynamoGraphDeployment to be processed..."
    if kubectl wait --for=condition=Ready dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s; then
        log_success "DynamoGraphDeployment is ready"
    else
        log_error "DynamoGraphDeployment failed to become ready within timeout"
        exit 1
    fi

    # Wait for pods to be running (this may take a while for image pulls)
    log_info "Waiting for pods to be running (this may take several minutes for image pulls)..."

    # Wait for frontend pod (main component we need for testing)
    log_info "Waiting for frontend pod..."
    if kubectl wait --for=condition=Ready pod -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=vllm-disagg-planner" -n "$NAMESPACE" --timeout=900s; then
        log_success "Frontend pod is ready"
    else
        log_error "Frontend pod failed to become ready within timeout"
        exit 1
    fi

    # Wait a bit more for all pods to be fully running
    log_info "Waiting for all pods to be running..."
    sleep 30
}

# Setup port forwarding
setup_port_forward() {
    log_info "Setting up port forwarding..."

    # Kill any existing port forward on the same port
    if lsof -ti:$LOCAL_PORT &> /dev/null; then
        log_warning "Port $LOCAL_PORT is already in use, attempting to free it..."
        kill $(lsof -ti:$LOCAL_PORT) 2>/dev/null || true
        sleep 2
    fi

    # Start port forwarding to frontend pod directly
    local frontend_pod=$(kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=vllm-disagg-planner" | awk 'NR==2 {print $1}')
    if [ -z "$frontend_pod" ]; then
        log_error "Frontend pod not found"
        return 1
    fi

    log_info "Port forwarding to pod: $frontend_pod"
    kubectl port-forward pod/"$frontend_pod" "$LOCAL_PORT:$FRONTEND_PORT" -n "$NAMESPACE" &
    PORT_FORWARD_PID=$!

    # Wait for port forwarding to be established
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

# Clean up port forwarding
cleanup_port_forward() {
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        log_info "Cleaning up port forwarding..."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        wait $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Clean up deployment
cleanup_deployment() {
    log_info "Cleaning up deployment..."
    kubectl delete -f "$YAML_FILE" -n "$NAMESPACE" --ignore-not-found

    # Wait for cleanup to complete
    log_info "Waiting for cleanup to complete..."
    kubectl wait --for=delete dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=120s || true

    log_info "Cleanup complete"
}

# Run the scaling test
run_test() {
    log_info "Running scaling test (10 req/s -> 20 req/s)..."

    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi

    if $python_cmd test_scaling_e2e.py --namespace "$NAMESPACE"; then
        log_success "Scaling test PASSED"
        return 0
    else
        log_error "Scaling test FAILED"
        return 1
    fi
}

# Main function
main() {
    # Parse namespace argument if provided
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--namespace NS]"
                echo ""
                echo "Run SLA planner scaling test (hardcoded 10 req/s -> 20 req/s scenario)"
                echo ""
                echo "Options:"
                echo "  --namespace NS    Kubernetes namespace (default: default)"
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
    log_info "Scenario: 10 req/s -> 20 req/s (1P1D -> 1P2D)"

    # Check prerequisites
    check_prerequisites

    # Setup trap for cleanup
    trap cleanup_port_forward EXIT

    # Check if we need to deploy
    if ! check_existing_deployment; then
        deploy_planner
    fi

    # Setup port forwarding
    if ! setup_port_forward; then
        log_error "Failed to setup port forwarding"
        exit 1
    fi

    # Run the test
    local test_result=0
    if ! run_test; then
        test_result=1
    fi

    # Always cleanup deployment
    cleanup_deployment

    if [ $test_result -eq 0 ]; then
        log_success "Test completed successfully!"
    else
        log_error "Test failed!"
    fi

    exit $test_result
}

# Run main function with all arguments
main "$@"