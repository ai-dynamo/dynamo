#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

NAMESPACE=${NAMESPACE:-default}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="$SCRIPT_DIR/scaling_e2e.py"
FRONTEND_PORT=8000
LOCAL_PORT=""
DEPLOYMENT_NAME="vllm-disagg-planner"
SAVE_RESULTS=false
MODE="throughput"
DEPLOYED_BY_US=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

find_free_local_port() {
    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi

    "$python_cmd" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

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

    if ! command -v aiperf &> /dev/null; then
        log_error "aiperf not found. This tool is required for load generation."
        log_error "Follow components/src/dynamo/planner/tests/manual/README.md for setup."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

check_existing_deployment() {
    log_info "Checking for existing deployment..."

    if kubectl get dynamographdeployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" &> /dev/null; then
        log_info "DynamoGraphDeployment $DEPLOYMENT_NAME already exists - skipping redeployment"

        local status
        status=$(kubectl get dynamographdeployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}')
        if [ "$status" = "successful" ]; then
            if kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=${NAMESPACE}-vllm-disagg-planner" --field-selector=status.phase=Running | grep -q .; then
                log_success "Existing deployment is ready"
                return 0
            fi
            log_warning "Existing deployment pods are not ready, will redeploy"
            return 1
        fi

        log_warning "Existing deployment is not ready (status: $status), will redeploy"
        return 1
    fi

    log_info "No existing deployment found"
    return 1
}

deploy_planner() {
    log_info "Deploying SLA planner..."

    if [ ! -f "$YAML_FILE" ]; then
        log_error "Deployment file $YAML_FILE not found"
        exit 1
    fi

    kubectl apply -f "$YAML_FILE" -n "$NAMESPACE"
    log_success "Deployment applied successfully"

    log_info "Waiting for DynamoGraphDeployment to be processed..."
    kubectl wait --for=condition=Ready dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s
    log_success "DynamoGraphDeployment is ready"

    log_info "Waiting for frontend pod..."
    kubectl wait --for=condition=Ready pod -l "nvidia.com/dynamo-component-type=frontend,nvidia.com/dynamo-namespace=${NAMESPACE}-vllm-disagg-planner" -n "$NAMESPACE" --timeout=900s
    log_success "Frontend pod is ready"

    log_info "Waiting for planner pod..."
    kubectl wait --for=condition=Ready pod -l "nvidia.com/dynamo-component-type=planner,nvidia.com/dynamo-namespace=${NAMESPACE}-vllm-disagg-planner" -n "$NAMESPACE" --timeout=900s
    sleep 30
}

setup_port_forward() {
    log_info "Setting up port forwarding..."
    LOCAL_PORT=$(find_free_local_port)
    log_info "Using local port $LOCAL_PORT for frontend port-forward"

    local frontend_service="vllm-disagg-planner-frontend"
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
    kubectl wait --for=delete dynamographdeployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=120s || true
}

cleanup() {
    cleanup_port_forward
    if [ "$DEPLOYED_BY_US" = true ]; then
        cleanup_deployment
    fi
}

run_test() {
    log_info "Running scaling test (graduated 8->18 req/s, mode=$MODE)..."

    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi

    local test_args="--namespace $NAMESPACE --mode $MODE --base-url http://localhost:$LOCAL_PORT"
    if [ "$SAVE_RESULTS" = true ]; then
        test_args="$test_args --save-results"
        log_info "Results will be saved to components/src/dynamo/planner/tests/e2e_scaling_results"
    fi

    $python_cmd "$TEST_FILE" $test_args
}

main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                if [[ "$MODE" != "throughput" && "$MODE" != "load" ]]; then
                    log_error "Invalid mode: $MODE (must be 'throughput' or 'load')"
                    exit 1
                fi
                shift 2
                ;;
            --save-results)
                SAVE_RESULTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--namespace NS] [--mode MODE] [--save-results]"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$MODE" = "load" ]; then
        YAML_FILE="$SCRIPT_DIR/disagg_planner_load.yaml"
    else
        YAML_FILE="$SCRIPT_DIR/disagg_planner_throughput.yaml"
    fi

    check_prerequisites
    trap cleanup EXIT

    if ! check_existing_deployment; then
        deploy_planner
        DEPLOYED_BY_US=true
    fi

    setup_port_forward
    run_test
}

main "$@"
