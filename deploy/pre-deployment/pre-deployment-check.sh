#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pre-deployment check script for Dynamo
# This script verifies that the Kubernetes cluster has the necessary prerequisites
# before deploying Dynamo platform.
#
# Checks performed:
# 1. kubectl connectivity - Verifies kubectl is installed and can connect to cluster
# 2. Default StorageClass - Ensures a default StorageClass is configured
# 3. Cluster GPU Resources - Validates GPU nodes are available
# 4. GPU Operator - Confirms GPU operator is installed and running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global state flags
MANAGED_GPU_DETECTED=false

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  Dynamo Pre-Deployment Check Script  ${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_section() {
    echo -e "\n${BLUE}--- $1 ---${NC}"
}

# Function to check if kubectl is available and cluster is accessible
check_kubectl() {
    print_section "Checking kubectl connectivity"

    if ! command -v kubectl &> /dev/null; then
        print_status $RED "❌ kubectl is not installed or not in PATH"
        print_status $YELLOW "Please install kubectl and ensure it's in your PATH"
        return 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        print_status $RED "❌ Cannot connect to Kubernetes cluster"
        print_status $YELLOW "Please ensure kubectl is configured to connect to your cluster"
        return 1
    fi

    print_status $GREEN "✅ kubectl is available and cluster is accessible"
    return 0
}

# Function to check for default storage class
check_default_storage_class() {
    print_section "Checking for default StorageClass"

    # Use JSONPath to find storage classes with the default annotation set to "true"
    local default_storage_classes
    default_storage_classes=$(kubectl get storageclass -o jsonpath='{range .items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")]}{.metadata.name}{"\n"}{end}' 2>/dev/null || echo "")

    if [[ -z "$default_storage_classes" ]]; then
        print_status $RED "❌ No default StorageClass found"
        print_status $YELLOW "\nDynamo requires a default StorageClass for persistent volume provisioning."
        print_status $BLUE "Please follow the instructions below to configure a default StorageClass before proceeding with deployment.\n"

        # Show available storage classes
        print_status $BLUE "Available StorageClasses in your cluster:"
        local all_storage_classes
        all_storage_classes=$(kubectl get storageclass 2>/dev/null || echo "")

        if [[ -z "$all_storage_classes" ]]; then
            print_status $YELLOW "  No StorageClasses found in the cluster"
        else
            echo -e "$all_storage_classes"

            local all_storage_class_names
            all_storage_class_names=$(kubectl get storageclass -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || echo "")

            print_status $BLUE "\nTo set a StorageClass as default, use the following command:"
            print_status $YELLOW "kubectl patch storageclass <storage-class-name> -p '{\"metadata\": {\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"

            if [[ -n "$all_storage_class_names" ]]; then
                local first_sc_name
                first_sc_name=$(echo "$all_storage_class_names" | head -n1)
                print_status $BLUE "\nExample with your first available StorageClass:"
                print_status $YELLOW "kubectl patch storageclass ${first_sc_name} -p '{\"metadata\": {\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"
            fi
        fi

        print_status $BLUE "\nFor more information on managing default StorageClasses, visit:"
        print_status $BLUE "https://kubernetes.io/docs/tasks/administer-cluster/change-default-storage-class/"
        return 1
    else
        print_status $GREEN "✅ Default StorageClass found"
        while IFS= read -r sc_name; do
            if [[ -n "$sc_name" ]]; then
                local provisioner
                default_sc=$(kubectl get storageclass "$sc_name" 2>/dev/null || echo "unknown")
                print_status $GREEN "  - ${default_sc}"
            fi
        done <<< "$default_storage_classes"

        # Check if there are multiple default storage classes (which can cause issues)
        local default_count
        default_count=$(echo "$default_storage_classes" | grep -c . || echo "0")
        if [[ $default_count -gt 1 ]]; then
            print_status $YELLOW "⚠️  Warning: Multiple default StorageClasses detected"
            print_status $YELLOW "   This may cause unpredictable behavior. Consider having only one default StorageClass."
        fi
        return 0
    fi
}

check_cluster_resources() {
    print_section "Checking cluster GPU resources"

    # Primary detection: GPU Feature Discovery label
    local labeled_gpu_nodes
    labeled_gpu_nodes=$(kubectl get nodes \
        -l nvidia.com/gpu.present=true \
        -o name 2>/dev/null || true)

    local labeled_count
    labeled_count=$(echo "$labeled_gpu_nodes" | grep -c . || true)

    if [[ $labeled_count -gt 0 ]]; then
        print_status $GREEN "✅ Found ${labeled_count} GPU node(s) via GFD labels"
        return 0
    fi

    # Fallback detection: allocatable GPU resources
    local allocatable_gpu_nodes
    allocatable_gpu_nodes=$(kubectl get nodes \
        -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' \
        2>/dev/null || true)

    local allocatable_count
    allocatable_count=$(echo "$allocatable_gpu_nodes" | awk '$2 != "" && $2 != "0"' | wc -l | tr -d ' ')

    if [[ $allocatable_count -gt 0 ]]; then
        MANAGED_GPU_DETECTED=true

        print_status $GREEN "✅ Found ${allocatable_count} GPU node(s) via allocatable GPU resources"
        print_status $BLUE "Managed GPU environment detected (GPU Operator labels not present)"

        echo "$allocatable_gpu_nodes" | awk '$2 != "" && $2 != "0"' | while read -r node gpu_count; do
            print_status $GREEN "  - ${node}: ${gpu_count} GPU(s)"
        done

        return 0
    fi

    print_status $RED "❌ No GPU nodes found in the cluster"
    print_status $YELLOW "No nodes detected with either:"
    print_status $YELLOW "  - nvidia.com/gpu.present=true label"
    print_status $YELLOW "  - allocatable nvidia.com/gpu resources"

    print_status $BLUE "Please ensure your cluster has GPU-enabled nodes configured."

    return 1
}

check_gpu_operator() {
    print_section "Checking GPU operator"

    # Managed GPU environments may not use GPU Operator
    if [[ "$MANAGED_GPU_DETECTED" == "true" ]]; then
        print_status $YELLOW "⚠️  GPU Operator not required in managed GPU environments"
        print_status $GREEN "✅ Managed GPU support detected via allocatable GPU resources"
        return 0
    fi

    # Check for GPU operator pods
    local gpu_operator_pods
    gpu_operator_pods=$(kubectl get pods -A \
        -l app.kubernetes.io/name=gpu-operator \
        --no-headers 2>/dev/null || true)

    # Fallback to legacy label
    if [[ -z "$gpu_operator_pods" ]]; then
        gpu_operator_pods=$(kubectl get pods -A \
            -l app=gpu-operator \
            --no-headers 2>/dev/null || true)
    fi

    if [[ -z "$gpu_operator_pods" ]]; then
        print_status $RED "❌ GPU operator not found in the cluster"
        print_status $YELLOW "Dynamo requires either:"
        print_status $YELLOW "  - NVIDIA GPU Operator"
        print_status $YELLOW "  - Managed GPU support with allocatable GPUs"

        return 1
    fi

    local running_pods
    running_pods=$(echo "$gpu_operator_pods" | awk '$4 == "Running"' | wc -l | tr -d ' ')

    local total_pods
    total_pods=$(echo "$gpu_operator_pods" | wc -l | tr -d ' ')

    if [[ $running_pods -eq 0 ]]; then
        print_status $RED "❌ GPU operator pods are not running"
        echo "$gpu_operator_pods"
        return 1

    elif [[ $running_pods -lt $total_pods ]]; then
        print_status $YELLOW "⚠️  GPU operator partially running: $running_pods/$total_pods pods running"
        echo "$gpu_operator_pods"
        print_status $GREEN "✅ GPU operator is available (with warnings)"
        return 0

    else
        print_status $GREEN "✅ GPU operator is running ($running_pods/$total_pods pods)"
        return 0
    fi
}

# Global variables to track check results (using simple arrays for compatibility)
CHECK_RESULTS=""
CHECK_ORDER=""

# Function to record check result
record_check_result() {
    local check_name="$1"
    local status="$2"

    # Append to results string with delimiter
    if [[ -z "$CHECK_RESULTS" ]]; then
        CHECK_RESULTS="${check_name}:${status}"
        CHECK_ORDER="${check_name}"
    else
        CHECK_RESULTS="${CHECK_RESULTS}|${check_name}:${status}"
        CHECK_ORDER="${CHECK_ORDER}|${check_name}"
    fi
}

# Function to get check result by name
get_check_result() {
    local check_name="$1"
    echo "$CHECK_RESULTS" | tr '|' '\n' | grep "^${check_name}:" | cut -d':' -f2
}

# Function to display check summary
display_check_summary() {
    print_section "Pre-Deployment Check Summary"

    local passed=0
    local failed=0

    # Split CHECK_ORDER by delimiter and iterate
    IFS='|' read -ra CHECKS <<< "$CHECK_ORDER"
    for check_name in "${CHECKS[@]}"; do
        local status=$(get_check_result "$check_name")
        if [[ "$status" == "PASS" ]]; then
            print_status $GREEN "✅ $check_name: PASSED"
            ((passed++))
        else
            print_status $RED "❌ $check_name: FAILED"
            ((failed++))
        fi
    done

    echo ""
    print_status $BLUE "Summary: $passed passed, $failed failed"

    if [[ $failed -eq 0 ]]; then
        print_status $GREEN "🎉 All pre-deployment checks passed!"
        print_status $GREEN "Your cluster is ready for Dynamo deployment."
        return 0
    else
        print_status $RED "❌ $failed pre-deployment check(s) failed."
        print_status $RED "Please address the issues above before proceeding with deployment."
        return 1
    fi
}

# Main execution
main() {
    print_header

    local overall_exit_code=0

    # Run checks and capture results
    if check_kubectl; then
        record_check_result "kubectl Connectivity" "PASS"
    else
        record_check_result "kubectl Connectivity" "FAIL"
        overall_exit_code=1
    fi

    if check_default_storage_class; then
        record_check_result "Default StorageClass" "PASS"
    else
        record_check_result "Default StorageClass" "FAIL"
        overall_exit_code=1
    fi

    if check_cluster_resources; then
        record_check_result "Cluster GPU Resources" "PASS"
    else
        record_check_result "Cluster GPU Resources" "FAIL"
        overall_exit_code=1
    fi

    if check_gpu_operator; then
        record_check_result "GPU Operator" "PASS"
    else
        record_check_result "GPU Operator" "FAIL"
        overall_exit_code=1
    fi

    # Display summary
    echo ""
    if ! display_check_summary; then
        overall_exit_code=1
    fi

    exit $overall_exit_code
}

# Run the script
main "$@"
