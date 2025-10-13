#!/bin/bash

# Pre-deployment check script for Dynamo
# This script verifies that the Kubernetes cluster has the necessary prerequisites
# before deploying Dynamo components.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
        print_status $YELLOW "Please configure a default StorageClass before proceeding with deployment.\n"

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
    print_section "Checking cluster gpu resources"

    local node_count
    node_count=$(kubectl get nodes -l nvidia.com/gpu.present=true 2>/dev/null | wc -l || echo "0")

    if [[ $node_count -eq 0 ]]; then
        print_status $RED "❌ No nodes found in the cluster"
        return 1
    else
        print_status $GREEN "✅ Found ${node_count} gpu node(s) in the cluster"
    fi

    # Show basic node information
    # print_status $BLUE "Node information:"
    # kubectl get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,STATUS:.status.conditions[-1].type,ROLES:.metadata.labels.'node-role\.kubernetes\.io/.*',VERSION:.status.nodeInfo.kubeletVersion 2>/dev/null || true
}

# Global variables to track check results
declare -A CHECK_RESULTS
declare -a CHECK_ORDER

# Function to record check result
record_check_result() {
    local check_name="$1"
    local status="$2"
    CHECK_RESULTS["$check_name"]="$status"
    CHECK_ORDER+=("$check_name")
}

# Function to display check summary
display_check_summary() {
    print_section "Pre-Deployment Check Summary"

    local passed=0
    local failed=0

    for check_name in "${CHECK_ORDER[@]}"; do
        local status="${CHECK_RESULTS[$check_name]}"
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
        record_check_result "Cluster Resources" "PASS"
    else
        record_check_result "Cluster Resources" "FAIL"
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
