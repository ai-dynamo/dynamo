#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail


NIXL_VERSION="0.6.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker daemon status
check_docker_daemon() {
    if ! docker info >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Function to check all required dependencies
check_dependencies() {
    echo "Checking required dependencies..."
    local missing_deps=()
    local warnings=()

    # Check wget
    if ! command_exists wget; then
        missing_deps+=("wget")
    else
        echo "✅ wget is available"
    fi

    # Check unzip
    if ! command_exists unzip; then
        missing_deps+=("unzip")
    else
        echo "✅ unzip is available"
    fi

    # Check kubectl
    if ! command_exists kubectl; then
        missing_deps+=("kubectl")
    else
        echo "✅ kubectl is available"
        # Test kubectl connectivity
        if ! kubectl cluster-info >/dev/null 2>&1; then
            warnings+=("kubectl is installed but cannot connect to cluster")
        else
            echo "✅ kubectl can connect to cluster"
        fi
    fi

    # Check Docker
    if ! command_exists docker; then
        missing_deps+=("docker")
    else
        echo "✅ docker is available"
        # Check Docker daemon
        if ! check_docker_daemon; then
            warnings+=("Docker is installed but daemon is not running or accessible")
        else
            echo "✅ Docker daemon is running"

            # Additional Docker toolchain checks
            if ! docker ps >/dev/null 2>&1; then
                warnings+=("Docker requires sudo or user is not in docker group - consider adding user to docker group")
            fi

            if ! docker buildx version >/dev/null 2>&1; then
                warnings+=("Docker buildx not available (may affect multi-architecture builds)")
            fi
        fi
    fi

    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo
        echo "❌ Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo
        echo "Please install the missing dependencies and try again."
        echo
        echo "Installation suggestions:"
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                wget)
                    echo "  wget: sudo apt-get install wget (Ubuntu/Debian) or yum install wget (RHEL/CentOS)"
                    ;;
                unzip)
                    echo "  unzip: sudo apt-get install unzip (Ubuntu/Debian) or yum install unzip (RHEL/CentOS)"
                    ;;
                kubectl)
                    echo "  kubectl: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
                    ;;
                docker)
                    echo "  docker: https://docs.docker.com/get-docker/"
                    ;;
            esac
        done
        return 1
    fi

    # Report warnings
    if [ ${#warnings[@]} -gt 0 ]; then
        echo
        echo "⚠️  Warnings:"
        for warning in "${warnings[@]}"; do
            echo "  - $warning"
        done
        echo
        printf "Do you want to continue despite these warnings? (y/N): "
        read continue_with_warnings
        case "$continue_with_warnings" in
            [Yy]|[Yy][Ee][Ss])
                echo "Continuing with warnings..."
                ;;
            *)
                echo "Please resolve the warnings and try again."
                return 1
                ;;
        esac
    fi

    echo "✅ All required dependencies are available"
    return 0
}

# Function to display available architectures
show_architectures() {
    echo "Available architectures:"
    echo "1) x86_64 (Intel/AMD 64-bit)"
    echo "2) aarch64 (ARM64)"
}

# Function to validate architecture input
validate_architecture() {
    local arch=$1
    case $arch in
        1|x86_64)
            echo "x86_64"
            return 0
            ;;
        2|aarch64)
            echo "aarch64"
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to prompt for registry
prompt_for_registry() {
    echo
    printf "Enter your Docker registry (e.g., my-registry, docker.io/username): "
    read REGISTRY
    if [ -z "$REGISTRY" ]; then
        echo "Error: Registry cannot be empty"
        exit 1
    fi
}

# Function to build nixlbench image
build_nixlbench() {
    local arch=$1
    local registry=$2

    echo "Building nixlbench image for architecture: $arch"
    echo "Registry: $registry"

    NIXL_BUILD_DIR="/tmp/nixlbench-${NIXL_VERSION}"
    rm -rf "${NIXL_BUILD_DIR}"
    mkdir -p "${NIXL_BUILD_DIR}"
    cd "${NIXL_BUILD_DIR}"

    echo "Downloading NIXL source..."
    wget https://github.com/ai-dynamo/nixl/archive/refs/tags/${NIXL_VERSION}.zip
    unzip "${NIXL_VERSION}.zip"
    cd "nixl-${NIXL_VERSION}/benchmark/nixlbench/contrib"
    read -p "Press Enter to continue"
    echo "Building Docker image..."
    ./build.sh --tag "${registry}/nixlbench:${NIXL_VERSION}-${arch}" --arch "${arch}"

    echo "Build completed successfully!"
    echo "Image: ${registry}/nixlbench:${NIXL_VERSION}-${arch}"
}

# Function to update deployment yaml
update_deployment() {
    local arch=$1
    local registry=$2
    local deployment_file="${SCRIPT_DIR}/nixlbench-deployment-${arch}.yaml"

    echo "Creating deployment file: $deployment_file"

    # Copy the original deployment file and update the image
    cp "${SCRIPT_DIR}/nixlbench-deployment.yaml" "$deployment_file"

    # Update the image field using sed
    sed -i "s|my-registry/nixlbench:version-arch|${registry}/nixlbench:${NIXL_VERSION}-${arch}|g" "$deployment_file"

    echo "Deployment file updated with image: ${registry}/nixlbench:${NIXL_VERSION}-${arch}"
}

# Function to prompt for steps to execute
prompt_for_steps() {
    echo
    echo "Select which steps to execute:"
    echo "1) Build nixlbench Docker image"
    echo "2) Update deployment YAML file"
    echo "3) Deploy to Kubernetes"
    echo
    echo "Enter the steps you want to execute (e.g., '1,2,3' for all, '1,2' to skip deployment, '3' for deployment only):"
    printf "Steps to execute: "
    read steps_input

    if [ -z "$steps_input" ]; then
        echo "Error: Please select at least one step"
        return 1
    fi

    # Parse the input and set flags
    EXECUTE_BUILD=false
    EXECUTE_UPDATE=false
    EXECUTE_DEPLOY=false

    # Convert comma-separated input to array
    IFS=',' read -ra STEPS <<< "$steps_input"
    for step in "${STEPS[@]}"; do
        # Remove whitespace
        step=$(echo "$step" | tr -d ' ')
        case "$step" in
            1)
                EXECUTE_BUILD=true
                ;;
            2)
                EXECUTE_UPDATE=true
                ;;
            3)
                EXECUTE_DEPLOY=true
                ;;
            *)
                echo "Warning: Invalid step '$step' ignored. Valid steps are 1, 2, 3"
                ;;
        esac
    done

    # Check if at least one valid step was selected
    if [ "$EXECUTE_BUILD" = false ] && [ "$EXECUTE_UPDATE" = false ] && [ "$EXECUTE_DEPLOY" = false ]; then
        echo "Error: No valid steps selected"
        return 1
    fi

    return 0
}

# Function to deploy to Kubernetes
deploy_to_k8s() {
    local arch=$1
    local deployment_file="${SCRIPT_DIR}/nixlbench-deployment-${arch}.yaml"

    echo "Deploying to Kubernetes..."
    kubectl apply -f "$deployment_file"
    echo "Deployment applied successfully!"
    echo
    echo "To check the status of your deployment:"
    echo "kubectl get pods -l app=nixl-benchmark"
    echo
    echo "To view logs:"
    echo "kubectl logs -l app=nixl-benchmark -f"
}

# Main script
main() {
    echo "NIXL Benchmark Build and Deploy Script"
    echo "======================================"
    echo

    # Check dependencies first
    if ! check_dependencies; then
        exit 1
    fi
    echo

    # Show available architectures
    show_architectures
    echo

    # Prompt for architecture
    while true; do
        printf "Select architecture (1-2 or enter x86_64/aarch64): "
        read arch_input

        if [ -z "$arch_input" ]; then
            echo "Error: Please select an architecture"
            continue
        fi

        SELECTED_ARCH=$(validate_architecture "$arch_input")
        if [ $? -eq 0 ]; then
            break
        else
            echo "Error: Invalid architecture. Please select 1, 2, x86_64, or aarch64"
        fi
    done

    echo "Selected architecture: $SELECTED_ARCH"

    # Prompt for registry (only if building or updating deployment)
    REGISTRY=""

    # Prompt for steps to execute
    while true; do
        if prompt_for_steps; then
            break
        fi
        echo "Please try again."
        echo
    done

    # Only prompt for registry if we need it
    if [ "$EXECUTE_BUILD" = true ] || [ "$EXECUTE_UPDATE" = true ]; then
        prompt_for_registry
    fi

    echo
    echo "Summary:"
    echo "- Architecture: $SELECTED_ARCH"
    if [ -n "$REGISTRY" ]; then
        echo "- Registry: $REGISTRY"
        echo "- Image will be: $REGISTRY/nixlbench:$NIXL_VERSION-$SELECTED_ARCH"
    fi
    echo "- Steps to execute:"
    if [ "$EXECUTE_BUILD" = true ]; then
        echo "  ✓ Build nixlbench Docker image"
    else
        echo "  ✗ Build nixlbench Docker image (skipped)"
    fi
    if [ "$EXECUTE_UPDATE" = true ]; then
        echo "  ✓ Update deployment YAML file"
    else
        echo "  ✗ Update deployment YAML file (skipped)"
    fi
    if [ "$EXECUTE_DEPLOY" = true ]; then
        echo "  ✓ Deploy to Kubernetes"
    else
        echo "  ✗ Deploy to Kubernetes (skipped)"
    fi
    echo

    printf "Proceed with selected steps? (y/N): "
    read confirm
    case "$confirm" in
        [Yy]|[Yy][Ee][Ss])
            ;;
        *)
            echo "Process cancelled."
            exit 0
            ;;
    esac

    # Execute selected steps
    if [ "$EXECUTE_BUILD" = true ]; then
        echo
        echo "=== Building nixlbench Docker image ==="
        build_nixlbench "$SELECTED_ARCH" "$REGISTRY"
    fi

    if [ "$EXECUTE_UPDATE" = true ]; then
        echo
        echo "=== Updating deployment YAML file ==="
        update_deployment "$SELECTED_ARCH" "$REGISTRY"
    fi

    if [ "$EXECUTE_DEPLOY" = true ]; then
        echo
        echo "=== Deploying to Kubernetes ==="
        # Check if deployment file exists
        deployment_file="${SCRIPT_DIR}/nixlbench-deployment-${SELECTED_ARCH}.yaml"
        if [ ! -f "$deployment_file" ]; then
            echo "Warning: Deployment file not found at $deployment_file"
            echo "You may need to run step 2 (Update deployment YAML file) first."
            printf "Do you want to continue with deployment anyway? (y/N): "
            read deploy_confirm
            case "$deploy_confirm" in
                [Yy]|[Yy][Ee][Ss])
                    ;;
                *)
                    echo "Deployment skipped."
                    EXECUTE_DEPLOY=false
                    ;;
            esac
        fi

        if [ "$EXECUTE_DEPLOY" = true ]; then
            deploy_to_k8s "$SELECTED_ARCH"
        fi
    fi

    echo
    echo "Process completed successfully!"
}

# Run main function
main "$@"
