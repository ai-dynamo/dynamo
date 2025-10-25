#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

RECIPES_DIR="$( cd "$( dirname "$0" )" && pwd )"
# Default values
NAMESPACE="${NAMESPACE:-dynamo}"
DEPLOYMENT=""
MODEL=""
FRAMEWORK=""
DRY_RUN=""

# Frameworks - following container/build.sh pattern
declare -A FRAMEWORKS=(["VLLM"]=1 ["TRTLLM"]=2 ["SGLANG"]=3)
DEFAULT_FRAMEWORK=VLLM

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] --model <model> --framework <framework> --deployment <deployment-type>"
    echo ""
    echo "Required Options:"
    echo "  --model <model>       Model name (e.g., llama-3-70b)"
    echo "  --framework <fw>      Framework one of ${!FRAMEWORKS[*]} (default: ${DEFAULT_FRAMEWORK})"
    echo "  --deployment <type>   Deployment type (e.g., agg, disagg etc, please refer to the README.md for available deployment types)"
    echo ""
    echo "Optional:"
    echo "  --namespace <ns>      Kubernetes namespace (default: dynamo)"
    echo "  --dry-run             Print commands without executing them"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE             Kubernetes namespace (default: dynamo)"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama-3-70b --framework vllm --deployment agg"
    echo "  $0 --model llama-3-70b --framework trtllm --deployment disagg-single-node"
    echo "  $0 --namespace my-ns --model llama-3-70b --framework vllm --deployment disagg-multi-node"
    exit 1
}

missing_requirement() {
    echo "ERROR: $1 requires an argument."
    usage
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="echo"
            shift
            ;;
        --model)
            if [ "$2" ]; then
                MODEL=$2
                shift 2
            else
                missing_requirement "$1"
            fi
            ;;
        --framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift 2
            else
                missing_requirement "$1"
            fi
            ;;
        --deployment)
            if [ "$2" ]; then
                DEPLOYMENT=$2
                shift 2
            else
                missing_requirement "$1"
            fi
            ;;
        --namespace)
            if [ "$2" ]; then
                NAMESPACE=$2
                shift 2
            else
                missing_requirement "$1"
            fi
            ;;
        -h|--help)
            usage
            ;;
        -*)
            error 'ERROR: Unknown option: ' "$1"
            ;;
        *)
            error "ERROR: Unknown argument: " "$1"
            ;;
    esac
done

if [ -z "$FRAMEWORK" ]; then
    FRAMEWORK=$DEFAULT_FRAMEWORK
fi

if [ -n "$FRAMEWORK" ]; then
    FRAMEWORK=${FRAMEWORK^^}
    if [[ -z "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
        error 'ERROR: Unknown framework: ' "$FRAMEWORK"
    fi
fi

# Validate required arguments
if [[ -z "$MODEL" ]] || [[ -z "$DEPLOYMENT" ]]; then
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: --model argument is required"
    fi
    if [[ -z "$DEPLOYMENT" ]]; then
        echo "ERROR: --deployment argument is required"
    fi
    echo ""
    usage
fi

# Construct paths based on new structure: recipes/<model>/<framework>/<deployment-type>/
MODEL_DIR="$RECIPES_DIR/$MODEL"
FRAMEWORK_DIR="$MODEL_DIR/${FRAMEWORK,,}"
DEPLOY_PATH="$FRAMEWORK_DIR/$DEPLOYMENT"

# Check if model directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: Model directory '$MODEL' does not exist in $RECIPES_DIR"
    echo "Available models:"
    ls -1 "$RECIPES_DIR" | grep -v "\.sh$\|\.md$\|model-cache$" | sed 's/^/  /'
    exit 1
fi

# Check if framework directory exists
if [[ ! -d "$FRAMEWORK_DIR" ]]; then
    echo "Error: Framework directory '${FRAMEWORK,,}' does not exist in $MODEL_DIR"
    echo "Available frameworks for $MODEL:"
    ls -1 "$MODEL_DIR" | grep -v "\.sh$\|\.md$" | sed 's/^/  /'
    exit 1
fi

# Check if deployment directory exists
if [[ ! -d "$DEPLOY_PATH" ]]; then
    echo "Error: Deployment type '$DEPLOYMENT' does not exist in $FRAMEWORK_DIR"
    echo "Available deployment types for $MODEL/${FRAMEWORK,,}:"
    ls -1 "$FRAMEWORK_DIR" | grep -v "\.sh$\|\.md$" | sed 's/^/  /'
    exit 1
fi

# Check if deployment files exist
DEPLOY_FILE="$DEPLOY_PATH/deploy.yaml"
PERF_FILE="$DEPLOY_PATH/perf.yaml"

if [[ ! -f "$DEPLOY_FILE" ]]; then
    echo "Error: Deployment file '$DEPLOY_FILE' not found"
    exit 1
fi

# Check if perf file exists (optional)
PERF_AVAILABLE=false
if [[ -f "$PERF_FILE" ]]; then
    PERF_AVAILABLE=true
    echo "Performance benchmark file found: $PERF_FILE"
else
    echo "Performance benchmark file not found: $PERF_FILE (skipping benchmarks)"
fi

# Show deployment information
echo "======================================"
echo "Dynamo Recipe Deployment"
echo "======================================"
echo "Model: $MODEL"
echo "Framework: ${FRAMEWORK,,}"
echo "Deployment Type: $DEPLOYMENT"
echo "Namespace: $NAMESPACE"
echo "======================================"

# Handle model downloading
MODEL_CACHE_DIR="$MODEL_DIR/model-cache"
echo "Creating PVC for model cache and downloading model..."
$DRY_RUN kubectl apply -n $NAMESPACE -f $MODEL_CACHE_DIR/model-cache.yaml
$DRY_RUN kubectl apply -n $NAMESPACE -f $MODEL_CACHE_DIR/model-download.yaml

# Wait for the model download to complete
MODEL_DOWNLOAD_JOB_NAME=$(grep "name:" $MODEL_CACHE_DIR/model-download.yaml | head -1 | awk '{print $2}')
echo "Waiting for job '$MODEL_DOWNLOAD_JOB_NAME' to complete..."
$DRY_RUN kubectl wait --for=condition=Complete job/$MODEL_DOWNLOAD_JOB_NAME -n $NAMESPACE --timeout=6000s

# Deploy the specified configuration
echo "Deploying $MODEL ${FRAMEWORK,,} $DEPLOYMENT configuration..."
$DRY_RUN kubectl apply -n $NAMESPACE -f $DEPLOY_FILE

# Launch the benchmark job (if available)
if [[ "$PERF_AVAILABLE" == "true" ]]; then
    echo "Launching benchmark job..."
    $DRY_RUN kubectl apply -n $NAMESPACE -f $PERF_FILE

    # Construct job name from the perf file
    JOB_NAME=$(grep "name:" $PERF_FILE | head -1 | awk '{print $2}')
    echo "Waiting for job '$JOB_NAME' to complete..."
    $DRY_RUN kubectl wait --for=condition=Complete job/$JOB_NAME -n $NAMESPACE --timeout=6000s

    # Print logs from the benchmark job
    echo "======================================"
    echo "Benchmark completed. Logs:"
    echo "======================================"
    $DRY_RUN kubectl logs job/$JOB_NAME -n $NAMESPACE
else
    echo "======================================"
    echo "Deployment completed successfully!"
    echo "No performance benchmark available for this configuration."
    echo "======================================"
fi