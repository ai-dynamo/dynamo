#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for Triton Server backend with Dynamo
# This runs the frontend and triton worker on the same node

set -e

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $FRONTEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_NAME="identity"
MODEL_REPO="${TRITON_DIR}/model_repo"
BACKEND_DIR="${TRITON_DIR}/backends"
LOG_VERBOSE=1
STORE_KV="${DYN_STORE_KV:-file}"  # Default to file-based KV (no etcd required)

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-repository)
            MODEL_REPO="$2"
            shift 2
            ;;
        --backend-directory)
            BACKEND_DIR="$2"
            shift 2
            ;;
        --log-verbose)
            LOG_VERBOSE="$2"
            shift 2
            ;;
        --store-kv)
            STORE_KV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Launch Triton Server backend with Dynamo frontend"
            echo ""
            echo "Options:"
            echo "  --model-name <name>         Model name to load (default: $MODEL_NAME)"
            echo "  --model-repository <path>   Path to model repository (default: $MODEL_REPO)"
            echo "  --backend-directory <path>  Path to Triton backends (default: $BACKEND_DIR)"
            echo "  --log-verbose <level>       Triton log verbosity 0-6 (default: $LOG_VERBOSE)"
            echo "  --store-kv <backend>        KV store backend: file, etcd, mem (default: $STORE_KV)"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DYN_STORE_KV     KV store backend (default: file)"
            echo "  DYN_HTTP_PORT    Frontend HTTP port (default: 8000)"
            echo "  DYN_SYSTEM_PORT  Worker metrics port (default: 8081)"
            echo ""
            echo "Ports:"
            echo "  HTTP:  8000 (configurable via DYN_HTTP_PORT)"
            echo "  gRPC:  8787 (KServe gRPC for tensor models)"
            echo ""
            echo "Additional arguments will be passed to tritonworker.py"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate paths
if [[ ! -d "$MODEL_REPO" ]]; then
    echo "Error: Model repository not found: $MODEL_REPO"
    exit 1
fi

if [[ ! -d "$BACKEND_DIR" ]]; then
    echo "Error: Backend directory not found: $BACKEND_DIR"
    exit 1
fi

echo "=== Triton Server with Dynamo ==="
echo "Model name:       $MODEL_NAME"
echo "Model repository: $MODEL_REPO"
echo "Backend directory: $BACKEND_DIR"
echo "Log verbose:      $LOG_VERBOSE"
echo "KV store:         $STORE_KV"
echo ""

# Set library path for Triton
export LD_LIBRARY_PATH="${TRITON_DIR}/lib:${BACKEND_DIR}:${LD_LIBRARY_PATH:-}"

# Export KV store setting for worker (read by @dynamo_worker decorator)
export DYN_STORE_KV="$STORE_KV"

# Run frontend in background
# --kserve-grpc-server enables the KServe gRPC endpoint for tensor models
echo "Starting Dynamo frontend..."
python3 -m dynamo.frontend --kserve-grpc-server --store-kv "$STORE_KV" &
FRONTEND_PID=$!

# Give frontend time to start
sleep 2

# Run triton worker in foreground
echo "Starting Triton worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 "${TRITON_DIR}/src/tritonworker.py" \
    --model-name "$MODEL_NAME" \
    --model-repository "$MODEL_REPO" \
    --backend-directory "$BACKEND_DIR" \
    --log-verbose "$LOG_VERBOSE" \
    "${EXTRA_ARGS[@]}"

