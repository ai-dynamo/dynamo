#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to launch vLLM with the DynamoConnector for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Cleanup: Kill any previous vLLM processes
# =============================================================================
echo "🧹 Cleaning up previous processes..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 2

# Check if port 8000 is in use
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is still in use, killing processes..."
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 1
fi

# =============================================================================
# Environment Setup
# =============================================================================

# Find virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
elif [ -d "$SCRIPT_DIR/.venv" ] && [ ! -L "$SCRIPT_DIR/.venv" ]; then
    VENV_PATH="$SCRIPT_DIR/.venv"
else
    echo "❌ No virtual environment found!"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Ensure our kvbm package is in the path
export PYTHONPATH="$REPO_ROOT/lib/bindings/kvbm/python:$PYTHONPATH"
export VLLM_SERVER_DEV_MODE=1

# =============================================================================
# Rust Logging Configuration
# =============================================================================
# Set RUST_LOG if not already set - enables tracing from Rust components
# export RUST_LOG="${RUST_LOG:-dynamo_kvbm=debug,dynamo_nova=info,warn}"
export RUST_LOG="${RUST_LOG:-debug}"
export DYN_LOG="${DYN_LOG:-debug}"

# =============================================================================
# vLLM Configuration
# =============================================================================

# Default model - use gpt2 for quick testing
MODEL="${MODEL:-gpt2}"

# Configure KV transfer for DynamoConnector
# The connector is at: kvbm.v2.vllm.schedulers.connector.DynamoConnector
kv_transfer_config='{
  "kv_connector": "DynamoConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
  "kv_connector_extra_config": {
    "leader": {
      "cache": { "host": { "cache_size_gb": 4.0 } },
      "tokio": { "worker_threads": 2 }
    },
    "worker": {
      "nixl": { "backends": { "UCX": {}, "POSIX": {} } },
      "tokio": { "worker_threads": 1 }
    }
  }
}'

# =============================================================================
# Display Configuration
# =============================================================================
echo ""
echo "🚀 Launching vLLM with DynamoConnector"
echo "=============================================================================="
echo ""
echo "Environment:"
echo "  SCRIPT_DIR:     $SCRIPT_DIR"
echo "  REPO_ROOT:      $REPO_ROOT"
echo "  VENV_PATH:      $VENV_PATH"
echo "  Python:         $(which python)"
echo "  vLLM version:   $(python -c 'import vllm; print(vllm.__version__)')"
echo "  kvbm version:   $(python -c 'import kvbm; print(kvbm.__version__)' 2>/dev/null || echo 'N/A')"
echo ""
echo "PYTHONPATH:"
echo "  $PYTHONPATH" | tr ':' '\n' | head -3
echo ""
echo "Rust Logging:"
echo "  RUST_LOG:       $RUST_LOG"
echo ""
echo "vLLM Configuration:"
echo "  Model:          $MODEL"
echo "  Host:           127.0.0.1"
echo "  Port:           8000"
echo "  Enforce Eager:  true"
echo ""
echo "KV Transfer Config:"
echo "$kv_transfer_config" | python -m json.tool 2>/dev/null || echo "$kv_transfer_config"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""
echo "=============================================================================="
echo ""

# =============================================================================
# Launch vLLM
# =============================================================================
# Using --enforce-eager to avoid CUDA graph compilation overhead during testing
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --kv-transfer-config "$kv_transfer_config" \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
