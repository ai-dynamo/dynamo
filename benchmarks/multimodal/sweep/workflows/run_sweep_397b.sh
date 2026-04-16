#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep wrapper: vllm-serve vs dynamo+EC at request rates 1, 2, 4.
# Builds dynamo in background while vllm-serve runs (no dynamo needed),
# then dynamo-ec configs use the freshly built dynamo.
#
# Usage: bash benchmarks/multimodal/sweep/workflows/run_sweep_397b.sh

set -e
source /opt/dynamo/venv/bin/activate

SWEEP_CONFIG=benchmarks/multimodal/sweep/experiments/embedding_cache/sweep_397b_rates.yaml

# Install aiperf from mounted repo
pip install --no-deps -e /aiperf

# Build dynamo in background — finishes during first vllm-serve model load.
# vllm-serve configs don't need dynamo; dynamo-ec configs run after.
(
  pip uninstall -y ai-dynamo ai-dynamo-runtime 2>/dev/null || true
  cd /workspace/lib/bindings/python && maturin develop --uv --release
  cd /workspace && uv pip install --no-deps -e .
  echo "=== DYNAMO BUILD COMPLETE ===" >&2
) &
BUILD_PID=$!

# Run full sweep — vllm-serve configs first, then dynamo-ec.
# The orchestrator runs configs in order, so dynamo build finishes
# well before dynamo-ec starts (~35 min model load vs ~5 min build).
cd /workspace
python -m benchmarks.multimodal.sweep --config "$SWEEP_CONFIG"

# Guard: if sweep somehow reached dynamo-ec before build finished
wait $BUILD_PID || { echo "ERROR: dynamo build failed" >&2; exit 1; }

echo "=== SWEEP COMPLETE ==="
