#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep wrapper: vllm-serve vs dynamo+EC at request rates 1, 2, 4.
# Runs vllm-serve sweep + dynamo build in parallel, then dynamo-ec sweep.
#
# Usage: bash benchmarks/multimodal/sweep/workflows/run_sweep_397b.sh

set -e
source /opt/dynamo/venv/bin/activate

# Install aiperf from mounted repo (latest main with PR #824)
pip install --no-deps -e /aiperf

# Phase 1: vllm-serve sweep + dynamo build in parallel
(
  pip uninstall -y ai-dynamo ai-dynamo-runtime 2>/dev/null || true
  cd /workspace/lib/bindings/python && maturin develop --uv --release
  cd /workspace && uv pip install --no-deps -e .
  echo "=== DYNAMO BUILD COMPLETE ===" >&2
) &
BUILD_PID=$!

cd /workspace
python -m benchmarks.multimodal.sweep \
  --config benchmarks/multimodal/sweep/experiments/embedding_cache/sweep_397b_vllm.yaml

# Phase 2: wait for build (should already be done), then dynamo-ec sweep
wait $BUILD_PID || { echo "ERROR: dynamo build failed" >&2; exit 1; }

python -m benchmarks.multimodal.sweep \
  --config benchmarks/multimodal/sweep/experiments/embedding_cache/sweep_397b_dynamo.yaml

echo "=== SWEEP COMPLETE ==="
