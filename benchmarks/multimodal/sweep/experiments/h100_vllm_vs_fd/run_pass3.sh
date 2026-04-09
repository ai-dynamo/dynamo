#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pass 3: dynamo-fd with unique images + tracing overhead A/B test.
# Run inside container on dlcluster H100.
set -eo pipefail

cd /workspace

# Init container env (NATS, etcd, HF_HOME, etc.)
set +u
source /dynamo-toolbox/init_container.sh
set -u

# Rebuild Dynamo in release mode (editable install)
bash /dynamo-toolbox/build.sh

# Verify [PERF] patches are importable
python -c "import benchmarks.multimodal.sweep.vllm_perf_patches; print('[OK] vllm_perf_patches importable')"

# Generate JSONL dataset: 240 requests x 4 unique images = 960 unique 2400x1080 PNGs
JSONL_PATH="benchmarks/multimodal/jsonl/240req_4img_960pool_400word_base64.jsonl"
if [ ! -f "$JSONL_PATH" ]; then
  echo "=== Generating JSONL dataset ==="
  cd benchmarks/multimodal/jsonl
  python main.py \
    -n 240 \
    --images-per-request 4 \
    --images-pool 960 \
    --user-text-tokens 400 \
    --image-size 2400 1080 \
    --seed 42
  cd /workspace
  echo ""
fi

# Run the sweep
python -m benchmarks.multimodal.sweep \
  --config benchmarks/multimodal/sweep/experiments/h100_vllm_vs_fd/sweep_pass3.yaml

echo ""
echo "=== Pass 3 complete ==="
echo "Results in: logs/04-09/h100-fd-pass3/"

# Show [PERF] line counts from server logs
echo ""
echo "=== [PERF] line counts ==="
find logs/04-09/h100-fd-pass3 -name "server.log" -exec sh -c \
  'echo "  $1: $(grep -c "\[PERF\]" "$1" 2>/dev/null || echo 0) lines"' _ {} \;
