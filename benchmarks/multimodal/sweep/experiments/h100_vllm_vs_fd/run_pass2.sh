#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pass 2: re-run sweep with [PERF] patches and server.log capture.
# Run inside container on dlcluster H100.
set -eo pipefail

cd /workspace

# Init container env (NATS, etcd, HF_HOME, etc.)
# Temporarily allow unbound vars — init_container.sh may reference unset PYTHONPATH
set +u
source /dynamo-toolbox/init_container.sh
set -u

# Rebuild Dynamo in release mode (editable install)
bash /dynamo-toolbox/build.sh

# Verify [PERF] patches are importable
python -c "import benchmarks.multimodal.sweep.vllm_perf_patches; print('[OK] vllm_perf_patches importable')"

# Run the sweep
python -m benchmarks.multimodal.sweep \
  --config benchmarks/multimodal/sweep/experiments/h100_vllm_vs_fd/sweep_pass2.yaml

echo ""
echo "=== Pass 2 complete ==="
echo "Results in: logs/04-08/h100-vllm-vs-fd-pass2/"

# Show [PERF] line counts from server logs
echo ""
echo "=== [PERF] line counts ==="
find logs/04-08/h100-vllm-vs-fd-pass2 -name "server.log" -exec sh -c \
  'echo "  $1: $(grep -c "\[PERF\]" "$1" 2>/dev/null || echo 0) lines"' _ {} \;
