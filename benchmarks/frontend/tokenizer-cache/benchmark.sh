#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Terminal 1: ./launch.sh off
# Terminal 2: ./benchmark.sh
# Stop the topology with Ctrl-C, then repeat with ./launch.sh on and compare results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTHER_CORES="1-$(($(nproc --all) - 1))"
OUTPUT_DIR="${SCRIPT_DIR}/../../results/tokenizer-cache/$(date +%Y%m%d-%H%M%S)"

echo "Waiting for Qwen/Qwen3-0.6B on http://127.0.0.1:8000"
for _ in {1..60}; do
    curl -sf http://127.0.0.1:8000/v1/models | grep -Fq "Qwen/Qwen3-0.6B" && break
    sleep 1
done
curl -sf http://127.0.0.1:8000/v1/models | grep -Fq "Qwen/Qwen3-0.6B" || {
    echo "Qwen/Qwen3-0.6B was not ready after 60 seconds" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"
echo "Writing AIPerf results to $OUTPUT_DIR"

taskset -c "$OTHER_CORES" \
uvx --from "git+https://github.com/cquil11/aiperf.git@8473e1545476c1d91932aa2402b642b416a23df6" \
    aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --tokenizer Qwen/Qwen3-0.6B \
    --url http://127.0.0.1:8000 \
    --url-strategy round-robin \
    --endpoint-type chat \
    --streaming \
    --no-fixed-schedule \
    --ignore-trace-delays \
    --concurrency 220 \
    --benchmark-duration 120 \
    --benchmark-grace-period 90 \
    --warmup-request-count 32 \
    --public-dataset weka_hf \
    --hf-weka-repo semianalysisai/cc-traces-weka-with-subagents-060526 \
    --num-dataset-entries 336 \
    --extra-inputs ignore_eos:true \
    --use-server-token-count \
    --export-level summary \
    --ui simple \
    --output-artifact-dir "$OUTPUT_DIR"
