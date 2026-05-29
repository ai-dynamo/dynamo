#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sequentially run all three configs (vllm-serve, dynamo-fd, dynamo-fd-ec)
# against the same hardware target and collect aiperf artifacts side-by-side
# under one dated directory ready for the 3-way comparison.
#
# Order per config: deploy → bench → retrieve → clean. Cleaning between
# configs frees the GPU for the next deployment so they share one node.
#
# Usage:
#   ./run-all-benchmarks.sh -n <namespace>                    # H100
#   ./run-all-benchmarks.sh -n <namespace> --hw gb200         # GB200
#   ./run-all-benchmarks.sh -n <namespace> --hw gb200 --skip-prep   # already PVC/download/dataset-ready
#
# Prep step (PVC + model download + data gen) runs once before the first
# deploy unless --skip-prep is passed.
set -euo pipefail

NAMESPACE=""
HW="h100"
SKIP_PREP="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --hw) HW="$2"; shift 2 ;;
    --skip-prep) SKIP_PREP="1"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$NAMESPACE" ]]; then
  echo "ERROR: -n <namespace> required" >&2; exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/run-benchmark.sh"
CONFIGS=(vllm-serve dynamo-fd dynamo-fd-ec)

TS_DIR="$(date +%m-%d)"
# Override the summary root via $BENCHMARK_RESULTS_DIR — also picked up by
# run-benchmark.sh's retrieve(), so the two paths stay in sync.
BASE_DIR="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
SUMMARY_DIR="$BASE_DIR/${TS_DIR}/qwen36-fp8-${HW}"
mkdir -p "$SUMMARY_DIR"
RUN_LOG="$SUMMARY_DIR/run-all-benchmarks.log"
echo "[run-all] hw=$HW namespace=$NAMESPACE" | tee -a "$RUN_LOG"
echo "[run-all] summary dir: $SUMMARY_DIR" | tee -a "$RUN_LOG"

# Prep is config-agnostic; pick any config to source so config.env loads cleanly.
if [[ "$SKIP_PREP" != "1" ]]; then
  for step in pvc download dataset; do
    echo "[prep] $step" | tee -a "$RUN_LOG"
    "$DRIVER" -n "$NAMESPACE" --hw "$HW" --config vllm-serve --step "$step" 2>&1 | tee -a "$RUN_LOG"
  done
fi

for cfg in "${CONFIGS[@]}"; do
  echo "" | tee -a "$RUN_LOG"
  echo "========== [$cfg] ==========" | tee -a "$RUN_LOG"
  for step in deploy bench retrieve clean; do
    "$DRIVER" -n "$NAMESPACE" --hw "$HW" --config "$cfg" --step "$step" 2>&1 | tee -a "$RUN_LOG"
  done
done

echo "" | tee -a "$RUN_LOG"
echo "[run-all] all three configs done." | tee -a "$RUN_LOG"
echo "[run-all] results: $SUMMARY_DIR/{vllm-serve,dynamo-fd,dynamo-fd-ec}/" | tee -a "$RUN_LOG"
echo "[run-all] each config's profile_export_aiperf.json holds the headline metrics." | tee -a "$RUN_LOG"
