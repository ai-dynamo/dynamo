#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sequentially run both configs (vllm-serve, dynamo-fd) against the same
# 8×H100 node and collect aiperf artifacts side-by-side under one dated
# directory ready for the 2-way comparison.
#
# Order per config: deploy → bench → retrieve → clean. Cleaning between
# configs frees the 8 GPUs for the next deployment so they share one node.
#
# Usage:
#   ./run-all-benchmarks.sh -n <namespace>                          # H100
#   ./run-all-benchmarks.sh -n <namespace> --skip-prep              # already PVC/download/dataset-ready
#   ./run-all-benchmarks.sh -n <namespace> --context <kube-context> # pin kubectl context for every call
#   KUBE_CONTEXT=<ctx> ./run-all-benchmarks.sh -n <namespace>       # env-var form, same effect
#   ./run-all-benchmarks.sh -n <namespace> --run-id <id>           # pin the run-isolation id (else a timestamp)
#
# Prep step (PVC + model download + data gen) runs once before the first
# deploy unless --skip-prep is passed.
set -euo pipefail

NAMESPACE=""
HW="h100"
SKIP_PREP="0"
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
RUN_ID="${RUN_ID:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --hw) HW="$2"; shift 2 ;;
    --skip-prep) SKIP_PREP="1"; shift ;;
    --context) KUBE_CONTEXT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Build the forwarded arg list once so every $DRIVER call passes
# --context through identically.
DRIVER_EXTRA=()
[[ -n "$KUBE_CONTEXT" ]] && DRIVER_EXTRA+=(--context "$KUBE_CONTEXT")
# Pin ONE run-id for the whole 2-config pass so vllm-serve and dynamo-fd
# land under the same isolated run dir and each step's retrieve agrees
# with its bench (the steps run as separate run-benchmark.sh processes).
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DRIVER_EXTRA+=(--run-id "$RUN_ID")
if [[ -z "$NAMESPACE" ]]; then
  echo "ERROR: -n <namespace> required" >&2; exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/run-benchmark.sh"
CONFIGS=(vllm-serve dynamo-fd)

TS_DIR="$(date +%m-%d)"
# Override the summary root via $BENCHMARK_RESULTS_DIR — also picked up by
# run-benchmark.sh's retrieve(), so the two paths stay in sync.
BASE_DIR="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
SUMMARY_DIR="$BASE_DIR/${TS_DIR}/qwen35-fp8-${HW}"
mkdir -p "$SUMMARY_DIR"
RUN_LOG="$SUMMARY_DIR/run-all-benchmarks.log"
echo "[run-all] hw=$HW namespace=$NAMESPACE run-id=$RUN_ID" | tee -a "$RUN_LOG"
echo "[run-all] summary dir: $SUMMARY_DIR" | tee -a "$RUN_LOG"

# Prep is config-agnostic; pick any config to source so the env loads cleanly.
if [[ "$SKIP_PREP" != "1" ]]; then
  for step in pvc download dataset; do
    echo "[prep] $step" | tee -a "$RUN_LOG"
    "$DRIVER" -n "$NAMESPACE" --hw "$HW" --config vllm-serve --step "$step" ${DRIVER_EXTRA[@]+"${DRIVER_EXTRA[@]}"} 2>&1 | tee -a "$RUN_LOG"
  done
fi

for cfg in "${CONFIGS[@]}"; do
  echo "" | tee -a "$RUN_LOG"
  echo "========== [$cfg] ==========" | tee -a "$RUN_LOG"
  for step in deploy bench retrieve clean; do
    "$DRIVER" -n "$NAMESPACE" --hw "$HW" --config "$cfg" --step "$step" ${DRIVER_EXTRA[@]+"${DRIVER_EXTRA[@]}"} 2>&1 | tee -a "$RUN_LOG"
  done
done

echo "" | tee -a "$RUN_LOG"
echo "[run-all] all configs done." | tee -a "$RUN_LOG"
echo "[run-all] results: $SUMMARY_DIR/{vllm-serve,dynamo-fd}/" | tee -a "$RUN_LOG"
echo "[run-all] next: python3 $HERE/compare.py $SUMMARY_DIR" | tee -a "$RUN_LOG"
