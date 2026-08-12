#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-config benchmark driver for Qwen3-VL-32B-Instruct-FP8 recipe.
# Applies the shared benchmark/perf.yaml template with config-specific
# env vars via envsubst, then submits the Job.
#
# Usage:
#   ./run-benchmark.sh --config agg        # benchmark aggregated deployment
#   ./run-benchmark.sh --config disagg     # benchmark disaggregated deployment
#   ./run-benchmark.sh --config agg --dry-run   # render YAML without applying
#
set -euo pipefail

CONFIG=""
DRY_RUN=0
NAMESPACE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config|-c) CONFIG="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --config {agg|disagg} [-n <namespace>] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: --config {agg|disagg} is required" >&2
  exit 2
fi

# ── Per-config knobs ──────────────────────────────────────────────────────────
case "$CONFIG" in
  agg)
    export BENCH_NAME="qwen3-vl-32b-fp8-agg-perf"
    export BENCH_ENDPOINT="qwen3-vl-32b-fp8-vllm-agg-frontend:8000"
    ;;
  disagg)
    export BENCH_NAME="qwen3-vl-32b-fp8-disagg-perf"
    export BENCH_ENDPOINT="qwen3-vl-32b-fp8-vllm-disagg-frontend:8000"
    ;;
  *)
    echo "ERROR: unknown config '$CONFIG'. Use 'agg' or 'disagg'." >&2
    exit 2
    ;;
esac

# ── Locate the shared template ────────────────────────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$HERE/perf.yaml"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: template not found: $TEMPLATE" >&2
  exit 1
fi

# ── Render via envsubst ───────────────────────────────────────────────────────
RENDERED=$(envsubst '${BENCH_NAME} ${BENCH_ENDPOINT}' < "$TEMPLATE")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "$RENDERED"
  exit 0
fi

# ── Apply ─────────────────────────────────────────────────────────────────────
NS_ARGS=""
if [[ -n "$NAMESPACE" ]]; then
  NS_ARGS="-n $NAMESPACE"
fi

echo "[run-benchmark] config=$CONFIG name=$BENCH_NAME endpoint=$BENCH_ENDPOINT"
echo "$RENDERED" | kubectl apply ${NS_ARGS} -f -
echo "[run-benchmark] Job '$BENCH_NAME' submitted."
echo "[run-benchmark] Monitor: kubectl ${NS_ARGS} logs -f job/$BENCH_NAME"
