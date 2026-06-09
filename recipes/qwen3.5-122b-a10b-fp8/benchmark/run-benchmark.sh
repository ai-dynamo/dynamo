#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified driver for the Qwen3.5-122B-A10B benchmark.
# Idempotent — re-running steps that already completed is a no-op.
#
# Available configs:
#   vllm-serve     : plain vllm serve, no Dynamo features
#   vllm-serve-ec  : vllm serve + Dynamo embedding cache (no frontend-decoding)
#   dynamo-fd      : Dynamo + frontend-decoding, no EC
#   dynamo-fd-ec   : Dynamo + frontend-decoding + EC
#
# Usage:
#   ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#   ./run-benchmark.sh -n <namespace> --hw h100 --config dynamo-fd-ec --step deploy
#   ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve-ec --step bench
#
# Steps: dataset | download | deploy | bench | retrieve | clean | all
set -euo pipefail

NAMESPACE=""
STEP="all"
HW="h100"
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --step)         STEP="$2";      shift 2 ;;
    --hw)           HW="$2";        shift 2 ;;
    --config)       CONFIG="$2";    shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NAMESPACE" ]] || { echo "ERROR: -n <namespace> required" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE="$(dirname "$HERE")"

case "$CONFIG" in
  vllm-serve)
    DEPLOY_KIND="deployment"; DEPLOY_NAME="qwen35-122b-vllm-serve"
    BENCH_POD="qwen35-122b-bench";      BENCH_FRONTEND="qwen35-122b-vllm-serve"
    BENCH_RUN_LABEL="vllm-serve" ;;
  vllm-serve-ec)
    DEPLOY_KIND="deployment"; DEPLOY_NAME="qwen35-122b-vllm-serve-ec"
    BENCH_POD="qwen35-122b-vs-ec-bench"; BENCH_FRONTEND="qwen35-122b-vllm-serve-ec"
    BENCH_RUN_LABEL="vllm-serve-ec" ;;
  dynamo-fd)
    DEPLOY_KIND="dgd"; DEPLOY_NAME="qwen35-122b-dynamo-fd"
    BENCH_POD="qwen35-122b-fd-bench";   BENCH_FRONTEND="qwen35-122b-dynamo-fd-frontend"
    BENCH_RUN_LABEL="dynamo-fd" ;;
  dynamo-fd-ec)
    DEPLOY_KIND="dgd"; DEPLOY_NAME="qwen35-122b-dynamo-fd-ec"
    BENCH_POD="qwen35-122b-fd-ec-bench"; BENCH_FRONTEND="qwen35-122b-dynamo-fd-ec-frontend"
    BENCH_RUN_LABEL="dynamo-fd-ec" ;;
  "")
    echo "ERROR: --config required" >&2
    echo "Available: vllm-serve  vllm-serve-ec  dynamo-fd  dynamo-fd-ec" >&2; exit 2 ;;
  *)
    echo "ERROR: unknown config '$CONFIG'" >&2
    echo "Available: vllm-serve  vllm-serve-ec  dynamo-fd  dynamo-fd-ec" >&2; exit 2 ;;
esac
export BENCH_POD BENCH_FRONTEND BENCH_RUN_LABEL

HW_ENV="$RECIPE/hw/${HW}.env"
[[ -f "$HW_ENV" ]] || { echo "ERROR: hw env not found: $HW_ENV" >&2; exit 2; }
command -v envsubst >/dev/null || { echo "ERROR: envsubst missing (brew install gettext)" >&2; exit 2; }

set -a; . "$HW_ENV"; set +a
echo "[hw]     $HW → image=$VLLM_IMAGE"
echo "[config] $CONFIG → kind=$DEPLOY_KIND name=$DEPLOY_NAME bench=$BENCH_POD"

K="kubectl -n $NAMESPACE"
TPL='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_POD $BENCH_FRONTEND $BENCH_RUN_LABEL'
APPLY() { envsubst "$TPL" <"$1" | $K apply -f -; }

# ── prep steps (config-agnostic, idempotent) ──────────────────────────────

download() {
  if $K get job qwen35-122b-model-download &>/dev/null; then
    [[ "$($K get job qwen35-122b-model-download -o jsonpath='{.status.succeeded}')" == "1" ]] \
      && { echo "[download] already complete"; return; }
    $K delete job qwen35-122b-model-download
  fi
  $K apply -f "$RECIPE/model-cache/model-download.yaml"
  $K wait --for=condition=Complete job/qwen35-122b-model-download --timeout=3600s
}

dataset() {
  if $K get job qwen35-122b-generate-datasets &>/dev/null; then
    [[ "$($K get job qwen35-122b-generate-datasets -o jsonpath='{.status.succeeded}')" == "1" ]] \
      && { echo "[dataset] already complete"; return; }
    $K delete job qwen35-122b-generate-datasets
  fi
  $K apply -f "$HERE/data-gen-job.yaml"
  $K wait --for=condition=Complete job/qwen35-122b-generate-datasets --timeout=1800s
  $K logs job/qwen35-122b-generate-datasets | tail -5
}

# ── config-specific lifecycle ─────────────────────────────────────────────

deploy() {
  APPLY "$RECIPE/deploy/${CONFIG}.yaml"
  case "$DEPLOY_KIND" in
    deployment)
      $K rollout status "deploy/$DEPLOY_NAME" --timeout=1800s ;;
    dgd)
      local sel_fe="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=frontend"
      local sel_wk="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=worker"
      echo "[deploy] waiting for Frontend..."
      $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
      echo "[deploy] waiting for VllmWorker..."
      $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=1800s ;;
  esac
}

bench() {
  $K delete pod "$BENCH_POD" --ignore-not-found
  APPLY "$HERE/perf.yaml"
  $K wait --for=condition=Ready "pod/$BENCH_POD" --timeout=300s
  echo "[bench] streaming logs — Ctrl-C to detach (run continues in pod)"
  $K logs -f "$BENCH_POD" || true
}

retrieve() {
  local dest="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}/$(date +%m-%d)/qwen35-122b/${CONFIG}"
  mkdir -p "$dest"
  $K exec "$BENCH_POD" -- \
      tar c --exclude='inputs.json' -C /perf-cache/artifacts/qwen35_122b "$BENCH_RUN_LABEL" \
    | tar x -C "$dest"
  echo "[retrieve] -> $dest"
  find "$dest" -name 'profile_export_aiperf.json' -print
}

clean() {
  $K delete pod "$BENCH_POD" --ignore-not-found
  case "$DEPLOY_KIND" in
    deployment)
      $K delete deploy "$DEPLOY_NAME" --ignore-not-found
      $K delete service "$DEPLOY_NAME" --ignore-not-found ;;
    dgd)
      $K delete dynamographdeployment "$DEPLOY_NAME" --ignore-not-found ;;
  esac
}

all() { download; dataset; deploy; bench; retrieve; }

case "$STEP" in
  download|dataset|deploy|bench|retrieve|clean|all) "$STEP" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
