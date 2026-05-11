#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end driver for the Qwen3.6-35B-A3B-FP8 dynamo-fd-ec benchmark.
# Same shape as vllm-serve/run-benchmark.sh — assumes PVCs +
# model-download + data-gen are already done (run those from the
# vllm-serve recipe first, or via this script's `pvc`, `download`,
# `dataset` subcommands).
#
# Usage:
#   ./run-benchmark.sh -n <namespace>                        # H100, full pipeline
#   ./run-benchmark.sh -n <namespace> --hw gb200             # GB200
#   ./run-benchmark.sh -n <namespace> --hw h100 --step <step>
#
# Steps: pvc | download | dataset | deploy | bench | retrieve | clean
# Hardware: h100 (default) | gb200 (or any hw/<hw>.env file)
set -euo pipefail

NAMESPACE=""
STEP="all"
HW="h100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --hw) HW="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$NAMESPACE" ]]; then
  echo "ERROR: -n <namespace> required" >&2; exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_ROOT="$(cd "$HERE/.." && pwd)"
HW_ENV="$RECIPE_ROOT/hw/${HW}.env"
if [[ ! -f "$HW_ENV" ]]; then
  echo "ERROR: hardware env file not found: $HW_ENV" >&2
  exit 2
fi
if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HW_ENV"; set +a
echo "[hw] $HW → image=$VLLM_IMAGE node=$HW_NODE_SELECTOR"

K="kubectl -n $NAMESPACE"
# Limit envsubst to our own hw vars so embedded ${MODEL_NAME} /
# ${KEEP_INPUTS_JSON:-} shell vars in perf.yaml's inline script stay literal.
TPL_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS'
APPLY_TPL() { envsubst "$TPL_VARS" <"$1" | $K apply -f -; }

pvc() {
  $K apply -f "$RECIPE_ROOT/model-cache/model-cache.yaml"
  $K get pvc
}

download() {
  if $K get job qwen36-model-download >/dev/null 2>&1; then
    if [[ "$($K get job qwen36-model-download -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[download] already complete"
      return
    fi
    $K delete job qwen36-model-download
  fi
  $K apply -f "$RECIPE_ROOT/model-cache/model-download.yaml"
  $K wait --for=condition=Complete job/qwen36-model-download --timeout=3600s
}

dataset() {
  if $K get job qwen36-generate-datasets >/dev/null 2>&1; then
    if [[ "$($K get job qwen36-generate-datasets -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[dataset] already complete"
      return
    fi
    $K delete job qwen36-generate-datasets
  fi
  $K apply -f "$RECIPE_ROOT/data-gen/generate-datasets-job.yaml"
  $K wait --for=condition=Complete job/qwen36-generate-datasets --timeout=1800s
}

deploy() {
  APPLY_TPL "$HERE/deploy.yaml"
  # Wait on operator-stamped labels — see deploy.yaml header comment for
  # why we don't match on `app.kubernetes.io/name`.
  local sel_fe="nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd-ec,nvidia.com/dynamo-component-type=frontend"
  local sel_wk="nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd-ec,nvidia.com/dynamo-component-type=worker"
  echo "[deploy] waiting for DGD Frontend pod to come up ..."
  $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
  echo "[deploy] waiting for VllmWorker pod ..."
  $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=1500s
}

bench() {
  $K delete pod qwen36-fd-ec-bench --ignore-not-found
  APPLY_TPL "$HERE/perf.yaml"
  $K wait --for=condition=Ready pod/qwen36-fd-ec-bench --timeout=300s
  echo "[bench] streaming logs — Ctrl-C to detach (run continues in pod)"
  $K logs -f qwen36-fd-ec-bench || true
}

retrieve() {
  local dest="$HOME/workspace/dynamo-tmp/logs/$(date +%m-%d)/qwen36-fp8-${HW}/dynamo-fd-ec"
  mkdir -p "$dest"
  $K exec qwen36-fd-ec-bench -- \
      tar c --exclude='inputs.json' -C /perf-cache artifacts \
    | tar x -C "$dest"
  echo "[retrieve] landed at $dest"
  find "$dest" -name 'profile_export_aiperf.json' -print
}

clean() {
  $K delete pod qwen36-fd-ec-bench --ignore-not-found
  $K delete dynamographdeployment qwen36-dynamo-fd-ec --ignore-not-found
  # PVCs intentionally NOT deleted.
}

all() {
  pvc
  download
  dataset
  deploy
  bench
  retrieve
}

case "$STEP" in
  pvc|download|dataset|deploy|bench|retrieve|clean|all) "$STEP" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
