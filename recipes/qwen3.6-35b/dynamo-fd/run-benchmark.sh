#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end driver for the Qwen3.6-35B-A3B-FP8 dynamo-fd (no EC)
# benchmark. Same shape as dynamo-fd-ec's driver; assumes PVCs +
# model-download + data-gen are already done.
#
# Usage:
#   ./run-benchmark.sh -n <namespace>                        # H100, deploy + bench
#   ./run-benchmark.sh -n <namespace> --hw gb200             # GB200
#   ./run-benchmark.sh -n <namespace> --hw h100 --step <step>
#
# Steps: deploy | bench | retrieve | clean
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

deploy() {
  APPLY_TPL "$HERE/deploy.yaml"
  local sel_fe="nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd,nvidia.com/dynamo-component-type=frontend"
  local sel_wk="nvidia.com/dynamo-graph-deployment-name=qwen36-dynamo-fd,nvidia.com/dynamo-component-type=worker"
  echo "[deploy] waiting for DGD Frontend pod ..."
  $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
  echo "[deploy] waiting for VllmWorker pod ..."
  $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=1500s
}

bench() {
  $K delete pod qwen36-fd-bench --ignore-not-found
  APPLY_TPL "$HERE/perf.yaml"
  $K wait --for=condition=Ready pod/qwen36-fd-bench --timeout=300s
  echo "[bench] streaming logs — Ctrl-C to detach"
  $K logs -f qwen36-fd-bench || true
}

retrieve() {
  local dest="$HOME/workspace/dynamo-tmp/logs/$(date +%m-%d)/qwen36-fp8-${HW}/dynamo-fd"
  mkdir -p "$dest"
  $K exec qwen36-fd-bench -- \
      tar c --exclude='inputs.json' -C /perf-cache artifacts \
    | tar x -C "$dest"
  echo "[retrieve] landed at $dest"
  find "$dest" -name 'profile_export_aiperf.json' -print
}

clean() {
  $K delete pod qwen36-fd-bench --ignore-not-found
  $K delete dynamographdeployment qwen36-dynamo-fd --ignore-not-found
}

all() {
  deploy
  bench
  retrieve
}

case "$STEP" in
  deploy|bench|retrieve|clean|all) "$STEP" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
