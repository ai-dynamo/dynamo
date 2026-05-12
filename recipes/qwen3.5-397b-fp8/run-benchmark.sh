#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified driver for the Qwen3.5-397B-A17B-FP8 3-way benchmark on
# 8×H100. Idempotent — re-running steps that already completed is a no-op.
#
# Two axes:
#   --hw <name>      → sources hw/<name>.env (VLLM_IMAGE, HW_NODE_SELECTOR, HW_TOLERATIONS)
#   --config <name>  → resolves to DEPLOY_KIND, DEPLOY_NAME, BENCH_POD inline (see CONFIGS table below)
#
# Usage:
#   ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#   ./run-benchmark.sh -n <namespace> --hw h100 --config dynamo-fd --step deploy
#
# Steps: pvc | download | dataset | deploy | bench | retrieve | clean | all
#   pvc/download/dataset are config-agnostic (idempotent prep).
#   deploy/bench/retrieve/clean are config-specific.
set -euo pipefail

NAMESPACE=""
STEP="all"
HW="h100"
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --hw) HW="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$NAMESPACE" ]]; then
  echo "ERROR: -n <namespace> required" >&2; exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"

# Per-config metadata. Keep this list in sync with the sibling config dirs.
#   DEPLOY_KIND branches deploy() + clean():
#     deployment → kubectl rollout status + delete Deployment+Service
#     dgd        → kubectl wait on operator-stamped DGD pod labels + delete DGD
case "$CONFIG" in
  vllm-serve)
    DEPLOY_KIND="deployment"
    DEPLOY_NAME="qwen35-vllm-serve"
    BENCH_POD="qwen35-bench"
    ;;
  dynamo-fd)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen35-dynamo-fd"
    BENCH_POD="qwen35-fd-bench"
    ;;
  "")
    echo "ERROR: --config <name> required" >&2
    echo "Available: vllm-serve dynamo-fd" >&2
    exit 2 ;;
  *)
    echo "ERROR: unknown config: $CONFIG" >&2
    echo "Available: vllm-serve dynamo-fd" >&2
    exit 2 ;;
esac

HW_ENV="$HERE/hw/${HW}.env"
if [[ ! -f "$HW_ENV" ]]; then
  echo "ERROR: hardware env file not found: $HW_ENV" >&2
  echo "Available: $(ls "$HERE/hw/" 2>/dev/null | tr '\n' ' ')" >&2
  exit 2
fi
CONFIG_DIR="$HERE/${CONFIG}"

if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a; . "$HW_ENV"; set +a
echo "[hw]     $HW → image=$VLLM_IMAGE node=$HW_NODE_SELECTOR"
echo "[config] $CONFIG → kind=$DEPLOY_KIND deploy=$DEPLOY_NAME bench-pod=$BENCH_POD"

K="kubectl -n $NAMESPACE"
# Limit envsubst to our own hw vars so embedded ${MODEL_NAME} /
# ${KEEP_INPUTS_JSON:-} shell vars in perf.yaml's inline script stay literal.
TPL_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS'
APPLY_TPL() { envsubst "$TPL_VARS" <"$1" | $K apply -f -; }

# ---------------- config-agnostic prep ----------------

pvc() {
  # We rely on a pre-provisioned `shared-model-cache` PVC in the
  # namespace (RWX, ≥600 GiB — typically FSx Lustre on AWS). The recipe
  # does not create one. Verify it exists and bail out loudly if not.
  if ! $K get pvc shared-model-cache >/dev/null 2>&1; then
    echo "ERROR: PVC 'shared-model-cache' not found in namespace $NAMESPACE." >&2
    echo "       Provision an RWX PVC named 'shared-model-cache' (≥600 GiB) and re-run." >&2
    exit 2
  fi
  $K get pvc shared-model-cache
}

download() {
  if $K get job qwen35-model-download >/dev/null 2>&1; then
    if [[ "$($K get job qwen35-model-download -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[download] already complete"
      return
    fi
    echo "[download] previous job present but not Complete — deleting and re-applying"
    $K delete job qwen35-model-download
  fi
  $K apply -f "$HERE/model-cache/model-download.yaml"
  $K wait --for=condition=Complete job/qwen35-model-download --timeout=3600s
}

dataset() {
  if $K get job qwen35-generate-datasets >/dev/null 2>&1; then
    if [[ "$($K get job qwen35-generate-datasets -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[dataset] already complete"
      return
    fi
    $K delete job qwen35-generate-datasets
  fi
  $K apply -f "$HERE/data-gen/generate-datasets-job.yaml"
  $K wait --for=condition=Complete job/qwen35-generate-datasets --timeout=1800s
  $K logs job/qwen35-generate-datasets | tail -20
}

# ---------------- config-specific lifecycle ----------------

deploy() {
  APPLY_TPL "$CONFIG_DIR/deploy.yaml"
  case "$DEPLOY_KIND" in
    deployment)
      # 397B FP8 cold boot from FSx: image pull + 400 GiB weight load +
      # torch.compile + cudagraph capture ≈ 25-40 min. Bump well past
      # the qwen3.6 recipe's 900s.
      $K rollout status "deploy/$DEPLOY_NAME" --timeout=2700s
      ;;
    dgd)
      local sel_fe="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=frontend"
      local sel_wk="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=worker"
      echo "[deploy] waiting for DGD Frontend pod ..."
      $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
      echo "[deploy] waiting for VllmWorker pod (397B FP8 cold load can take 30+ min) ..."
      $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=2700s
      ;;
    *)
      echo "ERROR: unknown DEPLOY_KIND=$DEPLOY_KIND" >&2; exit 2 ;;
  esac
}

bench() {
  $K delete pod "$BENCH_POD" --ignore-not-found
  APPLY_TPL "$CONFIG_DIR/perf.yaml"
  $K wait --for=condition=Ready "pod/$BENCH_POD" --timeout=300s
  echo "[bench] streaming logs — Ctrl-C to detach (the run continues in pod)"
  $K logs -f "$BENCH_POD" || true
}

retrieve() {
  # Override the destination root via $BENCHMARK_RESULTS_DIR if your
  # workspace layout differs from the default.
  local base="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
  local dest="$base/$(date +%m-%d)/qwen35-fp8-${HW}/${CONFIG}"
  mkdir -p "$dest"
  $K exec "$BENCH_POD" -- \
      tar c --exclude='inputs.json' -C /perf-cache artifacts \
    | tar x -C "$dest"
  echo "[retrieve] landed at $dest"
  find "$dest" -name 'profile_export_aiperf.json' -print
}

clean() {
  $K delete pod "$BENCH_POD" --ignore-not-found
  case "$DEPLOY_KIND" in
    deployment)
      $K delete deploy "$DEPLOY_NAME" --ignore-not-found
      $K delete service "$DEPLOY_NAME" --ignore-not-found
      ;;
    dgd)
      $K delete dynamographdeployment "$DEPLOY_NAME" --ignore-not-found
      ;;
  esac
  # Note: PVCs intentionally NOT deleted — that would force model re-download.
  # To wipe everything:
  #   kubectl -n $NS delete pvc shared-model-cache
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
