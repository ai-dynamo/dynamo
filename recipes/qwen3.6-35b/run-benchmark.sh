#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified driver for the Qwen3.6-35B-A3B-FP8 3-way benchmark.
# Idempotent — re-running steps that already completed is a no-op.
#
# Two axes:
#   --hw <name>      → sources hw/<name>.env (VLLM_IMAGE, HW_NODE_SELECTOR, HW_TOLERATIONS)
#   --config <name>  → resolves to DEPLOY_KIND, DEPLOY_NAME, BENCH_POD inline (see CONFIGS table below)
#
# Usage:
#   ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#   ./run-benchmark.sh -n <namespace> --hw gb200 --config dynamo-fd-ec --step deploy
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
#   BENCH_POD       — name of the aiperf Pod for this config.
#   BENCH_FRONTEND  — service name the bench Pod hits at $FRONTEND:8000.
#                     vllm-serve: a plain Service; DGDs: `<dgd-name>-frontend`
#                     created automatically by the dynamo operator.
#   BENCH_RUN_LABEL — sub-directory written under /perf-cache/artifacts/
#                     so the 3 configs' aiperf artifacts don't collide.
case "$CONFIG" in
  vllm-serve)
    DEPLOY_KIND="deployment"
    DEPLOY_NAME="qwen36-vllm-serve"
    BENCH_POD="qwen36-bench"
    BENCH_FRONTEND="qwen36-vllm-serve"
    BENCH_RUN_LABEL="vllm-serve"
    ;;
  dynamo-fd)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen36-dynamo-fd"
    BENCH_POD="qwen36-fd-bench"
    BENCH_FRONTEND="qwen36-dynamo-fd-frontend"
    BENCH_RUN_LABEL="dynamo-fd"
    ;;
  dynamo-fd-ec)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen36-dynamo-fd-ec"
    BENCH_POD="qwen36-fd-ec-bench"
    BENCH_FRONTEND="qwen36-dynamo-fd-ec-frontend"
    BENCH_RUN_LABEL="dynamo-fd-ec"
    ;;
  "")
    echo "ERROR: --config <name> required" >&2
    echo "Available: vllm-serve dynamo-fd dynamo-fd-ec" >&2
    exit 2 ;;
  *)
    echo "ERROR: unknown config: $CONFIG" >&2
    echo "Available: vllm-serve dynamo-fd dynamo-fd-ec" >&2
    exit 2 ;;
esac
export BENCH_POD BENCH_FRONTEND BENCH_RUN_LABEL

HW_ENV="$HERE/hw/${HW}.env"
if [[ ! -f "$HW_ENV" ]]; then
  echo "ERROR: hardware env file not found: $HW_ENV" >&2
  echo "Available: $(ls "$HERE/hw/" 2>/dev/null | tr '\n' ' ')" >&2
  exit 2
fi
DEPLOY_TPL="$HERE/deploy/${CONFIG}.yaml"

if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a; . "$HW_ENV"; set +a
echo "[hw]     $HW → image=$VLLM_IMAGE node=$HW_NODE_SELECTOR"
echo "[config] $CONFIG → kind=$DEPLOY_KIND deploy=$DEPLOY_NAME bench-pod=$BENCH_POD"

K="kubectl -n $NAMESPACE"
# Limit envsubst to our own template vars so embedded ${MODEL_NAME} /
# ${KEEP_INPUTS_JSON:-} shell vars inside perf.yaml's inline bash stay
# literal. $BENCH_* drive the shared perf.yaml; $VLLM_IMAGE / $HW_*
# drive deploy.yaml + perf.yaml.
TPL_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_POD $BENCH_FRONTEND $BENCH_RUN_LABEL'
APPLY_TPL() { envsubst "$TPL_VARS" <"$1" | $K apply -f -; }

# ---------------- config-agnostic prep ----------------

pvc() {
  # `shared-model-cache` is expected to be pre-provisioned in the namespace
  # (RWX, e.g. FSx Lustre). If your cluster doesn't pre-provision it, create
  # the PVC out-of-band — see README.md → "Storage: shared-model-cache".
  if ! $K get pvc shared-model-cache >/dev/null 2>&1; then
    echo "[pvc] ERROR: PVC 'shared-model-cache' not found in namespace '$NAMESPACE'" >&2
    echo "[pvc] See README.md → 'Storage: shared-model-cache' for provisioning guidance." >&2
    exit 1
  fi
  $K get pvc shared-model-cache
}

download() {
  if $K get job qwen36-model-download >/dev/null 2>&1; then
    if [[ "$($K get job qwen36-model-download -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[download] already complete"
      return
    fi
    echo "[download] previous job present but not Complete — deleting and re-applying"
    $K delete job qwen36-model-download
  fi
  $K apply -f "$HERE/model-cache/model-download.yaml"
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
  $K apply -f "$HERE/data-gen-job.yaml"
  $K wait --for=condition=Complete job/qwen36-generate-datasets --timeout=1800s
  $K logs job/qwen36-generate-datasets | tail -20
}

# ---------------- config-specific lifecycle ----------------

deploy() {
  APPLY_TPL "$DEPLOY_TPL"
  case "$DEPLOY_KIND" in
    deployment)
      $K rollout status "deploy/$DEPLOY_NAME" --timeout=900s
      ;;
    dgd)
      local sel_fe="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=frontend"
      local sel_wk="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=worker"
      echo "[deploy] waiting for DGD Frontend pod ..."
      $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
      echo "[deploy] waiting for VllmWorker pod ..."
      $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=1500s
      ;;
    *)
      echo "ERROR: unknown DEPLOY_KIND=$DEPLOY_KIND" >&2; exit 2 ;;
  esac
}

bench() {
  $K delete pod "$BENCH_POD" --ignore-not-found
  APPLY_TPL "$HERE/perf.yaml"
  $K wait --for=condition=Ready "pod/$BENCH_POD" --timeout=300s
  echo "[bench] streaming logs — Ctrl-C to detach (the run continues in pod)"
  $K logs -f "$BENCH_POD" || true
}

retrieve() {
  # Override the destination root via $BENCHMARK_RESULTS_DIR if your
  # workspace layout differs from the default.
  local base="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
  local dest="$base/$(date +%m-%d)/qwen36-fp8-${HW}/${CONFIG}"
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
