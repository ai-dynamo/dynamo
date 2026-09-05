#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified driver for the Qwen3.6-27B 3-way benchmark.
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
    DEPLOY_NAME="qwen3627-vllm-serve"
    BENCH_POD="qwen3627-bench"
    BENCH_FRONTEND="qwen3627-vllm-serve"
    BENCH_RUN_LABEL="vllm-serve"
    ;;
  dynamo-fd)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen3627-dynamo-fd"
    BENCH_POD="qwen3627-fd-bench"
    BENCH_FRONTEND="qwen3627-dynamo-fd-frontend"
    BENCH_RUN_LABEL="dynamo-fd"
    ;;
  dynamo-fd-ec)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen3627-dynamo-fd-ec"
    BENCH_POD="qwen3627-fd-ec-bench"
    BENCH_FRONTEND="qwen3627-dynamo-fd-ec-frontend"
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
  if $K get job qwen3627-model-download >/dev/null 2>&1; then
    if [[ "$($K get job qwen3627-model-download -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[download] already complete"
      return
    fi
    echo "[download] previous job present but not Complete — deleting and re-applying"
    $K delete job qwen3627-model-download
  fi
  $K apply -f "$HERE/model-cache/model-download.yaml"
  $K wait --for=condition=Complete job/qwen3627-model-download --timeout=3600s
}

dataset() {
  if $K get job qwen3627-generate-datasets >/dev/null 2>&1; then
    if [[ "$($K get job qwen3627-generate-datasets -o jsonpath='{.status.succeeded}')" == "1" ]]; then
      echo "[dataset] already complete"
      return
    fi
    $K delete job qwen3627-generate-datasets
  fi
  $K apply -f "$HERE/data-gen-job.yaml"
  $K wait --for=condition=Complete job/qwen3627-generate-datasets --timeout=1800s
  $K logs job/qwen3627-generate-datasets | tail -20
}

# ---------------- config-specific lifecycle ----------------

# `kubectl wait` errors out immediately with "no matching resources found"
# if a label selector matches zero pods at call time — a real race right
# after `kubectl apply` of a DGD, before the operator has created the Pod
# objects. Poll for at least one matching pod first.
wait_pod_created() {
  local selector="$1" timeout="$2" waited=0
  while [[ "$($K get pod -l "$selector" --no-headers 2>/dev/null | wc -l)" -eq 0 ]]; do
    if (( waited >= timeout )); then
      echo "ERROR: no pod matching '$selector' after ${timeout}s" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

deploy() {
  APPLY_TPL "$DEPLOY_TPL"
  case "$DEPLOY_KIND" in
    deployment)
      # 1800s matches deploy/vllm-serve.yaml's progressDeadlineSeconds so the
      # CLI watch doesn't give up before the Deployment controller would.
      $K rollout status "deploy/$DEPLOY_NAME" --timeout=1800s
      ;;
    dgd)
      local sel_fe="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=frontend"
      local sel_wk="nvidia.com/dynamo-graph-deployment-name=$DEPLOY_NAME,nvidia.com/dynamo-component-type=worker"
      echo "[deploy] waiting for DGD Frontend pod to be created ..."
      wait_pod_created "$sel_fe" 900
      echo "[deploy] waiting for DGD Frontend pod ..."
      $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
      echo "[deploy] waiting for VllmWorker pod to be created ..."
      wait_pod_created "$sel_wk" 1500
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
  echo "[bench] streaming logs — auto-detaches once the run reports complete"
  echo "[bench] (perf.yaml then sleeps 3600s in-pod so retrieve() can still exec in afterward)"
  # perf.yaml prints "Run complete." then sleeps 3600s so retrieve() can
  # exec in later. A plain `kubectl logs -f | awk '...{exit}'` pipeline does
  # NOT return early even though awk exits: bash waits for every stage of a
  # pipeline to finish, and `kubectl logs -f` itself only exits once the
  # container does (or the connection drops) — it won't get SIGPIPE until
  # its *next write*, and the pod goes quiet after "Run complete.", so it
  # would just block for the full 3600s anyway. Run the log follower in the
  # background instead, read it via a FIFO in the foreground, and kill it
  # explicitly once we see the marker.
  local logdir logfifo logpid found=0
  logdir="$(mktemp -d)"
  logfifo="$logdir/bench.fifo"
  mkfifo "$logfifo"
  $K logs -f "$BENCH_POD" >"$logfifo" 2>&1 &
  logpid=$!
  while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == "Run complete."* ]]; then
      found=1
      break
    fi
  done <"$logfifo"
  kill "$logpid" 2>/dev/null || true
  wait "$logpid" 2>/dev/null || true
  rm -rf "$logdir"
  # A stream that ends without ever printing "Run complete." — pod crashed,
  # aiperf errored out, connection dropped early — is a real failed run;
  # fail here so `set -e` aborts the sweep instead of silently continuing
  # into retrieve()/clean() as though the benchmark had passed.
  if [[ "$found" -ne 1 ]]; then
    echo "ERROR: benchmark pod log stream ended without 'Run complete.'" >&2
    return 1
  fi
}

retrieve() {
  # Override the destination root via $BENCHMARK_RESULTS_DIR if your
  # workspace layout differs from the default.
  local base="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
  local dest="$base/$(date +%m-%d)/qwen3627-${HW}/${CONFIG}"
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
