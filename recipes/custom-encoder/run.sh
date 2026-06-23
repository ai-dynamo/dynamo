#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the aggregated CustomEncoder demo recipe (ai-dynamo/dynamo#10832).
# Idempotent — re-running a completed step is a no-op.
#
# Usage:
#   ./run.sh -n <namespace> --image <tag> [--node <hostname>] [--step <step>]
#
# Flags:
#   -n, --namespace  Target namespace (required).
#   --image          Custom image built from PR #10832 (required except --step clean).
#                    e.g. nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-custom-encoder
#   --node           Pin Frontend + Worker to this kubernetes.io/hostname (optional).
#   --step           pvc | download | deploy | smoke | encoder-log | clean | all
#                    (default: all)
set -euo pipefail

NAMESPACE=""
IMAGE=""
NODE=""
STEP="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$NAMESPACE" ]] && { echo "ERROR: -n <namespace> required" >&2; exit 2; }
if [[ "$STEP" != "clean" && -z "$IMAGE" ]]; then
  echo "ERROR: --image <tag> required (custom image built from PR #10832 — see README)" >&2
  exit 2
fi
command -v envsubst >/dev/null 2>&1 || {
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
}

HERE="$(cd "$(dirname "$0")" && pwd)"
K="kubectl -n $NAMESPACE"

DGD_NAME="custom-encoder-agg"
DL_JOB="custom-encoder-model-download"
SMOKE_POD="custom-encoder-smoke"

export VLLM_IMAGE="$IMAGE"
if [[ -n "$NODE" ]]; then
  export HW_NODE_SELECTOR="{\"kubernetes.io/hostname\":\"$NODE\"}"
else
  export HW_NODE_SELECTOR="{}"
fi
export SMOKE_FRONTEND="${DGD_NAME}-frontend"

# Allow-lists keep pod-side ${MODEL_NAME} / ${ENCODER_CLASS} / ${FRONTEND} /
# ${IMAGE_URL} literal so they resolve from the container env at runtime.
DEPLOY_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR'
SMOKE_VARS='$SMOKE_FRONTEND'

pvc() {
  if ! $K get pvc shared-model-cache >/dev/null 2>&1; then
    echo "[pvc] ERROR: PVC 'shared-model-cache' not found in namespace '$NAMESPACE'" >&2
    echo "[pvc] See README.md → 'Storage' for a copy-pasteable manifest." >&2
    exit 1
  fi
  $K get pvc shared-model-cache
}

download() {
  if [[ "$($K get job "$DL_JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null)" == "1" ]]; then
    echo "[download] already complete"
    return
  fi
  $K delete job "$DL_JOB" --ignore-not-found
  $K apply -f "$HERE/model-cache/model-download.yaml"
  $K wait --for=condition=Complete "job/$DL_JOB" --timeout=1800s
}

encoder_log() {
  local sel="nvidia.com/dynamo-graph-deployment-name=$DGD_NAME,nvidia.com/dynamo-component-type=worker"
  if $K logs -l "$sel" --tail=-1 2>/dev/null | grep -m1 "Loaded CustomEncoder"; then
    echo "[encoder-log] custom encoder loaded in-process"
  else
    echo "[encoder-log] 'Loaded CustomEncoder' not seen yet (worker may still be starting)"
  fi
}

deploy() {
  envsubst "$DEPLOY_VARS" < "$HERE/deploy.yaml" | $K apply -f -
  local sel="nvidia.com/dynamo-graph-deployment-name=$DGD_NAME"
  # The operator takes a few seconds to reconcile the DGD into pods.
  # `kubectl wait -l` errors "no matching resources found" if it fires before
  # any pod exists, so poll for existence (Frontend + Worker = 2) first.
  echo "[deploy] waiting for operator to stamp DGD pods ..."
  for _ in $(seq 1 60); do
    [[ "$($K get pod -l "$sel" --no-headers 2>/dev/null | wc -l | tr -d ' ')" -ge 2 ]] && break
    sleep 5
  done
  echo "[deploy] waiting for Frontend Ready ..."
  $K wait --for=condition=Ready pod \
     -l "$sel,nvidia.com/dynamo-component-type=frontend" --timeout=600s
  echo "[deploy] waiting for VllmWorker Ready ..."
  $K wait --for=condition=Ready pod \
     -l "$sel,nvidia.com/dynamo-component-type=worker" --timeout=900s
  encoder_log || true
}

smoke() {
  $K delete pod "$SMOKE_POD" --ignore-not-found
  envsubst "$SMOKE_VARS" < "$HERE/smoke.yaml" | $K apply -f -
  echo "[smoke] polling for SMOKE PASS/FAIL marker (pod self-terminates) ..."
  for _ in $(seq 1 90); do
    local log
    log="$($K logs "$SMOKE_POD" 2>/dev/null || true)"
    if grep -q "SMOKE PASS" <<<"$log"; then echo "$log"; echo "[smoke] PASS"; return 0; fi
    if grep -q "SMOKE FAIL" <<<"$log"; then echo "$log"; echo "[smoke] FAIL"; return 1; fi
    local phase
    phase="$($K get pod "$SMOKE_POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [[ "$phase" == "Failed" ]] && { $K logs "$SMOKE_POD" || true; echo "[smoke] pod Failed"; return 1; }
    sleep 10
  done
  $K logs "$SMOKE_POD" --tail=80 || true
  echo "[smoke] TIMEOUT: no SMOKE marker in ~15 min" >&2
  return 1
}

clean() {
  $K delete pod "$SMOKE_POD" --ignore-not-found
  $K delete dynamographdeployment "$DGD_NAME" --ignore-not-found
  # PVC intentionally retained (model cache). To wipe:
  #   kubectl -n $NAMESPACE delete pvc shared-model-cache
}

all() {
  pvc
  download
  deploy
  smoke
}

case "$STEP" in
  encoder-log) encoder_log ;;
  pvc|download|deploy|smoke|clean|all) "$STEP" ;;
  *)
    echo "unknown step: $STEP" >&2
    echo "Available: pvc download deploy smoke encoder-log clean all" >&2
    exit 2 ;;
esac
