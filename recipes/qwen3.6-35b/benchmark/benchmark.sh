#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark driver for the Qwen3.6-35B-A3B-FP8 recipe.
# Assumes the target config has already been deployed via deploy/deploy.sh.
# Idempotent — re-running dataset is a no-op if the Job already Completed.
#
# Two axes:
#   --hw <name>      → sources ../hw/<name>.env (VLLM_IMAGE, HW_NODE_SELECTOR, HW_TOLERATIONS)
#                      The bench Pod lands on the same node as the deployed
#                      worker so latency is realistic.
#   --config <name>  → resolves to BENCH_POD, BENCH_FRONTEND, BENCH_RUN_LABEL
#                      inline (see CONFIGS table below).
#
# Usage (from the recipe root):
#   ./benchmark/benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#   ./benchmark/benchmark.sh -n <namespace> --hw gb200 --config dynamo-fd-ec
#   ./benchmark/benchmark.sh -n <namespace> --hw h100 --config dynamo-fd --step retrieve
#
# Steps: dataset | bench | retrieve | clean | all
#   dataset is config-agnostic (idempotent — one Job, shared across configs).
#   bench/retrieve/clean are config-specific.
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
RECIPE_ROOT="$(cd "$HERE/.." && pwd)"

# Per-config metadata — mirrors deploy/deploy.sh's table.
#   DEPLOY_NAME     — the Deployment / DGD name; we preflight on it to verify
#                     deploy has happened before applying perf.yaml.
#   DEPLOY_KIND     — drives which kubectl resource to probe.
#   BENCH_POD       — name of the aiperf Pod for this config.
#   BENCH_FRONTEND  — service name the bench Pod hits at $FRONTEND:8000.
#                     vllm-serve: a plain Service; DGDs: `<dgd-name>-frontend`
#                     stamped by the dynamo operator.
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

HW_ENV="$RECIPE_ROOT/hw/${HW}.env"
if [[ ! -f "$HW_ENV" ]]; then
  echo "ERROR: hardware env file not found: $HW_ENV" >&2
  echo "Available: $(ls "$RECIPE_ROOT/hw/" 2>/dev/null | tr '\n' ' ')" >&2
  exit 2
fi

if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a; . "$HW_ENV"; set +a
echo "[hw]     $HW → image=$VLLM_IMAGE node=$HW_NODE_SELECTOR"
echo "[config] $CONFIG → deploy=$DEPLOY_NAME bench-pod=$BENCH_POD frontend=$BENCH_FRONTEND"

K="kubectl -n $NAMESPACE"
TPL_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_POD $BENCH_FRONTEND $BENCH_RUN_LABEL'
APPLY_TPL() { envsubst "$TPL_VARS" <"$1" | $K apply -f -; }

# ---------------- preflight ----------------

require_deploy() {
  # The bench Pod hits ${BENCH_FRONTEND}:8000 — if no one stood that
  # endpoint up, the curl-loop in perf.yaml spins forever. Fail fast
  # with a pointer to deploy/.
  case "$DEPLOY_KIND" in
    deployment)
      if ! $K get deploy "$DEPLOY_NAME" >/dev/null 2>&1; then
        echo "ERROR: Deployment '$DEPLOY_NAME' not found in namespace '$NAMESPACE'." >&2
        echo "       Run deploy first: ./deploy/deploy.sh -n $NAMESPACE --hw $HW --config $CONFIG" >&2
        exit 1
      fi
      ;;
    dgd)
      if ! $K get dynamographdeployment "$DEPLOY_NAME" >/dev/null 2>&1; then
        echo "ERROR: DynamoGraphDeployment '$DEPLOY_NAME' not found in namespace '$NAMESPACE'." >&2
        echo "       Run deploy first: ./deploy/deploy.sh -n $NAMESPACE --hw $HW --config $CONFIG" >&2
        exit 1
      fi
      ;;
  esac
}

# ---------------- config-agnostic prep ----------------

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

bench() {
  require_deploy
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
  # Note: the deployed config is left running. Tear it down via
  # `./deploy/deploy.sh -n $NS --hw $HW --config $CONFIG --step clean`.
}

all() {
  dataset
  bench
  retrieve
}

case "$STEP" in
  dataset|bench|retrieve|clean|all) "$STEP" ;;
  *) echo "unknown step: $STEP" >&2; exit 2 ;;
esac
