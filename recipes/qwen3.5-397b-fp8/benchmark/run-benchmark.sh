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
# Optional:
#   --context <ctx>  → kubectl --context to pin every call (else KUBE_CONTEXT env, else current-context)
#   --run-id <id>    → run-isolation key for the artifact dir on the shared
#                      FSx PVC (else $RUN_ID env; bench mints a timestamp,
#                      retrieve discovers the newest run). See run_label_* below.
#
# Usage:
#   ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#   ./run-benchmark.sh -n <namespace> --hw h100 --config dynamo-fd --step deploy
#   KUBE_CONTEXT=my-cluster ./run-benchmark.sh -n <namespace> --hw h100 --config vllm-serve
#
# Steps: pvc | download | dataset | deploy | bench | retrieve | clean | all
#   pvc/download/dataset are config-agnostic (idempotent prep).
#   deploy/bench/retrieve/clean are config-specific.
set -euo pipefail

NAMESPACE=""
STEP="all"
HW="h100"
CONFIG=""
# Pin the kubectl context for every call in this script so a parallel
# `kubectl config use-context` (Claude tool runs, sibling sweeps) can't
# silently retarget the sweep mid-run. Falls back to KUBE_CONTEXT env
# var, then to the global current-context. CLI flag wins.
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
# Run-isolation key for the artifact dir (see run_label_* below). The
# orchestrator pins one value across deploy/bench/retrieve; standalone
# bench mints a timestamp and standalone retrieve discovers the newest.
RUN_ID="${RUN_ID:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --hw) HW="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --context) KUBE_CONTEXT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$NAMESPACE" ]]; then
  echo "ERROR: -n <namespace> required" >&2; exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
# The driver lives under recipe-root/benchmark/. Some files (hw/,
# model-download.yaml, deploy-<config>.yaml) live one level up — at the
# recipe root — because they're "what gets deployed" rather than
# "how we measure it". Compute the recipe root once.
RECIPE_ROOT="$(cd "$HERE/.." && pwd)"

# Per-config metadata. Add a new arm here AND ship a sibling
# ../deploy-<config>.yaml at the recipe root when adding a config.
#   DEPLOY_KIND branches deploy() + clean():
#     deployment → kubectl rollout status + delete Deployment+Service
#     dgd        → kubectl wait on operator-stamped DGD pod labels + delete DGD
case "$CONFIG" in
  vllm-serve)
    DEPLOY_KIND="deployment"
    DEPLOY_NAME="qwen35-vllm-serve"
    BENCH_POD="qwen35-bench"
    # Service the bench Pod hits at http://$FRONTEND:8000.
    BENCH_FRONTEND="qwen35-vllm-serve"
    # Config leaf of the run dir; full path is <ns>/<run-id>/<leaf>
    # (see run_label_* below).
    CONFIG_LABEL="vllm-serve"
    ;;
  dynamo-fd)
    DEPLOY_KIND="dgd"
    DEPLOY_NAME="qwen35-dynamo-fd"
    BENCH_POD="qwen35-fd-bench"
    # The Dynamo operator auto-creates a <dgd-name>-frontend Service.
    BENCH_FRONTEND="qwen35-dynamo-fd-frontend"
    CONFIG_LABEL="dynamo-fd"
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
# Export the per-config knobs so envsubst can substitute them into the
# shared root-level perf.yaml at apply time. BENCH_RUN_LABEL is the
# namespaced + run-id'd sub-path; it depends on RUN_ID so it's resolved
# and exported just-in-time by run_label_write (bench) / run_label_read
# (retrieve) rather than here.
export BENCH_POD BENCH_FRONTEND

HW_ENV="$RECIPE_ROOT/hw/${HW}.env"
if [[ ! -f "$HW_ENV" ]]; then
  echo "ERROR: hardware env file not found: $HW_ENV" >&2
  echo "Available: $(ls "$RECIPE_ROOT/hw/" 2>/dev/null | tr '\n' ' ')" >&2
  exit 2
fi
DEPLOY_FILE="$RECIPE_ROOT/deploy-${CONFIG}.yaml"
if [[ ! -f "$DEPLOY_FILE" ]]; then
  echo "ERROR: deploy template not found: $DEPLOY_FILE" >&2
  exit 2
fi

if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: envsubst missing. Install gettext-base (apt) or gettext (brew)." >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a; . "$HW_ENV"; set +a
echo "[hw]     $HW → image=$VLLM_IMAGE node=$HW_NODE_SELECTOR"
echo "[config] $CONFIG → kind=$DEPLOY_KIND deploy=$DEPLOY_NAME bench-pod=$BENCH_POD"

if [[ -n "$KUBE_CONTEXT" ]]; then
  K="kubectl --context=$KUBE_CONTEXT -n $NAMESPACE"
  echo "[ctx]    pinned to $KUBE_CONTEXT"
else
  K="kubectl -n $NAMESPACE"
  echo "[ctx]    using current-context $(kubectl config current-context 2>/dev/null || echo '<none>')"
fi
# Limit envsubst to our own hw + per-config vars so the embedded shell
# vars inside perf.yaml's inline bash (${MODEL_NAME}, ${KEEP_INPUTS_JSON:-},
# ${FRONTEND}, ${RUN_LABEL}, ${RUN_DIR}, ${AIPERF_GIT_REF}, ...) stay
# literal. The bash-side vars are resolved at runtime inside the Pod.
TPL_VARS='$VLLM_IMAGE $HW_NODE_SELECTOR $HW_TOLERATIONS $BENCH_POD $BENCH_FRONTEND $BENCH_RUN_LABEL'
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
  $K apply -f "$RECIPE_ROOT/model-download.yaml"
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
  $K apply -f "$HERE/data-gen-job.yaml"
  $K wait --for=condition=Complete job/qwen35-generate-datasets --timeout=1800s
  $K logs job/qwen35-generate-datasets | tail -20
}

# ---------------- run-dir isolation ----------------
# Every run writes to its own dir on the shared FSx PVC, keyed by
# namespace + run-id:
#   /perf-cache/artifacts/qwen35_fp8/<namespace>/<run-id>/<config>/
# This is load-bearing: the per-namespace `shared-model-cache` PVCs are
# distinct PV names over ONE FSx Lustre filesystem, so a fixed
# `.../<config>/` path lets two runs — even in different namespaces —
# read and clobber each other's artifacts. The namespace + run-id prefix
# gives each run its own dir. RUN_ID resolution:
#   - explicit --run-id / $RUN_ID wins. The orchestrator pins one id so
#     bench (writer) and retrieve (reader) agree across their separate
#     run-benchmark.sh invocations, and both configs group under it.
#   - bench with no id mints a fresh timestamp.
#   - retrieve with no id discovers the newest run for this
#     (namespace, config) on the still-running bench pod.
run_label_write() {
  [[ -z "$RUN_ID" ]] && RUN_ID="$(date +%Y%m%d-%H%M%S)"
  export BENCH_RUN_LABEL="${NAMESPACE}/${RUN_ID}/${CONFIG_LABEL}"
  echo "[run]    run-id=$RUN_ID label=$BENCH_RUN_LABEL"
}
run_label_read() {
  if [[ -z "$RUN_ID" ]]; then
    # ls -1t sorts newest-first; pick the first run-id that has this
    # config's sub-dir. Args are passed positionally to avoid quoting
    # the namespace path into the remote shell.
    local nsbase="/perf-cache/artifacts/qwen35_fp8/${NAMESPACE}"
    RUN_ID="$($K exec "$BENCH_POD" -- sh -c '
      for rid in $(ls -1t "$1" 2>/dev/null); do
        [ -d "$1/$rid/$2" ] && { echo "$rid"; break; }
      done' _ "$nsbase" "$CONFIG_LABEL" 2>/dev/null)"
    if [[ -z "$RUN_ID" ]]; then
      echo "ERROR: no run dir under $nsbase/*/$CONFIG_LABEL — pass --run-id." >&2
      exit 2
    fi
    echo "[retrieve] discovered newest run-id: $RUN_ID"
  fi
  export BENCH_RUN_LABEL="${NAMESPACE}/${RUN_ID}/${CONFIG_LABEL}"
}

# ---------------- config-specific lifecycle ----------------

deploy() {
  APPLY_TPL "$DEPLOY_FILE"
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
      # The Dynamo operator takes 5-10s to reconcile a DGD into pods —
      # `kubectl wait -l ...` issued immediately after `apply` returns
      # "no matching resources found" because the pods don't exist yet.
      # Poll for pod materialization (up to 60s) before waiting on Ready.
      echo "[deploy] waiting for operator to materialize DGD pods ..."
      for _ in $(seq 1 30); do
        [[ $($K get pod -l "$sel_fe" --no-headers 2>/dev/null | wc -l) -gt 0 ]] && break
        sleep 2
      done
      echo "[deploy] waiting for DGD Frontend pod Ready ..."
      $K wait --for=condition=Ready pod -l "$sel_fe" --timeout=900s
      echo "[deploy] waiting for VllmWorker pod (397B FP8 cold load can take 30+ min) ..."
      $K wait --for=condition=Ready pod -l "$sel_wk" --timeout=2700s
      ;;
    *)
      echo "ERROR: unknown DEPLOY_KIND=$DEPLOY_KIND" >&2; exit 2 ;;
  esac
}

bench() {
  run_label_write
  $K delete pod "$BENCH_POD" --ignore-not-found
  # Shared bench template at benchmark/perf.yaml. Per-config knobs
  # (pod name, frontend service) come from envsubst on $BENCH_POD /
  # $BENCH_FRONTEND (set by the case statement near the top); the
  # run-isolated $BENCH_RUN_LABEL is set by run_label_write above.
  APPLY_TPL "$HERE/perf.yaml"
  $K wait --for=condition=Ready "pod/$BENCH_POD" --timeout=300s
  # Clear any stale .aiperf-done sentinel BEFORE we start polling. The
  # run-isolated <namespace>/<run-id>/<config> path makes a cross-run
  # stale sentinel nearly impossible, but a re-invocation that reuses
  # the same --run-id (e.g. a retried bench) would still see the prior
  # tick's sentinel and exit "done" before the fresh aiperf finishes its
  # pip install. This rm keeps that case honest — cheap safety net.
  local sentinel="/perf-cache/artifacts/qwen35_fp8/${BENCH_RUN_LABEL}/.aiperf-done"
  $K exec "$BENCH_POD" -- rm -f "$sentinel" 2>/dev/null || true
  echo "[bench] pod Ready; polling for .aiperf-done sentinel (max ~50 min)"
  # Don't trust `kubectl logs -f` as a synchronization primitive — under
  # transient teleport / network hiccups it exits early and we'd race
  # past into retrieve+clean with empty artifacts (lost a vllm-serve
  # run that way once). The bench Pod writes a /perf-cache/artifacts/.../
  # .aiperf-done sentinel after aiperf finishes and the inputs.json
  # cleanup runs (see perf.yaml). Poll for it via `kubectl exec ls`.
  local deadline=$((SECONDS + 3000))   # 50 min
  while (( SECONDS < deadline )); do
    if $K exec "$BENCH_POD" -- test -f "$sentinel" 2>/dev/null; then
      echo "[bench] sentinel observed at $sentinel — aiperf done"
      return 0
    fi
    # Surface progress every minute by tailing the pod's stdout. This
    # is logging-only — the synchronization is the sentinel poll above.
    $K logs "$BENCH_POD" --tail=3 2>/dev/null || true
    sleep 60
  done
  echo "ERROR: bench sentinel $sentinel did not appear within 50 min." >&2
  echo "       Pod is left running for inspection; not running retrieve." >&2
  exit 2
}

retrieve() {
  run_label_read
  # Override the destination root via $BENCHMARK_RESULTS_DIR if your
  # workspace layout differs from the default.
  local base="${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}"
  local dest="$base/$(date +%m-%d)/qwen35-fp8-${HW}/${CONFIG}"
  mkdir -p "$dest/logs"
  # `kubectl exec -- tar c | tar x` is fragile across Teleport — under
  # transient network hiccups the exec stream returns "i/o timeout"
  # mid-stream and the local tar receives a truncated archive. Use
  # `kubectl cp` on the specific files instead (per convention/k8s.md):
  # the bench writes a fixed set of small artifacts (~few MB total
  # without inputs.json), each retried independently.
  local src="/perf-cache/artifacts/qwen35_fp8/${BENCH_RUN_LABEL}"
  echo "[retrieve] remote src: $src"
  for f in profile_export_aiperf.json profile_export_aiperf.csv \
           server_metrics_export.json server_metrics_export.csv \
           profile_export.jsonl; do
    $K cp "$BENCH_POD:$src/$f" "$dest/$f"
  done
  # The logs/aiperf.log is optional — present on most runs but absent
  # if aiperf crashed before writing it. Don't fail retrieve over it.
  $K cp "$BENCH_POD:$src/logs/aiperf.log" "$dest/logs/aiperf.log" 2>/dev/null || true
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
      # Use the fully-qualified resource path. The bare `dgd` /
      # `dynamographdeployment` shortname resolves to whichever API
      # version `kubectl` discovers first, which can be a not-yet-
      # served v1beta1 on operators that announced v1beta1 in their
      # APIService but haven't actually deployed it — `delete dgd`
      # then returns "not found" while the resource still exists at
      # v1alpha1 (observed on aws-dev-02 today, leaking an 8-GPU
      # qwen35-dynamo-fd DGD that --ignore-not-found silently
      # swallowed). Targeting v1alpha1 explicitly + verifying gone.
      $K delete dynamographdeployments.v1alpha1.nvidia.com "$DEPLOY_NAME" --ignore-not-found --wait=false
      # Verify the DGD actually went away — list via v1alpha1 path too.
      for _ in 1 2 3 4 5 6; do
        if ! $K get dynamographdeployments.v1alpha1.nvidia.com "$DEPLOY_NAME" >/dev/null 2>&1; then
          echo "[clean] DGD $DEPLOY_NAME removed"
          break
        fi
        sleep 5
      done
      if $K get dynamographdeployments.v1alpha1.nvidia.com "$DEPLOY_NAME" >/dev/null 2>&1; then
        echo "ERROR: DGD $DEPLOY_NAME still present after delete — manual cleanup required." >&2
        exit 2
      fi
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
