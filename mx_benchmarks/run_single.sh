#!/usr/bin/env bash
# Deploy a single-worker DGD (frontend + 1 worker) and run aiperf benchmark.
# No mx-target, no P2P, no scale-up — just baseline serving.
#
# Usage:
#   ./run_single.sh          # Deploy, benchmark, collect results
#   ./run_single.sh clean    # Clean up all resources for this RUN_ID
#
# Environment variables:
#   NAMESPACE        Kubernetes namespace (default: hwoo)
#   RUN_ID           Unique ID for resource isolation (default: default)
#   MODEL_NAME       HuggingFace model ID (default: meta-llama/Llama-3.3-70B-Instruct)
#   DATASET          Trace file name without .jsonl (default: staircase_trace)
#   SLICE_DURATION   Timeslice duration in seconds (default: 60)
#   RUN_TAG          Label for this run (default: timestamp YYYYMMDD_HHMMSS)
#   STAT             Statistic to plot: avg, p50, p90, p99, min, max (default: avg)
#   TP_SIZE          Tensor parallel size (default: 8)
#   DP_SIZE          Data parallel size (default: 1)
#   ENABLE_EP        Set to 1 to add --enable-expert-parallel
#   DYNAMO_IMAGE_TAG     Dynamo runtime image tag (default: dynamo-mx-runtime)
#   MX_SERVER_IMAGE_TAG  ModelExpress server image tag (default: 64f9249)
#   VLLM_EXTRA_ARGS  Space-separated extra vLLM args
#   VLLM_USE_DEEP_GEMM          Set to 0 to disable DeepGEMM
#   VLLM_USE_FLASHINFER_MOE_FP8 Set to 0 to disable FlashInfer MoE FP8
#   CLEANUP_ON_CRASH Set to 1 to auto-cleanup on crash (default: 0, leave pods)
#   KUBE_CONTEXT     Kubernetes context to use
#
# Dataset generation:
#   INPUT_TRACE      Base trace for synthesis (default: conversation_trace.jsonl)
#   MAX_ISL / MAX_OSL    Max sequence length filters (default: 8000)
#   STEP_RATES       Comma-separated rate values (default: 4,4,10,12,12,10,4,4)
#   STEP_DURATIONS   Comma-separated per-step durations (default: 60,60,60,60,60,60,60,60)
#   CONSTANT_RATE    Set to 1 to use constant-rate scheduling
set -euo pipefail

MODE="${1:-run}"
NAMESPACE="${NAMESPACE:-hwoo}"
RUN_ID="${RUN_ID:-default}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}"
DATASET="${DATASET:-staircase_trace}"
SLICE_DURATION="${SLICE_DURATION:-60}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAT="${STAT:-avg}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
_TP=$((TP_SIZE > 0 ? TP_SIZE : 1))
_DP=$((DP_SIZE > 0 ? DP_SIZE : 1))
TOTAL_GPUS=$((_TP * _DP))
RDMA_RESOURCE="${RDMA_RESOURCE:-rdma/ib}"
RDMA_COUNT="${RDMA_COUNT:-$TOTAL_GPUS}"
ENABLE_EP="${ENABLE_EP:-}"
DYNAMO_IMAGE_TAG="${DYNAMO_IMAGE_TAG:-dynamo-mx-runtime}"
MX_SERVER_IMAGE_TAG="${MX_SERVER_IMAGE_TAG:-64f9249}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-1}"
CLEANUP_ON_CRASH="${CLEANUP_ON_CRASH:-0}"
INPUT_TRACE="${INPUT_TRACE:-conversation_trace.jsonl}"
MAX_ISL="${MAX_ISL:-8000}"
MAX_OSL="${MAX_OSL:-8000}"
STEP_RATES="${STEP_RATES:-4,4,10,12,12,10,4,4}"
STEP_DURATIONS="${STEP_DURATIONS:-60,60,60,60,60,60,60,60}"

export NAMESPACE RUN_ID TP_SIZE DP_SIZE DYNAMO_IMAGE_TAG RDMA_RESOURCE RDMA_COUNT

if [[ "$MODE" != "run" && "$MODE" != "clean" && "$MODE" != "dry-run" ]]; then
  echo "Usage: $0 {run|clean|dry-run}"
  exit 1
fi

# Isolate kubeconfig for parallel cluster support
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
  _KUBECONFIG_COPY=$(mktemp "/tmp/kubeconfig-${RUN_ID}-XXXXXX")
  cp "${KUBECONFIG:-$HOME/.kube/config}" "$_KUBECONFIG_COPY"
  export KUBECONFIG="$_KUBECONFIG_COPY"
  kubectl config use-context "$KUBE_CONTEXT" >/dev/null
  echo "Using kube context: $KUBE_CONTEXT (isolated kubeconfig: $KUBECONFIG)"
fi

# -- Model -> PVC mapping --
declare -A MODEL_PVC_MAP=(
  ["meta-llama/Llama-3.3-70B-Instruct"]="model-cache-llama"
  ["Qwen/Qwen3-32B"]="model-cache-qwen3-32b"
  ["Qwen/Qwen3-0.6B"]="model-cache-qwen3-06b"
  ["deepseek-ai/DeepSeek-V3.2"]="model-cache-dsv3"
  ["deepseek-ai/DeepSeek-V3"]="model-cache-dsv3-0"
  ["moonshotai/Kimi-K2.5"]="model-cache-kimi-k25"
)

PVC_NAME="${MODEL_PVC_MAP[$MODEL_NAME]:-}"
if [[ -z "$PVC_NAME" ]]; then
  echo "ERROR: No PVC mapping for model '$MODEL_NAME'"
  for m in "${!MODEL_PVC_MAP[@]}"; do echo "  $m -> ${MODEL_PVC_MAP[$m]}"; done
  exit 1
fi

DEFAULT_MODEL="meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PVC="model-cache-llama"
DEFAULT_DYNAMO_TAG="dynamo-mx-runtime"
DEFAULT_MX_TAG="64f9249"
MODEL_BASE="${MODEL_NAME##*/}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE_FILE="${REPO_ROOT}/benchmarks/prefix_data_generator/${DATASET}.jsonl"
LOCAL_ARTIFACTS="${REPO_ROOT}/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}"

DGD_NAME="vllm-agg-single-${RUN_ID}"
ROLE_LABEL="dynamo/${RUN_ID}-role"
IMAGE_CACHE_LABEL="dynamo/${RUN_ID}-image-cache"
BENCHMARK_POD="benchmark-${RUN_ID}"
MX_SERVER_NAME="modelexpress-${RUN_ID}"
DAEMONSET_NAME="image-cache-${RUN_ID}"
FRONTEND="${DGD_NAME}-frontend"

source "${REPO_ROOT}/.venv/bin/activate"

# -- Generate dataset if needed --
if [[ "$MODE" != "clean" ]] && [[ ! -f "$TRACE_FILE" ]]; then
  DATAGEN_DIR="${REPO_ROOT}/benchmarks/prefix_data_generator"
  INPUT_TRACE_PATH="${DATAGEN_DIR}/${INPUT_TRACE}"
  if [[ ! -f "$INPUT_TRACE_PATH" ]]; then
    echo "ERROR: Base trace not found: $INPUT_TRACE_PATH"
    exit 1
  fi
  echo "Generating dataset: ${DATASET}.jsonl"
  (cd "$DATAGEN_DIR" && datagen synthesize \
    --input-file "$INPUT_TRACE" \
    --max-isl "$MAX_ISL" \
    --max-osl "$MAX_OSL" \
    --step-durations "$STEP_DURATIONS" \
    --step-rates "$STEP_RATES" \
    ${CONSTANT_RATE:+--constant-rate} \
    --output-file "${DATASET}.jsonl")
  [[ ! -f "$TRACE_FILE" ]] && { echo "ERROR: Dataset generation failed"; exit 1; }
fi

cd "${REPO_ROOT}/mx_benchmarks"

echo "=== Configuration (single worker) ==="
echo "  Namespace: $NAMESPACE"
echo "  RUN_ID:    $RUN_ID"
echo "  Model:     $MODEL_NAME"
echo "  Image tag: $DYNAMO_IMAGE_TAG"
echo "  PVC:       $PVC_NAME"
echo "  Dataset:   $DATASET"
echo "  Tag:       $RUN_TAG"
echo "  GPUs/worker: $TOTAL_GPUS (TP=$TP_SIZE x DP=$DP_SIZE)"
[[ "$ENABLE_EP" == "1" ]] && echo "  Expert parallel: enabled"
echo "===================="

# -- Helper: sed model/image/label substitutions --
model_sed() {
  local extra_insert=""
  if [[ "$ENABLE_EP" == "1" ]]; then
    extra_insert="${extra_insert}            - --enable-expert-parallel\n"
  fi
  if [[ "$DP_SIZE" -gt 1 ]]; then
    extra_insert="${extra_insert}            - --data-parallel-size\n"
    extra_insert="${extra_insert}            - \"${DP_SIZE}\"\n"
  fi
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    for arg in $VLLM_EXTRA_ARGS; do
      if [[ "$arg" == --* ]]; then
        extra_insert="${extra_insert}            - ${arg}\n"
      else
        extra_insert="${extra_insert}            - \"${arg}\"\n"
      fi
    done
  fi
  if [[ "$MODEL_NAME" == *"Kimi-K2.5"* ]]; then
    extra_insert="${extra_insert}            - --dyn-reasoning-parser\n"
    extra_insert="${extra_insert}            - kimi_k25\n"
    extra_insert="${extra_insert}            - --dyn-tool-call-parser\n"
    extra_insert="${extra_insert}            - kimi_k2\n"
  fi
  local extra_sed=""
  if [[ -n "$extra_insert" ]]; then
    extra_sed="/^ *- --load-format/i\\${extra_insert%\\n}"
  fi

  local kernel_env_sed=""
  if [[ "$VLLM_USE_DEEP_GEMM" == "0" ]]; then
    kernel_env_sed="${kernel_env_sed}        - name: VLLM_USE_DEEP_GEMM\n          value: \"0\"\n"
  fi
  if [[ "$VLLM_USE_FLASHINFER_MOE_FP8" == "0" ]]; then
    kernel_env_sed="${kernel_env_sed}        - name: VLLM_USE_FLASHINFER_MOE_FP8\n          value: \"0\"\n"
  fi
  local kernel_sed=""
  if [[ -n "$kernel_env_sed" ]]; then
    kernel_sed="/value: \"7200000\"/a\\${kernel_env_sed%\\n}"
  fi

  local tp_sed=""
  if [[ "$TP_SIZE" -eq 0 ]]; then
    tp_sed='/--tensor-parallel-size/,+1d'
  else
    tp_sed="s|- \"8\"|- \"$TP_SIZE\"|g"
  fi

  sed -e "s|$DEFAULT_MODEL|$MODEL_NAME|g" \
      -e "s|$DEFAULT_PVC|$PVC_NAME|g" \
      -e "s|:${DEFAULT_DYNAMO_TAG}|:${DYNAMO_IMAGE_TAG}|g" \
      -e "s|:${DEFAULT_MX_TAG}|:${MX_SERVER_IMAGE_TAG}|g" \
      -e "s|gpu: \"8\"|gpu: \"$TOTAL_GPUS\"|g" \
      -e "s|dynamo/role|${ROLE_LABEL}|g" \
      -e "s|dynamo/reserved|dynamo/${RUN_ID}-reserved|g" \
      -e "s|\([^/]\)modelexpress-server|\1${MX_SERVER_NAME}|g" \
      -e "s|//modelexpress-server|//${MX_SERVER_NAME}|g" \
      -e "s|^modelexpress-server|${MX_SERVER_NAME}|g" \
      -e "s|name: agg-mx-p2p-benchmark|name: ${BENCHMARK_POD}|g" \
      ${extra_sed:+-e "$extra_sed"} \
      ${kernel_sed:+-e "$kernel_sed"}
}

save_logs() {
  local logs_dir="${LOCAL_ARTIFACTS}/single/logs"
  mkdir -p "$logs_dir"
  echo "Collecting logs..."
  for svc in worker frontend; do
    local pod
    pod=$(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" \
      | grep "${DGD_NAME}.*${svc}" | head -1 || true)
    if [[ -z "$pod" ]]; then
      echo "  $svc: no pod found, skipping"
      continue
    fi
    echo "  $svc: $pod"
    kubectl logs "$pod" -n "$NAMESPACE" --all-containers 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/${svc}.log" || true
  done
  if kubectl get pod "$BENCHMARK_POD" -n "$NAMESPACE" &>/dev/null; then
    echo "  aiperf: $BENCHMARK_POD"
    kubectl logs "$BENCHMARK_POD" -n "$NAMESPACE" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/aiperf.log" || true
    kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- \
      tmux capture-pane -t benchmark -p -S - 2>/dev/null \
      | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/aiperf_tmux.log" || true
    [[ ! -s "$logs_dir/aiperf_tmux.log" ]] && rm -f "$logs_dir/aiperf_tmux.log"
  fi
  kubectl get "dgd/$DGD_NAME" -n "$NAMESPACE" -o yaml > "$logs_dir/dgd.yaml" 2>&1 || true
}

fetch_artifacts() {
  local remote_path="/perf-cache/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/single"
  local local_path="${LOCAL_ARTIFACTS}/single"
  if ! kubectl get pod "$BENCHMARK_POD" -n "$NAMESPACE" &>/dev/null; then
    echo "WARNING: Benchmark pod not found, skipping artifact fetch"
    return
  fi
  if ! kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- test -d "$remote_path" 2>/dev/null; then
    echo "WARNING: $remote_path does not exist on PVC"
    return
  fi
  mkdir -p "$local_path"
  echo "Fetching aiperf artifacts..."
  local files
  files=$(kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- \
    find "$remote_path" -maxdepth 1 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.log" -o -name "*.jsonl" \) \
    2>/dev/null || true)
  for f in $files; do
    local fname
    fname=$(basename "$f")
    echo "  $fname"
    kubectl cp "$NAMESPACE/${BENCHMARK_POD}:$f" "$local_path/$fname" 2>/dev/null || true
  done
}

clean_all() {
  echo "=== Cleaning up resources for RUN_ID=$RUN_ID ==="
  kubectl delete pod "$BENCHMARK_POD" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "dgd/$DGD_NAME" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "service/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "daemonset/${DAEMONSET_NAME}" -n "$NAMESPACE" --ignore-not-found --wait

  local taint_key="dynamo/${RUN_ID}-reserved"
  local tainted_nodes
  tainted_nodes=$(kubectl get nodes -l "$ROLE_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  for node in $tainted_nodes; do
    kubectl taint nodes "$node" "${taint_key}-" 2>/dev/null || true
  done

  local stale_nodes
  stale_nodes=$(kubectl get nodes -l "$ROLE_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  for node in $stale_nodes; do
    kubectl label node "$node" "${ROLE_LABEL}-" 2>/dev/null || true
  done
  stale_nodes=$(kubectl get nodes -l "$IMAGE_CACHE_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  for node in $stale_nodes; do
    kubectl label node "$node" "${IMAGE_CACHE_LABEL}-" 2>/dev/null || true
  done
  echo "Cleanup done."
}

# -- Main --

if [[ "$MODE" == "clean" ]]; then
  clean_all
  exit 0
fi

if [[ "$MODE" == "dry-run" ]]; then
  outdir="${LOCAL_ARTIFACTS}/dry-run"
  mkdir -p "$outdir"
  echo "===== Dry run: generating YAMLs to $outdir ====="

  strip_run_tolerations() {
    sed '/nodeSelector:/,+1d' \
    | sed "\\|dynamo/${RUN_ID}-reserved|,+3d" \
    | sed '/^[[:space:]]*tolerations:[[:space:]]*$/{N; /tolerations:\n[[:space:]]*$/d}'
  }

  dgd_out="$outdir/dgd-single.yaml"
  model_sed < agg_single.yaml \
    | sed "s|name: vllm-agg-single|name: ${DGD_NAME}|g" \
    | strip_run_tolerations \
    > "$dgd_out"
  echo "  $dgd_out ($(wc -l < "$dgd_out") lines)"

  perf_out="$outdir/perf.yaml"
  model_sed < perf.yaml \
    | sed -e "s/value: p2p/value: single/" \
          -e "s/value: vllm-agg-mx-p2p-frontend/value: $FRONTEND/" \
          -e "s/value: staircase_trace/value: $DATASET/" \
          -e "s/value: \"60\"/value: \"$SLICE_DURATION\"/" \
          -e "s/value: default_run/value: $RUN_TAG/" \
    > "$perf_out"
  echo "  $perf_out ($(wc -l < "$perf_out") lines)"

  mx_out="$outdir/modelexpress.yaml"
  model_sed < modelexpress-server.yaml | strip_run_tolerations > "$mx_out"
  echo "  $mx_out ($(wc -l < "$mx_out") lines)"
  echo "===== Done ====="
  exit 0
fi

clean_all
trap clean_all EXIT

# Setup node + image cache (single node only)
./setup-image-cache.sh source

TAINT_KEY="dynamo/${RUN_ID}-reserved"
echo "Tainting source node with ${TAINT_KEY}=source:NoSchedule..."
kubectl taint nodes -l "${ROLE_LABEL}=source" "${TAINT_KEY}=source:NoSchedule"

# Deploy modelexpress server
model_sed < modelexpress-server.yaml | kubectl apply -n "$NAMESPACE" -f -
kubectl rollout status "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --timeout=300s

# Deploy DGD
dgd_tmp="/tmp/dgd-${RUN_ID}-single.yaml"
model_sed < agg_single.yaml \
  | sed "s|name: vllm-agg-single|name: ${DGD_NAME}|g" \
  > "$dgd_tmp"
echo "DGD yaml saved to $dgd_tmp ($(wc -l < "$dgd_tmp") lines)"
kubectl apply -n "$NAMESPACE" -f "$dgd_tmp"

# Wait for DGD ready
echo "Waiting for DGD $DGD_NAME to be ready..."
wait_elapsed=0
while true; do
  ready=$(kubectl get "dgd/$DGD_NAME" -n "$NAMESPACE" \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
  if [[ "$ready" == "True" ]]; then
    echo "  DGD $DGD_NAME is ready."
    break
  fi

  # Check for crashes
  failed_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers \
    | grep "$DGD_NAME" \
    | awk '$3 ~ /CrashLoopBackOff|Error|OOMKilled|ImagePullBackOff/ || ($4+0 > 0) {print $1, $3, "restarts="$4}' || true)
  if [[ -n "$failed_pods" ]]; then
    echo "ERROR: Crashed pods detected:"
    echo "$failed_pods"
    if [[ "$CLEANUP_ON_CRASH" == "1" ]]; then
      save_logs
    else
      echo "Leaving pods for investigation (set CLEANUP_ON_CRASH=1 to auto-cleanup)"
    fi
    exit 1
  fi

  sleep 10
  wait_elapsed=$((wait_elapsed + 10))
  if [[ "$wait_elapsed" -ge 1800 ]]; then
    echo "ERROR: Timed out waiting for DGD after ${wait_elapsed}s"
    exit 1
  fi
  if (( wait_elapsed % 60 == 0 )); then
    echo "  waiting... (${wait_elapsed}s)"
  fi
done

# Deploy benchmark pod (needed for trace upload — PVC is writable here)
perf_tmp="/tmp/perf-${RUN_ID}.yaml"
model_sed < perf.yaml \
  | sed -e "s/value: p2p/value: single/" \
        -e "s/value: vllm-agg-mx-p2p-frontend/value: $FRONTEND/" \
        -e "s/value: staircase_trace/value: $DATASET/" \
        -e "s/value: \"60\"/value: \"$SLICE_DURATION\"/" \
        -e "s/value: default_run/value: $RUN_TAG/" \
  > "$perf_tmp"
echo "Perf yaml saved to $perf_tmp ($(wc -l < "$perf_tmp") lines)"
kubectl apply -n "$NAMESPACE" -f "$perf_tmp"

# Upload trace via benchmark pod (worker container may lack PVC write permission)
echo "Waiting for benchmark pod to be ready..."
kubectl wait "pod/$BENCHMARK_POD" -n "$NAMESPACE" --for=condition=Ready --timeout=300s
echo "Uploading trace to ${BENCHMARK_POD}:/perf-cache/traces/${DATASET}.jsonl ..."
kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- mkdir -p /perf-cache/traces
kubectl cp "$TRACE_FILE" "$NAMESPACE/${BENCHMARK_POD}:/perf-cache/traces/${DATASET}.jsonl"

# Wait for aiperf to finish
echo "Waiting for aiperf to finish..."
timeslice_path="/perf-cache/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/single/profile_export_aiperf_timeslices.json"
elapsed=0
timeout=3600
while true; do
  pod_phase=$(kubectl get pod "$BENCHMARK_POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [[ "$pod_phase" == "Succeeded" || "$pod_phase" == "Failed" ]]; then
    echo "  benchmark pod completed (phase=$pod_phase) after ~${elapsed}s"
    break
  fi
  if kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- test -f "$timeslice_path" 2>/dev/null; then
    echo "  aiperf finished after ~${elapsed}s"
    break
  fi
  sleep 30
  elapsed=$((elapsed + 30))
  if [[ "$elapsed" -ge "$timeout" ]]; then
    echo "WARNING: Timed out waiting for aiperf after ${timeout}s"
    break
  fi
  if (( elapsed % 60 == 0 )); then
    echo "  waiting... (${elapsed}s)"
  fi
done

# Collect logs and artifacts
save_logs
fetch_artifacts

# Cleanup PVC artifacts
remote_dir="/perf-cache/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/single"
kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- rm -rf "$remote_dir" 2>/dev/null || true

echo ""
echo "===== Done ====="
echo "Artifacts: ${LOCAL_ARTIFACTS}/single"
