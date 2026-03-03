#!/usr/bin/env bash
# Deploy and benchmark MX P2P vs disk loading with planner-driven scale-up.
#
# Unlike run.sh which uses manual sleep + scaleup_bench.py, this script
# deploys a Planner service that automatically scales mx-target when
# the frontend detects SLA breaches (TTFT spikes).
#
# Usage:
#   ./run_planner.sh              # Run both p2p and disk
#   ./run_planner.sh p2p          # Run only p2p
#   ./run_planner.sh disk         # Run only disk
#   ./run_planner.sh both         # Same as no argument
#   ./run_planner.sh clean        # Clean up all resources for this RUN_ID
#   ./run_planner.sh plot         # Re-generate comparison plot from existing artifacts
#   ./run_planner.sh table        # Print mx-target startup timing table from existing logs
#
# Environment variables (inherits all from run.sh, plus):
#   TTFT_TARGET          TTFT SLA target in ms (default: 2000)
#   MIN_REPLICAS         Minimum mx-target replicas (default: 0)
#   MAX_REPLICAS         Maximum mx-target replicas (default: 2)
#   PLANNER_INTERVAL     Load-based adjustment interval in seconds (default: 5)
#
# See run.sh header for the full list of shared environment variables.
set -euo pipefail

MODE="${1:-both}"
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
ENABLE_TORCH_COMPILE_CACHE="${ENABLE_TORCH_COMPILE_CACHE:-1}"
DEEPGEMM_SOURCE_WARMUP_MODE="${DEEPGEMM_SOURCE_WARMUP_MODE:-full}"
DEEPGEMM_TARGET_WARMUP_MODE="${DEEPGEMM_TARGET_WARMUP_MODE:-skip}"
ENABLE_REQUEST_LOGGING="${ENABLE_REQUEST_LOGGING:-0}"
INPUT_TRACE="${INPUT_TRACE:-conversation_trace.jsonl}"
MAX_ISL="${MAX_ISL:-8000}"
MAX_OSL="${MAX_OSL:-8000}"
STEP_RATES="${STEP_RATES:-4,4,10,12,12,10,4,4}"
STEP_DURATIONS="${STEP_DURATIONS:-60,60,60,60,60,60,60,60}"

# Planner-specific settings
TTFT_TARGET="${TTFT_TARGET:-2000}"
MIN_REPLICAS="${MIN_REPLICAS:-0}"
MAX_REPLICAS="${MAX_REPLICAS:-2}"
PLANNER_INTERVAL="${PLANNER_INTERVAL:-5}"

# Export for setup-image-cache.sh
export NAMESPACE RUN_ID TP_SIZE DP_SIZE DYNAMO_IMAGE_TAG RDMA_RESOURCE RDMA_COUNT

# Isolate kubeconfig for parallel cluster support
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
  _KUBECONFIG_COPY=$(mktemp "/tmp/kubeconfig-${RUN_ID}-XXXXXX")
  cp "${KUBECONFIG:-$HOME/.kube/config}" "$_KUBECONFIG_COPY"
  export KUBECONFIG="$_KUBECONFIG_COPY"
  kubectl config use-context "$KUBE_CONTEXT" >/dev/null
  echo "Using kube context: $KUBE_CONTEXT (isolated kubeconfig: $KUBECONFIG)"
fi

if [[ "$MODE" != "p2p" && "$MODE" != "disk" && "$MODE" != "both" && "$MODE" != "clean" && "$MODE" != "plot" && "$MODE" != "table" ]]; then
  echo "Usage: $0 {p2p|disk|both|clean|plot|table}"
  exit 1
fi

# -- Model -> PVC mapping (add new models here) --
declare -A MODEL_PVC_MAP=(
  ["meta-llama/Llama-3.3-70B-Instruct"]="model-cache-llama"
  ["Qwen/Qwen3-32B"]="model-cache-qwen3-32b"
  ["deepseek-ai/DeepSeek-V3.2"]="model-cache-dsv3"
  ["deepseek-ai/DeepSeek-V3"]="model-cache-dsv3-0"
  ["moonshotai/Kimi-K2.5"]="model-cache-kimi-k25"
)

PVC_NAME="${MODEL_PVC_MAP[$MODEL_NAME]:-}"
if [[ -z "$PVC_NAME" ]]; then
  echo "ERROR: No PVC mapping for model '$MODEL_NAME'"
  echo "Known models:"
  for m in "${!MODEL_PVC_MAP[@]}"; do
    echo "  $m -> ${MODEL_PVC_MAP[$m]}"
  done
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

# RUN_ID-scoped resource names (for parallel run isolation)
ROLE_LABEL="dynamo/${RUN_ID}-role"
IMAGE_CACHE_LABEL="dynamo/${RUN_ID}-image-cache"
BENCHMARK_POD="benchmark-${RUN_ID}"
MX_SERVER_NAME="modelexpress-${RUN_ID}"
DAEMONSET_NAME="image-cache-${RUN_ID}"

source "${REPO_ROOT}/.venv/bin/activate"

# -- Generate dataset if it doesn't exist (skip for clean mode) --
if [[ "$MODE" != "clean" ]] && [[ ! -f "$TRACE_FILE" ]]; then
  DATAGEN_DIR="${REPO_ROOT}/benchmarks/prefix_data_generator"
  INPUT_TRACE_PATH="${DATAGEN_DIR}/${INPUT_TRACE}"
  if [[ ! -f "$INPUT_TRACE_PATH" ]]; then
    echo "ERROR: Base trace not found: $INPUT_TRACE_PATH"
    exit 1
  fi
  echo "Dataset not found, generating: ${DATASET}.jsonl"
  echo "  Base trace: $INPUT_TRACE"
  echo "  Max ISL: $MAX_ISL | Max OSL: $MAX_OSL"
  echo "  Step rates: $STEP_RATES | Step durations: $STEP_DURATIONS"
  (cd "$DATAGEN_DIR" && datagen synthesize \
    --input-file "$INPUT_TRACE" \
    --max-isl "$MAX_ISL" \
    --max-osl "$MAX_OSL" \
    --step-durations "$STEP_DURATIONS" \
    --step-rates "$STEP_RATES" \
    ${CONSTANT_RATE:+--constant-rate} \
    --output-file "${DATASET}.jsonl")
  if [[ ! -f "$TRACE_FILE" ]]; then
    echo "ERROR: Dataset generation failed"
    exit 1
  fi
  echo "Dataset generated: $TRACE_FILE"
fi

cd "${REPO_ROOT}/mx_benchmarks"

echo "=== Configuration (planner mode) ==="
echo "  Mode:      $MODE"
echo "  Namespace: $NAMESPACE"
echo "  RUN_ID:    $RUN_ID"
echo "  Model:     $MODEL_NAME"
echo "  Image tag: $DYNAMO_IMAGE_TAG"
echo "  PVC:       $PVC_NAME"
echo "  Dataset:   $DATASET"
echo "  Tag:       $RUN_TAG"
echo "  TP size:   $TP_SIZE$(if [[ "$TP_SIZE" -eq 0 ]]; then echo " (auto)"; fi)"
echo "  DP size:   $DP_SIZE"
echo "  GPUs/worker: $TOTAL_GPUS (TP=$TP_SIZE x DP=$DP_SIZE)"
echo "  --- Planner ---"
echo "  TTFT target:      ${TTFT_TARGET}ms"
echo "  Min replicas:     $MIN_REPLICAS"
echo "  Max replicas:     $MAX_REPLICAS"
echo "  Interval:         ${PLANNER_INTERVAL}s"
[[ "$ENABLE_EP" == "1" ]] && echo "  Expert parallel: enabled"
[[ "$VLLM_USE_DEEP_GEMM" == "0" ]] && echo "  DeepGEMM: disabled"
[[ "$VLLM_USE_FLASHINFER_MOE_FP8" == "0" ]] && echo "  FlashInfer MoE FP8: disabled"
[[ "$ENABLE_TORCH_COMPILE_CACHE" == "0" ]] && echo "  Torch compile cache reuse: disabled"
[[ "$DEEPGEMM_SOURCE_WARMUP_MODE" != "full" ]] && echo "  DeepGEMM source warmup: $DEEPGEMM_SOURCE_WARMUP_MODE"
[[ "$DEEPGEMM_TARGET_WARMUP_MODE" != "skip" ]] && echo "  DeepGEMM target warmup: $DEEPGEMM_TARGET_WARMUP_MODE (independent, no PVC cache)"
[[ "$ENABLE_REQUEST_LOGGING" == "1" ]] && echo "  Request logging: enabled"
echo "===================="

# -- Helper functions --

model_sed() {
  # Build extra args insert string for sed
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
  # Model-specific args
  if [[ "$MODEL_NAME" == *"Kimi-K2.5"* ]]; then
    extra_insert="${extra_insert}            - --dyn-reasoning-parser\n"
    extra_insert="${extra_insert}            - kimi_k2\n"
    extra_insert="${extra_insert}            - --dyn-tool-call-parser\n"
    extra_insert="${extra_insert}            - kimi_k2\n"
  fi
  local extra_sed=""
  if [[ -n "$extra_insert" ]]; then
    extra_sed="/^ *- --load-format/i\\${extra_insert%\\n}"
  fi

  # Inject env vars to disable kernels in worker envs (after VLLM_RPC_TIMEOUT)
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

  # Build TP sed: either replace the value or remove the flag entirely
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
      -e "s|rdma/ib: \"8\"|${RDMA_RESOURCE}: \"$RDMA_COUNT\"|g" \
      -e "$tp_sed" \
      -e "s|dynamo/role|${ROLE_LABEL}|g" \
      -e "s|dynamo/reserved|dynamo/${RUN_ID}-reserved|g" \
      -e "s|\([^/]\)modelexpress-server|\1${MX_SERVER_NAME}|g" \
      -e "s|//modelexpress-server|//${MX_SERVER_NAME}|g" \
      -e "s|^modelexpress-server|${MX_SERVER_NAME}|g" \
      -e "s|name: agg-mx-p2p-benchmark|name: ${BENCHMARK_POD}|g" \
      ${extra_sed:+-e "$extra_sed"} \
      ${kernel_sed:+-e "$kernel_sed"}
}

# Substitute planner-specific values in the --config JSON string
planner_sed() {
  sed -e "s|\"ttft\":2000|\"ttft\":$TTFT_TARGET|g" \
      -e "s|\"min_endpoint\":0|\"min_endpoint\":$MIN_REPLICAS|g" \
      -e "s|\"max_endpoint\":2|\"max_endpoint\":$MAX_REPLICAS|g" \
      -e "s|\"decode_engine_num_gpu\":8|\"decode_engine_num_gpu\":$TOTAL_GPUS|g" \
      -e "s|\"load_adjustment_interval\":5|\"load_adjustment_interval\":$PLANNER_INTERVAL|g"
}

dgd_for_mode() {
  local m="$1"
  if [[ "$m" == "p2p" ]]; then
    echo "vllm-agg-mx-p2p-planner-${RUN_ID}"
  else
    echo "vllm-agg-disk-planner-${RUN_ID}"
  fi
}

# Default DGD names in the yaml templates (used as sed match targets)
default_dgd_for_mode() {
  local m="$1"
  if [[ "$m" == "p2p" ]]; then
    echo "vllm-agg-mx-p2p-planner"
  else
    echo "vllm-agg-disk-planner"
  fi
}

yaml_for_mode() {
  local m="$1"
  if [[ "$m" == "p2p" ]]; then
    echo "agg_mx_p2p_planner.yaml"
  else
    echo "agg_disk_planner.yaml"
  fi
}

clear_compile_cache() {
  echo "Clearing compile caches on PVC $PVC_NAME (torch.compile + DeepGEMM; source will recompile, target will reuse)..."
  kubectl delete pod "cache-clear-${RUN_ID}" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
  kubectl run "cache-clear-${RUN_ID}" --rm -i --restart=Never \
    --image=busybox \
    --overrides="{\"spec\":{\"volumes\":[{\"name\":\"m\",\"persistentVolumeClaim\":{\"claimName\":\"$PVC_NAME\"}}],\"containers\":[{\"name\":\"c\",\"image\":\"busybox\",\"command\":[\"sh\",\"-c\",\"rm -rf /models/.cache/vllm/compile_cache /models/.cache/deepgemm && echo done\"],\"volumeMounts\":[{\"name\":\"m\",\"mountPath\":\"/models\"}]}]}}" \
    -n "$NAMESPACE"
  echo "Compile caches cleared."
}

drop_page_caches() {
  local label_value="$1"  # "source" or "target"
  local nodes
  nodes=$(kubectl get nodes -l "${ROLE_LABEL}=${label_value}" -o jsonpath='{.items[*].metadata.name}')
  for node in $nodes; do
    local pod_name="drop-cache-${RUN_ID}-${node##*-}"
    echo "Dropping page caches on $node..."
    kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    kubectl run "$pod_name" --restart=Never \
      --image=busybox \
      --overrides="{\"spec\":{\"hostPID\":true,\"nodeName\":\"$node\",\"tolerations\":[{\"operator\":\"Exists\"}],\"containers\":[{\"name\":\"c\",\"image\":\"busybox\",\"command\":[\"sh\",\"-c\",\"echo before: ; grep -i cached /proc/meminfo ; nsenter -t 1 -m -- sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo ERROR: nsenter failed ; echo after: ; grep -i cached /proc/meminfo\"],\"securityContext\":{\"privileged\":true}}]}}" \
      -n "$NAMESPACE"
    # Wait for pod to finish, then read logs reliably
    kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$pod_name" -n "$NAMESPACE" --timeout=120s
    local output
    output=$(kubectl logs "$pod_name" -n "$NAMESPACE")
    echo "$output"
    kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    # Verify the drop actually worked by checking Cached value decreased
    local before_kb after_kb
    before_kb=$(echo "$output" | awk '/^before:/{found=1} found && /^Cached:/{print $2; exit}')
    after_kb=$(echo "$output" | awk '/^after:/{found=1} found && /^Cached:/{print $2; exit}')
    if [[ -n "$before_kb" && -n "$after_kb" ]]; then
      local before_gb after_gb
      before_gb=$(awk "BEGIN{printf \"%.1f\", $before_kb/1048576}")
      after_gb=$(awk "BEGIN{printf \"%.1f\", $after_kb/1048576}")
      echo "  Page cache: ${before_gb} GiB -> ${after_gb} GiB"
      if [[ "$after_kb" -ge "$before_kb" ]]; then
        echo "  WARNING: Page cache did not decrease — drop may have failed"
      fi
    else
      echo "  WARNING: Could not parse before/after cached values"
    fi
  done
}

check_target_resources() {
  local m="$1"
  local target_node
  target_node=$(kubectl get nodes -l "${ROLE_LABEL}=target" -o jsonpath='{.items[0].metadata.name}')
  if [[ -z "$target_node" ]]; then
    echo "ERROR: No node labeled ${ROLE_LABEL}=target found"
    exit 1
  fi

  echo "Waiting for target node $target_node to have $TOTAL_GPUS free GPUs..."
  local wait_elapsed=0
  while true; do
    local gpu_cap gpu_alloc gpu_free
    gpu_cap=$(kubectl get node "$target_node" -o jsonpath='{.status.capacity.nvidia\.com/gpu}')
    gpu_alloc=$(kubectl get pods --all-namespaces --field-selector="spec.nodeName=$target_node,status.phase!=Succeeded,status.phase!=Failed" \
      -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.requests.nvidia\.com/gpu}{"\n"}{end}{end}' \
      | awk '{s+=$1} END {print s+0}')
    gpu_free=$((gpu_cap - gpu_alloc))

    local ready=true
    if [[ "$m" == "p2p" ]]; then
      local rdma_cap rdma_alloc rdma_free
      rdma_cap=$(kubectl get node "$target_node" -o jsonpath="{.status.capacity.${RDMA_RESOURCE}}")
      rdma_alloc=$(kubectl get pods --all-namespaces --field-selector="spec.nodeName=$target_node,status.phase!=Succeeded,status.phase!=Failed" \
        -o jsonpath="{range .items[*]}{range .spec.containers[*]}{.resources.requests.${RDMA_RESOURCE}}{\"\\n\"}{end}{end}" \
        | awk '{s+=$1} END {print s+0}')
      rdma_free=$((rdma_cap - rdma_alloc))
      if [[ "$gpu_free" -ge "$TOTAL_GPUS" ]] && [[ "$rdma_free" -ge "$RDMA_COUNT" ]]; then
        echo "Target node $target_node: GPUs=${gpu_free}/${gpu_cap} free, RDMA=${rdma_free}/${rdma_cap} free"
        return
      fi
      ready=false
    else
      if [[ "$gpu_free" -ge "$TOTAL_GPUS" ]]; then
        echo "Target node $target_node: GPUs=${gpu_free}/${gpu_cap} free"
        return
      fi
      ready=false
    fi

    sleep 30
    wait_elapsed=$((wait_elapsed + 30))
    if (( wait_elapsed % 60 == 0 )); then
      echo "  waiting for resources... GPUs=${gpu_free}/${gpu_cap} free (need $TOTAL_GPUS) (${wait_elapsed}s)"
    fi
  done
}

wait_for_aiperf() {
  local m="$1"
  local dgd_name
  dgd_name=$(dgd_for_mode "$m")
  local timeslice_path="/models/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/${m}/profile_export_aiperf_timeslices.json"

  echo "Waiting for aiperf to finish ($m)..."
  local pod
  pod=$(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" | grep "${dgd_name}.*mx-source" | head -1 || true)
  if [[ -z "$pod" ]]; then
    echo "WARNING: No mx-source pod to poll, sleeping 300s instead"
    sleep 300
    return
  fi

  local elapsed=0
  local timeout=3600
  while ! kubectl exec "$pod" -n "$NAMESPACE" -- test -f "$timeslice_path" 2>/dev/null; do
    sleep 30
    elapsed=$((elapsed + 30))
    if [[ "$elapsed" -ge "$timeout" ]]; then
      echo "WARNING: Timed out waiting for aiperf after ${timeout}s"
      return
    fi
    echo "  waiting... (${elapsed}s)"
  done
  echo "  aiperf finished after ~${elapsed}s"
}

save_logs() {
  local m="$1"
  local dgd_name
  dgd_name=$(dgd_for_mode "$m")
  local logs_dir="${LOCAL_ARTIFACTS}/${m}/logs"
  mkdir -p "$logs_dir"

  echo "Collecting $m service logs..."
  for svc in mx-source mx-target frontend planner; do
    local pod
    pod=$(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" \
      | grep "${dgd_name}.*${svc}" | head -1 || true)
    if [[ -z "$pod" ]]; then
      echo "  $svc: no pod found, skipping"
      continue
    fi
    echo "  $svc: $pod"
    kubectl logs "$pod" -n "$NAMESPACE" --all-containers 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/${svc}.log" || true
    kubectl logs "$pod" -n "$NAMESPACE" --all-containers --previous 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/${svc}_previous.log" || true
    [[ ! -s "$logs_dir/${svc}_previous.log" ]] && rm -f "$logs_dir/${svc}_previous.log"
  done

  if kubectl get pod "$BENCHMARK_POD" -n "$NAMESPACE" &>/dev/null; then
    echo "  aiperf: $BENCHMARK_POD"
    kubectl logs "$BENCHMARK_POD" -n "$NAMESPACE" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$logs_dir/aiperf.log" || true
  fi

  # Save DGD spec and describe output for debugging
  echo "  DGD: $dgd_name"
  kubectl get "dgd/$dgd_name" -n "$NAMESPACE" -o yaml > "$logs_dir/dgd.yaml" 2>&1 || true
  kubectl describe "dgd/$dgd_name" -n "$NAMESPACE" > "$logs_dir/dgd-describe.txt" 2>&1 || true
}

fetch_aiperf_artifacts() {
  local m="$1"
  local remote_path="/perf-cache/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/${m}"
  local local_path="${LOCAL_ARTIFACTS}/${m}"

  if ! kubectl get pod "$BENCHMARK_POD" -n "$NAMESPACE" &>/dev/null; then
    echo "WARNING: Benchmark pod not found, skipping aiperf artifact fetch for $m"
    return
  fi
  if ! kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- test -d "$remote_path" 2>/dev/null; then
    echo "WARNING: $remote_path does not exist on PVC"
    return
  fi

  mkdir -p "$local_path"
  echo "Fetching aiperf artifacts ($m)..."
  local files
  files=$(kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- \
    find "$remote_path" -maxdepth 1 -type f \( -name "*.json" -o -name "*.csv" -o -name "*.log" -o -name "*.jsonl" \) \
    2>/dev/null || true)
  for f in $files; do
    local fname
    fname=$(basename "$f")
    echo "  $fname"
    local attempt
    for attempt in 1 2 3; do
      if kubectl cp "$NAMESPACE/${BENCHMARK_POD}:$f" "$local_path/$fname" 2>/dev/null; then
        break
      fi
      if [[ "$attempt" -lt 3 ]]; then
        echo "    retry $((attempt + 1))/3..."
        sleep 5
      else
        echo "    WARNING: failed to fetch $fname after 3 attempts, skipping"
      fi
    done
  done
}

# -- Run a single experiment --

run_experiment() {
  local m="$1"
  local dgd_name default_dgd dgd_yaml frontend
  dgd_name=$(dgd_for_mode "$m")
  default_dgd=$(default_dgd_for_mode "$m")
  dgd_yaml=$(yaml_for_mode "$m")
  frontend="${dgd_name}-frontend"

  echo ""
  echo "===== Running $m experiment (planner mode) ====="

  # Deploy modelexpress server (dynamo runtime requires mxclient connection)
  model_sed < modelexpress-server.yaml | kubectl apply -n "$NAMESPACE" -f -
  kubectl rollout status "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --timeout=300s

  # Clear stale torch.compile cache if requested
  clear_compile_cache

  # Deploy DGD with planner (sed the default DGD name + planner args)
  local dgd_tmp="/tmp/dgd-${RUN_ID}-${m}-planner.yaml"
  model_sed < "$dgd_yaml" \
    | sed "s|name: ${default_dgd}|name: ${dgd_name}|g" \
    | planner_sed \
    > "$dgd_tmp"
  # Disable target cache reuse if toggled off
  if [[ "$ENABLE_TORCH_COMPILE_CACHE" == "0" ]]; then
    sed -i '/mx-target:/,$ { s|/models/.cache/vllm/compile_cache|/tmp/vllm_compile_cache|g }' "$dgd_tmp"
  fi
  if [[ "$DEEPGEMM_SOURCE_WARMUP_MODE" != "full" ]]; then
    sed -i 's|value: "full"|value: "'"$DEEPGEMM_SOURCE_WARMUP_MODE"'"|g' "$dgd_tmp"
  fi
  if [[ "$DEEPGEMM_TARGET_WARMUP_MODE" != "skip" ]]; then
    # Target does its own warmup: change mode and use ephemeral cache instead of PVC
    sed -i 's|value: "skip"|value: "'"$DEEPGEMM_TARGET_WARMUP_MODE"'"|g' "$dgd_tmp"
    sed -i '/mx-target:/,$ { s|/models/.cache/deepgemm|/tmp/deepgemm|g }' "$dgd_tmp"
  fi
  if [[ "$ENABLE_REQUEST_LOGGING" == "1" ]]; then
    # Frontend: log routing decisions (insert after first envs: line, which is always Frontend)
    sed -i '0,/^      envs:$/{/^      envs:$/a\        - name: DYN_LOG\
          value: "info,dynamo_runtime::pipeline::network::egress::push_router=trace"
}' "$dgd_tmp"
    # Workers: log request arrivals (insert after each DYN_HEALTH_CHECK_ENABLED value)
    sed -i '/value: "false"/a\        - name: DYN_LOG\
          value: "info,dynamo_runtime::pipeline::network::ingress=trace"' "$dgd_tmp"
  fi
  echo "  DGD yaml saved to $dgd_tmp ($(wc -l < "$dgd_tmp") lines)"
  kubectl apply -n "$NAMESPACE" -f "$dgd_tmp"

  # Wait for DGD ready, checking for crashed pods every 30s
  echo "Waiting for DGD $dgd_name to be ready..."
  local wait_elapsed=0
  while true; do
    # Check if DGD is ready
    local ready
    ready=$(kubectl get "dgd/$dgd_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
    if [[ "$ready" == "True" ]]; then
      echo "  DGD $dgd_name is ready."
      break
    fi

    # Check for crashed/error pods (status or restart count > 0)
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers \
      | grep "$dgd_name" \
      | awk '$3 ~ /CrashLoopBackOff|Error|OOMKilled|ImagePullBackOff/ || ($4+0 > 0) {print $1, $3, "restarts="$4}' || true)
    if [[ -n "$failed_pods" ]]; then
      echo "ERROR: Crashed pods detected:"
      echo "$failed_pods"
      echo "Saving crash logs before cleanup..."
      save_logs "$m"
      echo "Cleaning up..."
      kubectl delete pod "$BENCHMARK_POD" -n "$NAMESPACE" --ignore-not-found --wait
      kubectl delete "dgd/$dgd_name" -n "$NAMESPACE" --ignore-not-found --wait
      kubectl delete "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
      return 1
    fi

    sleep 10
    wait_elapsed=$((wait_elapsed + 10))
    if [[ "$wait_elapsed" -ge 1800 ]]; then
      echo "ERROR: Timed out waiting for DGD $dgd_name after ${wait_elapsed}s"
      save_logs "$m"
      echo "Cleaning up..."
      kubectl delete pod "$BENCHMARK_POD" -n "$NAMESPACE" --ignore-not-found --wait
      kubectl delete "dgd/$dgd_name" -n "$NAMESPACE" --ignore-not-found --wait
      kubectl delete "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
      return 1
    fi
    if (( wait_elapsed % 60 == 0 )); then
      echo "  waiting... (${wait_elapsed}s)"
    fi
  done

  # Sanity-check that target nodes still have free GPUs
  check_target_resources "$m"

  # Upload trace file
  local source_pod
  source_pod=$(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" | grep "${dgd_name}.*mx-source" | head -1 || true)
  if [[ -z "$source_pod" ]]; then
    echo "ERROR: No mx-source pod found"
    kubectl get pods -n "$NAMESPACE" --no-headers
    exit 1
  fi
  # Sync DeepGEMM JIT cache from source (ephemeral /tmp) to PVC for target reuse
  echo "Syncing DeepGEMM JIT cache to PVC..."
  kubectl exec "$source_pod" -n "$NAMESPACE" -- \
    sh -c 'rm -rf /models/.cache/deepgemm && mkdir -p /models/.cache/deepgemm && cp -a /tmp/deepgemm/. /models/.cache/deepgemm/ 2>/dev/null && echo "synced $(find /models/.cache/deepgemm/cache -type d 2>/dev/null | wc -l) kernels" || echo "no cache to sync"'

  echo "Uploading trace to ${source_pod}:/models/traces/${DATASET}.jsonl ..."
  kubectl exec "$source_pod" -n "$NAMESPACE" -- mkdir -p /models/traces
  kubectl cp "$TRACE_FILE" "$NAMESPACE/${source_pod}:/models/traces/${DATASET}.jsonl"

  # Drop page caches on target nodes to ensure cold disk reads
  drop_page_caches target

  # Start aiperf (no manual scale-up — planner handles it)
  model_sed < perf.yaml \
    | sed -e "s/value: p2p/value: $m/" \
          -e "s/value: vllm-agg-mx-p2p-frontend/value: $frontend/" \
          -e "s/value: staircase_trace/value: $DATASET/" \
          -e "s/value: \"60\"/value: \"$SLICE_DURATION\"/" \
          -e "s/value: default_run/value: $RUN_TAG/" \
    | kubectl apply -n "$NAMESPACE" -f -

  # Wait for aiperf trace replay to complete (planner scales automatically)
  wait_for_aiperf "$m"

  # Collect logs while pods still exist
  save_logs "$m"

  # Fetch aiperf artifacts from PVC to local
  fetch_aiperf_artifacts "$m"

  # Remove artifacts from PVC to avoid piling up
  local remote_artifact_dir="/perf-cache/artifacts/${MODEL_BASE}/${DATASET}/${RUN_TAG}/${m}"
  echo "Cleaning up PVC artifacts at $remote_artifact_dir ..."
  kubectl exec "$BENCHMARK_POD" -n "$NAMESPACE" -- rm -rf "$remote_artifact_dir" 2>/dev/null || true

  # Cleanup this experiment's DGD, benchmark pod, and modelexpress
  echo "Cleaning up $m..."
  kubectl delete pod "$BENCHMARK_POD" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "dgd/$dgd_name" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
}

# -- Clean up all resources for this RUN_ID --

clean_all() {
  echo "=== Cleaning up all resources for RUN_ID=$RUN_ID (namespace=$NAMESPACE) ==="
  kubectl delete pod "$BENCHMARK_POD" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "dgd/vllm-agg-mx-p2p-planner-${RUN_ID}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "dgd/vllm-agg-disk-planner-${RUN_ID}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "deployment/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "service/${MX_SERVER_NAME}" -n "$NAMESPACE" --ignore-not-found --wait
  kubectl delete "daemonset/${DAEMONSET_NAME}" -n "$NAMESPACE" --ignore-not-found --wait

  # Remove taints from target nodes
  # Remove taints from reserved nodes
  local taint_key="dynamo/${RUN_ID}-reserved"
  echo "Removing taints (${taint_key})..."
  local tainted_nodes
  tainted_nodes=$(kubectl get nodes -l "$ROLE_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  for node in $tainted_nodes; do
    kubectl taint nodes "$node" "${taint_key}-" 2>/dev/null || true
  done

  # Remove node labels scoped to this RUN_ID
  echo "Removing node labels (${ROLE_LABEL}, ${IMAGE_CACHE_LABEL})..."
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

if [[ "$MODE" == "plot" ]]; then
  if [[ -d "$LOCAL_ARTIFACTS/p2p" ]] && [[ -d "$LOCAL_ARTIFACTS/disk" ]]; then
    echo "===== Generating comparison plot ====="
    python "${REPO_ROOT}/mx_benchmarks/plot_scaleup.py" \
      --artifacts-dir "$LOCAL_ARTIFACTS" \
      --slice-duration "$SLICE_DURATION" \
      --stat "$STAT"
    echo ""
    echo "===== Done ====="
    echo "Artifacts: $LOCAL_ARTIFACTS"
  else
    echo "ERROR: Need both $LOCAL_ARTIFACTS/p2p and $LOCAL_ARTIFACTS/disk to generate plot"
    exit 1
  fi
  exit 0
fi

if [[ "$MODE" == "table" ]]; then
  source "${REPO_ROOT}/mx_benchmarks/startup_timing.sh"
  print_startup_table
  exit 0
fi

# Cleanup any leftover resources for THIS RUN_ID
clean_all

# Ensure cleanup runs on any exit (crash, set -e, signal)
trap clean_all EXIT

# Setup source node image cache
./setup-image-cache.sh source

# Reserve MAX_REPLICAS target nodes upfront (taint to prevent GPU theft)
./setup-image-cache.sh target "$MAX_REPLICAS"
TAINT_KEY="dynamo/${RUN_ID}-reserved"
echo "Tainting source node with ${TAINT_KEY}=source:NoSchedule..."
kubectl taint nodes -l "${ROLE_LABEL}=source" "${TAINT_KEY}=source:NoSchedule"
echo "Tainting target nodes with ${TAINT_KEY}=target:NoSchedule..."
kubectl taint nodes -l "${ROLE_LABEL}=target" "${TAINT_KEY}=target:NoSchedule"

# Run experiment(s) — guard failures so set -e doesn't skip cleanup
rc=0
case "$MODE" in
  p2p)  run_experiment p2p || rc=1 ;;
  disk) run_experiment disk || rc=1 ;;
  both) run_experiment p2p || { echo "WARNING: p2p experiment failed, continuing to disk..."; rc=1; }
        run_experiment disk || { echo "WARNING: disk experiment failed"; rc=1; } ;;
esac

source "${REPO_ROOT}/mx_benchmarks/startup_timing.sh"
print_startup_table

# Plot comparison (only when both modes have results)
if [[ -d "$LOCAL_ARTIFACTS/p2p" ]] && [[ -d "$LOCAL_ARTIFACTS/disk" ]]; then
  echo ""
  echo "===== Generating comparison plot ====="
  python "${REPO_ROOT}/mx_benchmarks/plot_scaleup.py" \
    --artifacts-dir "$LOCAL_ARTIFACTS" \
    --slice-duration "$SLICE_DURATION" \
    --stat "$STAT"
fi

echo ""
echo "===== Done ====="
echo "Artifacts: $LOCAL_ARTIFACTS"
exit $rc
