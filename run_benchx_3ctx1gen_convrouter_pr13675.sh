#!/bin/bash
#SBATCH --job-name=core_dlfw_ci-benchx.3ctx1gen.convrouter.pr13675
#SBATCH --nodes=2
#SBATCH --partition=gb200
#SBATCH --account=core_dlfw_ci
#SBATCH --time=04:00:00
#SBATCH --output=bench/logs/run_benchx_3ctx1gen_convrouter_pr13675_%j.log
#SBATCH --error=bench/logs/run_benchx_3ctx1gen_convrouter_pr13675_%j.err

# =============================================================================
# benchx (feat/bench_x sha 11e16c) — 3 ctx + 1 gen with ConversationRouter,
# driven by dynamo.trtllm (etcd + nats + dynamo frontend) instead of
# trtllm-serve disaggregated.
#
# Layout:
#   NODE0 — etcd + nats + dynamo frontend + 3 ctx workers (GPUs 0-2)
#   NODE1 — 1 gen worker (GPU 0)
#
# RWLT sends X-Session-ID + X-Correlation-ID via send_conversation_routing_headers.
#
# Env:
#   CONCURRENCY  — single concurrency (default 48)
#   HOSTCACHE    — 1 = enable kv_cache_config.host_cache_size: 80GB on ctx
#                  0 = no host offloading (default)
#
# Submit:
#   sbatch --export=ALL,CONCURRENCY=48,HOSTCACHE=0 bench/run_benchx_3ctx1gen_convrouter_pr13675.sh
# =============================================================================

set -uo pipefail

CONCURRENCY="${CONCURRENCY:-48}"
HOSTCACHE="${HOSTCACHE:-0}"

if [ "$HOSTCACHE" = "1" ]; then HCTAG="hcon"; else HCTAG="hcoff"; fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/core_dlfw_ci/rihuo/trtllm_d236c0_release_dynamo_8db5ad.sqsh}"
EXP_NAME="run_benchx_3ctx1gen_convrouter_pr13675_${HCTAG}_c${CONCURRENCY}"

HF_TOKEN="${HF_TOKEN:-}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/core_dlfw_ci/rihuo/artificial-analysis}"
USER_DIR=$(dirname "$REPO_DIR")
MODEL_PATH="/lustre/fsw/core_dlfw_ci/rihuo/openai_gpt-oss-120b"
MODEL="openai/gpt-oss-120b"
TRAJECTORY_PATH="${REPO_DIR}/data/agentic_coding_v2_full.jsonl"
CONTAINER_MOUNTS="/lustre/:/lustre/"

NODE0=$(scontrol show hostnames $SLURM_NODELIST | sed -n '1p')
NODE1=$(scontrol show hostnames $SLURM_NODELIST | sed -n '2p')

# Dynamo ports
ETCD_PORT=2379
NATS_PORT=4222
FRONTEND_PORT=8000
DYNAMO_REQUEST_PLANE=tcp

# DYN_SYSTEM_PORT assignments (one per worker)
DYN_SYS_PORT_CTX_0=8081
DYN_SYS_PORT_CTX_1=8082
DYN_SYS_PORT_CTX_2=8083
DYN_SYS_PORT_GEN=8085

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/bench/results/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/metrics" "$REPO_DIR/bench/logs"

SRUN_PIDS=()
METRICS_PID=""

cleanup() {
    local exit_code=$?
    if [ -n "${METRICS_PID}" ] && kill -0 "${METRICS_PID}" 2>/dev/null; then
        echo "Stopping metrics sidecar (PID ${METRICS_PID})..."
        kill -TERM "${METRICS_PID}" 2>/dev/null || true
        wait "${METRICS_PID}" 2>/dev/null || true
    fi
    if [ "${#SRUN_PIDS[@]}" -gt 0 ]; then
        echo "Cleaning up ${#SRUN_PIDS[@]} background srun steps..."
        kill "${SRUN_PIDS[@]}" 2>/dev/null || true
        wait "${SRUN_PIDS[@]}" 2>/dev/null || true
    fi
    if [ $exit_code -eq 0 ]; then
        echo "Completed successfully"
    else
        echo "Failed (exit code: $exit_code)"
    fi
}
trap cleanup EXIT

start_bg() {
    "$@" &
    SRUN_PIDS+=($!)
}

require_alive() {
    local pid="$1"
    local name="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: ${name} exited unexpectedly"
        wait "$pid" || true
        exit 1
    fi
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-300}"
    local deadline=$((SECONDS + timeout))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if bash -c "echo > /dev/tcp/${host}/${port}" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: port ${host}:${port} is not open after waiting ${timeout} seconds"
    return 1
}

wait_for_dynamo_workers() {
    local host="$1"
    local port="$2"
    local expected_prefill="$3"
    local expected_decode="$4"
    local timeout="${5:-2700}"
    local report_interval="${6:-60}"
    local deadline=$((SECONDS + timeout))
    local last_report=$SECONDS

    echo "Polling http://${host}:${port}/health every 10s for ${expected_prefill} prefills and ${expected_decode} decodes"

    while [ "$SECONDS" -lt "$deadline" ]; do
        local response
        if response=$(curl -fsS --max-time 5 "http://${host}:${port}/health" 2>/dev/null); then
            local prefill_count decode_count
            prefill_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(sum(1 for i in data.get('instances', [])
          if i.get('endpoint') == 'generate' and i.get('component') == 'prefill'))
" 2>/dev/null || echo "0")
            decode_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(sum(1 for i in data.get('instances', [])
          if i.get('endpoint') == 'generate' and i.get('component') in ('decode', 'tensorrt_llm', 'backend')))
" 2>/dev/null || echo "0")

            if [ "${prefill_count}" -ge "${expected_prefill}" ] && [ "${decode_count}" -ge "${expected_decode}" ]; then
                echo "All workers ready: ${prefill_count} prefills, ${decode_count} decodes"
                return 0
            fi

            if [ $((SECONDS - last_report)) -ge "${report_interval}" ]; then
                echo "Waiting: have ${prefill_count}/${expected_prefill} prefills, ${decode_count}/${expected_decode} decodes"
                last_report=$SECONDS
            fi
        else
            if [ $((SECONDS - last_report)) -ge "${report_interval}" ]; then
                echo "Waiting: frontend not responding yet"
                last_report=$SECONDS
            fi
        fi
        sleep 10
    done

    echo "ERROR: workers did not register within ${timeout}s"
    return 1
}

verify_model_ready() {
    local host="$1"
    local port="$2"
    local timeout="${3:-120}"
    local deadline=$((SECONDS + timeout))

    echo "Verifying model is ready via /v1/models..."
    while [ "$SECONDS" -lt "$deadline" ]; do
        local response
        if response=$(curl -fsS --max-time 5 "http://${host}:${port}/v1/models" 2>/dev/null); then
            local model_count
            model_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
for m in models:
    print(f\"  model: {m.get('id', 'unknown')}\", file=sys.stderr)
print(len(models))
" 2>/dev/null || echo "0")
            if [ "${model_count}" -gt 0 ]; then
                echo "Model is serving (${model_count} model(s) available)"
                return 0
            fi
        fi
        sleep 5
    done
    echo "ERROR: /v1/models did not return any models within ${timeout}s"
    return 1
}

echo "============================================"
echo "$EXP_NAME (job $SLURM_JOB_ID) ctx=$NODE0 gen=$NODE1  CONCURRENCY=$CONCURRENCY HOSTCACHE=$HOSTCACHE"
echo "Container: $CONTAINER_IMAGE"
echo "Results: $RESULTS_DIR"
echo "============================================"

CTX_HCACHE_LINE=""
if [ "$HOSTCACHE" = "1" ]; then CTX_HCACHE_LINE="  host_cache_size: 85899345920"; fi

cat > "$RESULTS_DIR/ctx.yaml" << EOF
backend: pytorch
tensor_parallel_size: 1
pipeline_parallel_size: 1
moe_expert_parallel_size: 1
max_batch_size: 32
max_num_tokens: 20000
max_seq_len: 131072
trust_remote_code: true
disable_overlap_scheduler: true
enable_chunked_prefill: true
enable_attention_dp: false
num_postprocess_workers: 4
sampler_type: auto
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
  event_buffer_max_size: 16384
  dtype: fp8
  free_gpu_memory_fraction: 0.90
  enable_block_reuse: true
${CTX_HCACHE_LINE}
torch_compile_config:
  enable_fullgraph: true
  enable_piecewise_cuda_graph: true
  enable_userbuffers: false
  capture_num_tokens: [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192, 8704, 9216, 9728, 10240, 11264, 12288, 13312, 13914]
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
cache_transceiver_config:
  max_tokens_in_buffer: 131072
  backend: DEFAULT
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model: /lustre/fsw/core_dlfw_ci/rihuo/nvidia_gpt-oss-120b-Eagle3-v3
  eagle3_one_model: true
  eagle3_layers_to_capture: [23, 29, 35]
enable_iter_perf_stats: true
enable_iter_req_stats: true
print_iter_log: true
EOF

cat > "$RESULTS_DIR/gen.yaml" << 'EOF'
backend: pytorch
tensor_parallel_size: 1
pipeline_parallel_size: 1
moe_expert_parallel_size: 1
max_batch_size: 1024
max_num_tokens: 20000
max_seq_len: 131072
trust_remote_code: true
enable_chunked_prefill: true
enable_attention_dp: false
num_postprocess_workers: 4
sampler_type: auto
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
  event_buffer_max_size: 16384
  dtype: fp8
  free_gpu_memory_fraction: 0.90
  enable_block_reuse: true
torch_compile_config:
  enable_fullgraph: true
  enable_piecewise_cuda_graph: true
  enable_userbuffers: false
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cache_transceiver_config:
  max_tokens_in_buffer: 131072
  backend: DEFAULT
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model: /lustre/fsw/core_dlfw_ci/rihuo/nvidia_gpt-oss-120b-Eagle3-v3
  eagle3_one_model: true
  eagle3_layers_to_capture: [23, 29, 35]
enable_iter_perf_stats: true
enable_iter_req_stats: true
print_iter_log: true
EOF

COMMON_ENV="export TRTLLM_WORKER_DISABLE_GC=1 && \
export DYN_ROUTER_QUEUE_THRESHOLD=100000"

DYN_TCP_WORKER_POOL_SIZE=100000
DYN_TCP_QUEUE_SIZE=$((DYN_TCP_WORKER_POOL_SIZE * 4))
DYNAMO_WORKER_ENV="export ETCD_ENDPOINTS=http://${NODE0}:${ETCD_PORT} && \
export NATS_SERVER=nats://${NODE0}:${NATS_PORT} && \
export DYN_REQUEST_PLANE=${DYNAMO_REQUEST_PLANE} && \
export DYN_TCP_WORKER_POOL_SIZE=${DYN_TCP_WORKER_POOL_SIZE} && \
export DYN_TCP_WORK_QUEUE_SIZE=${DYN_TCP_QUEUE_SIZE}"

# ==============================================================================
# Stage 1: Start infrastructure services (etcd + nats) on NODE0
# ==============================================================================
echo "[$(date +%H:%M:%S)] Starting etcd and nats on $NODE0..."

start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c '
    set -euo pipefail
    rm -rf /tmp/etcd /tmp/nats
    mkdir -p /tmp/etcd /tmp/nats

    HOST_IP=$(hostname -I | awk "{print \$1}")
    echo "Infra node IP: ${HOST_IP}"

    echo "Starting nats-server..."
    nats-server -js -sd /tmp/nats &
    NATS_PID=$!

    echo "Starting etcd..."
    etcd \
        --data-dir /tmp/etcd \
        --listen-client-urls http://0.0.0.0:'"${ETCD_PORT}"' \
        --advertise-client-urls http://${HOST_IP}:'"${ETCD_PORT}"' &
    ETCD_PID=$!

    for i in $(seq 1 300); do
        if echo > /dev/tcp/localhost/'"${NATS_PORT}"' 2>/dev/null && \
           echo > /dev/tcp/localhost/'"${ETCD_PORT}"' 2>/dev/null; then
            echo "etcd and nats are ready"
            break
        fi
        sleep 1
    done

    echo "Infrastructure services running (nats PID: ${NATS_PID}, etcd PID: ${ETCD_PID})"
    wait
  '
INFRA_PID="${SRUN_PIDS[-1]}"
require_alive "${INFRA_PID}" "INFRA_PID"

echo "Waiting for nats (port ${NATS_PORT}) on ${NODE0}..."
wait_for_port "${NODE0}" "${NATS_PORT}" 300

echo "Waiting for etcd (port ${ETCD_PORT}) on ${NODE0}..."
wait_for_port "${NODE0}" "${ETCD_PORT}" 300
echo "Infrastructure services are ready"

# ==============================================================================
# Stage 2: Start TRTLLM workers via dynamo.trtllm
# ==============================================================================

# --- gen worker on $NODE1 GPU 0 (decode) ---
echo "[$(date +%H:%M:%S)] Starting gen worker on $NODE1 GPU 0..."
start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE1 --mpi=pmix \
  --output="$RESULTS_DIR/gen_worker.log" \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-entrypoint \
  --no-container-mount-home \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=0 && $COMMON_ENV && $DYNAMO_WORKER_ENV && \
    export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN} && \
    trtllm-llmapi-launch python3 -m dynamo.trtllm \
      --model-path $MODEL_PATH --served-model-name $MODEL \
      --disaggregation-mode decode \
      --extra-engine-args $RESULTS_DIR/gen.yaml \
      --request-plane ${DYNAMO_REQUEST_PLANE} \
      --publish-events-and-metrics"
GEN_PID="${SRUN_PIDS[-1]}"

# --- 3 ctx workers on $NODE0 GPUs 0-2 (prefill) ---
CTX_PIDS=()
for GPU in 0 1 2; do
  PORT_VAR="DYN_SYS_PORT_CTX_${GPU}"
  CTX_PORT="${!PORT_VAR}"
  echo "[$(date +%H:%M:%S)] Starting ctx worker on $NODE0 GPU $GPU (DYN_SYSTEM_PORT=${CTX_PORT})..."
  start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
    --output="$RESULTS_DIR/ctx_worker_g${GPU}.log" \
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-entrypoint \
    --no-container-mount-home \
    bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=${GPU} && $COMMON_ENV && $DYNAMO_WORKER_ENV && \
      export DYN_SYSTEM_PORT=${CTX_PORT} && \
      trtllm-llmapi-launch python3 -m dynamo.trtllm \
        --model-path $MODEL_PATH --served-model-name $MODEL \
        --disaggregation-mode prefill \
        --extra-engine-args $RESULTS_DIR/ctx.yaml \
        --request-plane ${DYNAMO_REQUEST_PLANE} \
        --publish-events-and-metrics"
  CTX_PIDS+=("${SRUN_PIDS[-1]}")
done

require_alive "${GEN_PID}" "GEN_PID"
for i in "${!CTX_PIDS[@]}"; do
  require_alive "${CTX_PIDS[$i]}" "CTX_PID_g${i}"
done

# ==============================================================================
# Stage 3: Start dynamo frontend on NODE0
# ==============================================================================
echo "[$(date +%H:%M:%S)] Starting dynamo frontend on $NODE0..."
start_bg srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --output="$RESULTS_DIR/frontend.log" \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-entrypoint \
  --no-container-mount-home \
  bash -c "$COMMON_ENV && \
    export ETCD_ENDPOINTS=http://${NODE0}:${ETCD_PORT} && \
    export NATS_SERVER=nats://${NODE0}:${NATS_PORT} && \
    export DYN_REQUEST_PLANE=${DYNAMO_REQUEST_PLANE} && \
    python3 -m dynamo.frontend \
      --http-port ${FRONTEND_PORT} \
      --request-plane ${DYNAMO_REQUEST_PLANE} \
      --router-mode kv \
      --no-router-kv-events \
      --router-ttl-secs 480.0"
FRONTEND_PID="${SRUN_PIDS[-1]}"
require_alive "${FRONTEND_PID}" "FRONTEND_PID"

# ==============================================================================
# Stage 4: Wait for all workers to register, then verify model is serving
# ==============================================================================
echo "[$(date +%H:%M:%S)] Waiting for servers (3 prefill + 1 decode)..."
if ! wait_for_dynamo_workers "${NODE0}" "${FRONTEND_PORT}" 3 1 2700 60; then
    echo "ERROR: workers did not become healthy"
    for f in $RESULTS_DIR/*.log; do echo "=== $(basename $f) ==="; tail -30 "$f"; done
    exit 1
fi

if ! verify_model_ready "${NODE0}" "${FRONTEND_PORT}" 120; then
    echo "ERROR: model is not serving"
    for f in $RESULTS_DIR/*.log; do echo "=== $(basename $f) ==="; tail -30 "$f"; done
    exit 1
fi

echo "[$(date +%H:%M:%S)] All workers healthy and model is serving"

# --- metrics sidecar ---
echo "[$(date +%H:%M:%S)] Starting metrics capture sidecar (interval=2s)..."
python3 "$REPO_DIR/bench/capture_metrics.py" \
  --endpoints "${NODE0}:${DYN_SYS_PORT_CTX_0},${NODE0}:${DYN_SYS_PORT_CTX_1},${NODE0}:${DYN_SYS_PORT_CTX_2},${NODE1}:${DYN_SYS_PORT_GEN}" \
  --labels "ctx_g0,ctx_g1,ctx_g2,gen_g0" \
  --output-dir "$RESULTS_DIR/metrics" \
  --interval 2 \
  > "$RESULTS_DIR/metrics/capture.stderr.log" 2>&1 &
METRICS_PID=$!
sleep 2

# --- RWLT @ CONCURRENCY ---
cat > "$RESULTS_DIR/rwlt_config.yaml" << RWLTEOF
base_url: http://${NODE0}:${FRONTEND_PORT}/v1
model: $MODEL
concurrencies: [${CONCURRENCY}]
phase_timeout_seconds: 1800
user_spawn_rate: 1.0
settling_time_seconds: 60
min_measurement_seconds: 300.0
min_total_trajectories: 30
min_trajectories_per_user: 3
trajectory_path: $TRAJECTORY_PATH
trajectories_per_user: 30
max_starting_line_offset: 10
seed: 42
timeout_seconds: 300.0
max_tokens: 16384
reasoning_effort: high
record_err_reasons: true
record_err_reasons_include_input: false
tool_calls_args_only: true
send_conversation_routing_headers: true
exp_prefix: ${EXP_NAME}
results_dir: $RESULTS_DIR/rwlt_results
RWLTEOF

echo "==== Running RWLT @ c=${CONCURRENCY} (X-Session-ID enabled) ===="
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && $COMMON_ENV && \
    uv run --isolated --with openai --with httpx --with pyyaml --with pydantic \
      python rwlt/run.py --config $RESULTS_DIR/rwlt_config.yaml" \
  > "$RESULTS_DIR/benchmark.log" 2>&1
echo "Bench exit: $?"

# --- shut down metrics sidecar ---
kill -TERM "$METRICS_PID" 2>/dev/null; wait "$METRICS_PID" 2>/dev/null || true
METRICS_PID=""
echo "Metrics samples per worker:"
for f in "$RESULTS_DIR/metrics"/*_metrics.jsonl; do
  [ -f "$f" ] && printf "  %s: %s lines\n" "$(basename "$f")" "$(wc -l < "$f")"
done

# --- final per-worker /metrics snapshot ---
echo "Dumping final /metrics per worker..."
for ENTRY in \
  "ctx_g0:${NODE0}:${DYN_SYS_PORT_CTX_0}" \
  "ctx_g1:${NODE0}:${DYN_SYS_PORT_CTX_1}" \
  "ctx_g2:${NODE0}:${DYN_SYS_PORT_CTX_2}" \
  "gen_g0:${NODE1}:${DYN_SYS_PORT_GEN}"; do
  ROLE="${ENTRY%%:*}"
  REST="${ENTRY#*:}"
  HOST="${REST%%:*}"
  PORT="${REST##*:}"
  OUT="$RESULTS_DIR/metrics/${ROLE}_${HOST}_${PORT}_final.prom"
  if curl -fsS --max-time 5 "http://${HOST}:${PORT}/metrics" -o "$OUT"; then
    printf "  %s: %s lines (%s)\n" "$(basename "$OUT")" "$(wc -l < "$OUT")" "http://${HOST}:${PORT}/metrics"
  else
    echo "  WARN: failed to fetch /metrics from ${ROLE} (${HOST}:${PORT})"
  fi
done

# --- full unified iter-stats plot ---
echo "==== Plotting iter_stats_unified.png ===="
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && uv run --isolated --with matplotlib python analysis/plot_unified.py $RESULTS_DIR" \
  > "$RESULTS_DIR/plot_iter_stats.log" 2>&1
echo "plot exit: $?"

echo "==== DONE ===="
echo "Outputs:"
echo "  RWLT summary:    $RESULTS_DIR/rwlt_results/*.txt"
echo "  Iter-stats plot: $RESULTS_DIR/iter_stats_unified.png"
echo "  Logs:            $RESULTS_DIR/"
