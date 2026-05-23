#!/bin/bash
#SBATCH --job-name=coreai_comparch_aarwlt-benchx.1ctx1gen.convrouter.pr13675
#SBATCH --nodes=1
#SBATCH --partition=gb200
#SBATCH --account=coreai_comparch_aarwlt
#SBATCH --time=01:30:00
#SBATCH --output=bench/logs/run_benchx_1ctx1gen_convrouter_pr13675_%j.log
#SBATCH --error=bench/logs/run_benchx_1ctx1gen_convrouter_pr13675_%j.err

# =============================================================================
# benchx (feat/bench_x sha 11e16c) — 1 ctx + 1 gen with ConversationRouter.
# 1:1 with conv router is mostly a sanity check (single ctx → no real routing
# choice), but exercises the same code path as 4:1 and serves as a baseline
# at the same concurrencies.
#
# RWLT sends X-Session-ID + X-Correlation-ID via send_conversation_routing_headers.
#
# Env:
#   CONCURRENCY  — single concurrency (default 48)
#   HOSTCACHE    — 1 = enable kv_cache_config.host_cache_size: 80GB on ctx
#                  0 = no host offloading (default)
#
# Submit:
#   sbatch --export=ALL,CONCURRENCY=48,HOSTCACHE=0 bench/run_benchx_1ctx1gen_convrouter.sh
# =============================================================================

set -uo pipefail

CONCURRENCY="${CONCURRENCY:-48}"
HOSTCACHE="${HOSTCACHE:-0}"

if [ "$HOSTCACHE" = "1" ]; then HCTAG="hcon"; else HCTAG="hcoff"; fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-gitlab-master.nvidia.com:5005#shobhitv/trtllm-builder/trtllm_11e16c_release}"
EXP_NAME="run_benchx_1ctx1gen_convrouter_pr13675_${HCTAG}_c${CONCURRENCY}"

HF_TOKEN="${HF_TOKEN:-}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
USER_DIR=$(dirname "$REPO_DIR")
MODEL="openai/gpt-oss-120b"
TRAJECTORY_PATH="${REPO_DIR}/data/agentic_coding_v2_full.jsonl"
CONTAINER_MOUNTS="/lustre/:/lustre/,/etc/hostname:/etc/host_hostname,$REPO_DIR/bench/patches/router_benchx_pr13675.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/router.py,$REPO_DIR/bench/patches/openai_server_benchx_pr13675.py:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/openai_server.py"

NODE0=$(scontrol show hostnames $SLURM_NODELIST | sed -n '1p')

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/bench/results/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/metrics" "$REPO_DIR/bench/logs"

echo "============================================"
echo "$EXP_NAME (job $SLURM_JOB_ID) on $NODE0  CONCURRENCY=$CONCURRENCY HOSTCACHE=$HOSTCACHE"
echo "Container: $CONTAINER_IMAGE"
echo "Results: $RESULTS_DIR"
echo "============================================"

cat > "$RESULTS_DIR/master.yaml" << EOF
hostname: "0.0.0.0"
port: 8000
context_servers:
  router:
    type: conversation
    args:
      match_threshold: 0.75
      use_token_ids: false
      hash_skip_count: 0
      max_sessions: 100000
  num_instances: 1
  urls:
    - "${NODE0}:8001"
generation_servers:
  num_instances: 1
  urls:
    - "${NODE0}:8002"
EOF

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
stream_interval: 100
trust_remote_code: true
disable_overlap_scheduler: true
enable_chunked_prefill: true
enable_attention_dp: false
num_postprocess_workers: 0
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
  speculative_model: nvidia/gpt-oss-120b-Eagle3-next
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
stream_interval: 100
trust_remote_code: true
enable_chunked_prefill: true
enable_attention_dp: false
num_postprocess_workers: 0
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
  speculative_model: nvidia/gpt-oss-120b-Eagle3-next
enable_iter_perf_stats: true
enable_iter_req_stats: true
print_iter_log: true
EOF

echo "Master:"; cat "$RESULTS_DIR/master.yaml"; echo

COMMON_ENV="export HF_HOME=$USER_DIR/cache && \
export HF_HUB_CACHE=$USER_DIR/cache/hub && \
export HF_TOKEN=$HF_TOKEN && \
export TRANSFORMERS_CACHE=$USER_DIR/cache && \
export TIKTOKEN_ENCODINGS_BASE=$USER_DIR/cache/tiktoken_encodings && \
export TRTLLM_SERVER_DISABLE_GC=1 && \
export TRTLLM_WORKER_DISABLE_GC=1 && \
mkdir -p $USER_DIR/cache/hub"

# --- gen worker on $NODE0 GPU 1 port 8002 ---
echo "[$(date +%H:%M:%S)] Starting gen worker on $NODE0 GPU 1 port 8002..."
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=1 && $COMMON_ENV && \
    trtllm-llmapi-launch trtllm-serve $MODEL --extra_llm_api_options $RESULTS_DIR/gen.yaml \
      --host 0.0.0.0 --port 8002" \
  > "$RESULTS_DIR/gen_g0.log" 2>&1 &
GEN_PID=$!

# --- ctx worker on $NODE0 GPU 0 port 8001 ---
echo "[$(date +%H:%M:%S)] Starting ctx worker on $NODE0 GPU 0 port 8001..."
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=0 && $COMMON_ENV && \
    trtllm-llmapi-launch trtllm-serve $MODEL --extra_llm_api_options $RESULTS_DIR/ctx.yaml \
      --host 0.0.0.0 --port 8001" \
  > "$RESULTS_DIR/ctx_g0.log" 2>&1 &
CTX_PID=$!

# --- master ---
echo "[$(date +%H:%M:%S)] Starting master on $NODE0..."
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "$COMMON_ENV && trtllm-serve disaggregated --config_file $RESULTS_DIR/master.yaml \
    --server_start_timeout 7200 --request_timeout 7200" \
  > "$RESULTS_DIR/master.log" 2>&1 &
MASTER_PID=$!

# --- health wait ---
echo "[$(date +%H:%M:%S)] Waiting for servers (1 ctx + 1 gen)..."
HEALTHY=0
for i in $(seq 1 600); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$NODE0:8000/health 2>/dev/null)
  if [ "$CODE" = "200" ]; then echo "[$(date +%H:%M:%S)] Health OK"; HEALTHY=1; break; fi
  sleep 3
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "ERROR: not healthy"
  for f in $RESULTS_DIR/*.log; do echo "=== $(basename $f) ==="; tail -30 "$f"; done
  kill $MASTER_PID $GEN_PID $CTX_PID 2>/dev/null; wait 2>/dev/null; exit 1
fi

# --- /kv_cache_events probe (initial) ---
for ENDPOINT in $NODE0:8001 $NODE0:8002; do
  R=$(curl -s -X POST --max-time 5 "http://$ENDPOINT/kv_cache_events")
  echo "$R" > "$RESULTS_DIR/kv_events_probe_$(echo $ENDPOINT | tr ':' '_').json"
done

# --- metrics sidecar ---
echo "[$(date +%H:%M:%S)] Starting metrics capture sidecar (interval=2s)..."
python3 "$REPO_DIR/bench/capture_metrics.py" \
  --endpoints "${NODE0}:8001,${NODE0}:8002" \
  --labels "ctx_g0,gen_g0" \
  --output-dir "$RESULTS_DIR/metrics" \
  --interval 2 \
  > "$RESULTS_DIR/metrics/capture.stderr.log" 2>&1 &
METRICS_PID=$!
sleep 2

# --- RWLT @ CONCURRENCY ---
cat > "$RESULTS_DIR/rwlt_config.yaml" << RWLTEOF
base_url: http://${NODE0}:8000/v1
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
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && $COMMON_ENV && \
    uv run --isolated --with openai --with httpx --with pyyaml --with pydantic \
      python rwlt/run.py --config $RESULTS_DIR/rwlt_config.yaml" \
  > "$RESULTS_DIR/benchmark.log" 2>&1
echo "Bench exit: $?"

# --- final probe + metrics shutdown ---
for ENDPOINT in $NODE0:8001 $NODE0:8002; do
  R=$(curl -s -X POST --max-time 5 "http://$ENDPOINT/kv_cache_events")
  echo "$R" > "$RESULTS_DIR/kv_events_final_$(echo $ENDPOINT | tr ':' '_').json"
done

kill -TERM $METRICS_PID 2>/dev/null; wait $METRICS_PID 2>/dev/null || true
echo "Metrics samples per worker:"
for f in "$RESULTS_DIR/metrics"/*_metrics.jsonl; do [ -f "$f" ] && printf "  %s: %s lines\n" "$(basename $f)" "$(wc -l < $f)"; done

# --- full unified iter-stats plot ---
echo "==== Plotting iter_stats_unified.png ===="
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && uv run --isolated --with matplotlib python analysis/plot_unified.py $RESULTS_DIR" \
  > "$RESULTS_DIR/plot_iter_stats.log" 2>&1
echo "plot exit: $?"

echo "==== DONE ===="
echo "Outputs:"
echo "  RWLT summary:    $RESULTS_DIR/rwlt_results/*.txt"
echo "  Iter-stats plot: $RESULTS_DIR/iter_stats_unified.png"
echo "  Master log:      $RESULTS_DIR/master.log"
kill $MASTER_PID $GEN_PID $CTX_PID 2>/dev/null
wait 2>/dev/null
