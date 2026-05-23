#!/bin/bash
#SBATCH --job-name=core_dlfw_ci-trtllm_serve.agg_tp1
#SBATCH --nodes=1
#SBATCH --partition=gb200
#SBATCH --account=core_dlfw_ci
#SBATCH --time=04:00:00
#SBATCH --output=bench/logs/trtllm_serve_agg_tp1_%j.log
#SBATCH --error=bench/logs/trtllm_serve_agg_tp1_%j.err

# =============================================================================
# trtllm-serve aggregated TP1 on GB200 (single worker, no disaggregation).
#
# Layout:
#   NODE0 GPU 0  — single trtllm-serve aggregated worker (port 8000)
#
# Env:
#   CONCURRENCY    — single concurrency (default 48)
#
# Submit:
#   sbatch --export=ALL,CONCURRENCY=48 bench/trtllm-serve-agg-tp1.sh
# =============================================================================

set -uo pipefail

CONCURRENCY="${CONCURRENCY:-48}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/core_dlfw_ci/rihuo/dynamo-trtllm-rihuo-arm64-1-2-0-0dd537-publisherfix.sqsh}"
EXP_NAME="trtllm_serve_agg_tp1_c${CONCURRENCY}"

HF_TOKEN="${HF_TOKEN:-}"
REPO_DIR="${REPO_DIR:-/lustre/fsw/core_dlfw_ci/rihuo/artificial-analysis}"
USER_DIR=$(dirname "$REPO_DIR")
MODEL_PATH="/lustre/fsw/core_dlfw_ci/rihuo/openai_gpt-oss-120b"
MODEL="openai/gpt-oss-120b"
TRAJECTORY_PATH="${REPO_DIR}/data/agentic_coding_v2_full.jsonl"
CONTAINER_MOUNTS="/lustre/:/lustre/"

NODE0=$(scontrol show hostnames $SLURM_NODELIST | sed -n '1p')

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${EXP_NAME}_${TIMESTAMP}_${SLURM_JOB_ID:-unknown}"
RESULTS_DIR="$REPO_DIR/bench/results/trtllm_serve/${RUN_TAG}"
mkdir -p "$RESULTS_DIR" "$REPO_DIR/bench/logs"
cp -- "${BASH_SOURCE[0]}" "$RESULTS_DIR/" 2>/dev/null || true

{
    echo "exp_name: $EXP_NAME"
    echo "job_id: ${SLURM_JOB_ID:-unknown}"
    echo "timestamp: $TIMESTAMP"
    echo "container_image: $CONTAINER_IMAGE"
    echo "concurrency: $CONCURRENCY"
} > "$RESULTS_DIR/job_info.txt"

echo "============================================"
echo "$EXP_NAME (job $SLURM_JOB_ID) on $NODE0  CONCURRENCY=$CONCURRENCY"
echo "Container: $CONTAINER_IMAGE"
echo "Results:   $RESULTS_DIR"
echo "============================================"

# ---------- engine.yaml (aggregated TP1) ----------
cat > "$RESULTS_DIR/engine.yaml" << 'EOF'
tensor_parallel_size: 1
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 256
kv_cache_config:
  dtype: fp8
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.9
moe_config:
  backend: TRTLLM
max_seq_len: 131072
max_batch_size: 256
max_num_tokens: 20000
num_postprocess_workers: 4
print_iter_log: true
enable_chunked_prefill: true
gpus_per_node: 8
trust_remote_code: true
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model: /lustre/fsw/core_dlfw_ci/rihuo/nvidia_gpt-oss-120b-Eagle3-v3
  eagle3_one_model: true
  eagle3_layers_to_capture: [23, 29, 35]
EOF

echo "Engine:"; cat "$RESULTS_DIR/engine.yaml"; echo

COMMON_ENV="export TRTLLM_SERVER_DISABLE_GC=1 && \
export TRTLLM_WORKER_DISABLE_GC=1 && \
export TIKTOKEN_ENCODINGS_BASE=/lustre/fsw/core_dlfw_ci/rihuo/tiktoken_encodings"

# --- aggregated worker on $NODE0 GPU 0 (TP1) port 8000 ---
echo "[$(date +%H:%M:%S)] Starting trtllm-serve aggregated (TP1) on $NODE0 GPU 0 port 8000..."
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=0 && $COMMON_ENV && \
    trtllm-llmapi-launch trtllm-serve $MODEL_PATH --extra_llm_api_options $RESULTS_DIR/engine.yaml \
      --host 0.0.0.0 --port 8000 --tp_size 1" \
  > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!

# --- health wait ---
echo "[$(date +%H:%M:%S)] Waiting for trtllm-serve to become healthy..."
HEALTHY=0
for i in $(seq 1 1200); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$NODE0:8000/health 2>/dev/null)
  if [ "$CODE" = "200" ]; then echo "[$(date +%H:%M:%S)] Health OK"; HEALTHY=1; break; fi
  sleep 3
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "ERROR: not healthy"
  for f in $RESULTS_DIR/*.log; do echo "=== $(basename $f) ==="; tail -50 "$f"; done
  kill $SERVER_PID 2>/dev/null; wait 2>/dev/null; exit 1
fi

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
send_conversation_routing_headers: false
exp_prefix: ${EXP_NAME}
results_dir: $RESULTS_DIR/rwlt_results
RWLTEOF

echo "==== Running RWLT @ c=${CONCURRENCY} ===="
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && $COMMON_ENV && \
    uv run --isolated --with openai --with httpx --with pyyaml --with pydantic \
      python rwlt/run.py --config $RESULTS_DIR/rwlt_config.yaml" \
  > "$RESULTS_DIR/benchmark.log" 2>&1
echo "Bench exit: $?"

# --- shutdown ---
echo "[$(date +%H:%M:%S)] Stopping worker..."
kill -TERM $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "==== DONE ===="
echo "Outputs:"
echo "  RWLT summary:    $RESULTS_DIR/rwlt_results/*.txt"
echo "  Logs:            $RESULTS_DIR/{server,benchmark}.log"
