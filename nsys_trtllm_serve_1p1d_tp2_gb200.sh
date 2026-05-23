#!/bin/bash
#SBATCH --job-name=core_dlfw_ci-nsys.trtllm_serve.1p1d_tp2
#SBATCH --nodes=1
#SBATCH --partition=gb200
#SBATCH --account=core_dlfw_ci
#SBATCH --time=04:00:00
#SBATCH --output=bench/logs/nsys_trtllm_serve_1p1d_tp2_gb200_%j.log
#SBATCH --error=bench/logs/nsys_trtllm_serve_1p1d_tp2_gb200_%j.err

# =============================================================================
# nsys-profiled trtllm-serve 1 ctx (TP1) + 1 gen (TP2) on GB200.
#
# Launch approach: trtllm-serve disaggregated (master + ctx + gen workers),
# mirroring bench/trtllm_serve.sh. Dir/path/env setup mirrors
# run_benchx_1ctx1gen_dynamo_kvrouter.sh (lustre/fsw/core_dlfw_ci/rihuo paths,
# arm64 dynamo-trtllm container — trtllm-serve is the underlying TRT-LLM tool).
#
# Layout:
#   NODE0 GPU 0       — ctx worker  (TP1, port 8001)
#   NODE0 GPUs 1,2    — gen worker  (TP2, port 8002)
#   NODE0             — trtllm-serve disaggregated master (port 8000)
#
# Each worker process is wrapped with `nsys profile`. Per-rank .nsys-rep files
# are emitted to $RESULTS_DIR/nsys/.
#
# Env:
#   CONCURRENCY    — single concurrency for the profile run (default 16; keep
#                    small to bound report size)
#   HOSTCACHE      — 1 = enable kv_cache_config.host_cache_size: 80GB on ctx
#                    0 = no host offloading (default)
#   NSYS_DELAY     — seconds to wait before starting capture (default 0 = full
#                    capture; set e.g. 600 to skip warmup)
#   NSYS_DURATION  — seconds of capture (default empty = until process exits)
#
# Submit:
#   sbatch --export=ALL,CONCURRENCY=16 bench/nsys_trtllm_serve_1p1d_tp2_gb200.sh
# =============================================================================

set -uo pipefail

CONCURRENCY="${CONCURRENCY:-48}"
HOSTCACHE="${HOSTCACHE:-0}"
NSYS_DELAY="${NSYS_DELAY:-0}"
NSYS_DURATION="${NSYS_DURATION:-}"

if [ "$HOSTCACHE" = "1" ]; then HCTAG="hcon"; else HCTAG="hcoff"; fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/core_dlfw_ci/rihuo/dynamo-trtllm-rihuo-arm64-1-2-0-0dd537-publisherfix.sqsh}"
EXP_NAME="nsys_trtllm_serve_1p1d_tp2_gb200_${HCTAG}_c${CONCURRENCY}"

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
RESULTS_DIR="$REPO_DIR/bench/results/nsys/${RUN_TAG}"
NSYS_RESULT_DIR="$REPO_DIR/bench/nsys_reports/${RUN_TAG}"
mkdir -p "$RESULTS_DIR" "$NSYS_RESULT_DIR" "$REPO_DIR/bench/logs"
cp -- "${BASH_SOURCE[0]}" "$RESULTS_DIR/" 2>/dev/null || true

{
    echo "exp_name: $EXP_NAME"
    echo "job_id: ${SLURM_JOB_ID:-unknown}"
    echo "timestamp: $TIMESTAMP"
    echo "container_image: $CONTAINER_IMAGE"
    echo "concurrency: $CONCURRENCY"
    echo "nsys_delay: $NSYS_DELAY"
    echo "nsys_duration: ${NSYS_DURATION:-(full)}"
} > "$RESULTS_DIR/job_info.txt"

echo "============================================"
echo "$EXP_NAME (job $SLURM_JOB_ID) on $NODE0  CONCURRENCY=$CONCURRENCY HOSTCACHE=$HOSTCACHE"
echo "Container: $CONTAINER_IMAGE"
echo "Results:   $RESULTS_DIR"
echo "nsys dir:  $NSYS_RESULT_DIR"
echo "============================================"

# ---------- master.yaml: ctx TP1 (port 8001) + gen TP2 (port 8002) ----------
cat > "$RESULTS_DIR/master.yaml" << EOF
hostname: "0.0.0.0"
port: 8000
context_servers:
  num_instances: 1
  urls:
    - "${NODE0}:8001"
generation_servers:
  num_instances: 1
  urls:
    - "${NODE0}:8002"
EOF

# ---------- ctx.yaml (inlined from bench/ctx_config.yaml) ----------
cat > "$RESULTS_DIR/ctx.yaml" << 'EOF'
backend: pytorch
tensor_parallel_size: 2
pipeline_parallel_size: 1
moe_expert_parallel_size: 1
max_batch_size: 128
max_num_tokens: 20000
max_seq_len: 131072
stream_interval: 10
trust_remote_code: true
disable_overlap_scheduler: false
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
  free_gpu_memory_fraction: 0.9
  enable_block_reuse: true
torch_compile_config:
  enable_fullgraph: true
  enable_piecewise_cuda_graph: true
  enable_userbuffers: false
  capture_num_tokens:
  - 128
  - 256
  - 384
  - 512
  - 640
  - 768
  - 896
  - 1024
  - 1152
  - 1280
  - 1408
  - 1536
  - 1664
  - 1792
  - 1920
  - 2048
  - 2176
  - 2304
  - 2432
  - 2560
  - 2688
  - 2816
  - 2944
  - 3072
  - 3200
  - 3328
  - 3456
  - 3584
  - 3712
  - 3840
  - 3968
  - 4096
  - 4224
  - 4352
  - 4480
  - 4608
  - 4736
  - 4864
  - 4992
  - 5120
  - 5248
  - 5376
  - 5504
  - 5632
  - 5760
  - 5888
  - 6016
  - 6144
  - 6272
  - 6400
  - 6528
  - 6656
  - 6784
  - 6912
  - 7040
  - 7168
  - 7296
  - 7424
  - 7552
  - 7680
  - 7808
  - 7936
  - 8064
  - 8192
  - 8320
  - 8448
  - 8576
  - 8704
  - 8832
  - 8960
  - 9088
  - 9216
  - 9344
  - 9472
  - 9600
  - 9728
  - 9856
  - 9984
  - 10112
  - 10240
  - 10368
  - 10496
  - 10624
  - 10752
  - 10880
  - 11008
  - 11136
  - 11264
  - 11392
  - 11520
  - 11648
  - 11776
  - 11904
  - 12032
  - 12160
  - 12288
  - 12416
  - 12544
  - 12672
  - 12800
  - 12928
  - 13056
  - 13184
  - 13312
  - 13440
  - 13568
  - 13696
  - 13824
  - 13952
  - 14080
  - 14208
  - 14336
  - 14464
  - 14592
  - 14720
  - 14848
  - 14976
  - 15104
  - 15232
  - 15360
  - 15488
  - 15616
  - 15744
  - 15872
  - 16000
  - 16128
  - 16256
  - 16384
  - 16512
  - 16640
  - 16768
  - 16896
  - 17024
  - 17152
  - 17280
  - 17408
  - 17536
  - 17664
  - 17792
  - 17920
  - 18048
  - 18176
  - 18304
  - 18432
  - 18560
  - 18688
  - 18816
  - 18944
  - 19072
  - 19200
  - 19328
  - 19456
  - 19584
  - 19712
  - 19840
  - 19968
  - 20000
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
cache_transceiver_config:
  max_tokens_in_buffer: 131072
  backend: DEFAULT
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model: /lustre/fsw/core_dlfw_ci/rihuo/nvidia_gpt-oss-120b-Eagle3-v3
  eagle3_one_model: true
  eagle3_layers_to_capture:
  - 23
  - 29
  - 35
enable_iter_perf_stats: true
enable_iter_req_stats: false
print_iter_log: false
EOF
if [ "$HOSTCACHE" = "1" ]; then
    sed -i '/^kv_cache_config:/a\  host_cache_size: 85899345920' "$RESULTS_DIR/ctx.yaml"
fi

# ---------- gen.yaml (inlined from gen_config.yaml) ----------
cat > "$RESULTS_DIR/gen.yaml" << 'EOF'
backend: pytorch
tensor_parallel_size: 2
pipeline_parallel_size: 1
moe_expert_parallel_size: 1
max_batch_size: 128
max_num_tokens: 512
max_seq_len: 131072
stream_interval: 10
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
  free_gpu_memory_fraction: 0.9
  enable_block_reuse: true
torch_compile_config:
  enable_fullgraph: true
  enable_piecewise_cuda_graph: true
  enable_userbuffers: false
  capture_num_tokens:
  - 4
  - 8
  - 12
  - 16
  - 20
  - 24
  - 28
  - 32
  - 36
  - 40
  - 44
  - 48
  - 52
  - 56
  - 60
  - 64
  - 68
  - 72
  - 76
  - 80
  - 84
  - 88
  - 92
  - 96
  - 100
  - 104
  - 108
  - 112
  - 116
  - 120
  - 124
  - 128
  - 132
  - 136
  - 140
  - 144
  - 148
  - 152
  - 156
  - 160
  - 164
  - 168
  - 172
  - 176
  - 180
  - 184
  - 188
  - 192
  - 196
  - 200
  - 204
  - 208
  - 212
  - 216
  - 220
  - 224
  - 228
  - 232
  - 236
  - 240
  - 244
  - 248
  - 252
  - 256
  - 260
  - 264
  - 268
  - 272
  - 276
  - 280
  - 284
  - 288
  - 292
  - 296
  - 300
  - 304
  - 308
  - 312
  - 316
  - 320
  - 324
  - 328
  - 332
  - 336
  - 340
  - 344
  - 348
  - 352
  - 356
  - 360
  - 364
  - 368
  - 372
  - 376
  - 380
  - 384
  - 388
  - 392
  - 396
  - 400
  - 404
  - 408
  - 412
  - 416
  - 420
  - 424
  - 428
  - 432
  - 436
  - 440
  - 444
  - 448
  - 452
  - 456
  - 460
  - 464
  - 468
  - 472
  - 476
  - 480
  - 484
  - 488
  - 492
  - 496
  - 500
  - 504
  - 508
  - 512
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
cache_transceiver_config:
  max_tokens_in_buffer: 131072
  backend: DEFAULT
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model: /lustre/fsw/core_dlfw_ci/rihuo/nvidia_gpt-oss-120b-Eagle3-v3
  eagle3_one_model: true
  eagle3_layers_to_capture:
  - 23
  - 29
  - 35
enable_iter_perf_stats: true
enable_iter_req_stats: false
print_iter_log: false
EOF

echo "Master:"; cat "$RESULTS_DIR/master.yaml"; echo

COMMON_ENV="export TRTLLM_SERVER_DISABLE_GC=1 && \
export TRTLLM_WORKER_DISABLE_GC=1 && \
export TIKTOKEN_ENCODINGS_BASE=/lustre/fsw/core_dlfw_ci/rihuo/tiktoken_encodings"

# TRT-LLM profiler hooks — drive cudaProfilerStart/Stop window and NVTX detail.
# Active iteration window is the `start-stop` range below; outside that nsys
# (when launched with `-c cudaProfilerApi`) records nothing.
WORKER_PROFILE_ENV="export TLLM_PROFILE_RECORD_GC=1 && \
export TLLM_PROFILE_START_STOP=30-130 && \
export TLLM_NVTX_DEBUG=1 && \
export TLLM_LLMAPI_ENABLE_NVTX=1"

# ---------- nsys command builder ----------
# We profile each worker process. nsys writes one .nsys-rep per srun task, so
# for TP>1 we get one report per rank via %q{SLURM_PROCID}.
NSYS_BIN="${NSYS_BIN:-/lustre/fsw/core_dlfw_ci/rihuo/nsys-extracted/opt/nvidia/nsight-systems/2025.5.1/bin/nsys}"
NSYS_BASE_OPTS="-t cuda,nvtx --sample=none --cpuctxsw=none --backtrace=none --force-overwrite=true --cuda-graph-trace=node -c cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true"
if [ "$NSYS_DELAY" != "0" ] && [ -n "$NSYS_DELAY" ]; then
  NSYS_BASE_OPTS="$NSYS_BASE_OPTS --delay=${NSYS_DELAY}"
fi
if [ -n "$NSYS_DURATION" ]; then
  NSYS_BASE_OPTS="$NSYS_BASE_OPTS --duration=${NSYS_DURATION}"
fi

# --- gen worker on $NODE0 GPUs 2,3 (TP2) port 8002 ---
echo "[$(date +%H:%M:%S)] Starting gen worker (TP2) on $NODE0 GPUs 2,3 port 8002 under nsys..."
srun --overlap --ntasks=2 --ntasks-per-node=2 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=2,3 && $COMMON_ENV && $WORKER_PROFILE_ENV && \
    ${NSYS_BIN} profile ${NSYS_BASE_OPTS} \
      -o ${NSYS_RESULT_DIR}/gen_rank%q{SLURM_PROCID} \
      trtllm-llmapi-launch trtllm-serve $MODEL_PATH --extra_llm_api_options $RESULTS_DIR/gen.yaml \
        --host 0.0.0.0 --port 8002 --tp_size 2" \
  > "$RESULTS_DIR/gen.log" 2>&1 &
GEN_PID=$!

# --- ctx worker on $NODE0 GPUs 0,1 (TP2) port 8001 ---
echo "[$(date +%H:%M:%S)] Starting ctx worker (TP2) on $NODE0 GPUs 0,1 port 8001 under nsys..."
srun --overlap --ntasks=2 --ntasks-per-node=2 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "cd $REPO_DIR && export CUDA_VISIBLE_DEVICES=0,1 && $COMMON_ENV && $WORKER_PROFILE_ENV && \
    ${NSYS_BIN} profile ${NSYS_BASE_OPTS} \
      -o ${NSYS_RESULT_DIR}/ctx_rank%q{SLURM_PROCID} \
      trtllm-llmapi-launch trtllm-serve $MODEL_PATH --extra_llm_api_options $RESULTS_DIR/ctx.yaml \
        --host 0.0.0.0 --port 8001 --tp_size 2" \
  > "$RESULTS_DIR/ctx.log" 2>&1 &
CTX_PID=$!

# --- master (disaggregated server) ---
echo "[$(date +%H:%M:%S)] Starting trtllm-serve disaggregated master on $NODE0 port 8000..."
srun --overlap --ntasks=1 --nodes=1 --nodelist=$NODE0 --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir="$USER_DIR" --container-remap-root --container-writable \
  --no-container-entrypoint \
  bash -c "$COMMON_ENV && trtllm-serve disaggregated --config_file $RESULTS_DIR/master.yaml \
    --server_start_timeout 7200 --request_timeout 7200" \
  > "$RESULTS_DIR/master.log" 2>&1 &
MASTER_PID=$!

# --- health wait ---
echo "[$(date +%H:%M:%S)] Waiting for servers (1 ctx TP2 + 1 gen TP2)..."
HEALTHY=0
for i in $(seq 1 1200); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$NODE0:8000/health 2>/dev/null)
  if [ "$CODE" = "200" ]; then echo "[$(date +%H:%M:%S)] Health OK"; HEALTHY=1; break; fi
  sleep 3
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "ERROR: not healthy"
  for f in $RESULTS_DIR/*.log; do echo "=== $(basename $f) ==="; tail -50 "$f"; done
  kill $MASTER_PID $GEN_PID $CTX_PID 2>/dev/null; wait 2>/dev/null; exit 1
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

# --- shut workers down so nsys finalizes the .nsys-rep files ---
echo "[$(date +%H:%M:%S)] Stopping workers so nsys finalizes reports..."
kill -TERM $MASTER_PID $GEN_PID $CTX_PID 2>/dev/null
wait $MASTER_PID 2>/dev/null
wait $GEN_PID 2>/dev/null
wait $CTX_PID 2>/dev/null

echo "==== DONE ===="
echo "Outputs:"
echo "  RWLT summary:    $RESULTS_DIR/rwlt_results/*.txt"
echo "  nsys reports:    $NSYS_RESULT_DIR/*.nsys-rep"
echo "  Logs:            $RESULTS_DIR/{ctx,gen,master,benchmark}.log"
ls -lh "$NSYS_RESULT_DIR"/ 2>/dev/null || true
