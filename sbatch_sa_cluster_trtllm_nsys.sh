#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm5_nvfp4_ISL1K_OSL1K_ctx6dep2_gen1dep32_batch256_eplb288_mtp0_nsys
#SBATCH --nodes=12
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=12
#SBATCH --account=restricted
#SBATCH --time=4:00:00
#SBATCH --output=/data/home/rihuo/srt-slurm/outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch_2

set -euo pipefail

SRTCTL_SOURCE="/data/home/rihuo/srt-slurm"
OUTPUT_BASE="/data/home/rihuo/srt-slurm/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
CONTAINER_IMAGE="/data/home/rihuo/tensorrtllm-runtime-1-1-0-dev-3.sqsh"
MODEL_PATH="/data/home/rihuo/nvidia_GLM-5-NVFP4"
MODEL_NAME="nvidia_GLM-5-NVFP4"
NSYS_PROFILE_DIR="/data/home/rihuo/nsys-profile/${SLURM_JOB_ID}"
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks,${NSYS_PROFILE_DIR}:/nsys-profile"
NSYS_DELAY_SECONDS=2100
NSYS_DURATION_SECONDS=300
NSYS_PREFIX_BASE="nsys profile --trace=cuda,nvtx,osrt,python-gil --python-sampling=true --python-sampling-frequency=100 --delay=${NSYS_DELAY_SECONDS} --duration=${NSYS_DURATION_SECONDS} --cuda-graph-trace=node --force-overwrite=true --trace-fork-before-exec=true"
TRTLLM_COMMON_ENV="export ENROOT_ALLOW_DEV=yes && export MIMALLOC_PURGE_DELAY=0 && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_ENABLE_PDL=1 && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1"

mkdir -p "${LOG_DIR}"
mkdir -p "${NSYS_PROFILE_DIR}"
exec 2>&1

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
if [ "${#ALL_NODES[@]}" -lt 12 ]; then
    echo "ERROR: expected at least 12 nodes, got ${#ALL_NODES[@]}"
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"
PREFILL_NODE_A="${ALL_NODES[1]}"
PREFILL_NODE_B="${ALL_NODES[2]}"
PREFILL_NODE_C="${ALL_NODES[3]}"
DECODE_NODES=("${ALL_NODES[@]:4:8}")
DECODE_NODELIST="$(IFS=,; echo "${DECODE_NODES[*]}")"

CTX0_PORT=8001
CTX1_PORT=8002
CTX2_PORT=8003
CTX3_PORT=8004
CTX4_PORT=8005
CTX5_PORT=8006
GEN_PORT=8007
DISAGG_PORT=8000

SRUN_PIDS=()

cleanup() {
    local exit_code=$?

    if [ "${#SRUN_PIDS[@]}" -gt 0 ]; then
        echo ""
        echo "Cleaning up ${#SRUN_PIDS[@]} background srun steps..."
        kill "${SRUN_PIDS[@]}" 2>/dev/null || true
        wait "${SRUN_PIDS[@]}" 2>/dev/null || true
    fi

    if [ $exit_code -eq 0 ]; then
        echo "✓ Sweep completed successfully"
    else
        echo "✗ Sweep failed (exit code: $exit_code)"
    fi
    echo "End: $(date)"
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

wait_for_http_ok() {
    local host="$1"
    local port="$2"
    local path="${3:-/health}"
    local timeout="${4:-2700}"
    local deadline=$((SECONDS + timeout))

    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -fsS --max-time 5 "http://${host}:${port}${path}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done

    echo "ERROR: endpoint http://${host}:${port}${path} is not ready after waiting ${timeout} seconds"
    return 1
}

write_prefill_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
tensor_parallel_size: 2
moe_expert_parallel_size: 2
pipeline_parallel_size: 1
enable_attention_dp: true
disable_overlap_scheduler: true
trust_remote_code: true
custom_tokenizer: "glm_moe_dsa"
max_batch_size: 16
max_num_tokens: 16384
max_seq_len: 1064
print_iter_log: true
cuda_graph_config: null
moe_config:
  backend: CUTEDSL
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.6
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
EOF
}

write_decode_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
tensor_parallel_size: 32
moe_expert_parallel_size: 32
pipeline_parallel_size: 1
enable_attention_dp: true
enable_lm_head_tp_in_adp: false
trust_remote_code: true
custom_tokenizer: "glm_moe_dsa"
max_batch_size: 256
max_num_tokens: 256
max_seq_len: 2088
print_iter_log: true
stream_interval: 100
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 24
    - 32
    - 40
    - 48
    - 56
    - 64
    - 72
    - 80
    - 88
    - 96
    - 104
    - 112
    - 120
    - 128
    - 136
    - 144
    - 152
    - 160
    - 168
    - 176
    - 184
    - 192
    - 200
    - 208
    - 216
    - 224
    - 232
    - 240
    - 248
    - 256
moe_config:
  backend: CUTEDSL
  use_low_precision_moe_combine: true
  load_balancer:
    layer_updates_per_iter: 1
    num_slots: 288
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.7
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
nvfp4_gemm_config:
  allowed_backends:
    - cutlass
    - cublaslt
    - cutedsl
    - cuda_core
EOF
}

write_disagg_config() {
    local output_path="$1"
    cat > "${output_path}" <<EOF
hostname: 0.0.0.0
port: ${DISAGG_PORT}
backend: pytorch
context_servers:
  num_instances: 6
  urls:
    - "${PREFILL_NODE_A}:${CTX0_PORT}"
    - "${PREFILL_NODE_A}:${CTX1_PORT}"
    - "${PREFILL_NODE_B}:${CTX2_PORT}"
    - "${PREFILL_NODE_B}:${CTX3_PORT}"
    - "${PREFILL_NODE_C}:${CTX4_PORT}"
    - "${PREFILL_NODE_C}:${CTX5_PORT}"
generation_servers:
  num_instances: 1
  urls:
    - "${DECODE_NODES[0]}:${GEN_PORT}"
EOF
}

echo "=========================================="
echo "🚀 TensorRT-LLM Disaggregated Sweep (nsys)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Nsys profile dir: ${NSYS_PROFILE_DIR} (delay=${NSYS_DELAY_SECONDS}s, duration=${NSYS_DURATION_SECONDS}s)"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Head node: ${HEAD_NODE}"
echo "Prefill nodes: ${PREFILL_NODE_A}, ${PREFILL_NODE_B}, ${PREFILL_NODE_C}"
echo "Decode nodes: ${DECODE_NODELIST}"
echo ""

write_prefill_config "${LOG_DIR}/trtllm_prefill.yaml"
write_decode_config "${LOG_DIR}/trtllm_decode.yaml"
write_disagg_config "${LOG_DIR}/trtllm_disagg.yaml"

echo "Starting context servers"
start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_A}" \
    --output "${LOG_DIR}/${PREFILL_NODE_A}_prefill_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=0,1 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_0_${PREFILL_NODE_A}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX0_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX0_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_A}" \
    --output "${LOG_DIR}/${PREFILL_NODE_A}_prefill_w1.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=2,3 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_1_${PREFILL_NODE_A}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX1_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX1_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_B}" \
    --output "${LOG_DIR}/${PREFILL_NODE_B}_prefill_w2.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=0,1 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_2_${PREFILL_NODE_B}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX2_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX2_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_B}" \
    --output "${LOG_DIR}/${PREFILL_NODE_B}_prefill_w3.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=2,3 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_3_${PREFILL_NODE_B}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX3_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX3_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_C}" \
    --output "${LOG_DIR}/${PREFILL_NODE_C}_prefill_w4.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=0,1 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_4_${PREFILL_NODE_C}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX4_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX4_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE_C}" \
    --output "${LOG_DIR}/${PREFILL_NODE_C}_prefill_w5.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=2,3 && ${NSYS_PREFIX_BASE} --output /nsys-profile/prefill_5_${PREFILL_NODE_C}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX5_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX5_PID="${SRUN_PIDS[-1]}"

echo "Starting generation server"
start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 8 \
    --ntasks 32 \
    --nodelist "${DECODE_NODELIST}" \
    --output "${LOG_DIR}/${DECODE_NODES[0]}_decode_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${NSYS_PREFIX_BASE} --output /nsys-profile/decode_w0_node%q{SLURMD_NODENAME}_rank%q{SLURM_PROCID} trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${GEN_PORT} --extra_llm_api_options /logs/trtllm_decode.yaml"
GEN_PID="${SRUN_PIDS[-1]}"

for pid_name in CTX0_PID CTX1_PID CTX2_PID CTX3_PID CTX4_PID CTX5_PID GEN_PID; do
    require_alive "${!pid_name}" "${pid_name}"
done

echo "Waiting for context and generation servers"
wait_for_http_ok "${PREFILL_NODE_A}" "${CTX0_PORT}" "/health" 2700
wait_for_http_ok "${PREFILL_NODE_A}" "${CTX1_PORT}" "/health" 2700
wait_for_http_ok "${PREFILL_NODE_B}" "${CTX2_PORT}" "/health" 2700
wait_for_http_ok "${PREFILL_NODE_B}" "${CTX3_PORT}" "/health" 2700
wait_for_http_ok "${PREFILL_NODE_C}" "${CTX4_PORT}" "/health" 2700
wait_for_http_ok "${PREFILL_NODE_C}" "${CTX5_PORT}" "/health" 2700
wait_for_http_ok "${DECODE_NODES[0]}" "${GEN_PORT}" "/health" 2700

echo "Starting disaggregated orchestration server"
start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/${HEAD_NODE}_disagg.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${NSYS_PREFIX_BASE} --output /nsys-profile/disagg_${HEAD_NODE} trtllm-serve disaggregated -c /logs/trtllm_disagg.yaml -t 7200 -r 7200"
DISAGG_PID="${SRUN_PIDS[-1]}"
require_alive "${DISAGG_PID}" "DISAGG_PID"

echo "Waiting for disaggregated server to be ready..."
if ! wait_for_http_ok "${HEAD_NODE}" "${DISAGG_PORT}" "/health" 2700; then
    echo "ERROR: disaggregated server did not become healthy"
    exit 1
fi

echo "Server is healthy - starting benchmark"
srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/benchmark.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "export SRTCTL_FRONTEND_TYPE=dynamo && bash /srtctl-benchmarks/sa-bench/bench.sh http://localhost:${DISAGG_PORT} 1024 1024 8192 inf /model ${MODEL_NAME} true 44 12 32 0.8 16 2 glm_moe_dsa true random"
