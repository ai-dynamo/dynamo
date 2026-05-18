#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=glm5_nvfp4_ISL8K_OSL1K_ctx1dep2_gen4tep8_batch4_allconc_eplb0_mtp3
#SBATCH --nodes=10
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=10
#SBATCH --account=restricted
#SBATCH --time=0
#SBATCH --output=/data/home/rihuo/srt-slurm/outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch_2

set -euo pipefail

SRTCTL_SOURCE="/data/home/rihuo/srt-slurm"
OUTPUT_BASE="/data/home/rihuo/srt-slurm/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
CONTAINER_IMAGE="/data/home/rihuo/dynamo-vllm-rihuo-arm64-1-2-0-1-3-0rc15-tot.sqsh"
EVAL_CONTAINER_IMAGE="/data/home/rihuo/sglang-v0.5.10.post1-cu130.sqsh"
MODEL_PATH="/data/home/rihuo/nvidia_GLM-5-NVFP4"
MODEL_NAME="nvidia_GLM-5-NVFP4"
INFMAX_WORKSPACE="/data/home/rihuo/InferenceMAX"
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks,${INFMAX_WORKSPACE}:/infmax-workspace"
TRTLLM_COMMON_ENV="export ENROOT_ALLOW_DEV=yes && export MIMALLOC_PURGE_DELAY=0 && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_ENABLE_PDL=1 && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1"

mkdir -p "${LOG_DIR}"
exec 2>&1

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
if [ "${#ALL_NODES[@]}" -lt 10 ]; then
    echo "ERROR: expected at least 10 nodes, got ${#ALL_NODES[@]}"
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"
PREFILL_NODE="${ALL_NODES[1]}"
DECODE_NODES_0=("${ALL_NODES[2]}" "${ALL_NODES[3]}")
DECODE_NODES_1=("${ALL_NODES[4]}" "${ALL_NODES[5]}")
DECODE_NODES_2=("${ALL_NODES[6]}" "${ALL_NODES[7]}")
DECODE_NODES_3=("${ALL_NODES[8]}" "${ALL_NODES[9]}")
DECODE_NODELIST_0="$(IFS=,; echo "${DECODE_NODES_0[*]}")"
DECODE_NODELIST_1="$(IFS=,; echo "${DECODE_NODES_1[*]}")"
DECODE_NODELIST_2="$(IFS=,; echo "${DECODE_NODES_2[*]}")"
DECODE_NODELIST_3="$(IFS=,; echo "${DECODE_NODES_3[*]}")"

CTX0_PORT=8001
GEN0_PORT=8002
GEN1_PORT=8003
GEN2_PORT=8004
GEN3_PORT=8005
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
max_batch_size: 2
max_num_tokens: 16640
max_seq_len: 8232
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
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
EOF
}

write_decode_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
allreduce_strategy: MNNVL
tensor_parallel_size: 8
moe_expert_parallel_size: 8
pipeline_parallel_size: 1
enable_attention_dp: false
enable_lm_head_tp_in_adp: false
trust_remote_code: true
max_batch_size: 4
max_num_tokens: 16
max_seq_len: 9256
print_iter_log: true
stream_interval: 100
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
nvfp4_gemm_config:
  allowed_backends:
    - cutlass
    - cublaslt
    - cutedsl
    - cuda_core
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
EOF
}

write_disagg_config() {
    local output_path="$1"
    cat > "${output_path}" <<EOF
hostname: 0.0.0.0
port: ${DISAGG_PORT}
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "${PREFILL_NODE}:${CTX0_PORT}"
generation_servers:
  num_instances: 4
  urls:
    - "${DECODE_NODES_0[0]}:${GEN0_PORT}"
    - "${DECODE_NODES_1[0]}:${GEN1_PORT}"
    - "${DECODE_NODES_2[0]}:${GEN2_PORT}"
    - "${DECODE_NODES_3[0]}:${GEN3_PORT}"
EOF
}

echo "=========================================="
echo "🚀 TensorRT-LLM Disaggregated Sweep"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Head node: ${HEAD_NODE}"
echo "Prefill node: ${PREFILL_NODE}"
echo "Decode nodes: ${DECODE_NODELIST_0}, ${DECODE_NODELIST_1}, ${DECODE_NODELIST_2}, ${DECODE_NODELIST_3}"
echo ""

write_prefill_config "${LOG_DIR}/trtllm_prefill.yaml"
write_decode_config "${LOG_DIR}/trtllm_decode.yaml"
write_disagg_config "${LOG_DIR}/trtllm_disagg.yaml"

echo "Starting context server"
start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE}" \
    --output "${LOG_DIR}/${PREFILL_NODE}_prefill_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && export CUDA_VISIBLE_DEVICES=0,1 && trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${CTX0_PORT} --extra_llm_api_options /logs/trtllm_prefill.yaml"
CTX0_PID="${SRUN_PIDS[-1]}"

echo "Starting generation servers"
start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_0}" \
    --output "${LOG_DIR}/${DECODE_NODES_0[0]}_decode_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${GEN0_PORT} --extra_llm_api_options /logs/trtllm_decode.yaml"
GEN0_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_1}" \
    --output "${LOG_DIR}/${DECODE_NODES_1[0]}_decode_w1.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${GEN1_PORT} --extra_llm_api_options /logs/trtllm_decode.yaml"
GEN1_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_2}" \
    --output "${LOG_DIR}/${DECODE_NODES_2[0]}_decode_w2.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${GEN2_PORT} --extra_llm_api_options /logs/trtllm_decode.yaml"
GEN2_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_3}" \
    --output "${LOG_DIR}/${DECODE_NODES_3[0]}_decode_w3.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && trtllm-llmapi-launch trtllm-serve serve /model --backend pytorch --host 0.0.0.0 --port ${GEN3_PORT} --extra_llm_api_options /logs/trtllm_decode.yaml"
GEN3_PID="${SRUN_PIDS[-1]}"

for pid_name in CTX0_PID GEN0_PID GEN1_PID GEN2_PID GEN3_PID; do
    require_alive "${!pid_name}" "${pid_name}"
done

echo "Waiting for context and generation servers"
wait_for_http_ok "${PREFILL_NODE}" "${CTX0_PORT}" "/health" 2700
wait_for_http_ok "${DECODE_NODES_0[0]}" "${GEN0_PORT}" "/health" 2700
wait_for_http_ok "${DECODE_NODES_1[0]}" "${GEN1_PORT}" "/health" 2700
wait_for_http_ok "${DECODE_NODES_2[0]}" "${GEN2_PORT}" "/health" 2700
wait_for_http_ok "${DECODE_NODES_3[0]}" "${GEN3_PORT}" "/health" 2700

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
    bash -c "${TRTLLM_COMMON_ENV} && trtllm-serve disaggregated -c /logs/trtllm_disagg.yaml -t 7200 -r 7200"
DISAGG_PID="${SRUN_PIDS[-1]}"
require_alive "${DISAGG_PID}" "DISAGG_PID"

echo "Waiting for disaggregated server to be ready..."
if ! wait_for_http_ok "${HEAD_NODE}" "${DISAGG_PORT}" "/health" 2700; then
    echo "ERROR: disaggregated server did not become healthy"
    exit 1
fi

echo "Server is healthy - starting post eval (lm-eval)"
srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/eval.out" \
    --container-image "${EVAL_CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    --export="ALL,MODEL_NAME=${MODEL_NAME},EVAL_CONC=24,RUN_EVAL=true,IS_MULTINODE=true,FRAMEWORK=trtllm,PRECISION=fp4,MODEL=/model,PREFILL_TP=2,PREFILL_EP=2,PREFILL_DP_ATTN=true,PREFILL_NUM_WORKERS=1,DECODE_TP=8,DECODE_EP=8,DECODE_DP_ATTN=false,DECODE_NUM_WORKERS=4" \
    bash -c "bash /srtctl-benchmarks/lm-eval/bench.sh http://localhost:${DISAGG_PORT} /infmax-workspace"
