#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo TRTLLM variant of sbatch_sa_cluster.sh
# Uses etcd + nats + dynamo frontend instead of trtllm-serve disaggregated
#
#SBATCH --job-name=glm5_nvfp4_dynamo_ISL1K_OSL1K_ctx6dep2_gen1dep32_batch256_eplb288_mtp0
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
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks"
TRTLLM_COMMON_ENV="export ENROOT_ALLOW_DEV=yes && export MIMALLOC_PURGE_DELAY=0 && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_ENABLE_PDL=1 && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1"

# Dynamo ports
ETCD_PORT=2379
NATS_PORT=4222
FRONTEND_PORT=8000
DYNAMO_REQUEST_PLANE=nats

# DYN_SYSTEM_PORT assignments (one per worker, starting at 8081)
DYN_SYS_PORT_CTX0=8081
DYN_SYS_PORT_CTX1=8082
DYN_SYS_PORT_CTX2=8083
DYN_SYS_PORT_CTX3=8084
DYN_SYS_PORT_CTX4=8085
DYN_SYS_PORT_CTX5=8086
DYN_SYS_PORT_GEN=8087

mkdir -p "${LOG_DIR}"
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

# Mirrors check_dynamo_health() in src/srtctl/core/health.py
# Parses dynamo /health JSON to count prefill and decode workers registered via etcd.
# TRTLLM decode workers report component="tensorrt_llm".
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
            # Count instances where endpoint=="generate" and component=="prefill"
            prefill_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(sum(1 for i in data.get('instances', [])
          if i.get('endpoint') == 'generate' and i.get('component') == 'prefill'))
" 2>/dev/null || echo "0")
            # Count instances where endpoint=="generate" and component is decode/tensorrt_llm/backend
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

# Query /v1/models to verify the model is loaded and serving
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

echo "=========================================="
echo "Dynamo TRTLLM Disaggregated Sweep"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Head node (infra + frontend): ${HEAD_NODE}"
echo "Prefill nodes: ${PREFILL_NODE_A}, ${PREFILL_NODE_B}, ${PREFILL_NODE_C}"
echo "Decode nodes: ${DECODE_NODELIST}"
echo ""

write_prefill_config "${LOG_DIR}/trtllm_prefill.yaml"
write_decode_config "${LOG_DIR}/trtllm_decode.yaml"

# ==============================================================================
# Stage 1: Start infrastructure services (etcd + nats) on head node
# ==============================================================================
echo "Starting etcd and nats on head node ${HEAD_NODE}..."

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/${HEAD_NODE}_infra.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
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

        # Wait for both services to be ready
        for i in $(seq 1 300); do
            if echo > /dev/tcp/localhost/'"${NATS_PORT}"' 2>/dev/null && \
               echo > /dev/tcp/localhost/'"${ETCD_PORT}"' 2>/dev/null; then
                echo "etcd and nats are ready"
                break
            fi
            sleep 1
        done

        echo "Infrastructure services running (nats PID: ${NATS_PID}, etcd PID: ${ETCD_PID})"

        # Keep running until killed
        wait
    '
INFRA_PID="${SRUN_PIDS[-1]}"
require_alive "${INFRA_PID}" "INFRA_PID"

echo "Waiting for nats (port ${NATS_PORT}) on ${HEAD_NODE}..."
wait_for_port "${HEAD_NODE}" "${NATS_PORT}" 300

echo "Waiting for etcd (port ${ETCD_PORT}) on ${HEAD_NODE}..."
wait_for_port "${HEAD_NODE}" "${ETCD_PORT}" 300
echo "Infrastructure services are ready"

# ==============================================================================
# Stage 2: Start TRTLLM workers via dynamo.trtllm
# ==============================================================================
DYNAMO_WORKER_ENV="export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=${DYNAMO_REQUEST_PLANE} && export DYN_TCP_WORKER_POOL_SIZE=256 && export DYN_TCP_WORK_QUEUE_SIZE=512 && export DYN_LOG=debug"

echo "Starting prefill workers (6x TP2/EP2)"

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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX0} && export CUDA_VISIBLE_DEVICES=0,1 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX1} && export CUDA_VISIBLE_DEVICES=2,3 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX2} && export CUDA_VISIBLE_DEVICES=0,1 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX3} && export CUDA_VISIBLE_DEVICES=2,3 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX4} && export CUDA_VISIBLE_DEVICES=0,1 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX5} && export CUDA_VISIBLE_DEVICES=2,3 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
CTX5_PID="${SRUN_PIDS[-1]}"

echo "Starting decode worker (1x TP32/EP32 across 8 nodes)"
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
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane ${DYNAMO_REQUEST_PLANE}"
GEN_PID="${SRUN_PIDS[-1]}"

for pid_name in CTX0_PID CTX1_PID CTX2_PID CTX3_PID CTX4_PID CTX5_PID GEN_PID; do
    require_alive "${!pid_name}" "${pid_name}"
done

# ==============================================================================
# Stage 3: Start dynamo frontend on head node
# ==============================================================================
echo "Starting dynamo frontend on ${HEAD_NODE}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/${HEAD_NODE}_frontend.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=${DYNAMO_REQUEST_PLANE} && export DYN_LOG=debug && python3 -m dynamo.frontend --http-port ${FRONTEND_PORT} --request-plane ${DYNAMO_REQUEST_PLANE}"
FRONTEND_PID="${SRUN_PIDS[-1]}"
require_alive "${FRONTEND_PID}" "FRONTEND_PID"

# ==============================================================================
# Stage 4: Wait for all workers to register, then verify model is serving
# ==============================================================================
EXPECTED_PREFILL=6
EXPECTED_DECODE=1

echo "Waiting for ${EXPECTED_PREFILL} prefill and ${EXPECTED_DECODE} decode workers to register..."
if ! wait_for_dynamo_workers "${HEAD_NODE}" "${FRONTEND_PORT}" "${EXPECTED_PREFILL}" "${EXPECTED_DECODE}" 2700 60; then
    echo "ERROR: workers did not become healthy"
    exit 1
fi

if ! verify_model_ready "${HEAD_NODE}" "${FRONTEND_PORT}" 120; then
    echo "ERROR: model is not serving"
    exit 1
fi

echo "All workers healthy and model is serving - starting benchmark"

# ==============================================================================
# Stage 5: Run benchmark
# ==============================================================================
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
    bash -c "export SRTCTL_FRONTEND_TYPE=dynamo && bash /srtctl-benchmarks/sa-bench/bench.sh http://localhost:${FRONTEND_PORT} 1024 1024 8192 inf /model ${MODEL_NAME} true 44 12 32 0.8 16 2 glm_moe_dsa true random"
