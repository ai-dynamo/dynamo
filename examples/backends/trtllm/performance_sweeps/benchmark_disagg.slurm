#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
MULTI_ROUND="${MULTI_ROUND:-8}"

# set MOUNT_DIR
MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=disaggr-test


STREAMING=true
CTX_GPU_FRAC=0.75
CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448

num_ctx_servers=$1
ctx_tp_size=$2
ctx_batch_size=$3
ctx_max_num_tokens=$4
ctx_enable_attention_dp=$5
num_gen_servers=$6
gen_tp_size=$7
gen_batch_size=$8
gen_max_num_tokens=$9
gen_enable_attention_dp=${10}
gen_gpu_memory_fraction=${11}
eplb_num_slots=${12}
mtp_size=${13}
concurrency_list=${14}
gen_nodes=${15}
kind=${16}
model_path=${17}
served_model_name=${18}
image=${19}
isl=${20}
osl=${21}

ctx_max_seq_len=$((${isl} + 203))
gen_max_seq_len=$((${isl} + ${osl} + 203))

WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/${kind}-bm-${isl}-${osl}
SCRIPTS_DIR=${WORK_DIR}/
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

sub_dir=${LOG_DIR}/ctx${num_ctx_servers}_gen${num_gen_servers}_dep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}

echo "concurrency_list: ${concurrency_list}"

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    sub_dir=${LOG_DIR}/ctx${num_ctx_servers}_gen${num_gen_servers}_tep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
fi

full_logdir=${sub_dir}
artifacts_dir=${full_logdir}/genai_perf_artifacts
mkdir -p ${artifacts_dir}


# Set clock
srun ${set_clock_cmd}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${model_path}:${model_path}

# start the container
srun -l --container-image=${image} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up."

# generate the yaml file
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
	-n 1 -N 1 \
        python3 ${SCRIPTS_DIR}/scripts/gen_yaml.py --config ${full_logdir}/config.yaml \
            --model ${model_path} \
            --num_ctx_servers ${num_ctx_servers} \
            --ctx_tp_size ${ctx_tp_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --ctx_max_seq_len ${ctx_max_seq_len} \
            --ctx_free_gpu_memory_fraction ${CTX_GPU_FRAC} \
            --cache_transceiver_max_num_tokens ${CACHE_TRANSCEIVER_MAX_NUM_TOKENS} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_batch_size ${gen_batch_size} \
            --gen_max_num_tokens ${gen_max_num_tokens} \
            --gen_max_seq_len ${gen_max_seq_len} \
            --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
            --eplb_num_slots ${eplb_num_slots} \
            $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
            $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
            $(if [ "${mtp_size}" -gt 0 ]; then echo "--mtp_size ${mtp_size}"; fi)

echo "YAML file generated."

nsys_on=""
# nsys_on=${full_logdir}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

export HEAD_NODE="${nodes[0]}"
export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

# Create a temporary file to store PIDs
PID_FILE=$(mktemp)
trap 'cleanup_and_exit' EXIT

cleanup_and_exit() {
    if [ -f "$PID_FILE" ]; then
        echo "Cleaning up spawned processes..."
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Sending TERM to process $pid"
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Process $pid still running, sending KILL"
                    kill -KILL "$pid" 2>/dev/null
                fi
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
}

# start the server
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
	--oversubscribe \
	--overlap \
	--container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/start_frontend.sh &> ${full_logdir}/output_server.log &
SERVER_PID=$!
echo "$SERVER_PID" >> "$PID_FILE"

# wait for the server to start
sleep 10

PREFILL_COUNT=$(grep 'prefill_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$PREFILL_COUNT" ]; then
    echo "Error: Failed to extract prefill_count from instance_config.yaml"
    exit 1
fi
echo "Prefill Count: $PREFILL_COUNT"

# start the prefill workers
prefill_pids=()
for ((i=1; i<=PREFILL_COUNT; i++)); do
  echo "Running Prefill Worker: ${i}"
  node_idx=$((i-1))
  echo "Running Prefill Nodes: ${nodes[node_idx]}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix --overlap -w ${nodes[node_idx]} \
      --oversubscribe \
      --overlap \
      --ntasks 4 \
      --nodes 1 \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/prefill_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${served_model_name} ${model_path} 'prefill' &> ${full_logdir}/output_workers.log &
  prefill_pids+=($!)
  echo "$!" >> "$PID_FILE"
done

DECODE_COUNT=$(grep 'decode_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$DECODE_COUNT" ]; then
    echo "Error: Failed to extract decode_count from instance_config.yaml"
    exit 1
fi
echo "Decode Count: $DECODE_COUNT"

num_gen_nodes=$((gen_nodes/num_gen_servers))
decode_start_idx=$PREFILL_COUNT
for ((i=1; i<=DECODE_COUNT; i++)); do
  echo "Running Decode Worker: ${i}"
  decode_node_list=()
  for ((j=0; j<num_gen_nodes; j++)); do
    node_idx=$((decode_start_idx + (i-1)*num_gen_nodes + j))
    decode_node_list+=("${nodes[node_idx]}")
  done
  decode_nodes_csv=$(IFS=, ; echo "${decode_node_list[*]}")
  echo "Running Decode Nodes: ${decode_nodes_csv}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix \
      -w ${decode_nodes_csv} \
      --nodes ${num_gen_nodes} \
      --ntasks $gen_tp_size \
      --oversubscribe \
      --overlap \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/decode_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${served_model_name} ${model_path} 'decode' &> ${full_logdir}/output_workers.log &
  echo "$!" >> "$PID_FILE"
done

total_gpus=$((ctx_gpus + gen_gpus))

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts},${artifacts_dir}:${artifacts_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
	-w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${served_model_name} ${MULTI_ROUND} ${num_gen_servers} "${concurrency_list}" ${STREAMING} ${full_logdir} ${total_gpus} ${artifacts_dir} ${model_path} ${isl} ${osl} ${kind} > ${full_logdir}/bench.log 2>&1


# Cleanup will be handled by the EXIT trap
