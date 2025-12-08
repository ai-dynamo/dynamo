#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
MULTI_ROUND="${MULTI_ROUND:-8}"

# set MOUNT_DIR
MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=disaggr-test-$(date +%s)-$$

STREAMING=true
CTX_GPU_FRAC=0.85

num_ctx_servers=$1
ctx_tp_size=$2
ctx_ep_size=$3
ctx_enable_attention_dp=$4
ctx_batch_size=$5
ctx_max_num_tokens=$6
num_gen_servers=$7
gen_tp_size=$8
gen_ep_size=$9
gen_batch_size=${10}
gen_max_num_tokens=${11}
gen_enable_attention_dp=${12}
gen_gpu_memory_fraction=${13}
eplb_num_slots=${14}
mtp_size=${15}
concurrency_list=${16}
gen_nodes=${17}
kind=${18}
model_path=${19}
served_model_name=${20}
image=${21}
isl=${22}
osl=${23}
benchmark_kind=${24}
ntasks_per_node=${25}

CACHE_TRANSCEIVER_MAX_NUM_TOKENS=${CACHE_TRANSCEIVER_MAX_NUM_TOKENS:-$((${isl} + ${osl} + 512))}

ctx_max_seq_len=$((${isl} + 203))
gen_max_seq_len=$((${isl} + ${osl} + 203))

WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/${kind}-bm-${isl}-${osl}
SCRIPTS_DIR=${WORK_DIR}/
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

sub_dir=${LOG_DIR}/ctx${num_ctx_servers}_gen${num_gen_servers}_dp${gen_tp_size}_ep${gen_ep_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}

echo "concurrency_list: ${concurrency_list}"

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

if [ "${gen_enable_attention_dp}" = "false" ]; then
    sub_dir=${LOG_DIR}/ctx${num_ctx_servers}_gen${num_gen_servers}_tp${gen_tp_size}_ep${gen_ep_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
fi

full_logdir=${sub_dir}
mkdir -p ${full_logdir}

# Set clock
#srun ${set_clock_cmd}

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
            --ctx_ep_size ${ctx_ep_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --ctx_max_seq_len ${ctx_max_seq_len} \
            --ctx_free_gpu_memory_fraction ${CTX_GPU_FRAC} \
            --cache_transceiver_max_num_tokens ${CACHE_TRANSCEIVER_MAX_NUM_TOKENS} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_ep_size ${gen_ep_size} \
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
export ETCD_ENDPOINTS="http://${HEAD_NODE}:2379"
export NATS_SERVER="nats://${HEAD_NODE}:4222"

echo "HEAD_NODE: ${HEAD_NODE}"
echo "HEAD_NODE_ID: ${HEAD_NODE_ID}"

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

DECODE_COUNT=$(grep 'decode_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$DECODE_COUNT" ]; then
    echo "Error: Failed to extract decode_count from instance_config.yaml"
    exit 1
fi
echo "Decode Count: $DECODE_COUNT"

num_gen_nodes=$((gen_nodes/num_gen_servers))
for ((i=1; i<=DECODE_COUNT; i++)); do
  echo "Running Decode Worker: ${i}"
  decode_node_list=()
  for ((j=0; j<num_gen_nodes; j++)); do
    node_idx=$(((i-1)*num_gen_nodes + j))
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
      --gpus-per-node $ntasks_per_node \
      --overlap \
      -e UCX_NET_DEVICES,TRTLLM_UCX_INTERFACE \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/decode_config.yaml ${ctx_gpus} ${served_model_name} ${model_path} 'decode' $gen_enable_attention_dp &> ${full_logdir}/output_decode_worker_${i}.log &
  echo "$!" >> "$PID_FILE"
done

PREFILL_COUNT=$(grep 'prefill_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$PREFILL_COUNT" ]; then
    echo "Error: Failed to extract prefill_count from instance_config.yaml"
    exit 1
fi
echo "Prefill Count: $PREFILL_COUNT"

# start the prefill workers
prefill_pids=()
prefill_start_idx=$gen_nodes
for ((i=1; i<=PREFILL_COUNT; i++)); do
  echo "Running Prefill Worker: ${i}"
  node_idx=$((prefill_start_idx + (i-1)))
  echo "Running Prefill Nodes: ${nodes[node_idx]}"
  srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -w ${nodes[node_idx]} \
        --oversubscribe \
        --overlap \
        --ntasks $(( ctx_tp_size < ntasks_per_node ? ctx_tp_size : ntasks_per_node )) \
        --gpus-per-node $ntasks_per_node \
        --nodes 1 \
        -e UCX_NET_DEVICES,TRTLLM_UCX_INTERFACE \
        bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/prefill_config.yaml ${ctx_gpus} ${served_model_name} ${model_path} 'prefill' $ctx_enable_attention_dp &> ${full_logdir}/output_prefill_worker_${i}.log &
  prefill_pids+=($!)
  echo "$!" >> "$PID_FILE"
done


# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --container-env HEAD_NODE_IP,HEAD_NODE,SCRIPTS_DIR \
        --mpi=pmix --overlap -N 1 -n 1 \
	    -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${served_model_name} ${MULTI_ROUND} ${num_gen_servers} "${concurrency_list}" ${STREAMING} ${full_logdir} ${ctx_gpus} ${gen_gpus} ${model_path} ${isl} ${osl} ${kind} ${benchmark_kind} > ${full_logdir}/bench.log 2>&1


# Cleanup will be handled by the EXIT trap
