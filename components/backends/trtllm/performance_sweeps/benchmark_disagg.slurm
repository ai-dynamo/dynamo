#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
MULTI_ROUND="${MULTI_ROUND:-8}"

# set MOUNT_DIR
MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=disaggr-test-$(date +%s)-$$

config=$1
concurrency_list=$2

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"model"}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-4}
BENCHMARK_KIND=${BENCHMARK_KIND:-"sa"}

if [[ "${BENCHMARK_KIND}" != "sa" && "${BENCHMARK_KIND}" != "aiperf" ]]; then
    echo "ERROR: BENCHMARK_KIND must be either 'sa' or 'aiperf'. Got '${BENCHMARK_KIND}'."
    exit 1
fi


if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH environment variable must be set."
    exit 1
fi

if [[ -z "${ISL}" ]]; then
    echo "ERROR: ISL environment variable must be set."
    exit 1
fi

if [[ -z "${OSL}" ]]; then
    echo "ERROR: OSL environment variable must be set."
    exit 1
fi


WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/dynamo-disagg-bm-${ISL}-${OSL}
SCRIPTS_DIR=${WORK_DIR}/
mkdir -p ${LOG_DIR}
echo "trying to submit job"

config=$(realpath "$config")

sub_dir=$LOG_DIR/$(python3 -c "import yaml, uuid; print(yaml.safe_load(open('${config}')).get('name', 'benchmark-' + uuid.uuid4().hex))")

echo "concurrency_list: ${concurrency_list}"

full_logdir=${sub_dir}
mkdir -p ${full_logdir}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${MODEL_PATH}:${MODEL_PATH},${config}:${config}

# start the container
srun -l --container-image=${IMAGE} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up."

srun --container-name ${CONTAINER_NAME} \
    --container-mounts ${container_mounts} \
    -N 1 -n 1 \
    python3 ${SCRIPTS_DIR}/scripts/gen_instance_config.py ${config} ${full_logdir}

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

if [[ -n $(scontrol show config | grep "GresTypes" | grep "gpu") ]]; then
    gpus_per_node="--gpus-per-node ${NTASKS_PER_NODE}"
else
    gpus_per_node=""
fi

DECODE_COUNT=$(grep 'decode_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$DECODE_COUNT" ]; then
    echo "Error: Failed to extract decode_count from instance_config.yaml"
    exit 1
fi
echo "Decode Count: $DECODE_COUNT"

decode_tp_size=$(grep 'decode_tp_size:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$decode_tp_size" ]; then
    echo "Error: Failed to extract decode_tp_size from instance_config.yaml"
    exit 1
fi
echo "Decode TP Size: $decode_tp_size"

num_decode_nodes=$(((decode_tp_size+NTASKS_PER_NODE-1)/NTASKS_PER_NODE))
decode_nodes=$((num_decode_nodes * DECODE_COUNT))
for ((i=1; i<=DECODE_COUNT; i++)); do
  echo "Running Decode Worker: ${i}"
  decode_node_list=()
  for ((j=0; j<num_decode_nodes; j++)); do
    node_idx=$(((i-1)*num_decode_nodes + j))
    decode_node_list+=("${nodes[node_idx]}")
  done
  decode_nodes_csv=$(IFS=, ; echo "${decode_node_list[*]}")
  echo "Running Decode Nodes: ${decode_nodes_csv}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix \
      -w ${decode_nodes_csv} \
      --nodes ${num_decode_nodes} $gpus_per_node \
      --ntasks $decode_tp_size \
      --oversubscribe \
      --overlap \
      -e UCX_NET_DEVICES,TRTLLM_UCX_INTERFACE \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${config} ${MODEL_PATH} ${SERVED_MODEL_NAME} "decode" &> ${full_logdir}/output_decode_worker_${i}.log &
  echo "$!" >> "$PID_FILE"
done

PREFILL_COUNT=$(grep 'prefill_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$PREFILL_COUNT" ]; then
    echo "Error: Failed to extract prefill_count from instance_config.yaml"
    exit 1
fi
echo "Prefill Count: $PREFILL_COUNT"

prefill_tp_size=$(grep 'prefill_tp_size:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$prefill_tp_size" ]; then
    echo "Error: Failed to extract prefill_tp_size from instance_config.yaml"
    exit 1
fi
echo "Prefill TP Size: $prefill_tp_size"

# start the prefill workers
prefill_pids=()
prefill_start_idx=$decode_nodes
for ((i=1; i<=PREFILL_COUNT; i++)); do
  echo "Running Prefill Worker: ${i}"
  node_idx=$((prefill_start_idx + (i-1)))
  echo "Running Prefill Nodes: ${nodes[node_idx]}"
  srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -w ${nodes[node_idx]} \
        --oversubscribe \
        --overlap \
        --ntasks $(( prefill_tp_size < NTASKS_PER_NODE ? prefill_tp_size : NTASKS_PER_NODE )) \
        --nodes 1 $gpus_per_node \
        -e UCX_NET_DEVICES,TRTLLM_UCX_INTERFACE \
        bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${config} ${MODEL_PATH} ${SERVED_MODEL_NAME} 'prefill' &> ${full_logdir}/output_prefill_worker_${i}.log &
  prefill_pids+=($!)
  echo "$!" >> "$PID_FILE"
done

ctx_gpus=$((prefill_tp_size*PREFILL_COUNT))
decode_gpus=$((decode_tp_size*DECODE_COUNT))

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --container-env HEAD_NODE_IP,HEAD_NODE,SCRIPTS_DIR \
        --mpi=pmix --overlap -N 1 -n 1 \
	    -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${SERVED_MODEL_NAME} ${MULTI_ROUND} ${DECODE_COUNT} "${concurrency_list}" true ${full_logdir} ${ctx_gpus} ${decode_gpus} ${MODEL_PATH} ${ISL} ${OSL} "dynamo-disagg" ${BENCHMARK_KIND} > ${full_logdir}/bench.log 2>&1


# Cleanup will be handled by the EXIT trap
