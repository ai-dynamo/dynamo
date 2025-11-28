#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
MULTI_ROUND="${MULTI_ROUND:-8}"

MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=aggr-test

STREAMING=true
GPU_FRAC=0.8

tp_size=$1
ep_size=$2
max_batch_size=$3
max_num_tokens=$4
enable_attention_dp=$5
concurrency_list=$6
mtp_size=$7
kind=$8
isl=$9
osl=${10}
model_path=${11}
served_model_name=${12}
image=${13}

echo "tp_size=$tp_size ep_size=$ep_size max_batch_size=$max_batch_size max_num_tokens=$max_num_tokens enable_attention_dp=$enable_attention_dp concurrency_list=$concurrency_list mtp_size=$mtp_size kind=$kind isl=$isl osl=$osl model_path=$model_path served_model_name=$served_model_name image=$image"

max_seq_len=$((${isl} + ${osl}))

WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/${kind}-bm-${isl}-${osl}
SCRIPTS_DIR=${WORK_DIR}/
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

sub_dir=${LOG_DIR}/ctx0_gen1_dep${tp_size}_batch${max_batch_size}_mtp${mtp_size}
if [ "${enable_attention_dp}" = "false" ]; then
    sub_dir=${LOG_DIR}/ctx0_gen1_tep${tp_size}_batch${max_batch_size}_mtp${mtp_size}
fi

full_logdir=${sub_dir}
artifacts_dir=${full_logdir}/aiperf_artifacts
mkdir -p ${artifacts_dir}


set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
srun ${set_clock_cmd}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${model_path}:${model_path}

# start the container
srun -l --container-image=${image} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up."

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
        bash ${SCRIPTS_DIR}/scripts/start_frontend.sh  &> ${full_logdir}/output_server.log &
SERVER_PID=$!
echo "$SERVER_PID" >> "$PID_FILE"

# wait for the server to start
sleep 10

# start the workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
        --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        bash -x ${WORK_DIR}/scripts/start_agg_worker.sh ${model_path} ${max_batch_size} ${max_num_tokens} ${tp_size} ${ep_size} ${enable_attention_dp} ${GPU_FRAC} ${max_seq_len} ${mtp_size} ${served_model_name} &> ${full_logdir}/output_workers.log &
WORKERS_PID=$!
echo "$WORKERS_PID" >> "$PID_FILE"

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts},${artifacts_dir}:${artifacts_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
	-w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${served_model_name} ${MULTI_ROUND} 1 "${concurrency_list}" ${STREAMING} ${full_logdir} ${tp_size} ${artifacts_dir} ${model_path} ${isl} ${osl} ${kind} > ${full_logdir}/bench.log 2>&1

# Cleanup will be handled by the EXIT trap