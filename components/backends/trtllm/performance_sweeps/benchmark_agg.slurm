#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=gtc_inference
#SBATCH --time=04:00:00
#SBATCH --job-name="gtc_inference-job-1:aggr-test"

if [[ -z ${MODEL_PATH} ]]; then
    echo "ERROR: MODEL_PATH was not set."
    echo "ERROR: MODEL_PATH must be set to either the HuggingFace ID or locally " \
         "downloaded path to the model weights. Since Deepseek R1 is large, it is " \
         "recommended to pre-download them to a shared location and provide the path."
    exit 1
fi

if [[ -z ${SERVED_MODEL_NAME} ]]; then
    echo "WARNING: SERVED_MODEL_NAME was not set. It will be derived from MODEL_PATH."
fi


IMAGE="${IMAGE:-""}"

ISL="${ISL:-8150}"
OSL="${OSL:-1024}"
MULTI_ROUND="${MULTI_ROUND:-8}"

MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=aggr-test

STREAMING=true
GPU_FRAC=0.8

max_seq_len=$((${ISL} + ${OSL}))

tp_size=$1
ep_size=$2
max_batch_size=$3
max_num_tokens=$4
enable_attention_dp=$5
concurrency_list=$6
mtp_size=$7
kind=$8


WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/${kind}-bm-${ISL}-${OSL}
SCRIPTS_DIR=${WORK_DIR}/
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

sub_dir=${LOG_DIR}/dep${tp_size}_batch${max_batch_size}_mtp${mtp_size}
if [ "${enable_attention_dp}" = "false" ]; then
    sub_dir=${LOG_DIR}/tep${tp_size}_batch${max_batch_size}_mtp${mtp_size}
fi

full_logdir=${sub_dir}
artifacts_dir=${full_logdir}/genai_perf_artifacts
mkdir -p ${artifacts_dir}


set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
srun ${set_clock_cmd}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${MODEL_PATH}:${MODEL_PATH}

# start the container
srun -l --container-image=${IMAGE} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up."

export HEAD_NODE="${nodes[0]}"
export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

# start the server
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
	--oversubscribe \
	--overlap \
	--container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/start_frontend.sh  &> ${full_logdir}/output_server.log &

# wait for the server to start
sleep 10

# start the workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
        --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        bash -x ${WORK_DIR}/scripts/start_agg_worker.sh ${MODEL_PATH} ${max_batch_size} ${max_num_tokens} ${tp_size} ${ep_size} ${enable_attention_dp} ${GPU_FRAC} ${max_seq_len} ${mtp_size} ${SERVED_MODEL_NAME} &> ${full_logdir}/output_server.log &

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts},${artifacts_dir}:${artifacts_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
	-w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${SERVED_MODEL_NAME} ${MULTI_ROUND} 1 "${concurrency_list}" ${STREAMING} ${full_logdir} ${tp_size} ${artifacts_dir} ${MODEL_PATH} ${ISL} ${OSL} ${kind} > ${full_logdir}/bench.log 2>&1 

# try to kill the server and workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
        kill -9 $(ps aux | grep '[p]ython3' | awk '{print $2}') >/dev/null 2>&1 || true
wait
