#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
#SBATCH --account=gtc_inference
#SBATCH --time=03:00:00
#SBATCH --job-name="gtc_inference-job-1:disaggr-test"

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

# set MOUNT_DIR
MOUNT_DIR="${MOUNT_DIR:-${PWD}}"

WORK_DIR=${MOUNT_DIR}
LOG_DIR=$WORK_DIR/${ISL}-${OSL}-$(date +%Y%m%d_%H%M%S)


CONTAINER_NAME=disaggr-test
SCRIPTS_DIR=${WORK_DIR}/
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

STREAMING=true
CTX_GPU_FRAC=0.75
CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448


ctx_max_seq_len=$((${ISL} + 203))
gen_max_seq_len=$((${ISL} + ${OSL} + 203))

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
mkdir -p ${full_logdir}

artifacts_root_dir=${full_logdir}/genai_perf_artifacts
mkdir -p ${artifacts_root_dir}


# Set clock
srun ${set_clock_cmd}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${MODEL_PATH}:${MODEL_PATH}

# start the container
srun -l --container-image=${IMAGE} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up."

# generate the yaml file
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
	-n 1 -N 1 \
        python3 ${SCRIPTS_DIR}/disagg/gen_yaml.py --config ${full_logdir}/config.yaml \
            --model ${MODEL_PATH} \
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

# start the server
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
	--oversubscribe \
	--overlap \
	--container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/disagg/start_server.sh ${full_logdir}/config.yaml &> ${full_logdir}/output_server.log &

sleep 10

PREFILL_COUNT=$(grep 'prefill_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
echo "Prefill Count: $PREFILL_COUNT"

# start the prefill workers
prefill_pids=()
for ((i=1; i<=PREFILL_COUNT; i++)); do
  echo "Running Prefill Worker: ${i}"
  node_idx=$(( (i-1) % ${#nodes[@]}))
  echo "Running Prefill Nodes: ${nodes[node_idx]}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix --overlap -w ${nodes[node_idx]} \
      --oversubscribe \
      --overlap \
      --ntasks 4 \
      --nodes 1 \
      bash ${SCRIPTS_DIR}/disagg/start_worker.sh ${full_logdir}/prefill_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${SERVED_MODEL_NAME} ${MODEL_PATH} 'prefill' &> ${full_logdir}/output_workers.log &
  prefill_pids+=($!)
done

DECODE_COUNT=$(grep 'decode_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
echo "Decode Count: $DECODE_COUNT"

# Assign decode workers to nodes not used by prefill workers (i.e., skip nodes[0..$PREFILL_COUNT-1])
num_decode_nodes=$((gen_tp_size/4))
num_total_nodes=${#nodes[@]}
decode_start_idx=$PREFILL_COUNT  # nodes[0] is head, nodes[1..PREFILL_COUNT-1] are prefill

for ((i=1; i<=DECODE_COUNT; i++)); do
  echo "Running Decode Worker: ${i}"
  decode_node_list=()
  for ((j=0; j<num_decode_nodes; j++)); do
    node_idx=$(( (decode_start_idx + (i-1)*num_decode_nodes + j) % (num_total_nodes - decode_start_idx) + decode_start_idx ))
    decode_node_list+=("${nodes[node_idx]}")
  done
  decode_nodes_csv=$(IFS=, ; echo "${decode_node_list[*]}")
  echo "Running Decode Nodes: ${decode_nodes_csv}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix \
      -w ${decode_nodes_csv} \
      --nodes ${num_decode_nodes} \
      --ntasks $gen_tp_size \
      --oversubscribe \
      --overlap \
      bash ${SCRIPTS_DIR}/disagg/start_worker.sh ${full_logdir}/decode_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${SERVED_MODEL_NAME} ${MODEL_PATH} 'decode' &> ${full_logdir}/output_workers.log &
done

total_gpus=$((ctx_gpus + gen_gpus))

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts},${artifacts_root_dir}:${artifacts_root_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
	-w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/disagg/bench.sh ${SERVED_MODEL_NAME} ${MULTI_ROUND} ${num_gen_servers} "${concurrency_list}" ${STREAMING} ${full_logdir} ${total_gpus} ${artifacts_root_dir} ${MODEL_PATH} ${ISL} ${OSL} > ${full_logdir}/bench.log 2>&1 


# srun -l --container-name=${CONTAINER_NAME} \
#         --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
#         --mpi=pmix --overlap -N 1 -n 1 \
#         bash ${SCRIPTS_DIR}/disagg/post_process.sh ${full_logdir} ${gen_tp_size} ${mtp_size} ${nsys_on} > ${full_logdir}/post_process.log 2>&1

# try to kill the server and workers
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
        kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
wait
