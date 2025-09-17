#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

: "${ACCOUNT:?ACCOUNT not defined}"
: "${PARTITION:?PARTITION not defined}"
: "${TIME_LIMIT:?TIME_LIMIT not defined}"

: "${MODEL_DIR:?MODEL_DIR not defined}"
: "${CONFIG_DIR:?CONFIG_DIR not defined}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE not defined}"

GPU_TYPE="gb200-fp8"
GPUS_PER_NODE=4
: "${NETWORK_INTERFACE:=enP6p9s0np0}"

# COMMAND_LINE ARGS
PREFILL_NODES=$1
PREFILL_WORKERS=$2
DECODE_NODES=$3
DECODE_WORKERS=$4
N_ADDITIONAL_FRONTENDS=$5
ISL=$6
OSL=$7
CONCURRENCIES=$8
REQUEST_RATE=$9

# Should not need retries

profiler_args="type=vllm; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req-rate=${REQUEST_RATE}"

USE_INIT_LOCATIONS=()
if [[ $PREFILL_NODES -eq 6 ]] && [[ $PREFILL_WORKERS -eq 3 ]] && [[ $DECODE_NODES -eq 12 ]] && [[ $DECODE_WORKERS -eq 1 ]]; then 
    USE_INIT_LOCATIONS=(--use-init-location)
fi

command=(
    python3 submit_job_script.py 
    --account $ACCOUNT --partition $PARTITION --time-limit $TIME_LIMIT 
    --template job_script_template.j2 
    --model-dir $MODEL_DIR --config-dir $CONFIG_DIR 
    --container-image $CONTAINER_IMAGE 

    --gpu-type $GPU_TYPE --gpus-per-node $GPUS_PER_NODE --network-interface $NETWORK_INTERFACE 

    --prefill-nodes $PREFILL_NODES --prefill-workers $PREFILL_WORKERS
    --decode-nodes $DECODE_NODES --decode-workers $DECODE_WORKERS
    --enable-multiple-frontends --num-additional-frontends $N_ADDITIONAL_FRONTENDS ${USE_INIT_LOCATIONS[@]}

    --profiler "${profiler_args}"
)

"${command[@]}"
