#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

usage() {
    cat << 'USAGE'
This script aims to provide a one-liner call to the submit_job_script.py,
so that the deployment process can be further simplified.

To use this script, fill in the following script and run it under your `slurm_jobs` directory:
======== begin script area ========
export SLURM_ACCOUNT=
export SLURM_PARTITION=
export TIME_LIMIT=

# Add path to your DSR1-FP8 model directory here
export MODEL_PATH=

# This path should contain the deepep.json and optionally init expert locations.
# Please refer to the README for more detail.
export CONFIG_DIR=

# Add path to your container image here, either as a link or as a cached file
export CONTAINER_IMAGE=

bash submit_disagg.sh \
$PREFILL_NODES $PREFILL_WORKERS $DECODE_NODES $DECODE_WORKERS \
$ADDITIONAL_FRONTENDS \
$ISL $OSL $CONCURRENCIES $REQUEST_RATE
======== end script area ========
USAGE
}

check_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} not specified" >&2
        usage >&2
        exit 1
    fi
}

check_env SLURM_ACCOUNT
check_env SLURM_PARTITION
check_env TIME_LIMIT

check_env MODEL_PATH
check_env CONFIG_DIR
check_env CONTAINER_IMAGE

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

RETRIES=1 # defaults to retry the job 1 time to avoid transient errors

# Should not need retries

profiler_args="type=vllm; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req-rate=${REQUEST_RATE}"

USE_INIT_LOCATIONS=()
if [[ $PREFILL_NODES -eq 6 ]] && [[ $PREFILL_WORKERS -eq 3 ]] && [[ $DECODE_NODES -eq 12 ]] && [[ $DECODE_WORKERS -eq 1 ]]; then
    USE_INIT_LOCATIONS=(--use-init-location)
fi

command=(
    python3 submit_job_script.py
    --account $SLURM_ACCOUNT --partition $SLURM_PARTITION --time-limit $TIME_LIMIT
    --template job_script_template.j2
    --model-dir $MODEL_PATH --config-dir $CONFIG_DIR
    --container-image $CONTAINER_IMAGE

    --gpu-type $GPU_TYPE --gpus-per-node $GPUS_PER_NODE --network-interface $NETWORK_INTERFACE

    --prefill-nodes $PREFILL_NODES --prefill-workers $PREFILL_WORKERS
    --decode-nodes $DECODE_NODES --decode-workers $DECODE_WORKERS
    --enable-multiple-frontends --num-additional-frontends $N_ADDITIONAL_FRONTENDS ${USE_INIT_LOCATIONS[@]}

    --profiler "${profiler_args}"

    --retries $RETRIES
)

"${command[@]}"
