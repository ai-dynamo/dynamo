#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

prefill_workers=$1
decode_workers=$2
total_gpus=$3

chosen_isl=$4
chosen_osl=$5
chosen_concurrencies=$6

echo "Profiling for model with PrefillDP=${prefill_workers}, DecodeDP=${decode_workers}"

head_node="localhost"
head_port="8000"

SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1"
MODEL_PATH=/model/

random_seed=$(python3 -c "import random; print(random.randint(0, 65535))")
random_seed=$RANDOM
echo "Chosen random seed ${random_seed}"

source /scripts/benchmark_utils.sh

wait_for_model $head_node $head_port $prefill_workers $decode_workers 5 900 60

set -e
warmup_model $head_node $head_port $SERVED_MODEL_NAME $MODEL_PATH "${chosen_isl}x${chosen_osl}x10000x10000x250"
set +e

aiperf_warmup_workers=$(python3 -c "print(max(${DP:-0}, ${prefill_workers:-0}, ${decode_workers:-0}))")

IFS='x' read -r -a concurrency_list <<< "$chosen_concurrencies"

profile_folder="/logs/gap_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $profile_folder

tmp_work_dir=$(mktemp -d -t aiperf-XXXXXXXX)
for concurrency in ${concurrency_list[@]}; do
    export_folder="${tmp_work_dir}/concurrency_${concurrency}"
    mkdir -p $export_folder
    export_model_name=${SERVED_MODEL_NAME//\//_}
    export_file="${export_model_name}_generation_${concurrency}.json"

    echo "Run benchmark for concurrency $concurrency; ISL $chosen_isl; OSL $chosen_osl"
    command=(
        aiperf profile
        -m ${SERVED_MODEL_NAME}
        --tokenizer ${MODEL_PATH}
        --endpoint-type chat
        --endpoint /v1/chat/completions
        --url "${head_node}:${head_port}"
        --streaming

        --concurrency ${concurrency}
        --warmup-request-count $(( 2*aiperf_warmup_workers ))
        --request-count $(( 5*concurrency ))

        --synthetic-input-tokens-mean ${chosen_isl} --synthetic-input-tokens-stddev 0
        --output-tokens-mean ${chosen_osl} --output-tokens-stddev 0
        --extra-inputs "max_tokens:${chosen_osl}" --extra-inputs "min_tokens:${chosen_osl}"

        --artifact-dir ${export_folder}
        --profile-export-file ${export_file}

        --random-seed ${random_seed}

        --tokenizer-trust-remote-code
        --num-dataset-entries 3000
    )

    set -e
    ${command[@]}
    set +e

    cp $export_folder/*/*_aiperf.json $profile_folder
done
