#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

n_prefill=$1
n_decode=$2
total_gpus=$3

chosen_isl=$4
chosen_osl=$5
concurrency_list=$6

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"
chosen_req_rate=$7

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[@]}; ${chosen_req_rate}"

head_node="localhost"
head_port="8000"

SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1"
MODEL_PATH=/model/

source /scripts/benchmark_utils.sh

wait_for_model $head_node $head_port $n_prefill $n_decode 5 900 60

sleep 300

set -e
warmup_model $head_node $head_port $SERVED_MODEL_NAME $MODEL_PATH "${chosen_isl}x${chosen_osl}x10000x10000x250"
set +e

profile_folder="/logs/sglang_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $profile_folder

for max_concurrency in ${chosen_concurrencies[@]}; do

    chosen_n_requests=$((5*max_concurrency))

    export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}.json"

    command=(
        python3 -m sglang.bench_serving
        --base-url "http://${head_node}:${head_port}"
        --model ${SERVED_MODEL_NAME} --tokenizer ${MODEL_PATH}
        --backend sglang-oai
        --dataset-name random --random-input ${chosen_isl} --random-output ${chosen_osl}
        --random-range-ratio 1
        --num-prompts ${chosen_n_requests} --request-rate ${chosen_req_rate} --max-concurrency ${max_concurrency}
        --output-file $export_file
    )

    echo "Running command ${command[@]}"

    ${command[@]}

    echo "-----------------------------------------"
done
