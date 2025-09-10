#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example script adapted from https://github.com/kedarpotdar-nv/bench_serving/tree/dynamo-fix.

model_name="deepseek-ai/DeepSeek-R1"
model_path="/model/"
head_node="localhost"
head_port=8000

source /scripts/benchmark_utils.sh
work_dir="/scripts/vllm/"
cd $work_dir

chosen_isl=$3
chosen_osl=$4
concurrency_list=$5
IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"
chosen_req_rate=$6

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[@]}; ${chosen_req_rate}"

wait_for_model $head_node $head_port 5 2400 60

set -e
warmup_model $head_node $head_port $model_name $model_path "${chosen_isl}x${chosen_osl}x10000x10000x${chosen_req_rate}"
set +e

result_dir="/logs/vllm_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $result_dir

set -e
for concurrency in "${chosen_concurrencies[@]}"
do
    num_prompts=$((concurrency * 5))
    echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts, writing to file ${result_dir}"
    result_filename="isl_${chosen_isl}_osl_${chosen_osl}_concurrency_${concurrency}_req_rate_${chosen_req_rate}.json"

    set -x
    python3 benchmark_serving.py \
        --model ${model_name} --tokenizer ${model_path} \
        --host $head_node --port $head_port \
        --backend "dynamo" --endpoint /v1/chat/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len $chosen_isl \
        --random-output-len $chosen_osl \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --request-rate ${chosen_req_rate} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --save-result --result-dir $result_dir --result-filename $result_filename
    set +x

    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done
set +e
