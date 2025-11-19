#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example script adapted from https://github.com/kedarpotdar-nv/bench_serving/tree/dynamo-fix.

model_name="deepseek-ai/DeepSeek-R1"
model_path="/model/"
head_node="localhost"
head_port=8000

n_prefill=$1
n_decode=$2
total_gpus=$3

source /scripts/benchmark_utils.sh
work_dir="/scripts/vllm/"
cd $work_dir

chosen_isl=$4
chosen_osl=$5
concurrency_list=$6
IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"
chosen_req_rate=$7

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[@]}; ${chosen_req_rate}"

wait_for_model_timeout=1500 # 25 minutes
wait_for_model_check_interval=5 # check interval -> 5s
wait_for_model_report_interval=60 # wait_for_model report interval -> 60s

wait_for_model $head_node $head_port $n_prefill $n_decode $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval

set -e
# Warmup the model
warmup_isl=$chosen_isl
warmup_osl=$chosen_osl
warmup_prompts=10000
warmup_concurrencies=10000
warmup_req_rate=250
set -x
python3 benchmark_serving.py \
    --model ${model_name} --tokenizer ${model_path} \
    --host $head_node --port $head_port \
    --backend "dynamo" --endpoint /v1/completions \
    --disable-tqdm \
    --dataset-name random \
    --num-prompts "$warmup_prompts" \
    --random-input-len $warmup_isl \
    --random-output-len $warmup_osl \
    --random-range-ratio 0.8 \
    --ignore-eos \
    --request-rate ${warmup_req_rate} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --max-concurrency "$warmup_concurrencies"
set +x
set +e

result_dir="/logs/vllm_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $result_dir

set -e
for concurrency in "${chosen_concurrencies[@]}"
do
    num_prompts=$((concurrency * 5))
    echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts, writing to file ${result_dir}"
    result_filename="isl_${chosen_isl}_osl_${chosen_osl}_concurrency_${concurrency}_req_rate_${chosen_req_rate}_gpus${total_gpus}.json"

    set -x
    python3 benchmark_serving.py \
        --model ${model_name} --tokenizer ${model_path} \
        --host $head_node --port $head_port \
        --backend "dynamo" --endpoint /v1/completions \
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
