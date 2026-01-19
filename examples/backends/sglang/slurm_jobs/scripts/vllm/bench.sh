#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example script adapted from https://github.com/kedarpotdar-nv/bench_serving/tree/dynamo-fix.

model_name="deepseek-ai/DeepSeek-R1"
model_path="/model/"
head_node="localhost"
head_port=8000

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$((prefill_gpus+decode_gpus))

source /scripts/benchmark_utils.sh
work_dir="/scripts/vllm/"
cd $work_dir

chosen_isl=$5
chosen_osl=$6
concurrency_list=$7
IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"
chosen_req_rate=$8

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[@]}; ${chosen_req_rate}"

wait_for_model_timeout=3000
wait_for_model_check_interval=5
wait_for_model_report_interval=60

wait_for_model $head_node $head_port $n_prefill $n_decode \
    $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval

set -e

# Warmup defaults
warmup_isl=$chosen_isl
warmup_osl=$chosen_osl
warmup_req_rate=250
warmup_concurrency_list=(1 4 32 128 256)

# Ensure all chosen concurrencies are in warmup list
for c in "${chosen_concurrencies[@]}"; do
    found=false
    for w in "${warmup_concurrency_list[@]}"; do
        if [[ "$c" == "$w" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        warmup_concurrency_list+=("$c")
    fi
done

# Optional: sort warmup list numerically
IFS=$'\n' warmup_concurrency_list=($(sort -n <<<"${warmup_concurrency_list[*]}"))
unset IFS

echo "Final warmup list: ${warmup_concurrency_list[@]}"

# Warmup
for warmup_concurrency in "${warmup_concurrency_list[@]}"
do
    echo "Warming up model with concurrency $warmup_concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    num_prompts=$((warmup_concurrency * 5))
    set -x
    python3 -u benchmark_serving.py \
        --model ${model_name} --tokenizer ${model_path} \
        --host $head_node --port $head_port \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len $warmup_isl \
        --random-output-len $warmup_osl \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --request-rate ${warmup_req_rate} \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$warmup_concurrency"
    set +x
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
done
set +e

result_dir="/logs/vllm_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p $result_dir

set -e
for concurrency in "${chosen_concurrencies[@]}"
do
    num_prompts=$((concurrency * 5))
    echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts, writing to file ${result_dir}"
    result_filename="isl_${chosen_isl}_osl_${chosen_osl}_concurrency_${concurrency}_req_rate_${chosen_req_rate}_ctx_${prefill_gpus}_gen_${decode_gpus}_gpus_${total_gpus}.json"

    set -x
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    python3 -u benchmark_serving.py \
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

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done
