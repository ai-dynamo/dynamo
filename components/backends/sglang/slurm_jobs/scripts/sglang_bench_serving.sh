#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

### Benchmark configuration and setup
# Benchmarking script setup - ISL/OSL/concurrencies/request_rate
chosen_isl=1024
chosen_osl=1024
chosen_req_rate=250
chosen_concurrencies=(2 10 20 50 100 200 500 1000 2000 2500 3000 3500 4000 4500 5000 7500 10000 12500 15000 16250 17500 18750 20000)

# Model config setup - frontend URL, model name, and path
head_node="localhost"
head_port="8000"
SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1"
MODEL_PATH=/model/

# This file contains `wait_for_model` and `warmup_model`
source /scripts/benchmark_utils.sh

### Benchmark runs
# 1. wait for model to come alive - `wait_for_model`
# 2. warms up the model - `warmup_model`
# 3. benchmark model - for concurrency in concurrencies; do <benchmark script>; done
wait_for_model $head_node $head_port 5 2400 60

set -e
warmup_model $head_node $head_port $SERVED_MODEL_NAME $MODEL_PATH "${chosen_isl}x${chosen_osl}x10000x10000x${chosen_req_rate}"
set +e

for max_concurrency in ${chosen_concurrencies[@]}; do

    chosen_n_requests=$((5*max_concurrency))

    command=(
        python3 -m sglang.bench_serving
        --base-url "http://${head_node}:${head_port}"
        --model ${SERVED_MODEL_NAME} --tokenizer ${MODEL_PATH}
        --backend sglang-oai
        --dataset-name random --random-input ${chosen_isl} --random-output ${chosen_osl}
        --random-range-ratio 1
        --num-prompts ${chosen_n_requests} --request-rate ${chosen_req_rate} --max-concurrency ${max_concurrency}
    )

    echo "Running command ${command[@]}"

    ${command[@]}

    echo "-----------------------------------------"
done

