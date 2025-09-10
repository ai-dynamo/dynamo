#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wait_for_model() {

    local model_host=$1
    local model_port=$2
    local poll=${3:-1}
    local timeout=${4:-600}
    local report_every=${5:-60}

    local health_addr="http://${model_host}:${model_port}/health"
    echo "Polling ${health_addr} every ${poll} seconds"

    local start_ts=$(date +%s)
    local report_ts=$(date +%s)

    while :; do
        curl_result=$(curl ${health_addr} 2>/dev/null)
        health=$(grep '"status":"healthy"' <<< $curl_result)
        if [[ -n $health ]]; then
            echo "Model is alive. Health response: ${curl_result}; "
            return 0;
        fi

        time_now=$(date +%s)
        if [[ $((time_now - start_ts)) -ge $timeout ]]; then
            echo "Model did not get healthy in ${timeout} seconds"
            exit 2;
        fi

        if [[ $((time_now - report_ts)) -ge $report_every ]]; then
            echo "Waiting for model to come alive. Current result: ${curl_result}"
            report_ts=$time_now
        fi

        sleep $poll
    done
}

warmup_model() {
    service_host=$1
    service_port=$2
    served_model_name=$3
    model_path=$4
    config=$5

    IFS='x' read -r -a config_list <<< "$config"
    isl=${config_list[0]}
    osl=${config_list[1]}
    num_prompts=${config_list[2]}
    concurrency=${config_list[3]}
    request_rate=${config_list[4]}

    command=(
        python3 -m sglang.bench_serving
        --base-url "http://${service_host}:${service_port}"
        --model ${served_model_name} --tokenizer ${model_path}
        --backend sglang-oai
        --dataset-name random --random-input ${isl} --random-output ${osl}
        --random-range-ratio 1
        --num-prompts ${num_prompts} --request-rate ${request_rate} --max-concurrency ${concurrency}
    )

    echo "Config ${config}. Running command ${command[@]}"

    ${command[@]}
}
