#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wait_for_model() {

    local model_host=$1
    local model_port=$2
    local n_prefill=${3:-1}
    local n_decode=${4:-1}
    local poll=${5:-1}
    local timeout=${6:-600}
    local report_every=${7:-60}

    local health_addr="http://${model_host}:${model_port}/health"
    echo "Polling ${health_addr} every ${poll} seconds to check whether ${n_prefill} prefills and ${n_decode} decodes are alive"

    local start_ts=$(date +%s)
    local report_ts=$(date +%s)

    while :; do
        # Curl timeout - our primary use case here is to launch it at the first node (localhost), so no timeout is needed.
        curl_result=$(curl ${health_addr} 2>/dev/null)
        # Python path - Use of `check_server_health.py` is self-constrained outside of any packaging.
        check_result=$(python3 /scripts/check_server_health.py $n_prefill $n_decode <<< $curl_result)
        if [[ $check_result == *"Model is ready."* ]]; then
            echo $check_result
            return 0
        fi

        time_now=$(date +%s)
        if [[ $((time_now - start_ts)) -ge $timeout ]]; then
            echo "Model did not get healthy in ${timeout} seconds"
            exit 2;
        fi

        if [[ $((time_now - report_ts)) -ge $report_every ]]; then
            echo $check_result
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

    model_name="deepseek-ai/DeepSeek-R1"
    model_path="deepseek-ai/DeepSeek-R1-0528"
    head_node="localhost"
    head_port="8000"
    chosen_isl=1024
    chosen_osl=1024
    chosen_req_rate="inf"
    chosen_concurrencies=(1 2 4 8 16 32 64 128)

	for concurrency in ${chosen_concurrencies[@]}
	do
	    num_prompts=$((concurrency * 5))

	    command=(
		python3 -m sglang.bench_serving
		--base-url "http://${head_node}:${head_port}"
		--model ${model_name} --tokenizer ${model_path}
		--backend sglang-oai
		--dataset-name random --random-input ${chosen_isl} --random-output ${chosen_osl}
		--random-range-ratio 1
		--num-prompts ${num_prompts} --request-rate ${chosen_req_rate} --max-concurrency ${concurrency}
	    )

	    echo "Running with concurrency: ${concurrency}, num_prompts: ${num_prompts}"
	    "${command[@]}"
	done
}