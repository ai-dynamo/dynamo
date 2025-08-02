#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

usage() {
    echo "Usage: $0 <ip> [port] [--type e2e|custom_completions|warmup]"
    echo "  ip: server IP address"
    echo "  port: server port (defaults to 8000)"
    echo "  --type: endpoint type - 'e2e' for chat completions, 'custom_completions' for completions, 'warmup' for warmup phases"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

IP=$1
PORT=8000
TYPE="e2e"

# Check if second argument is a port number or an option
if [[ $# -gt 1 && $2 =~ ^[0-9]+$ ]]; then
    PORT=$2
    shift 2
else
    shift 1
fi

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TYPE="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ "$TYPE" != "e2e" && "$TYPE" != "custom_completions" && "$TYPE" != "warmup" ]]; then
    echo "Error: --type must be 'e2e', 'custom_completions', or 'warmup'"
    usage
fi

MODEL="deepseek-ai/DeepSeek-R1"
ARTIFACT_DIR="/benchmarks/"

if [[ "$TYPE" == "e2e" ]]; then
    # E2E chat completions configuration
    ISL=8000
    OSL=256
    CONCURRENCY_ARRAY=(1 2 4 16 64 256 512 1024 2048 4096 8192)

    for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
        echo "Run e2e concurrency: $concurrency"

        genai-perf profile \
            --model ${MODEL} \
            --tokenizer ${MODEL} \
            --endpoint-type chat \
            --endpoint /v1/chat/completions \
            --streaming \
            --url ${IP}:${PORT} \
            --synthetic-input-tokens-mean ${ISL} \
            --synthetic-input-tokens-stddev 0 \
            --output-tokens-mean ${OSL} \
            --output-tokens-stddev 0 \
            --extra-inputs max_tokens:${OSL} \
            --extra-inputs min_tokens:${OSL} \
            --extra-inputs ignore_eos:true \
            --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
            --concurrency ${concurrency} \
            --request-count $(($concurrency*10)) \
            --num-dataset-entries $(($concurrency*12)) \
            --random-seed 100 \
            --artifact-dir ${ARTIFACT_DIR} \
            -- \
            -v \
            --max-threads ${concurrency} \
            -H 'Authorization: Bearer NOT USED' \
            -H 'Accept: text/event-stream'
    done

elif [[ "$TYPE" == "warmup" ]]; then
    echo "Starting warmup phases..."

    # Phase configurations: "ISL OSL CONCURRENCY_LIST"
    PHASES=(
        "500 100 1,2,4,8"
        "2000 100 1,2,4,8"
        "4000 256 1,2,8,64"
    )

    for i in "${!PHASES[@]}"; do
        phase_num=$((i + 1))
        phase_config=(${PHASES[$i]})
        ISL=${phase_config[0]}
        OSL=${phase_config[1]}
        concurrency_list=${phase_config[2]}

        echo "Phase $phase_num: ISL=$ISL, OSL=$OSL"

        # Convert comma-separated list to array
        IFS=',' read -ra CONCURRENCY_ARRAY <<< "$concurrency_list"

        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            echo "Run warmup phase $phase_num, concurrency: $concurrency, ISL: $ISL, OSL: $OSL"

            genai-perf profile \
                --model ${MODEL} \
                --tokenizer ${MODEL} \
                --endpoint-type chat \
                --endpoint /v1/chat/completions \
                --streaming \
                --url ${IP}:${PORT} \
                --synthetic-input-tokens-mean ${ISL} \
                --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean ${OSL} \
                --output-tokens-stddev 0 \
                --extra-inputs max_tokens:${OSL} \
                --extra-inputs min_tokens:${OSL} \
                --extra-inputs ignore_eos:true \
                --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
                --concurrency ${concurrency} \
                --request-count $(($concurrency)) \
                --warmup-request-count $(($concurrency)) \
                --num-dataset-entries $(($concurrency*12)) \
                --random-seed 100 \
                --artifact-dir ${ARTIFACT_DIR} \
                -- \
                -v \
                --max-threads ${concurrency} \
                -H 'Authorization: Bearer NOT USED' \
                -H 'Accept: text/event-stream'

            echo "Sleeping for 5 seconds..."
            sleep 5
        done

        echo "Phase $phase_num complete"
    done

else
    # Custom completions configuration
    OSL=5
    INPUT_FILE=data.jsonl
    CONCURRENCY_ARRAY=(8192)

    for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
        echo "Run custom_completions concurrency: $concurrency"

        genai-perf profile \
            --model ${MODEL} \
            --tokenizer ${MODEL} \
            --endpoint-type completions \
            --streaming \
            --url ${IP}:${PORT} \
            --input-file ${INPUT_FILE} \
            --extra-inputs max_tokens:${OSL} \
            --extra-inputs min_tokens:${OSL} \
            --extra-inputs ignore_eos:true \
            --concurrency ${concurrency} \
            --request-count ${concurrency} \
            --random-seed 100 \
            --artifact-dir ${ARTIFACT_DIR} \
            --warmup-requests 10 \
            -- \
            -v \
            --max-threads 256 \
            -H 'Authorization: Bearer NOT USED' \
            -H 'Accept: text/event-stream'
    done
fi