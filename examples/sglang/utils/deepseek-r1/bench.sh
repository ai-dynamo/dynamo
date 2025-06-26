#!/bin/bash

usage() {
    echo "Usage: $0 <ip> [port] [--type e2e|custom_completions]"
    echo "  ip: server IP address"
    echo "  port: server port (defaults to 8000)"
    echo "  --type: endpoint type - 'e2e' for chat completions, 'custom_completions' for completions (defaults to e2e)"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

IP=$1
PORT=${2:-8000}
TYPE="e2e"

# Parse remaining arguments
shift 2 2>/dev/null || shift 1
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TYPE="$2"
            shift 2
            ;s
        *)
            usage
            ;;
    esac
done

if [[ "$TYPE" != "e2e" && "$TYPE" != "custom_completions" ]]; then
    echo "Error: --type must be 'e2e' or 'custom_completions'"
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
            --warmup-request-count $(($concurrency*2)) \
            --num-dataset-entries $(($concurrency*12)) \
            --random-seed 100 \
            --artifact-dir ${ARTIFACT_DIR} \
            -- \
            -v \
            --max-threads ${concurrency} \
            -H 'Authorization: Bearer NOT USED' \
            -H 'Accept: text/event-stream'
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
            -v -v \
            --max-threads 256 \
            -H 'Authorization: Bearer NOT USED' \
            -H 'Accept: text/event-stream'
    done
fi