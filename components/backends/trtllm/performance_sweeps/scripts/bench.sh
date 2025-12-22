#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Check if SCRIPTS_DIR is set, if not try to infer it or exit
if [ -z "${SCRIPTS_DIR:-}" ]; then
    # Try to infer SCRIPTS_DIR from the current script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
    echo "SCRIPTS_DIR not set, inferred as: ${SCRIPTS_DIR}"
fi

# Verify SCRIPTS_DIR exists and contains expected structure
if [ ! -d "${SCRIPTS_DIR}/scripts/bench" ]; then
    echo "Error: SCRIPTS_DIR (${SCRIPTS_DIR}) does not contain expected structure"
    echo "Expected: ${SCRIPTS_DIR}/scripts/bench to exist"
    exit 1
fi

model=${1}
multi_round=${2}
num_ctx_servers=${3}
num_gen_servers=${4}
concurrency_list=${5}
streaming=${6}
log_path=${7}
prefill_gpus=${8}
decode_gpus=${9}
total_gpus=$((prefill_gpus+decode_gpus))
model_path=${10}
isl=${11}
osl=${12}
kind=${13}
benchmark_kind=${14}

if [ "$#" -ne 14 ]; then
    echo "Error: Expected 14 arguments, got $#"
    echo "Usage: $0 <model> <multi_round> <num_ctx_servers> <num_gen_servers> <concurrency_list> <streaming> <log_path> <prefill_gpus> <decode_gpus> <model_path> <isl> <osl> <kind> <benchmark_kind>"
    exit 1
fi

echo "Arguments:"
echo "  model: $model"
echo "  multi_round: $multi_round"
echo "  num_ctx_servers: $num_ctx_servers"
echo "  num_gen_servers: $num_gen_servers"
echo "  concurrency_list: $concurrency_list"
echo "  streaming: $streaming"
echo "  log_path: $log_path"
echo "  prefill_gpus: $prefill_gpus"
echo "  decode_gpus: $decode_gpus"
echo "  total_gpus: $total_gpus"
echo "  model_path: $model_path"
echo "  isl: $isl"
echo "  osl: $osl"
echo "  kind: $kind"
echo "  benchmark_kind: $benchmark_kind"

if ! ( [[ "$benchmark_kind" == "sa" || "$benchmark_kind" == "aiperf" ]] ); then
    echo "Invalid benchmark kind! Expected 'sa' or 'aiperf'"
    exit 0
fi

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

set -x
config_file=${log_path}/config.yaml


hostname=$HEAD_NODE
port=8000

echo "Hostname: ${hostname}, Port: ${port}"

apt update
apt install curl


# try client

# The configuration is dumped to a JSON file which hold details of the OAI service
# being benchmarked.
deployment_config=$(cat << EOF
{
  "kind": "${kind}",
  "model": "${model}",
  "total_gpus": "${total_gpus}"
}
EOF
)

mkdir -p "${log_path}"
if [ -f "${log_path}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${log_path}/deployment_config.json"
fi
echo "${deployment_config}" > "${log_path}/deployment_config.json"

health_addr="http://$hostname:${port}/health"
echo "Polling ${health_addr} every 5 seconds to check whether ${num_ctx_servers} prefills and ${num_gen_servers} decodes are alive"

start_ts=$(date +%s)
report_ts=$(date +%s)

while :; do
    # Curl timeout - our primary use case here is to launch it at the first node (localhost), so no timeout is needed.
    curl_result=$(curl ${health_addr} 2>/dev/null)
    # Python path - Use of `check_server_health.py` is self-constrained outside of any packaging.
    check_result=$(python3 $SCRIPTS_DIR/scripts/check_server_health.py $num_ctx_servers $num_gen_servers <<< $curl_result)
    if [[ $check_result == *"Model is ready."* ]]; then
        echo $check_result
        break;
    fi

    time_now=$(date +%s)
    if [[ $((time_now - start_ts)) -ge 1200 ]]; then
        echo "Model did not get healthy in 1200 seconds"
        exit 2;
    fi

    if [[ $((time_now - report_ts)) -ge 20 ]]; then
        echo $check_result
        report_ts=$time_now
    fi

    sleep 5
done

curl -v  -w "%{http_code}" "${hostname}:${port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "'${model}'",
  "messages": [
  {
    "role": "user",
    "content": "Tell me a story as if we were playing dungeons and dragons."
  }
  ],
  "stream": true,
  "max_tokens": 30
}'

mkdir -p ${log_path}/results
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    original_concurrency=$concurrency
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}

    if [[ "$benchmark_kind" == "sa" ]]; then
        python3 ${SCRIPTS_DIR}/scripts/bench/benchmark_serving.py \
            --served-model-name ${model} \
            --model ${model_path} \
            --dataset-name random \
            --num-prompts "$(($concurrency * 2))" \
            --random-input-len ${isl} \
            --random-output-len ${osl} \
            --random-range-ratio 0.8 \
            --ignore-eos \
	    --use-chat-template \
            --backend "dynamo" \
	    --endpoint "/v1/completions" \
	    --percentile-metrics ttft,tpot,itl,e2el \
            --max-concurrency "$concurrency" \
	    --host ${hostname} \
            --port ${port}
	python3 ${SCRIPTS_DIR}/scripts/bench/benchmark_serving.py \
            --served-model-name ${model} \
            --model ${model_path} \
            --dataset-name random \
            --num-prompts "$num_prompts" \
            --random-input-len ${isl} \
            --random-output-len ${osl} \
            --random-range-ratio 0.8 \
            --use-chat-template \
            --ignore-eos \
            --use-chat-template \
            --backend "dynamo" \
            --endpoint "/v1/completions" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --max-concurrency "$concurrency" \
            --host ${hostname} \
            --port ${port} \
            --save-result \
            --result-dir "${log_path}/results" \
            --result-filename "results_concurrency_${original_concurrency}_gpus_${total_gpus}_ctx_${prefill_gpus}_gen_${decode_gpus}.json"
    else
        aiperf profile \
    	    --model ${model} \
    	    --tokenizer ${model_path} \
    	    --endpoint-type completions \
    	    --endpoint /v1/completions \
    	    --streaming \
    	    --url ${hostname}:${port} \
    	    --synthetic-input-tokens-mean ${isl} \
    	    --synthetic-input-tokens-stddev 0 \
    	    --output-tokens-mean ${osl} \
    	    --output-tokens-stddev 0 \
    	    --extra-inputs max_tokens:${osl} \
    	    --extra-inputs min_tokens:${osl} \
    	    --extra-inputs ignore_eos:true \
	    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    	    --concurrency $concurrency \
    	    --request-count $num_prompts \
    	    --warmup-request-count $(($concurrency*2)) \
	    --num-dataset-entries ${num_prompts} \
    	    --random-seed 100 \
    	    --artifact-dir "${log_path}/results/concurrency_${original_concurrency}" \
    	    --ui simple \
	    -v \
    	    -H 'Authorization: Bearer NOT USED' \
    	    -H 'Accept: text/event-stream'    
    fi

    echo "Benchmark with concurrency ${concurrency} done"
done


job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi
