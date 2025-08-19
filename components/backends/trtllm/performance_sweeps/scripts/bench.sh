#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

model=$1
multi_round=$2
num_gen_servers=$3
concurrency_list=$4
streaming=$5
log_path=$6
total_gpus=$7
artifacts_dir=$8
model_path=$9
isl=${10}
osl=${11}
kind=${12}

if [ "$#" -ne 12 ]; then
    echo "Error: Expected 12 arguments, got $#"
    echo "Usage: $0 <model> <multi_round> <num_gen_servers> <concurrency_list> <streaming> <log_path> <total_gpus> <artifacts_dir> <model_path> <isl> <osl> <kind>"
    exit 1
fi

echo "Arguments:"
echo "  model: $model"
echo "  multi_round: $multi_round"
echo "  num_gen_servers: $num_gen_servers"
echo "  concurrency_list: $concurrency_list"
echo "  streaming: $streaming"
echo "  log_path: $log_path"
echo "  total_gpus: $total_gpus"
echo "  artifacts_dir: $artifacts_dir"
echo "  model_path: $model_path"
echo "  isl: $isl"
echo "  osl: $osl"
echo "  kind: $kind"



# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

set -x
config_file=${log_path}/config.yaml


# install genai-perf
pip install genai-perf

# Create artifacts root directory if it doesn't exist
if [ ! -d "${artifacts_dir}" ]; then
    mkdir -p "${artifacts_dir}"
fi

hostname=$HEAD_NODE_IP
port=8000

echo "Hostname: ${hostname}, Port: ${port}"

apt update
apt install curl


# try client

do_get_logs(){
    worker_log_path=$1
    output_folder=$2
    grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" ${worker_log_path} > ${output_folder}/gen_only.txt || true
    grep -a "'num_generation_tokens': 0" ${worker_log_path} > ${output_folder}/ctx_only.txt || true
}

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

mkdir -p "${artifacts_dir}"
if [ -f "${artifacts_dir}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${artifacts_dir}/deployment_config.json"
fi
echo "${deployment_config}" > "${artifacts_dir}/deployment_config.json"

# TODO: This is a temporary fix to check if the server is up.
# We should use a more robust health check mechanism.

# Loop up to 50 times
for ((i=1; i<=50; i++)); do
    # Run curl and capture response and HTTP code
    response=$(curl -s -w "\n%{http_code}" "${hostname}:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${model}\",
        \"messages\": [
           {
            \"role\": \"user\",
            \"content\": \"Tell me a story as if we were playing dungeons and dragons.\"
           }
        ],
        \"stream\": true,
        \"max_tokens\": 30
      }")

    # Extract HTTP code
    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" = "200" ]; then
        echo "Success on attempt $i"
        # Optional: Print the response body (excluding HTTP code)
        echo "$response" | sed '$d'
        break
    else
        echo "Attempt $i failed (HTTP $http_code)."

        # Wait: 100 seconds after first failure, 10 seconds after subsequent
        if [ "$i" -eq 1 ]; then
            sleep 300
        else
            sleep 10
        fi
    fi
done

if [ "$http_code" != "200" ]; then
    echo "Server did not respond correctly after 50 attempts."
    exit 1
fi

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

cp ${log_path}/output_workers.log ${log_path}/workers_start.log
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}
    genai-perf profile \
    	--model ${model} \
    	--tokenizer ${model_path} \
    	--endpoint-type chat \
    	--endpoint /v1/chat/completions \
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
    	--concurrency ${concurrency} \
    	--request-count $(($concurrency*10)) \
    	--warmup-request-count $(($concurrency*2)) \
	    --num-dataset-entries ${num_prompts} \
    	--random-seed 100 \
    	--artifact-dir ${artifacts_dir} \
    	-- \
    	-v \
    	--max-threads ${concurrency} \
    	-H 'Authorization: Bearer NOT USED' \
    	-H 'Accept: text/event-stream'
    echo "Benchmark with concurrency ${concurrency} done"
    do_get_logs ${log_path}/output_workers.log ${log_path}/concurrency_${concurrency}
    echo -n "" > ${log_path}/output_workers.log
done


job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi
