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

WAIT_TIME=300

model=${1}
multi_round=${2}
num_gen_servers=${3}
concurrency_list=${4}
streaming=${5}
log_path=${6}
prefill_gpus=${7}
decode_gpus=${8}
total_gpus=$((prefill_gpus+decode_gpus))
model_path=${9}
isl=${10}
osl=${11}
kind=${12}

if [ "$#" -ne 12 ]; then
    echo "Error: Expected 12 arguments, got $#"
    echo "Usage: $0 <model> <multi_round> <num_gen_servers> <concurrency_list> <streaming> <log_path> <prefill_gpus> <decode_gpus> <model_path> <isl> <osl> <kind>"
    exit 1
fi

echo "Arguments:"
echo "  model: $model"
echo "  multi_round: $multi_round"
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

# Wait for server to become healthy (up to 50 attempts)
failed=true
for ((i=1; i<=50; i++)); do
    sleep $((i == 1 ? WAIT_TIME : 20))
    response=$(curl -s -w "\n%{http_code}" "${hostname}:${port}/health")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    # NOTE: This can be replaced with simply checking /v1/models endpoint after this PR:
    # https://github.com/ai-dynamo/dynamo/pull/2683
    if [[ "$http_code" == "200" ]] && echo "$body" | grep -q '"status":"healthy"' && echo "$body" | grep -q '"endpoints":\[[^]]*"dyn://dynamo.tensorrt_llm.generate"'; then
        if [[ "$kind" == *disagg* ]]; then
            if echo "$body" | grep -q '"dyn://dynamo.prefill.generate"'; then
                echo "Health check succeeded on attempt $i"
                echo "$body"
                failed=false
                break
            else
                echo "Attempt $i: prefill generate endpoint not found in etcd."
            fi
        else
            echo "Health check succeeded on attempt $i"
            echo "$body"
            failed=false
            break
        fi
    else
        echo "Attempt $i failed: /health not ready (HTTP $http_code)."
    fi
done

if [[ "$failed" == "true" ]]; then
    echo "Server did not respond with healthy status after 50 attempts."
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

python3 ${SCRIPTS_DIR}/scripts/bench/benchmark_serving.py \
        --served-model-name ${model} \
        --model ${model_path} \
        --dataset-name random \
        --num-prompts "${multi_round}" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --use-chat-template \
        --backend "dynamo" \
        --endpoint "/v1/completions" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "1" \
        --host ${hostname} \
        --port ${port}

mkdir -p ${log_path}/results
echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    original_concurrency=${concurrency}
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}

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

    echo "Benchmark with concurrency ${concurrency} done"
done


job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi
