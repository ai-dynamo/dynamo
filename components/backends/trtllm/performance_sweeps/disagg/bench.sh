#!/bin/bash

# 添加错误处理
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# 添加参数检查
if [ "$#" -lt 6 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 model_name multi_round concurrency_list streaming log_path total_gpus artifacts_root_dir model_path isl osl"
    exit 1
fi


model=$1
multi_round=$2
num_gen_servers=$3
concurrency_list=$4
streaming=$5
log_path=$6
total_gpus=$7
artifacts_root_dir=$8
model_path=$9
isl=${10}
osl=${11}


if [ $# -lt 12 ]; then
  echo "Usage: $0 $model $multi_round $num_gen_servers $concurrency_list $streaming $log_path $total_gpus $artifacts_root_dir $model_path $isl $osl"
  exit 1
fi

# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

set -x
config_file=${log_path}/config.yaml

# check if the config file exists every 10 seconds timeout 1800 seconds
timeout=1800

# Create artifacts root directory if it doesn't exist
if [ ! -d "${artifacts_root_dir}" ]; then
    mkdir -p "${artifacts_root_dir}"
fi

# Find the next available artifacts directory index
index=0
while [ -d "${artifacts_root_dir}/artifacts_${index}" ]; do
    index=$((index + 1))
done

# Create the new artifacts directory
artifact_dir="${artifacts_root_dir}/artifacts_${index}"
mkdir -p "${artifact_dir}"

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

pip install genai-perf

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
    	--artifact-dir ${artifact_dir} \
    	-- \
    	-v \
    	--max-threads ${concurrency} \
    	-H 'Authorization: Bearer NOT USED' \
    	-H 'Accept: text/event-stream'
    echo "Benchmark with concurrency ${concurrency} done"
    do_get_logs ${log_path}/output_workers.log ${log_path}/concurrency_${concurrency}
    echo -n "" > ${log_path}/output_workers.log
done

# The configuration is dumped to a JSON file which hold details of the OAI service
# being benchmarked.
deployment_config=$(cat << EOF
{
  "kind": "dynamo_trtllm_wideep",
  "model": "${model}",
  "total_gpus": "${total_gpus}"
}
EOF
)

mkdir -p "${artifact_dir}"
if [ -f "${artifact_dir}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${artifact_dir}/deployment_config.json"
fi
echo "${deployment_config}" > "${artifact_dir}/deployment_config.json"


job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi

echo "Benchmark done, gracefully shutting down server and workers..."
kill -9 $(ps aux | grep '[s]tart_server.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[s]tart_worker.sh' | awk '{print $2}') >/dev/null 2>&1 || true
kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
sleep 20  # Give processes some time to clean up

# Check if there are any remaining processes
if pgrep -f "trtllm-serve"; then
    echo "Warning: Some processes may still be running"
else
    echo "All processes successfully terminated"
fi
