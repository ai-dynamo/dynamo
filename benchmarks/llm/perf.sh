#/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Default Values
model="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
url="http://localhost:8000"
mode="aggregated"
artifacts_root_dir="artifacts_root"
deployment_kind="dynamo"
concurrency_list="1,2,4,8,16,32,64,128,256"

# Input Sequence Length (isl) 3000 and Output Sequence Length (osl) 150 are
# selected for chat use case. Note that for other use cases, the results and
# tuning would vary.
isl=3000
osl=150

tp=0
dp=0
prefill_tp=0
prefill_dp=0
decode_tp=0
decode_dp=0

print_help() {
  echo "Usage: $0 [OPTIONS]"
  echo
  echo "Options:"
  echo "  --tensor-parallelism, --tp <int>           Tensor parallelism (default: $tp)"
  echo "  --data-parallelism, --dp <int>             Data parallelism (default: $dp)"
  echo "  --prefill-tensor-parallelism, --prefill-tp <int>   Prefill tensor parallelism (default: $prefill_tp)"
  echo "  --prefill-data-parallelism, --prefill-dp <int>     Prefill data parallelism (default: $prefill_dp)"
  echo "  --decode-tensor-parallelism, --decode-tp <int>     Decode tensor parallelism (default: $decode_tp)"
  echo "  --decode-data-parallelism, --decode-dp <int>       Decode data parallelism (default: $decode_dp)"
  echo "  --model <model_id>                         Hugging Face model ID to benchmark (default: $model)"
  echo "  --input-sequence-length, --isl <int>       Input sequence length (default: $isl)"
  echo "  --output-sequence-length, --osl <int>      Output sequence length (default: $osl)"
  echo "  --url <http://host:port>                   Target URL for inference requests (default: $url)"
  echo "  --concurrency <list>                       Comma-separated concurrency levels (default: $concurrency_list)"
  echo "  --mode <aggregated|disaggregated>          Serving mode (default: $mode)"
  echo "  --artifacts-root-dir <path>                Root directory to store benchmark results (default: $artifacts_root_dir)"
  echo "  --deployment-kind <type>                   Deployment tag used for pareto chart labels (default: $deployment_kind)"
  echo "  --help                                     Show this help message and exit"
  echo
  exit 0
}

# Parse command line arguments
# The defaults can be overridden by command line arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --tensor-parallelism|--tp)
      tp="$2"
      shift 2
      ;;
    --data-parallelism|--dp)
      dp="$2"
      shift 2
      ;;
    --prefill-tensor-parallelism|--prefill-tp)
      prefill_tp="$2"
      shift 2
      ;;
    --prefill-data-parallelism|--prefill-dp)
      prefill_dp="$2"
      shift 2
      ;;
    --decode-tensor-parallelism|--decode-tp)
      decode_tp="$2"
      shift 2
      ;;
    --decode-data-parallelism|--decode-dp)
      decode_dp="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --input-sequence-length|--isl)
      isl="$2"
      shift 2
      ;;
    --output-sequence-length|--osl)
      osl="$2"
      shift 2
      ;;
    --url)
      url="$2"
      shift 2
      ;;
    --concurrency)
      concurrency_list="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --artifacts-root-dir)
      artifacts_root_dir="$2"
      shift 2
      ;;
    --deployment-kind)
      deployment_kind="$2"
      shift 2
      ;;
    --help)
      print_help
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to validate if concurrency values are positive integers
validate_concurrency() {
  for val in "${concurrency_array[@]}"; do
    if ! [[ "$val" =~ ^[0-9]+$ ]] || [ "$val" -le 0 ]; then
      echo "Error: Invalid concurrency value '$val'. Must be a positive integer." >&2
      exit 1
    fi
  done
}

IFS=',' read -r -a concurrency_array <<< "$concurrency_list"
# Validate concurrency values
validate_concurrency

if [ "${mode}" == "aggregated" ]; then
  if [ "${tp}" == "0" ] && [ "${dp}" == "0" ]; then
    echo "--tensor-parallelism and --data-parallelism must be set for aggregated mode."
    exit 1
  fi
  echo "Starting benchmark for the deployment service with the following configuration:"
  echo "  - Tensor Parallelism: ${tp}"
  echo "  - Data Parallelism: ${dp}"
elif [ "${mode}" == "disaggregated" ]; then
  if [ "${prefill_tp}" == "0" ] && [ "${prefill_dp}" == "0" ] && [ "${decode_tp}" == "0" ] && [ "${decode_dp}" == "0" ]; then
    echo "--prefill-tensor-parallelism, --prefill-data-parallelism, --decode-tensor-parallelism and --decode-data-parallelism must be set for disaggregated mode."
    exit 1
  fi
  echo "Starting benchmark for the deployment service with the following configuration:"
  echo "  - Prefill Tensor Parallelism: ${prefill_tp}"
  echo "  - Prefill Data Parallelism: ${prefill_dp}"
  echo "  - Decode Tensor Parallelism: ${decode_tp}"
  echo "  - Decode Data Parallelism: ${decode_dp}"
else
  echo "Unknown mode: ${mode}. Only 'aggregated' and 'disaggregated' modes are supported."
  exit 1
fi

echo "--------------------------------"
echo "WARNING: This script does not validate tensor_parallelism=${tp} and data_parallelism=${dp} settings."
echo "         The user is responsible for ensuring these match the deployment configuration being benchmarked."
echo "         Incorrect settings may lead to misleading benchmark results."
echo "--------------------------------"


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

# Print warning about existing artifacts directories
if [ $index -gt 0 ]; then
    echo "--------------------------------"
    echo "WARNING: Found ${index} existing artifacts directories:"
    for ((i=0; i<index; i++)); do
        if [ -f "${artifacts_root_dir}/artifacts_${i}/deployment_config.json" ]; then
            echo "artifacts_${i}:"
            cat "${artifacts_root_dir}/artifacts_${i}/deployment_config.json"
            echo "--------------------------------"
        fi
    done
    echo "Creating new artifacts directory: artifacts_${index}"
    echo "--------------------------------"
fi

echo "Running aiperf with:"
echo "Model: $model"
echo "ISL: $isl"
echo "OSL: $osl"
echo "Concurrency levels: ${concurrency_array[@]}"

# Concurrency levels to test
for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"

  # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
  # `ignore_eos` since they are not in the official OpenAI spec.
  aiperf profile \
    --model ${model} \
    --tokenizer ${model} \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url ${url} \
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
    --num-dataset-entries $(($concurrency*12)) \
    --random-seed 100 \
    --artifact-dir ${artifact_dir} \
    --ui simple \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'

done

# The configuration is dumped to a JSON file which hold details of the OAI service
# being benchmarked.
deployment_config=$(cat << EOF
{
  "kind": "${deployment_kind}",
  "model": "${model}",
  "input_sequence_length": ${isl},
  "output_sequence_length": ${osl},
  "tensor_parallelism": ${tp},
  "data_parallelism": ${dp},
  "prefill_tensor_parallelism": ${prefill_tp},
  "prefill_data_parallelism": ${prefill_dp},
  "decode_tensor_parallelism": ${decode_tp},
  "decode_data_parallelism": ${decode_dp},
  "mode": "${mode}"
}
EOF
)

mkdir -p "${artifact_dir}"
if [ -f "${artifact_dir}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${artifact_dir}/deployment_config.json"
fi
echo "${deployment_config}" > "${artifact_dir}/deployment_config.json"

echo "Benchmarking Successful!!!"
