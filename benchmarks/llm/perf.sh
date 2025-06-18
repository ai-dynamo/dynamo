#/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Parse command line arguments
model="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
url="http://localhost:8000"
mode="aggregated"
artifacts_root_dir="artifacts_root"
deployment_kind="dynamo"

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

# The defaults can be overridden by command line arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --tensor-parallelism)
      tp="$2"
      shift 2
      ;;
    --data-parallelism)
      dp="$2"
      shift 2
      ;;
    --prefill-tensor-parallelism)
      prefill_tp="$2"
      shift 2
      ;;
    --prefill-data-parallelism)
      prefill_dp="$2"
      shift 2
      ;;
    --decode-tensor-parallelism)
      decode_tp="$2"
      shift 2
      ;;
    --decode-data-parallelism)
      decode_dp="$2"
      shift 2
      ;;
      --model)
      model="$2"
      shift 2
      ;;
    --input-sequence-length)
      isl="$2"
      shift 2
      ;;
    --output-sequence-length)
      osl="$2"
      shift 2
      ;;
    --url)
      url="$2"
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

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

# Concurrency levels to test
for concurrency in 1 2 4 8 16 32 64 128 256; do

  # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
  # `ignore_eos` since they are not in the official OpenAI spec.
  genai-perf profile \
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
    -- \
    -v \
    --max-threads 256 \
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
