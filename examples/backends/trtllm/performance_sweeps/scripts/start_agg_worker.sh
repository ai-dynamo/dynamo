#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

model_path=$1
max_batch=$2
max_num_tokens=$3
tp_size=$4
ep_size=$5
enable_attention_dp=$6
gpu_fraction=$7
max_seq_len=$8
mtp=$9
model_name=${10}

# Validate all required parameters
if [ -z "${model_path}" ] || [ -z "${max_batch}" ] || [ -z "${max_num_tokens}" ] || [ -z "${tp_size}" ] || [ -z "${ep_size}" ] || [ -z "${enable_attention_dp}" ] || [ -z "${gpu_fraction}" ] || [ -z "${max_seq_len}" ] || [ -z "${mtp}" ] || [ -z "${model_name}" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 model_path max_batch max_num_tokens tp_size ep_size enable_attention_dp gpu_fraction max_seq_len mtp model_name"
    echo ""
    echo "Parameters:"
    echo "  model_path: Path to the model"
    echo "  max_batch: Maximum batch size (integer)"
    echo "  max_num_tokens: Maximum number of tokens (integer)"
    echo "  tp_size: Tensor parallel size (integer)"
    echo "  ep_size: Expert parallel size (integer)"
    echo "  enable_attention_dp: Enable attention data parallel (true/false)"
    echo "  gpu_fraction: GPU memory fraction (float 0.0-1.0)"
    echo "  max_seq_len: Maximum sequence length (integer)"
    echo "  mtp: MTP size (integer)"
    echo "  model_name: Name of the model to serve"
    exit 1
fi

# Validate numeric parameters
if ! [[ "${max_batch}" =~ ^[0-9]+$ ]]; then
    echo "Error: max_batch must be a positive integer, got: ${max_batch}"
    exit 1
fi

if ! [[ "${max_num_tokens}" =~ ^[0-9]+$ ]]; then
    echo "Error: max_num_tokens must be a positive integer, got: ${max_num_tokens}"
    exit 1
fi

if ! [[ "${tp_size}" =~ ^[0-9]+$ ]]; then
    echo "Error: tp_size must be a positive integer, got: ${tp_size}"
    exit 1
fi

if ! [[ "${ep_size}" =~ ^[0-9]+$ ]]; then
    echo "Error: ep_size must be a positive integer, got: ${ep_size}"
    exit 1
fi

if ! [[ "${gpu_fraction}" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "${gpu_fraction} <= 0" | bc -l) )) || (( $(echo "${gpu_fraction} > 1" | bc -l) )); then
    echo "Error: gpu_fraction must be a float between 0.0 and 1.0, got: ${gpu_fraction}"
    exit 1
fi

if ! [[ "${max_seq_len}" =~ ^[0-9]+$ ]]; then
    echo "Error: max_seq_len must be a positive integer, got: ${max_seq_len}"
    exit 1
fi

if ! [[ "${mtp}" =~ ^[0-9]+$ ]]; then
    echo "Error: mtp must be a positive integer, got: ${mtp}"
    exit 1
fi

# Validate enable_attention_dp is true or false
if [ "${enable_attention_dp}" != "true" ] && [ "${enable_attention_dp}" != "false" ]; then
    echo "Error: enable_attention_dp must be 'true' or 'false', got: ${enable_attention_dp}"
    exit 1
fi

# echo all parameters
echo "model_path: ${model_path}"
echo "max_batch: ${max_batch}"
echo "max_num_tokens: ${max_num_tokens}"
echo "tp_size: ${tp_size}"
echo "ep_size: ${ep_size}"
echo "enable_attention_dp: ${enable_attention_dp}"
echo "gpu_fraction: ${gpu_fraction}"
echo "max_seq_len: ${max_seq_len}"
echo "mtp: ${mtp}"

# check enable_attention_dp is true or false
if [ ${enable_attention_dp} == "true" ]; then
    enable_attention_dp_flag="true"
    moe_backend="CUTLASS"
else
    enable_attention_dp_flag="false"
    moe_backend="TRTLLM"
fi

extra_llm_api_file=/tmp/extra-llm-api-config.yml

if [ ${mtp} -gt 0 ]; then
cat << EOF > ${extra_llm_api_file}
tensor_parallel_size: ${tp_size}
moe_expert_parallel_size: ${ep_size}
trust_remote_code: true
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch}
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: ${gpu_fraction}
    enable_block_reuse: false
print_iter_log: true
enable_attention_dp: ${enable_attention_dp_flag}
stream_interval: 10
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: ${mtp}
moe_config:
    backend: ${moe_backend}
EOF
else
cat << EOF > ${extra_llm_api_file}
tensor_parallel_size: ${tp_size}
moe_expert_parallel_size: ${ep_size}
trust_remote_code: true
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${max_batch}
enable_attention_dp: ${enable_attention_dp_flag}
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: ${gpu_fraction}
    enable_block_reuse: false
stream_interval: 10
moe_config:
    backend: ${moe_backend}
EOF
fi



echo "extra_llm_api_file generated: ${extra_llm_api_file}"
cat ${extra_llm_api_file}

echo "TRT_LLM_VERSION: $TRT_LLM_VERSION"
echo "TRT_LLM_GIT_COMMIT: $TRT_LLM_GIT_COMMIT"

# start the server
trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path $model_path \
    --served-model-name $model_name \
    --max-num-tokens ${max_num_tokens} \
    --max-batch-size ${max_batch} \
    --max-seq-len ${max_seq_len} \
    --extra-engine-args ${extra_llm_api_file}


