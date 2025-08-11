#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
max_batch_size: ${max_batch}
max_num_tokens: ${max_num_tokens}
max_seq_len: ${max_seq_len}
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
    max_num_tokens: 37376
EOF
else
cat << EOF > ${extra_llm_api_file}
tensor_parallel_size: ${tp_size}
moe_expert_parallel_size: ${ep_size}
max_batch_size: ${max_batch}
max_num_tokens: ${max_num_tokens}
max_seq_len: ${max_seq_len}
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
    max_num_tokens: 37376
EOF
fi



echo "extra_llm_api_file generated: ${extra_llm_api_file}"
cat ${extra_llm_api_file}

echo "TRT_LLM_VERSION: $TRT_LLM_VERSION"
echo "TRT_LLM_GIT_COMMIT: $TRT_LLM_GIT_COMMIT"

# start the server
trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path $model_path --served-model-name $model_name --extra-engine-args ${extra_llm_api_file}

