#!/bin/bash
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

# default values
model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
p_tensor_parallel_size=1
d_tensor_parallel_size=1
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=1024
gpu_memory_utilization=0.9
enable_chunked_prefill=False
block_size=64

num_p=2
num_d=2
total_rank=$((num_p + num_d))
curr_kv_rank=0

# Function to display usage
usage() {
    echo "Usage: $0 [--model <model>] [--p_tensor_parallel_size <size>] [--d_tensor_parallel_size <size>] [--max_model_len <len>] [--max_num_batched_tokens <tokens>] [--max_num_seqs <seqs>] [--gpu_memory_utilization <utilization>] [--enable_chunked_prefill <True/False>] [--num_p <p>] [--num_d <d>]"
    exit 1
}

# Parse the command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --p_tensor_parallel_size)
            p_tensor_parallel_size="$2"
            shift 2
            ;;
        --d_tensor_parallel_size)
            d_tensor_parallel_size="$2"
            shift 2
            ;;
        --max_model_len)
            max_model_len="$2"
            shift 2
            ;;
        --max_num_batched_tokens)
            max_num_batched_tokens="$2"
            shift 2
            ;;
        --max_num_seqs)
            max_num_seqs="$2"
            shift 2
            ;;
        --gpu_memory_utilization)
            gpu_memory_utilization="$2"
            shift 2
            ;;
        --enable_chunked_prefill)
            enable_chunked_prefill="$2"
            shift 2
            ;;
        --num_p)
            num_p="$2"
            shift 2
            ;;
        --num_d)
            num_d="$2"
            shift 2
            ;;
        --total_rank)
            total_rank="$2"
            shift 2
            ;;
        --curr_kv_rank)
            curr_kv_rank="$2"
            shift 2
            ;;
        --block_size)
            block_size="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# rank here is GPU rank
curr_rank=0

echo "total rank: "${total_rank}

for (( i=1; i<=num_d; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $d_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    echo "starting gpu rank "${cuda_devices}" (decode)"

    CUDA_VISIBLE_DEVICES=${cuda_devices} python3 worker.py \
    --remote-prefill \
    --model ${model} \
    --max-model-len ${max_model_len} \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --enable-chunked-prefill ${enable_chunked_prefill} \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --enforce-eager \
    --enable-prefix-caching \
    --tensor-parallel-size ${d_tensor_parallel_size} \
    --block-size ${block_size} \
    --kv-transfer-config '{"kv_connector":"dynamoNixlConnector"}' & disown
    curr_rank=$((curr_rank + d_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done

for (( i=1; i<=num_p; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $p_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    echo "starting gpu rank "${cuda_devices}" (prefill)"

    CUDA_VISIBLE_DEVICES=${cuda_devices} python3 prefill_worker.py \
    --model ${model} \
    --max-model-len ${max_model_len} \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --enable-chunked-prefill ${enable_chunked_prefill} \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --enforce-eager \
    --tensor-parallel-size ${p_tensor_parallel_size} \
    --block-size ${block_size} \
    --kv-transfer-config '{"kv_connector":"dynamoNixlConnector"}' & disown
    curr_rank=$((curr_rank + p_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done

