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

set -e
set -x

# Parser arguments and print help for lack of arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <endpoint-host>"
    exit 1
fi

FIRST_HOST=$1
# NATS and etcd should be running on first node
# NATS_SERVER and ETCD_ENDPOINTS should be set

export NATS_SERVER="nats://${FIRST_HOST}:4222"
export ETCD_ENDPOINTS="http://${FIRST_HOST}:2379"


BLOCK_SIZE=128
MAX_MODEL_LEN=10000
MAX_NUM_SEQS=2


CHAT_MODEL_NAME="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"


cd /workspace/examples/python_rs/llm/vllm_nixl


echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=0,1 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &

echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=2,3 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &

echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=4,5 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &


echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=6,7 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}'


