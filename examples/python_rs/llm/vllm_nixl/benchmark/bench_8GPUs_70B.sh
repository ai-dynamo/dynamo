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

ENDPOINT_HOST="localhost"
ENDPOINT_PORT="8080"
ENDPOINT_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT"

MEAN_INPUT_TOKENS=3000
MEAN_OUTPUT_TOKENS=150
IO_PREFIX="isl_shared_0_isl_unique_${MEAN_INPUT_TOKENS}_osl_${MEAN_OUTPUT_TOKENS}"

CHAT_MODEL_NAME="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"

nats-server \
    -js \
    -p 4222 \
    -m 8222 &

echo "Waiting for NATS server to start..."
sleep 5

echo "Starting etcd server..."
etcd &

echo "Waiting for etcd server to start..."
sleep 5

echo "Starting HTTP server endpoint..."
http --host $ENDPOINT_HOST --port $ENDPOINT_PORT &

echo "Waiting for HTTP server to start..."
sleep 5

echo "Adding model to HTTP server..."
llmctl http add chat-models $CHAT_MODEL_NAME test-nixl.vllm.generate

echo "Waiting for model to be added..."
sleep 15

echo "Activating Triton environment..."

cd /workspace/examples/python_rs/llm/vllm_nixl


echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=0,1 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len 10000 \
    --max-num-seqs 2 \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &

echo "Starting prefill worker..."

CUDA_VISIBLE_DEVICES=2,3 python3 prefill_worker.py \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --max-model-len 10000 \
    --max-num-seqs 2 \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &


echo "Starting decode worker..."

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 worker.py \
    --remote-prefill \
    --model $CHAT_MODEL_NAME \
    --tensor-parallel-size 4 \
    --max-model-len 3500 \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' &

echo "Running benchmark..."

CONFIG_PREFIX="prefill_tp2dp2_generate_tp4dp1"

ARTIFACT_DIR_PREFIX="./artifacts/$IO_PREFIX/$CONFIG_PREFIX"

mkdir -p $ARTIFACT_DIR_PREFIX

for p in {0..7}; do
    CONCURRENCY=$((2**p))
    echo "Running benchmark for concurrency $CONCURRENCY..."
    python3 /workspace/examples/python_rs/llm/vllm_nixl/benchmark/benchmark.py \
        --isl-shared 0 \
        --isl-unique $MEAN_INPUT_TOKENS \
        --osl $MEAN_OUTPUT_TOKENS \
        --model $CHAT_MODEL_NAME \
        --tokenizer $CHAT_MODEL_NAME \
        --url $ENDPOINT_URL \
        --artifact-dir $ARTIFACT_DIR_PREFIX \
        --load-type concurrency \
        --load-value $CONCURRENCY
done


pkill -f nats-server   || true
pkill -f etcd          || true
pkill -f "http --host $ENDPOINT_HOST --port $ENDPOINT_PORT" || true
pkill -f python3 || true

# Start vllm serve baseline using 2 GPUs

VLLM_CONFIGURE_LOGGING=0 vllm serve \
    $CHAT_MODEL_NAME \
    --tensor-parallel-size 8 \
    --port $ENDPOINT_PORT \
    --host $ENDPOINT_HOST &

sleep 15

echo "Running vllm serve baseline benchmark..."

CONFIG_PREFIX="baseline_tp8dp1"

ARTIFACT_DIR_PREFIX="./artifacts/$IO_PREFIX/$CONFIG_PREFIX"

mkdir -p $ARTIFACT_DIR_PREFIX

for p in {0..7}; do
    CONCURRENCY=$((2**p))
    echo "Running benchmark for concurrency $CONCURRENCY..."
    python3 /workspace/examples/python_rs/llm/vllm_nixl/benchmark/benchmark.py \
        --isl-shared 0 \
        --isl-unique $MEAN_INPUT_TOKENS \
        --osl $MEAN_OUTPUT_TOKENS \
        --model $CHAT_MODEL_NAME \
        --tokenizer $CHAT_MODEL_NAME \
        --url $ENDPOINT_URL \
        --artifact-dir $ARTIFACT_DIR_PREFIX \
        --load-type concurrency \
        --load-value $CONCURRENCY
done

# Kill all python3 processes from vllm serve

pkill -f python3 || true

echo "Generating plots..."

# Seaborn and matplotlib are not installed in the Triton environment
uv pip install matplotlib seaborn
python3 /workspace/examples/python_rs/llm/vllm_nixl/benchmark/process_results.py ./artifacts/$IO_PREFIX "I/O $MEAN_INPUT_TOKENS/$MEAN_OUTPUT_TOKENS $CHAT_MODEL_NAME"

echo "Done!"
