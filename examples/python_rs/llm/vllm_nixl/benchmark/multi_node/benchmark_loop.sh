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

# Parser arguments and print help for lack of arguments
# Arguments: endpoint host, configuration prefix

if [ -z "$1" ]; then
    echo "Usage: $0 <endpoint-host> <configuration-prefix>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <endpoint-host> <configuration-prefix>"
    exit 1
fi

FIRST_HOST=$1
CONFIG_PREFIX=$2

ENDPOINT_HOST=$FIRST_HOST
ENDPOINT_PORT="8080"
ENDPOINT_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT"

MEAN_INPUT_TOKENS=3000
MEAN_OUTPUT_TOKENS=150

IO_PREFIX="isl_shared_0_isl_unique_${MEAN_INPUT_TOKENS}_osl_${MEAN_OUTPUT_TOKENS}"

CHAT_MODEL_NAME="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"


echo "Running benchmark..."

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
