#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


set -ex


export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install hf-transfer

rm -rf default.etcd

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../utils.sh
env | grep VLLM_

setup_cleanup_trap

nats-server -js -p 4222 -m 8222 &
etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379 &

wait_for_nats "http://127.0.0.1:8222" || exit 1
wait_for_etcd "http://127.0.0.1:2379" || exit 1

cd /workspace/examples/llm
CONFIG_FILE="configs/agg.yaml"
SERVED_MODEL_NAME=$(yq .Frontend.model $CONFIG_FILE)
HOST="127.0.0.1"
PORT=$(yq .Frontend.port $CONFIG_FILE)

dynamo serve graphs.agg:Frontend -f $CONFIG_FILE &
wait_for_server "$HOST:$PORT" $SERVED_MODEL_NAME || exit 1

echo "Checking if responses are deterministic..."
first_response=""
for i in {1..10}; do
    echo "Request $i"
    response=$(curl -s -X POST http://$HOST:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": '$SERVED_MODEL_NAME',
                "messages": [
                    {
                        "role": "user",
                        "content": "What is capital of France?"
                    }
                ],
                "seed": 1,
                "temperature": 0.0,
                "stream": false,
                "max_tokens": 100
            }')

    completion=$(echo "$response" | jq .choices[0].message.content)

    if [ $i -eq 1 ]; then
        first_response="$completion"
        if [[ "$first_response" != *"Paris"* ]]; then
            echo "=============================================================="
            echo "❌ FAILURE: Expected 'Paris' in the response but it was not found"
            echo "Got: $first_response"
            echo "=============================================================="
            exit 1
        fi
        echo "First response: $first_response"
    else
        if [ "$first_response" != "$completion" ]; then
            echo "=============================================================="
            echo "❌ FAILURE: Response $i differs from first response"
            echo "Expected: $first_response"
            echo "Got: $completion"
            echo "=============================================================="
            exit 1
        fi
    fi
done

echo "=============================================================="
echo "✅ SUCCESS: All responses are identical - deterministic behavior confirmed"
echo "=============================================================="
exit 0