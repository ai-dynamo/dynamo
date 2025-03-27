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
# Arguments: endpoint host

if [ -z "$1" ]; then
    echo "Usage: $0 <endpoint-host> <configuration-prefix>"
    exit 1
fi

FIRST_HOST=$1

export NATS_SERVER="nats://${FIRST_HOST}:4222"
export ETCD_ENDPOINTS="http://${FIRST_HOST}:2379"

# NATS and etcd should be running on first node
# NATS_SERVER and ETCD_ENDPOINTS should be set

ENDPOINT_HOST=$FIRST_HOST
ENDPOINT_PORT="8080"
ENDPOINT_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT"


CHAT_MODEL_NAME="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"

nats-server \
    -addr $FIRST_HOST \
    -js \
    -p 4222 \
    -m 8222 &

echo "Waiting for NATS server to start..."
sleep 5

echo "Starting etcd server..."
etcd \
	--listen-client-urls http://$FIRST_HOST:2379 \
	--advertise-client-urls http://$FIRST_HOST:2379 &

echo "Waiting for etcd server to start..."
sleep 5

echo "Starting HTTP server endpoint..."
http --host $ENDPOINT_HOST --port $ENDPOINT_PORT &

echo "Waiting for HTTP server to start..."
sleep 5

echo "Adding model to HTTP server..."
llmctl http add chat-models $CHAT_MODEL_NAME test-nixl.vllm.generate
