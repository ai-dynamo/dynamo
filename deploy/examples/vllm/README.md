<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Prerequisites

Start required services (etcd and NATS):

   Option A: Using [Docker Compose](/deploy/docker-compose.yml) (Recommended)
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```

   Option B: Manual Setup

    - [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
        - example: `nats-server -js --trace`
    - [etcd](https://etcd.io) server
        - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally

## Build docker

```
./container/build.sh
```

## Run container

```
./container/run.sh -it
```
## Run deployment

This figure shows an overview of the major components to deploy:

```
                                                 +----------------+
                                          +------| prefill worker |-------+
                                   notify |      |                |       |
                                 finished |      +----------------+       | pull
                                          v                               v
+------+      +-----------+      +------------------+    push     +---------------+
| HTTP |----->| processor |----->| decode/monolith  |------------>| prefill queue |
|      |<-----|           |<-----|      worker      |             |               |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+

```

### Disaggregated vLLM deployment

Serve following components:

- processor: Processor routes the requests to the (decode) workers. Three scheduling strategies are supported: random and kv.
- kv router: The KV Router is a component that aggregates KV Events from all the workers and maintains
a prefix tree of the cached tokens. It makes decisions on which worker to route requests
to based on the length of the prefix match and the load on the workers.

- decode worker: runs on gpu = 0
- prefill worker: runs on gpu = 1

```bash

cd /workspace/deploy/examples/vllm

dynamo serve disaggregated.processor:Processor  \
   --Processor.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   --Processor.tokenizer=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   --Processor.block-size=64 \
   --Processor.max-model-len=16384 \
   --Processor.router=kv \
   --Router.min-workers=1 \
   --Router.block-size=64 \
   --Router.model-name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   --VllmWorker.remote-prefill=true \
   --VllmWorker.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   --VllmWorker.enforce-eager=true \
   --VllmWorker.tensor-parallel-size=1 \
   --VllmWorker.kv-transfer-config='{"kv_connector": "DynamoNixlConnector"}' \
   --VllmWorker.block-size=64  \
   --VllmWorker.max-num-batched-tokens=16384 \
   --VllmWorker.max-model-len=16384 \
   --VllmWorker.router=kv \
   --VllmWorker.enable-prefix-caching=true \
   --PrefillWorker.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
   --PrefillWorker.enforce-eager=true \
   --PrefillWorker.block-size=64 \
   --PrefillWorker.max-model-len=16384 \
   --PrefillWorker.max-num-batched-tokens=16384 \
   --PrefillWorker.kv-transfer-config='{"kv_connector": "DynamoNixlConnector"}' \
   --PrefillWorker.cuda-visible-device-offset=1
```


Add model to dynamo and start http server.
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynamo-init.Processor.chat_completions

TRT_LOG=DEBUG http --port 8181
```
### Client

In another terminal:
```bash
# this test request has around 200 tokens isl

curl localhost:8181/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```

### Close deployment

Kill all python processes and clean up metadata files:

```
pkill -9 -f python
```