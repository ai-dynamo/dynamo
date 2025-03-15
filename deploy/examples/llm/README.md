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

# LLM Deployment Examples

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations.

## Components

- workers: Prefill and decode worker handles actual LLM inference
- router: Handles API requests and routes them to appropriate workers based on specified strategy
- frontend: OpenAI compatible http server handles incoming requests

## Deployment Architectures

### Monolith
Single-instance deployment where both prefill and decode are done by the same worker.

### Disaggregated
Distributed deployment where prefill and decode are done by separate workers that can scale independently.

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

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

### Build docker

```
./container/build.sh
```

### Run container

```
./container/run.sh -it
```
## Run Deployment Architecture Locally

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

### Example architectures

#### Router based monolith
```bash
cd /workspace/deploy/examples/llm
dynamo serve monolith.ex_router:Frontend -f ./configs/monolith/router.yaml
```

#### Routerless monolith
```bash
cd /workspace/deploy/examples/llm
dynamo serve monolith.ex_routerless:Frontend -f ./configs/monolith/routerless.yaml
```

#### Router based disaggregated serving
```bash
cd /workspace/deploy/examples/llm
dynamo serve disaggregated.ex_router:Frontend -f ./configs/disaggregated/router.yaml
```

#### Routerless disaggregated serving
```bash
cd /workspace/deploy/examples/llm
dynamo serve disaggregated.ex_routerless:Frontend -f ./configs/disaggregated/routerless.yaml
```

### Client

In another terminal:
```bash
# this test request has around 200 tokens isl

curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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