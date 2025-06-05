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

# vLLM Deployment Examples

This directory contains examples for deploying vLLM models aggregated with with DP.

## Prerequisites

1. Install vLLM:
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout d459fae0a2c464e28680bc6d564c1de1b295029e
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

2. Start required services:
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

## Running the Server

### Aggregated Deployment with Multiple disconnected DP engines

Serves the leader AsyncLLM engine + number of dp ranks you specify
```bash
cd examples/vllm_v1
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

or to run with P/D disagg:

```bash
cd examples/vllm_v1
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

To run other dp ranks headless on same node or other nodes can run

```
CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 vllm serve Qwen/Qwen3-0.6B -dp 1 -dpr 1 --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 62300 --data-parallel-size-local 1 --enforce-eager --headless --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq"}' --enable-prefix-caching &
```

To test can run this curl reqeust. KV Routing will mean this will keep routing to a single node, so you will need to switch it up to see routing to different dp workers.

```
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
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

TODO:
- Currently if you run more than one instance or worker on the same node this will fail because the ZmqKvPublishers will overlap ports, need to add some port offsetting to manage that.
```
  ServiceArgs:
    workers: 1  # 2 workers not supported
```
- It would be best to distill the vLLM serve into a VllmHeadlessWorker using - run_headless(self.engine_args). This is relatively simple, the main difficulty here is if you want to add the ZmqKvEventPublisher to these nodes (which would be easier for multi-node because then you just need to set-up nats and not worry about port stuff) they will have a different lease_id than the leader worker. This is a problem because we don't actually route requests to these dp_ranks directly but in the KV Router and KV Indexer it will see these KVEvents as coming from a seperate "worker". We still need to route the KVEvents through the leader AsyncLLM engine and that engine will take care of routing to the dp ranks.
  - To address this we could create a concept of worker groups? IE components whose lease_ids are tied to a single leader worker?


For more detailed explenations, refer to the main [LLM examples README](../llm/README.md).
