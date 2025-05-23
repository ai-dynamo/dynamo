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

This directory contains examples for deploying vLLM (v0) models in both aggregated and disaggregated configurations.

> [!NOTE]
> Different than `/examples/llm`, this example uses `dynamo-run` to handle the (de)tokenization and routing. `dynamo-run` is a rust-based CLI designed for high-performance pre/post-processing and routing. Read more about `dynamo-run`: [dynamo_run.md](../docs/guides/dynamo_run.md).

## Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml)
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build and run docker

```bash
# On an x86 machine
./container/build.sh --framework vllm

# On an ARM machine (ex: GB200)
./container/build.sh --framework vllm --platform linux/arm64

./container/run.sh -it --framework vllm
```

> [!WARNING]
> Starting the container not in `--privileged` mode might result in significant CPU bottlenecks. Please turn on `--privileged` if you experience any performance issues.


## Run Deployment

```bash
# aggregated
cd $DYNAMO_HOME/examples/vllm_v0
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml

# aggregated with kv
cd $DYNAMO_HOME/examples/vllm_v0
dynamo serve graphs.agg:Frontend -f ./configs/agg_kv.yaml

# disaggregated
cd $DYNAMO_HOME/examples/vllm_v0
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml

# disaggregated with kv
cd $DYNAMO_HOME/examples/vllm_v0
dynamo serve graphs.disagg:Frontend -f ./configs/disagg_kv.yaml
```

## Client

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