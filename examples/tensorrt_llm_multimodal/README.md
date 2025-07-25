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

# LLM Deployment Examples using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.


## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture.
Note that this TensorRT-LLM version does not support all the options yet.

Note: TensorRT-LLM disaggregation does not support conditional disaggregation yet. You can only configure the deployment to always use aggregate or disaggregated serving.

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml)
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build docker

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
./container/build.sh --framework tensorrtllm

# On an ARM machine:
./container/build.sh --framework tensorrtllm --platform linux/arm64
```

> [!NOTE]
> Because of a known issue of C++11 ABI compatibility within the NGC pytorch container,
> we rebuild TensorRT-LLM from source. See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
> for more information.
>
> Hence, when running this script for the first time, the time taken by this script can be
> quite long.


### Run container

```
./container/run.sh --framework tensorrtllm -it
```
## Run Deployment

This figure shows an overview of the major components to deploy:



```

+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| processor |----->|      Worker      |------------>|     Prefill   |
|      |<-----|           |<-----|                  |<------------|     Worker    |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+

```

Note: The above architecture illustrates all the components. The final components
that get spawned depend upon the chosen graph.

### Example architectures

#### Aggregated serving
```bash
cd /workspace/examples/tensorrt_llm_multimodal
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Aggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm_multimodal
dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml
```

#### Disaggregated serving
```bash
cd /workspace/examples/tensorrt_llm_multimodal
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

#### Disaggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm_multimodal
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml
```
#### Example Client
```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
{"id":"chatcmpl-3b112e60192e490fb92a0a474d7f2013","choices":[{"index":0,"message":{"content":"TheThe image depicts a turbulent sea under a stormy sky, with large waves crashing against each other. The water is dark and choppy, with white foam forming at the crests of the waves. The sky above is overcast and gray, with thick clouds that suggest an impending storm. The overall atmosphere of the image is one of power and intensity, capturing the raw energy of the ocean in a moment of turmoil.<|eot|>","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1752191807,"model":"meta-llama/Llama-4-Scout-17B-16E-Instruct","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}
```

#### Additional notes (Must do before inference)
1. All vision models mentioned here are supported : https://github.com/NVIDIA/TensorRT-LLM/tree/v1.0.0rc0/examples/pytorch (Except Llama-4-Maverick it is WIP)
2. Apply [this](https://gist.github.com/chang-l/81cec031267f92b7a6e2b7a70a4c76e1) patch to your tensorrt_llm installation inside the dynamo container to enable llama4 models
Location : `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_llama.py`
