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

# Using `dynamo serve` to deploy inference graphs locally

This guide explains how to create, configure, and deploy inference graphs for large language models using the `dynamo serve` command.

## Table of Contents

- [What are inference graphs?](#what-are-inference-graphs)
- [Creating an inference graph](#creating-an-inference-graph)
- [Serving the inference graph](#deploying-the-inference-graph)
- [Guided Example](#guided-example)

## What are inference graphs?

Inference graphs are compositions of service components that work together to handle LLM inference. A typical graph might include:

- Frontend: OpenAI-compatible HTTP server that handles incoming requests
- Processor: Processes requests before passing to workers
- Router: Routes requests to appropriate workers based on specified strategy
- Workers: Handle the actual LLM inference (prefill and decode phases)

## Creating an inference graph

Once you've written your various Dynamo services (docs on how to write these can be found [here](https://github.com/ai-dynamo/dynamo/blob/main/deploy/dynamo/sdk/docs/sdk/README.md)), you can create an inference graph by composing these services together using the following two mechanisms:

### 1. Dependencies with `depends()`

```python
from components.worker import VllmWorker

class Processor:
    worker = depends(VllmWorker)

    # Now you can call worker methods directly
    async def process(self, request):
        result = await self.worker.generate(request)
```

Benefits of `depends()`:

- Automatically ensures dependent services are deployed
- Creates type-safe client connections between services
- Allows calling dependent service methods directly

### 2. Dynamic composition with `.link()`

```python
# From examples/llm/graphs/agg.py
from components.frontend import Frontend
from components.processor import Processor
from components.worker import VllmWorker

Frontend.link(Processor).link(VllmWorker)
```

This creates a graph where:

- Frontend depends on Processor
- Processor depends on VllmWorker

The `.link()` method is useful for:

- Dynamically building graphs at runtime
- Selectively activating specific dependencies
- Creating different graph configurations from the same components

## Deploying the inference graph

Once you've defined your inference graph and its configuration, you can deploy it locally using the `dynamo serve` command! We recommend running the `--dry-run` command so you can see what arguments will be pasesd into your final graph. And then

Lets walk through an example.

## Guided Example

The files referenced here can be found [here](https://github.com/ai-dynamo/dynamo/blob/main/examples/llm/components). You will need 1 GPU minimum to run this example. This example can be run from the `examples/llm` directory

### 1. Define your components

In this example we'll be deploying an aggregated serving graph. Our components include:

1. Frontend - OpenAI-compatible HTTP server that handles incoming requests
2. Processor - Runs processing steps and routes the request to a worker
3. VllmWorker - Handles the prefill and decode phases of the request

```python
# components/frontend.py
class Frontend:
    worker = depends(VllmWorker)
    worker_routerless = depends(VllmWorkerRouterLess)
    processor = depends(Processor)

    ...
```

```python
# components/processor.py
class Processor(ProcessMixIn):
    worker = depends(VllmWorker)
    router = depends(Router)

    ...
```

```python
# components/worker.py
class VllmWorker:
    prefill_worker = depends(PrefillWorker)

    ...
```

Note that our prebuilt components have the maximal set of dependancies needed to run the component. This allows you to plug in different components to the same graph to create different architectures. When you write your own components, you can be as flexible as you'd like.

### 2. Define your graph

```python
# graphs/agg.py
from components.frontend import Frontend
from components.processor import Processor
from components.worker import VllmWorker

Frontend.link(Processor).link(VllmWorker)
```

### 3. Define your configuration

We've provided a set of basic configurations for this example [here](https://github.com/ai-dynamo/dynamo/blob/main/examples/llm/configs/agg.yaml). All of these can be changed and also be overridden by passing in CLI flags to serve!

### 4. Serve your graph

As a prerequisite, ensure you have NATS and etcd running by running the docker compose in the deploy directory. You can find it [here](https://github.com/ai-dynamo/dynamo/blob/main/deploy/docker-compose.yml).

```bash
docker compose up -d
```

Note that the we point toward the first node in our graph. In this case, it's the `Frontend` service.

```bash
# check out the configuration that will be used when we serve
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml --dry-run
```

This will print out something like

```bash
Service Configuration:
{
  "Common": {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "block-size": 64,
    "max-model-len": 16384,
  },
  "Frontend": {
    "served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "endpoint": "dynamo.Processor.chat/completions",
    "port": 8000
  },
  "Processor": {
    "router": "round-robin",
    "common-configs": [model, block-size, max-model-len]
  },
  "VllmWorker": {
    "enforce-eager": true,
    "max-num-batched-tokens": 16384,
    "enable-prefix-caching": true,
    "router": "random",
    "tensor-parallel-size": 1,
    "ServiceArgs": {
      "workers": 1
    },
    "common-configs": [model, block-size, max-model-len]
  }
}

Environment Variable that would be set:
DYNAMO_SERVICE_CONFIG={"Common": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384}, "Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}, "common-configs": ["model", "block-size", "max-model-len"]}}
```

You can override any of these configuration options by passing in CLI flags to serve. For example, to change the routing strategy, you can run

```bash
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml --Processor.router=random --dry-run
```

Which will print out something like

```bash
  #...
  "Processor": {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "block-size": 64,
    "max-model-len": 16384,
    "router": "random"
  },
  #...
```

Once you're ready - simply remove the `--dry-run` flag to serve your graph!

```bash
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

Once everything is running, you can test your graph by making a request to the frontend from a different window.

```bash
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

## Close deployment

> [!IMPORTANT]
> We are aware of an issue where vLLM subprocesses might not be killed when `ctrl-c` is pressed.
> We are working on addressing this. Relevant vLLM issues can be found [here](https://github.com/vllm-project/vllm/pull/8492) and [here](https://github.com/vllm-project/vllm/issues/6219#issuecomment-2439257824).

To stop the serve, you can press `ctrl-c` which will kill the different components. In order to kill the remaining vLLM subprocesses you can run `nvidia-smi` and `kill -9` the remaining processes or run `pkill python3` from inside of the container.
