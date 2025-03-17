<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA Dynamo

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments. Dynamo is designed to be inference engine agnostic (supports TRT-LLM, vLLM, SGLang or others) and captures LLM-specific capabilities such as:

- **Disaggregated prefill & decode inference** – Maximizes GPU throughput and facilitates trade off between throughput and latency.
- **Dynamic GPU scheduling** – Optimizes performance based on fluctuating demand
- **LLM-aware request routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated data transfer** – Reduces inference response time using NIXL.
- **KV cache offloading** – Leverages multiple memory hierarchies for higher system throughput

Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

| [Quick Start](#quick-start) | [Distributed LLM Serving](#serving-a-distributed-llm-serving-solution) | [Architecture](docs/architecture.md)| [Disaggregated Serving and KV Routing Examples](examples/llm) |

## Quick Start

### Installation

The following examples require a few system level packages.
We also leverage `uv` to manage a Python virtual environment.

#### System Packages

```
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev curl libucx0
```

#### Install uv Package Manager [Optional]

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

#### Install Dynamo


```bash
uv venv dynamo-venv
source dynamo-venv/bin/activate
uv pip install ai-dynamo[all]
```

### Running and Interacting with an LLM Locally

To run a model and interact with it locally you can call `dynamo
run` with a hugging face model. `dynamo run` supports several backends
including: `mistralrs`, `sglang`, `vllm`, and `tensorrtllm`.

#### Example Command

```
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

Once launched you'll be able to chat directly with the model from the
command line.

#### Example Output
```
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
2025-03-17T14:03:14.122652Z  INFO dynamo_run: CPU mode. Rebuild with `--features cuda|metal|vulkan` for better performance
2025-03-17T14:03:14.124227Z  INFO candle_hf_hub: Token file not found "/root/.cache/huggingface/token"
INFO 03-17 14:03:16 __init__.py:190] Automatically detected platform cuda.
INFO 03-17 14:03:16 nixl.py:16] NIXL is available
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.02s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.15s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.14s/it]
Capturing CUDA graph shapes: 100%|██████████| 35/35 [00:17<00:00,  2.04it/s]
2025-03-17T14:03:50.190381Z  INFO dynamo_run::input::text: Ctrl-c to exit
```

```
? User › Hello, how are you?
✔ User · Hello, how are you?
Okay, so I'm trying to figure out how to respond to the user's greeting. They said, "Hello, how are you?" and then followed it with "Hello! I'm just a program, but thanks for asking." Hmm, I need to come up with a suitable reply. ...
```

### Serving a Distributed LLM Serving Solution

Dynamo provides a simple way to spin up a local set of inference
components including:

- **OpenAI Compatible Frontend** – High performance OpenAI compatible http api server written in Rust.
- **Basic and Kv Aware Router** – Route and load balance traffic to a set of workers.
- **Workers** – Set of pre-configured LLM serving engines.

To run a minimal configuration you can use a pre-configured
example.

> [!NOTE]
> The following assumes docker and docker-compose are installed.

#### Start Dynamo Distributed Runtime Services

First start the Dynamo Distributed Runtime services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

#### Example Output

```bash
[+] Running 3/3
 ✔ Network deploy_default          Created   0.1s
 ✔ Container deploy-etcd-server-1  Started   0.7s
 ✔ Container deploy-nats-server-1  Started   0.7s
```


#### Start Dynamo LLM Serving Components

Next serve a minimal configuration with an http server, basic
round-robin router, and a single worker.


```bash
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

#### Example Output
```bash
Added new chat model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
+------------+------------------------------------------+-----------+-----------+------------------+
| MODEL TYPE | MODEL NAME                               | NAMESPACE | COMPONENT | ENDPOINT         |
+------------+------------------------------------------+-----------+-----------+------------------+
| chat       | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | dynamo    | Processor | chat/completions |
+------------+------------------------------------------+-----------+-----------+------------------+
2025-03-17T14:48:51.223378Z  INFO dynamo_llm::http::service::discovery: added Chat model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
2025-03-17T14:48:51.811831Z  INFO dynamo_runtime::pipeline::network::tcp::server: tcp transport service on 10.20.56.81:44999
2025-03-17T14:48:51.812385Z  INFO dynamo_llm::http::service::discovery: added Chat model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
2025-03-17T14:48:51.812451Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8000 address="0.0.0.0:8000"
...
```

#### Send a Request

```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

#### Example Output
```
{
  "id": "5a8b4199-d005-47e2-83ed-bd8a6ec64b43",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "Alright, the user said, \"Hello, how are you?\" and then mentioned \"How can I assist you today?\".\n\nI should respond in a friendly and open manner to encourage them to ask for help.\n\nMaybe something like, \"I'm doing well, thank you! How can I assist you today?\"\n\nThat should cover it and keep the conversation going.\n</think>\n\nI'm doing well, thank you! How can I assist you today?",
        "refusal": null,
        "tool_calls": null,
        "role": "assistant",
        "function_call": null,
        "audio": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "created": 1742234675,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "service_tier": null,
  "system_fingerprint": null,
  "object": "chat.completion",
  "usage": null
}
```

