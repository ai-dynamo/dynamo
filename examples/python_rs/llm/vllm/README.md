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

# vLLM Integration with Triton Distributed

This example demonstrates how to use Triton Distributed to serve large language models with the vLLM engine, enabling efficient model serving with both monolithic and disaggregated deployment options.

## Prerequisites

Start required services (etcd and NATS):

   Option A: Using [Docker Compose](/deploy/docker-compose.yml) (Recommended)
   ```bash
   docker compose -f ./deploy/docker-compose.yml up -d
   ```

   Option B: Manual Setup

    - [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
        - example: `nats-server -js --trace`
    - [etcd](https://etcd.io) server
        - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally


## Building the Environment

The example is designed to run in a containerized environment using Triton Distributed, vLLM, and associated dependencies. To build the container:

```bash
# Build image
./container/build.sh --framework VLLM
```

## Launching the Environment
```
# Run image interactively
./container/run.sh --framework VLLM -it
```

## Deployment

### 1. HTTP Server

Run the server logging (with debug level logging):
```bash
DYN_LOG=DEBUG http
```
By default the server will run on port 8080.

Add model to the server:
```bash
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynemo.vllm.generate
```

##### Example Output
```
+------------+------------------------------------------+-----------+-----------+----------+
| MODEL TYPE | MODEL NAME                               | NAMESPACE | COMPONENT | ENDPOINT |
+------------+------------------------------------------+-----------+-----------+----------+
| chat       | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | dynemo | vllm      | generate |
+------------+------------------------------------------+-----------+-----------+----------+
```

### 2. Workers

#### 2.1. Monolithic Deployment

In a separate terminal run the vllm worker:

```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Launch worker
cd /workspace/examples/python_rs/llm/vllm
python3 -m monolith.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager
```

##### Example Output

```
INFO 03-02 05:30:36 __init__.py:190] Automatically detected platform cuda.
WARNING 03-02 05:30:36 nixl.py:43] NIXL is not available

INFO 03-02 05:30:43 config.py:542] This model supports multiple tasks: {'embed', 'score', 'generate', 'classify', 'reward'}. Defaulting to 'generate'.
INFO 03-02 05:30:43 base_engine.py:43] Initializing engine client
INFO 03-02 05:30:43 api_server.py:206] Started engine process with PID 1151
INFO 03-02 05:30:44 config.py:542] This model supports multiple tasks: {'embed', 'score', 'generate', 'classify', 'reward'}. Defaulting to 'generate'.

<SNIP>

INFO 03-02 05:32:20 llm_engine.py:476] init engine (profile, create kv cache, warmup model) took 4.22 seconds

```

#### 2.2. Disaggregated Deployment

This deployment option splits the model serving across prefill and decode workers, enabling more efficient resource utilization.

**Terminal 1 - Prefill Worker:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Launch prefill worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"DynemoNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

##### Example Output

```
INFO 03-02 05:59:44 worker.py:269] the current vLLM instance can use total_gpu_memory (47.54GiB) x gpu_memory_utilization (0.40) = 19.01GiB
INFO 03-02 05:59:44 worker.py:269] model weights take 14.99GiB; non_torch_memory takes 0.06GiB; PyTorch activation peak memory takes 1.19GiB; the rest of the memory reserved for KV Cache is 2.78GiB.
INFO 03-02 05:59:44 executor_base.py:110] # CUDA blocks: 1423, # CPU blocks: 2048
INFO 03-02 05:59:44 executor_base.py:115] Maximum concurrency for 10 tokens per request: 2276.80x
INFO 03-02 05:59:47 llm_engine.py:476] init engine (profile, create kv cache, warmup model) took 3.41 seconds
```

**Terminal 2 - Decode Worker:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Launch decode worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1,2 python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"DynemoNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

The disaggregated deployment utilizes separate GPUs for prefill and decode operations, allowing for optimized resource allocation and improved performance. For more details on the disaggregated deployment, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill.html).

##### Example Output

```
INFO 03-02 05:59:44 worker.py:269] the current vLLM instance can use total_gpu_memory (47.54GiB) x gpu_memory_utilization (0.40) = 19.01GiB
INFO 03-02 05:59:44 worker.py:269] model weights take 14.99GiB; non_torch_memory takes 0.06GiB; PyTorch activation peak memory takes 1.19GiB; the rest of the memory reserved for KV Cache is 2.78GiB.
INFO 03-02 05:59:44 executor_base.py:110] # CUDA blocks: 1423, # CPU blocks: 2048
INFO 03-02 05:59:44 executor_base.py:115] Maximum concurrency for 10 tokens per request: 2276.80x
INFO 03-02 05:59:47 llm_engine.py:476] init engine (profile, create kv cache, warmup model) took 3.41 seconds
```


### 3. Client

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

##### Example Output

```json
{
    "id": "5b04e7b0-0dcd-4c45-baa0-1d03d924010c",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The capital of France is Paris. Paris is a major city known for iconic landmarks like the Eiffel Tower and the Louvre Museum."
        },
        "index": 0,
        "finish_reason": "stop"
    }],
    "created": 1739548787,
    "model": "vllm",
    "object": "chat.completion",
    "usage": null,
    "system_fingerprint": null
}
```

### 4. Multi-Node Deployment

The vLLM workers can be deployed across multiple nodes by configuring the NATS and etcd connection endpoints through environment variables. This enables distributed inference across a cluster.

Set the following environment variables on each node before running the workers:

```bash
export NATS_SERVER="nats://<nats-server-host>:<nats-server-port>"
export ETCD_ENDPOINTS="http://<etcd-server-host1>:<etcd-server-port>,http://<etcd-server-host2>:<etcd-server-port>",...
```

For disaggregated deployment, you will also need to pass the `kv_ip` and `kv_port` to the workers in the `kv_transfer_config` argument:

```bash
...
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":<rank>,"kv_parallel_size":2,"kv_ip":<master_node_ip>,"kv_port":<kv_port>}'
```


### 5. KV Router Deployment

The KV Router is a component that aggregates KV Events from all the workers and maintains a prefix tree of the cached tokens. It makes decisions on which worker to route requests to based on the length of the prefix match and the load on the workers.

You can run the router and workers in separate terminal sessions or use the `kv-router-run.sh` script to launch them all at once in their own tmux sessions.

#### Deploying using tmux

The helper script `kv-router-run.sh` will launch the router and workers in their own tmux sessions.
kv-router-run.sh <number_of_workers> <routing_strategy> Optional[<model_name>]

Example:
```bash
# Launch 8 workers with prefix routing strategy and use deepseek-ai/DeepSeek-R1-Distill-Llama-8B as the model
bash /workspace/examples/python_rs/llm/vllm/scripts/kv-router-run.sh 8 prefix deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# List tmux sessions
tmux ls

# Attach to the tmux sessions
tmux a -t v-1 # worker 1 - use cmd + b, d to detach
tmux a -t v-router # kv router - use cmd + b, d to detach

# Close the tmux sessions
tmux ls | grep 'v-' | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
```

#### Deploying using separate terminals

**Terminal 1 - Router:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Launch prefill worker
cd /workspace/examples/python_rs/llm/vllm
RUST_LOG=info python3 -m kv_router.router \
    --routing-strategy prefix \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --min-workers 1
```

You can choose only the prefix strategy for now:
- `prefix`: Route requests to the worker that has the longest prefix match.

**Terminal 2 - Processor:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Processor must take the same args as the worker
# This is temporary until we communicate the ModelDeploymentCard over etcd
cd /workspace/examples/python_rs/llm/vllm
RUST_LOG=info python3 -m kv_router.processor \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enable-prefix-caching \
    --block-size 64 \
    --max-model-len 16384
```

**Terminal 3 and 4 - Workers:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate

# Launch Worker 1 and Worker 2 with same command
cd /workspace/examples/python_rs/llm/vllm
RUST_LOG=info python3 -m kv_router.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enable-prefix-caching \
    --block-size 64 \
    --max-model-len 16384
```

Note: Must enable prefix caching for KV Router to work
Note: block-size must be 64, otherwise Router won't work (accepts only 64 tokens)

**Terminal 5 - Client:**
Don't forget to add the model to the server:
```bash
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynemo.process.chat/completions
```

```bash
curl localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
##### Example Output

```json
{
    "id": "f435d1aa-d423-40a0-a616-00bc428a3e32",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Alright, the user is playing a character in a D&D setting. They want a detailed background for their character, set in the world of Eldoria, particularly in the city of Aeloria. The user mentioned it's about an intrepid explorer"
            },
            "index": 0,
            "finish_reason": "length"
        }
    ],
    "created": 1740020570,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "object": "chat.completion",
    "usage": null,
    "system_fingerprint": null
}
```
### 6. Preprocessor and backend

This deployment splits the pre-processing and backend for model serving.

Run following commands in 4 terminals:

**Terminal 1 - vLLM Worker:**
```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm

RUST_LOG=info python3 -m preprocessor.worker --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Terminal 2 - preprocessor:**

```bash
# Activate virtual environment
source /opt/dynemo/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm

RUST_LOG=info python3 -m preprocessor.processor --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Terminal 3 - HTTP Server**

Run the server logging (with debug level logging):
```bash
DYN_LOG=DEBUG http
```
By default the server will run on port 8080.

Add model to the server:
```bash
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B dynemo.preprocessor.generate
```

**Terminal 4 - client**

```bash

curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### 7. Known Issues and Limitations

- vLLM is not working well with the `fork` method for multiprocessing and TP > 1. This is a known issue and a workaround is to use the `spawn` method instead. See [vLLM issue](https://github.com/vllm-project/vllm/issues/6152).
- `kv_rank` of `kv_producer` must be smaller than of `kv_consumer`.
- Instances with the same `kv_role` must have the same `--tensor-parallel-size`.
- Currently only `--pipeline-parallel-size 1` is supported for XpYd disaggregated deployment.
