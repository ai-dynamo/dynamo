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

> **NOTE**: This example is based on an internal NVIDIA library that will soon be publicly released. The example won't work until the official release.

## Prerequisites

Start required services (etcd and NATS):

   Option A: Using [Docker Compose](/runtime/rust/docker-compose.yml) (Recommended)
   ```bash
   docker-compose up -d
   ```

   Option B: Manual Setup

    - [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
        - example: `nats-server -js --trace`
    - [etcd](https://etcd.io) server
        - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally

## Build docker

```
./container/build.sh --framework VLLM_NIXL --target dev --build-context nixl=<path to downloaded nixl repo @ c53bb19a6a114e9093071bd1f2904f996ae1839b>
```

## Run container

```
./container/run.sh --framework VLLM_NIXL --target dev -it
```

All of the commands below are run inside the same container.

## Run deployment

Add model to triton and start http server.


```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B triton-init.process.chat/completions
TRT_LOG=DEBUG http --port 8181
```

To launch disaggregated vllm deployment, there are three major components:
1. Processor
2. KV Router
3. Disaggregated Router
4. Prefill and Decode Workers

### Processor

```
# Processor must take the same args as the worker
# This is temporary until we communicate the ModelDeploymentCard over etcd
cd /workspace/examples/python_rs/llm/vllm
RUST_LOG=info python3 processor.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enable-prefix-caching \
    --block-size 64 \
    --max-model-len 16384
```

### KV Router

The KV Router is a component that aggregates KV Events from all the workers and maintains a prefix tree of the cached tokens. It makes decisions on which worker to route requests to based on the length of the prefix match and the load on the workers.

To launch the KV Router, run the following command:
```
RUST_LOG=info python3 router.py \
    --routing-strategy prefix \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --min-workers 1
```

You can choose only the prefix strategy for now:
- `prefix`: Route requests to the worker that has the longest prefix match.


### Disaggregated Router

The disaggregated router determines whether a request should be send to a
remote prefill engine or a local prefill engine for prefilling based on the
prefill length. When prefilling locally, the vllm scheduler will prioritize
prefill request and pause any ongoing decode requests.
Each decoder worker will launch its own prefill worker.
For now, we use a simple heuristic to route to prefill engine
if prefill length (including prefix catch hit) is greater than a threshold.
This threshold can by dynamically adjusted at runtime through etcd.

To check the current threshold (this will print out all kv pairs in etcd):
```
curl -s -L http://localhost:2379/v3/kv/range -X POST   -d '{"key":"AA==", "range_end":"AA=="}' |   jq -r '.kvs[] | "KEY: \(.key | @base64d)\nVALUE: \(.value | @base64d)\n---"'
```

To update the threshold:
```
ETCDCTL_API=3 etcdctl --endpoints=http://localhost:2379 put 'public/components/disagg_router/models/chat/deepseek-ai/DeepSeek-R1-Distill-Llama-8B' '{"max_local_prefill_length": <new_threshold>}'
```

### Workers

```
# start prefill worker in Terminal 1
# for xPyD, please first start the decode workers
# so that prefill workers can get metadata of all decode workers
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=0 python prefill_worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"TritonNixlConnector"}'

# start decode worker in Terminal 2
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=1,2 python3 worker.py \
    --remote-prefill \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"TritonNixlConnector"}'
```

Alternatively, we also provide a script to launch all workers in one go:
```
./start_single_node.sh
# Usage [--model <model>] [--p_tensor_parallel_size <size>] [--d_tensor_parallel_size <size>] [--max_model_len <len>] [--max_num_batched_tokens <tokens>] [--max_num_seqs <seqs>] [--gpu_memory_utilization <utilization>] [--enable_chunked_prefill <True/False>] [--num_p <p>] [--num_d <d>]
```

If torch GLOO backend is complaining about file name too long, set
```
export GLOO_SOCKET_IFNAME=lo
```

## Client

In another terminal:
```
# around 200 tokens isl
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

## Run genai-perf

`genai-perf` is a tool for profiling and benchmarking LLM servers. It is already installed in the container. For more details, please refer to the [genai-perf README](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html).

```
genai-perf profile \
  -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url localhost:8181 \
  --endpoint-type chat \
  --streaming \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 150 \
  --extra-inputs min_tokens:150 \
  --extra-inputs max_tokens:150 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency 10 \
  --request-count 40 \
  -- -v \
  --async
```

## Close deployment

Kill all python processes and clean up metadata files:

```
pkill -9 -f python
rm -r /tmp/nixl
```

## TODOs, limitations, known issues

- [ ] Add etcd for discovery
- [ ] Multi-node deployment support
- [ ] Enable chunked prefill
- [ ] Support mixed tp
- [ ] Process many remote prefill in one iteration
- [ ] Support recompute preemption
- [ ] Make sure decode does not preempt blocks before xfer finishes
- [ ] Layer wise transfer
- [ ] Non blocking send in prefill (cache manager should check xfer status)
- [ ] Test under load
- [ ] Support pp > 1
- [ ] Check why adding extra seed input is crashing vllm with remote prefill
- [ ] Unified worker for both prefill and decode
- [x] Require sending two parallel requests to start decode for the first time
- [x] Concurrency > 2 is not working
- [x] Parse cmdline args
- [x] Manual nixl example with tp1
- [x] Zero copy
- [x] Conditional remote prefill
- [x] Manual example with tp > 1
- [x] Run on triton distributed runtime
- [x] add oai http endpoint
- [x] Sample only on decode, do note return remote prefill response
- [x] Check if all transfers finished before moving to decode
- [x] Enable async output processing - could be working
