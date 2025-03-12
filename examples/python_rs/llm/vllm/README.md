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

All of the commands below are run inside the same container.

## Run deployment

This figure shows an overview of the major components to deploy:

```
                                                 +----------------+
                                          +------| prefill worker |-------+
                                   notify |      |   (optional)   |       |
                                 finished |      +----------------+       | pull
                                          v                               v
+------+      +-----------+      +------------------+    push     +---------------+
| HTTP |----->| processor |----->| decode/monolith  |------------>| prefill queue |
|      |<-----|           |<-----|      worker      | (if disagg) |   (optional)  |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|    (optional)    |
                                 +------------------+

```

Add model to dynamo and start http server.
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynamo-init.process.chat/completions
TRT_LOG=DEBUG http --port 8181
```

### Processor

Processor routes the requests to the (decode) workers. Three scheduling strategies are supported: 1. random, 2. round-robin, 3. kv (see [Kv Router](#kv-router)).

```
# Processor must take the same args as the (decoder) worker
# This is temporary until we communicate the ModelDeploymentCard over etcd
RUST_LOG=info python3 processor.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --block-size 64 \
    --max-model-len 16384 \
    --router <random/round-robin/kv>
```

Alternatively, the processor can be bypassed by directly hitting the worker endpoints:
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynamo-init.vllm.generate

# monolithic
CUDA_VISIBLE_DEVICES=0 python3 routerless/worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager

# disaggregated
CUDA_VISIBLE_DEVICES=0 python routerless/prefill_worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}'
CUDA_VISIBLE_DEVICES=1 python3 routerless/worker.py \
    --remote-prefill \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}'
```

### Kv Router

The KV Router is a component that aggregates KV Events from all the workers and maintains
a prefix tree of the cached tokens. It makes decisions on which worker to route requests
to based on the length of the prefix match and the load on the workers.
There are three steps needed to enable the kv router:
1. Use `--router kv` in the processor.
2. Use `--router kv` and `--enable-prefix-caching` in all the (decode) workers.
3. Launch the kv router in a separate terminal.
   ```
   RUST_LOG=info python3 kv_router.py \
       --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
       --block-size 64 \
       --min-workers 1
   ```
   where `--min-workers` is the number of (decode) workers.

You can choose only the prefix strategy for now:
- `prefix`: Route requests to the worker that has the longest prefix match.

### Disaggregated Router

The disaggregated router determines whether a request should be send to a
remote prefill engine or a local prefill engine for prefilling based on the
prefill length. If kv router is enabled, the disaggregated router will use
the absolute prefill length (actual prefill length - prefix hit length) to make
the decision.

When prefilling locally, the vllm scheduler will prioritize
prefill request and pause any ongoing decode requests.

To enable the disaggregated router, add the following commands in the decode workers:
```
python worker.py \
...
--conditional-disagg \
--max-local-prefill-length <length>
```

### Worker

#### Monolithic

Only kv router is supported for monolithic deployment.

```
CUDA_VISIBLE_DEVICES=0 python3 worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --block-size 64 \
    --max-model-len 16384 \
    <optional kv router args: --router kv --enable-prefix-caching>
```

#### Disaggregated

Kv router and disaggregated router are supported and can be turned on/off individually.

```
# start prefill worker in one terminal
# Note: prefix caching is not supported in the prefill for now
CUDA_VISIBLE_DEVICES=0 python3 prefill_worker.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' \
    --block-size 64 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384

# start decode worker in another terminal
CUDA_VISIBLE_DEVICES=1 python3 worker.py \
    --remote-prefill \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' \
    --block-size 64 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    <optional kv router args: --router kv --enable-prefix-caching>
    <optional disaggregated router args: --conditional-disagg --max-local-prefill-length <length>>
```

### Multi-Node Deployment

For multi-node deployment, etcd, nats, processor, and kv router
are only required on the head node. The only components that need
to be deployed on all nodes are the workers.

Set the following environment variables on each node before running the workers:
```bash
export NATS_SERVER="nats://<nats-server-host>:<nats-server-port>"
export ETCD_ENDPOINTS="http://<etcd-server-host>:<etcd-server-port>"
```


### Common Issues

If torch GLOO backend is complaining about file name too long, set
```
export GLOO_SOCKET_IFNAME=lo
```

## Client

In another terminal:
```
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
```

## TODOs, limitations, known issues

- [ ] Add etcd for discovery
- [ ] Multi-node deployment support
- [ ] Enable chunked prefill
- [ ] Process many remote prefill in one iteration
- [ ] Support recompute preemption
- [ ] Make sure decode does not preempt blocks before xfer finishes
- [ ] Layer wise transfer
- [ ] Non blocking send in prefill (cache manager should check xfer status)
- [ ] Test under load
- [ ] Support pp > 1
- [ ] Check why adding extra seed input is crashing vllm with remote prefill
- [ ] Unified worker for both prefill and decode
- [x] Support mixed tp
- [x] Require sending two parallel requests to start decode for the first time
- [x] Concurrency > 2 is not working
- [x] Parse cmdline args
- [x] Manual nixl example with tp1
- [x] Zero copy
- [x] Conditional remote prefill
- [x] Manual example with tp > 1
- [x] Run on dynamo distributed runtime
- [x] add oai http endpoint
- [x] Sample only on decode, do note return remote prefill response
- [x] Check if all transfers finished before moving to decode
- [x] Enable async output processing - could be working
