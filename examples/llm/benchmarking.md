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

# LLM Deployment Benchmarking Guide

This guide provides detailed steps on benchmarking Large Language Models (LLMs) in single and multi nodes configurations.

> [!NOTE]
> We advice trying out the [LLM Deployment Examples](./README.md) before benchmarking.

## Build the Benchmarking Image

1\. Clone Dynamo repository
```bash
git clone --single-branch --depth=1 -b main https://github.com/ai-dynamo/dynamo.git
```

2\. Build vLLM image
```bash
./container/build.sh
```
Note: Make sure to run it in the `dynamo` directory cloned.

## Disaggregated Single Node Benchmarking

One H100 80GB x8 node is required for this setup.

With the Dynamo repository and the benchmarking image available, perform the following steps:

1\. Run benchmarking container
```bash
./container/run.sh -it
```
Note: Make sure to run it in the `dynamo` directory cloned.

2\. Download model
```bash
huggingface-cli download neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
```

3\. Start NATS and ETCD
```bash
nats-server -js -p 4222 -m 8222 &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 &
```

4\. Update `/workspace/examples/llm/configs/disagg.yaml` to the following
```yml
Frontend:
  served_model_name: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  endpoint: dynamo.Processor.chat/completions
  port: 8000

Processor:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  router: round-robin

VllmWorker:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  block-size: 128
  max-model-len: 16384
  remote-prefill: true
  conditional-disagg: true
  max-local-prefill-length: 10
  max-prefill-queue-size: 2
  tensor-parallel-size: 4
  ServiceArgs:
    workers: 1
    resources:
      gpu: 4

PrefillWorker:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  block-size: 128
  max-model-len: 4096
  max-num-batched-tokens: 4096
  tensor-parallel-size: 1
  ServiceArgs:
    workers: 4
    resources:
      gpu: 1
```
Key settings:
* **PrefillWorker**: Four separate processes (each with 1 GPU) handle the initial prefill (context embedding) phase.
* **VllmWorker**: One process (using 4 GPUs) generates output tokens (the "decode" phase).
* **kv-transfer-config**: Uses the DynamoNixlConnector for remote KV cache transfer, so that the KV cache can be passed between prefill workers and the generation worker.
* **FP8 Model**: Reduces memory usage for both the model and KV cache, easing remote transfer.
* **Block Size 128**: Batches token processing for more efficient chunk transfers to GPUs.

5\. Start disaggregated service
```bash
cd /workspace/examples/llm
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml 1> disagg.log 2>&1 &
```
Note: Check the `disagg.log` to make sure the service is fully started before collecting performance numbers.

6\. Create and run the benchmarking script
```bash
#/bin/bash

model=neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic

isl=3000
osl=150

for concurrency in 1 2 4 8 16 32 64 128 256; do

  genai-perf profile \
    --model ${model} \
    --tokenizer ${model} \
    --service-kind openai \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url http://localhost:8000 \
    --synthetic-input-tokens-mean ${isl} \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean ${osl} \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:${osl} \
    --extra-inputs min_tokens:${osl} \
    --extra-inputs ignore_eos:true \
    --concurrency ${concurrency} \
    --request-count $(($concurrency*10)) \
    --warmup-request-count $(($concurrency*2)) \
    --num-dataset-entries $(($concurrency*12)) \
    --random-seed 100 \
    -- \
    -v \
    --max-threads 256 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'

done
```
Key settings:
* **streaming**: Enables token streaming for more realistic performance measurements.
* **concurrency**: Number of simultaneous requests.
* **isl**: Input sequence length.
* **osl**: Output sequence length requested.

## Disaggregated Multi Node Benchmarking

Two H100 80GB x8 node is required for this setup.

With the Dynamo repository and the benchmarking image available, perform the following steps:

1\. Run benchmarking container (node 0 & 1)
```bash
./container/run.sh -it
```
Note: Make sure to run it in the `dynamo` directory cloned.

2\. Download model (node 0 & 1)
```bash
huggingface-cli download neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
```

3\. Start NATS and ETCD (node 0)
```bash
nats-server -js -p 4222 -m 8222 &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 &
```

4\. Config NATS and ETCD (node 1)
```bash
export NATS_SERVER="nats://<node_0_ip_addr>"
export ETCD_ENDPOINTS="<node_0_ip_addr>:2379"
```
Note: Node 1 must be able to reach Node 0 over the network for the above services.

5\. Update `/workspace/examples/llm/configs/disagg_router.yaml` to the following (node 0 & 1)
```yml
Frontend:
  served_model_name: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  endpoint: dynamo.Processor.chat/completions
  port: 8000

Processor:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  block-size: 128
  max-model-len: 16384
  router: kv

Router:
  model-name: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  min-workers: 1

VllmWorker:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  block-size: 128
  max-model-len: 16384
  max-num-batched-tokens: 16384
  remote-prefill: true
  conditional-disagg: true
  max-local-prefill-length: 10
  max-prefill-queue-size: 2
  tensor-parallel-size: 8
  router: kv
  enable-prefix-caching: true
  ServiceArgs:
    workers: 1
    resources:
      gpu: 8

PrefillWorker:
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  block-size: 128
  max-model-len: 4096
  max-num-batched-tokens: 4096
  tensor-parallel-size: 1
  ServiceArgs:
    workers: 8
    resources:
      gpu: 1

```

6\. Start frontend, processor and decode worker (node 0)
```bash
cd /workspace/examples/llm
sed -i "s/.link(PrefillWorker)//" graphs/disagg_router.py
dynamo serve graphs.disagg_router:Frontend -f configs/disagg_router.yaml 1> disagg_router.log 2>&1 &
```
Note: Check the `disagg_router.log` to make sure the service is fully started before collecting performance numbers.

7\. Start prefill workers (node 1)
```bash
cd /workspace/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f configs/disagg_router.yaml 1> prefill_worker.log 2>&1 &
```
Note: Check the `prefill_worker.log` to make sure the service is fully started before collecting performance numbers.

8. Collect the performance numbers as shown on the [Disaggregated Single Node Benchmarking](#disaggregated-single-node-benchmarking) section above.

## vLLM Aggregated Baseline Benchmarking

One (or Two) H100 80GB x8 node is required for this setup.

With the Dynamo repository and the benchmarking image available, perform the following steps:

1\. Run benchmarking container
```bash
./container/run.sh -it
```
Note: Make sure to run it in the `dynamo` directory cloned.

2\. Download model
```bash
huggingface-cli download neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
```

3\. Start vLLM serve
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --tensor-parallel-size 4 --port 8001 1> vllm_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --tensor-parallel-size 4 --port 8002 1> vllm_1.log 2>&1 &
```
Notes:
* Check the `vllm_0.log` and `vllm_1.log` to make sure the service is fully started before collecting performance numbers.
* `vllm serve` may also be started similarly on the second node for collecting multi node numbers.

4\. Install NGINX
```bash
apt update && apt install -y nginx
```

5\. Update `/etc/nginx/nginx.conf` to the following
```conf
worker_processes 1;
worker_rlimit_nofile 4096;
events {
    worker_connections 2048;
    multi_accept on;
    use epoll;
}
http {
    upstream backend_servers {
        least_conn;
        server 127.0.0.1:8001 max_fails=3 fail_timeout=10000s;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=10000s;
    }
    server {
        listen 8000;
        location / {
            proxy_pass http://backend_servers;
            proxy_http_version 1.1;
            proxy_read_timeout 240s;
        }
    }
}
```
Key settings:
* **least_conn**: To load balance across vLLM servers.
* **server**: Select all the upstream vLLM servers to load balance across, including those at a different node, if any.

6\. Restart NGINX
```bash
service nginx restart
```

7\. Collect the performance numbers as shown on the [Disaggregated Single Node Benchmarking](#disaggregated-single-node-benchmarking) section above.
