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

# Running KVBM in vLLM

This guide explains how to leverage KVBM (KV Block Manager) to mange KV cache and do KV offloading in vLLM.

To learn what KVBM is, please check [here](https://docs.nvidia.com/dynamo/latest/architecture/kvbm_intro.html)

## Quick Start

To use KVBM in vLLM, you can follow the steps below:

```bash
# start up etcd for KVBM leader/worker registration and discovery
docker compose -f deploy/docker-compose.yml up -d

# build a container containing vllm and kvbm
./container/build.sh --framework vllm --enable-kvbm

# launch the container
./container/run.sh --framework vllm -it --mount-workspace --use-nixl-gds

# enable kv offloading to CPU memory
# 4 means 4GB of CPU memory would be used
export DYN_KVBM_CPU_CACHE_GB=4

# enable kv offloading to disk
# 8 means 8GB of disk would be used
export DYN_KVBM_DISK_CACHE_GB=8

# [DYNAMO] start dynamo frontend
python -m dynamo.frontend --http-port 8000 &

# [DYNAMO] serve an LLM model using KVBM with dynamo
python -m dynamo.vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --connector kvbm &

# make a call to LLM
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

Alternatively, can use "vllm serve" with KVBM by replacing the above two [DYNAMO] cmds with below:
```bash
vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "dynamo.llm.vllm_integration.connector"}' deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## Enable and View KVBM Metrics

Follow below steps to enable metrics collection and view via Grafana dashboard:
```bash
# Start the basic services (etcd & natsd), along with Prometheus and Grafana
docker compose -f deploy/docker-compose.yml --profile metrics up -d

# start vllm with DYN_SYSTEM_ENABLED set to true and DYN_SYSTEM_PORT port to 6880.
# NOTE: Make sure port 6880 (for KVBM worker metrics) and port 6881 (for KVBM leader metrics) are available.
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=6880 vllm serve --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "dynamo.llm.vllm_integration.connector"}' deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# optional if firewall blocks KVBM metrics ports to send prometheus metrics
sudo ufw allow 6880/tcp
sudo ufw allow 6881/tcp
```

View grafana metrics via http://localhost:3001 (default login: dynamo/dynamo) and look for KVBM Dashboard

## Benchmark KVBM

Once vllm serve is ready, follow below steps to use LMBenchmark to benchmark KVBM performance:
```bash
git clone https://github.com/LMCache/LMBenchmark.git

# show case of running the synthetic multi-turn chat dataset.
# we are passing model, endpoint, output file prefix and qps to the sh script.
cd LMBenchmark/synthetic-multi-round-qa
./long_input_short_output_run.sh \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    "http://localhost:8000" \
    "benchmark_kvbm" \
    1

# Average TTFT and other perf numbers would be in the output from above cmd
```
More details about how to use LMBenchmark could be found [here](https://github.com/LMCache/LMBenchmark).

`NOTE`: if metrics are enabled as mentioned in the above section, you can observe KV offloading, and KV onboarding in the grafana dashboard.

To compare, you can run `vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B` to turn KVBM off as the baseline.
