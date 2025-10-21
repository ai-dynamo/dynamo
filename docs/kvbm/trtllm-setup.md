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

# Running KVBM in TensorRT-LLM

This guide explains how to leverage KVBM (KV Block Manager) to manage KV cache and do KV offloading in TensorRT-LLM (trtllm).

To learn what KVBM is, please check [here](kvbm_architecture.md)

> [!Note]
> - Ensure that `etcd` and `nats` are running before starting.
> - KVBM does not currently support CUDA graphs in TensorRT-LLM.
> - KVBM only supports TensorRT-LLM’s PyTorch backend.
> - Disable partial reuse `enable_partial_reuse: false` in the LLM API config’s `kv_connector_config` to increase offloading cache hits.
> - KVBM requires TensorRT-LLM v1.1.0rc5 or newer.
> - Enabling KVBM metrics with TensorRT-LLM is still a work in progress.

## Quick Start

To use KVBM in TensorRT-LLM, you can follow the steps below:

```bash
# start up etcd for KVBM leader/worker registration and discovery
docker compose -f deploy/docker-compose.yml up -d

# Build a container that includes TensorRT-LLM and KVBM.
./container/build.sh --framework trtllm --enable-kvbm

# launch the container
./container/run.sh --framework trtllm -it --mount-workspace --use-nixl-gds

# Configure KVBM cache tiers (choose one of the following options):

# Option 1: CPU cache only (GPU -> CPU offloading)
# 4 means 4GB of pinned CPU memory would be used
export DYN_KVBM_CPU_CACHE_GB=4

# Option 2: Both CPU and Disk cache (GPU -> CPU -> Disk tiered offloading)
export DYN_KVBM_CPU_CACHE_GB=4
# 8 means 8GB of disk would be used
export DYN_KVBM_DISK_CACHE_GB=8

# [Experimental] Option 3: Disk cache only (GPU -> Disk direct offloading, bypassing CPU)
# NOTE: this option is only experimental and it might give out the best performance.
# NOTE: disk offload filtering is not support when using this option.
export DYN_KVBM_DISK_CACHE_GB=8

# Note: You can also use DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS or
# DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS to specify exact block counts instead of GB

# Allocating memory and disk storage can take some time.
# We recommend setting a higher timeout for leader–worker initialization.
# 1200 means 1200 seconds timeout
export DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS=1200

# enable disk zerofill fallback for KVBM
# set to true to enable fallback behavior when disk operations fail
export DYN_KVBM_DISK_ZEROFILL_FALLBACK=true
```

> [!NOTE]
> When disk offloading is enabled, to extend SSD lifespan, disk offload filtering would be enabled by default. The current policy is only offloading KV blocks from CPU to disk if the blocks have frequency equal or more than `2`. Frequency is determined via doubling on cache hit (init with 1) and decrement by 1 on each time decay step.
>
> To disable disk offload filtering, set `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` to true or 1.

```bash
# write an example LLM API config
# Note: Disable partial reuse "enable_partial_reuse: false" in the LLM API config’s "kv_connector_config" to increase offloading cache hits.
cat > "/tmp/kvbm_llm_api_config.yaml" <<EOF
backend: pytorch
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
kv_connector_config:
  connector_module: dynamo.llm.trtllm_integration.connector
  connector_scheduler_class: DynamoKVBMConnectorLeader
  connector_worker_class: DynamoKVBMConnectorWorker
EOF

# [DYNAMO] start dynamo frontend
python3 -m dynamo.frontend --http-port 8000 &

# [DYNAMO] To serve an LLM model with dynamo
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml &

# make a call to LLM
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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

Alternatively, can use "trtllm-serve" with KVBM by replacing the above two [DYNAMO] cmds with below:
```bash
trtllm-serve Qwen/Qwen3-0.6B --host localhost --port 8000 --backend pytorch --extra_llm_api_options /tmp/kvbm_llm_api_config.yaml
```

## Enable and View KVBM Metrics

Follow below steps to enable metrics collection and view via Grafana dashboard:
```bash
# Start the basic services (etcd & natsd), along with Prometheus and Grafana
docker compose -f deploy/docker-compose.yml --profile metrics up -d

# set env var DYN_KVBM_METRICS to true, when launch via dynamo
# Optionally set DYN_KVBM_METRICS_PORT to choose the /metrics port (default: 6880).
DYN_KVBM_METRICS=true \
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml &

# optional if firewall blocks KVBM metrics ports to send prometheus metrics
sudo ufw allow 6880/tcp
```

View grafana metrics via http://localhost:3001 (default login: dynamo/dynamo) and look for KVBM Dashboard

## Benchmark KVBM

Once the model is loaded ready, follow below steps to use LMBenchmark to benchmark KVBM performance:
```bash
git clone https://github.com/LMCache/LMBenchmark.git

# show case of running the synthetic multi-turn chat dataset.
# we are passing model, endpoint, output file prefix and qps to the sh script.
cd LMBenchmark/synthetic-multi-round-qa
./long_input_short_output_run.sh \
    "Qwen/Qwen3-0.6B" \
    "http://localhost:8000" \
    "benchmark_kvbm" \
    1

# Average TTFT and other perf numbers would be in the output from above cmd
```
More details about how to use LMBenchmark could be found [here](https://github.com/LMCache/LMBenchmark).

`NOTE`: if metrics are enabled as mentioned in the above section, you can observe KV offloading, and KV onboarding in the grafana dashboard.

To compare, you can remove the `kv_connector_config` section from the LLM API config and run `trtllm-serve` with the updated config as the baseline.
```bash
cat > "/tmp/llm_api_config.yaml" <<EOF
backend: pytorch
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.80
EOF

# run trtllm-serve for the baseline for comparison
trtllm-serve Qwen/Qwen3-0.6B --host localhost --port 8000 --backend pytorch --extra_llm_api_options /tmp/llm_api_config.yaml &
```
