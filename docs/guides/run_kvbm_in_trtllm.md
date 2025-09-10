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

This guide explains how to leverage KVBM (KV Block Manager) to mange KV cache and do KV offloading in TensorRT-LLM (trtllm).

To learn what KVBM is, please check [here](https://docs.nvidia.com/dynamo/latest/architecture/kvbm_intro.html)

> [!Note]
> - Ensure that `etcd` and `nats` are running before starting.
> - KVBM does not currently support CUDA graphs in TensorRT-LLM.
> - KVBM only supports TensorRT-LLM’s PyTorch backend.
> - To enable disk cache offloading, you must first enable a CPU memory cache offloading.
> - Disable partial reuse `enable_partial_reuse: false` in the LLM API config’s `kv_connector_config` to increase offloading cache hits.
> - KVBM requires TensorRT-LLM at commit ce580ce4f52af3ad0043a800b3f9469e1f1109f6 or newer.
> - Enabling KVBM metrics with TensorRT-LLM is still a work in progress.

## Quick Start

To use KVBM in TensorRT-LLM, you can follow the steps below:

```bash
# start up etcd for KVBM leader/worker registration and discovery
docker compose -f deploy/docker-compose.yml up -d

# Build a container that includes TensorRT-LLM and KVBM. Note: KVBM integration is only available in TensorRT-LLM commit dcd110cfac07e577ce01343c455917832b0f3d5e or newer.
# When building with the --tensorrtllm-commit option, you may notice that https://github.com keeps prompting for a username and password.
# This happens because cloning TensorRT-LLM can hit GitHub’s rate limit.
# To work around this, you can keep pressing "Enter" or "Return.".
# Setting "export GIT_LFS_SKIP_SMUDGE=1" may also reduce the number of prompts.
./container/build.sh --framework trtllm --tensorrtllm-commit dcd110cfac07e577ce01343c455917832b0f3d5e --enable-kvbm

# launch the container
./container/run.sh --framework trtllm -it --mount-workspace --use-nixl-gds

# enable kv offloading to CPU memory
# 60 means 60GB of pinned CPU memory would be used
export DYN_KVBM_CPU_CACHE_GB=60

# enable kv offloading to disk. Note: To enable disk cache offloading, you must first enable a CPU memory cache offloading.
# 20 means 20GB of disk would be used
export DYN_KVBM_DISK_CACHE_GB=20

# Allocating memory and disk storage can take some time.
# We recommend setting a higher timeout for leader–worker initialization.
# 1200 means 1200 seconds timeout
export DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS=1200
```

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

# start dynamo frontend
python3 -m dynamo.frontend --http-port 8000 &

# To serve an LLM model with dynamo
python3 -m dynamo.trtllm \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --extra-engine-args /tmp/kvbm_llm_api_config.yaml &

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

# Optionally, we could also serve an LLM with trtllm-serve to utilize the KVBM feature.
trtllm-serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --host localhost --port 8001 --backend pytorch --extra_llm_api_options /tmp/kvbm_llm_api_config.yaml

```
