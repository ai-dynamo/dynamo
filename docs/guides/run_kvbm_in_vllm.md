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
# build a container containing vllm and kvbm
./container/build.sh --framework kvbm

# launch the container
./container/run.sh --framework kvbm -it --mount-workspace --use-nixl-gds

# enable using kvbm instead of vllm's own kv cache manager
export DYNAMO_KVBM_MANAGER=kvbm

# enable kv offloading to CPU memory
# 4 means 4GP of CPU memory would be used
export DYNAMO_KVBM_CPU_CACHE=4

# enable kv offloading to disk
# 8 means *GP of disk would be used
export DYNAMO_KVBM_DISK_CACHE=8

# serve an example LLM model
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B

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