---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Running Deepseek R1 with Wide EP

Dynamo supports running Deepseek R1 with data parallel attention and wide expert parallelism. Each data parallel attention rank is a separate dynamo component that will emit its own KV Events and Metrics. vLLM controls the expert parallelism using the flag `--enable-expert-parallel`

## Instructions

The following script can be adapted to run Deepseek R1 with a variety of different configuration. The current configuration uses 2 nodes, 16 GPUs, and a dp of 16. Follow the [vLLM Backend](README.md) Getting Started section on each node, and then run these two commands.

node 0
```bash
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 0 --gpus-per-node 8 --master-addr <node 0 addr>
```

node 1
```bash
./launch/dsr1_dep.sh --num-nodes 2 --node-rank 1 --gpus-per-node 8 --master-addr <node 0 addr>
```

### Testing the Deployment

On node 0 (where the frontend was started) send a test request to verify your deployment:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```
