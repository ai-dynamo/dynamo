<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multinode Examples

## Multi-node sized models

SGLang allows you to deploy multi-node sized models by adding in the `dist-init-addr`, `nnodes`, and `node-rank` arguments. Below we demonstrate and example of deploying DeepSeek R1 for disaggregated serving across 4 nodes. This example requires 4 nodes of 8xH100 GPUs.

**Prerequisite**: Building the Dynamo container.

```bash
cd $DYNAMO_ROOT
docker build -f container/Dockerfile.sglang-wideep . -t dynamo-wideep --no-cache
```

You can use a specific tag from the [lmsys dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags) by adding `--build-arg SGLANG_IMAGE_TAG=<tag>` to the build command.

**Step 1**: Ensure that your configuration file has the required arguments. Here's an example configuration that runs prefill and the model in TP16:

Node 1: Run HTTP ingress, processor, and 8 shards of the prefill worker
```bash
# run ingress
python3 -m dynamo.frontend --http-port=8000 &
# run prefill worker
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dp-size 16 \
  --dist-init-addr ${HEAD_PREFILL_NODE_IP}:29500 \
  --nnodes 2 \
  --node-rank 0 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --load-balance-method round_robin \
  --host 0.0.0.0 \
  --mem-fraction-static 0.82
```

Node 2: Run the remaining 8 shards of the prefill worker
```bash
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dp-size 16 \
  --dist-init-addr ${HEAD_PREFILL_NODE_IP}:29500 \
  --nnodes 2 \
  --node-rank 1 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --host 0.0.0.0 \
  --load-balance-method round_robin \
  --mem-fraction-static 0.82
```

Node 3: Run the first 8 shards of the decode worker
```bash
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dp-size 16 \
  --dist-init-addr ${HEAD_DECODE_NODE_IP}:29500 \
  --nnodes 2 \
  --node-rank 0 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --host 0.0.0.0 \
  --prefill-round-robin-balance \
  --mem-fraction-static 0.82 \
  --cuda-graph-max-bs 8
```

Node 4: Run the remaining 8 shards of the decode worker
```bash
python3 -m dynamo.sglang \
  --model-path /model/ \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dp-size 16 \
  --dist-init-addr ${HEAD_DECODE_NODE_IP}:29500 \
  --nnodes 2 \
  --node-rank 1 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 30001 \
  --host 0.0.0.0 \
  --prefill-round-robin-balance \
  --mem-fraction-static 0.82 \
  --cuda-graph-max-bs 8
```

**Step 2**: Run inference
SGLang typically requires a warmup period to ensure the DeepGEMM kernels are loaded. We recommend running a few warmup requests and ensuring that the DeepGEMM kernels load in.

```bash
curl ${HEAD_PREFILL_NODE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of the tennis world, where champions rise and fall with each Grand Slam, lies the legend of the Golden Racket of Wimbledon. Once wielded by the greatest players of antiquity, this mythical racket is said to bestow unparalleled precision, grace, and longevity upon its rightful owner. For centuries, it remained hidden, its location lost to all but the most dedicated scholars of the sport. You are Roger Federer, the Swiss maestro whose elegant play and sportsmanship have already cemented your place among the legends, but whose quest for perfection remains unquenched even as time marches on. Recent dreams have brought you visions of this ancient artifact, along with fragments of a map that seems to lead to its resting place. Your journey will take you through the hallowed grounds of tennis history, from the clay courts of Roland Garros to the hidden training grounds of forgotten champions, and finally to a secret chamber beneath Centre Court itself. Character Background: Develop a detailed background for Roger Federer in this quest. Describe his motivations for seeking the Golden Racket, his tennis skills and personal weaknesses, and any connections to the legends of the sport that came before him. Is he driven by a desire to extend his career, to secure his legacy as the greatest of all time, or perhaps by something more personal? What price might he be willing to pay to claim this artifact, and what challenges from rivals past and present might stand in his way?"
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```

