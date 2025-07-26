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

# Gemma 3 with Variable Sliding Window Attention

This guide demonstrates how to deploy google/gemma-3-1b-it with Variable Sliding Window Attention (VSWA) using Dynamo. Since google/gemma-3-1b-it is a small model, each aggregated, decode, or prefill worker only requires one H100 GPU or one GB200 GPU.
VSWA is a mechanism in which a model’s layers alternate between multiple sliding window sizes. An example of this is Gemma 3, which incorporates both global attention layers and sliding window layers.

## Notes
* To run Gemma 3 with VSWA, ensure that the container has TensorRT-LLM v1.0.0rc4 installed.

## Limitation
* The current KV event-based KV routing does not work well with VSWA. The Dynamo team is actively working on adding support to distinguish between events from different layer groups. As a workaround, Dynamo can perform KV routing without relying on KV events by instead leveraging the prompt prefix of previous requests, refers to Approximate KV Routing. If a request is routed to a worker and another request with the same prefix arrives shortly afterward, it can be routed to the same worker due to the high probability of a large cache hit.
* The Approximate KV indexer uses a TTL (Time to Live) of 120 seconds.

### Aggregated Serving
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=engine_configs/gemma3/vswa_agg.yaml
./launch/agg.sh
```

### Aggregated Serving with Approximate KV Routing
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=engine_configs/gemma3/vswa_agg.yaml
export ENABLE_KV_EVENTS=false # disable kv events to use Approximate KV Routing
./launch/agg_router.sh
```

#### Disaggregated Serving
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=engine_configs/gemma3/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=engine_configs/gemma3/vswa_decode.yaml
./launch/disagg.sh
```

#### Disaggregated Serving with Approximate KV Routing
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=engine_configs/gemma3/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=engine_configs/gemma3/vswa_decode.yaml
export ENABLE_KV_EVENTS=false # disable kv events to use Approximate KV Routing
./launch/disagg_router.sh
```
