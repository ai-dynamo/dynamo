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

> [!Note]
> - Ensure that required services such as `nats` and `etcd` are running before starting.
> - Request access to `google/gemma-3-1b-it` on Hugging Face and set your `HF_TOKEN` environment variable for authentication.
> - It's recommended to continue using the VSWA feature with the Dynamo 0.5.0 release and the TensorRT-LLM dynamo runtime image nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0. The 0.5.1 release bundles TensorRT-LLM v1.1.0rc5, which has a regression that breaks VSWA.

## Aggregated Serving
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_agg.yaml
./launch/agg.sh
```

## Aggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_agg.yaml
./launch/agg_router.sh
```

## Disaggregated Serving
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_decode.yaml
./launch/disagg.sh
```

## Disaggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=$DYNAMO_HOME/recipes/gemma3/trtllm/vswa_decode.yaml
./launch/disagg_router.sh
```
