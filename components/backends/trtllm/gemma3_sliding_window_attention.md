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
* To run Gemma 3 with VSWA and KV Routing with KV block reuse, ensure that the container is built using commit ID `c9eebcb4541d961ab390f0bd0a22e2c89f1bcc78` from Tensorrt-LLM.
```bash
./container/build.sh --framework trtllm --tensorrtllm-commit c9eebcb4541d961ab390f0bd0a22e2c89f1bcc78
```
* The 1.0.0rc4 release version of TensorRT-LLM can also run Gemma 3 with VSWA, but KV block reuse cannot be turned on in that version.

### Aggregated Serving
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=engine_configs/gemma3/vswa_agg.yaml
./launch/agg.sh
```

### Aggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export AGG_ENGINE_ARGS=engine_configs/gemma3/vswa_agg.yaml
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

#### Disaggregated Serving with KV Routing
```bash
cd $DYNAMO_HOME/components/backends/trtllm
export MODEL_PATH=google/gemma-3-1b-it
export SERVED_MODEL_NAME=$MODEL_PATH
export PREFILL_ENGINE_ARGS=engine_configs/gemma3/vswa_prefill.yaml
export DECODE_ENGINE_ARGS=engine_configs/gemma3/vswa_decode.yaml
./launch/disagg_router.sh
```
