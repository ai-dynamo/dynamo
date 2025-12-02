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



# KV Cache Transfer in Disaggregated Serving

In disaggregated serving architectures, KV cache must be transferred between prefill and decode workers. TensorRT-LLM supports two methods for this transfer:

## Default Method: NIXL
By default, TensorRT-LLM uses **NIXL** (NVIDIA Inference Xfer Library) with UCX (Unified Communication X) as backend for KV cache transfer between prefill and decode workers. [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's high-performance communication library designed for efficient data transfer in distributed GPU environments.

### Specify Backends for NIXL

TODO: Add instructions for how to specify different backends for NIXL.

## Alternative Method: UCX

TensorRT-LLM can also leverage **UCX** (Unified Communication X) directly for KV cache transfer between prefill and decode workers. There are two ways to enable UCX as the KV cache transfer backend:

1. **Recommended:** Set `cache_transceiver_config.backend: UCX` in your engine configuration YAML file.
2. Alternatively, set the environment variable `TRTLLM_USE_UCX_KV_CACHE=1` and configure `cache_transceiver_config.backend: DEFAULT` in the engine configuration YAML.

This flexibility allows users to choose the most suitable method for their deployment and compatibility requirements.
