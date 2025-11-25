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

## Default Method: UCX
By default, TensorRT-LLM uses UCX (Unified Communication X) for KV cache transfer between prefill and decode workers. UCX provides high-performance communication optimized for GPU-to-GPU transfers.

## Beta Method: NIXL
TensorRT-LLM also supports using **NIXL** (NVIDIA Inference Xfer Library) for KV cache transfer. [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's high-performance communication library designed for efficient data transfer in distributed GPU environments.

**Note:** NIXL support in TensorRT-LLM is currently beta and may have some sharp edges.

## Using NIXL for KV Cache Transfer

**Note:** NIXL backend for TensorRT-LLM is currently only supported on AMD64 (x86_64) architecture. If you're running on ARM64, you'll need to use the default UCX method for KV cache transfer.

To enable NIXL for KV cache transfer in disaggregated serving:

1. **Build the container with NIXL support:**
   The TensorRT-LLM wheel must be built from source with NIXL support. The `./container/build.sh` script caches previously built TensorRT-LLM wheels to reduce build time. If you have previously built a TensorRT-LLM wheel without NIXL support, you must delete the cached wheel to force a rebuild with NIXL support.

   **Remove cached TensorRT-LLM wheel (only if previously built without NIXL support):**
   ```bash
   rm -rf /tmp/trtllm_wheel
   ```

   **Build the container with NIXL support:**
   ```bash
   ./container/build.sh --framework trtllm \
     --tensorrtllm-git-url https://github.com/NVIDIA/TensorRT-LLM.git \
     --tensorrtllm-commit v1.2.0rc2
   ```

2. **Run the containerized environment:**
   See [run container](./README.md#run-container) section to learn how to start the container image built in previous step.

3. **Start the disaggregated service:**
   See [disaggregated serving](./README.md#disaggregated-serving) to see how to start the deployment.

4. **Send the request:**
   See [client](./README.md#client) section to learn how to send the request to deployment.

**Important:** Ensure that ETCD and NATS services are running before starting the service.
