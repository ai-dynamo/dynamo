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

### Configuring the NIXL Backend

You can select the backend for NIXL-based KV cache transfer by setting the `TRTLLM_NIXL_KVCACHE_BACKEND` environment variable. For example, to use the `LIBFABRIC` backend:

```bash
export TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC
```

Supported values include `UCX` and `LIBFABRIC`, depending on your cluster's hardware and software environment. Choose the backend that best matches your system for optimal performance and compatibility.

## Alternative Method: Direct UCX



