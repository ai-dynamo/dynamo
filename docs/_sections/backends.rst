..
    SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Backends
========

NVIDIA Dynamo supports multiple inference backends to provide flexibility and performance optimization for different use cases and model architectures. Backends are the underlying engines that execute AI model inference, each optimized for specific scenarios, hardware configurations, and performance requirements.

Overview
--------

Dynamo's multi-backend architecture allows you to:

* **Choose the optimal engine** for your specific workload and hardware
* **Switch between backends** without changing your application code
* **Leverage specialized optimizations** from each backend
* **Scale flexibly** across different deployment scenarios

Supported Backends
------------------

Dynamo currently supports the following high-performance inference backends:

.. toctree::
   :maxdepth: 1

   vLLM <../backends/vllm/README>
   SGLang <../backends/sglang/README>
   TensorRT-LLM <../backends/trtllm/README>
