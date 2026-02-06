..
    SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

KV Block Manager
================
The Dynamo KV Block Manager (KVBM) is a scalable runtime component designed to handle memory allocation, management, and remote sharing of Key-Value (KV) blocks for inference tasks across heterogeneous and distributed environments. It acts as a unified memory layer for frameworks like vLLM, SGLang, and TRT-LLM.

It offers:

* A **unified memory API** that spans GPU memory (future), pinned host memory, remote RDMA-accessible memory, local or distributed pool of SSDs and remote file/object/cloud storage systems.
* Support for evolving **block lifecycles** (allocate → register → match) with event-based state transitions that storage can subscribe to.
* Integration with **NIXL**, a dynamic memory exchange layer used for remote registration, sharing, and access of memory blocks over RDMA/NVLink.

The Dynamo KV Block Manager serves as a reference implementation that emphasizes modularity and extensibility. Its pluggable design enables developers to customize components and optimize for specific performance, memory, and deployment needs.

.. list-table::
   :widths: 20 5 75
   :header-rows: 1

   * -
     -
     - Feature
   * - **Backend**
     - ✅
     - Local
   * -
     - ✅
     - Kubernetes
   * - **LLM Framework**
     - ✅
     - vLLM
   * -
     - ✅
     - TensorRT-LLM
   * -
     - ❌
     - SGLang
   * - **Serving Type**
     - ✅
     - Aggregated
   * -
     - ✅
     - Disaggregated

.. toctree::
   :hidden:

   Overview <self>
   Quick Start <README.md>
   User Guide <kvbm_guide.md>
   Design <kvbm_design.md>
   LMCache Integration <../integrations/lmcache_integration.md>
   FlexKV Integration <../integrations/flexkv_integration.md>
   SGLang HiCache <../integrations/sglang_hicache.md>