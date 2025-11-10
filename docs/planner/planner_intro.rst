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

Planner
=======

The planner monitors the state of the system and adjusts workers to ensure that the system runs efficiently.

Currently, the planner can scale the number of vllm workers up and down based on the kv cache load and prefill queue size:

Key features include:

* **SLA-based scaling** that uses predictive modeling and performance interpolation to proactively meet TTFT and ITL targets
* **Graceful scaling** that ensures no requests are dropped during scale-down operations

.. admonition:: 🚀 Quick Start
   :class: seealso

   **New to SLA Planner?** Start with the [SLA Planner Quick Start Guide](/docs/planner/sla_planner_quickstart.md) for a complete, step-by-step workflow.

   **Prerequisites**: SLA-based planner requires pre-deployment profiling (2-4 hours on real silicon or a few minutes using simulator) before deployment. The Quick Start guide includes everything you need.

.. list-table::
   :widths: 20 5 75
   :header-rows: 1

   * -
     -
     - Feature
   * - **Backend**
     - ❌
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
     - ✅
     - SGLang
   * - **Serving Type**
     - ✅
     - Aggregated
   * -
     - ✅
     - Disaggregated
   * - **Planner Actions**
     - ❌
     - Load-based scaling up/down prefill/decode workers
   * -
     - ✅
     - SLA-based scaling up/down prefill/decode workers [1]_
   * -
     - ❌
     - Adjusting engine knobs

.. [1] Supported with some limitations.

.. toctree::
   :hidden:

   Overview <self>
   SLA Planner Quick Start <sla_planner_quickstart>
   SLA-Driven Profiling <../benchmarks/sla_driven_profiling.md>
   SLA-based Planner <sla_planner.md>
