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

..
   Main Page
..

Welcome to NVIDIA Dynamo
========================

The NVIDIA Dynamo Platform is a high-performance, low-latency inference framework designed to serve all AI models—across any framework, architecture, or deployment scale.

.. admonition:: 💎 Discover the latest developments!
   :class: seealso

   This guide is a snapshot of a specific point in time. For the latest information, examples, and Release Assets, see the `Dynamo GitHub repository <https://github.com/ai-dynamo/dynamo/releases/latest>`_.

Quickstart
==========
.. include:: _includes/quick_start_local.rst

..
   Sidebar
..

.. toctree::
   :hidden:
   :caption: Getting Started

   Quickstart <self>
   Support Matrix <reference/support-matrix.md>
   Feature Matrix <reference/feature-matrix.md>
   Release Artifacts <reference/release-artifacts.md>
   Examples <_sections/examples>

.. toctree::
   :hidden:
   :caption: Kubernetes Deployment

   Deployment Guide <kubernetes/README>
   Observability (K8s) <kubernetes/observability/metrics>
   Multinode <kubernetes/deployment/multinode-deployment>

.. toctree::
   :hidden:
   :caption: User Guides

   KV Cache Aware Routing <components/router/router_guide.md>
   Disaggregated Serving Guide <features/disaggregated_serving/README.md>
   KV Cache Offloading <components/kvbm/kvbm_guide.md>
   Benchmarking <benchmarks/benchmarking.md>
   Multimodality Support <features/multimodal/README.md>
   Tool Calling <agents/tool-calling.md>
   LoRA Adapters <features/lora/README.md>
   Observability (Local) <observability/README>
   Fault Tolerance <fault_tolerance/README>
   Writing Python Workers in Dynamo <development/backend-guide.md>

.. toctree::
   :hidden:
   :caption: Components

   Backends <_sections/backends>
   Frontend <components/frontend/README>
   Router <components/router/README>
   Planner <components/planner/README>
   Profiler <components/profiler/README>
   KVBM <components/kvbm/README>

.. toctree::
   :hidden:
   :caption: Integrations

   LMCache <integrations/lmcache_integration.md>
   SGLang HiCache <integrations/sglang_hicache.md>
   FlexKV <integrations/flexkv_integration.md>
   KV Events for Custom Engines <integrations/kv_events_custom_engines.md>

.. toctree::
   :hidden:
   :caption: Design Docs

   Overall Architecture <design_docs/architecture.md>
   Architecture Flow <design_docs/dynamo_flow.md>
   Disaggregated Serving <design_docs/disagg_serving.md>
   Distributed Runtime <design_docs/distributed_runtime.md>
   Request Plane <design_docs/request_plane.md>
   Event Plane <design_docs/event_plane.md>
   Router Design <design_docs/router_design.md>
   KVBM Design <design_docs/kvbm_design.md>
   Planner Design <design_docs/planner_design.md>
