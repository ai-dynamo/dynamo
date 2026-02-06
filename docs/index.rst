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

   Deployment Guide <_sections/k8s_deployment>
   Observability (K8s) <_sections/k8s_observability>
   Multinode <_sections/k8s_multinode>

.. toctree::
   :hidden:
   :caption: User Guides

   KV Cache Offloading <components/kvbm/kvbm_guide.md>
   KV Aware Routing <components/router/router_guide.md>
   Tool Calling <agents/tool-calling.md>
   Multimodality Support <features/multimodal/README.md>
   LoRA Adapters <features/lora/README.md>
   Finding Best Initial Configs <performance/aiconfigurator.md>
   Benchmarking <benchmarks/benchmarking.md>
   Tuning Disaggregated Performance <performance/tuning.md>
   Writing Python Workers in Dynamo <development/backend-guide.md>
   Observability (Local) <_sections/observability>
   Fault Tolerance <_sections/fault_tolerance>
   Glossary <reference/glossary.md>

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
   :caption: Design Docs

   Overall Architecture <design_docs/architecture.md>
   Architecture Flow <design_docs/dynamo_flow.md>
   Disaggregated Serving <design_docs/disagg_serving.md>
   Distributed Runtime <design_docs/distributed_runtime.md>
   Router Design <design_docs/router_design.md>
   Request Plane <design_docs/request_plane.md>
   Event Plane <design_docs/event_plane.md>
   Planner Design <design_docs/planner_design.md>
