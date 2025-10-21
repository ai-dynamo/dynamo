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
   Installation <_sections/installation>
   Support Matrix <reference/support-matrix.md>
   Architecture <_sections/architecture>
   Examples <_sections/examples>

.. toctree::
   :hidden:
   :caption: Kubernetes Deployment

   Quickstart (K8s) <../kubernetes/README.md>
   Detailed Installation Guide <../kubernetes/installation_guide.md>
   Creating Deployments <../kubernetes/create_deployment.md>
   API Reference <../kubernetes/api_reference.md>
   Dynamo Operator <../kubernetes/dynamo_operator.md>
   Metrics <../kubernetes/metrics.md>
   Logging <../kubernetes/logging.md>
   Multinode <../kubernetes/multinode-deployment.md>
   Minikube Setup <../kubernetes/minikube.md>

.. toctree::
   :hidden:
   :caption: Components

   Backends <_sections/backends>
   Router <router/README>
   Planner <planner/planner_intro>
   KVBM <kvbm/kvbm_intro>

.. toctree::
   :hidden:
   :caption: Developer Guide

   Benchmarking Guide <benchmarks/benchmarking.md>
   KV Router A/B Testing <benchmarks/kv-router-ab-testing.md>
   SLA Planner (Autoscaling) Quickstart <planner/sla_planner_quickstart>
   Logging <observability/logging.md>
   Health Checks <observability/health-checks.md>
   Tuning Disaggregated Serving Performance <performance/tuning.md>
   Writing Python Workers in Dynamo <development/backend-guide.md>
   Glossary <reference/glossary.md>
