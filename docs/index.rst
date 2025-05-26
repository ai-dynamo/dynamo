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

Welcome to NVIDIA Dynamo
========================

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.

Dive in: Examples
-----------------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`Hello World </examples/hello_world>`
        :link: /examples/hello_world
        :link-type: doc

        Demonstrates the basic concepts of Dynamo by creating a simple multi-service pipeline.

    .. grid-item-card:: :doc:`LLM Deployment </examples/llm_deployment>`
        :link: /examples/llm_deployment
        :link-type: doc

        Presents examples and reference implementations for deploying Large Language Models (LLMs) in various configurations.

    .. grid-item-card:: :doc:`Multinode </examples/multinode>`
        :link: /examples/multinode
        :link-type: doc

        Demonstrates deployment for disaggregated serving on 3 nodes using `nvidia/Llama-3.1-405B-Instruct-FP8`.

    .. grid-item-card:: :doc:`TensorRT-LLM </examples/trtllm>`
        :link: /examples/trtllm
        :link-type: doc

        Presents TensorRT-LLM examples and reference implementations for deploying Large Language Models (LLMs) in various configurations.

Overview
--------

The NVIDIA Dynamo Platform is a high-performance, low-latency inference platform
designed to serve all AI models—across any framework, architecture, or deployment scale.
Dynamo is inference engine agnostic, supporting TRT-LLM, vLLM, SGLang, and others, and captures
LLM-specific capabilities such as:

* **Disaggregated prefill & decode inference** - Maximizes GPU throughput and facilitates trade off between throughput and latency.
* **Dynamic GPU scheduling** - Optimizes performance based on fluctuating demand.
* **LLM-aware request routing** - Eliminates unnecessary KV cache re-computation.
* **Accelerated data transfer** - Reduces inference response time using NIXL.
* **KV cache offloading** - Leverages several memory hierarchies for higher system throughput.

Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source
and driven by a transparent, OSS (Open Source Software) first development approach.

.. toctree::
   :hidden:

   Welcome to Dynamo <self>
   Support Matrix <support_matrix.md>
   Getting Started <get_started.md>

.. toctree::
   :hidden:
   :caption: Architecture & Features

   High Level Architecture <architecture/architecture.md>
   Distributed Runtime <architecture/distributed_runtime.md>
   Disaggregated Serving <architecture/disagg_serving.md>
   KV Block Manager <architecture/kvbm_intro.rst>
   KV Cache Routing <architecture/kv_cache_routing.md>
   Planner <guides/planner.md>

.. toctree::
   :hidden:
   :caption: Dynamo Command Line Interface

   CLI Overview <guides/cli_overview.md>
   Running Dynamo (dynamo run) <guides/dynamo_run.md>
   Serving Inference Graphs (dynamo serve) <guides/dynamo_serve.md>
   Building Dynamo (dynamo build) <guides/dynamo_build.md>
   Deploying Inference Graphs (dynamo deploy) <guides/dynamo_deploy/README.md>

.. toctree::
   :hidden:
   :caption: Usage Guides

   Writing Python Workers in Dynamo <guides/backend.md>
   Disaggregation and Performance Tuning <guides/disagg_perf_tuning.md>
   KV Cache Router Performance Tuning <guides/kv_router_perf_tuning.md>
   Planner Benchmark Example <guides/planner_benchmark/benchmark_planner.md>

.. toctree::
   :hidden:
   :caption: Deployment Guides

   Dynamo Cloud Kubernetes Platform <guides/dynamo_deploy/dynamo_cloud.md>
   Deploying Dynamo Inference Graphs to Kubernetes using the Dynamo Cloud Platform <guides/dynamo_deploy/operator_deployment.md>
   Manual Helm Deployment <guides/dynamo_deploy/manual_helm_deployment.md>
   Minikube Setup Guide <guides/dynamo_deploy/minikube.md>

.. toctree::
   :hidden:
   :caption: API

   SDK Reference <API/sdk.md>
   Python API <API/python_bindings.md>

.. toctree::
   :hidden:
   :caption: Examples

   Hello World Example <examples/hello_world.md>
   LLM Deployment Examples <examples/llm_deployment.md>
   Multinode Examples <examples/multinode.md>
   LLM Deployment Examples using TensorRT-LLM <examples/trtllm.md>



