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

The NVIDIA Dynamo Platform is a high-performance, low-latency inference framework designed to serve all AI models—across any framework, architecture, or deployment scale.

.. admonition:: 💎 Discover the latest developments!
   :class: seealso

   This guide is a snapshot of the `Dynamo GitHub Repository <https://github.com/ai-dynamo/dynamo>`_ at a specific point in time. For the latest information and examples, see:

   - `Dynamo README <https://github.com/ai-dynamo/dynamo/blob/main/README.md>`_
   - `Architecture and features doc <https://github.com/ai-dynamo/dynamo/blob/main/docs/architecture/>`_
   - `Usage guides <https://github.com/ai-dynamo/dynamo/tree/main/docs/guides>`_
   - `Dynamo examples repo <https://github.com/ai-dynamo/examples>`_


Quick Start
-----------------
Follow the :doc:`Quick Guide to install Dynamo Platform <guides/dynamo_deploy/quickstart>`.


Dive in: Examples
-----------------

The examples below assume you build the latest image yourself from source. If using a prebuilt image follow the examples from the corresponding branch.

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


.. toctree::
   :hidden:

   Welcome to Dynamo <self>
   Support Matrix <support_matrix.md>

.. toctree::
   :hidden:
   :caption: Architecture & Features

   High Level Architecture <architecture/architecture.md>
   Distributed Runtime <architecture/distributed_runtime.md>
   Disaggregated Serving <architecture/disagg_serving.md>
   KV Block Manager <architecture/kvbm_intro.rst>
   KV Cache Routing <architecture/kv_cache_routing.md>
   Planner <architecture/planner_intro.rst>
   Dynamo Architecture Flow <architecture/dynamo_flow.md>

.. toctree::
   :hidden:
   :caption: Using Dynamo

   Running Inference Graphs Locally (dynamo-run) <guides/dynamo_run.md>
   Deploying Inference Graphs <guides/dynamo_deploy/README.md>

.. toctree::
   :hidden:
   :caption: Usage Guides

   Writing Python Workers in Dynamo <guides/backend.md>
   Disaggregation and Performance Tuning <guides/disagg_perf_tuning.md>
   KV Cache Router Performance Tuning <guides/kv_router_perf_tuning.md>
   Working with Dynamo Kubernetes Operator <guides/dynamo_deploy/dynamo_operator.md>

.. toctree::
   :hidden:
   :caption: Deployment Guides

   Dynamo Deploy Quickstart <guides/dynamo_deploy/quickstart.md>
   Dynamo Cloud Kubernetes Platform <guides/dynamo_deploy/dynamo_cloud.md>
   Manual Helm Deployment <guides/dynamo_deploy/manual_helm_deployment.md>
   GKE Setup Guide <guides/dynamo_deploy/gke_setup.md>
   Minikube Setup Guide <guides/dynamo_deploy/minikube.md>
   Model Caching with Fluid <guides/dynamo_deploy/model_caching_with_fluid.md>

.. toctree::
   :hidden:
   :caption: Benchmarking

   Planner Benchmark Example <guides/planner_benchmark/README.md>


.. toctree::
   :hidden:
   :caption: API

   Python API <API/python_bindings.md>
   NIXL Connect API <API/nixl_connect/README.md>

.. toctree::
   :hidden:
   :caption: Examples

   Aggregated and Disaggregated Deployment <examples/disagg_skeleton.md>
   LLM Deployment Examples <examples/llm_deployment.md>
   Multinode Examples <examples/multinode.md>
   LLM Deployment Examples using TensorRT-LLM <examples/trtllm.md>

.. toctree::
   :hidden:
   :caption: Reference

   Glossary <dynamo_glossary.md>
   KVBM Reading <architecture/kvbm_reading.md>


