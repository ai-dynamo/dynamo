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

The NVIDIA Dynamo Platform is a high-performance, low-latency inference framework designed to serve generative AI and reasoning models—across any framework, architecture, or deployment scale. Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

Large language models are quickly outgrowing the memory and compute budget of any single GPU. Tensor-parallelism solves the capacity problem by spreading each layer across many GPUs—and sometimes many servers—but it creates a new one: how do you coordinate those shards, route requests, and share KV cache fast enough to feel like one accelerator? This orchestration gap is exactly what NVIDIA Dynamo is built to close.

.. admonition:: 💎 Discover the latest developments!
   :class: seealso

   This guide is a snapshot of the `Dynamo GitHub Repository <https://github.com/ai-dynamo/dynamo>`_ at a specific point in time. For the latest information and examples, see:

   - `Dynamo README <https://github.com/ai-dynamo/dynamo/blob/main/README.md>`_
   - `Architecture and Features <https://github.com/ai-dynamo/dynamo/blob/main/docs/architecture/>`_
   - `Usage Guides <https://github.com/ai-dynamo/dynamo/tree/main/docs/guides>`_
   - `Dynamo Examples <https://github.com/ai-dynamo/dynamo/tree/main/examples>`_


Quick Start
-----------------

Local Deployment
~~~~~~~~~~~~~~~~

Get started with Dynamo locally in just a few commands:

**1. Install Dynamo**

.. code-block:: bash

   # Install uv (recommended Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install Dynamo
   uv venv venv
   source venv/bin/activate
   uv pip install "ai-dynamo[sglang]"  # or [vllm], [trtllm]

**2. Start etcd/NATS** 

.. code-block:: bash

   # Start etcd and NATS using Docker Compose
   docker compose -f deploy/docker-compose.yml up -d

**3. Run Dynamo**

.. code-block:: bash

   # Start the OpenAI compatible frontend
   python -m dynamo.frontend
   
   # In another terminal, start an SGLang worker
   python -m dynamo.sglang.worker deepseek-ai/DeepSeek-R1-Distill-Llama-8B

**4. Test your deployment**

.. code-block:: bash

   curl localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
          "messages": [{"role": "user", "content": "Hello!"}], 
          "max_tokens": 50}'

Kubernetes Deployment  
~~~~~~~~~~~~~~~~~~~~~

For deployments on Kubernetes, follow the :doc:`Dynamo Platform Quickstart Guide <guides/dynamo_deploy/quickstart>`.


Dive in: Dynamo Examples
-----------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`Hello World <examples/runtime/hello_world/README>`
        :link: examples/runtime/hello_world/README
        :link-type: doc

        Demonstrates the basic concepts of Dynamo by creating a simple GPU-unaware graph using Python bindings.

    .. grid-item-card:: :doc:`LLM Serving with vLLM <components/backends/vllm/README>`
        :link: components/backends/vllm/README
        :link-type: doc

        Examples and reference implementations for deploying LLM inference workflows in various configurations with vLLM.

    .. grid-item-card:: :doc:`Deploy DeepSeek R1 Disaggregated with WideEP using SGLang <components/backends/sglang/docs/dsr1-wideep-gb200.md>`
        :link: components/backends/sglang/docs/dsr1-wideep-gb200.md
        :link-type: doc

        Demonstrates disaggregated serving of DeepSeek R1 using Wide Expert Parallelism using SGLang.

    .. grid-item-card:: :doc:`Deploy with TensorRT-LLM <components/backends/trtllm/README>`
        :link: components/backends/trtllm/README
        :link-type: doc

        Presents TensorRT-LLM examples and reference implementations for deploying LLMs in various configurations.


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

   Writing Python Workers in Dynamo <guides/backend.md>
   Disaggregation and Performance Tuning <guides/disagg_perf_tuning.md>
   KV Cache Router Performance Tuning <guides/kv_router_perf_tuning.md>
   Working with Dynamo Kubernetes Operator <guides/dynamo_deploy/dynamo_operator.md>
   Configuring Metrics for Observability <guides/metrics.md>

.. toctree::
   :hidden:
   :caption: Deployment Guides

   Deploying Dynamo on Kubernetes <guides/dynamo_deploy/README.md>
   Manual Helm Deployment <deploy/helm/README.md>
   Minikube Setup Guide <guides/dynamo_deploy/minikube.md>
   Model Caching with Fluid <guides/dynamo_deploy/model_caching_with_fluid.md>

.. toctree::
   :hidden:
   :caption: Examples

   Hello World <examples/runtime/hello_world/README.md>
   LLM Deployment Examples using VLLM <components/backends/vllm/README.md>
   Multinode Examples using SGLang <components/backends/sglang/docs/multinode-examples.md>
   LLM Deployment Examples using TensorRT-LLM <components/backends/trtllm/README.md>
   Planner Benchmark Example <guides/planner_benchmark/README.md>

.. toctree::
   :hidden:
   :caption: Reference

   Glossary <dynamo_glossary.md>
   NIXL Connect API <API/nixl_connect/README.md>
   KVBM Reading <architecture/kvbm_reading.md>


