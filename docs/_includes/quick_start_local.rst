..
   SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

This guide covers running Dynamo **using the CLI on your local machine or VM**.

.. important::

   **Looking to deploy on Kubernetes instead?**
   See the `Kubernetes Installation Guide <../kubernetes/installation_guide.html>`_
   and `Kubernetes Quickstart <../kubernetes/README.html>`_ for cluster deployments.

**1. Install Dynamo**

.. code-block:: bash

   # Install uv (recommended Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv venv
   source venv/bin/activate
   uv pip install pip

Install system dependencies and the Dynamo wheel for your chosen backend:

**SGLang**

.. code-block:: bash

   sudo apt install python3-dev
   uv pip install --prerelease=allow "ai-dynamo[sglang]"

**TensorRT-LLM**

.. code-block:: bash

   sudo apt install python3-dev
   pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"

.. note::

   TensorRT-LLM requires ``pip`` due to a transitive Git URL dependency that
   ``uv`` doesn't resolve. We recommend using the TensorRT-LLM container for
   broader compatibility. See the `TRT-LLM backend guide <../backends/trtllm/README.html>`_
   for details.

**vLLM**

.. code-block:: bash

   sudo apt install python3-dev libxcb1
   uv pip install --prerelease=allow "ai-dynamo[vllm]"

**Containers**

Pull and run prebuilt images from NVIDIA NGC. Container names follow the pattern
``nvcr.io/nvidia/ai-dynamo/{backend}-runtime:{version}``:

.. code-block:: bash

   # SGLang
   docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1
   docker run --rm -it --gpus all --network host nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1

   # TensorRT-LLM
   docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1
   docker run --rm -it --gpus all --network host nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1

   # vLLM
   docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
   docker run --rm -it --gpus all --network host nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1

See `Release Artifacts <../reference/release-artifacts.html#container-images>`_ for available
versions and backend guides for run instructions: `SGLang <../backends/sglang/README.html>`_ |
`TensorRT-LLM <../backends/trtllm/README.html>`_ | `vLLM <../backends/vllm/README.html>`_

**2. Sanity Check (Optional)**

Verify your system configuration and dependencies:

.. code-block:: bash

   python3 deploy/sanity_check.py

**3. Run Dynamo**

Start the frontend, then start a worker for your chosen backend:

.. code-block:: bash

   # Start the OpenAI compatible frontend (default port is 8000)
   # --store-kv file avoids needing etcd (frontend and workers must share a disk)
   python3 -m dynamo.frontend --store-kv file

In another terminal, start a worker:

**SGLang**

.. code-block:: bash

   python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --store-kv file

**TensorRT-LLM**

.. code-block:: bash

   python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --store-kv file

**vLLM**

.. code-block:: bash

   python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --store-kv file \
     --kv-events-config '{"enable_kv_cache_events": false}'

.. note::

   For dependency-free local development, disable KV event publishing (avoids NATS):

   - **vLLM:** Add ``--kv-events-config '{"enable_kv_cache_events": false}'``
   - **SGLang:** No flag needed (KV events disabled by default)
   - **TensorRT-LLM:** Do not use ``--publish-events-and-metrics``

   The warning ``Cannot connect to ModelExpress server/transport error. Using direct download.``
   is expected when running without NATS and can be safely ignored.

**4. Test your deployment**

.. code-block:: bash

   curl localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "Hello!"}],
          "max_tokens": 50}'
