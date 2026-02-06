..
   SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

This guide covers running Dynamo **using the CLI on your local machine or VM**.

.. important::

   **Looking to deploy on Kubernetes instead?**
   See the `Kubernetes Installation Guide <../kubernetes/installation_guide.html>`_
   and `Kubernetes Quickstart <../kubernetes/README.html>`_ for cluster deployments.

**Install Dynamo**

**Option A: Containers (Recommended)**

Containers have all dependencies pre-installed. No setup required.

.. code-block:: bash

   # SGLang
   docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1

   # TensorRT-LLM
   docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1

   # vLLM
   docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1

.. tip::

   To run frontend and worker in the same container, either:

   - Run processes in background with ``&`` (see Run Dynamo section below), or
   - Open a second terminal and use ``docker exec -it <container_id> bash``

See `Release Artifacts <../reference/release-artifacts.html#container-images>`_ for available
versions and backend guides for run instructions: `SGLang <../backends/sglang/README.html>`_ |
`TensorRT-LLM <../backends/trtllm/README.html>`_ | `vLLM <../backends/vllm/README.html>`_

**Option B: Install from PyPI**

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

.. note::

   For CUDA 13 (B300/GB300), the container is recommended. See
   `SGLang install docs <https://docs.sglang.io/get_started/install.html>`_ for details.

**TensorRT-LLM**

.. code-block:: bash

   sudo apt install python3-dev
   pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
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

**Run Dynamo**

.. tip::

   **(Optional)** Before running Dynamo, verify your system configuration:
   ``python3 deploy/sanity_check.py``

Start the frontend, then start a worker for your chosen backend.

.. tip::

   To run in a single terminal (useful in containers), append ``> logfile.log 2>&1 &``
   to run processes in background. Example: ``python3 -m dynamo.frontend --store-kv file > dynamo.frontend.log 2>&1 &``

.. code-block:: bash

   # Start the OpenAI compatible frontend (default port is 8000)
   # --store-kv file avoids needing etcd (frontend and workers must share a disk)
   python3 -m dynamo.frontend --store-kv file

In another terminal (or same terminal if using background mode), start a worker:

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
   - **TensorRT-LLM:** No flag needed (KV events disabled by default)

   **TensorRT-LLM only:** The warning ``Cannot connect to ModelExpress server/transport error. Using direct download.``
   is expected and can be safely ignored.

**Test Your Deployment**

.. code-block:: bash

   curl localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "Hello!"}],
          "max_tokens": 50}'
