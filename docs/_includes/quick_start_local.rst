This guide covers running Dynamo **using the CLI on your local machine or VM**.

.. important::

   **Looking to deploy on Kubernetes instead?**
   See the `Kubernetes Installation Guide <../kubernetes/installation_guide.html>`_
   and `Kubernetes Quickstart <../kubernetes/README.html>`_ for cluster deployments.

**1. Install Dynamo**

Backend engines require Python development headers for JIT compilation:

.. code-block:: bash

   sudo apt install python3-dev

**Option A: Install from PyPI**

.. code-block:: bash

   # Install uv (recommended Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv venv
   source venv/bin/activate
   uv pip install pip

   # For vLLM or SGLang:
   uv pip install --prerelease=allow "ai-dynamo[sglang]"  # or [vllm]

   # For TensorRT-LLM (requires pip and NVIDIA PyPI):
   pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"

.. note::

   TensorRT-LLM requires ``pip`` due to a transitive Git URL dependency that
   ``uv`` doesn't resolve. We recommend using the `TensorRT-LLM container
   <../reference/release-artifacts.html#container-images>`_ for broader
   compatibility. See the `TRT-LLM backend guide <../backends/trtllm/README.html>`_
   for details.

**Option B: Docker**

Pull and run prebuilt images from NVIDIA NGC. Container names follow the pattern
``nvcr.io/nvidia/ai-dynamo/{backend}-runtime:{version}``:

.. code-block:: bash

   # Examples:
   docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
   docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1
   docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1

   # Run with GPU access
   docker run --rm -it --gpus all --network host \
     nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1

See `Release Artifacts <../reference/release-artifacts.html#container-images>`_ for available
versions and backend guides for run instructions: `vLLM <../backends/vllm/README.html>`_ |
`SGLang <../backends/sglang/README.html>`_ | `TensorRT-LLM <../backends/trtllm/README.html>`_

**2. Sanity Check (Optional)**

Verify your system configuration and dependencies:

.. code-block:: bash

   python3 deploy/sanity_check.py

**3. Run Dynamo**

.. code-block:: bash

   # Start the OpenAI compatible frontend (default port is 8000)
   # --store-kv file avoids needing etcd (frontend and workers must share a disk)
   python -m dynamo.frontend --store-kv file

   # In another terminal, start an SGLang worker
   python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --store-kv file

.. note::

   vLLM workers publish KV cache events by default, which requires NATS. For
   dependency-free local development with vLLM, add
   ``--kv-events-config '{"enable_kv_cache_events": false}'``. This keeps local
   prefix caching enabled while disabling event publishing.

**4. Test your deployment**

.. code-block:: bash

   curl localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "Hello!"}],
          "max_tokens": 50}'
