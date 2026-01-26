Get started with Dynamo locally in just a few commands:

**1. Install Dynamo**

Backend engines require Python development headers for JIT compilation:

.. code-block:: bash

   sudo apt install python3-dev

Install uv and create a virtual environment:

.. code-block:: bash

   # Install uv (recommended Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install Dynamo
   uv venv venv
   source venv/bin/activate
   uv pip install pip

   # For vLLM or SGLang:
   uv pip install --prerelease=allow "ai-dynamo[sglang]"  # or [vllm]

   # For TensorRT-LLM (requires pip, not uv):
   pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"

.. note::

   TensorRT-LLM uses ``pip`` instead of ``uv`` due to URL-based dependencies,
   and requires NVIDIA's PyPI for the TRT-LLM wheel. The wheel may not be
   available for all CUDA versions—see `Release Artifacts
   <../reference/release-artifacts.html#python-wheels>`_ for availability. For
   broader compatibility, we recommend using the `prebuilt TRT-LLM container
   <../reference/release-artifacts.html#container-images>`_. See the
   `TRT-LLM backend guide <../backends/trtllm/README.html>`_ for details.

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


