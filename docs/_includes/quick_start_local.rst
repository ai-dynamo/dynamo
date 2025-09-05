Get started with Dynamo locally in just a few commands:

**1. Install Dynamo**

.. code-block:: bash

   # Install uv (recommended Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install Dynamo
   uv venv venv
   source venv/bin/activate
   uv pip install "ai-dynamo[sglang]==0.4.1"  # or [vllm], [trtllm]

**2. Start etcd/NATS**

.. code-block:: bash

   # Fetch and start etcd and NATS using Docker Compose
   curl -fsSL -o docker-compose.yml https://raw.githubusercontent.com/ai-dynamo/dynamo/release/0.4.1/deploy/docker-compose.yml
   docker compose -f docker-compose.yml up -d

**3. Run Dynamo**

.. code-block:: bash

   # Start the OpenAI compatible frontend (default port is 8000)
   python -m dynamo.frontend

   # In another terminal, start an SGLang worker
   python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B

**4. Test your deployment**

.. code-block:: bash

   curl localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "Hello!"}],
          "max_tokens": 50}'


