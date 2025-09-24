Pip (PyPI)
----------

Install a pre-built wheel from PyPI.

.. code-block:: bash

   # Create a virtual environment and activate it
   uv venv venv
   source venv/bin/activate

   # Install Dynamo from PyPI (choose one backend extra)
   uv pip install "ai-dynamo[sglang]==my-tag"  # or [vllm], [trtllm]


Pip from source
---------------

Install directly from a local checkout for development.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ai-dynamo/dynamo.git
   cd dynamo

   # Create a virtual environment and activate it
   uv venv venv
   source venv/bin/activate
   uv pip install ".[sglang]"  # or [vllm], [trtllm]


Docker
------

Pull and run prebuilt images from NVIDIA NGC (`nvcr.io`).

.. code-block:: bash

   # Run a container (mount your workspace if needed)
   docker run --rm -it \
     --gpus all \
     --network host \
     nvcr.io/nvidia/ai-dynamo/sglang-runtime:my-tag  # or vllm, tensorrtllm
