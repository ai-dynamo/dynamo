<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
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
-->

# Getting Started


## Development Environment

This section describes how to set up your development environment.

### Recommended Setup: Using Dev Container

We recommend using our pre-configured development container:

1. Install prerequisites:

   - [Docker](https://www.docker.com/products/docker-desktop)
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. Get the code:

   ```bash
   git clone https://github.com/ai-dynamo/dynamo.git
   cd dynamo
   ```

3. Open in Visual Studio Code:

   1. Launch Visual Studio Code
   2. Click the button in the bottom-left corner
   3. Select **Reopen in Container**

Visual Studio Code builds and starts a container with all necessary dependencies for Dynamo development.

### Alternative Setup: Manual Installation

If you don't want to use the dev container, you can set the environment up manually:

1. Ensure you have:

   - Ubuntu 24.04 (recommended)
   - x86_64 CPU
   - Python 3.x
   - Git

   See [Support Matrix](support_matrix.md) for more information.

2. **If you plan to use vLLM or SGLang**, you must also install:
   - etcd
   - NATS.io

   Before starting dynamo, run both etcd and NATS.io in separate processes.

3. Install required system packages:
   ```bash
   apt-get update
   DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
   ```

4. Set up the Python environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. Install Dynamo:
   ```bash
   pip install "ai-dynamo[all]"
   ```

> [!Important]
> To ensure compatibility, use the examples in the release branch or tag that matches the version you installed.


## Building the Dynamo Base Image

Deploying your Dynamo pipelines to Kubernetes requires you to build and push a Dynamo base image to your container registry.
You can use any private container registry of your choice, including:

- [Docker Hub](https://hub.docker.com/)
- [NVIDIA NGC Container Registry](https://catalog.ngc.nvidia.com/)


To build it:

```bash
./container/build.sh
docker tag dynamo:latest-vllm <your-registry>/dynamo-base:latest-vllm
docker login <your-registry>
docker push <your-registry>/dynamo-base:latest-vllm
```

This documentation describes these frameworks:

- `--framework vllm` build:
   See [LLM Deployment Examples](examples/llm_deployment.md).

- `--framework tensorrtllm` build:
   See [TRTLLM Deployment Examples](examples/trtllm.md).

After building, use this image by setting the `DYNAMO_IMAGE` environment variable to point to your built image:

```bash
export DYNAMO_IMAGE=<your-registry>/dynamo-base:latest-vllm
```


## Running and Interacting with an LLM Locally

Dynamo supports several backends, including `mistralrs`, `sglang`, `vllm`, and `tensorrtllm`.
Use example commands below tp launch a model.

### Example Command

```bash
python -m dynamo.frontend [--http-port 8080]
python -m dynamo.vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

```bash
? User › Hello, how are you?
✔ User · Hello, how are you?
Okay, so I'm trying to figure out how to respond to the user's greeting.
They said, "Hello, how are you?" and then followed it with "Hello! I'm just a program, but thanks for asking."
Hmm, I need to come up with a suitable reply. ...
```


## LLM Serving

Dynamo provides a simple way to spin up a local set of inference components including:

- **OpenAI-compatible Frontend**:
   High-performance OpenAI compatible http api server written in Rust.

- **Basic and Kv Aware Router**:
   Route and load balance traffic to a set of workers.

- **Workers**:
   Set of pre-configured LLM serving engines.

To run a minimal configuration, use a pre-configured example.

### Start Dynamo Distributed Runtime Services

To start the Dynamo Distributed Runtime services the first time:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Start Dynamo LLM Serving Components

[Explore the VLLM Example](../examples/vllm/README.md)


## Local Development

If you use vscode or cursor, use the `.devcontainer` folder built on [Microsoft's Extension](https://code.visualstudio.com/docs/devcontainers/containers).
For instructions, see the Dynamo repository's [devcontainer README](https://github.com/ai-dynamo/dynamo/blob/main/.devcontainer/README.md).

Otherwise, to develop locally, we recommend working inside of the container:

```bash
./container/build.sh
./container/run.sh -it --mount-workspace

cargo build --release
mkdir -p /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/dynamo-run /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin

uv pip install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
```

### Conda Environment

Alternatively, use a Conda environment:

```bash
conda activate <ENV_NAME>

pip install nixl # Or install https://github.com/ai-dynamo/nixl from source

cargo build --release

# To install ai-dynamo-runtime from source
cd lib/bindings/python
pip install .

cd ../../../
pip install .[all]

# To test
docker compose -f deploy/docker-compose.yml up -d
python -m dynamo.frontend [--http-port 8080]
python -m dynamo.vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```
