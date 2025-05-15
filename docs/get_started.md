# Getting Started

## Development Environment

For a consistent development environment, use the provided devcontainer configuration. This requires:
- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

To use the devcontainer:
1. Open the project in VS Code.
2. Click the button in the bottom-left corner.
3. Select **Reopen in Container**.

This builds and starts a container with all the necessary dependencies for Dynamo development.


## Installation

 ```{note}
- The following examples require system level packages.
- We recommend Ubuntu 24.04 with a x86_64 CPU. See the [Support Matrix](support_matrix.md).
```

```
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
python3 -m venv venv
source venv/bin/activate

pip install ai-dynamo[all]
```

```{note}
To ensure compatibility, please refer to the examples in the release branch or tag that matches the version you installed.
```

## Building the Dynamo Base Image

Although not needed for local development, deploying your Dynamo pipelines to Kubernetes will require you to build and push a Dynamo base image to your container registry. You can use any container registry of your choice, such as:
- Docker Hub (docker.io)
- NVIDIA NGC Container Registry (nvcr.io)
- Any private registry

To build it:

```bash
./container/build.sh
docker tag dynamo:latest-vllm <your-registry>/dynamo-base:latest-vllm
docker login <your-registry>
docker push <your-registry>/dynamo-base:latest-vllm
```

This documentation describes these frameworks:
- `--framework vllm` build: see [here](examples/llm_deployment.md).
- `--framework tensorrtllm` build: see [here](examples/trtllm.md).

After building, use this image by setting the `DYNAMO_IMAGE` environment variable to point to your built image:
```bash
export DYNAMO_IMAGE=<your-registry>/dynamo-base:latest-vllm
```

## Running and Interacting with an LLM Locally

To run a model and interact with it locally,  call `dynamo run` with a hugging face model. `dynamo run` supports several backends including: `mistralrs`, `sglang`, `vllm`, and `tensorrtllm`.

### Example Command

```
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

```
? User › Hello, how are you?
✔ User · Hello, how are you?
Okay, so I'm trying to figure out how to respond to the user's greeting.
They said, "Hello, how are you?" and then followed it with "Hello! I'm just a program, but thanks for asking."
Hmm, I need to come up with a suitable reply. ...
```

## LLM Serving

Dynamo provides a simple way to spin up a local set of inference components including:

- **OpenAI Compatible Frontend**–High performance OpenAI compatible http api server written in Rust.
- **Basic and Kv Aware Router**–Route and load balance traffic to a set of workers.
- **Workers**–Set of pre-configured LLM serving engines.

To run a minimal configuration you can use a pre-configured
example.

### Start Dynamo Distributed Runtime Services

First start the Dynamo Distributed Runtime services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```
### Start Dynamo LLM Serving Components

Next serve a minimal configuration with an http server, basic
round-robin router, and a single worker.

```bash
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

### Send a Request

```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

## Local Development

If you use vscode or cursor, use the .devcontainer folder built on [Microsofts Extension](https://code.visualstudio.com/docs/devcontainers/containers). For instructions see the [ReadMe](https://github.com/ai-dynamo/dynamo/blob/main/.devcontainer/README.md) for more details.

Otherwise, to develop locally, we recommend working inside of the container

```bash
./container/build.sh
./container/run.sh -it --mount-workspace

cargo build --release
mkdir -p /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/http /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/llmctl /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/dynamo-run /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin

uv pip install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
```


### Conda Environment

Alternately, you can use a conda environment

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
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```


