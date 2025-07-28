<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# LLM Deployment Examples using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.

# User Documentation

- [Deployment Architectures](#deployment-architectures)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Build docker](#build-docker)
  - [Run container](#run-container)
  - [Run deployment](#run-deployment)
    - [Single Node deployment](#single-node-deployments)
    - [Multinode deployment](#multinode-deployment)
  - [Client](#client)
  - [Benchmarking](#benchmarking)
- [Disaggregation Strategy](#disaggregation-strategy)
- [KV Cache Transfer](#kv-cache-transfer-in-disaggregated-serving)
- [More Example Architectures](#more-example-architectures)
  - [Llama 4 Maverick Instruct + Eagle Speculative Decoding](./llama4_plus_eagle.md)

# Quick Start

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture.

Note: TensorRT-LLM disaggregation does not support conditional disaggregation yet. You can configure the deployment to always use either aggregate or disaggregated serving.

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../../deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build docker

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
./container/build.sh --framework tensorrtllm

# On an ARM machine:
./container/build.sh --framework tensorrtllm --platform linux/arm64

# Build the container with the default experimental TensorRT-LLM commit
# WARNING: This is for experimental feature testing only.
# The container should not be used in a production environment.
./container/build.sh --framework tensorrtllm --use-default-experimental-tensorrtllm-commit
```

### Run container

```
./container/run.sh --framework tensorrtllm -it
```
## Run Deployment

This figure shows an overview of the major components to deploy:



```

+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| processor |----->|      Worker1     |------------>|    Worker2    |
|      |<-----|           |<-----|                  |<------------|               |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+

```

**Note:** The diagram above shows all possible components in a deployment. Depending on the chosen disaggregation strategy, you can configure whether Worker1 handles prefill and Worker2 handles decode, or vice versa. For more information on how to select and configure these strategies, see the [Disaggregation Strategy](#disaggregation-strategy) section below.

### Single-Node Deployments

> [!IMPORTANT]
> Below we provide some simple shell scripts that run the components for each configuration. Each shell script is simply running the `dynamo-run` to start up the ingress and using `python3` to start up the workers. You can easily take each command and run them in separate terminals.

#### Aggregated
```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/agg.sh
```

#### Disaggregated

> [!IMPORTANT]
> Disaggregated serving supports two strategies for request flow: `"prefill_first"` and `"decode_first"`. By default, the script below uses the `"decode_first"` strategy, which can reduce response latency by minimizing extra hops in the return path. You can switch strategies by setting the `DISAGGREGATION_STRATEGY` environment variable.

```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/disagg.sh
```

### Multinode Deployment

For comprehensive instructions on multinode serving, see the [multinode-examples.md](./multinode/multinode-examples.md) guide. It provides step-by-step deployment examples and configuration tips for running Dynamo with TensorRT-LLM across multiple nodes. While the walkthrough uses DeepSeek-R1 as the model, you can easily adapt the process for any supported model by updating the relevant configuration files. You can see [Llama4+eagle](./llama4_plus_eagle.md) guide to learn how to use these scripts when a single worker fits on the single node.

### Client

Below is an example of image being sent to `Llama-4-Maverick-17B-128E-Instruct` model

Request :
```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```
Response :

```
{"id":"unknown-id","choices":[{"index":0,"message":{"content":"The image depicts a serene landscape featuring a large rock formation, likely El Capitan in Yosemite National Park, California. The scene is characterized by a winding road that curves from the bottom-right corner towards the center-left of the image, with a few rocks and trees lining its edge.\n\n**Key Features:**\n\n* **Rock Formation:** A prominent, tall, and flat-topped rock formation dominates the center of the image.\n* **Road:** A paved road winds its way through the landscape, curving from the bottom-right corner towards the center-left.\n* **Trees and Rocks:** Trees are visible on both sides of the road, with rocks scattered along the left side.\n* **Sky:** The sky above is blue, dotted with white clouds.\n* **Atmosphere:** The overall atmosphere of the","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1753322607,"model":"meta-llama/Llama-4-Maverick-17B-128E-Instruct","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}
```

NOTE: To send a request to a multi-node deployment, target the node which is running `dynamo-run in=http`.

### Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../benchmarks/llm/perf.sh)


## Disaggregation Strategy

The disaggregation strategy controls how requests are distributed between the prefill and decode workers in a disaggregated deployment.

By default, Dynamo uses a `decode first` strategy: incoming requests are initially routed to the decode worker, which then forwards them to the prefill worker in round-robin fashion. The prefill worker processes the request and returns results to the decode worker for any remaining decode operations.

When using KV routing, however, Dynamo switches to a `prefill first` strategy. In this mode, requests are routed directly to the prefill worker, which can help maximize KV cache reuse and improve overall efficiency for certain workloads. Choosing the appropriate strategy can have a significant impact on performance, depending on your use case.

The disaggregation strategy can be set using the `DISAGGREGATION_STRATEGY` environment variable. You can set the strategy before launching your deployment, for example:
```bash
DISAGGREGATION_STRATEGY="prefill_first" ./launch/disagg.sh
```

## KV Cache Transfer in Disaggregated Serving

Dynamo with TensorRT-LLM supports two methods for transferring KV cache in disaggregated serving: UCX (default) and NIXL (experimental). For detailed information and configuration instructions for each method, see the [KV cache transfer guide](./kv-cache-tranfer.md).

## Using Pre-computed Embeddings (Experimental)

Dynamo with TensorRT-LLM supports providing pre-computed embeddings directly in an inference request. This bypasses the need for the model to process an image and generate embeddings itself, which is useful for performance optimization or when working with custom, pre-generated embeddings.

### Enabling the Feature

This is an experimental feature that requires a small patch to your local TensorRT-LLM installation.

1.  **Apply Patches:** You will need to apply two patches to the TensorRT-LLM Python files. These patches enable the `default_multimodal_input_loader` to accept pre-computed embeddings.
    *   Patch 1: [`302b73b`](https://github.com/chang-l/TensorRT-LLM/commit/302b73be5108f58a6795075e5231a31872e42ddd)
    *   Patch 2: [`5b7613b`](https://github.com/chang-l/TensorRT-LLM/commit/5b7613bbc78d830efb7c320a3090c3ef862aa0ab)

    > **Note:** These changes are to Python files only and **do not** require you to recompile TensorRT-LLM. You can apply them by simply replacing the relevant files in your installation.

### How to Use

Once the patches are applied, you can send requests with paths to local embedding files.

-   **Format:** Provide the embedding as part of the `messages` array, using the `image_url` content type.
-   **URL:** The `url` field should contain the absolute or relative path to your embedding file on the local filesystem.
-   **File Types:** Supported embedding file extensions are `.pt`, `.pth`, and `.bin`. Dynamo will automatically detect these extensions.

When a request with a supported embedding file is received, Dynamo will load the tensor from the file and pass it directly to the model for inference, skipping the image-to-embedding pipeline.

### Example Request

Here is an example of how to send a request with a pre-computed embedding file.

```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the content represented by the embeddings"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/path/to/your/embedding.pt"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```

## Supported Multimodal Models

Multimodel models listed [here](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/examples/pytorch/README.md) are supported by dynamo.

## More Example Architectures

- [Llama 4 Maverick Instruct + Eagle Speculative Decoding](./llama4_plus_eagle.md)
