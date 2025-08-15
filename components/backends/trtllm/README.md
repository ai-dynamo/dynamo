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

# LLM Deployment using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

---

## Table of Contents
- [Feature Support Matrix](#feature-support-matrix)
- [Quick Start](#quick-start)
- [Single Node Examples](#single-node-examples)
- [Advanced Examples](#advanced-examples)
- [Disaggregation Strategy](#disaggregation-strategy)
- [KV Cache Transfer](#kv-cache-transfer-in-disaggregated-serving)
- [Client](#client)
- [Benchmarking](#benchmarking)
- [Multimodal Support](#multimodal-support)
- [Performance Sweep](#performance-sweep)

## Feature Support Matrix

### Core Dynamo Features

| Feature | TensorRT-LLM | Notes |
|---------|--------------|-------|
| [**Disaggregated Serving**](../../../docs/architecture/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../../docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | Not supported yet |
| [**KV-Aware Routing**](../../../docs/architecture/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../../docs/architecture/sla_planner.md) | 🚧 | Planned |
| [**Load Based Planner**](../../../docs/architecture/load_planner.md) | 🚧 | Planned |
| [**KVBM**](../../../docs/architecture/kvbm_architecture.md) | 🚧 | Planned |

### Large Scale P/D and WideEP Features

| Feature            | TensorRT-LLM | Notes                                                                 |
|--------------------|--------------|-----------------------------------------------------------------------|
| **WideEP**         | ✅           |                                                                 |
| **DP Rank Routing**| ✅           |                                                                 |
| **GB200 Support**  | ✅           |                                                                 |

## Quick Start

Below we provide a guide that lets you run all of our the common deployment patterns on a single node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build container

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
./container/build.sh --framework trtllm

# On an ARM machine:
./container/build.sh --framework trtllm --platform linux/arm64

# Build the container with the default experimental TensorRT-LLM commit
# WARNING: This is for experimental feature testing only.
# The container should not be used in a production environment.
./container/build.sh --framework trtllm --use-default-experimental-tensorrtllm-commit
```

### Run container

```bash
./container/run.sh --framework trtllm -it
```

## Single Node Examples

> [!IMPORTANT]
> Below we provide some simple shell scripts that run the components for each configuration. Each shell script is simply running the `python3 -m dynamo.frontend <args>` to start up the ingress and using `python3 -m dynamo.trtllm <args>` to start up the workers. You can easily take each command and run them in separate terminals.

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

### Aggregated
```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/agg.sh
```

### Aggregated with KV Routing
```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/agg_router.sh
```

### Disaggregated

> [!IMPORTANT]
> Disaggregated serving supports two strategies for request flow: `"prefill_first"` and `"decode_first"`. By default, the script below uses the `"decode_first"` strategy, which can reduce response latency by minimizing extra hops in the return path. You can switch strategies by setting the `DISAGGREGATION_STRATEGY` environment variable.

```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/disagg.sh
```

### Disaggregated with KV Routing

> [!IMPORTANT]
> Disaggregated serving with KV routing uses a "prefill first" workflow by default. Currently, Dynamo supports KV routing to only one endpoint per model. In disaggregated workflow, it is generally more effective to route requests to the prefill worker. If you wish to use a "decode first" workflow instead, you can simply set the `DISAGGREGATION_STRATEGY` environment variable accordingly.

```bash
cd $DYNAMO_HOME/components/backends/trtllm
./launch/disagg_router.sh
```

### Aggregated with Multi-Token Prediction (MTP) and DeepSeek R1
```bash
cd $DYNAMO_HOME/components/backends/trtllm

export AGG_ENGINE_ARGS=./engine_configs/deepseek_r1/mtp/mtp_agg.yaml
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"
# nvidia/DeepSeek-R1-FP4 is a large model
export MODEL_PATH="nvidia/DeepSeek-R1-FP4"
./launch/agg.sh
```

Notes:
- MTP is only available within the container built with the experimental TensorRT-LLM commit. Please add --use-default-experimental-tensorrtllm-commit to the arguments of the build.sh script.

  Example: `./container/build.sh --framework trtllm --use-default-experimental-tensorrtllm-commit`

- There is a noticeable latency for the first two inference requests. Please send warm-up requests before starting the benchmark.
- MTP performance may vary depending on the acceptance rate of predicted tokens, which is dependent on the dataset or queries used while benchmarking. Additionally, `ignore_eos` should generally be omitted or set to `false` when using MTP to avoid speculating garbage outputs and getting unrealistic acceptance rates.

## Advanced Examples

Below we provide a selected list of advanced examples. Please open up an issue if you'd like to see a specific example!

### Multinode Deployment

For comprehensive instructions on multinode serving, see the [multinode-examples.md](./multinode/multinode-examples.md) guide. It provides step-by-step deployment examples and configuration tips for running Dynamo with TensorRT-LLM across multiple nodes. While the walkthrough uses DeepSeek-R1 as the model, you can easily adapt the process for any supported model by updating the relevant configuration files. You can see [Llama4+eagle](./llama4_plus_eagle.md) guide to learn how to use these scripts when a single worker fits on the single node.

### Speculative Decoding
- **[Llama 4 Maverick Instruct + Eagle Speculative Decoding](./llama4_plus_eagle.md)**

### Kubernetes Deployment

For complete Kubernetes deployment instructions, configurations, and troubleshooting, see [TensorRT-LLM Kubernetes Deployment Guide](deploy/README.md).

### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which is running `python3 -m dynamo.frontend <args>`.

### Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../../benchmarks/llm/perf.sh)


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


## Request Migration

You can enable [request migration](../../../docs/architecture/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
python3 -m dynamo.trtllm ... --migration-limit=3
```

This allows a request to be migrated up to 3 times before failing. See the [Request Migration Architecture](../../../docs/architecture/request_migration.md) documentation for details on how this works.

## Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which is running `python3 -m dynamo.frontend <args>`.

## Benchmarking

To benchmark your deployment with GenAI-Perf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../../benchmarks/llm/perf.sh)

## Multimodal support

TRTLLM supports multimodal models with dynamo. You can provide multimodal inputs in the following ways:

- By sending image URLs
- By providing paths to pre-computed embedding files

Please note that you should provide **either image URLs or embedding file paths** in a single request.

### Aggregated

Here are quick steps to launch Llama-4 Maverick BF16 in aggregated mode
```bash
cd $DYNAMO_HOME/components/backends/trtllm

export AGG_ENGINE_ARGS=./engine_configs/multinode/agg.yaml
export SERVED_MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
export MODEL_PATH="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
./launch/agg.sh
```
### Example Requests

#### With Image URL

Below is an example of an image being sent to `Llama-4-Maverick-17B-128E-Instruct` model

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

### Disaggregated

Here are quick steps to launch in disaggregated mode.

The following is an example of launching a model in disaggregated mode. While this example uses `Qwen/Qwen2-VL-7B-Instruct`, you can adapt it for other models by modifying the environment variables for the model path and engine configurations.
```bash
cd $DYNAMO_HOME/components/backends/trtllm

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen2-VL-7B-Instruct"}
export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"decode_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"engine_configs/multimodal/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"engine_configs/multimodal/decode.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

./launch/disagg.sh
```

For a large model like `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, a multi-node setup is required for disaggregated serving, while aggregated serving can run on a single node. This is because the model with a disaggregated configuration is too large to fit on a single node's GPUs. For instance, running this model in disaggregated mode requires a setup of 2 nodes with 8xH200 GPUs or 4 nodes with 4xGB200 GPUs.

In general, disaggregated serving can run on a single node, provided the model fits on the GPU. The multi-node requirement in this example is specific to the size and configuration of the `meta-llama/Llama-4-Maverick-17B-128E-Instruct` model.

To deploy `Llama-4-Maverick-17B-128E-Instruct` in disaggregated mode, you will need to follow the multi-node setup instructions, which can be found [here](multinode/multinode-multimodal-example.md).

### Using Pre-computed Embeddings (Experimental)

Dynamo with TensorRT-LLM supports providing pre-computed embeddings directly in an inference request. This bypasses the need for the model to process an image and generate embeddings itself, which is useful for performance optimization or when working with custom, pre-generated embeddings.

#### Enabling the Feature

This is an experimental feature that requires using a specific TensorRT-LLM commit.
To enable it build the dynamo container with the `--tensorrtllm-commit` flag, followed by the commit hash:

```bash
./container/build.sh --framework trtllm --tensorrtllm-commit b4065d8ca64a64eee9fdc64b39cb66d73d4be47c
```

#### How to Use

Once the container is built, you can send requests with paths to local embedding files.

-   **Format:** Provide the embedding as part of the `messages` array, using the `image_url` content type.
-   **URL:** The `url` field should contain the absolute or relative path to your embedding file on the local filesystem.
-   **File Types:** Supported embedding file extensions are `.pt`, `.pth`, and `.bin`. Dynamo will automatically detect these extensions.

When a request with a supported embedding file is received, Dynamo will load the tensor from the file and pass it directly to the model for inference, skipping the image-to-embedding pipeline.

#### Example Request

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

### Supported Multimodal Models

Multimodel models listed [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/inputs/utils.py#L221) are supported by dynamo.

## Performance Sweep

For detailed instructions on running comprehensive performance sweeps across both aggregated and disaggregated serving configurations, see the [TensorRT-LLM Benchmark Scripts for DeepSeek R1 model](./performance_sweeps/README.md). This guide covers recommended benchmarking setups, usage of provided scripts, and best practices for evaluating system performance.
