<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running SGLang with Dynamo

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
- [Dynamo SGLang Integration](#dynamo-sglang-integration)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Single Node Examples](#run-single-node-examples)
- [Multi-Node and Advanced Examples](#advanced-examples)
- [Deploy on SLURM or Kubernetes](#deployment)

## Feature Support Matrix

### Core Dynamo Features

| Feature | SGLang | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../design_docs/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../design_docs/disagg_serving.md#conditional-disaggregation) | 🚧 | WIP [PR](https://github.com/sgl-project/sglang/pull/7730) |
| [**KV-Aware Routing**](../../router/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../planner/sla_planner.md) | ✅ |  |
| [**Multimodal EPD Disaggregation**](multimodal_epd.md) | ✅ |  |
| [**KVBM**](../../kvbm/kvbm_architecture.md) | ❌ | Planned |


## Dynamo SGLang Integration

Dynamo SGLang integrates SGLang engines into Dynamo's distributed runtime, enabling advanced features like disaggregated serving, KV-aware routing, and request migration while maintaining full compatibility with SGLang's engine arguments.

### Argument Handling

Dynamo SGLang uses SGLang's native argument parser, so **most SGLang engine arguments work identically**. You can pass any SGLang argument (like `--model-path`, `--tp`, `--trust-remote-code`) directly to `dynamo.sglang`.

#### Dynamo-Specific Arguments

| Argument | Description | Default | SGLang Equivalent |
|----------|-------------|---------|-------------------|
| `--endpoint` | Dynamo endpoint in `dyn://namespace.component.endpoint` format | Auto-generated based on mode | N/A |
| `--migration-limit` | Max times a request can migrate between workers for fault tolerance. See [Request Migration Architecture](../../fault_tolerance/request_migration.md). | `0` (disabled) | N/A |
| `--dyn-tool-call-parser` | Tool call parser for structured outputs (takes precedence over `--tool-call-parser`) | `None` | `--tool-call-parser` |
| `--dyn-reasoning-parser` | Reasoning parser for CoT models (takes precedence over `--reasoning-parser`) | `None` | `--reasoning-parser` |
| `--use-sglang-tokenizer` | Use SGLang's tokenizer instead of Dynamo's | `False` | N/A |
| `--custom-jinja-template` | Use custom chat template for that model (takes precedence over default chat template in model repo) | `None` | `--chat-template` |

#### Tokenizer Behavior

- **Default (`--use-sglang-tokenizer` not set)**: Dynamo handles tokenization/detokenization via our blazing fast frontend and passes `input_ids` to SGLang
- **With `--use-sglang-tokenizer`**: SGLang handles tokenization/detokenization, Dynamo passes raw prompts

> [!NOTE]
> When using `--use-sglang-tokenizer`, only `v1/chat/completions` is available through Dynamo's frontend.

### Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources for other requests.

#### Cancellation Support Matrix

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ⚠️ | ✅ |

> [!WARNING]
> ⚠️ SGLang backend currently does not support cancellation during remote prefill phase in disaggregated mode.

For more details, see the [Request Cancellation Architecture](../../fault_tolerance/request_cancellation.md) documentation.

## Installation

### Install latest release
We suggest using uv to install the latest release of ai-dynamo[sglang]. You can install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`

<details>
<summary>Expand for instructions</summary>

```bash
# create a virtual env
uv venv --python 3.12 --seed
# install the latest release (which comes bundled with a stable sglang version)
uv pip install "ai-dynamo[sglang]"
```

</details>

### Install editable version for development

<details>
<summary>Expand for instructions</summary>

This requires having rust installed. We also recommend having a proper installation of the cuda toolkit as sglang requires `nvcc` to be available.

```bash
# create a virtual env
uv venv --python 3.12 --seed
# build dynamo runtime bindings
uv pip install maturin
cd $DYNAMO_HOME/lib/bindings/python
maturin develop --uv
cd $DYNAMO_HOME
# installs sglang supported version along with dynamo
# include the prerelease flag to install flashinfer rc versions
uv pip install -e .
# install any sglang version >= 0.5.4.post1
uv pip install "sglang[all]==0.5.4.post1"
```

</details>

### Using docker containers

<details>
<summary>Expand for instructions</summary>

We are in the process of shipping pre-built docker containers that contain installations of DeepEP, DeepGEMM, and NVSHMEM in order to support WideEP and P/D. For now, you can quickly build the container from source with the following command.

```bash
cd $DYNAMO_ROOT
docker build \
  -f container/Dockerfile.sglang-wideep \
  -t dynamo-sglang \
  --no-cache \
  .
```

And then run it using

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-sglang:latest
```

</details>

## Quick Start

Below we provide a guide that lets you run all of our common deployment patterns on a single node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

> [!TIP]
> Each example corresponds to a simple bash script that runs the OpenAI compatible server, processor, and optional router (written in Rust) and LLM engine (written in Python) in a single terminal. You can easily take each command and run them in separate terminals.
>
> Additionally - because we use sglang's argument parser, you can pass in any argument that sglang supports to the worker!


### Aggregated Serving

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/agg.sh
```

### Aggregated Serving with KV Routing

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/agg_router.sh
```

### Aggregated Serving for Embedding Models

Here's an example that uses the [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) model.

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/agg_embed.sh
```

<details>
<summary>Send the following request to verify your deployment:</summary>

```bash
curl localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "Hello, world!"
  }'
```

</details>

### Disaggregated serving

See [SGLang Disaggregation](sglang-disaggregation.md) to learn more about how sglang and dynamo handle disaggregated serving.


```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/disagg.sh
```

### Disaggregated Serving with KV Aware Prefill Routing

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/disagg_router.sh
```

### Disaggregated Serving with Mixture-of-Experts (MoE) models and DP attention

You can use this configuration to test out disaggregated serving with dp attention and expert parallelism on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd $DYNAMO_HOME/components/backends/sglang
./launch/disagg_dp_attn.sh
```

### Testing the Deployment

Send a test request to verify your deployment:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"
    }
    ],
    "stream": true,
    "max_tokens": 30
  }'
```

## Advanced Examples

Below we provide a selected list of advanced examples. Please open up an issue if you'd like to see a specific example!

### Run a multi-node sized model
- **[Run a multi-node model](multinode-examples.md)**

### Large scale P/D disaggregation with WideEP
- **[Run DeepSeek-R1-FP8 on H100s](dsr1-wideep-h100.md)**
- **[Run DeepSeek-R1-FP8 on GB200s](dsr1-wideep-gb200.md)**

### Hierarchical Cache (HiCache)
- **[Enable SGLang Hierarchical Cache (HiCache)](sgl-hicache-example.md)**

### Multimodal Encode-Prefill-Decode (EPD) Disaggregation with NIXL
- **[Run a multimodal model with EPD Disaggregation](multimodal_epd.md)**

## Deployment

We currently provide deployment examples for Kubernetes and SLURM.

## Kubernetes
- **[Deploying Dynamo with SGLang on Kubernetes](../../../components/backends/sglang/deploy/README.md)**

## SLURM
- **[Deploying Dynamo with SGLang on SLURM](../../../components/backends/sglang/slurm_jobs/README.md)**
