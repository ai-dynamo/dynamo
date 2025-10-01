<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running SGLang with Dynamo

This directory contains an SGLang component for Dynamo and reference implementations for deploying Large Language Models (LLMs) in various configurations using SGLang. SGLang internally uses ZMQ to communicate between the ingress and the engine processes. For Dynamo, we leverage the runtime to communicate directly with the engine processes and handle ingress and pre/post processing on our end.

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
- [Quick Start](#quick-start)
- [Single Node Examples](#run-single-node-examples)
- [Multi-Node and Advanced Examples](#advanced-examples)
- [Deploy on SLURM or Kubernetes](#deployment)

## Feature Support Matrix

### Core Dynamo Features

| Feature | SGLang | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../../docs/architecture/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../../docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | WIP [PR](https://github.com/sgl-project/sglang/pull/7730) |
| [**KV-Aware Routing**](../../../docs/architecture/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../../docs/architecture/sla_planner.md) | ✅ |  |
| [**Load Based Planner**](../../../docs/architecture/load_planner.md) | ❌ | Planned |
| [**KVBM**](../../../docs/architecture/kvbm_architecture.md) | ❌ | Planned |

### Large Scale P/D and WideEP Features

| Feature             | SGLang | Notes                                                        |
|---------------------|--------|--------------------------------------------------------------|
| **WideEP**          | ✅     | Full support on H100s/GB200                                  |
| **DP Rank Routing** | 🚧     | Direct routing supported. Dynamo KV router does not router to DP worker |
| **GB200 Support**   | ✅     |                                                              |


## Dynamo SGLang Integration

Dynamo SGLang integrates SGLang engines into Dynamo's distributed runtime, enabling advanced features like disaggregated serving, KV-aware routing, and request migration while maintaining full compatibility with SGLang's engine arguments.

### Argument Handling

Dynamo SGLang uses SGLang's native argument parser, so **most SGLang engine arguments work identically**. You can pass any SGLang argument (like `--model-path`, `--tp`, `--trust-remote-code`) directly to `dynamo.sglang`.

#### Dynamo-Specific Arguments

| Argument | Description | Default | SGLang Equivalent |
|----------|-------------|---------|-------------------|
| `--endpoint` | Dynamo endpoint in `dyn://namespace.component.endpoint` format | Auto-generated based on mode | N/A |
| `--migration-limit` | Max times a request can migrate between workers | `0` (disabled) | N/A |
| `--dyn-tool-call-parser` | Tool call parser for structured outputs (takes precedence over `--tool-call-parser`) | `None` | `--tool-call-parser` |
| `--dyn-reasoning-parser` | Reasoning parser for CoT models (takes precedence over `--reasoning-parser`) | `None` | `--reasoning-parser` |
| `--use-sglang-tokenizer` | Use SGLang's tokenizer instead of Dynamo's | `False` | N/A |
| `--custom-jinja-template` | Use custom chat template for that model (takes precedence over default chat template in model repo) | `None` | `--chat-template` |

#### Tokenizer Behavior

- **Default (`--use-sglang-tokenizer` not set)**: Dynamo handles tokenization and passes `input_ids` to SGLang
- **With `--use-sglang-tokenizer`**: SGLang handles tokenization, Dynamo passes raw prompts

> **Note**: When using `--use-sglang-tokenizer`, only `v1/chat/completions` endpoints are available through Dynamo's frontend.

## SGLang Quick Start

Below we provide a guide that lets you run all of our common deployment patterns on a single node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Install `ai-dynamo[sglang]`

#### Install latest release
We suggest using uv to install the latest release of ai-dynamo[sglang]. You can install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# create a virtual env
uv venv --python 3.12 --seed
# install the latest release
uv pip install "ai-dynamo[sglang]"
```

#### Installing editable version for development

<details>
<summary>Instructions</summary>

This requires having rust installed. We also recommend having a proper installation of the cuda toolkit as sglang requires `nvcc` to be available.

```bash
# create a virtual env
uv venv --python 3.12 --seed
# build dynamo runtime bindings
uv pip install maturin
cd $DYNAMO_HOME/lib/bindings/python
maturin develop --uv
cd $DYNAMO_HOME
uv pip install .
export PYTHONPATH="${PYTHONPATH}:$(pwd)/components/backends/sglang/src"
# install target sglang version (you can choose any version)
# we include the prerelease flag in order to install flashinfer rc versions
uv pip install --prerelease=allow sglang[all]==0.4.9.post6
```

</details>

#### Using prebuilt docker containers

<details>
<summary>Instructions</summary>

```bash
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:my-tag
```

</details>

#### Building docker container from source

<details>
<summary>Instructions</summary>

```bash
./container/build.sh --framework sglang
# run container using prebuild wheel
./container/run.sh --framework sglang -it
# mount workspace for development
./container/run.sh --framework sglang --mount-workspace
```

</details>

## Run Single Node Examples

> [!IMPORTANT]
> Each example corresponds to a simple bash script that runs the OpenAI compatible server, processor, and optional router (written in Rust) and LLM engine (written in Python) in a single terminal. You can easily take each command and run them in separate terminals.
>
> Additionally - because we use sglang's argument parser, you can pass in any argument that sglang supports to the worker!


### Aggregated Serving

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/agg.sh
```

### Aggregated Serving with KV Routing

> [!NOTE]
> Until sglang releases a version > v0.5.0rc0, you will have to install from source to use kv_routing. You can do this by running `git clone https://github.com/sgl-project/sglang.git && cd sglang && uv pip install -e "python[all]"`. We will update this section once sglang releases a newer version.

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/agg_router.sh
```

### Disaggregated serving

<details>
<summary>Under the hood: SGLang Load Balancer vs Dynamo Discovery</summary>

SGLang uses a mini load balancer to route requests to handle disaggregated serving. The load balancer functions as follows:

1. The load balancer receives a request from the client
2. A random `(prefill, decode)` pair is selected from the pool of available workers
3. Request is sent to both `prefill` and `decode` workers via asyncio tasks
4. Internally disaggregation is done from prefill -> decode

Because Dynamo has a discovery mechanism, we do not use a load balancer. Instead, we first route to a random prefill worker, select a random decode worker, and then send the request to both. Internally, SGLang's bootstrap server (which is a part of the `tokenizer_manager`) is used in conjuction with NIXL to handle the kv transfer.

</details>

> [!IMPORTANT]
> Disaggregated serving in SGLang currently requires each worker to have the same tensor parallel size [unless you are using an MLA based model](https://github.com/sgl-project/sglang/pull/5922)

```bash
cd $DYNAMO_HOME/components/backends/sglang
./launch/disagg.sh
```

### Disaggregated Serving with Mixture-of-Experts (MoE) models and DP attention

You can use this configuration to test out disaggregated serving with dp attention and expert parallelism on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd $DYNAMO_HOME/components/backends/sglang
./launch/disagg_dp_attn.sh
```

When using MoE models, you can also use the our implementation of the native SGLang endpoints to record expert distribution data. The `disagg_dp_attn.sh` script automatically sets up the SGLang HTTP server, the environment variable that controls the expert distribution recording directory, and sets up the expert distribution recording mode to `stat`. You can learn more about expert parallelism load balancing [here](docs/expert-distribution-eplb.md).

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

## Request Migration

You can enable [request migration](../../../docs/architecture/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
python3 -m dynamo.sglang ... --migration-limit=3
```

This allows a request to be migrated up to 3 times before failing. See the [Request Migration Architecture](../../../docs/architecture/request_migration.md) documentation for details on how this works.

## Advanced Examples

Below we provide a selected list of advanced examples. Please open up an issue if you'd like to see a specific example!

### Run a multi-node sized model
- **[Run a multi-node model](docs/multinode-examples.md)**

### Large scale P/D disaggregation with WideEP
- **[Run DeepSeek-R1 on 104+ H100s](docs/dsr1-wideep-h100.md)**
- **[Run DeepSeek-R1-FP8 on GB200s](docs/dsr1-wideep-gb200.md)**

### Hierarchical Cache (HiCache)
- **[Enable SGLang Hierarchical Cache (HiCache)](docs/sgl-hicache-example.md)**

## Deployment

We currently provide deployment examples for Kubernetes and SLURM.

## Kubernetes
- **[Deploying Dynamo with SGLang on Kubernetes](deploy/README.md)**

## SLURM
- **[Deploying Dynamo with SGLang on SLURM](slurm_jobs/README.md)**
