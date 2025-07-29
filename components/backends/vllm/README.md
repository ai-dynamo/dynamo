<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LLM Deployment using vLLM

This directory contains a Dynamo vllm engine and reference implementations for deploying Large Language Models (LLMs) in various configurations using vLLM. For Dynamo integration, we leverage vLLM's native KV cache events, NIXL based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

## Use the Latest Release

We recommend using the latest stable release of Dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

---

## Table of Contents
- [Feature Support Matrix](#feature-support-matrix)
- [Quick Start](#quick-start)
- [Single Node Examples](#run-single-node-examples)
- [Advanced Examples](#advanced-examples)
- [Deploy on Kubernetes](#kubernetes-deployment)
- [Configuration](#configuration)

## Feature Support Matrix

### Core Dynamo Features

| Feature | vLLM | Notes |
|---------|------|-------|
| [**Disaggregated Serving**](../../docs/architecture/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | WIP |
| [**KV-Aware Routing**](../../docs/architecture/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../docs/architecture/sla_planner.md) | ✅ |  |
| [**Load Based Planner**](../../docs/architecture/load_planner.md) | 🚧 | WIP |
| [**KVBM**](../../docs/architecture/kvbm_architecture.md) | 🚧 | WIP |

### Large Scale P/D and WideEP Features

| Feature            | vLLM | Notes                                                                 |
|--------------------|------|-----------------------------------------------------------------------|
| **WideEP**         | ✅   | Support for PPLX / DeepEP not verified                                           |
| **DP Rank Routing**| ✅   | Supported via external control of DP ranks |
| **GB200 Support**  | 🚧   | Container functional on main |

## Quick Start

Below we provide a guide that lets you run all of our the common deployment patterns on a single node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Pull or build container

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd like to build your own container from source:

```bash
./container/build.sh --framework VLLM
```

### Run container

```bash
./container/run.sh -it --framework VLLM [--mount-workspace]
```

This includes the specific commit [vllm-project/vllm#19790](https://github.com/vllm-project/vllm/pull/19790) which enables support for external control of the DP ranks.

## Run Single Node Examples

> [!IMPORTANT]
> Below we provide simple shell scripts that run the components for each configuration. Each shell script runs `python3 -m dynamo.frontend` to start the ingress and uses `python3 -m dynamo.vllm` to start the vLLM workers. You can also run each command in separate terminals for better log visibility.

This figure shows an overview of the major components to deploy:

```
+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| dynamo    |----->|   vLLM Worker    |------------>|  vLLM Prefill |
|      |<-----| ingress   |<-----|                  |<------------|    Worker     |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+
```

Note: The above architecture illustrates all the components. The final components that get spawned depend upon the chosen deployment pattern.

### Aggregated Serving

```bash
# requires one gpu
cd components/backends/vllm
bash launch/agg.sh
```

### Aggregated Serving with KV Routing

```bash
# requires two gpus
cd components/backends/vllm
bash launch/agg_router.sh
```

### Disaggregated Serving

```bash
# requires two gpus
cd components/backends/vllm
bash launch/disagg.sh
```

### Disaggregated Serving with KV Routing

```bash
# requires three gpus
cd components/backends/vllm
bash launch/disagg_router.sh
```

### Single Node Data Parallel Attention / Expert Parallelism

This example is not meant to be performant but showcases Dynamo routing to data parallel workers

```bash
# requires four gpus
cd components/backends/vllm
bash launch/dep.sh
```

> [!TIP]
> Run a disaggregated example and try adding another prefill worker once the setup is running! The system will automatically discover and utilize the new worker.

## Advanced Examples

Below we provide a selected list of advanced deployments. Please open up an issue if you'd like to see a specific example!

### Kubernetes Deployment

For Kubernetes deployment, YAML manifests are provided in the `deploy/` directory. These define DynamoGraphDeployment resources for various configurations:

- `agg.yaml` - Aggregated serving
- `agg_router.yaml` - Aggregated serving with KV routing
- `disagg.yaml` - Disaggregated serving
- `disagg_router.yaml` - Disaggregated serving with KV routing
- `disagg_planner.yaml` - Disaggregated serving with [SLA Planner](../../../docs/architecture/sla_planner.md). See [SLA Planner Deployment Guide](../../../docs/guides/dynamo_deploy/sla_planner_deployment.md) for more details.

#### Prerequisites

- **Dynamo Cloud**: Follow the [Quickstart Guide](../../../docs/guides/dynamo_deploy/quickstart.md) to deploy Dynamo Cloud first.

- **Container Images**: We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd prefer to use your own registry, build and push your own image:
  ```bash
  ./container/build.sh --framework VLLM
  # Tag and push to your container registry
  # Update the image references in the YAML files
  ```

- **Pre-Deployment Profiling (if Using SLA Planner)**: Follow the [pre-deployment profiling guide](../../../docs/architecture/pre_deployment_profiling.md) to run pre-deployment profiling. The results will be saved to the `profiling-pvc` PVC and queried by the SLA Planner.

- **Port Forwarding**: After deployment, forward the frontend service to access the API:
  ```bash
  kubectl port-forward deployment/vllm-v1-disagg-frontend-<pod-uuid-info> 8080:8000
  ```

#### Deploy to Kubernetes

Example with disagg:
Export the NAMESPACE  you used in your Dynamo Cloud Installation.

```bash
cd dynamo
cd components/backends/vllm/deploy
kubectl apply -f disagg.yaml -n $NAMESPACE
```

To change `DYN_LOG` level, edit the yaml file by adding

```yaml
...
spec:
  envs:
    - name: DYN_LOG
      value: "debug" # or other log levels
  ...
```

### Testing the Deployment

Send a test request to verify your deployment:

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

## Configuration

vLLM workers are configured through command-line arguments. Key parameters include:

- `--endpoint`: Dynamo endpoint in format `dyn://namespace.component.endpoint`
- `--model`: Model to serve (e.g., `Qwen/Qwen3-0.6B`)
- `--is-prefill-worker`: Enable prefill-only mode for disaggregated serving
- `--metrics-endpoint-port`: Port for publishing KV metrics to Dynamo

See `args.py` for the full list of configuration options and their defaults.

The [documentation](https://docs.vllm.ai/en/v0.9.2/configuration/serve_args.html?h=serve+arg) for the vLLM CLI args points to running 'vllm serve --help' to see what CLI args can be added. We use the same argument parser as vLLM.

## Request Migration

In a [Distributed System](#distributed-system), a request may fail due to connectivity issues between the Frontend and the Backend.

The Frontend will automatically track which Backends are having connectivity issues with it and avoid routing new requests to the Backends with known connectivity issues.

For ongoing requests, there is a `--migration-limit` flag which can be set on the Backend that tells the Frontend how many times a request can be migrated to another Backend should there be a loss of connectivity to the current Backend.

For example,
```bash
python3 -m dynamo.vllm ... --migration-limit=3
```
indicates a request to this model may be migrated up to 3 times to another Backend, before failing the request, should the Frontend detects a connectivity issue to the current Backend.

The migrated request will continue responding to the original request, allowing for a seamless transition between Backends, and a reduced overall request failure rate at the Frontend for enhanced user experience.
