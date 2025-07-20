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

# LLM Deployment Examples using vLLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using vLLM. For Dynamo integration, we leverage vLLM's native KV cache events, NIXL based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture. vLLM supports aggregated, disaggregated, and KV-routed serving patterns.

## Getting Started

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml):

```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build and Run docker

```bash
./container/build.sh
```

```bash
./container/run.sh -it [--mount-workspace]
```

This includes the specific commit [vllm-project/vllm#19790](https://github.com/vllm-project/vllm/pull/19790) which enables support for external control of the DP ranks.

## Run Deployment

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

### Example Architectures

> [!IMPORTANT]
> Below we provide simple shell scripts that run the components for each configuration. Each shell script runs `dynamo run` to start the ingress and uses `python3 main.py` to start the vLLM workers. You can run each command in separate terminals for better log visibility.

#### Aggregated Serving

```bash
# requires one gpu
cd examples/vllm
bash launch/agg.sh
```

#### Aggregated Serving with KV Routing

```bash
# requires two gpus
cd examples/vllm
bash launch/agg_router.sh
```

#### Disaggregated Serving

```bash
# requires two gpus
cd examples/vllm
bash launch/disagg.sh
```

#### Disaggregated Serving with KV Routing

```bash
# requires three gpus
cd examples/vllm
bash launch/disagg_router.sh
```

#### Single Node Data Parallel Attention / Expert Parallelism

This example is not meant to be performant but showcases dynamo routing to data parallel workers

```bash
# requires four gpus
cd examples/vllm
bash launch/dep.sh
```


> [!TIP]
> Run a disaggregated example and try adding another prefill worker once the setup is running! The system will automatically discover and utilize the new worker.

### Kubernetes Deployment

For Kubernetes deployment, YAML manifests are provided in the `deploy/` directory. These define DynamoGraphDeployment resources for various configurations:

- `agg.yaml` - Aggregated serving
- `agg_router.yaml` - Aggregated serving with KV routing
- `disagg.yaml` - Disaggregated serving
- `disagg_router.yaml` - Disaggregated serving with KV routing

#### Prerequisites

- **Dynamo Cloud**: Follow the [Quickstart Guide](../../docs/guides/dynamo_deploy/quickstart.md) to deploy Dynamo Cloud first.

- **Container Images**: The deployment files currently require access to `nvcr.io/nvidian/nim-llm-dev/vllm-runtime`. If you don't have access, build and push your own image:
  ```bash
  ./container/build.sh --framework VLLM
  # Tag and push to your container registry
  # Update the image references in the YAML files
  ```

- **Port Forwarding**: After deployment, forward the frontend service to access the API:
  ```bash
  kubectl port-forward deployment/vllm-v1-disagg-frontend-<pod-uuid-info> 8080:8000
  ```

#### Deploy to Kubernetes

Example with disagg:

```bash
cd ~/dynamo/examples/vllm/deploy
kubectl apply -f disagg.yaml
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
