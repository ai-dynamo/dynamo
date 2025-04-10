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

# End-to-end deployment of a Dynamo NIM to Kubernetes

## Deployment Paths in Dynamo

Dynamo provides two distinct deployment paths, each serving different purposes:

1. **Manual Deployment with Helm Charts** (`deploy/Kubernetes/`)
   - Used for manually deploying inference graphs to Kubernetes
   - Contains Helm charts and configurations for deploying individual inference pipelines
   - Documentation:
        - [Deploying Dynamo Inference Graphs to Kubernetes using Helm](../Kubernetes/pipeline/README.md)
        - [Dynamo Deploy Guide](../../docs/guides/dynamo_deploy.md)

2. **Dynamo Cloud Platform** (`deploy/dynamo/helm/`)
   - Contains the infrastructure components required for the Dynamo cloud platform
   - Used when deploying with the `dynamo deploy` CLI commands
   - Provides a managed deployment experience
   - This README focuses on setting up this platform infrastructure

Choose the appropriate deployment path based on your needs:
- Use `deploy/Kubernetes/` if you want to manually manage your inference graph deployments
- Use `deploy/dynamo/helm/` if you want to use the Dynamo cloud platform and CLI tools

## Building docker images for Dynamo cloud components

You can build and push Docker images for the Dynamo cloud components (API server, API store, and operator) to any container registry of your choice. Here's how to build each component:

### Prerequisites
- [Earthly](https://earthly.dev/) installed
- Docker installed and running
- Access to a container registry of your choice

### Building and Pushing Images

First, set the required environment variables:
```bash
export CI_REGISTRY_IMAGE=<CONTAINER_REGISTRY>/<ORGANIZATION>
export CI_COMMIT_SHA=<TAG>
```

As a description of the placeholders:
- `<CONTAINER_REGISTRY>/<ORGANIZATION>`: Your container registry and organization name (e.g., `nvcr.io/myorg`, `docker.io/myorg`, etc.)
- `<TAG>`: The tag you want to use for the image (e.g., `latest`, `0.0.1`, etc.)

Note: Make sure you're logged in to your container registry before pushing images. For example:
```bash
docker login <CONTAINER_REGISTRY>
```

You can build each component individually or build all components at once:

#### Option 1: Build All Components at Once
```bash
earthly --push +all-docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

#### Option 2: Build Components Individually

1. **API Store**
```bash
cd deploy/dynamo/api-store
earthly --push +docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

2. **Operator**
```bash
cd deploy/dynamo/operator
earthly --push +docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

## Deploy Dynamo cloud components to Kubernetes

For detailed deployment instructions, please refer to the [Helm deployment guide](./helm/README.md) which walks through installing and configuring the Dynamo cloud components on your Kubernetes cluster.

## Hello World example
See [examples/hello_world/README.md#deploying-to-kubernetes-using-dynamo-cloud-and-dynamo-deploy-cli](../../examples/hello_world/README.md#deploying-to-kubernetes-using-dynamo-cloud-and-dynamo-deploy-cli)