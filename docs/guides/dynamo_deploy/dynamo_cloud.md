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

# 🚀 Dynamo Cloud Kubernetes Platform (Dynamo Deploy)

The Dynamo Cloud platform is a comprehensive solution for deploying and managing Dynamo inference graphs (also referred to as pipelines) in Kubernetes environments. It provides a streamlined experience for deploying, scaling, and monitoring your inference services. You can interface with Dynamo Cloud using the `deploy` subcommand available in the Dynamo CLI (e.g `dynamo deploy`)

## Overview

The Dynamo cloud platform consists of several key components:

- **Dynamo Operator**: A Kubernetes operator that manages the lifecycle of Dynamo inference graphs from build ➡️ deploy.
- **API Store**: Stores and manages service configurations and metadata related to Dynamo deployments. Needs to be exposed externally.
- **Custom Resources**: Kubernetes custom resources for defining and managing Dynamo services

These components work together to provide a seamless deployment experience, handling everything from containerization to scaling and monitoring.

![Dynamo Deploy](../../images/dynamo-deploy.png)

## Prerequisites

Before getting started with the Dynamo cloud platform, ensure you have:

- A Kubernetes cluster (version 1.24 or later)
- [Earthly](https://earthly.dev/) installed for building components
- Docker installed and running
- Access to a container registry (e.g., Docker Hub, NVIDIA NGC, etc.)
- `kubectl` configured to access your cluster
- Helm installed (version 3.0 or later)

## Building Docker Images for Dynamo Cloud Components

The Dynamo cloud platform components need to be built and pushed to a container registry before deployment. You can build these components individually or all at once.

### Setting Up Environment Variables

First, set the required environment variables for building and pushing images:

```bash
# Set your container registry and organization
export CI_REGISTRY_IMAGE=<CONTAINER_REGISTRY>/<ORGANIZATION>
# Set the image tag (e.g., latest, 0.0.1, etc.)
export CI_COMMIT_SHA=<TAG>
```

Where:
- `<CONTAINER_REGISTRY>/<ORGANIZATION>`: Your container registry and organization name
  - Examples: `nvcr.io/myorg`, `docker.io/myorg`
- `<TAG>`: The version tag for your images
  - Examples: `latest`, `0.0.1`, `v1.0.0`

> [!IMPORTANT]
> Make sure you're logged in to your container registry before pushing images:
> ```bash
> docker login <CONTAINER_REGISTRY>
> ```

### Building Components

You have two options for building the components:

#### Option 1: Build All Components at Once

This is the simplest approach and builds and pushes all components in a single command:

```bash
earthly --push +all-docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

#### Option 2: Build Components Individually

If you need to build components separately:

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

## Deploying the Dynamo Cloud Platform

Once you've built and pushed the components, you can deploy the platform to your Kubernetes cluster.

### Prerequisites

Make sure you're in the correct directory:
```bash
cd deploy/dynamo/helm
```

Set your namespace (this will be used for all deployments):
```bash
export KUBE_NS=hello-world    # Change this to your preferred namespace
```

### Deployment Steps

1. **Create Namespace and Set Context**

```bash
# Create a new namespace
kubectl create namespace $KUBE_NS

# Set the namespace as your default context
kubectl config set-context --current --namespace=$KUBE_NS

# [Optional] Create image pull secrets if your registry requires authentication
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=<your-registry> \
  --docker-username=<your-username> \
  --docker-password=<your-password> \
  --namespace=$KUBE_NS
```

2. **Deploy Using the Helm Chart**

Set the required environment variables:
```bash
export NGC_TOKEN=$NGC_API_TOKEN
export NAMESPACE=$KUBE_NS
export CI_COMMIT_SHA=<TAG>  # Use the same tag you used when building the images
export CI_REGISTRY_IMAGE=<CONTAINER_REGISTRY>/<ORGANIZATION>  # Use the same registry/org you used when building the images
export RELEASE_NAME=$KUBE_NS
```

Deploy the platform:
```bash
./deploy.sh
```

3. **Expose Dynamo Cloud Externally**

You must also expose the `dynamo-store` service within the namespace externally. This will be the endpoint the CLI uses to interface with Dynamo Cloud. You might setup an Ingress, use an `ExternalService` with Istio, or simply port-forward. In our docs, we refer to this externally available endpoint as `DYNAMO_CLOUD`.

## Next Steps

After deploying the Dynamo cloud platform, you can:

1. Deploy your first inference graph using the [Dynamo CLI](operator_deployment.md)
2. Deploy Dynamo LLM pipelines to Kubernetes using the [Dynamo CLI](../../../examples/llm/README.md)!
3. Manage your deployments using the Dynamo CLI

For more detailed information about deploying inference graphs, see the [Dynamo Deploy Guide](README.md).