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

# Deploying Dynamo Inference Graphs to Kubernetes using the Dynamo Cloud Platform

This guide walks you through deploying an inference graph created with the Dynamo SDK onto a Kubernetes cluster using the Dynamo cloud platform and the Dynamo deploy CLI. The Dynamo cloud platform provides a streamlined experience for deploying and managing your inference services.

## Prerequisites

Before proceeding with deployment, ensure you have:

- [Dynamo Python package](../../get_started.md#alternative-setup-manual-installation) installed
- A Kubernetes cluster
- [Dynamo Base Image](../../get_started.md#building-the-dynamo-base-image) built
- [Dynamo cloud platform](dynamo_cloud.md) deployed on your Cluster
- Ubuntu 24.04 as the base image for your services
- Required dependencies:
  - Helm package manager
  - Rust packages and toolchain


## Understanding the Deployment Process

The deployment process involves two main steps:

1. **Local Build (`dynamo build`)**
   - Creates a Dynamo service archive containing:
     - Service code and dependencies
     - Service configuration and metadata
     - Runtime requirements
     - Service graph definition
   - This archive is used as input for the remote build process

2. **Remote Image Build**
   - A `dynamo-image-builder` pod is created in your cluster
   - This pod:
     - Takes the Dynamo service archive
     - Containerizes it using the specified base image
     - Pushes the final container image to your cluster's registry
   - The build process is managed by the Dynamo operator

## Deployment Steps for your Inference Graph

### 1. Configure Environment Variables

First, set up your environment variables for working with Dynamo Cloud. You have two options for accessing the `dynamo-store` service:

#### Option 1: Using Port-Forward (Local Development)
This is the simplest approach for local development and testing:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Set your Kubernetes namespace (must match the namespace where Dynamo cloud is installed)
export KUBE_NS=<your-namespace>

# In a separate terminal, run port-forward to expose the dynamo-store service locally
kubectl port-forward svc/dynamo-store 8080:80 -n $KUBE_NS

# Set DYNAMO_CLOUD to use the local port-forward endpoint
export DYNAMO_CLOUD=http://localhost:8080
```

#### Option 2: Using Ingress/VirtualService (Production)
For production environments, you should use proper ingress configuration:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Set your Kubernetes namespace (must match the namespace where Dynamo cloud is installed)
export KUBE_NS=<your-namespace>

# Set DYNAMO_CLOUD to your externally accessible endpoint
# This could be your Ingress hostname or VirtualService URL
export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com  # Replace with your actual endpoint
```

``` {note}
The `DYNAMO_CLOUD` environment variable is required for all Dynamo deployment commands. Make sure it's set before running any deployment operations.
```

Export the [Dynamo Base Image](../../get_started.md#building-the-dynamo-base-image) as DYNAMO_IMAGE env var from the prerequisites.
```bash
export DYNAMO_IMAGE=<your-registry>/<your-image-name>:<your-tag>
```

### 2. Build the Dynamo Deployment package with your inference graph.

```bash
DYNAMO_TAG=$(dynamo build <your-graph-name> | grep "Successfully built" | awk '{ print $3 }' | sed 's/\.$//')
```

### 3. Deploy your dynamo graph package to Kubernetes

Deploy your service using the Dynamo deployment command:

```bash
# Set your Helm release name
export DEPLOYMENT_NAME=<your-deployment-name>

# Create the deployment
dynamo deployment create $DYNAMO_TAG -n $DEPLOYMENT_NAME
```

Follow the [Hello World example](../../examples/hello_world.md) for more details.

#### Managing Deployments

Once you have deployments running, you can manage them using Dynamo CLI.
Either export the endpoint into the env var `DYNAMO_CLOUD`

```bash
DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com  # If production, replace with your actual endpoint
DYNAMO_CLOUD=http://localhost:8080 # If local port forward
```

or add it as the endpoint with the ` -e` flag to the commands below.

To see a list of all deployments in your namespace:

```bash
dynamo deployment list
```
This command displays a table of all deployments.

To get detailed information about a specific deployment:

```bash
dynamo deployment get $DEPLOYMENT_NAME
```

To update a specific deployment:

```bash
dynamo deployment update $DEPLOYMENT_NAME [--config-file FILENAME] [--env ENV_VAR]
```

To remove a deployment and all its associated resources:

```bash
dynamo deployment delete $DEPLOYMENT_NAME
```

```{warning}
This command permanently deletes the deployment and all associated resources. Make sure you have any necessary backups before proceeding.
```

### 4. Test the Deployment

The deployment process creates several pods:
1. A `dynamo-image-builder` pod for building the container image
2. Service pods prefixed with `$DEPLOYMENT_NAME` once the build is complete

To test your deployment:

```bash
# Forward the service port to localhost
kubectl -n ${KUBE_NS} port-forward svc/${DEPLOYMENT_NAME}-<your-service>3000:3000

# Test the API endpoint (adjust per your service)
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

## Using Kubernetes Secrets for Environment Variables

Dynamo supports securely injecting environment variables from Kubernetes secrets into your deployment. This is only supported when deploying with `--target kubernetes`.

### Creating a Secret

First, create a Kubernetes secret containing your sensitive values:

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic dynamo-env-secrets \
  --from-literal=huggingface.token=$HF_TOKEN \
  --from-literal=another_secret.key=value \
  -n $KUBE_NS
```

### Referencing Secrets in Your Deployment

You can reference secret keys in your deployment using the `--env-from-secret` flag:

- `--env-from-secret HF_TOKEN=huggingface.token` will set the `HF_TOKEN` environment variable from the `huggingface.token` key in the secret.
- `--env-from-secret ANOTHER_SECRET=another_secret.key` will set the `ANOTHER_SECRET` environment variable from the same-named key in the secret.
- You can also mix normal envs: `--env NORMAL_ENV_KEY=value`.

By default, Dynamo will look for a secret named `dynamo-env-secrets`. You can override this with the `--env-secrets-name` flag or the `DYNAMO_ENV_SECRETS` environment variable.

### Example Full Command

```bash
dynamo deploy $DYNAMO_TAG -n $DEPLOYMENT_NAME -f ./configs/agg.yaml \
  --env NORMAL_ENV_KEY=value \
  --env-from-secret HF_TOKEN=huggingface.token \
  --env-from-secret ANOTHER_SECRET=another_secret.key \
  --target kubernetes
```
