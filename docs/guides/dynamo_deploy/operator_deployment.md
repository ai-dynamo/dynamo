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

- [Dynamo Python package](../../get_started.md#installation) installed
- A Kubernetes cluster with the [Dynamo cloud platform](dynamo_cloud.md) installed
- Ubuntu 24.04 as the base image for your services
- Required dependencies:
  - Helm package manager
  - Rust packages and toolchain

You must have first followed the instructions in [deploy/dynamo/helm/README.md](https://github.com/ai-dynamo/dynamo/blob/main/deploy/dynamo/cloud/helm/README.md) to install Dynamo Cloud on your Kubernetes cluster.

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
   - A `yatai-dynamonim-image-builder` pod is created in your cluster
   - This pod:
     - Takes the Dynamo service archive
     - Containerizes it using the specified base image
     - Pushes the final container image to your cluster's registry
   - The build process is managed by the Dynamo operator

## Deployment Steps

### 1. Configure Environment Variables

First, set up your environment variables for working with Dynamo Cloud. You have two options for accessing the `dynamo-store` service:

#### Option 1: Using Port-Forward (Local Development)
This is the simplest approach for local development and testing:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Set your Kubernetes namespace (must match the namespace where Dynamo cloud is installed)
export KUBE_NS=hello-world

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
export KUBE_NS=hello-world

# Set DYNAMO_CLOUD to your externally accessible endpoint
# This could be your Ingress hostname or VirtualService URL
export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com  # Replace with your actual endpoint
```

``` {note}
The `DYNAMO_CLOUD` environment variable is required for all Dynamo deployment commands. Make sure it's set before running any deployment operations.
```

### 2. Build the Dynamo Base Image

Before building your service, you need to ensure the base image is properly set up:

1. For detailed instructions on building and pushing the Dynamo base image, see the [Building the Dynamo Base Image](../../get_started.md#building-the-dynamo-base-image) section in the main README.

2. Export the image from the previous step to your environment.
```bash
# Export the image from the previous step to your environment
export DYNAMO_IMAGE=<your-registry>/<your-image-name>:<your-tag>

# Navigate to your project directory
cd $PROJECT_ROOT/examples/hello_world

# Build the service and capture the tag
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk '{ print $3 }' | sed 's/\.$//')
```

### 3. Deploy to Kubernetes

Deploy your service using the Dynamo deployment command:

```bash
# Set your Helm release name
export DEPLOYMENT_NAME=hello-world

# Create the deployment
dynamo deployment create $DYNAMO_TAG --no-wait -n $DEPLOYMENT_NAME
```

#### Managing Deployments

Once you have deployments running, you can manage them using the following commands:

To see a list of all deployments in your namespace:

```bash
dynamo deployment list
```
This command displays a table of all deployments.

To get detailed information about a specific deployment:

```bash
dynamo deployment get $DEPLOYMENT_NAME
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
1. A `yatai-dynamonim-image-builder` pod for building the container image
2. Service pods prefixed with `$DEPLOYMENT_NAME` once the build is complete

To test your deployment:

```bash
# Forward the service port to localhost
kubectl -n ${KUBE_NS} port-forward svc/${DEPLOYMENT_NAME}-frontend 3000:3000

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

## Expected Output

When you send a request with "test" as input, you'll see how the text flows through each service:

```
Frontend: Middle: Backend: test-mid-back
```

This demonstrates the service pipeline:
1. The Frontend receives "test"
2. The Middle service adds "-mid" to create "test-mid"
3. The Backend service adds "-back" to create "test-mid-back"
