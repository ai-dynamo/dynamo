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

# Hello World Example

## Overview

This example demonstrates the basic concepts of Dynamo by creating a simple multi-service pipeline. It shows how to:

1. Create and connect multiple Dynamo services
2. Pass data between services using Dynamo's runtime
3. Set up a simple HTTP API endpoint
4. Deploy and interact with a Dynamo service graph

Pipeline Architecture:

```
Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
```

## Component Descriptions

### Frontend Service
- Serves as the entry point for external HTTP requests
- Exposes a `/generate` HTTP API endpoint that clients can call
- Processes incoming text and passes it to the Middle service

### Middle Service
- Acts as an intermediary service in the pipeline
- Receives requests from the Frontend
- Appends "-mid" to the text and forwards it to the Backend

### Backend Service
- Functions as the final service in the pipeline
- Processes requests from the Middle service
- Appends "-back" to the text and yields tokens

## Running the Example Locally

1. Launch all three services using a single command:

```bash
cd /workspace/examples/hello_world
dynamo serve hello_world:Frontend
```

The `dynamo serve` command deploys the entire service graph, automatically handling the dependencies between Frontend, Middle, and Backend services.

2. Send request to frontend using curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "test"
}'
```

## Deploying to and Running the Example in Kubernetes

There are two ways to deploy the hello world example:
1. Manually using helm charts
2. Using the Dynamo cloud Kubernetes platform and the Dynamo deploy CLI.

#### Deploying with helm charts

The instructions for deploying the hello world example using helm charts can be found at [Deploying Dynamo Inference Graphs to Kubernetes using Helm](../../docs/guides/dynamo_deploy.md). The guide covers:

1. Setting up a local Kubernetes cluster with MicroK8s
2. Installing required dependencies like NATS and etcd
3. Building and containerizing the pipeline
4. Deploying using Helm charts
5. Testing the deployment

#### Deploying with the Dynamo cloud platform

This example can be deployed to a Kubernetes cluster using Dynamo cloud and the Dynamo deploy CLI.

##### Prerequisites

Before deploying, ensure you have:
- Dynamo CLI installed
- Ubuntu 24.04 as the base image
- Required dependencies:
  - Helm package manager
  - Dynamo SDK and CLI tools
  - Rust packages and toolchain

You must have first followed the instructions in [deploy/dynamo/helm/README.md](../../deploy/dynamo/helm/README.md) to create your Dynamo cloud deployment.

##### Understanding the Build and Deployment Process

The deployment process involves two distinct build steps:

1. **Local `dynamo build`**: This step creates a Dynamo service archive that contains:
   - Your service code and dependencies
   - Service configuration and metadata
   - Runtime requirements
   - The service graph definition

2. **Remote Image Build**: When you create a deployment, a `yatai-dynamonim-image-builder` pod is created in your cluster. This pod:
   - Takes the Dynamo service archive created in step 1
   - Containerizes it using the specified base image
   - Pushes the final container image to your cluster's registry

##### Deployment Steps

1. **Login to Dynamo Cloud**

```bash
export PROJECT_ROOT=$(pwd)
export KUBE_NS=hello-world  # Must match your Kubernetes namespace
export DYNAMO_CLOUD=https://${KUBE_NS}.dev.aire.nvidia.com
```

The `DYNAMO_CLOUD` environment variable is required for all Dynamo deployment commands. Make sure it's set before running any deployment operations.

2. **Build the Dynamo Base Image**

> [!NOTE]
> For instructions on building and pushing the Dynamo base image, see the [Building the Dynamo Base Image](../../README.md#building-the-dynamo-base-image) section in the main README.

```bash
# Set runtime image name
export DYNAMO_IMAGE=<dynamo_docker_image_name>

# Prepare your project for deployment.
cd $PROJECT_ROOT/examples/hello_world
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk '{ print $3 }' | sed 's/\.$//')
```

3. **Deploy to Kubernetes**

```bash
echo $DYNAMO_TAG
export DEPLOYMENT_NAME=ci-hw
dynamo deployment create $DYNAMO_TAG -n $DEPLOYMENT_NAME
```

4. **Test the deployment**

Once you create the Dynamo deployment, a pod prefixed with `yatai-dynamonim-image-builder` will begin running. Once it finishes running, it will create the pods necessary. Once the pods prefixed with `$HELM_RELEASE` are up and running, you can test out your example!

Find your frontend pod using one of these methods:

```bash
# Method 1: List all pods and find the frontend pod manually
kubectl get pods -n ${KUBE_NS} | grep frontend | cat

# Method 2: Use a label selector to find the frontend pod automatically
export FRONTEND_POD=$(kubectl get pods -n ${KUBE_NS} | grep "${DEPLOYMENT_NAME}-frontend" | sort -k1 | tail -n1 | awk '{print $1}')

# Forward the pod's port to localhost
kubectl port-forward pod/$FRONTEND_POD 8000:8000 -n ${KUBE_NS}

# Note: We forward directly to the pod's port 8000 rather than the service port because the frontend component listens on port 8000 internally.

# Test the API endpoint
curl -X 'POST' 'http://localhost:8000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

## Expected Output

When you send the request with "test" as input, the response will show how the text flows through each service:

```
Frontend: Middle: Backend: test-mid-back
```

This demonstrates how:
1. The Frontend receives "test"
2. The Middle service adds "-mid" to create "test-mid"
3. The Backend service adds "-back" to create "test-mid-back"
