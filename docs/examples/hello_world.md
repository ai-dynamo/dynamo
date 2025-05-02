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

This example can be deployed to a Kubernetes cluster using [Dynamo Cloud](../../docs/guides/dynamo_deploy/dynamo_cloud.md) and the Dynamo CLI.

### Prerequisites

You must have first followed the instructions in [deploy/dynamo/helm/README.md](https://github.com/ai-dynamo/dynamo/blob/main/deploy/dynamo/helm/README.md) to create your Dynamo cloud deployment.

### Deployment Steps

For detailed deployment instructions, please refer to the [Operator Deployment Guide](../../docs/guides/dynamo_deploy/operator_deployment.md). The following are the specific commands for the hello world example:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Configure environment variables (see operator_deployment.md for details)
export KUBE_NS=hello-world
export DYNAMO_CLOUD=http://localhost:8080  # If using port-forward
# OR
# export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com  # If using Ingress/VirtualService

# Build the Dynamo base image (see operator_deployment.md for details)
export DYNAMO_IMAGE=<your-registry>/<your-image-name>:<your-tag>

# Build the service
cd $PROJECT_ROOT/examples/hello_world
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk '{ print $3 }' | sed 's/\.$//')

# Deploy to Kubernetes
export DEPLOYMENT_NAME=ci-hw
dynamo deployment create $DYNAMO_TAG -n $DEPLOYMENT_NAME
```

### Testing the Deployment

Once the deployment is complete, you can test it using:

```bash
# Find your frontend pod
export FRONTEND_POD=$(kubectl get pods -n ${KUBE_NS} | grep "${DEPLOYMENT_NAME}-frontend" | sort -k1 | tail -n1 | awk '{print $1}')

# Forward the pod's port to localhost
kubectl port-forward pod/$FRONTEND_POD 8000:8000 -n ${KUBE_NS}

# Test the API endpoint
curl -X 'POST' 'http://localhost:8000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

For more details on managing deployments, testing, and troubleshooting, please refer to the [Operator Deployment Guide](../../docs/guides/dynamo_deploy/operator_deployment.md).

## Expected Output

When you send the request with "test" as input, the response will show how the text flows through each service:

```
Frontend: Middle: Backend: test-mid-back
```

This demonstrates how:
1. The Frontend receives "test"
2. The Middle service adds "-mid" to create "test-mid"
3. The Backend service adds "-back" to create "test-mid-back"