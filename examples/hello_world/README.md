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

This example demonstrates a basic Dynamo inference graph with three components: a frontend HTTP server, a middle processing layer, and a backend inference engine. This simple architecture showcases Dynamo's core concepts and how to compose services together.

## Architecture Overview

The example implements a three-tier architecture:

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

Each component has a specific role:
- **Frontend**: Handles HTTP requests and provides a REST API endpoint at `/generate`
- **Middle**: Processes requests and coordinates between frontend and backend
- **Backend**: Performs the actual inference computation

## Running Locally

### 1. Launch the Services

You can launch all three services using a single command:

```bash
cd /workspace/examples/hello_world
dynamo serve hello_world:Frontend
```

This command will:
1. Start the frontend HTTP server
2. Initialize the middle processing layer
3. Launch the backend inference engine
4. Establish the necessary connections between components

### 2. Test the API

Once the services are running, you can test the API using curl:

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "test"
}'
```

The API will respond with a stream of generated text based on your input.

## Deploying to Kubernetes

There are two ways to deploy the hello world example:
1. Manually using helm charts
2. Using the Dynamo cloud Kubernetes platform and the Dynamo deploy CLI.

### Deploying with helm charts

The instructions for deploying the hello world example using helm charts can be found at [Deploying Dynamo Inference Graphs to Kubernetes using Helm](../../docs/guides/dynamo_deploy.md). The guide covers:

1. Setting up a local Kubernetes cluster with MicroK8s
2. Installing required dependencies like NATS and etcd
3. Building and containerizing the pipeline
4. Deploying using Helm charts
5. Testing the deployment

### Deploying with the Dynamo cloud platform

This example can be deployed to a Kubernetes cluster using Dynamo cloud and the Dynamo deploy CLI.

#### Prerequisites

Before deploying, ensure you have:
- Dynamo CLI installed
- Ubuntu 24.04 as the base image
- Required dependencies:
  - Helm package manager
  - Dynamo SDK and CLI tools
  - Rust packages and toolchain

You must have first followed the instructions in [deploy/dynamo/helm/README.md](../../deploy/dynamo/helm/README.md) to create your Dynamo cloud deployment.

#### Deployment Steps

1. **Login to Dynamo Server**

```bash
export PROJECT_ROOT=$(pwd)
export KUBE_NS=hello-world  # Must match your Kubernetes namespace
export DYNAMO_SERVER=https://${KUBE_NS}.dev.aire.nvidia.com
dynamo server login --api-token TEST-TOKEN --endpoint $DYNAMO_SERVER
```

2. **Build the Dynamo Image**

```bash
export DYNAMO_IMAGE=<dynamo_docker_image_name>
# example dynamo image if you have access to nvcr.io: nvcr.io/nvidian/nim-llm-dev/dynamo-base:cd05fbb91cdeae15efaf56b099b9951db065fd8d-26362190-vllm
cd $PROJECT_ROOT/examples/hello_world
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')
```

3. **Deploy to Kubernetes**

```bash
echo $DYNAMO_TAG
export HELM_RELEASE=ci-hw
dynamo deployment create $DYNAMO_TAG --no-wait -n $HELM_RELEASE
```

To delete an existing Dynamo deployment:

```bash
kubectl delete dynamodeployment $HELM_RELEASE
```

4. **Test the deployment**

Once you create the Dynamo deployment, a pod prefixed with `yatai-dynamonim-image-builder` will begin running. Once it finishes running, it will create the pods necessary. Once the pods prefixed with `$HELM_RELEASE` are up and running, you can test out your example!

```bash
# Forward the service port to localhost
kubectl -n ${KUBE_NS} port-forward svc/${HELM_RELEASE}-frontend 3000:3000

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

For more complex examples, see the [LLM deployment examples](../../examples/llm/README.md)