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

This example can also be deployed to a Kubernetes cluster using Dynamo cloud and the Dynamo deploy CLI.

### Prerequisites

Before deploying, ensure you have:
- Dynamo CLI installed
- Ubuntu 24.04 as the base image
- Required dependencies:
  - Helm package manager
  - Dynamo SDK and CLI tools
  - Rust packages and toolchain

You must have first followed the instructions in [deploy/dynamo/helm/README.md](../../deploy/dynamo/helm/README.md) to create your Dynamo cloud deployment.

### Deployment Steps

1. **Login to Dynamo Server**

```bash
export KUBE_NS=hannahz-hello-world  # Must match your Kubernetes namespace
export DYNAMO_SERVER=https://${KUBE_NS}.dev.aire.nvidia.com
dynamo server login --api-token TEST-TOKEN --endpoint $DYNAMO_SERVER
```

2. **Build the Dynamo Image**

```bash
export DYNAMO_IMAGE=nvcr.io/nvidian/nim-llm-dev/dynamo-base:097fb745a43e85b8c9e5ad0cf217e03290c865e8
cd /workspaces/ai-dynamo/examples/hello_world
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')
```

3. **Deploy to Kubernetes**

```bash
echo $DYNAMO_TAG
dynamo deployment create $DYNAMO_TAG --no-wait -n ci-hw
```

To delete an existing Dynamo deployment:

```bash
kubectl delete dynamodeployment ci-hw
```

## Additional Resources

- For more complex examples, see the [LLM deployment examples](../../examples/llm/README.md)
- Learn about Dynamo's architecture and key features in the [architecture documentation](../../docs/architecture.md)
- Explore the [Dynamo SDK documentation](../../deploy/dynamo/sdk/docs/sdk/README.md) for detailed API references