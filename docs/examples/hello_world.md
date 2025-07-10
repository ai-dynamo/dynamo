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

# Hello World Example: Basic Pipeline

## Overview

This example demonstrates the basic concepts of Dynamo by creating a simple multi-service pipeline. It shows how to:

1. Create and connect multiple Dynamo services
2. Pass data between services using Dynamo's runtime
3. Set up a simple HTTP API endpoint
4. Deploy and interact with a Dynamo service graph

Graph Architecture:

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

Make sure you are running etcd and nats
```bash
sudo systemctl start etcd
sudo systemctl start nats-server
```

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

# Deploy to Kubernetes

You should first deploy the Dynamo Cloud Platform.
If you are a **👤 Dynamo User** first follow the [Quickstart Guide](../guides/dynamo_deploy/quickstart.md).
If you are a **🧑‍💻 Dynamo Contributor** and you have changed the platform code you would have to rebuild the dynamo platform. To do so please look at the [Cloud Guide](../guides/dynamo_deploy/dynamo_cloud.md).

## Deploy your service using a DynamoGraphDeployment CR.

```bash
kubectl apply -f examples/hello_world/deploy/hello_world.yaml -n ${NAMESPACE}
```

## Testing the Deployment

Once the deployment is complete, you can test it using commands below.
Do the port forward in another terminal if needed.

```bash
export DEPLOYMENT_NAME=hello-world
# Forward the pod's port to localhost
kubectl port-forward svc/$DEPLOYMENT_NAME-frontend 8000:8000 -n ${NAMESPACE}
```

```bash
# Test the API endpoint
curl -N -X POST http://localhost:8000/generate \
  -H "accept: text/event-stream" \
  -H "Content-Type: application/json" \
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
