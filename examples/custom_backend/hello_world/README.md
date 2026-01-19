<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This is the simplest Dynamo example demonstrating a basic service using Dynamo's distributed runtime. It showcases the fundamental concepts of creating endpoints and workers in the Dynamo runtime system.

## Architecture

```text
Client (dynamo_worker)
      │
      ▼
┌─────────────┐
│   Backend   │  Dynamo endpoint (/generate)
└─────────────┘
```

## Components

- **Backend**: A Dynamo service with an endpoint that receives text input and streams back greetings for each comma-separated word
- **Client**: A Dynamo worker that connects to and sends requests to the backend service, then prints out the response

## Implementation Details

The example demonstrates:

- **Endpoint Definition**: Using the `@dynamo_endpoint` decorator to create streaming endpoints
- **Worker Setup**: Using the `@dynamo_worker()` decorator to create distributed runtime workers
- **Service Creation**: Creating services and endpoints using the distributed runtime API
- **Streaming Responses**: Yielding data for real-time streaming
- **Client Integration**: Connecting to services and processing streams
- **Logging**: Basic logging configuration with `configure_dynamo_logging`

## Getting Started

### Prerequisites

Dynamo must be installed. No external services are required for local development—the example uses file-based KV storage by default.

### Running the Example

First, start the backend service:
```bash
cd examples/custom_backend/hello_world
DYN_STORE_KV=file python hello_world.py
```

Second, in a separate terminal, run the client:
```bash
cd examples/custom_backend/hello_world
DYN_STORE_KV=file python client.py
```

> **Note**: Setting `DYN_STORE_KV=file` uses file-based storage instead of etcd.
> Both the backend and client must use the same KV backend to discover each other.

The client will connect to the backend service and print the streaming results.

### Expected Output

When running the client, you should see streaming output like:
```text
Hello world!
Hello sun!
Hello moon!
Hello star!
```

## Code Structure

### Backend Service (`hello_world.py`)

- **`content_generator`**: A dynamo endpoint that processes text input and yields greetings
- **`worker`**: A dynamo worker that sets up the service, creates the endpoint, and serves it

### Client (`client.py`)

- **`worker`**: A dynamo worker that connects to the backend service and processes the streaming response

## Deployment to Kubernetes

Note that this a very simple degenerate example which does not demonstrate the standard Dynamo FrontEnd-Backend deployment. The hello-world client is not a web server, it is a one-off function which sends the predefined text "world,sun,moon,star" to the backend. The example is meant to show the HelloWorldWorker. As such you will only see the HelloWorldWorker pod in deployment. The client will run and exit and the pod will not be operational.


Follow the [Quickstart Guide](/docs/kubernetes/README.md) to install Dynamo Kubernetes Platform.
Then deploy to kubernetes using

```bash
export NAMESPACE=<your-namespace>
cd dynamo
kubectl apply -f examples/custom_backend/hello_world/deploy/hello_world.yaml -n ${NAMESPACE}
```

to delete your deployment:

```bash
kubectl delete dynamographdeployment hello-world -n ${NAMESPACE}
```
