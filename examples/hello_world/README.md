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

## Running the Example

1. Launch all three services using a single command:

```bash
cd /workspace/examples/hello_world
dynamo serve hello_world:Frontend
```

The `dynamo serve` command deploys the entire service graph, automatically handling the dependencies between Frontend, Middle, and Backend services.

2. Send request to frontend using curl:

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "test"
}'
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