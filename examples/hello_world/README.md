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

## Overview

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


## Unified serve
1. Launch all three services using a single command -

```bash
cd /workspace/examples/hello_world
dynamo serve hello_world:Frontend
```

2. Send request to frontend using curl -

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "test"
}'
```

## Deploying to Kubernetes using Dynamo cloud and Dynamo deploy CLI

**Pre-Requisites**

The dynamo CLI must be installed first and the image must be Ubuntu 24.04-based.

You must have first followed the instructions in [deploy/dynamo/helm/README.md](../../deploy/dynamo/helm/README.md) to create your Dynamo cloud deployment.

**Step 1**: Login to dynamo server

```bash
  export KUBE_NS=hello-world # this must match the name of your Kubernetes namespace
  export DYNAMO_SERVER=https://${KUBE_NS}.dev.aire.nvidia.com
  dynamo server login --api-token TEST-TOKEN --endpoint $DYNAMO_SERVER
```

**Step 2**:  use a framework-less base image and build a dynamo nim

```bash
  export DYNAMO_IMAGE=nvcr.io/nvidian/nim-llm-dev/dynamo-base:097fb745a43e85b8c9e5ad0cf217e03290c865e8
  cd /workspaces/ai-dynamo/examples/hello_world
  DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')
```

**Step 3**: Deploy!

```bash
  echo $DYNAMO_TAG
  dynamo deploy $DYNAMO_TAG --no-wait -n ci-hw
```