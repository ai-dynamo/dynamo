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

# Multimodal Deployment Examples

This directory provides example workflows and reference implementations for deploying a multimodal model using Dynamo.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Multimodal Aggregated Serving

### Components

- workers: For aggregated serving, we have two workers, [encode_worker](components/encode_worker.py) for encoding and [decode_worker](components/decode_worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have two workers, [encode_worker](components/encode_worker.py) and [decode_worker](components/decode_worker.py).
The encode worker is responsible for encoding the image and passing the embeddings to the decode worker via a combination of NATS and RDMA.
The work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
Its decode worker then prefills and decodes the prompt, just like the [LLM aggregated serving](../llm/README.md) example.
By separating the encode from the prefill and decode stages, we can have a more flexible deployment and scale the
encode worker independently from the prefill and decode workers if needed.

This figure shows the flow of the graph:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --> decode_worker
  decode_worker --> processor
  decode_worker --image_url--> encode_worker
  encode_worker --embeddings--> decode_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
# Serve a LLaVA 1.5 7B model:
dynamo serve graphs.agg:Frontend -f ./configs/agg-llava.yaml
# Serve a Qwen2.5-VL model:
# dynamo serve graphs.agg:Frontend -f ./configs/agg-qwen.yaml
# Serve a Phi3V model:
# dynamo serve graphs.agg:Frontend -f ./configs/agg-phi3v.yaml
```

### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llava-hf/llava-1.5-7b-hf",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 300,
      "temperature": 0.0,
      "stream": false
    }'
```

If serving the example Qwen model, replace `"llava-hf/llava-1.5-7b-hf"` in the `"model"` field with `"Qwen/Qwen2.5-VL-7B-Instruct"`. If serving the example Phi3V model, replace `"llava-hf/llava-1.5-7b-hf"` in the `"model"` field with `"microsoft/Phi-3.5-vision-instruct"`.

You should see a response similar to this:
```json
{"id": "c37b946e-9e58-4d54-88c8-2dbd92c47b0c", "object": "chat.completion", "created": 1747725277, "model": "llava-hf/llava-1.5-7b-hf", "choices": [{"index": 0, "message": {"role": "assistant", "content": " In the image, there is a city bus parked on a street, with a street sign nearby on the right side. The bus appears to be stopped out of service. The setting is in a foggy city, giving it a slightly moody atmosphere."}, "finish_reason": "stop"}]}
```

## Multimodal Disaggregated Serving

### Components

- workers: For disaggregated serving, we have three workers, [encode_worker](components/encode_worker.py) for encoding, [decode_worker](components/decode_worker.py) for decoding, and [prefill_worker](components/prefill_worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have three workers, [encode_worker](components/encode_worker.py), [decode_worker](components/decode_worker.py), and [prefill_worker](components/prefill_worker.py).
For the Llava model, embeddings are only required during the prefill stage. As such, the encode worker is connected directly to the prefill worker.
The encode worker is responsible for encoding the image and passing the embeddings to the prefill worker via a combination of NATS and RDMA.
Its work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](../llm/README.md) example.

This figure shows the flow of the graph:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --> decode_worker
  decode_worker --> processor
  decode_worker --> prefill_worker
  prefill_worker --> decode_worker
  prefill_worker --image_url--> encode_worker
  encode_worker --embeddings--> prefill_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llava-hf/llava-1.5-7b-hf",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 300,
      "temperature": 0.0,
      "stream": false
    }'
```

You should see a response similar to this:
```json
{"id": "c1774d61-3299-4aa3-bea1-a0af6c055ba8", "object": "chat.completion", "created": 1747725645, "model": "llava-hf/llava-1.5-7b-hf", "choices": [{"index": 0, "message": {"role": "assistant", "content": " This image shows a passenger bus traveling down the road near power lines and trees. The bus displays a sign that says \"OUT OF SERVICE\" on its front."}, "finish_reason": "stop"}]}
```

***Note***: disaggregation is currently only confirmed to work with LLaVA. Qwen VL and PhiV are not confirmed to be supported.

## Deployment with Dynamo Operator

These multimodal examples can be deployed to a Kubernetes cluster using [Dynamo Cloud](../../docs/guides/dynamo_deploy/dynamo_cloud.md) and the Dynamo CLI.

### Prerequisites

You must have first followed the instructions in [deploy/cloud/helm/README.md](../../deploy/cloud/helm/README.md) to install Dynamo Cloud on your Kubernetes cluster.

**Note**: The `KUBE_NS` variable in the following steps must match the Kubernetes namespace where you installed Dynamo Cloud. You must also expose the `dynamo-store` service externally. This will be the endpoint the CLI uses to interface with Dynamo Cloud.

### Deployment Steps

For detailed deployment instructions, please refer to the [Operator Deployment Guide](../../docs/guides/dynamo_deploy/operator_deployment.md). The following are the specific commands for the multimodal examples:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Configure environment variables (see operator_deployment.md for details)
export KUBE_NS=dynamo-cloud
export DYNAMO_CLOUD=http://localhost:8080  # If using port-forward
# OR
# export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com  # If using Ingress/VirtualService

# Build the Dynamo base image (see operator_deployment.md for details)
export DYNAMO_IMAGE=<your-registry>/<your-image-name>:<your-tag>

# Build the service
cd $PROJECT_ROOT/examples/multimodal
DYNAMO_TAG=$(dynamo build graphs.agg:Frontend | grep "Successfully built" |  awk '{ print $NF }' | sed 's/\.$//')
# For disaggregated serving:
# DYNAMO_TAG=$(dynamo build graphs.disagg:Frontend | grep "Successfully built" |  awk '{ print $NF }' | sed 's/\.$//')

# Deploy to Kubernetes
export DEPLOYMENT_NAME=multimodal-agg
# For aggregated serving with LLaVA:
dynamo deploy $DYNAMO_TAG -n $DEPLOYMENT_NAME -f ./configs/agg-llava.yaml
# For aggregated serving with Qwen2.5-VL:
# dynamo deploy $DYNAMO_TAG -n $DEPLOYMENT_NAME -f ./configs/agg-qwen.yaml
# For aggregated serving with Phi3V:
# dynamo deploy $DYNAMO_TAG -n $DEPLOYMENT_NAME -f ./configs/agg-phi3v.yaml
# For disaggregated serving:
# export DEPLOYMENT_NAME=multimodal-disagg
# dynamo deploy $DYNAMO_TAG -n $DEPLOYMENT_NAME -f ./configs/disagg.yaml
```

**Note**: To avoid rate limiting from unauthenticated requests to HuggingFace (HF), you can provide your `HF_TOKEN` as a secret in your deployment. See the [operator deployment guide](../../docs/guides/dynamo_deploy/operator_deployment.md#referencing-secrets-in-your-deployment) for instructions on referencing secrets like `HF_TOKEN` in your deployment configuration.

**Note**: Optionally add `--Planner.no-operation=false` at the end of the deployment command to enable the planner component to take scaling actions on your deployment.

### Testing the Deployment

Once the deployment is complete, you can test it. If you have ingress available for your deployment, you can directly call the url returned
in `dynamo deployment get ${DEPLOYMENT_NAME}` and skip the steps to find and forward the frontend pod.

```bash
# Find your frontend pod
export FRONTEND_POD=$(kubectl get pods -n ${KUBE_NS} | grep "${DEPLOYMENT_NAME}-frontend" | sort -k1 | tail -n1 | awk '{print $1}')

# Forward the pod's port to localhost
kubectl port-forward pod/$FRONTEND_POD 8000:8000 -n ${KUBE_NS}

# Test the API endpoint
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/llava-1.5-7b-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "What is in this image?" },
          { "type": "image_url", "image_url": { "url": "http://images.cocodataset.org/test2017/000000155781.jpg" } }
        ]
      }
    ],
    "max_tokens": 300,
    "temperature": 0.0,
    "stream": false
  }'
```

If serving the example Qwen model, replace `"llava-hf/llava-1.5-7b-hf"` in the `"model"` field with `"Qwen/Qwen2.5-VL-7B-Instruct"`. If serving the example Phi3V model, replace `"llava-hf/llava-1.5-7b-hf"` in the `"model"` field with `"microsoft/Phi-3.5-vision-instruct"`.

For more details on managing deployments, testing, and troubleshooting, please refer to the [Operator Deployment Guide](../../docs/guides/dynamo_deploy/operator_deployment.md).
