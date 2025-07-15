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

- workers: For aggregated serving, we have two workers, [encode_worker](../../examples/multimodal/encode_worker.py) for encoding and [decode_worker](../../examples/multimodal/decode_worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have two workers, [encode_worker](../../examples/multimodal/components/encode_worker.py) and [decode_worker](../../examples/multimodal/decode_worker.py).
The encode worker is responsible for encoding the image and passing the embeddings to the decode worker via a combination of NATS and RDMA.
The work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
Its decode worker then prefills and decodes the prompt, just like the [LLM aggregated serving](llm_deployment.md) example.
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

- workers: For disaggregated serving, we have three workers, [encode_worker](../../examples/multimodal/encode_worker.py) for encoding, [decode_worker](../../examples/multimodal/decode_worker.py) for decoding, and [prefill_worker](../../examples/multimodal/prefill_worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have three workers, [encode_worker](../../examples/multimodal/encode_worker.py), [decode_worker](../../examples/multimodal/decode_worker.py), and [prefill_worker](../../examples/multimodal/prefill_worker.py).
For the Llava model, embeddings are only required during the prefill stage. As such, the encode worker is connected directly to the prefill worker.
The encode worker is responsible for encoding the image and passing the embeddings to the prefill worker via a combination of NATS and RDMA.
Its work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](llm_deployment.md) example.

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

These multimodal examples can be deployed to a Kubernetes cluster using [Dynamo Cloud](../guides/dynamo_deploy/dynamo_cloud.md) and the Dynamo CLI.

### Prerequisites

You must have first followed the instructions in [deploy/cloud/helm/README.md](../../deploy/cloud/helm/README.md) to install Dynamo Cloud on your Kubernetes cluster.

**Note**: The `KUBE_NS` variable in the following steps must match the Kubernetes namespace where you installed Dynamo Cloud. You must also expose the `dynamo-store` service externally. This will be the endpoint the CLI uses to interface with Dynamo Cloud.

### Deployment Steps

For detailed deployment instructions, please refer to the [Operator Deployment Guide](../guides/dynamo_deploy/operator_deployment.md). The following are the specific commands for the multimodal examples:

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

# TODO: Apply Dynamo graph deployment for the example
```

**Note**: To avoid rate limiting from unauthenticated requests to HuggingFace (HF), you can provide your `HF_TOKEN` as a secret in your deployment. See the [operator deployment guide](../guides/dynamo_deploy/operator_deployment.md#referencing-secrets-in-your-deployment) for instructions on referencing secrets like `HF_TOKEN` in your deployment configuration.

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

For more details on managing deployments, testing, and troubleshooting, please refer to the [Operator Deployment Guide](../guides/dynamo_deploy/operator_deployment.md).

## Multimodal Aggregated Video Serving

This example demonstrates deploying an aggregated multimodal model that can process video inputs.

### Components

- workers: For video serving, we have two workers, [video_encode_worker](../../examples/multimodal/video_encode_worker.py) for decoding video into frames, and [video_decode_worker](../../examples/multimodal/video_decode_worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have two workers, `video_encode_worker` and `video_decode_worker`.
The `video_encode_worker` is responsible for decoding the video into a series of frames. Unlike the image pipeline which generates embeddings, this pipeline passes the raw frames directly to the `video_decode_worker`. This transfer is done efficiently using RDMA.
The `video_decode_worker` then receives these frames, and performs prefill and decode steps with the model. Separating the video processing from the language model inference allows for flexible scaling.

This figure shows the flow of the graph:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --> video_decode_worker
  video_decode_worker --> processor
  video_decode_worker --video_url--> video_encode_worker
  video_encode_worker --frames--> video_decode_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
# Serve a LLaVA-NeXT-Video-7B model:
dynamo serve graphs.agg_video:Frontend -f ./configs/agg_video.yaml
```

### Client

In another terminal:
```bash
curl -X 'POST'   'http://localhost:8000/v1/chat/completions'   -H 'Content-Type: application/json'   -d '{
    "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the video in detail"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 300,
    "stream": false
  }' | jq
```

You should see a response describing the video's content similar to
```json
{
  "id": "b5714626-5889-4bb7-8c51-f3bca65b4683",
  "object": "chat.completion",
  "created": 1749772533,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " Sure! The video features a group of anthropomorphic animals who appear human-like. They're out in a meadow, which is a large, open area covered in grasses, and have given human qualities like speaking and a desire to go on adventures. The animals are seen play-fighting with each other clearly seen glancing at the camera when they sense it, blinking, and Roman the second can be directly heard by the camera reciting the line, \"When the challenge becomes insane, the behavior becomes erratic.\" A white rabbit is the first in shot and he winks the left eye and flips the right ear before shaking with the mouse and squirrel friends on a blurry rock ledge under the sky. At some point, the rabbit turns towards the camera and starts playing with the thing, and there's a distant mountain in the background. Furthermore, a little animal from a tree in the background flies with two rocks, and it's joined by the rest of the group of friends. That outro is an elder turtle in the Ramden musical style saturated with a horn-like thing pattern."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Multimodal Disaggregated Video Serving

This example demonstrates deploying a disaggregated multimodal model that can process video inputs.

### Dependency

Video example relies on `av` package for video preprocessing inside the encode_worker.
Please install `av` inside the dynamo container to enable video example.

`pip install av`

### Components

- workers: For disaggregated video serving, we have three workers, [video_encode_worker](../../examples/multimodal/video_encode_worker.py) for decoding video into frames, [video_decode_worker](../../examples/multimodal/video_decode_worker.py) for decoding, and [video_prefill_worker](../../examples/multimodal/video_prefill_worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: HTTP endpoint to handle incoming requests.

### Graph

In this graph, we have three workers, `video_encode_worker`, `video_decode_worker`, and `video_prefill_worker`.
For the LLaVA-NeXT-Video-7B model, frames are only required during the prefill stage. As such, the `video_encode_worker` is connected directly to the `video_prefill_worker`.
The `video_encode_worker` is responsible for decoding the video into a series of frames and passing them to the `video_prefill_worker` via RDMA.
The `video_prefill_worker` performs the prefilling step and forwards the KV cache to the `video_decode_worker` for decoding.

This figure shows the flow of the graph:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --> video_decode_worker
  video_decode_worker --> processor
  video_decode_worker --> video_prefill_worker
  video_prefill_worker --> video_decode_worker
  video_prefill_worker --video_url--> video_encode_worker
  video_encode_worker --frames--> video_prefill_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
# Serve a LLaVA-NeXT-Video-7B model:
dynamo serve graphs.disagg_video:Frontend -f ./configs/disagg_video.yaml
```

### Client

In another terminal:
```bash
curl -X 'POST'   'http://localhost:8000/v1/chat/completions'   -H 'Content-Type: application/json'   -d '{
    "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the video in detail"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 300,
    "stream": false
  }' | jq
```

You should see a response describing the video's content similar to
```json
{
  "id": "d1d641b1-4daf-48d3-9d06-6a60743b5a42",
  "object": "chat.completion",
  "created": 1749775300,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " The video features two animals in a lush, green outdoor environment. On the ground, there is a white rabbit with big brown eyes, a playful expression, and two antlers. The rabbit is accompanied by a uniquely colored bird with orange pupils, possibly a squirrel or a hamster, sitting on its head. These two animals seem to have embarked on an unlikely journey, flying together in the sky. The backdrop showcases rolling green hills and trees under the pleasant weather. The sky is clear, indicating a beautiful day. The colors and contrast suggest the landscape is during spring or summer, signifying the rabbit and bird could also be engaging in outdoor activities during those seasons. Overall, it's a charming scene depicting an unlikely yet harmonious pair, enjoying a surprise adventure in nature."
      },
      "finish_reason": "stop"
    }
  ]
}
```


## Deploying Multimodal Examples on Kubernetes

This guide will help you quickly deploy and clean up the multimodal example services in Kubernetes.

### Prerequisites

- **Dynamo Cloud** is already deployed in your target Kubernetes namespace.
- You have `kubectl` access to your cluster and the correct namespace set in `$NAMESPACE`.


### Create a secret with huggingface token

```bash
export HF_TOKEN="huggingfacehub token with read permission to models"
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=$HF_TOKEN -n $KUBE_NS || true
```

---

Choose the example you want to deploy or delete. The YAML files are located in `examples/multimodal/deploy/k8s/`.

### Deploy the Multimodal Example

```bash
kubectl apply -f examples/multimodal/deploy/k8s/<Example yaml file> -n $NAMESPACE
```

### Uninstall the Multimodal Example


```bash
kubectl delete -f examples/multimodal/deploy/k8s/<Example yaml file> -n $NAMESPACE
```

### Using a different dynamo container

To customize the container image used in your deployment, you will need to update the manifest before applying it.

You can use [`yq`](https://github.com/mikefarah/yq?tab=readme-ov-file#install), a portable command-line YAML processor.

Please follow the [installation instructions](https://github.com/mikefarah/yq?tab=readme-ov-file#install) for your platform if you do not already have `yq` installed. After installing `yq`, you can generate and apply your manifest as follows:


```bash
export DYNAMO_IMAGE=my-registry/my-image:tag

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(DYNAMO_IMAGE)' $EXAMPLE_FILE > my_example_manifest.yaml

# install the dynamo example
kubectl apply -f my_example_manifest.yaml -n $NAMESPACE

# uninstall the dynamo example
kubectl delete -f my_example_manifest.yaml -n $NAMESPACE

```
