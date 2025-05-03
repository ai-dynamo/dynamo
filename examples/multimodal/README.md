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

This directory contains examples and reference implementations for deploying a multimodal model with Dynamo.

## Components

- workers: For aggregated serving, we have two workers, [encode_worker](components/encode_worker.py) for encoding and [vllm_worker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the vllm worker.
- frontend: Http endpoint to handle incoming requests.


#### Multimodal Aggregated serving

In this deployment, we have two workers, [encode_worker](components/encode_worker.py) and [vllm_worker](components/worker.py).
The encode worker is responsible for encoding the image and passing the embeddings to the vllm worker via NATS.
The vllm worker then prefills and decodes the prompt, just like the [LLM aggregated serving](../llm/README.md) example.
By separating the encode from the prefill and decode stages, we can have a more flexible deployment and scale the
encode worker independently from the prefill and decode workers if needed.

This figure shows the flow of the deployment:
```

+------+      +-----------+      +------------------+      image url       +---------------+
| HTTP |----->| processor |----->|   vllm worker    |--------------------->| encode worker |
|      |<-----|           |<-----|                  |<---------------------|               |
+------+      +-----------+      +------------------+   image embeddings   +---------------+

```

```bash
cd $DYNAMO_HOME/examples/multimodal
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

### Client

In another terminal:
```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "model":"llava-hf/llava-1.5-7b-hf",
  "image":"http://images.cocodataset.org/test2017/000000155781.jpg",
  "prompt":"Describe the image",
  "max_tokens":300
}' | jq
```

You should see a response similar to this:
```
" The image features a close-up view of the front of a bus, with a prominent neon sign clearly displayed. The bus appears to be slightly past its prime condition, beyond its out-of-service section. Inside the bus, we see a depth of text, with the sign saying \"out of service\". A wide array of windows line the side of the double-decker bus, making its overall appearance quite interesting and vintage."
```
