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
The examples are based on the [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model.

## Multimodal Aggregated Serving

### Components

- workers: For aggregated serving, we have two workers, [encode_worker](components/encode_worker.py) for encoding and [vllm_worker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the vllm worker.
- frontend: Http endpoint to handle incoming requests.

### Deployment

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

## Multimodal Disaggregated serving

### Components

- workers: For disaggregated serving, we have three workers, [encode_worker](components/encode_worker.py) for encoding, [vllm_worker](components/worker.py) for decoding, and [prefill_worker](components/prefill_worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the vllm worker.
- frontend: Http endpoint to handle incoming requests.

### Deployment

In this deployment, we have three workers, [encode_worker](components/encode_worker.py), [vllm_worker](components/worker.py), and [prefill_worker](components/prefill_worker.py).
For the Llava model, embeddings are only required during the prefill stage. As such, the encode worker is connected directly to the prefill worker.
The encode worker handles image encoding and transmits the resulting embeddings to the prefill worker via NATS.
The prefill worker performs the prefilling step and forwards the KV cache to the vllm worker for decoding.
For more details on the roles of the prefill and vllm workers, refer to the [LLM disaggregated serving](../llm/README.md) example.

This figure shows the flow of the deployment:
```

+------+      +-----------+      +------------------+      +------------------+      image url       +---------------+
| HTTP |----->| processor |----->|   vllm worker    |----->|  prefill worker  |--------------------->| encode worker |
|      |<-----|           |<-----|  (decode worker) |<-----|                  |<---------------------|               |
+------+      +-----------+      +------------------+      +------------------+   image embeddings   +---------------+

```


```bash
cd $DYNAMO_HOME/examples/multimodal
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
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
  "image":"http://images.cocodataset.org/val2017/000000324158.jpg",
  "prompt":"Describe the mood and setting of this image in two sentences. What time of day do you think it is?",
  "max_tokens":300
}' | jq
```

You should see a response similar to this:
```
" The image depicts a man moving across a field on a skateboard. The setting appears to be joyful, and this activity suggests that the man is enjoying an outdoor adventure. Additionally, a pet dog is probably accompanying, contributing to the positive mood. The mood and setting of the image appear lively and shoal. The sun is most likely low in the sky, as this would produce a nice daylight."
```
