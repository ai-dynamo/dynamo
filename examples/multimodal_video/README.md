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

# Multimodal Video Deployment Examples

This directory provides example workflows and reference implementations for deploying a multimodal video model using Dynamo.
The examples are based on the [LLaVA-NeXT-Video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) model.

## Multimodal Aggregated Serving

### Components

- workers: For aggregated serving, we have two workers, [encode_worker](components/encode_worker.py) for video frame encoding and [decode_worker](components/decode_worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: Http endpoint to handle incoming requests.

### Deployment

In this deployment, we have two workers, [encode_worker](components/encode_worker.py) and [decode_worker](components/decode_worker.py).
The encode worker is responsible for encoding the video frames and passing the embeddings to the decode worker via NIXL.
The decode worker then prefills and decodes the prompt, just like the [LLM aggregated serving](../llm/README.md) example.
By separating the encode from the prefill and decode stages, we can have a more flexible deployment and scale the
encode worker independently from the prefill and decode workers if needed.

This figure shows the flow of the deployment:
```

+------+      +-----------+      +------------------+      video url       +---------------+
| HTTP |----->| processor |----->|  decode worker   |--------------------->| encode worker |
|      |<-----|           |<-----|                  |<---------------------|               |
+------+      +-----------+      +------------------+   video frames       +---------------+

```

```bash
cd $DYNAMO_HOME/examples/multimodal_video
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "/tmp/sample-5s_new.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 300,
    "stream": false
  }'
```

You should see a response similar to this:
```
{
  "id": "997c21b8-48d5-4812-a72d-adeb3fd022bb",
  "object": "chat.completion",
  "created": 1749489568,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " The video depicts a busy daytime scene on a four-lane highway with several cars driving in different lanes. The central vehicle is a small red sports car with white and black stripes on the hood and a number \"1\" painted on it, possibly indicating a racing number or sponsorship. It's driving at a relatively high speed down the highway with a red corridor in one lane, passing an orange traffic light and overtaking another vehicle. The driver appears skilled and focused, looking ahead through the left window, with sunglasses on, and appears to be maintaining a safe distance from the car ahead, which belongs to a car company.\n\n\nIn the scene from the perspective of a different vehicle, we see a car with manufacturing company branding and a red and white logo, and a green motorcycle or scooter going past an orange arrow sign with a speed limit of 40, possibly indicating caution-speed or a bend ahead. The big blue sky is cloudy and the conditions suggest it's either dawn or dusk.\n\n\nFrom the perspective of the central vehicle, we can infer that there is an orange truck on the far left, and we're currently parallel to it on the right as it passes, but we can't see the driver's face as the camera angle is angled away. The road is wet, and there's a"
      },
      "finish_reason": "length"
    }
  ]
}
```

## Multimodal Disaggregated serving

### Components

- workers: For disaggregated serving, we have three workers, [encode_worker](components/encode_worker.py) for video frame encoding, [decode_worker](components/decode_worker.py) for decoding, and [prefill_worker](components/prefill_worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the decode worker.
- frontend: Http endpoint to handle incoming requests.

### Deployment

In this deployment, we have three workers, [encode_worker](components/encode_worker.py), [decode_worker](components/decode_worker.py), and [prefill_worker](components/prefill_worker.py).
For the LLaVA-NeXT-Video model, embeddings are only required during the prefill stage. As such, the encode worker is connected directly to the prefill worker.
The encode worker handles video frames and transmits them to the prefill worker via NIXL.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](../llm/README.md) example.

This figure shows the flow of the deployment:
```

+------+      +-----------+      +------------------+      +------------------+      video url       +---------------+
| HTTP |----->| processor |----->|  decode worker   |----->|  prefill worker  |--------------------->| encode worker |
|      |<-----|           |<-----|  (decode worker) |<-----|                  |<---------------------|               |
+------+      +-----------+      +------------------+      +------------------+   video embeddings   +---------------+

```


```bash
cd $DYNAMO_HOME/examples/multimodal_video
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "/tmp/sample-5s_new.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 300,
    "stream": false
  }'
```

You should see a response similar to this:
```
{
  "id": "fc069933-0072-4823-9ba3-ba14cd525250",
  "object": "chat.completion",
  "created": 1749489527,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " The video appears to be a fast-motion recording taken from a car's windshield. The view captures a scene of two red sports cars racing on a nearby road, with the camera zooming in and out slightly to show details such as the cars' tires and headlights. As the view progresses, the red sports cars are seen driving slowly and passing by several other smaller vehicles, showcasing the contrast between the fast-motion effect. The man in the front-right seat of the car seems focused on the road and continues his conversation despite the high speed, maintaining eye contact with the camera while the camera pans up and down the row of seats in the car. The road shifts and a highway sign is visible, although there are cars driving on that side, with trees and a bridge indicated in the background before returning to the focus on the road. The red car passes by another red sports car on its left, mirroring the continuous motion despite speeding up and passing multiple vehicles. The video blankes to a stop followed by drawing the windshield back into view, with the words \"McFADD\" displayed on the bottom right and another car going by in the right lane."
      },
      "finish_reason": "stop"
    }
  ]
}
```
