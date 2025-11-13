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

This directory provides example workflows and reference implementations for deploying a multimodal model using Dynamo and vLLM v1.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Multimodal Aggregated Serving

### Components

- workers: For aggregated serving, we have two workers, [VllmEncodeWorker](components/encode_worker.py) for encoding and [VllmPDWorker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the VllmEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have two workers, [VllmEncodeWorker](components/encode_worker.py) and [VllmPDWorker](components/worker.py).
The VllmEncodeWorker is responsible for encoding the image and passing the embeddings to the VllmPDWorker via a combination of NATS and RDMA.
The work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
Its VllmPDWorker then prefills and decodes the prompt, just like the [LLM aggregated serving](../../docs/backends/vllm/README.md) example.
By separating the encode from the prefill and decode stages, we can have a more flexible deployment and scale the
VllmEncodeWorker independently from the prefill and decode workers if needed.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> encode_worker
  encode_worker --> processor
  encode_worker --embeddings--> pd_worker
  pd_worker --> encode_worker
```

***Note*** Aggregated serving supports LLaVA 1.5 7B and Qwen2.5-VL-7B-Instruct today. Phi3V support will be added in the future. Disaggregated serving is currently only confirmed for LLaVA (see note below).

```bash
cd $DYNAMO_HOME/examples/multimodal
# Serve a LLaVA 1.5 7B model:
bash launch/agg.sh --model llava-hf/llava-1.5-7b-hf
# Serve a Qwen2.5-VL model:
bash launch/agg.sh --model Qwen/Qwen2.5-VL-7B-Instruct
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

If serving the example Qwen model, replace `"llava-hf/llava-1.5-7b-hf"` in the `"model"` field with `"Qwen/Qwen2.5-VL-7B-Instruct"`.

You should see a response similar to this:
```json
{"id": "c37b946e-9e58-4d54-88c8-2dbd92c47b0c", "object": "chat.completion", "created": 1747725277, "model": "llava-hf/llava-1.5-7b-hf", "choices": [{"index": 0, "message": {"role": "assistant", "content": " In the image, there is a city bus parked on a street, with a street sign nearby on the right side. The bus appears to be stopped out of service. The setting is in a foggy city, giving it a slightly moody atmosphere."}, "finish_reason": "stop"}]}
```

## Multimodal Disaggregated Serving

### Components

- workers: For disaggregated serving, we have three workers, [VllmEncodeWorker](components/encode_worker.py) for encoding, [VllmDecodeWorker](components/worker.py) for decoding, and [VllmPDWorker](components/worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the VllmEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have three workers, [VllmEncodeWorker](components/encode_worker.py), [VllmDecodeWorker](components/worker.py), and [VllmPDWorker](components/worker.py).
For the Llava model, embeddings are only required during the prefill stage. As such, the VllmEncodeWorker is connected directly to the prefill worker.
The VllmEncodeWorker is responsible for encoding the image and passing the embeddings to the prefill worker via a combination of NATS and RDMA.
Its work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](../../docs/backends/vllm/README.md) example.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> encode_worker
  encode_worker --> processor
  encode_worker --embeddings--> prefill_worker
  prefill_worker --> encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/disagg.sh --model llava-hf/llava-1.5-7b-hf
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

## Llama 4 family Serving

The family of Llama 4 models is natively multimodal, however, different
from Llava, they do not directly consume image embedding as input
(see the [support metrics](https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation_1)
from vLLM for the types of multi-modal inputs supported by the model).
Therefore, encoder worker will not be used in the following example and the
encoding will be done along side with prefill.

`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` will be used as an example
for the content below. And the system will be H100x8 which can hold one instance
of the model per node.

### Multimodal Aggregated Serving

#### Components

- workers: For aggregated serving, we have one worker, [VllmPDWorker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the VllmPDWorker.
- frontend: HTTP endpoint to handle incoming requests.

#### Workflow

In this workflow, we have [VllmPDWorker](components/worker.py) which will encode the image, prefill and decode the prompt, just like the [LLM aggregated serving](../../docs/backends/vllm/README.md) example.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> pd_worker
  pd_worker --> processor
```

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/agg_llama.sh
```

#### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
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
{"id": "b8f060fa95584e34b9204eaba7b105cc", "object": "chat.completion", "created": 1752706281, "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The image depicts a street scene with a trolley bus as the central focus. The trolley bus is positioned on the left side of the road, facing the camera, and features a white and yellow color scheme. A prominent sign on the front of the bus reads \"OUT OF SERVICE\" in orange letters.\n\n**Key Elements:**\n\n* **Trolley Bus:** The bus is the main subject of the image, showcasing its distinctive design and color.\n* **Sign:** The \"OUT OF SERVICE\" sign is clearly visible on the front of the bus, indicating its current status.\n* **Street Scene:** The surrounding environment includes trees, buildings, and power lines, creating a sense of context and atmosphere.\n* **Lighting:** The image is characterized by a misty or foggy quality, with soft lighting that adds to the overall ambiance.\n\n**Overall Impression:**\n\nThe image presents a serene and somewhat melancholic scene, with the out-of-service trolley bus serving as a focal point. The misty atmosphere and soft lighting contribute to a dreamy or nostalgic feel, inviting the viewer to reflect on the scene."}, "finish_reason": "stop"}]}
```

### Multimodal Disaggregated Serving

#### Components

- workers: For disaggregated serving, we have two workers, [VllmDecodeWorker](components/worker.py) for decoding, and [VllmPDWorker](components/worker.py) for encoding and prefilling.
- processor: Tokenizes the prompt and passes it to the VllmPDWorker.
- frontend: HTTP endpoint to handle incoming requests.

#### Workflow

In this workflow, we have two workers, [VllmDecodeWorker](components/worker.py), and [VllmPDWorker](components/worker.py).
The prefill worker performs the encoding and prefilling steps and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](/docs/backends/vllm/README.md) example.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> prefill_worker
  prefill_worker --> processor
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/disagg_llama.sh --head-node

# On a separate node that has finished standard dynamo setup, i.e.
# the worker node needs NATS_SERVER and ETCD_ENDPOINTS environment variables
# pointing to the head node's external IP address for distributed coordination
cd $DYNAMO_HOME/examples/multimodal
bash launch/disagg_llama.sh
```

#### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
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
{"id": "6cc99123ad6948d685b8695428238d4b", "object": "chat.completion", "created": 1752708043, "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "choices": [{"index": 0, "message": {"role": "assistant", "content": "The image depicts a street scene with a trolley bus as the central focus. The trolley bus is positioned on the left side of the road, facing the camera, and features a white and yellow color scheme. A prominent sign on the front of the bus reads \"OUT OF SERVICE\" in orange letters.\n\n**Key Elements:**\n\n* **Trolley Bus:** The bus is the main subject of the image, showcasing its distinctive design and color.\n* **Sign:** The \"OUT OF SERVICE\" sign is clearly visible on the front of the bus, indicating its current status.\n* **Street Scene:** The surrounding environment includes trees, buildings, and power lines, creating a sense of context and atmosphere.\n* **Lighting:** The image is characterized by a misty or foggy quality, with soft lighting that adds to the overall mood.\n\n**Overall Impression:**\n\nThe image presents a serene and somewhat melancholic scene, with the out-of-service trolley bus serving as a focal point. The misty atmosphere and soft lighting contribute to a contemplative ambiance, inviting the viewer to reflect on the situation."}, "finish_reason": "stop"}]}
```

## Multimodal Aggregated Video Serving

This example demonstrates deploying an aggregated multimodal model that can process video inputs.

### Components

- workers: For video serving, we use the [VideoEncodeWorker](components/video_encode_worker.py) for decoding video into frames, and send the frames to [VllmPDWorker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the VideoEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have two workers, [VideoEncodeWorker](components/video_encode_worker.py) and [VllmPDWorker](components/worker.py).
The VideoEncodeWorker is responsible for decoding the video into a series of frames. Unlike the image pipeline which generates embeddings,
this pipeline passes the raw frames directly to the VllmPDWorker via a combination of NATS and RDMA.
Its VllmPDWorker then prefills and decodes the prompt, just like the [LLM aggregated serving](../../docs/backends/vllm/README.md) example.
By separating the video processing from the prefill and decode stages, we can have a more flexible deployment and scale the
VideoEncodeWorker independently from the prefill and decode workers if needed.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --video_url--> video_encode_worker
  video_encode_worker --> processor
  video_encode_worker --frames--> pd_worker
  pd_worker --> video_encode_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/video_agg.sh
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
  "id": "7587e7d152014bae8e5c4e25f9fda0ed",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": " The video takes us away to a lively world of wildlife and natural beauty, featuring a white rabbit in a vibrant forest setting. At the beginning of the clip, the white rabbit is seen standing on a rock, facing towards the right side of the frame, with bushes and trees in the backdrop. The rabbit appears to be alert, given its ears are up and its ears perked in the air. As the clip progresses, the movement of the rabbit brings it around a tree, where its legs are partially hidden by the dense vegetation. It then sits down and grooms its fur, a behavior that suggests it is comfortable in its surroundings. \n\nThe scene then switches to a close-up shot of the rabbit, giving us a better view of its features and expressions. In this camera angle, the rabbit appears more dynamic and alert, with its breathing more visible, signaling its health and well-being. The camera pans out, and we see the rabbit heading towards the left side of the screen, possibly curious or hunting for food, with its ears perked up again. The lush greenery of the forest unfolds in the background, adding to the feeling of a wild and thriving environment.\n\n\nThe rabbit, upturned slightly while walking, finds a pile of dirt and rocks and sits there, fully clothed, perhaps taking a break from its exploration. There's a mention of a blue bird that appears to perch atop a log, adding a touch of whimsy to the scene. Lastly, the rabbit is observed relaxing on the rocks, resting comfortably, and looking off to the right side—a moment of tranquility in a bustling ecosystem. Throughout the clip, the rabbit's outfit remains the same, allowing for a clear focus on its behavior and characteristics while fitting in its habitat.",
        "role": "assistant",
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1756251832,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "object": "chat.completion",
  "usage": null
}
```

## Multimodal Disaggregated Video Serving

This example demonstrates deploying a disaggregated multimodal model that can process video inputs.

### Components

- workers: For disaggregated video serving, we have three workers, [VideoEncodeWorker](components/video_encode_worker.py) for decoding video into frames,
[VllmDecodeWorker](components/worker.py) for decoding, and [VllmPDWorker](components/worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the VideoEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have three workers, [VideoEncodeWorker](components/video_encode_worker.py), [VllmDecodeWorker](components/worker.py), and [VllmPDWorker](components/worker.py).
For the LLaVA-NeXT-Video-7B model, frames are only required during the prefill stage. As such, the VideoEncodeWorker is connected directly to the prefill worker.
The VideoEncodeWorker is responsible for decoding the video into a series of frames and passing them to the prefill worker via RDMA.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](/docs/backends/vllm/README.md) example.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --video_url--> video_encode_worker
  video_encode_worker --> processor
  video_encode_worker --frames--> prefill_worker
  prefill_worker --> video_encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/video_disagg.sh
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
  "id": "7587e7d152014bae8e5c4e25f9fda0ed",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": " The video takes us away to a lively world of wildlife and natural beauty, featuring a white rabbit in a vibrant forest setting. At the beginning of the clip, the white rabbit is seen standing on a rock, facing towards the right side of the frame, with bushes and trees in the backdrop. The rabbit appears to be alert, given its ears are up and its ears perked in the air. As the clip progresses, the movement of the rabbit brings it around a tree, where its legs are partially hidden by the dense vegetation. It then sits down and grooms its fur, a behavior that suggests it is comfortable in its surroundings. \n\nThe scene then switches to a close-up shot of the rabbit, giving us a better view of its features and expressions. In this camera angle, the rabbit appears more dynamic and alert, with its breathing more visible, signaling its health and well-being. The camera pans out, and we see the rabbit heading towards the left side of the screen, possibly curious or hunting for food, with its ears perked up again. The lush greenery of the forest unfolds in the background, adding to the feeling of a wild and thriving environment.\n\n\nThe rabbit, upturned slightly while walking, finds a pile of dirt and rocks and sits there, fully clothed, perhaps taking a break from its exploration. There's a mention of a blue bird that appears to perch atop a log, adding a touch of whimsy to the scene. Lastly, the rabbit is observed relaxing on the rocks, resting comfortably, and looking off to the right side—a moment of tranquility in a bustling ecosystem. Throughout the clip, the rabbit's outfit remains the same, allowing for a clear focus on its behavior and characteristics while fitting in its habitat.",
        "role": "assistant",
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1756251832,
  "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
  "object": "chat.completion",
  "usage": null
}
```
## Multimodal Aggregated Audio Serving

This example demonstrates deploying an aggregated multimodal model that can process audio inputs.

### Components

- workers: For audio serving, we use the [AudioEncodeWorker](components/audio_encode_worker.py) for decoding audio into audio embeddings, and send the embeddings to [VllmPDWorker](components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the AudioEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have two workers, [AudioEncodeWorker](components/audio_encode_worker.py) and [VllmPDWorker](components/worker.py).
The AudioEncodeWorker is responsible for decoding the audio into embeddings.
Its VllmPDWorker then prefills and decodes the prompt, just like the [LLM aggregated serving](../../docs/backends/vllm/README.md) example.
By separating the audio processing from the prefill and decode stages, we can have a more flexible deployment and scale the
AudioEncodeWorker independently from the prefill and decode workers if needed.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --audio_url--> audio_encode_worker
  audio_encode_worker --> processor
  audio_encode_worker --embeddings--> pd_worker
  pd_worker --> audio_encode_worker
```

```bash
pip install vllm["audio"] accelerate # multimodal audio models dependency
cd $DYNAMO_HOME/examples/multimodal
bash launch/audio_agg.sh
```

### Client

In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen/Qwen2-Audio-7B-Instruct",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is recited in the audio?"
            },
            {
              "type": "audio_url",
              "audio_url": {
                "url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"
              }
            }
          ]
        }
      ],
      "max_tokens": 6000,
      "temperature": 0.8,
      "stream": false
    }' | jq
```

You should see a response describing the audio's content similar to
```json
{
  "id": "e2d8d67c37634b309400974eaa058ce8",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "The original content of this audio is:'yet these thoughts affected Hester Pynne less with hope than apprehension.'",
        "refusal": null,
        "tool_calls": null,
        "role": "assistant",
        "function_call": null,
        "audio": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "created": 1756368148,
  "model": "Qwen/Qwen2-Audio-7B-Instruct",
  "service_tier": null,
  "system_fingerprint": null,
  "object": "chat.completion",
  "usage": null
}
```

## Multimodal Disaggregated Audio Serving

This example demonstrates deploying a disaggregated multimodal model that can process audio inputs.

### Components

- workers: For disaggregated audio serving, we have three workers, [AudioEncodeWorker](components/audio_encode_worker.py) for decoding audio into embeddings,
[VllmDecodeWorker](components/worker.py) for decoding, and [VllmPDWorker](components/worker.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the AudioEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

### Workflow

In this workflow, we have three workers, [AudioEncodeWorker](components/audio_encode_worker.py), [VllmDecodeWorker](components/worker.py), and [VllmPDWorker](components/worker.py).
For the Qwen/Qwen2-Audio-7B-Instruct model, audio embeddings are only required during the prefill stage. As such, the AudioEncodeWorker is connected directly to the prefill worker.
The AudioEncodeWorker is responsible for decoding the audio into embeddings and passing them to the prefill worker via RDMA.
The prefill worker performs the prefilling step and forwards the KV cache to the decode worker for decoding.
For more details on the roles of the prefill and decode workers, refer to the [LLM disaggregated serving](../../docs/backends/vllm/README.md) example.

This figure illustrates the workflow:
```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --audio_url--> audio_encode_worker
  audio_encode_worker --> processor
  audio_encode_worker --embeddings--> prefill_worker
  prefill_worker --> audio_encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

```bash
pip install vllm["audio"] accelerate # multimodal audio models dependency
cd $DYNAMO_HOME/examples/multimodal
bash launch/audio_disagg.sh
```