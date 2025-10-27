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

# Multimodal Support

TRTLLM supports multimodal models with dynamo. You can provide multimodal inputs in the following ways:

- By sending image URLs
- By providing paths to pre-computed embedding files

Please note that you should provide **either image URLs or embedding file paths** in a single request.

## Aggregated

Here are quick steps to launch Llama-4 Maverick BF16 in aggregated mode
```bash
cd $DYNAMO_HOME

export AGG_ENGINE_ARGS=./recipes/llama4/trtllm/multimodal/agg.yaml
export SERVED_MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
export MODEL_PATH="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
./launch/agg.sh
```
## Example Requests

### With Image URL

Below is an example of an image being sent to `Llama-4-Maverick-17B-128E-Instruct` model

Request :
```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```
Response :

```
{"id":"unknown-id","choices":[{"index":0,"message":{"content":"The image depicts a serene landscape featuring a large rock formation, likely El Capitan in Yosemite National Park, California. The scene is characterized by a winding road that curves from the bottom-right corner towards the center-left of the image, with a few rocks and trees lining its edge.\n\n**Key Features:**\n\n* **Rock Formation:** A prominent, tall, and flat-topped rock formation dominates the center of the image.\n* **Road:** A paved road winds its way through the landscape, curving from the bottom-right corner towards the center-left.\n* **Trees and Rocks:** Trees are visible on both sides of the road, with rocks scattered along the left side.\n* **Sky:** The sky above is blue, dotted with white clouds.\n* **Atmosphere:** The overall atmosphere of the","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1753322607,"model":"meta-llama/Llama-4-Maverick-17B-128E-Instruct","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}
```

## Disaggregated

Here are quick steps to launch in disaggregated mode.

The following is an example of launching a model in disaggregated mode. While this example uses `Qwen/Qwen2-VL-7B-Instruct`, you can adapt it for other models by modifying the environment variables for the model path and engine configurations.
```bash
cd $DYNAMO_HOME

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen2-VL-7B-Instruct"}
export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"decode_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"recipes/qwen2-vl-7b-instruct/trtllm/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"recipes/qwen2-vl-7b-instruct/trtllm/decode.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

./launch/disagg.sh
```

For a large model like `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, a multi-node setup is required for disaggregated serving, while aggregated serving can run on a single node. This is because the model with a disaggregated configuration is too large to fit on a single node's GPUs. For instance, running this model in disaggregated mode requires a setup of 2 nodes with 8xH200 GPUs or 4 nodes with 4xGB200 GPUs.

In general, disaggregated serving can run on a single node, provided the model fits on the GPU. The multi-node requirement in this example is specific to the size and configuration of the `meta-llama/Llama-4-Maverick-17B-128E-Instruct` model.

To deploy `Llama-4-Maverick-17B-128E-Instruct` in disaggregated mode, you will need to follow the multi-node setup instructions, which can be found [here](./multinode/multinode-multimodal-example.md).

## Using Pre-computed Embeddings (Experimental)

Dynamo with TensorRT-LLM supports providing pre-computed embeddings directly in an inference request. This bypasses the need for the model to process an image and generate embeddings itself, which is useful for performance optimization or when working with custom, pre-generated embeddings.

### How to Use

Once the container is built, you can send requests with paths to local embedding files.

-   **Format:** Provide the embedding as part of the `messages` array, using the `image_url` content type.
-   **URL:** The `url` field should contain the absolute or relative path to your embedding file on the local filesystem.
-   **File Types:** Supported embedding file extensions are `.pt`, `.pth`, and `.bin`. Dynamo will automatically detect these extensions.

When a request with a supported embedding file is received, Dynamo will load the tensor from the file and pass it directly to the model for inference, skipping the image-to-embedding pipeline.

### Example Request

Here is an example of how to send a request with a pre-computed embedding file.

```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the content represented by the embeddings"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/path/to/your/embedding.pt"
                    }
                }
            ]
        }
    ],
    "stream": false,
    "max_tokens": 160
}'
```
## Encode-Prefill-Decode (EPD) Flow with NIXL

Dynamo with the TensorRT-LLM backend supports multimodal models in Encode -> Decode -> Prefill fashion, enabling you to process embeddings seperately in a seperate worker. For detailed setup instructions, example requests, and best practices, see the [Multimodal EPD Support Guide](./multimodal_epd.md).

## Supported Multimodal Models

Multimodel models listed [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/inputs/utils.py#L221) are supported by dynamo.