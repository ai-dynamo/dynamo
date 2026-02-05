---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# KServe gRPC frontend

## Motivation

[KServe v2 API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2) is one of the industry standard protocol for machine learning model inference. Triton inference server is one of the inference solutions that comply with KServe v2 API and it has gained a lot of adoption. To quickly enable Triton users to explore with Dynamo benefits, Dynamo provides a KServe gRPC frontend.

This documentation assumes readers are familiar with the usage of KServe v2 API and focuses on explaining the Dynamo parts that work together to support KServe API and how users may migrate existing KServe deployment to Dynamo.

## Supported Endpoints

* `ModelInfer` endpoint: KServe Standard endpoint as described [here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1)
* `ModelStreamInfer` endpoint: Triton extension endpoint that provide bi-directional streaming version of the inference RPC to allow a sequence of inference requests/responses to be sent over a GRPC stream, as described [here](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto#L84-L92)
* `ModelMetadata` endpoint: KServe standard endpoint as described [here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata-1)
* `ModelConfig` endpoint: Triton extension endpoint as described [here](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md)

## Starting the Frontend

To start the KServe frontend, run the below command
```
python -m dynamo.frontend --kserve-grpc-server
```

## Registering a Backend

Similar to HTTP frontend, the registered backend will be auto-discovered and added to the frontend list of serving model. To register a backend, the same `register_llm()` API will be used. Currently the frontend support serving of the following model type and model input combination:
* `ModelType::Completions` and `ModelInput::Text`: Combination for LLM backend that uses custom preprocessor
* `ModelType::Completions` and `ModelInput::Token`: Combination for LLM backend that uses Dynamo preprocessor (i.e. Dynamo vLLM / SGLang / TRTLLM backend)
* `ModelType::TensorBased` and `ModelInput::Tensor`: Combination for backend that is used for generic tensor based inference

The first two combinations are backed by OpenAI Completions API, see [OpenAI Completions section](#openai-completions) for more detail. Whereas the last combination is most aligned with KServe API and the users can replace existing deployment with Dynamo once their backends implements adaptor for `NvCreateTensorRequest/NvCreateTensorResponse`, see [Tensor section](#tensor) for more detail:

### OpenAI Completions

Most of the Dynamo features are tailored for LLM inference and the combinations that are backed by OpenAI API can enable those features and are best suited for exploring those Dynamo features. However, this implies specific conversion between generic tensor based messages and OpenAI message and imposes specific structure of the KServe request message.

#### Model Metadata / Config

The metadata and config endpoint will report the registered backend to have the below, note that this is not the exact response.
```
{
    name: $MODEL_NAME,
    version: 1,
    platform: "dynamo",
    backend: "dynamo", # model config specific
    inputs: [
        {
            name: "text_input",
            datatype: "BYTES",
            shape: [1]
        },
        {
            name: "streaming",
            datatype: "BOOL",
            shape: [1],
            optional: true
        }
    ]
    outputs: [
        {
            name: "text_output",
            datatype: "BYTES",
            shape: [-1]
        },
        {
            name: "finish_reason",
            datatype: "BYTES",
            shape: [-1],
            optional: true
        }
    ]
}
```

#### Inference

On receiving inference request, the following conversion will be performed:
* `text_input`: the element is expected to contain the user prompt string and will be converted to `prompt` field in OpenAI Completion request
* `streaming`: the element will be converted to `stream` field in OpenAI Completion request
On receiving model response, the following conversion will be performed:
* `text_output`: each element corresponds to one choice in OpenAI Completion response, and the content will be set to `text` of the choice.
* `finish_reason`: each element corresponds to one choice in OpenAI Completion response, and the content will be set to `finish_reason` of the choice.

### Tensor

This combination is used when the user is migrating an existing KServe based backend into Dynamo ecosystem.

#### Model Metadata / Config

When registering the backend, the backend must provide the model's metadata as tensor based deployment is generic and the frontend can't make any assumptions like for OpenAI Completions model. There are two methods to provide model metadata:
* [TensorModelConfig](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/protocols/tensor.rs): This is Dynamo defined structure for model metadata, the backend can provide the model metadata as shown in this [example](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/tests/test_tensor.py). For metadata provided in such way, the following field will be set to a fixed value: `version: 1`, `platform: "dynamo"`, `backend: "dynamo"`. Note that for model config endpoint, the rest of the fields will be set to their default values.
* [triton_model_config](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/protocols/tensor.rs): For users that already have Triton model config and require the full config to be returned for client side logic, they can set the config in `TensorModelConfig::triton_model_config` which will supersedes other fields in `TensorModelConfig` and be used for endpoint responses. `triton_model_config` is expected to be the serialized string of the `ModelConfig` protobuf message, see [echo_tensor_worker.py](https://github.com/ai-dynamo/dynamo/blob/main/tests/frontend/grpc/echo_tensor_worker.py) for example.

#### Inference

When receiving inference request, the backend will receive [NvCreateTensorRequest](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/protocols/tensor.rs) and be expected to return [NvCreateTensorResponse](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/protocols/tensor.rs), which are the mapping of ModelInferRequest / ModelInferResponse protobuf message in Dynamo.

## Python Bindings

The frontend may be started via Python binding, this is useful when integrating Dynamo in existing system that desire the frontend to be run in the same process with other components. See [server.py](https://github.com/ai-dynamo/dynamo/blob/main/lib/bindings/python/examples/kserve_grpc_service/server.py) for example.
