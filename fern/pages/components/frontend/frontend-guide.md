---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Frontend Guide

This guide covers the KServe gRPC frontend configuration and integration for the Dynamo Frontend.

## KServe gRPC Frontend

### Motivation

[KServe v2 API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2) is one of the industry-standard protocols for machine learning model inference. Triton inference server is one of the inference solutions that comply with KServe v2 API and it has gained a lot of adoption. To quickly enable Triton users to explore with Dynamo benefits, Dynamo provides a KServe gRPC frontend.

This documentation assumes readers are familiar with the usage of KServe v2 API and focuses on explaining the Dynamo parts that work together to support KServe API and how users may migrate existing KServe deployment to Dynamo.

## Supported Endpoints

* `ModelInfer` endpoint: KServe Standard endpoint as described [here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1)
* `ModelStreamInfer` endpoint: Triton extension endpoint that provide bi-directional streaming version of the inference RPC to allow a sequence of inference requests/responses to be sent over a GRPC stream, as described [here](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto#L84-L92)
* `ModelMetadata` endpoint: KServe standard endpoint as described [here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata-1)
* `ModelConfig` endpoint: Triton extension endpoint as described [here](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md)

## Starting the Frontend

To start the KServe frontend, run the below command:

```bash
python -m dynamo.frontend --kserve-grpc-server
```

## gRPC Performance Tuning

The gRPC server supports optional HTTP/2 flow control tuning via environment variables. These can be set before starting the server to optimize for high-throughput streaming workloads.

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `DYN_GRPC_INITIAL_CONNECTION_WINDOW_SIZE` | HTTP/2 connection-level flow control window size in bytes | tonic default (64KB) |
| `DYN_GRPC_INITIAL_STREAM_WINDOW_SIZE` | HTTP/2 per-stream flow control window size in bytes | tonic default (64KB) |

### Example: High-ISL/OSL configuration for streaming workloads

```bash
# For 128 concurrent 15k-token requests
export DYN_GRPC_INITIAL_CONNECTION_WINDOW_SIZE=16777216  # 16MB
export DYN_GRPC_INITIAL_STREAM_WINDOW_SIZE=1048576      # 1MB
python -m dynamo.frontend --kserve-grpc-server
```

If these variables are not set, the server uses tonic's default values.

<Note>
Tune these values based on your workload. Connection window should accommodate `concurrent_requests x request_size`. Memory overhead equals the connection window size (shared across all streams). See [gRPC performance best practices](https://grpc.io/docs/guides/performance/) and [gRPC channel arguments](https://grpc.github.io/grpc/core/group__grpc__arg__keys.html) for more details.
</Note>

## Registering a Backend

Similar to HTTP frontend, the registered backend will be auto-discovered and added to the frontend list of serving model. To register a backend, the same `register_llm()` API will be used. Currently the frontend support serving of the following model type and model input combination:

* `ModelType::Completions` and `ModelInput::Text`: Combination for LLM backend that uses custom preprocessor
* `ModelType::Completions` and `ModelInput::Token`: Combination for LLM backend that uses Dynamo preprocessor (i.e. Dynamo vLLM / SGLang / TRTLLM backend)
* `ModelType::TensorBased` and `ModelInput::Tensor`: Combination for backend that is used for generic tensor-based inference

The first two combinations are backed by OpenAI Completions API, see [OpenAI Completions section](#openai-completions) for more detail. Whereas the last combination is most aligned with KServe API and the users can replace existing deployment with Dynamo once their backends implements adaptor for `NvCreateTensorRequest/NvCreateTensorResponse`, see [Tensor section](#tensor) for more detail:

### OpenAI Completions

Most of the Dynamo features are tailored for LLM inference and the combinations that are backed by OpenAI API can enable those features and are best suited for exploring those Dynamo features. However, this implies specific conversion between generic tensor-based messages and OpenAI message and imposes specific structure of the KServe request message.

#### Model Metadata / Config

The metadata and config endpoint will report the registered backend to have the below, note that this is not the exact response.

```json
{
    "name": "$MODEL_NAME",
    "version": 1,
    "platform": "dynamo",
    "backend": "dynamo",
    "inputs": [
        {
            "name": "text_input",
            "datatype": "BYTES",
            "shape": [1]
        },
        {
            "name": "streaming",
            "datatype": "BOOL",
            "shape": [1],
            "optional": true
        }
    ],
    "outputs": [
        {
            "name": "text_output",
            "datatype": "BYTES",
            "shape": [-1]
        },
        {
            "name": "finish_reason",
            "datatype": "BYTES",
            "shape": [-1],
            "optional": true
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

This combination is used when the user is migrating an existing KServe-based backend into Dynamo ecosystem.

#### Model Metadata / Config

When registering the backend, the backend must provide the model's metadata as tensor-based deployment is generic and the frontend can't make any assumptions like for OpenAI Completions model. There are two methods to provide model metadata:

* [TensorModelConfig](https://github.com/ai-dynamo/dynamo/tree/main/lib/llm/src/protocols/tensor.rs): This is Dynamo defined structure for model metadata, the backend can provide the model metadata as shown in this [example](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/tests/test_tensor.py). For metadata provided in such way, the following field will be set to a fixed value: `version: 1`, `platform: "dynamo"`, `backend: "dynamo"`. Note that for model config endpoint, the rest of the fields will be set to their default values.
* [triton_model_config](https://github.com/ai-dynamo/dynamo/tree/main/lib/llm/src/protocols/tensor.rs): For users that already have Triton model config and require the full config to be returned for client side logic, they can set the config in `TensorModelConfig::triton_model_config` which supersedes other fields in `TensorModelConfig` and be used for endpoint responses. `triton_model_config` is expected to be the serialized string of the `ModelConfig` protobuf message, see [echo_tensor_worker.py](https://github.com/ai-dynamo/dynamo/blob/main/tests/frontend/grpc/echo_tensor_worker.py) for example.

#### Inference

When receiving inference request, the backend will receive [NvCreateTensorRequest](https://github.com/ai-dynamo/dynamo/tree/main/lib/llm/src/protocols/tensor.rs) and be expected to return [NvCreateTensorResponse](https://github.com/ai-dynamo/dynamo/tree/main/lib/llm/src/protocols/tensor.rs), which are the mapping of ModelInferRequest / ModelInferResponse protobuf message in Dynamo.

## Python Bindings

The frontend may be started via Python binding, this is useful when integrating Dynamo in existing system that desire the frontend to be run in the same process with other components. See [server.py](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/examples/kserve_grpc_service/server.py) for example.

## Integration

### With Router

The frontend includes an integrated router for request distribution. Configure routing mode:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

See [Router Documentation](../router/README.md) for routing configuration details.

### With Backends

Backends auto-register with the frontend when they call `register_llm()`. Supported backends:

- [vLLM Backend](../../backends/vllm/README.md)
- [SGLang Backend](../../backends/sglang/README.md)
- [TensorRT-LLM Backend](../../backends/trtllm/README.md)

## See Also

| Document | Description |
|----------|-------------|
| [Frontend Overview](README.md) | Quick start and feature matrix |
| [Router Documentation](../router/README.md) | KV-aware routing configuration |
