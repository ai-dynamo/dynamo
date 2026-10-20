<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Triton sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Flags and behavior may change without notice.

`dynamo-triton-sidecar` connects Dynamo to a separately managed NVIDIA Triton Inference Server over its KServe v2 gRPC API, its KServe v2 HTTP/REST API, or both.
It registers one Dynamo endpoint for every ready Triton model and forwards unary model-infer requests without loading Triton or GPU libraries into the sidecar process.
By default each model is served as a KServe tensor model; `--task classify` and `--task embeddings` serve text models through the frontend's OpenAI `/v1/classify` and `/v1/embeddings` endpoints instead (see [OpenAI classify and embeddings](#openai-classify-and-embeddings)).

Choose the protocols at runtime by passing their endpoints. At least one is required.

```bash
# gRPC only
dynamo-triton-sidecar --grpc-endpoint localhost:8001 --discovery-backend file

# HTTP only
dynamo-triton-sidecar --http-endpoint localhost:8000 --discovery-backend file

# Both
dynamo-triton-sidecar \
  --grpc-endpoint localhost:8001 \
  --http-endpoint localhost:8000 \
  --discovery-backend file
```

| Flag              | Environment Variable        | Value                                                                       |
| :---------------- | :-------------------------- | :-------------------------------------------------------------------------- |
| `--grpc-endpoint` | `DYN_SIDECAR_GRPC_ENDPOINT` | `host:port` or a plaintext `http://` / `grpc://` URL                        |
| `--http-endpoint` | `DYN_SIDECAR_HTTP_ENDPOINT` | `host:port` or an `http://` / `https://` URL, optionally with a path prefix |

When both protocols are enabled:

- Model discovery and model configuration use gRPC.
- Startup waits until both endpoints report ready, and a model is served only when it is ready on both.
- Inference requests alternate between gRPC and HTTP.

The HTTP transport uses Triton's binary tensor data extension for every input and output, so tensors travel as raw bytes as they do over gRPC.
Triton's JSON model configuration is converted to the same protobuf `ModelConfig` that gRPC returns, so Dynamo exposes identical model metadata on either protocol.
The `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs` flags also govern the HTTP transport; `--grpc-connections` applies only to gRPC.

Ready models are discovered through Triton's `RepositoryIndex` API. Use one
or more `--model-name <name>` options to skip repository auto-discovery.

## OpenAI classify and embeddings

`--task` (`DYN_TRITON_TASK`) selects the endpoint every registered model serves:

| Task               | Frontend Endpoint            | Registration                        |
| :----------------- | :--------------------------- | :---------------------------------- |
| `tensor` (default) | KServe gRPC tensor inference | `Tensor` input, `TensorBased` model |
| `classify`         | `POST /v1/classify`          | `Text` input, `Classify` model      |
| `embeddings`       | `POST /v1/embeddings`        | `Text` input, `Embedding` model     |

These tasks match the Python Triton worker's `--task classify` and `--task embeddings`.
The model must declare exactly one `TYPE_STRING` input with `dims: [1]` or `dims: [-1]` and exactly one `TYPE_FP32` output, typically an ensemble that tokenizes inside Triton.
Name the tensors with `--classify-input-name` / `--classify-output-name` or `--embeddings-input-name` / `--embeddings-output-name` when the model has more than one candidate.
A model that does not fit fails startup with its name in the error.

```bash
dynamo-triton-sidecar --grpc-endpoint localhost:8001 --task classify --discovery-backend file
```

- Each request's text prompts are sent to Triton as one `BYTES` tensor, shaped `[N, 1]` for a batching model and `[N]` otherwise, and Triton returns one `FP32` row per prompt.
- When auto-discovering models, ensemble steps (tokenizer, encoder, and so on) stay loaded in Triton but are not registered.
- Token-ID input is rejected, as are request controls owned by the Triton model: `use_activation`, `add_special_tokens`, `truncate_prompt_tokens`, `truncation_side`, `mm_processor_kwargs`, `cache_salt`, and non-zero `priority` for classify; `dimensions`, `add_special_tokens`, and `truncate_prompt_tokens` for embeddings.
- Classify labels come from the output's `label_filename`.
  Triton exposes labels only through its classification extension, which rounds scores, so the sidecar reads probabilities from the raw output and fetches the labels once per model with one extra classification request.
- Embeddings travel to the frontend as base64 little-endian `FP32`; the frontend returns floats unless the client asks for `encoding_format: "base64"`.
- Health-check probes answer from Triton's model readiness without running inference.

## Support matrix

| Feature                           | Supported                                  |
| :-------------------------------- | :----------------------------------------- |
| KServe v2 unary tensor inference  | Yes                                        |
| KServe v2 gRPC transport          | Yes                                        |
| KServe v2 HTTP/REST transport     | Yes, with the binary tensor data extension |
| Multiple models per Triton server | Yes                                        |
| Typed and raw tensor contents     | Yes                                        |
| Classification output parameters  | Passed through                             |
| Bidirectional streaming inference | No                                         |
| Shared-memory tensor I/O          | No                                         |
| OpenAI `/v1/classify` (text in)   | Yes, with `--task classify`                |
| OpenAI `/v1/embeddings` (text in) | Yes, with `--task embeddings`              |
| OpenAI `/v1/pooling`              | No                                         |
| Dynamic model repository updates  | No; restart the sidecar                    |
