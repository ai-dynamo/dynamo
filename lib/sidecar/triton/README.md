<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Triton sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are experimental.
> Flags and behavior may change without notice.

`dynamo-triton-sidecar` connects Dynamo to a separately managed NVIDIA Triton Inference Server over its KServe v2 gRPC API, its KServe v2 HTTP/REST API, or both.
It registers one Dynamo endpoint for every ready Triton model and forwards unary model-infer requests without loading Triton or GPU libraries into the sidecar process.
By default each model is served as a KServe tensor model; `--classify <model>` and `--embeddings <model>` serve the named text models through the frontend's OpenAI `/v1/classify` and `/v1/embeddings` endpoints instead (see [OpenAI classify and embeddings](#openai-classify-and-embeddings)).

Choose the protocols at runtime by passing their endpoints.
At least one is required.

```bash
# gRPC only, on localhost:8001
dynamo-triton-sidecar --grpc --discovery-backend file

# HTTP only, on localhost:8000
dynamo-triton-sidecar --http --discovery-backend file

# Both, on Triton's default ports
dynamo-triton-sidecar --grpc --http --discovery-backend file

# Both, on a remote server
dynamo-triton-sidecar \
  --grpc triton.example.com:8001 \
  --http https://triton.example.com/prefix \
  --discovery-backend file
```

| Flag     | Environment Variable        | Value                                                                               | With No Value    |
| :------- | :-------------------------- | :---------------------------------------------------------------------------------- | :--------------- |
| `--grpc` | `DYN_SIDECAR_GRPC_ENDPOINT` | port, `host:port`, or a plaintext `http://` / `grpc://` URL                         | `localhost:8001` |
| `--http` | `DYN_SIDECAR_HTTP_ENDPOINT` | port, `host:port`, or an `http://` / `https://` URL, optionally with a path prefix  | `localhost:8000` |

A bare port, such as `--grpc 9001`, means `localhost:<port>`.
The environment variables take the same values as `--grpc` and `--http` and apply only when neither form of the flag is passed.
At startup, the sidecar writes each endpoint it uses back to its environment variable in full, such as `http://localhost:8000`, so subprocesses see the same endpoints.

`--grpc-endpoint` and `--http-endpoint` are the same as `--grpc` and `--http`, matching the flags of the other Dynamo sidecars: each takes a required `host:port` or URL, with no bare-port or no-value default.
When one is passed along with its short form, the last one on the command line wins; either flag on the command line overrides the environment variable.

When both protocols are enabled:

- Model discovery and model configuration use gRPC.
- Startup waits until both endpoints report ready, and a model is served only when it is ready on both.
- Inference requests alternate between gRPC and HTTP.

The HTTP transport uses Triton's binary tensor data extension for every input and output, so tensors travel as raw bytes as they do over gRPC.
Triton's JSON model configuration is converted to the same protobuf `ModelConfig` that gRPC returns, so Dynamo exposes identical model metadata on either protocol.
The `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs` flags also govern the HTTP transport; `--grpc-connections` applies only to gRPC.

Ready models are discovered through Triton's `RepositoryIndex` API.
Use one or more `--model-name <name>` options to skip repository auto-discovery.

## OpenAI classify and embeddings

Each model is served through one endpoint, chosen by the flags that name it:

| Flag                                                         | Frontend Endpoint            | Registration                        |
| :----------------------------------------------------------- | :--------------------------- | :---------------------------------- |
| (none, the default) or `--model-name <model>`                | KServe gRPC Tensor Inference | `Tensor` input, `TensorBased` model |
| `--classify <model>` (`DYN_TRITON_CLASSIFY_MODELS`)          | `POST /v1/classify`          | `Text` input, `Classify` model      |
| `--embeddings <model>` (`DYN_TRITON_EMBEDDINGS_MODELS`)      | `POST /v1/embeddings`        | `Text` input, `Embedding` model     |

Repeat `--classify` and `--embeddings`, or pass a comma-separated list, to serve several models; one Triton server can serve tensor, classify, and embeddings models side by side.
A model cannot be passed to both flags.
The named models are served in addition to any `--model-name` models; when auto-discovering, each named model must be ready in Triton.

```bash
dynamo-triton-sidecar --grpc \
  --classify text_classifier \
  --embeddings text_embedder \
  --discovery-backend file
```

A classify or embeddings model must declare exactly one `TYPE_STRING` input with `dims: [1]` or `dims: [-1]` and exactly one `TYPE_FP32` output, typically an ensemble that tokenizes inside Triton.
A named model that does not fit fails startup with its name in the error.

- Each request's text prompts are sent to Triton as one `BYTES` tensor, shaped `[N, 1]` for a batching model and `[N]` otherwise, and Triton returns one `FP32` row per prompt.
- Ensemble steps (tokenizer, encoder, and so on) that are not named are served as tensor models like any other model; pass `--model-name` to register only the models you want.
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
| OpenAI `/v1/classify` (text in)   | Yes, with `--classify <model>`             |
| OpenAI `/v1/embeddings` (text in) | Yes, with `--embeddings <model>`           |
| OpenAI `/v1/pooling`              | No                                         |
| Dynamic model repository updates  | No; restart the sidecar                    |
