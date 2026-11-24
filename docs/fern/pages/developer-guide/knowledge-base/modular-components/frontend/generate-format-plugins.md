---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Generate Format Plugins
subtitle: How the frontend serves engine-native token-in/token-out routes through format plugins
---

The Dynamo frontend serves engine-native `/generate` routes through format plugins. A format plugin is a Rust crate that owns one client-facing wire format.

The frontend owns routing, dispatch, streaming, and metrics. The format plugin owns every request field and every response field. The frontend reads no field of the client body.

This page is for developers who change a built-in format or add a format.

## Built-In Formats

Dynamo includes two built-in formats. The frontend serves a route only after you enable it.

| Format | Route | Enable with | Path override | Worker capability |
|---|---|---|---|---|
| `sglang` | `POST`, `PUT` `/generate` | `DYN_SGLANG_ENABLE_GENERATE=1` | `DYN_HTTP_SVC_SGLANG_GENERATE_PATH` | `sglang_generate` |
| `vllm` | `POST /inference/v1/generate` | `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE=1` | `DYN_HTTP_SVC_VLLM_GENERATE_PATH` | `vllm_inference_v1_generate` |

`DYN_GENERATE_FORMAT` selects the format for the `/generate` route. The default is `sglang`. The `/inference/v1/generate` route always uses the `vllm` format.

Each worker advertises a capability in its runtime data. The frontend sends a request only to workers that advertise the capability of the format.

## Code Locations

| Path | Contents | Owner |
|---|---|---|
| `lib/llm/src/generate_format.rs` | The `GenerateFormat` and `GenerateExchange` traits, and the format registry | Frontend |
| `lib/llm/src/generate_format/error.rs` | The error types that map worker errors to HTTP status and request metrics | Frontend |
| `lib/llm/src/http/service/format_generate.rs` | The shared HTTP handler for all formats | Frontend |
| `lib/generate-format-catalog` | The default catalog, which registers the built-in formats | Frontend |
| `lib/generate-formats/sglang` | The SGLang format | SGLang backend |
| `lib/generate-formats/vllm` | The vLLM format | vLLM backend |

The backend teams own their format crates. A change to SGLang or vLLM request rules stays in the crate of that backend.

## Request Flow

The shared handler does these steps for each request:

1. The handler reads the body and calls `GenerateFormat::open`.
2. `open` decodes the body and returns an `Opened` value. `Opened` contains the request ID, the stream flag, the model name, and the input token count.
3. The handler makes sure that the frontend is ready. Then it applies the semantic validation result from `Opened`.
4. The handler selects the worker set for the model.
5. The handler calls the `Bind` closure from `Opened` with a `Worker` value. `Worker` contains the model name, the KV cache block size, and the LoRA name.
6. `Bind` returns a `Bound` value. `Bound` contains the routed `PreprocessedRequest` and a `GenerateExchange`.
7. The handler dispatches the request to the worker through the router.
8. The handler gives each worker output to `GenerateExchange::chunk`.
9. At the end of the stream, the handler calls `GenerateExchange::finish`.

The format receives the worker set before it builds the request. Thus, the format can use the KV cache block size, for example to compute multimodal routing hashes.

## The Format Interface

### GenerateFormat

`GenerateFormat` describes one wire format. These methods are required:

- `name` returns the name that `DYN_GENERATE_FORMAT` uses.
- `capability` returns the runtime capability that a worker advertises.
- `open` starts one client request.

The other methods have default implementations. They control `PUT` support, body errors, frontend errors, worker errors, and the JSON shape of an error body. Override them to keep the error behavior of the native engine API.

### Opened and Bind

`open` decodes the body and returns errors for malformed input. `open` does not do semantic validation. Put semantic validation in `Opened::validation`. The handler applies `Opened::validation` after the readiness check.

`Bind` is a closure that the handler calls one time. It builds the `PreprocessedRequest` for the selected worker set.

### GenerateExchange

`GenerateExchange` assembles client responses from worker outputs. The handler calls its methods in stream order, and never concurrently.

- `chunk` receives one `LLMEngineOutput`. For a streamed request, `chunk` gives each client chunk to `emit`. For a unary request, `chunk` keeps the data until `finish`.
- The `prefill` flag on `chunk` marks the final output of the prefill worker in disaggregated serving.
- `finish` returns the unary response. For a streamed request, `finish` returns an optional final chunk.

## Native Payloads

A format sends the native request body to the worker in `PreprocessedRequest::extra_args`. The worker reads the body from the key that the format uses.

The body in `extra_args` is not the complete native request. The format moves the prompt token IDs out of the body and into `PreprocessedRequest::token_ids`. The worker puts the canonical token IDs back into the native request before it validates the request and sends it to the engine.

| Format | Request key | Response key |
|---|---|---|
| `sglang` | `extra_args.sglang_tito` | `engine_data.sglang_response` |
| `vllm` | `extra_args.vllm_tito` | Typed fields of `LLMEngineOutput` |

The format also fills the canonical fields of `PreprocessedRequest`. The canonical fields include the token IDs, the stop conditions, the sampling options, and the routing hints. The KV router uses the canonical token IDs.

The SGLang worker returns each native SGLang response in `engine_data`. The SGLang format merges these responses into the client response. The vLLM format builds the client response from the typed fields, such as token IDs, log probabilities, and the finish reason.

## Streaming

The `sglang` format supports unary and streamed requests. The `vllm` format supports only unary requests. It rejects `stream=true`.

The handler owns the Server-Sent Events (SSE) transport and client disconnect handling. The format owns the content of each chunk.

## Migration

Requests on `/generate` routes do not migrate to another worker. The frontend sets the migration limit to 0 for these routes.

The native body holds request state that the canonical fields do not describe. An example is the token budget in `sglang_tito.sampling_params.max_new_tokens`. A retry must update that state, and only the format can read it. Thus, migration support for a format needs changes in the crate of that format.

## Errors

The format classifies worker errors with `GenerateFormat::worker_error`. The `ErrorPhase` argument tells the format whether the failure occurred at dispatch or in the response stream.

A `WorkerError` holds the HTTP status, the public message, and the error type for request metrics. `ClientError` holds a client error. `ClientError::new` returns a 400 error.

## Add a Custom Format

To serve a format that Dynamo does not include, link a custom catalog into the frontend.

1. Create a Rust crate that implements `GenerateFormat` and `GenerateExchange`.
2. Create a catalog crate. Add `dynamo-generate-format-catalog` to this crate as a dependency under an alias. In the `register` function of your catalog, call `register` of the built-in catalog to keep the built-in formats.
3. Register your format with a name that no other format uses. The registry rejects duplicate names.
4. In `lib/bindings/python/Cargo.toml`, point the `dynamo-generate-format-catalog` dependency at your catalog crate with `package = "<your-crate>"`. The bindings call `dynamo_generate_format_catalog::register`, so the dependency name does not change.
5. Set `DYN_GENERATE_FORMAT` to the name of your format.
6. Make sure that each worker advertises the capability of your format in its runtime data.

The frontend installs the registry one time at startup. An unknown format name causes a startup error.

For the built-in catalog, refer to [`lib/generate-format-catalog/src/lib.rs`](https://github.com/ai-dynamo/dynamo/blob/main/lib/generate-format-catalog/src/lib.rs).

## Related Pages

- [Protocol Field Handling](protocol-field-handling.md)
- [Python Route Extensions](python-route-extensions.md)
- [Frontend Overview](overview.md)
