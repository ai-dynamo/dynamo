---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Rust API
max-toc-depth: 2
---

# Rust API Reference

NVIDIA Dynamo's core infrastructure is implemented in Rust across multiple workspace crates.
API documentation is hosted on [docs.rs](https://docs.rs) for published crates.

## Core Crates

| Name | Description |
| --- | --- |
| [`dynamo-llm`](https://docs.rs/dynamo-llm/latest) | LLM serving abstractions and pipeline orchestration |
| [`dynamo-runtime`](https://docs.rs/dynamo-runtime/latest) | Distributed runtime for service discovery and RPC |
| [`dynamo-kv-router`](https://docs.rs/dynamo-kv-router/latest) | KV cache-aware request routing |
| [`dynamo-memory`](https://docs.rs/dynamo-memory/latest) | Memory allocation and management for inference |
| [`kvbm-logical`](https://docs.rs/kvbm-logical/latest) | Block allocation and lifecycle for KV cache |

## Supporting Crates

| Name | Description |
| --- | --- |
| [`dynamo-config`](https://docs.rs/dynamo-config/latest) | Configuration parsing and environment utilities |
| [`dynamo-async-openai`](https://docs.rs/dynamo-async-openai/latest) | Async OpenAI-compatible client for inference endpoints |
| [`dynamo-parsers`](https://docs.rs/dynamo-parsers/latest) | Output parsing for tool calling and reasoning |

## Development & Testing

| Name | Description |
| --- | --- |
| [`dynamo-mocker`](https://docs.rs/dynamo-mocker/latest) | Mock LLM scheduler and KV manager for testing |

## Bindings

| Crate | Language | Source | Description |
| --- | --- | --- | --- |
| `dynamo-codegen` | Python | [lib/bindings/python/codegen](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/codegen) | PyO3 codegen (produces the `dynamo._core` module) |
| `libdynamo_llm` | C | [lib/bindings/c](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/c) | C bindings for the LLM library |

## Building Locally

```bash
cargo doc --no-deps --workspace --open
```

This generates HTML documentation for all workspace crates and opens it in your browser.
The output is written to `target/doc/`.
