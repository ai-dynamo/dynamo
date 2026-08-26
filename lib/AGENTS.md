<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust workspace — Agent Guide

Everything under `lib/` is Rust, except `lib/gpu_memory_service/`, which is a Python
package. The root [`AGENTS.md`](../AGENTS.md) covers build, test, lint, and PR
conventions for the whole repository; this file is the crate map and the rules that
apply across the workspace.

The authoritative crate list is the `[workspace] members` array in the root
[`Cargo.toml`](../Cargo.toml). The grouping below is by role, so that a crate you land
in can be placed against its neighbours. Descriptions are taken from each crate's own
`Cargo.toml` `description` or its `src/lib.rs` module docs.

## Crate map

### Runtime and transport

| Directory | Package | Role |
|-----------|---------|------|
| `runtime/` | `dynamo-runtime` | `DistributedRuntime`, the namespace/component/endpoint model, and the request, event, and discovery transports. See [`runtime/AGENTS.md`](runtime/AGENTS.md). |
| `memory/` | `dynamo-memory` | Storage abstraction for the v2 block manager: memory descriptors, concrete storage types, NIXL registration. |
| `tokens/` | `dynamo-tokens` | Token sequences, block creation, and hashing — including `PositionalLineageHash`. |
| `truthy/` | `dynamo-truthy` | Single owner of the truthy/falsy vocabulary accepted from environment variables, headers, and config values. |

### LLM serving

| Directory | Package | Role |
|-----------|---------|------|
| `llm/` | `dynamo-llm` | The largest crate. HTTP and gRPC entrypoints, preprocessor, protocols, local model handling, migration, block manager, and the LLM-side KV router integration. Also carries a worker/frontend wire-compatibility policy in [`llm/CLAUDE.md`](llm/CLAUDE.md). |
| `backend-common/` | `dynamo-backend-common` | Shared runtime glue for Rust backends: the `LLMEngine` trait an engine author implements, the `Worker` lifecycle owner, and a `run()` helper for each backend's `main.rs`. |
| `sidecar/` | `dynamo-sidecar-common`, `dynamo-{vllm,sglang,trtllm}-sidecar` | Sidecars that connect Dynamo workers to inference engines over the engines' native gRPC APIs, with the engine in a separate process. See [`sidecar/README.md`](sidecar/README.md). |
| `router-plugins/catalog/` | `dynamo-worker-selection-policy-catalog` | Build-time catalog of worker-selection policies. |
| `rl/` | `dynamo-rl` | Worker discovery surface for reinforcement-learning workflows; workers opt in via `DYN_ENABLE_RL` / `--enable-rl`. |

### Routing

| Directory | Package | Role |
|-----------|---------|------|
| `kv-router/` | `dynamo-kv-router` | Radix tree indexer, scheduling, sequence tracking, and the router services. Several subdirectories carry their own agent files. |
| `kv-hashing/` | `dynamo-kv-hashing` | The request → `PositionalLineageHash` contract that gives KV cache identity a single definition across router, consolidator, KVBM, and the frameworks. |

### KV block management (KVBM)

| Directory | Package | Role |
|-----------|---------|------|
| `kvbm-common/` | `kvbm-common` | Shared identifiers and layout handles for the KVBM crates. |
| `kvbm-config/` | `kvbm-config` | Configuration for the Tokio, Rayon, and messenger runtimes KVBM uses. |
| `kvbm-logical/` | `kvbm-logical` | Logical block lifecycle: state transitions, block registry and deduplication, pool management, and the event pipeline. |
| `kvbm-physical/` | `kvbm-physical` | Physical layout and transfer management — mapping blocks to memory, registering them for RDMA transfer through NIXL, and moving them between storage tiers. |
| `kvbm-engine/` | `kvbm-engine` | Distributed coordination primitives for KVBM. |
| `kvbm-kernels/` | `kvbm-kernels` | CUDA kernels that convert KV cache blocks between the memory layouts used by different inference frameworks. |
| `kvbm-consolidator/` | `kvbm-consolidator` | Consumes KV cache events from several sources and publishes one deduplicated, router-compatible stream. |

### Simulation and benchmarking

| Directory | Package | Role |
|-----------|---------|------|
| `mocker/` | `dynamo-mocker` | Mock LLM scheduler and KV manager that simulates cache management, scheduling, and token timing without a GPU. |
| `bench/` | `dynamo-bench` | Lightweight HTTP benchmarks against Dynamo endpoints. |
| `data-gen/` | `dynamo-data-gen` | Schemas and primitives for generated data, including Mooncake replay traces. |

### Bindings

| Directory | Package | Role |
|-----------|---------|------|
| `bindings/python/` | `dynamo-py3` (module `dynamo._core`) | The PyO3 extension module, published as the `ai-dynamo-runtime` wheel. |
| `bindings/c/` | `libdynamo_llm` | C API surface. |
| `bindings/kvbm/` | `kvbm-py3` (wheel `kvbm`) | PyO3 bindings for KVBM. |

> [!NOTE]
> `bindings/python/` and `bindings/kvbm/` each declare their own `[workspace]` table in
> their `Cargo.toml`, which excludes them from the root workspace. A plain
> `cargo build` at the repository root does not build them; use `maturin` as the root
> [`AGENTS.md`](../AGENTS.md) Build section describes.

## Build and test

Building the whole workspace is slow. Prefer one crate:

```bash
cargo build -p dynamo-runtime
cargo test  -p dynamo-kv-router
cargo clippy -p dynamo-llm
```

Format only the files your change touched rather than running `cargo fmt --all`, which
rewrites the whole tree and makes a diff unreviewable:

```bash
rustfmt lib/runtime/src/distributed.rs
```

After changing Rust that the Python layer calls, rebuild the extension module — the
Python side imports a compiled artifact, so a `cargo build` alone changes nothing it
can see:

```bash
cd lib/bindings/python && maturin develop --uv
```

## Concepts

The architecture is documented once, under
[`docs/fern/pages/developer-guide/knowledge-base/`](../docs/fern/pages/developer-guide/knowledge-base/).
Read it there rather than reconstructing it from source:

- [Distributed runtime](../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/distributed-runtime.md)
- [Architecture flow](../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture-flow.md)
- [Disaggregated serving](../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md)
- [Routing concepts](../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md)
- [KVBM overview](../docs/fern/pages/developer-guide/knowledge-base/modular-components/kvbm/overview.md)

## Nested agent files

Crates and modules may carry their own `AGENTS.md` or `CLAUDE.md` with local rules.
Before editing a path, inspect that path and each ancestor under `lib/` for both file
names rather than relying on a directory list in this guide. Read every file you find.
This guide frames the workspace; a nested guide provides the more specific rule.
