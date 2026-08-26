<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python components — Agent Guide

This is the Python extensibility layer: the frontend, the engine backends, the
planner, the profiler, and the standalone services that surround them. It is one
half of the `dynamo` package. The other half — the compiled runtime — is built from
Rust in `lib/`, and the boundary between the two is the subject of the section
below.

The root [`AGENTS.md`](../../../AGENTS.md) covers build, test, lint, and PR
conventions. Read [`.ai/python-guidelines.md`](../../../.ai/python-guidelines.md)
before writing code here and
[`.ai/pytest-guidelines.md`](../../../.ai/pytest-guidelines.md) before writing
tests.

## The boundary with Rust

The `dynamo` namespace is assembled from two source roots at install time:

| Source root | Wheel | Built by |
|-------------|-------|----------|
| `components/src/dynamo/` | `ai-dynamo` | hatchling, from the root [`pyproject.toml`](../../../pyproject.toml) |
| `lib/bindings/python/src/dynamo/` | `ai-dynamo-runtime` | maturin, from the PyO3 crate `dynamo-py3` |

So `dynamo.frontend` and `dynamo.runtime` look like siblings but come from
different builds. `dynamo.runtime` is a thin re-export layer over the compiled
extension module `dynamo._core`; `DistributedRuntime`, `Endpoint`, `Client`, and
`Context` are Rust types surfaced through PyO3.

> [!NOTE]
> Editing Rust and then running Python changes nothing on its own — Python imports
> a compiled artifact. Rebuild the extension module after any Rust change a Python
> caller depends on:
>
> ```bash
> cd lib/bindings/python && maturin develop --uv
> ```

## Running a component

Every runnable package has a `__main__.py`, so the invocation is uniform:

```bash
python3 -m dynamo.frontend --help
python3 -m dynamo.vllm --help
```

`dynamo.common` is the exception: it is a library of shared code, not a service,
and has no `__main__.py`.

## Package map

### Serving path

| Package | Role |
|---------|------|
| `frontend/` | The API gateway: OpenAI-compatible HTTP and KServe gRPC endpoints. Carries a configuration-boundary rule in [`frontend/CLAUDE.md`](frontend/CLAUDE.md). |
| `vllm/`, `sglang/`, `trtllm/` | The three engine backends. Each `README.md` points at the backend documentation; `sglang/` carries its own agent file. |
| `tokenspeed/` | A further backend built on the same `LLMEngine` contract. |
| `common/` | Shared library code for everything above: the backend framework, protocols, configuration, HTTP helpers, LoRA, multimodal, snapshot, and storage. `common/backend/` has its own agent file describing the `Worker`/`BaseEngine` lifecycle every engine implements. |

### Routing

| Package | Role |
|---------|------|
| `router/` | Backend-agnostic standalone KV-aware router service. |
| `global_router/` | Hierarchical router between the frontend and local routers in different pool namespaces, in disaggregated or aggregated mode. |
| `kv_dc_relay/` | Discovers inference pools, consumes their ordered KV events, and supervises one Cuckoo-filter producer per local pool. |
| `kv_state_agent/` | Standalone KV state-agent host. |
| `thunderagent_router/` | Experimental; not a released component. |

### Planning and analysis

| Package | Role |
|---------|------|
| `planner/` | SLA-driven autoscaling controller for Dynamo inference graphs. |
| `global_planner/` | Centralized scaling execution for multi-DGD planner deployments. |
| `profiler/` | Profiling entrypoints; the documentation lives under `docs/fern/`. |
| `mocker/` | Python entrypoint for the mock engine implemented in `lib/mocker`, which simulates scheduling and token timing without a GPU. |
| `replay/` | Replay entrypoints for single runs and for the router, planner, and online adapters. The shared offline implementation is owned by AISimulate. |
| `squeeze_evolve/` | Experimental; not a released component. |

## Testing

```bash
pytest -m unit tests/
```

Markers are strict. Several packages also carry their own `tests/` directory next
to the code; run those the same way. Anything that needs a GPU must be gated by
the `gpu_0` … `gpu_8` markers described in the root
[`AGENTS.md`](../../../AGENTS.md).

## Concepts

Architecture is documented once, under
[`docs/fern/pages/developer-guide/knowledge-base/`](../../../docs/fern/pages/developer-guide/knowledge-base/):

- [Architecture flow](../../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture-flow.md)
- [Distributed runtime](../../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/distributed-runtime.md)
- [Disaggregated serving](../../../docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md)
- [Routing concepts](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md)

Backend-specific pages sit under
[`modular-components/backends/`](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/),
one directory per engine.
