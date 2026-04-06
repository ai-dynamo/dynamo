---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Configuration, arguments, and operational details for the vLLM backend
---

# Reference Guide

## Overview

The vLLM backend in Dynamo integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting.

Dynamo vLLM uses vLLM's native argument parser — all vLLM engine arguments are passed through directly. Dynamo adds its own arguments for disaggregation mode, KV transfer, and prompt embeddings.

## Argument Reference

The vLLM backend accepts all upstream vLLM engine arguments plus Dynamo-specific arguments. The authoritative source is always the CLI:

```bash
python -m dynamo.vllm --help
```

The `--help` output is organized into the following groups:

- **Dynamo Runtime Options** — Namespace, discovery backend, request/event plane, endpoint types, tool/reasoning parsers, and custom chat templates. These are common across all Dynamo backends and use `DYN_*` env vars.
- **Dynamo vLLM Options** — Disaggregation mode, tokenizer selection, sleep mode, multimodal flags, vLLM-Omni pipeline configuration, headless mode, and ModelExpress. These use `DYN_VLLM_*` env vars.
- **vLLM Engine Options** — All native vLLM arguments (`--model`, `--tensor-parallel-size`, `--kv-transfer-config`, `--kv-events-config`, `--enable-prefix-caching`, etc.). See the [vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Tool and Reasoning Parsers

Use `--dyn-tool-call-parser` and `--dyn-reasoning-parser` to match the model's output format when the model emits tool calls and/or reasoning content. The current supported values are documented in [Tool Calling](../../agents/tool-calling.md#supported-tool-call-parsers) and [Reasoning](../../agents/reasoning.md#supported-reasoning-parsers).

### Prompt Embeddings

Dynamo supports [vLLM prompt embeddings](https://docs.vllm.ai/en/stable/features/prompt_embeds.html) — pre-computed embeddings bypass tokenization in the Rust frontend and are decoded to tensors in the worker.

- Enable with `--enable-prompt-embeds` (disabled by default)
- Embeddings are sent as base64-encoded PyTorch tensors via the `prompt_embeds` field in the Completions API
- NATS must be configured with a 15MB max payload for large embeddings (already set in default deployments)

## Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's built-in hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```

See the high-level notes in [Router Design](../../design-docs/router-design.md#deterministic-event-ids) on deterministic event IDs.

## Graceful Shutdown

vLLM workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete (configurable via `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
3. **Resource cleanup**: Engine resources and temporary files (Prometheus dirs, LoRA adapters) are released

All vLLM endpoints use `graceful_shutdown=True`, meaning they wait for in-flight requests to finish before exiting. An internal `VllmEngineMonitor` also checks engine health every 2 seconds and initiates shutdown if the engine becomes unresponsive.

For more details, see [Graceful Shutdown](../../fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type has a specialized health check payload that validates the full inference pipeline:

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_tokens=1`) using the model's BOS token |
| Prefill | Same payload structure as decode, adapted for prefill request format |
| vLLM-Omni | Short generation request via AsyncOmni with the model's BOS token |

Health checks are registered with the Dynamo runtime and called by the frontend or Kubernetes liveness probes. The payload can be overridden via `DYN_HEALTH_CHECK_PAYLOAD` environment variable. See [Health Checks](../../observability/health-checks.md) for the broader health check architecture.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources.

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../fault-tolerance/request-cancellation.md) documentation.

## Request Migration

Dynamo supports [request migration](../../fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../fault-tolerance/request-migration.md) documentation for configuration details.

## See Also

- **[Examples](vllm-examples.md)**: All deployment patterns with launch scripts
- **[vLLM README](README.md)**: Quick start and feature overview
- **[Observability](vllm-observability.md)**: Metrics and monitoring setup
- **[Router Guide](../../components/router/router-guide.md)**: KV-aware routing configuration
- **[Fault Tolerance](../../fault-tolerance/README.md)**: Request migration, cancellation, and graceful shutdown
