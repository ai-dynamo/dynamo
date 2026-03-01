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

### Key Configuration Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model to serve (e.g., `Qwen/Qwen3-0.6B`) |
| `--disaggregation-mode` | `agg` | Worker role: `prefill`, `decode`, or `agg` (aggregated) |
| `--kv-transfer-config` | `None` | JSON string for vLLM KVTransferConfig (e.g., `'{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`) |
| `--kv-events-config` | `None` | JSON string for KV event publishing config |
| `--enable-prompt-embeds` | `false` | Enable [prompt embeddings](prompt-embeddings.md) feature (opt-in) |
| `--block-size` | vLLM default | Block size for KV cache management |
| `--enforce-eager` | `false` | Disable CUDA graphs for quick deployment (remove for production) |
| `--data-parallel-rank` | `None` | DP rank for expert parallelism deployments |
| `--data-parallel-size` | `None` | Total DP size for expert parallelism deployments |
| `--enable-expert-parallel` | `false` | Enable expert parallelism for MoE models |
| `--metrics-endpoint-port` | `None` | Port for publishing KV metrics to Dynamo |

### Upstream vLLM Arguments

Dynamo uses the same argument parser as vLLM. Run `vllm serve --help` to see all available CLI arguments, or consult the [vLLM serve args documentation](https://docs.vllm.ai/en/v0.9.2/configuration/serve_args.html).

See `args.py` in the Dynamo source for the full list of Dynamo-specific overrides.

### Prompt Embeddings

The `--enable-prompt-embeds` flag enables accepting pre-computed prompt embeddings via the API:

- **Default behavior:** Prompt embeddings DISABLED — requests with `prompt_embeds` will fail
- **Error without flag:** `ValueError: You must set --enable-prompt-embeds to input prompt_embeds`

See [Prompt Embeddings](prompt-embeddings.md) for full documentation.

## Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's builtin hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```

See the high-level notes in [Router Design](../../design-docs/router-design.md#deterministic-event-ids) on deterministic event IDs.

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
- **[Prometheus](prometheus.md)**: Metrics and monitoring setup
- **[Router Guide](../../components/router/router-guide.md)**: KV-aware routing configuration
- **[Fault Tolerance](../../fault-tolerance/README.md)**: Request migration, cancellation, and graceful shutdown
