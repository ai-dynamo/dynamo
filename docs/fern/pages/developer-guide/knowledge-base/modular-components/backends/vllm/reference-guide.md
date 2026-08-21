---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Configuration, arguments, and operational details for the vLLM backend
---

## Overview

The vLLM backend in Dynamo integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting.

Dynamo vLLM uses vLLM's native argument parser — all vLLM engine arguments are passed through directly. Dynamo adds its own arguments for disaggregation mode, KV transfer, and prompt embeddings.

## Argument Reference

The vLLM backend accepts all upstream vLLM engine arguments plus Dynamo-specific arguments. The authoritative source is always the CLI:

```bash
python -m dynamo.vllm --help
```

The `--help` output is organized into the following groups:

- **Dynamo Runtime Options** — Namespace, discovery backend, request/event plane, endpoint types, tool/reasoning parsers, and custom chat templates. These are common across all Dynamo backends and use `DYN_*` env vars. See [Runtime Configuration](../../../../../reference/components/runtime-configuration.mdx) for the full field reference.
- **Dynamo vLLM Options** — Disaggregation mode, tokenizer selection, sleep mode, multimodal flags, vLLM-Omni pipeline configuration, headless mode, and ModelExpress. These use `DYN_VLLM_*` env vars. See [vLLM Configuration](../../../../../reference/backends/vllm-configuration.mdx) for the full field reference.
- **vLLM Engine Options** — All native vLLM arguments (`--model`, `--tensor-parallel-size`, `--kv-transfer-config`, `--kv-events-config`, `--enable-prefix-caching`, etc.). See the [vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Tool and Reasoning Parsers

Use `--dyn-tool-call-parser` and `--dyn-reasoning-parser` to match the model's output format when the model emits tool calls and/or reasoning content. The current supported values are documented in [Tool Call Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#supported-tool-call-parsers) and [Reasoning Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#supported-reasoning-parsers).

To set the thinking mode used when a request omits an explicit control, pass
`--dyn-default-thinking-mode enabled|disabled` or set
`DYN_DEFAULT_THINKING_MODE`. Request-level thinking controls, including
adaptive thinking, take precedence. See
[Deployment-Level Thinking Default](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#deployment-level-thinking-default).

For reasoning models with structured output (`response_format`, JSON schema,
or required/named tool choice), configure both reasoning parsers on the worker:

```bash
python -m dynamo.vllm --model <model> \
  --reasoning-parser <vllm-parser> \
  --dyn-reasoning-parser <dynamo-parser>
```

The vLLM parser delays grammar enforcement until reasoning ends; the Dynamo
parser populates `reasoning_content`. Parser names can differ between registries.

### Priority Scheduling

vLLM engine-level request priority is controlled by the upstream vLLM
`--scheduling-policy priority` argument.

```bash
python -m dynamo.vllm \
    --model <model> \
    --scheduling-policy priority
```

Clients still send the Dynamo API value directly:
`nvext.agent_hints.priority`. Higher values mean higher priority at the Dynamo
API layer. Dynamo converts that value before passing it to vLLM, which uses a
different native priority polarity internally.

Do not negate `nvext.agent_hints.priority` in the client for vLLM. If you are
also using the router queue, configure the frontend-side
`--router-queue-threshold` separately; vLLM engine scheduling only applies
after a request reaches the worker.

For the cross-layer behavior, see
[Priority Scheduling](../../../../../use-cases/agents/priority-scheduling.md). For the upstream
flag definition, see the
[vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Prompt Embeddings

Dynamo supports [vLLM prompt embeddings](https://docs.vllm.ai/en/stable/features/prompt_embeds.html) — pre-computed embeddings bypass tokenization in the Rust frontend and are decoded to tensors in the worker.

- Enable with `--enable-prompt-embeds` (disabled by default)
- Embeddings are sent as base64-encoded PyTorch tensors via the `prompt_embeds` field in the Completions API
- NATS must be configured with a 15MB max payload for large embeddings (already set in default deployments)

## KV Transfer Connectors for Disaggregated Serving

Disaggregated serving moves the KV cache from the prefill worker to the decode worker over a connector you select with `--kv-transfer-config`. Dynamo supports two NIXL connectors, which differ in who moves the blocks:

| Connector | Direction | Handoff |
|-----------|-----------|---------|
| `NixlConnector` | Decode READs from prefill | Decode learns prefill's block locations from the prefill response |
| `NixlPushConnector` | Prefill WRITEs into decode | Decode registers its allocated blocks with prefill, which pushes into them |

Configure the same connector on both the prefill and decode workers:

```bash
python -m dynamo.vllm ... \
  --kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_both"}'
```

Push mode lets the two legs overlap. Because decode never needs anything computed during prefill — only the prefill engine's identity — the frontend dispatches both legs at once, so decode is already holding allocated blocks when prefill finishes and the transfer starts immediately. Pull mode cannot overlap: its handoff carries block IDs that do not exist until prefill has run.

To make that possible, a prefill worker running `NixlPushConnector` publishes its NIXL side-channel address and engine ID to discovery. If it cannot, the handoff falls back to running sequentially — still correct, because the prefill worker holds its finished blocks until decode's late registration arrives, but without the overlap.

> [!NOTE]
> Push mode reports no cached-prompt-token details in usage, and does not short-circuit a request that reaches a stop condition during the one-token prefill step. Both follow from dispatching decode before the prefill response exists, and apply equally to SGLang's bootstrap handoff.

### Data Parallelism and Push Mode

vLLM gives each data-parallel rank its own NIXL side channel, at `VLLM_NIXL_SIDE_CHANNEL_PORT` plus the rank index. A Dynamo worker registers one address, so it can only advertise push coordinates when it fronts exactly one rank.

vLLM also names the NIXL agent per rank: the engine ID is `<base>_dp<rank>` whenever `--data-parallel-size` is greater than 1 or the rank is non-zero, and the unsuffixed base ID otherwise — so tensor- or expert-parallel deployments with a single rank keep the base ID. Dynamo mirrors that derivation when it advertises. Publishing the wrong form is not a loud failure: the peer rejects the handshake with `Remote NIXL agent engine ID mismatch` and the transfer falls back.

This is satisfied by every deployment except vLLM's *internal* DP load balancing with `--data-parallel-size` greater than 1, where a single worker process fronts all ranks:

| Deployment | Ranks per worker | Advertises |
|------------|------------------|------------|
| No data parallelism | 1 | ✅ |
| External DP load balancing | 1 | ✅ |
| Hybrid DP load balancing, one local rank | 1 | ✅ |
| Hybrid DP load balancing, several local ranks | many | ❌ |
| Internal DP load balancing, `--data-parallel-size` > 1 | all | ❌ |

A worker that cannot advertise logs a warning at startup and runs the sequential handoff. Push mode still works — it just loses the overlap — so this is a performance limitation, not a correctness one. Use external or hybrid DP load balancing with one rank per worker to get the overlapped path; both are recommended for multi-worker DP deployments regardless.

## Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's built-in hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```

See the high-level notes in [Router Design](../../router/router-design.md#deterministic-event-ids) on deterministic event IDs.

## Graceful Shutdown

vLLM workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete (configurable via `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
3. **Resource cleanup**: Engine resources and temporary files (Prometheus dirs, LoRA adapters) are released

All vLLM endpoints use `graceful_shutdown=True`, meaning they wait for in-flight requests to finish before exiting. An internal `VllmEngineMonitor` also checks engine health every 2 seconds and initiates shutdown if the engine becomes unresponsive.

For more details, see [Graceful Shutdown](../../../../../kubernetes/fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type has a specialized health check payload that validates the full inference pipeline:

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_tokens=1`) using the model's BOS token |
| Prefill | Same payload structure as decode, adapted for prefill request format |
| vLLM-Omni | Short generation request via AsyncOmni with the model's BOS token |

Health checks are registered with the Dynamo runtime and called by the frontend or Kubernetes liveness probes. The payload can be overridden via `DYN_HEALTH_CHECK_PAYLOAD` environment variable. See [Observability Architecture](../../../concepts/observability-architecture.md#active-worker-health-checks) for the active health-check design.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources.

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../../concepts/fault-tolerance/request-cancellation-architecture.md) documentation.

## Request Migration

Dynamo supports [request migration](../../../../../kubernetes/fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../../../../kubernetes/fault-tolerance/request-migration.md) documentation for configuration details.

## See Also

- **[Examples](../../../../../recipes/cli-templates/vllm.mdx)**: Local deployment launch scripts
- **[vLLM README](overview.md)**: Quick start and feature overview
- **[Observability](observability.md)**: Metrics and monitoring setup
- **[Configuration and Tuning](../../router/configuration-and-tuning.md)**: KV-aware routing configuration
- **[Fault Tolerance](../../../../../kubernetes/fault-tolerance/introduction.md)**: Request migration, cancellation, and graceful shutdown
