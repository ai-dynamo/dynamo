---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Architecture, configuration, and operational details for the SGLang backend
---

## Overview

The SGLang backend in Dynamo uses a modular architecture where `main.py` dispatches to specialized initialization modules based on the worker type. Each worker type has its own init module, request handler, health check, and registration logic.

Dynamo SGLang uses SGLang's native argument parser -- all SGLang engine arguments (e.g., `--model-path`, `--tp`, `--trust-remote-code`) are passed through directly. Dynamo adds its own arguments for worker mode selection, tokenizer control, and disaggregation configuration.

### Worker Types

| Worker Type | Description |
|------------|-------------|
| **Decode** *(default)* | Standard LLM inference (aggregated or disaggregated decode) |
| **Prefill** | Disaggregated prefill phase (`--disaggregation-mode prefill`) |
| **Embedding** | Text embedding models (`--embedding-worker`) |
| **Multimodal Encode** | Frontend-facing: vision encoding, embeddings generation (`--enable-multimodal --disaggregation-mode encode`) |
| **Native Multimodal P/D** | Normal prefill/decode workers that pass raw media to SGLang (`--enable-multimodal --disaggregation-mode prefill/decode`) |
| **Internal Multimodal Worker** | E/PD or E/P/D worker that consumes embeddings from a separate encode worker (`--enable-multimodal --dedicated-mm-encoder --disaggregation-mode pd/decode`) |
| **Internal Multimodal Prefill** | Prefill phase for E/P/D multimodal disaggregation (`--enable-multimodal --dedicated-mm-encoder --disaggregation-mode prefill`) |
| **Image Diffusion** | Image generation via DiffGenerator (`--image-diffusion-worker`) |
| **Video Generation** | Text/image-to-video via DiffGenerator (`--video-generation-worker`) |
| **LLM Diffusion** | Diffusion language models like LLaDA (`--dllm-algorithm <algo>`) |

## Argument Reference

### Dynamo-Specific Arguments

These arguments are added by Dynamo on top of SGLang's native arguments. For the complete field-by-field reference of every Dynamo SGLang flag, default, and environment variable, see [SGLang Configuration](../../../../../reference/backends/sglang-configuration.mdx).

| Argument | Env Var | Default | Description |
|----------|---------|---------|-------------|
| `--endpoint` | `DYN_ENDPOINT` | Auto-generated | Dynamo endpoint in `dyn://namespace.component.endpoint` format |
| `--use-sglang-tokenizer` | `DYN_SGL_USE_TOKENIZER` | `false` | **[Deprecated]** Use `--dyn-chat-processor sglang` on the frontend instead. See [SGLang Chat Processor](chat-processor.md). |
| `--dyn-tool-call-parser` | `DYN_TOOL_CALL_PARSER` | `None` | [Tool call](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#supported-tool-call-parsers) parser (overrides SGLang's `--tool-call-parser`) |
| `--dyn-reasoning-parser` | `DYN_REASONING_PARSER` | `None` | [Reasoning](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#supported-reasoning-parsers) parser for chain-of-thought models |
| `--dyn-default-thinking-mode` | `DYN_DEFAULT_THINKING_MODE` | `None` | Deployment-level `enabled` or `disabled` [thinking default](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#deployment-level-thinking-default); explicit request controls take precedence |
| `--custom-jinja-template` | `DYN_CUSTOM_JINJA_TEMPLATE` | `None` | Custom chat template path (incompatible with `--use-sglang-tokenizer`) |
| `--embedding-worker` | `DYN_SGL_EMBEDDING_WORKER` | `false` | Run as embedding worker (also sets SGLang's `--is-embedding`) |
| `--enable-multimodal` | `DYN_SGL_ENABLE_MULTIMODAL` | `false` | Allow [multimodal](multimodal.md) inputs on this worker |
| `--dedicated-mm-encoder` | `DYN_SGL_DEDICATED_MM_ENCODER` | `false` | Select the internal encode-worker topology for multimodal PD/P/D workers |
| `--image-diffusion-worker` | `DYN_SGL_IMAGE_DIFFUSION_WORKER` | `false` | Run as [image diffusion](../../../../../use-cases/diffusion/workflows/text-to-image.md#sglang) worker |
| `--video-generation-worker` | `DYN_SGL_VIDEO_GENERATION_WORKER` | `false` | Run as [video generation](../../../../../use-cases/diffusion/workflows/text-to-video.md#sglang) worker |
| `--disagg-config` | `DYN_SGL_DISAGG_CONFIG` | `None` | Path to YAML disaggregation config file |
| `--disagg-config-key` | `DYN_SGL_DISAGG_CONFIG_KEY` | `None` | Key to select from disaggregation config (e.g., `prefill`, `decode`) |
| `--engine-route` | `DYN_SGLANG_ENGINE_ROUTES` | `None` | Expose a startup-allowlisted SGLang Engine or tokenizer-manager method under `/engine/<path>`; repeat the flag or separate environment descriptors with whitespace |

> [!NOTE]
> `--disagg-config` and `--disagg-config-key` must be provided together. The selected section is written to a temp YAML file and passed to SGLang's `--config` flag.

The current supported parser names for both flags are documented in [Tool Call Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#supported-tool-call-parsers) and [Reasoning Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#supported-reasoning-parsers).

## Tokenizer Behavior

By default, Dynamo handles tokenization and detokenization through its Rust-based frontend, passing `input_ids` to SGLang. This enables all frontend endpoints (`v1/chat/completions`, `v1/completions`, `v1/embeddings`).

For SGLang-native preprocessing (tool calling, reasoning parsing, chat templates), use `--dyn-chat-processor sglang` on the frontend. See [SGLang Chat Processor](chat-processor.md) for architecture and usage.

> [!WARNING]
> `--use-sglang-tokenizer` is deprecated. Use `--dyn-chat-processor sglang` on the frontend instead, which provides the same SGLang-native processing with KV router support and the completions endpoint.

## Request Cancellation

When a client disconnects, Dynamo automatically cancels the in-flight request across all workers, freeing compute resources. A background cancellation monitor detects disconnection and aborts the SGLang request.

| Mode | Prefill | Decode |
|------|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ⚠️ | ✅ |

> [!WARNING]
> Cancellation during remote prefill in disaggregated mode is not currently supported.

For details on the cancellation architecture, see [Request Cancellation](../../../concepts/fault-tolerance/request-cancellation-architecture.md).

## Graceful Shutdown

SGLang workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete
3. **Deferred handlers**: SGLang's internal signal handlers (captured during startup via monkey-patching `loop.add_signal_handler`) are invoked after the graceful period

This ensures zero dropped requests during rolling updates or scale-down events.

For more details, see [Graceful Shutdown](../../../../../kubernetes/fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type has a specialized health check payload that validates the full inference pipeline:

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_new_tokens=1`) |
| Prefill | Wrapped prefill-specific request structure |
| Image Diffusion | Minimal image generation request |
| Video Generation | Minimal video generation request |
| Embedding | Standard embedding request |

Health checks are registered with the Dynamo runtime and called by the frontend or Kubernetes liveness probes. See [Observability Architecture](../../../concepts/observability-architecture.md#active-worker-health-checks) for the active health-check design.

## Metrics and KV Events

### Prometheus Metrics

Enable metrics with `--enable-metrics` on the worker. Set `DYN_SYSTEM_PORT` to expose the `/metrics` endpoint:

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --enable-metrics
```

Both SGLang engine metrics (`sglang:*` prefix) and Dynamo runtime metrics (`dynamo_*` prefix) are served from the same endpoint.

For metric details, see [SGLang Observability](observability.md). For a visualization
walkthrough, see [Metrics and Dashboards](../../../../../cli/operations/observability.mdx#view-metrics-and-dashboards).

### KV Events

`--router-mode kv` configures the frontend router but does not enable SGLang's KV event publisher.
For event-driven KV routing, pass `--kv-events-config` to every aggregated worker. In a standard
disaggregated deployment, pass it to every prefill worker:

```bash
python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B \
  --stream-interval 20 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
```

The SGLang JSON must include `"publisher":"zmq"` and an explicit `"endpoint"`. The `"topic"`
field is optional. SGLang does not recognize `{"enable_kv_cache_events":true}` as event
enablement. Do not reuse this vLLM-only form.

Without `--kv-events-config`, SGLang does not publish KV events. The configuration makes SGLang emit
raw cache events over ZMQ to its Dynamo worker. Dynamo normalizes and coalesces the events. It then
updates the worker-local indexer and republishes the state updates to the router. This publication
uses Dynamo's event plane. Dynamo enables the worker-local indexer after it creates the publisher.
The worker-local indexer does not enable the SGLang event source.

SGLang does not support Dynamo's experimental conditional-disaggregation bypass. Therefore,
standard disaggregated deployments do not need KV event publication on decode workers. For workers
that share a network namespace, use a unique ZMQ endpoint port.

For approximate routing without engine events, omit `--kv-events-config` from all workers. Start
the frontend with `--no-router-kv-events`. You can instead set `DYN_ROUTER_USE_KV_EVENTS=false`.

For DP attention mode (`--enable-dp-attention`), the publisher handles multiple DP ranks per node, each with its own KV event stream.

### Recommended Stream Interval

Start with `--stream-interval 20` on SGLang workers. The default interval of `1` can produce one
engine output for each generated token. An interval of `20` reduces host-side output processing
and Dynamo bridge crossings under load. The tradeoff is coarser stream updates.

The SGLang frontend processor has a separate default interval of `20`. To override this value, set
`DYN_SGLANG_STREAM_INTERVAL` on the frontend. The worker flag does not propagate to the frontend.
When fine-grained stream updates are more important than host efficiency, use a lower interval.

## Engine Routes

SGLang workers expose operational endpoints via Dynamo's system server:

| Route | Description |
|-------|-------------|
| `/engine/control/start_profile` | Start PyTorch profiling |
| `/engine/control/stop_profile` | Stop profiling and save traces |
| `/engine/control/release_memory_occupation` | Release GPU memory for maintenance |
| `/engine/control/resume_memory_occupation` | Resume GPU memory after release |
| `/engine/control/update_weights_from_disk` | Update model weights from disk |
| `/engine/control/update_weights_from_tensor` | Update model weights from tensor payload |
| `/engine/control/update_weights_from_distributed` | Update model weights from distributed source |
| `/engine/control/update_weights_from_ipc` | Update model weights from IPC payload |
| `/engine/control/update_weight_version` | Update weight version metadata |

### Configurable SGLang Method Routes

Set `--engine-route` or `DYN_SGLANG_ENGINE_ROUTES` to expose additional SGLang methods on a worker's system server without changing Dynamo. Each descriptor uses this grammar:

```text
<path>[=<method>][:engine|tm]
```

- `<path>` becomes `/engine/<path>` and can contain `/`-separated segments.
- `<method>` defaults to `<path>`. Use `server_info=get_server_info` to expose an alias. A path containing `/` therefore needs an explicit method.
- The target defaults to the `sgl.Engine` instance. Set `:tm` to target its tokenizer manager.
- Repeat `--engine-route` for multiple routes. Separate multiple `DYN_SGLANG_ENGINE_ROUTES` descriptors with whitespace.

For per-worker generation control with NVIDIA Collective Communications Library (NCCL) weight updates, expose this allowlist:

```bash
export DYN_SGLANG_ENGINE_ROUTES="server_info=get_server_info pause_generation:tm flush_cache init_weights_update_group update_weights_from_distributed:tm destroy_weights_update_group continue_generation:tm"
python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B
```

Call these routes on each worker's system-server address. Send inference requests to the Dynamo frontend and reserve the system server for worker control.

SGLang 0.5.15 exposes `get_internal_state` only on the tokenizer manager, where it returns a list with one state dictionary per data-parallel rank. The Engine `get_server_info` adapter merges the server arguments and scheduler topology into the top-level dictionary expected by control clients, so `/engine/server_info` returns that dictionary directly without a `result` envelope.

The `update_weights_from_distributed:tm` form accepts SGLang's tokenizer-manager request schema, including `weight_version`. The public SGLang Engine wrapper omits that field. This configurable `/engine/update_weights_from_distributed` route is separate from Dynamo's fixed `/engine/control/update_weights_from_distributed` route.

Configured Engine methods receive JSON object keys as keyword arguments. For tokenizer-manager methods with an annotated dataclass or msgspec request, Dynamo constructs the request object from the JSON body and supplies `None` for an HTTP request parameter when the SGLang signature requires one. Async methods are awaited, sync Engine wrappers run outside the server loop, and nested results are converted to JSON. Empty request bodies become `{}`. Configured routes accept the ordinary HTTP methods handled by the Dynamo system server.

Dynamo parses and resolves the complete route list during worker startup. Startup fails for malformed descriptors, unknown targets, duplicate paths, private methods, missing methods, or non-callable attributes. An unconfigured method has no route. Request content cannot select or replace the method captured for a configured path, and failure values such as `{"success": false}` remain failures in the response.

> [!WARNING]
> Treat each configured descriptor as remote administrative access to that SGLang method. Expose only the required allowlist and restrict network access to the Dynamo system server.

## See Also

- **[Examples](../../../../../recipes/cli-templates/sglang.mdx)**: Local deployment launch scripts
- **[Disaggregation](disaggregation.md)**: P/D architecture and KV transfer
- **[Diffusion](../../../../../use-cases/diffusion/overview.md)**: LLM, image, and video diffusion models
- **[Configuration and Tuning](../../router/configuration-and-tuning.md)**: KV-aware routing configuration
