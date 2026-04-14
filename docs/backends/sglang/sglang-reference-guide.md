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
| **Multimodal Encode** | Frontend-facing: vision encoding, embeddings generation (`--multimodal-encode-worker`) |
| **Multimodal Worker** | LLM inference with multimodal data (`--multimodal-worker`) |
| **Multimodal Prefill** | Prefill phase for multimodal disaggregation (`--multimodal-worker --disaggregation-mode prefill`) |
| **Image Diffusion** | Image generation via DiffGenerator (`--image-diffusion-worker`) |
| **Video Generation** | Text/image-to-video via DiffGenerator (`--video-generation-worker`) |
| **LLM Diffusion** | Diffusion language models like LLaDA (`--dllm-algorithm <algo>`) |

## Argument Reference

### Dynamo-Specific Arguments

These arguments are added by Dynamo on top of SGLang's native arguments.

| Argument | Env Var | Default | Description |
|----------|---------|---------|-------------|
| `--endpoint` | `DYN_ENDPOINT` | Auto-generated | Dynamo endpoint in `dyn://namespace.component.endpoint` format |
| `--use-sglang-tokenizer` | `DYN_SGL_USE_TOKENIZER` | `false` | **[Deprecated]** Use `--dyn-chat-processor sglang` on the frontend instead. See [SGLang Chat Processor](sglang-chat-processor.md). |
| `--dyn-tool-call-parser` | `DYN_TOOL_CALL_PARSER` | `None` | [Tool call](../../agents/tool-calling.md#supported-tool-call-parsers) parser (overrides SGLang's `--tool-call-parser`) |
| `--dyn-reasoning-parser` | `DYN_REASONING_PARSER` | `None` | [Reasoning](../../agents/reasoning.md#supported-reasoning-parsers) parser for chain-of-thought models |
| `--custom-jinja-template` | `DYN_CUSTOM_JINJA_TEMPLATE` | `None` | Custom chat template path (incompatible with `--use-sglang-tokenizer`) |
| `--embedding-worker` | `DYN_SGL_EMBEDDING_WORKER` | `false` | Run as embedding worker (also sets SGLang's `--is-embedding`) |
| `--multimodal-encode-worker` | `DYN_SGL_MULTIMODAL_ENCODE_WORKER` | `false` | Run as [multimodal](../../features/multimodal/multimodal-sglang.md) encode worker (frontend-facing) |
| `--multimodal-worker` | `DYN_SGL_MULTIMODAL_WORKER` | `false` | Run as multimodal LLM worker |
| `--image-diffusion-worker` | `DYN_SGL_IMAGE_DIFFUSION_WORKER` | `false` | Run as [image diffusion](sglang-diffusion.md#image-diffusion) worker |
| `--video-generation-worker` | `DYN_SGL_VIDEO_GENERATION_WORKER` | `false` | Run as [video generation](sglang-diffusion.md#video-generation) worker |
| `--disagg-config` | `DYN_SGL_DISAGG_CONFIG` | `None` | Path to YAML disaggregation config file |
| `--disagg-config-key` | `DYN_SGL_DISAGG_CONFIG_KEY` | `None` | Key to select from disaggregation config (e.g., `prefill`, `decode`) |

<Note>
`--disagg-config` and `--disagg-config-key` must be provided together. The selected section is written to a temp YAML file and passed to SGLang's `--config` flag.
</Note>

The current supported parser names for both flags are documented in [Tool Calling](../../agents/tool-calling.md#supported-tool-call-parsers) and [Reasoning](../../agents/reasoning.md#supported-reasoning-parsers).

## Tokenizer Behavior

By default, Dynamo handles tokenization and detokenization through its Rust-based frontend, passing `input_ids` to SGLang. This enables all frontend endpoints (`v1/chat/completions`, `v1/completions`, `v1/embeddings`).

For SGLang-native preprocessing (tool calling, reasoning parsing, chat templates), use `--dyn-chat-processor sglang` on the frontend. See [SGLang Chat Processor](sglang-chat-processor.md) for architecture and usage.

<Warning>
`--use-sglang-tokenizer` is deprecated. Use `--dyn-chat-processor sglang` on the frontend instead, which provides the same SGLang-native processing with KV router support and the completions endpoint.
</Warning>

## Request Cancellation

When a client disconnects, Dynamo automatically cancels the in-flight request across all workers, freeing compute resources. A background cancellation monitor detects disconnection and aborts the SGLang request.

| Mode | Prefill | Decode |
|------|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ⚠️ | ✅ |

<Warning>Cancellation during remote prefill in disaggregated mode is not currently supported.</Warning>

For details on the cancellation architecture, see [Request Cancellation](../../fault-tolerance/request-cancellation.md).

## Graceful Shutdown

SGLang workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete
3. **Deferred handlers**: SGLang's internal signal handlers (captured during startup via monkey-patching `loop.add_signal_handler`) are invoked after the graceful period

This ensures zero dropped requests during rolling updates or scale-down events.

For more details, see [Graceful Shutdown](../../fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type has a specialized health check payload that validates the full inference pipeline:

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_new_tokens=1`) |
| Prefill | Wrapped prefill-specific request structure |
| Image Diffusion | Minimal image generation request |
| Video Generation | Minimal video generation request |
| Embedding | Standard embedding request |

Health checks are registered with the Dynamo runtime and called by the frontend or Kubernetes liveness probes. See [Health Checks](../../observability/health-checks.md) for the broader health check architecture.

## Metrics and KV Events

### Prometheus Metrics

Enable metrics with `--enable-metrics` on the worker. Set `DYN_SYSTEM_PORT` to expose the `/metrics` endpoint:

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --enable-metrics
```

Both SGLang engine metrics (`sglang:*` prefix) and Dynamo runtime metrics (`dynamo_*` prefix) are served from the same endpoint.

For metric details, see [SGLang Observability](sglang-observability.md). For visualization setup, see [Prometheus + Grafana](../../observability/prometheus-grafana.md).

### KV Events

When configured with `--kv-events-config`, workers publish KV cache events (block creation/deletion) for the [KV-aware router](../../components/router/README.md). Events are published via ZMQ from SGLang's scheduler and relayed through Dynamo's event plane.

For DP attention mode (`--enable-dp-attention`), the publisher handles multiple DP ranks per node, each with its own KV event stream.

## Engine Routes

SGLang workers expose operational endpoints via Dynamo's system server:

| Route | Description |
|-------|-------------|
| `/engine/start_profile` | Start PyTorch profiling |
| `/engine/stop_profile` | Stop profiling and save traces |
| `/engine/release_memory_occupation` | Release GPU memory for maintenance |
| `/engine/resume_memory_occupation` | Resume GPU memory after release |
| `/engine/update_weights_from_distributor` | Update model weights from distributor |
| `/engine/update_weights_from_disk` | Update model weights from disk |
| `/engine/update_weight_version` | Update weight version metadata |

## See Also

- **[Examples](sglang-examples.md)**: All deployment patterns
- **[Disaggregation](sglang-disaggregation.md)**: P/D architecture and KV transfer
- **[Diffusion](sglang-diffusion.md)**: LLM, image, and video diffusion models
- **[Configuration and Tuning](../../components/router/router-configuration.md)**: KV-aware routing configuration
