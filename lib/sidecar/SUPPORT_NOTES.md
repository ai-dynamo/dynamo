<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar support notes

> [!NOTE]
> Sidecar integration is under active development. These notes may become stale as Dynamo and the native engines evolve. Check the versions you deploy.

## vLLM

The standard generation path requires `n=1`. It accepts image, video, and audio input, but rejects prompt embeddings and per-request `mm_processor_kwargs`. Basic structured output is implemented with a narrower set of constraint controls than the in-process backend.

Runtime weight updates and sleep/wake controls require the engine to advertise the corresponding capabilities. Configure speculative decoding in the native engine; model, algorithm, routing, disaggregation, and telemetry combinations require separate qualification.

Premature clean stream closure is a non-migratable protocol error. See the [shared operational limits](#operational-limits) for migration eligibility, observability, planning, cache offloading, and shutdown behavior.

## SGLang

The standard gRPC generation path requires `n=1` and rejects multimodal input. Basic structured output is implemented with a narrower set of constraint controls than the in-process backend.

The sidecar does not register the engine's LoRA adapter inventory or expose Dynamo-managed adapter loading/unloading. Forwarding an adapter name internally does not make it available through OpenAI model lookup. The separate, opt-in native `/generate` passthrough can select a preloaded adapter using `lora_path`; native passthrough and manually published adapter records are separate integration choices.

Configure HiCache and speculative decoding in the native engine. The sidecar reports HiCache capacity for recognized configurations and handles EAGLE-family routing metadata and draft-token reserves. Model, algorithm, routing, disaggregation, and telemetry combinations require separate qualification.

Premature clean stream closure is eligible for request migration. See the [shared operational limits](#operational-limits) for migration eligibility, observability, planning, cache offloading, and shutdown behavior.

## TensorRT-LLM

The standard generation path requires `n=1`. It accepts images, but rejects separate encoder handoffs, video, and audio. Basic structured output is implemented with a narrower set of constraint controls than the in-process backend. Output logprobs are supported; prompt logprobs are not.

The sidecar publishes routing-load snapshots. These differ from the per-iteration Forward Pass Metrics (FPM) required by load-based planning. Premature clean stream closure and KV-session/transfer failures are non-migratable.

Configure speculative decoding in the native engine. The sidecar preserves generation output and native handoffs, but model, algorithm, routing, disaggregation, and telemetry combinations require separate qualification. The aggregated convenience launcher rejects custom config files; merge the required `guided_decoding_backend: xgrammar` setting into your configuration and start the native engine directly.

See the [shared operational limits](#operational-limits) for migration eligibility, observability, planning, cache offloading, and shutdown behavior.

## Operational limits

Dynamo runtime metrics, logs, and request spans are implemented. Scrape native engine metrics separately; the sidecars do not provide full engine-internal tracing or parity with existing dashboards.

A supported service-level agreement (SLA) autoscaling workflow is not available for these sidecars. Kubernetes worker discovery is implemented, but built-in per-iteration Forward Pass Metrics (FPM) emission/relay is missing, so default load-based scaling has no input and cannot make scaling decisions. A separate predictive-scaling path does not establish an integrated sidecar workflow. See [Planner Limitations](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/overview.md#current-limitations).

The Kubernetes operator rejects sidecar mode combined with GPU Memory Service (GMS), enabled checkpointing, or shadow-engine failover. See the [sidecar deployment contract](../../docs/fern/pages/reference/kubernetes-api/dynamo-component-deployment.mdx#dynamo-sidecar-mode).

Enable request migration with `--migration-limit > 0`; it is off by default. Eligible requests can retry connection failures, timeouts, and supported interrupted streams, continuing from delivered tokens. Structured-output requests and explicit worker pins are excluded, and not all engine errors are migratable. See each engine's support notes for error-specific restrictions. Request migration is distinct from shadow-engine failover.

The shared worker unregisters, waits, and cleans up during shutdown. The sidecars do not report engine quiescence, so prefill draining uses a bounded wait rather than proving that engine KV transfers have completed.

Configure cache offloading in the native engine, with any required plugin installed in the engine container. Sidecar separation does not prevent engine-local caching. Offload-aware routing, storage tiers, and combinations with prefill/decode transfer require separate qualification; a prefix-cache hit or KV event alone does not prove them. Implementation support does not certify every model and feature combination.
