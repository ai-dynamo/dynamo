<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar support notes

> [!NOTE]
> Sidecar integration is under active development. These notes can become out of date when Dynamo or a native engine changes.

Check the versions in your deployment.

## vLLM

The standard generation path requires one completion per request (`n=1`). It accepts image, video, and audio input. It does not accept prompt embeddings or per-request `mm_processor_kwargs`. The sidecar provides basic structured output. It supports fewer constraint controls than the in-process backend.

Runtime weight updates require the engine to report support for weight updates. Sleep and wake controls require engine support for sleep mode. Configure speculative decoding in the native engine. Test each model, algorithm, routing, disaggregation, and telemetry combination separately.

If an output stream closes early without an error, the sidecar treats the closure as a protocol error. Request migration does not retry this error. See the [shared operational limits](#operational-limits) for migration rules, observability, planning, cache offloading, and shutdown behavior.

## SGLang

The standard gRPC generation path requires one completion per request (`n=1`). It accepts image, video, and audio input as `http://`, `https://`, or `data:` URLs. The sidecar rejects per-request `mm_processor_kwargs` on media inputs. The sidecar provides basic structured output. It supports fewer constraint controls than the in-process backend.

Run one Dynamo sidecar per native SGLang encoder. Enable `--route-to-encoder` on the prefill or aggregated sidecar and set `SGLANG_EXTERNAL_ENCODER_HANDOFF=1` on each language server. Set `--encoder-advertise-endpoint` if the language server reaches E at a different address from its sidecar. The staged path supports Qwen2-VL, Qwen2.5-VL and Qwen3-VL (including MoE) with TP=PP=DP=1.

Dynamo sends all images and videos in a request to one encoder. Text requests bypass E. Chat accepts `image_url` and `video_url`. Native `/generate` accepts a URL string or flat list for each of `image_data` and `video_data`. The original `input_ids` determine their order. Image-only requests retain the v1 handoff.

The launcher defaults to NIXL/UCX. P or PD registers host memory for the encoder to write into. Install NIXL with its UCX plugin on both ends, or select ZMQ with `--encoder-transfer-backend zmq_to_scheduler`. In E/P/D, D receives the layout and video timing to reconstruct token positions. The P→D KV transfer backend is configured separately. See the [E/PD and E/P/D launcher](sglang/launch/encoded.sh).

To serve video, deploy video-capable builds on every E and every participating language worker for the model in its Dynamo namespace. EncoderRouter binds one encoder worker set and selects replicas round-robin. A mixed rollout can leave the router without a unique encoder set or send a video to an image-only encoder, which rejects it.

For encoders that advertise their media policy, the E sidecar fetches remote media under Dynamo's DNS and redirect checks and the native encoder's allowed domains, size limit, and timeout. The complete Stage request, including base64 expansion, must fit the 64 MiB gRPC limit. On older image-only encoders the native process fetches its own image URLs.

The staged path accepts images and videos. Audio uses the regular generation path when supported by the native engine. Media-aware KV routing on wide-token SGLang builds is not supported. See the [multimodal compatibility limits](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#wide-token-sglang-builds).

SGLang's Rust server downloads HTTP and HTTPS media URLs from any host. This includes `127.0.0.1` and SGLang's own HTTP API. The Rust server does not support `--allowed-media-domains` (v0.5.23 ignores it; later builds refuse to start with it), so limit the engine's network access instead, for example with a Kubernetes NetworkPolicy.

SGLang's Rust server does not support LoRA adapters.

Configure HiCache and speculative decoding in the native engine. The sidecar reports HiCache capacity for configurations that it recognizes. It also handles EAGLE-family routing metadata and draft-token reserves; the reserves need [sgl-project/sglang#45859](https://github.com/sgl-project/sglang/pull/45859), and without it an EAGLE engine gets no token budget. Test each model, algorithm, routing, disaggregation, and telemetry combination separately.

Request migration can retry a stream that closes early without an error. See the [shared operational limits](#operational-limits) for migration rules, observability, planning, cache offloading, and shutdown behavior.

## TensorRT-LLM

The standard generation path requires one completion per request (`n=1`). It accepts images and video when the engine reports support for them. It does not accept separate encoder handoffs or audio. The sidecar provides basic structured output. It supports fewer constraint controls than the in-process backend. The sidecar returns output logprobs (token log probabilities). It does not return prompt logprobs.

The sidecar publishes routing-load snapshots. These are not the per-iteration Forward Pass Metrics (FPM) that load-based planning needs. Request migration does not retry an output stream that closes early without an error. It also does not retry KV-session or KV-transfer failures.

Configure speculative decoding in the native engine. The sidecar passes generation output and native handoff data. Test each model, algorithm, routing, disaggregation, and telemetry combination separately.

The aggregated launcher does not accept custom configuration files. Add `guided_decoding_backend: xgrammar` to your engine configuration. Then start the native engine directly.

See the [shared operational limits](#operational-limits) for migration rules, observability, planning, cache offloading, and shutdown behavior.

## Operational limits

Dynamo provides runtime metrics, logs, and request spans. Collect native engine metrics separately. The sidecars do not provide full tracing inside the engine. They do not provide full support for existing dashboards.

The sidecars do not have a supported workflow for service-level agreement (SLA) autoscaling. The planner can discover workers in Kubernetes. The sidecars do not publish or forward per-iteration Forward Pass Metrics (FPM). The default load-based planner needs these metrics to make scaling decisions. Predictive scaling uses a separate code path. That path does not provide a supported autoscaling workflow for sidecars. See [Planner Limitations](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/planner/overview.md#current-limitations).

GPU Memory Service (GMS), checkpoint/restore, and shadow-engine failover require separate integration. A Dynamo engine container or upstream engine settings can provide components for these integrations. Each backend and deployment mode needs its own setup and tests. These requirements do not mean that every combination works.

The Kubernetes operator rejects sidecar mode with GMS, enabled checkpointing, or shadow-engine failover. This restriction applies to all engine images. Changing the engine image alone does not enable these operator-managed features. See the [sidecar deployment contract](../../docs/fern/pages/reference/kubernetes-api/dynamo-component-deployment.mdx#dynamo-sidecar-mode).

To enable request migration, set `--migration-limit` to a value greater than `0`. Request migration is off by default. Eligible requests can retry connection failures, timeouts, and some interrupted streams. Retries continue from the tokens already sent to the client. Request migration excludes structured-output requests and requests pinned to a worker. Some engine errors cannot trigger request migration. See each engine's notes for the error restrictions. Request migration and shadow-engine failover are separate features.

During shutdown, the shared worker removes its registration. It then waits and releases its resources. Sidecars do not report when all engine work has stopped. The prefill worker therefore waits for a limited time. This wait does not prove that engine KV transfers are complete.

Configure cache offloading in the native engine. Install any required plugin in the engine container. Running the sidecar in a separate process does not prevent caching inside the engine. Test offload-aware routing, storage tiers, and prefill/decode transfer combinations separately. A prefix-cache hit or KV event alone does not prove that these combinations work. Support for a feature does not mean that all model and feature combinations are verified.
