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

Dynamo can select a standalone SGLang encoder (E) for an aggregated language worker (PD). Split prefill and decode workers with an external encoder are not supported: the sidecar refuses `--route-to-encoder` on a prefill or decode worker, and the launcher refuses `--split-pd`.

The encoder is SGLang's unmodified HTTP encoder (`--encoder-only`). Run one Dynamo sidecar per encoder with `--disaggregation-mode encode`. In that mode the sidecar speaks no gRPC, and `--grpc-endpoint host:port` names the encoder's HTTP listener. The encoder reports no metadata, so `--encoder-model-path` names the model to register. Use the language worker's model path. Set `--encoder-served-model-name` when the language worker has a served model name, and `--encoder-tokenizer-path` when the tokenizer is stored elsewhere. Set `--encoder-advertise-endpoint host:port` when the language worker reaches the encoder at a different address than the sidecar does.

Start the language worker as SGLang's Rust server with `SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF=1`, `--language-only` and `--encoder-transfer-backend zmq_to_scheduler`. Start its sidecar with `--route-to-encoder`. SGLang requires PP=1 and one DP rank in this mode. The [E/PD launcher](sglang/launch/encoded.sh) starts all of these. Set `--tp-size N` for tensor parallelism or `--encoder-tp-size N` for a different encoder degree. Each role takes a separate GPU group by default. Use `--colocate` to put E and PD on the same GPUs.

For a request with media, EncoderRouter picks one encoder round-robin and sends it all of the request's media. The encoder sidecar sends the encoder one native `/encode` request per modality, in the order image, video, audio, under the Dynamo request ID. It returns a descriptor that names the encoder and the item count of each modality. The language sidecar forwards the descriptor to SGLang as `encoder_handoff`, with the same request ID as `rid` and the original token IDs. SGLang registers its receive address with that encoder, and the embeddings and their layout travel from E to PD over ZMQ (`zmq_to_scheduler`). No other transfer backend is supported.

`/encode` answers only after the embeddings reach the language worker, so the encoder sidecar leaves it open and waits 200 ms for an early failure. An unreachable encoder fails the request as unavailable, and media or a modality that the encoder rejects fails it as an invalid request. The sidecar logs a failure that arrives later. On a deployment that requires E, an encode failure ends the request before language generation. When a request is cancelled or the language worker definitively rejects it, the frontend tells the selected encoder sidecar to drop that request's open `/encode` calls.

Text requests bypass E. Chat accepts `image_url`, `video_url`, `audio_url` and inline `input_audio`. Native `/generate` accepts a URL string or flat list for each of `image_data`, `video_data` and `audio_data`. The original `input_ids` determine their order. The native encoder and the loaded model decide which modalities work. Dynamo does not check them before the encoder does. The encoder process downloads the media URLs itself, so limit its network access like the Rust server's.

The request ID is SGLang's `rid` for a media request, and the encoder keys its work by it. The encoder delivers a request's embeddings once, so a media request cannot be retried on another language worker.

The encoder sidecar polls the encoder's `/health` every 5 seconds. It stops serving when the encoder cannot be reached or three polls in a row fail. EncoderRouter selects replicas round-robin within one encoder worker set. Multiple encoder worker sets make the target ambiguous.

Chat also requires a compatible frontend tokenizer and chat template. For `moonshotai/Kimi-VL-A3B-Instruct`, the pinned Dynamo Rust tokenizer does not recognize the checkpoint's `kimi_vl` tiktoken configuration. Set `DYN_CHAT_PROCESSOR=sglang`, `DYN_ENDPOINT_TYPES=chat` and `DYN_TRUST_REMOTE_CODE=true` to use the checkpoint tokenizer with the SGLang chat processor in a frontend environment with SGLang installed. Native `/generate` accepts pretokenized input when enabled with `DYN_SGLANG_ENABLE_GENERATE=1`.

Media-aware KV routing on wide-token SGLang builds is not supported. See the [multimodal compatibility limits](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#wide-token-sglang-builds).

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
