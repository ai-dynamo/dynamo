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

Run one Dynamo sidecar per native SGLang encoder. Enable `--route-to-encoder` on the prefill or aggregated sidecar and set `SGLANG_EXTERNAL_ENCODER_HANDOFF=1` on each language server. The launcher uses DP=PP=1. Set `--tp-size N` for tensor parallelism or `--encoder-tp-size N` for a different encoder degree. Each role takes a separate GPU group by default. Use `--colocate` to put E and the language workers on the same GPUs.

The local launcher creates a temporary, private receipt key shared by its SGLang workers. Across hosts, generate 32 random bytes once with `(umask 077 && openssl rand 32 > encoder-handoff.key)`. Mount the file on each native worker and set `SGLANG_ENCODER_HANDOFF_KEY_FILE` to its path. Set `SGLANG_ENCODER_HANDOFF_ENDPOINT=grpc://host:port` on E if it binds a wildcard or P/PD reaches it at a different address. An explicit sidecar `--encoder-advertise-endpoint` must match that signed endpoint.

Discovery matches the processor layout and receipt key fingerprint across E, P/PD and D. E and P/PD also have to advertise the same embedding transport (NIXL or ZMQ). D consumes metadata and can use either setup. Restart workers with a new key to rotate it. E signs the descriptor and its address. P/PD and D verify the receipt before using it. Each worker validates its available media types on the request. An image-capable encoder can serve images even when only the language worker has video support. On a deployment that requires E, a Stage failure ends the request before language generation.

After Stage, the router directs cleanup for a cancelled or definitively rejected request to the selected E. Lost or ambiguous handoffs expire in E's native request registry. Once P/PD claims a transfer, it owns the receive buffer and its completion fence.

Dynamo sends all media in a request to one encoder. Text requests bypass E. Chat accepts `image_url`, `video_url`, `audio_url` and inline `input_audio`. Native `/generate` accepts a URL string or flat list for each of `image_data`, `video_data` and `audio_data`. The original `input_ids` determine their order. SGLang advertises the staged media capabilities of the loaded model. Dynamo forwards the descriptor to P/PD and D with the original token IDs.

| Model family | Image | Video | Separate audio input |
|---|---|---|---|
| Qwen2-VL, Qwen2.5-VL, Qwen3-VL/MoE, Qwen3.5/MoE, Intern-S2 Preview | Yes | Yes | — |
| GLM-5-Next (GLM-5.3-Flash), GLM-4V/MoE, MiniMax-M3 | Yes | Yes | — |
| Qwen3-Omni-MoE | Yes | Yes | Yes |
| Muse Spark | Yes | Yes | With a tokenizer that defines the soft-audio delimiters |
| MiMo-V2 with the corresponding towers | Layout implemented | Clips without an audio track | Layout implemented |
| Dots3-Note | Layout implemented | Image items only | Layout implemented |
| Kimi-VL, Kimi K2.5, K3, K3.1, Inkling2 | Yes | Image items only | — |
| Qwen2-Audio, Qwen3-ASR | — | — | Yes |

The native SGLang EPD launch list at `9fc0e2a6` includes 18 registered visual architectures. The staged image path covers all 18 and adds GLM-4V and GLM-4V-MoE. Of these 20, 14 accept direct video input. The pinned revision lists Qwen2.5-Omni in its launch check but has no registered vision implementation for it. [Native EPD launch list](https://github.com/sgl-project/sglang/blob/9fc0e2a6/python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L366-L387) · [SGLang implementation](https://github.com/sgl-project/sglang/pull/51754)

The MiMo-V2.5 and dots3-note-prev checkpoints failed to start standalone E during verification. MiMo accessed the absent decoder during weight loading, and Dots initialized the decoder MoE.

Kimi K3/K3.1 process video as timestamped image chunks in unified serving. Dots3-Note interleaves frames, text and optional audio. Submit image items for Kimi and Dots3-Note vision. Dots3-Note also accepts separate audio items. Qwen-family handoffs use v1 for images and v2 for video. Other visual processors use v3, which carries the native model type, the embedding width and the row count of each item.

Chat also requires a compatible frontend tokenizer and chat template. For `moonshotai/Kimi-VL-A3B-Instruct`, the pinned Dynamo Rust tokenizer does not recognize the checkpoint's `kimi_vl` tiktoken configuration. Set `DYN_CHAT_PROCESSOR=sglang`, `DYN_ENDPOINT_TYPES=chat` and `DYN_TRUST_REMOTE_CODE=true` to use the checkpoint tokenizer with the SGLang chat processor in a frontend environment with SGLang installed. Pass `-- --trust-remote-code --skip-tokenizer-init` to the launcher. The Rust ingress then accepts the prepared token IDs, and the SGLang receiver still initializes the visual processor. Native `/generate` accepts pretokenized input when enabled with `DYN_SGLANG_ENABLE_GENERATE=1`.

Qwen3-ASR-0.6B ships a vocabulary and merges. For a Rust tokenizer that requires `tokenizer.json`, export the same checkpoint tokenizer with `AutoTokenizer.from_pretrained(model_dir).backend_tokenizer.save(f"{model_dir}/tokenizer.json")`.

For Chat with `Qwen/Qwen2-Audio-7B-Instruct`, place the official processor template beside `config.json` in the local model directory. The tokenizer's bundled template omits audio placeholders. Native `/generate` accepts token IDs produced with the processor template.

```python
from pathlib import Path
from transformers import AutoProcessor

model_dir = Path("/path/to/Qwen2-Audio-7B-Instruct")
template = AutoProcessor.from_pretrained(model_dir).chat_template
(model_dir / "chat_template.jinja").write_text(template, encoding="utf-8")
```

GPU validation covered the following checkpoints.

| Checkpoint | Verified path |
|---|---|
| Qwen2-Audio-7B-Instruct | Single-clip and ordered two-clip questions on Chat and `/generate` passed in E/PD over NIXL and split E/P/D over ZMQ (16/16 requests). Reversing the WAVs reversed the answer. Chat used the official processor template. |
| Qwen3-ASR-0.6B | Dynamo transcribed two distinct WAVs on Chat and `/generate` with split E/P/D over NIXL and E/PD over ZMQ (8/8 requests). |
| Qwen3-Omni-30B-A3B-Instruct | The full thinker answered all eight targeted image/audio ordering questions on Chat and `/generate` in E/PD over each of NIXL and ZMQ. With a generic prompt, each transport scored 2/8. |
| GLM-4.6V-Flash | E/PD with TP1 NIXL and TP4 ZMQ passed single-image, ordered two-image, video and mixed requests on Chat and `/generate`. Split E/P/D with TP4, ZMQ E→P and NIXL P→D passed mixed requests on both APIs. |
| Kimi-VL-A3B-Instruct | TP1 E/PD with NIXL passed single-image and ordered two-image requests on both APIs using the tokenizer settings above. |
| GLM-5.3-Flash | With the full FP8 checkpoint, E/PD and E/P/D over NIXL each matched native SGLang on 16/16 Chat and `/generate` answers. All three deployments scored 4/8 with the pure-blue clip and 8/8 with the brighter clip. |
| Qwen2.5-VL-3B-Instruct | An earlier PR revision passed 33 Chat and `/generate` requests across E/PD and E/P/D using both NIXL and ZMQ. |

For requests with audio, E and the language worker must advertise the same audio model type. The v4 descriptor records native feature lengths, embedding rows, widths and identities. P/PD binds the rows to their placeholders, and split D reconstructs positions using the same metadata. Vision-only requests retain their v1–v3 formats.

The launcher defaults to NIXL/UCX. P or PD registers host memory for the encoder to write into. Install NIXL with its UCX plugin on both ends, or select ZMQ with `--encoder-transfer-backend zmq_to_scheduler`. In E/P/D, D receives the layout and video timing to reconstruct token positions. The P→D KV transfer backend is configured separately. See the [E/PD and E/P/D launcher](sglang/launch/encoded.sh).

EncoderRouter selects replicas round-robin within a compatible worker set. Multiple eligible encoder sets make the target ambiguous.

For encoders that advertise their media policy, the E sidecar fetches remote media under Dynamo's DNS and redirect checks and the native encoder's allowed domains, size limit, and timeout. The complete Stage request, including base64 expansion, must fit the 64 MiB gRPC limit. When E omits the media policy, its native process fetches the source URLs.

Deploy a native encoder that reports its 16-byte instance ID in both GetEncoderInfo and Stage. A replacement encoder registers a new ID. The sidecar rejects a receipt from an unexpected instance and attempts to release it on the producing endpoint.

The E sidecar gives each native Stage RPC a 180-second deadline after fetching the media. Set `DYN_SGLANG_ENCODER_STAGE_TIMEOUT_SECS` on the encoder sidecar to change that deadline.

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
