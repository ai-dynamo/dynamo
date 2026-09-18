<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM sidecar

> [!WARNING]
> **Experimental.** This sidecar and its deployment example are experimental and
> not yet packaged for distribution (see [Packaging](#packaging)). The manifest,
> flags, and behavior may change without notice.

`dynamo-trtllm-sidecar` connects a Dynamo worker to TensorRT-LLM's OpenEngine
(`openengine.v1`) gRPC server — the `Inference.Generate` streaming RPC. It is a
standalone Rust executable composed with `dynamo_backend_common::run`, and is
also compiled into `ai-dynamo-runtime` for the importable
`dynamo.trtllm.sidecar` launcher:
TensorRT-LLM runs as its own process while the sidecar owns Dynamo worker
registration, request conversion, transport, cancellation, and abort.

## Supported

- Aggregated generation
- Disaggregated (prefill/decode) serving — see [Disaggregation](#disaggregation)
- Token requests through Dynamo preprocessing
- Sampling, stop conditions, structured output (JSON schema / regex / grammar /
  structural tag), and logprobs
- Streaming delta tokens with a terminal usage/finish summary
- Cancellation via `Control.Abort` and by closing the gRPC stream

The integration does **not** support multimodal input, LoRA, encode workers,
beam search, or `n > 1`.

Data-parallel rank targeting is rejected: the server answers
`openengine-target-dp-rank` with `UNIMPLEMENTED`, so a request carrying a rank
hint is refused up front instead of failing in the engine.
`KvSessionRef.dp_rank` still carries a disaggregated session's KV affinity,
inside the request body.

> [!NOTE]
> `Control.GetModelInfo` supplies the registered context length (and the default
> `max_tokens` for requests that omit one) unless `--context-length` supplies it
> instead. `Control.Abort` cancels an in-flight request; closing the `Generate`
> stream also aborts it, so cancellation is covered either way.
>
> `Control`'s LoRA RPCs (`LoadLora`, `UnloadLora`, `ListLoras`) and KV-event
> RPCs (`GetKvEventSources`, `SubscribeKvEvents`) return `UNIMPLEMENTED`: the
> LLM API has no runtime adapter load/unload entry point, and KV events are
> published out of band. The sidecar uses neither.

## Protocol

The gRPC types are vendored, like the vLLM and SGLang sidecars': `proto/`
carries the `openengine.v1` contract from
[`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) `v0.1.0`,
compiled by `build.rs` with `tonic-build`. The pinned revision is the git commit
behind the Buf Schema Registry module commit (`768a93c7b44e`) TensorRT-LLM's
server is generated from, so the two sides agree; `proto/README.md` records the
commit and per-file SHA-256.

Building needs only `protoc`, which the workspace already requires for `lib/llm`
and the other sidecars — there is no registry to configure and no token to
obtain.

To bump the protocol, re-copy `proto/openengine/v1/*.proto` from a newer upstream
commit, update the revision, checksums, and `build.rs`'s `PROTOS` list together,
and re-run the tests.

## Run

Start TensorRT-LLM with its OpenEngine gRPC server. Published releases carry the
server from `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000`
onward; earlier tags have no OpenEngine servicer at all.

The bindings are a separate install even on those releases. The servicer imports
`openengine.v1` directly, so a stock image refuses to start the listener:

```text
Error: Failed to import OpenEngine support: No module named 'openengine'.
```

```bash
# Install the two packages directly rather than through the
# `tensorrt_llm[openengine]` extra: the extra makes pip re-resolve
# TensorRT-LLM's whole dependency closure, which fails on an image whose
# site-packages is read-only. Both are pinned to BSR module commit
# 768a93c7b44e, the revision `proto/` was generated from. The protobuf package
# is additionally pinned by gencode version -- a gencode newer than the image's
# protobuf runtime fails at import -- so raise it only with the image's
# protobuf.
python -m pip install --extra-index-url https://buf.build/gen/python \
  "openengine-openengine-grpc-python==1.78.1.1.20260730172104+768a93c7b44e" \
  "openengine-openengine-protocolbuffers-python==33.5.0.1.20260730172104+768a93c7b44e"

python -m tensorrt_llm.commands.serve <model> \
  --grpc --grpc-protocol openengine --host 127.0.0.1 --port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker:

```bash
dynamo-trtllm-sidecar \
  --grpc-endpoint 127.0.0.1:50051 \
  --model-path <model>
```

The context length comes from `--context-length` (or `TRTLLM_CONTEXT_LENGTH`)
when supplied, and from `Control.GetModelInfo` otherwise; a disagreement is
logged at WARN and the configured value wins. Supply it whenever the engine was
started without `--max_seq_len`, because TensorRT-LLM then leaves
`max_context_length` unset. With neither source the sidecar retries until
`--grpc-startup-deadline-secs` and then exits, rather than registering a worker
that would reject every request omitting `max_tokens`.

Use `DYN_SIDECAR_GRPC_ENDPOINT` instead of `--grpc-endpoint` when the endpoint is
provided through the environment.

## Disaggregation

Prefill and decode run as two workers, each with its own TensorRT-LLM engine and
its own sidecar, selected with `--disaggregation-mode`:

```bash
# Prefill (context) worker — registers under the `prefill` component.
dynamo-trtllm-sidecar --disaggregation-mode prefill \
  --grpc-endpoint 127.0.0.1:50051 --model-path <model>

# Decode (generation) worker.
dynamo-trtllm-sidecar --disaggregation-mode decode \
  --grpc-endpoint 127.0.0.1:50052 --model-path <model>
```

`launch/disagg.sh` brings up the whole topology on two GPUs — frontend, both
engines with a cache transceiver, and both sidecars:

```bash
./lib/sidecar/trtllm/launch/disagg.sh --model <model>
```

Run `--help` for the ports, GPU assignment, and transceiver backend it accepts.

Both engines must be started with a KV cache transceiver so they can move KV
cache between themselves (`cache_transceiver_config`); without it the engines
cannot complete the handoff. Use the default `NIXL` backend — it picks its own
underlying transport (UCX where there is no RDMA fabric) and is the path Dynamo
uses elsewhere for disaggregation.

OpenEngine has no request-type field, so the phase is carried on the wire like
this:

- The prefill worker sets `extra.request_type = "context_only"` and caps
  generation at one token. The server answers with a terminal `PrefillReady`
  event holding a `KvSessionRef`; there is no `finished` event for a context
  request.
- The sidecar encodes that `KvSessionRef` as the opaque JSON Dynamo carries in
  `PrefillResult.disaggregated_params`, and emits it on the prefill worker's
  terminal chunk. The prefill worker streams no tokens to the client.
- The decode worker decodes that JSON back into `kv.session` on its own
  `Generate` request, which the server maps to `generation_only`. It streams the
  full completion and reports the authoritative usage.

The handoff JSON mirrors `KvSessionRef` field-for-field (`session_id`,
`transfer_backend`, `endpoints`, `dp_rank`, `attributes`) and is never
interpreted between the two workers. See `src/disagg.rs`.

## Deploy on Kubernetes

`deploy/agg.yaml` runs a frontend and one worker pod serving `Qwen/Qwen3-0.6B`
on one GPU. `deploy/disagg.yaml` runs prefill and decode as separate worker
pods. Read the disaggregated manifest's header before applying it: it requests
`rdma/ib` on both engines, which you drop if your fabric does not expose it.

You need a cluster on **v1.29+** (or v1.28 with the `SidecarContainers` gate)
with the Dynamo operator and a GPU node — the engine runs as a native sidecar —
plus a namespace, a registry, and two images:

- **The sidecar image.** Not published yet (see [Packaging](#packaging)), so
  build it from `lib/sidecar/Dockerfile`; it carries all three engine-specific
  executables. Multi-arch if your nodes are mixed:
  `docker buildx build --platform linux/amd64,linux/arm64 -f lib/sidecar/Dockerfile -t <your-registry>/dynamo-sidecar:1.3.0 --push .`
- **The engine image.** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000`
  or newer, with the pinned bindings from [Run](#run) layered on: the release
  ships the servicer but not the `openengine.v1` package that both it and the
  manifests' health probes import.

Set both images in the manifest, add `imagePullSecrets` for a private registry,
then:

```bash
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="$HF_TOKEN" -n <namespace>
kubectl apply -f lib/sidecar/trtllm/deploy/agg.yaml -n <namespace>
kubectl get pods -n <namespace> -w    # every worker pod reaches 2/2
```

Then port-forward `svc/trtllm-sidecar-agg-frontend` (or
`svc/trtllm-sidecar-disagg-frontend`) on 8000 and call `/v1/models` and
`/v1/chat/completions`.

## Tuning

Per-token gRPC serialization is the sidecar's main throughput cost versus an
in-process backend. `stream_interval: N` emits one chunk per `N` decode steps
instead of one per token, trading burstier delivery for throughput. On a single
GB200 (Qwen3-0.6B, 2000-in / 256-out) raising it from 1 to 5 roughly doubled
output throughput at high concurrency (~6.3k → ~12k tok/s) and lowered TTFT.
The manifests set `5`; `launch/disagg.sh` leaves it at the engine default.

## Packaging

No published sidecar image yet. Build and push it yourself — see
[Build the image](../README.md#build-the-image). One minimal CPU-only image
carries the vLLM, SGLang, and TensorRT-LLM executables.
