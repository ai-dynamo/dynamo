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
standalone Rust executable composed with `dynamo_backend_common::run`:
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

The integration does **not** support multimodal input, LoRA, KV-aware routing,
encode workers, beam search, or `n > 1`.

> [!NOTE]
> The sidecar requires a server implementing the OpenEngine `Control` service.
> `GetModelInfo` supplies the registered context length (and the default
> `max_tokens` for requests that omit one), so the sidecar refuses to start if
> `Control` is missing rather than serving with an unknown context window.
> `Control.Abort` cancels an in-flight request; closing the `Generate` stream
> also aborts it, so cancellation is covered either way.
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

Start TensorRT-LLM with its OpenEngine gRPC server. This requires the OpenEngine
Python bindings and a TensorRT-LLM build with OpenEngine gRPC support:

```bash
python -m pip install --extra-index-url https://buf.build/gen/python \
  "tensorrt_llm[openengine]"

python -m tensorrt_llm.commands.serve <model> \
  --grpc --grpc-protocol openengine --host 0.0.0.0 --port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker:

```bash
dynamo-trtllm-sidecar \
  --trtllm-endpoint 127.0.0.1:50051 \
  --model-path <model>
```

The context length is read from the server at startup (`Control.GetModelInfo`),
so there is no flag to keep in sync with the engine's `max_seq_len`.

Use `TRTLLM_GRPC_ENDPOINT` instead of `--trtllm-endpoint` when the endpoint is
provided through the environment.

## Disaggregation

Prefill and decode run as two workers, each with its own TensorRT-LLM engine and
its own sidecar, selected with `--disaggregation-mode`:

```bash
# Prefill (context) worker — registers under the `prefill` component.
dynamo-trtllm-sidecar --disaggregation-mode prefill \
  --trtllm-endpoint 127.0.0.1:50051 --model-path <model>

# Decode (generation) worker.
dynamo-trtllm-sidecar --disaggregation-mode decode \
  --trtllm-endpoint 127.0.0.1:50052 --model-path <model>
```

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

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` deploys a frontend and one worker pod. The worker runs the
sidecar next to a TensorRT-LLM engine and serves `Qwen/Qwen3-0.6B` on one GPU.

There is no published sidecar image yet (see [Packaging](#packaging)), so you
build and push your own.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node. The engine runs as a native
  sidecar (`initContainers` with `restartPolicy: Always`), which requires that
  version.
- `kubectl` set to that cluster, and a namespace to deploy into.
- A Hugging Face token for the model.
- A container registry you can push to and the cluster can pull from.
- A TensorRT-LLM engine image with OpenEngine gRPC support (serving
  `--grpc-protocol openengine`, implementing the `Control` service, and with the
  OpenEngine Python bindings installed for the health probes). No published
  release ships this yet, so `deploy/agg.yaml` leaves it as the placeholder
  `<trtllm-image-with-openengine>` for you to fill in.

### 1. Build and push the sidecar image

Build a multi-arch image so it runs on any node — `amd64` (x86) or `arm64`
(GB200/Grace):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/trtllm/Dockerfile \
  -t <your-registry>/dynamo-trtllm-sidecar:1.3.0 --push .
```

To build faster for one arch, pass just that platform (e.g. `linux/arm64` for
GB200/Grace).

### 2. Point the manifest at your image

In `deploy/agg.yaml`, set the `main` worker image to the one you just pushed.
If your registry is private, add `imagePullSecrets` to the worker pod spec.

### 3. Create the Hugging Face token secret

Read the token from an env var so it stays out of your shell history (or use
`--from-file` / an external secret manager):

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" -n <namespace>
```

### 4. Deploy

```bash
kubectl apply -f lib/sidecar/trtllm/deploy/agg.yaml -n <namespace>
```

Wait for the worker pod to reach `2/2 Running`:

```bash
kubectl get pods -n <namespace> -w
```

### 5. Send a request

Port-forward the frontend and call it:

```bash
kubectl port-forward -n <namespace> svc/trtllm-sidecar-agg-frontend 8000:8000 &

curl -s localhost:8000/v1/models | jq .

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' | jq .
```

`/v1/models` should list `Qwen/Qwen3-0.6B`, and the chat call returns a reply.

## Tuning

The engine streams tokens to the sidecar over gRPC. By default it sends one
message per token, and that per-token serialization is the sidecar's main
throughput cost versus an in-process backend. The `trtllm-engine-config`
ConfigMap in `deploy/agg.yaml` sets `stream_interval`, which emits one chunk per
`N` decode steps instead:

- Higher `N` → fewer, larger gRPC messages → higher throughput under load.
- Trade-off: the client receives tokens in bursts of `N`.

On a single GB200 (Qwen3-0.6B, 2000-in / 256-out) raising `stream_interval` from
1 to 5 roughly doubled output throughput at high concurrency (~6.3k → ~12k
tok/s) and even lowered TTFT. `5` keeps streaming smooth while capturing nearly
all the gain.

## Packaging

There is no published image yet. That is deferred to a follow-up change. Once
the sidecar crate is published, you just install it onto a minimal base image.
Until then, build and push your own as shown above.
