<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM sidecar

> [!WARNING]
> **Experimental.** This sidecar and its deployment examples are experimental
> and not yet packaged for distribution. The manifests, flags, and behavior may
> change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC services:

- `vllm.Inference` for generation
- `vllm.Control` for model and server discovery
- Standard gRPC health for startup readiness

It is a standalone Rust executable.

## Supported

- Aggregated generation
- NIXL prefill/decode generation
- Token and text requests through Dynamo preprocessing
- Sampling, stop conditions, structured output, logprobs, cache options, and priority
- Opaque `kv_transfer_params` handoff
- Data-parallel rank routing and KV-event source discovery
- Opt-in image URL and data-URI requests; exact KV routing is qualified for
  Qwen2-VL, Qwen2.5-VL, and Qwen3-VL

The released vLLM v0.27.1 protocol does not support multimodal requests. The
image profile requires a vLLM build from revision
`3d1f5cee1552b8208f3009c75f8bc856f27e0eff` and the explicit sidecar
`--enable-multimodal` flag. Without that opt-in, the sidecar rejects media before
submitting an RPC so a v0.27.1 server cannot silently discard the newer media
fields. The image profile accepts `http://`, `https://`, and `data:` sources; it
does not accept decoded/RDMA media, audio, video, or processor overrides.

LoRA, encode workers, beam search, `n > 1`, and Dynamo tool-call and reasoning
parsers remain unsupported. Parser defaults returned by Control are
intentionally not advertised to the Dynamo frontend because the current
inference protocol does not preserve all parser-related request semantics.

## Run

Start vLLM with its gRPC listener:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --host 127.0.0.1 --grpc-port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker explicitly:

```bash
dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051
```

For the compatible multimodal vLLM revision, opt in explicitly:

```bash
dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051 \
  --enable-multimodal
```

Use `VLLM_GRPC_ENDPOINT` instead of `--vllm-endpoint` when the endpoint is
provided through the environment. `DYN_ENABLE_MULTIMODAL=true` is equivalent to
the multimodal flag.

The sidecar discovers `model_id`, the served name, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through `vllm.Control`. `model_id` must be readable locally or fetchable by Dynamo for tokenization and chat templates.

The sidecar currently supports one vLLM frontend hosting the complete data-parallel group starting at rank 0. Control reports the global size; Dynamo forwards the selected rank as `x-data-parallel-rank` gRPC metadata on each generation request. Partial and hybrid rank ownership are unsupported because the protocol does not report the locally hosted rank count, and a nonzero starting rank is rejected. When KV routing is enabled, Control must return one unique ZMQ event source for every rank in the group.

For an opted-in multimodal model, the sidecar resolves the model's chat image
placeholder token from its local or Hugging Face configuration and attaches it
to every ZMQ source. This lets backend events use the same canonical image hash
as frontend routing for the qualified Qwen families. Other multimodal families,
or a configuration fetch/token-resolution failure, keep inference available but
omit exact multimodal source metadata and receive only ordinary text-prefix KV
routing. The current Control API reports the model ID but not its Hugging Face
revision, so a vLLM launch using `--revision` should expose the same local model
directory as its model ID and mount that path into the sidecar. Otherwise the
sidecar can only fetch the repository's default revision and may disable or
misconfigure exact routing.

The forwarded 64-hex vLLM media identifier contains Dynamo's 64-bit routing hash
and becomes part of vLLM's encoder/prefix-cache identity. Treat passed-through
HTTP URLs as immutable: changing the bytes behind one URL can reuse stale media
cache state. User-supplied media UUIDs retain their vLLM semantics and disable
exact Dynamo multimodal credit.

Aggregated serving is the default. Set the existing `--disaggregation-mode` to `prefill` or `decode` only for non-aggregated deployments; the current Control API does not report engine role.

The sidecar opens eight gRPC connections by default. This avoided
connection-level throttling in high-concurrency sidecar tests. Override the
pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry
interval, and a five-minute deadline for establishing the full connection
pool. Override them with `--grpc-connect-attempt-timeout-secs`,
`--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the
corresponding `DYN_SIDECAR_GRPC_*` environment variables.

Each request owns its response stream but borrows a channel from the shared pool. Aggregate and prefill cancellation drops only that request's stream. Decode cancellation first submits the decode request and retains its stream until the first output token or a response containing `finish_info`, so a NIXL receiver can complete and release the transferred KV; it then drops the stream. If the stream ends early, returns a gRPC error, or produces an invalid response after cancellation, the sidecar logs the failure and reports the request as cancelled. vLLM automatically aborts the corresponding engine request while the pooled HTTP/2 connection remains available to other requests. The sidecar does not call the Control `Abort` RPC.

## Test without vLLM or a GPU

Use the CPU-only `dynamo-vllm-mocker-server` to exercise the same Inference, Control, and health contracts:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --vllm-endpoint 127.0.0.1:50051
```

See [`../../mocker/servers/vllm/README.md`](../../mocker/servers/vllm/README.md)
for aggregated and prefill/decode examples, supported Mocker configuration,
and fidelity limits.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` runs an aggregated deployment (a frontend plus one worker pod
that colocates the sidecar with a vLLM engine). `deploy/disagg.yaml` runs
disaggregated prefill/decode with NIXL KV transfer.

There is no published vLLM sidecar image yet, so you build and push your own from
`Dockerfile` — the same pattern as the TensorRT-LLM and SGLang sidecars.

The sidecar waits for both the Control and Inference services through the
standard gRPC health API before registering the worker. The deployment
manifests retain lightweight socket probes for container lifecycle monitoring.
The examples pair the exact vLLM v0.27.1 engine image with its released text-only
profile. To exercise images, replace the engine image with a build of revision
`3d1f5cee1552b8208f3009c75f8bc856f27e0eff` and add
`--enable-multimodal` to the sidecar arguments.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node (multiple GPUs plus an RDMA fabric
  for `disagg.yaml`). The engine runs as a native sidecar (`initContainers` with
  `restartPolicy: Always`), which requires that version.
- `kubectl` set to that cluster, and a namespace to deploy into.
- A Hugging Face token for the model.
- A container registry you can push to and the cluster can pull from.

### 1. Build and push the sidecar image

Build a multi-arch image so it runs on any node — `amd64` (x86) or `arm64`
(GB200/Grace):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/vllm/Dockerfile \
  -t <your-registry>/dynamo-vllm-sidecar:1.3.0 --push .
```

### 2. Point the manifest at your image

In `deploy/agg.yaml` (and `deploy/disagg.yaml`), set the `main` worker image to
the one you pushed. Add `imagePullSecrets` if your registry is private.

### 3. Create the Hugging Face token secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" -n <namespace>
```

### 4. Deploy

```bash
kubectl apply -f lib/sidecar/vllm/deploy/agg.yaml -n <namespace>
```

Wait for the worker pod to reach `2/2 Running`:

```bash
kubectl get pods -n <namespace> -w
```

### 5. Send a request

```bash
kubectl port-forward -n <namespace> svc/vllm-sidecar-agg-frontend 8000:8000 &

curl -s localhost:8000/v1/models | jq .

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' | jq .
```

### Disaggregated

`deploy/disagg.yaml` runs prefill and decode as separate worker pods with NIXL
KV transfer. It needs multiple GPUs and an RDMA fabric, and both worker pods
must reach `2/2 Running`. Apply it the same way and call the frontend as above.

## Packaging

There is no published image yet; the quick start above builds one from
`Dockerfile`. Official packaging is deferred to a follow-up change.
