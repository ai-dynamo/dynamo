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
- Capability-gated RL pause/resume, sleep/wake, weight-transfer, and weight-version controls through native gRPC
- Image URL and data-URI inputs, including media UUIDs

The protocol does not support LoRA, encode workers, beam search, `n > 1`,
preprocessed multimodal features, audio/video media, or Dynamo tool-call and
reasoning parsers. Parser defaults returned by Control are intentionally not
advertised to the Dynamo frontend because the current inference protocol does
not preserve all parser-related request semantics.

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

Use `VLLM_GRPC_ENDPOINT` instead of `--vllm-endpoint` when the endpoint is
provided through the environment.

### RL workflows

Start vLLM with the capabilities required by the workflow, then opt the sidecar into RL discovery:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --grpc-port 50051 \
  --enable-sleep-mode \
  --weight-transfer-config '{"backend":"nccl"}'

DYN_SYSTEM_PORT=8081 dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051 \
  --enable-rl
```

`--enable-rl` (or `DYN_ENABLE_RL=true`) requires the Dynamo system server (`DYN_SYSTEM_PORT=0` or a positive port) and registers `dyn://<namespace>.<component>.rl`, which lets the Dynamo frontend discover this worker and its `/engine/control/*` and `/engine/update/*` routes through `/v1/rl/workers`. The sidecar advertises pause/resume, sleep-status, and weight-version controls when the vLLM server reports the RL gRPC API; mutating sleep/wake routes require `--enable-sleep-mode`, weight-transfer routes require `--weight-transfer-config`, and draft updates require speculative decoding support.

The update request bodies match vLLM's RL HTTP schemas: `init_weight_transfer_engine` requires `{"init_info": {...}}`, `update_weights` requires `{"update_info": {...}}`, `finish_weight_update` accepts `{"weight_version": "..."}`, and `update_weight_version` requires `{"new_version": "..."}`. Weight tensors remain on the configured NCCL, IPC, or sparse-NCCL transport; only backend metadata crosses gRPC.

The RL endpoint and engine routes are unauthenticated administrative surfaces that can pause serving, release GPU memory, and replace model weights. Enable them only on trusted request and system networks.

The sidecar discovers `model_id`, the served name, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through `vllm.Control`. `model_id` must be readable locally or fetchable by Dynamo for tokenization and chat templates. Parser defaults are not advertised because the current inference protocol cannot preserve all parser-related request semantics.

The sidecar currently supports one vLLM frontend hosting the complete data-parallel group starting at rank 0. Control reports the global size; Dynamo forwards the selected rank as `x-data-parallel-rank` gRPC metadata on each generation request. Partial and hybrid rank ownership are unsupported because the protocol does not report the locally hosted rank count, and a nonzero starting rank is rejected. When KV routing is enabled, Control must return one unique ZMQ event source for every rank in the group.

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

The mocker does not advertise RL capabilities; use a compatible vLLM server for RL route testing.

See [`../../mocker/servers/vllm/README.md`](../../mocker/servers/vllm/README.md)
for aggregated and prefill/decode examples, supported Mocker configuration,
and fidelity limits.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` runs an aggregated deployment (a frontend plus one worker pod
that colocates the sidecar with a vLLM engine). `deploy/disagg.yaml` runs
disaggregated prefill/decode with NIXL KV transfer.

There is no published sidecar image yet, so build and push the unified image
from `lib/sidecar/Dockerfile`. It contains the vLLM, SGLang, and TensorRT-LLM
sidecar executables; the deployment selects `dynamo-vllm-sidecar`.

The sidecar waits for both the Control and Inference services through the standard gRPC health API before registering the worker. The deployment manifests retain lightweight socket probes for container lifecycle monitoring. The engine image must include a `vllm-rs` build compatible with the vendored protocol.

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
  -f lib/sidecar/Dockerfile \
  -t <your-registry>/dynamo-sidecar:1.3.0 --push .
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

There is no published unified sidecar image yet. The quick start above builds
`dynamo-sidecar` from `lib/sidecar/Dockerfile`. The same image contains the
vLLM, SGLang, and TensorRT-LLM executables; each deployment selects the matching
executable with its container command. Official packaging is deferred to a
follow-up change.
