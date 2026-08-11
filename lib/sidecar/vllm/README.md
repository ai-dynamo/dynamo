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

The initial protocol does not support multimodal input, LoRA, KV-aware data
parallel routing, encode workers, beam search, or `n > 1`.

## Run

Start vLLM with its released gRPC listener:

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

The sidecar discovers `model_id`, the served name, context length, KV capacity, and scheduler limits through `vllm.Control`. `model_id` must be readable locally or fetchable by Dynamo for tokenization and chat templates. Parser defaults are not advertised because the current inference protocol cannot preserve all parser-related request semantics.

Data-parallel registration is omitted because Control reports global topology, not the rank range hosted by the connected frontend.

Aggregated serving is the default. Set the existing `--disaggregation-mode` to `prefill` or `decode` only for non-aggregated deployments; the current Control API does not report engine role.

The sidecar opens eight gRPC connections by default. This avoided
connection-level throttling in high-concurrency sidecar tests. Override the
pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry
interval, and a five-minute deadline for establishing the full connection
pool. Override them with `--grpc-connect-attempt-timeout-secs`,
`--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the
corresponding `DYN_SIDECAR_GRPC_*` environment variables.

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
