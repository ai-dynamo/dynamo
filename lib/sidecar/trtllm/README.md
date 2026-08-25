<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM sidecar

> [!WARNING]
> **Experimental.** This sidecar and its deployment example are experimental and
> not yet packaged for distribution (see [Packaging](#packaging)). The manifest,
> flags, and behavior may change without notice.

`dynamo-trtllm-sidecar` connects a Dynamo worker to TensorRT-LLM's native
`trtllm.TrtllmService` gRPC `Generate` service. It is a standalone Rust
executable composed with `dynamo_backend_common::run`: TensorRT-LLM runs as its
own process while the sidecar owns Dynamo worker registration, request
conversion, transport, cancellation, and abort.

## Supported

- Aggregated generation
- Token requests through Dynamo preprocessing
- Sampling, stop conditions, structured output (JSON schema / regex / grammar /
  structural tag), and logprobs
- Streaming delta tokens with a terminal usage/finish summary
- `Abort` on cancellation

The initial protocol does **not** support disaggregated (prefill/decode)
serving, multimodal input, LoRA, KV-aware routing, encode workers, beam search,
or `n > 1`. Disaggregation is excluded because the `Generate` response contract
carries no context-phase handoff.

## Run

Start TensorRT-LLM with its native gRPC listener (TRT-LLM `1.3.0rc21`; rc22 has a broken `--grpc`):

```bash
python -m tensorrt_llm.commands.serve <model> --grpc --host 0.0.0.0 --port 50051
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

Use `TRTLLM_GRPC_ENDPOINT` instead of `--trtllm-endpoint` when the endpoint is
provided through the environment.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` deploys a frontend and one worker pod. The worker runs the
sidecar next to a TensorRT-LLM engine and serves `Qwen/Qwen3-0.6B` on one GPU.

There is no published sidecar image yet (see [Packaging](#packaging)), so build
and push the unified image from `lib/sidecar/Dockerfile`. It contains all three
engine-specific sidecar executables; this deployment selects
`dynamo-trtllm-sidecar`.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node. The engine runs as a native
  sidecar (`initContainers` with `restartPolicy: Always`), which requires that
  version.
- `kubectl` set to that cluster, and a namespace to deploy into.
- A Hugging Face token for the model.
- A container registry you can push to and the cluster can pull from.

### 1. Build and push the sidecar image

Follow [Build the image](../README.md#build-the-image) and push the result to
a registry your cluster can pull from. This deployment selects
`dynamo-trtllm-sidecar`.

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

There is no published unified sidecar image yet. The image contains the vLLM,
SGLang, and TensorRT-LLM executables and uses a minimal CPU-only base. Until
official packaging is available, build and push `dynamo-sidecar` as described in
[Build the image](../README.md#build-the-image).
