<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust EPP Development

This directory contains the native Rust Envoy `ext_proc` Endpoint Picker Plugin (EPP) for Gateway
API Inference Extension (GAIE). It builds a single Rust binary, `dynamo-ext-proc`, and does not use
the Go EPP or CGO bridge.

Use this file for developer build, test, and image workflows. User-facing GAIE setup belongs in the
published Kubernetes Gateway API documentation.

## Prerequisites

- Rust and Cargo for host-native builds and tests.
- Docker with BuildKit/buildx for image builds.
- `kind` only when using `make image-kind`.

The Makefile runs Cargo commands from the repository root so the crate participates in the full
Dynamo workspace.

## Host-Native Build

From this directory:

```bash
cd deploy/inference-gateway/ext-proc
```

Build the release binary:

```bash
make build
```

Build a debug binary:

```bash
make build-debug
```

The release binary is written to:

```text
target/release/dynamo-ext-proc
```

## Development Checks

Run the Rust checks from this directory:

```bash
make fmt
make check
make clippy
make test
```

These targets map to Cargo commands for the `dynamo-ext-proc` package.

## Image Builds

Build and load a local image:

```bash
make image-load
```

Build and push an image:

```bash
export DOCKER_SERVER=ghcr.io/nvidia/dynamo
export IMAGE_TAG=ghcr.io/nvidia/dynamo/dynamo-rust-epp:<tag>
make image-push
```

Build, load, and import into a kind cluster:

```bash
export KIND_CLUSTER=kind
make image-kind
```

Build and push a multi-architecture image:

```bash
export DOCKER_SERVER=ghcr.io/nvidia/dynamo
export IMAGE_TAG=ghcr.io/nvidia/dynamo/dynamo-rust-epp:<tag>
make image-multiarch-push
```

Useful image variables:

| Variable | Default | Purpose |
|---|---|---|
| `DOCKER_SERVER` | `dynamo` | Registry or registry namespace used to form `IMAGE_REPO`. |
| `IMAGE_TAG` | `$(DOCKER_SERVER)/dynamo-rust-epp:$(git describe ...)` | Full image reference to build. |
| `DYNAMO_DIR` | Repository root auto-detected from this directory | Named Docker build context for the Dynamo workspace. |
| `PLATFORMS` | Host architecture | Platform for local image builds. |
| `MULTIARCH_PLATFORMS` | `linux/amd64,linux/arm64` | Platforms for multi-architecture builds. |
| `DOCKER_PROXY` | unset | Optional image prefix or mirror for base images. |
| `EXTRA_BUILD_ARGS` | unset | Extra arguments passed to `docker buildx build`. |
| `KIND_CLUSTER` | `kind` | kind cluster name for `make image-kind`. |

Run `make info` to print the resolved build values.

Common image targets:

| Target | Purpose |
|---|---|
| `make image-build` | Build the image with the default buildx builder. |
| `make image-load` | Build the image and load it into the local Docker daemon. |
| `make image-push` | Build the image and push it to `IMAGE_TAG`. |
| `make image-kind` | Build, load, and import the image into `KIND_CLUSTER`. |
| `make image-multiarch-push` | Build and push a multi-architecture image. |
| `make image-local-build` | Build with a temporary local buildx builder. |
| `make image-local-load` | Build with a temporary local buildx builder and load locally. |
| `make image-local-push` | Build with a temporary local buildx builder and push. |

## Runtime Notes for Developers

The Rust EPP serves Envoy `ext_proc` gRPC on port `9002` and plaintext gRPC health on port `9003`.
It serves TLS on the `ext_proc` port by default. Set `DYN_SECURE_SERVING=false` only for local
debugging with a plaintext h2c gateway.

The common local environment variables are:

| Variable | Default | Purpose |
|---|---|---|
| `DYN_NAMESPACE_PREFIX` | unset | Preferred Dynamo discovery namespace prefix. |
| `DYN_NAMESPACE` | unset | Exact Dynamo discovery namespace fallback. If unset, the binary uses `vllm-agg`. |
| `DYN_COMPONENT_NAME` | `backend` | Dynamo component that exposes the `generate` endpoint. |
| `DYN_ENFORCE_DISAGG` | `false` | Deprecated and ignored. Registered worker types determine routing topology and readiness. |
| `DYN_KUBE_DISCOVERY_MODE` | `pod` | Kubernetes discovery identity mode. The Rust EPP currently rejects `container`. |
| `DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD` | unset | Load-shedding: KV cache block utilization threshold (`0.0`–`1.0`). A worker is overloaded when `active_decode_blocks / kv_total_blocks` exceeds this. |
| `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD` | unset | Load-shedding: absolute prefill token count threshold. A worker is overloaded when `active_prefill_tokens` exceeds this. |
| `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC` | unset | Load-shedding: prefill token threshold as a fraction of `max_num_batched_tokens`. |
| `DYN_SHED_RETRY_AFTER_SECS` | unset | Load-shedding: optional `Retry-After` header value (seconds) sent on a 429 shed response. |
| `RUST_LOG` | `info` | Tracing log filter. |

## Load Shedding (Request Rejection)

The Rust EPP can proactively reject requests when every eligible worker is
overloaded, instead of routing them and wasting prefill compute on requests that
would time out. This is the gateway-path equivalent of the Dynamo frontend's
HTTP 529 admission control, and it **reuses the exact same detection logic** —
the `KvWorkerMonitor` from `lib/llm` (`dynamo-llm`).

### How it works

1. On startup the EPP constructs a `KvWorkerMonitor` bound to the decode
   `KvRouter`'s `Client`, and (in disaggregated serving) hands it to the
   `PrefillRouter` so the prefill `Client` is attached on activation.
2. The monitor subscribes to the namespace `kv_metrics` event stream, tracks
   per-worker load (`active_decode_blocks`, `kv_total_blocks`,
   `active_prefill_tokens`), and publishes the overloaded worker set to the
   router `Client`s. The KV/prefill schedulers already exclude overloaded workers
   and cannot route to them.
3. On each request, before tokenizing or routing, the EPP checks whether any
   eligible worker is free. If all discovered workers are overloaded, it returns
   an explicit **HTTP 429** (with an optional `Retry-After` header) via an Envoy
   `ext_proc` immediate response — so the gateway rejects the request without
   forwarding it to any worker, and the client / failover can back off and retry.

Because the monitor lives in-process, the EPP reads Dynamo's own KV-router load
signals directly (no metrics scraping and no CGO bridge).

### Enabling it

Load shedding is **opt-in and off by default**. If none of the
`DYN_ACTIVE_*_THRESHOLD` variables are set, the monitor reports "not configured"
and the EPP never sheds — behavior is identical to before. Set one or more
thresholds (see the table above) to enable it, for example:

```bash
export DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD=0.85
export DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD=10000
export DYN_SHED_RETRY_AFTER_SECS=1
```

### Reference

The threshold semantics, tuning guidance, dual-threshold (decode vs prefill)
model, and Prometheus metrics are documented for the frontend and apply
identically here, since the same `KvWorkerMonitor` computes overload:
[Request Rejection (frontend)](../../../docs/fault-tolerance/request-rejection.md).

The main differences on the EPP path are:

- Configuration is via the environment variables above rather than frontend CLI
  flags or the `/busy_threshold` HTTP endpoint.
- Rejections surface as **HTTP 429** (with `Retry-After`) through the gateway,
  rather than the frontend's HTTP 529.

## Cleaning

Clean the Rust package build artifacts:

```bash
make clean
```
