<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo EPP Development

This directory contains the native Rust Envoy `ext_proc` Endpoint Picker Plugin (EPP) for Gateway
API Inference Extension (GAIE). It builds a single Rust binary, `dynamo-ext-proc`, packages it as
`/epp` in the container image, and implements the upstream
[Lightweight Endpoint Picker (LW-EPP)](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/lwepp/README.md)
interface with the full Dynamo KV-aware router.

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
export IMAGE_TAG=ghcr.io/nvidia/dynamo/dynamo-epp:<tag>
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
export IMAGE_TAG=ghcr.io/nvidia/dynamo/dynamo-epp:<tag>
make image-multiarch-push
```

Useful image variables:

| Variable | Default | Purpose |
|---|---|---|
| `DOCKER_SERVER` | `dynamo` | Registry or registry namespace used to form `IMAGE_REPO`. |
| `IMAGE_TAG` | `$(DOCKER_SERVER)/dynamo-epp:$(git describe ...)` | Full image reference to build. |
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

The EPP serves Envoy `ext_proc` gRPC on port `9002` and plaintext gRPC health on port `9003`.
It serves TLS on the `ext_proc` port by default. Set `DYN_SECURE_SERVING=false` only for local
debugging with a plaintext h2c gateway.

The common local environment variables are:

| Variable | Default | Purpose |
|---|---|---|
| `DYN_NAMESPACE_PREFIX` | unset | Preferred Dynamo discovery namespace prefix. |
| `DYN_NAMESPACE` | unset | Exact Dynamo discovery namespace fallback. If unset, the binary uses `vllm-agg`. |
| `DYN_COMPONENT_NAME` | `backend` | Dynamo component that exposes the `generate` endpoint. |
| `DYN_ENFORCE_DISAGG` | `false` | Deprecated and ignored. Registered worker types determine routing topology and readiness. |
| `DYN_KUBE_DISCOVERY_MODE` | `pod` | Kubernetes discovery identity mode. The EPP currently rejects `container`. |
| `RUST_LOG` | `info` | Tracing log filter. |

## Cleaning

Clean the Rust package build artifacts:

```bash
make clean
```
