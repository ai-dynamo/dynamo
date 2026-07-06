<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo EPP Development

This directory contains the standard Dynamo Endpoint Picker Plugin (EPP) for Gateway API Inference
Extension (GAIE). The implementation builds a Rust Dynamo routing library and links it into the Go
EPP binary through CGO.

Use this file for developer build, test, and image workflows. User-facing GAIE setup belongs in the
published Kubernetes Gateway API documentation.

## Prerequisites

- Docker with BuildKit/buildx for image builds.
- Go and a C toolchain for host-native Go builds.
- Rust and Cargo for host-native Dynamo library builds.
- `kind` only when using `make image-kind` or `make all-kind`.

Image builds are self-contained and do not require a host-built Dynamo library. Host-native binary
builds do require the local Dynamo static library artifacts.

## Image Builds

From this directory:

```bash
cd deploy/inference-gateway/epp
```

Build and load a local image:

```bash
make image-load
```

Build with a temporary local buildx builder and load the image:

```bash
make all
```

Build and push an image:

```bash
export DOCKER_SERVER=ghcr.io/nvidia/dynamo
export IMAGE_TAG=ghcr.io/nvidia/dynamo/dynamo-epp:<tag>
make image-push
```

Build and push through the aggregate target:

```bash
make all-push
```

Build, load, and import into a kind cluster:

```bash
export KIND_CLUSTER=kind
make image-kind
```

Build and import through the aggregate target:

```bash
make all-kind
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
| `make all` | Alias for `make image-local-load`. |
| `make all-push` | Alias for `make image-push`. |
| `make all-kind` | Alias for `make image-kind`. |

## Host-Native Build

Host-native `go build` needs the Dynamo C API static library and header copied into this project.
Build those artifacts first:

```bash
make dynamo-lib
```

Then build the EPP binary:

```bash
make build
```

The combined target builds the library and binary:

```bash
make build-with-lib
```

The binary is written to:

```text
deploy/inference-gateway/epp/bin/epp
```

`make dynamo-lib` builds `libdynamo_llm` from the repository root and copies:

- `libdynamo_llm_capi.a` to `pkg/plugins/dynamo_kv_scorer/lib/`
- `llm_engine.h` to `pkg/plugins/dynamo_kv_scorer/include/`

## Development Checks

Run the Go checks from this directory:

```bash
make fmt
make vet
make tidy
make test
```

`make test` runs with `CGO_ENABLED=1`, so it needs the same local C library artifacts as
host-native builds.

## Cleaning

Remove local Go build artifacts:

```bash
make clean
```

This removes `bin/` and runs `go clean`. It does not clean the repository-level Cargo target
directory used by `make dynamo-lib`.
