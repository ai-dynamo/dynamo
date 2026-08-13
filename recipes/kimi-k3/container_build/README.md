<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo bf7542 + vLLM 65b7662 ARM64 image

This directory contains the complete recipe used to build Dynamo commit
`bf7542fd26613495cc2a59ded28848861e1fee3c` on the ARM64 vLLM nightly at commit
`65b7662d3fcb773afaf751ab29ac6960a0cf011d`.

The resulting image contains Dynamo `1.4.0` and vLLM `0.26.1rc1.dev602+g65b7662d3`.

## Prerequisites

- A native `aarch64` Linux build host.
- Docker with Buildx and enough local disk for a large CUDA image build.
- Git and Python 3.
- Network access to GitHub, Docker Hub, NGC, Quay, PyPI, crates.io, and the
  public dependency hosts used by Dynamo's generated Dockerfile.
- Credentials for the selected output registry. Authenticate with
  `docker login`; do not put registry keys in this directory.
- NVIDIA Container Toolkit and a visible GPU when the default GPU validation
  is enabled.

The build performs a fresh source checkout, creates a new isolated BuildKit
builder, and passes `--no-cache --pull`. It does not import a prior build cache.

## Build and push

Run the recipe from this directory:

```bash
chmod +x build.sh validate-image.sh
IMAGE=registry.example.com/project/dynamo-vllm:arm64 \
  ./build.sh
```

`IMAGE` is required and has no default. It must be a writable tag, not a
digest. The script pushes the image, resolves the registry manifest digest,
writes the immutable reference to `out/image-pin.txt`, and validates that
digest on the local GPU.

For a build host without a GPU, defer validation:

```bash
IMAGE=<registry>/<repository>:<tag> VALIDATE_GPU=0 ./build.sh
./validate-image.sh "$(cat out/image-pin.txt)"
```

The validation script sends its Python program over standard input. This
avoids bind-mount failures on compute nodes backed by shared filesystems.

## Pinned inputs

All source and image pins are in `versions.env`. The important inputs are:

| Input | Pin |
| --- | --- |
| Dynamo | `bf7542fd26613495cc2a59ded28848861e1fee3c` |
| vLLM ARM64 manifest | `sha256:3ae6337cbc8423ce6af3286a38b759df8c218bfdb29e1d0353cabc273a22fb0b` |
| CUDA build base | `sha256:399b4d7b6401b02ff1de7216e2c7f3d56448728a0b6097e912c4f864b12091ec` |
| ARM64 manylinux builder | `sha256:b9dd5b2d6885fae144119ac934978003bcc413087ea08f602a960257205ec246` |

The script renders Dynamo's official vLLM CUDA 13.0 runtime Dockerfile from
the pinned Dynamo source. `prepare_dockerfile.py` then adds only:

- a build-time import and version check;
- an ARM-compatible check for `vllm._C_stable_libtorch`;
- labels recording the Dynamo and vLLM source pins.

The generated Dockerfile must match the SHA-256 recorded in `versions.env` or
the build stops before Docker runs.

## Validation

The build-time check verifies the packaged Python modules without loading the
CUDA extension on a GPU-less BuildKit worker. Post-push validation pulls the
immutable registry digest and verifies:

- the host and image are ARM64;
- CUDA is available;
- the exact Dynamo and vLLM versions are installed;
- `vllm._C_stable_libtorch` loads;
- Dynamo core, frontend, and vLLM modules import;
- a PyTorch matrix multiplication launches and synchronizes on the GPU.

Do not replace the stable-libtorch import with `vllm._C`. This vLLM ARM64
nightly does not ship an extension under that name.

## Outputs

The ignored `out/` directory contains:

- `build.log`: complete BuildKit output;
- `build-metadata.json`: BuildKit inputs and resulting descriptor;
- `image-inspect.txt`: registry inspection;
- `image-pin.txt`: immutable `image@sha256` reference;
- `manifest.json`: raw OCI manifest;
- `gpu-validation.log`: native GPU validation output, when enabled.

The known reference build produced manifest
`sha256:ad7c8b13f1711086ba10daf048d254309296617e4aa04190bb1a78c14a4aa78a`.
Registry manifests can differ when upstream package repositories or mutable
build tooling change, even though the principal source and container inputs
are pinned. Treat `out/image-pin.txt` as the authoritative output of each new
build.
