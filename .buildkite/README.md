<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Buildkite hosted remote-builder pilot

This pipeline tests whether Buildkite hosted agents can replace Dynamo's
self-managed BuildKit worker discovery, bootstrap, refresh, and fallback logic.
It is intentionally separate from the required GitHub Actions checks.

The first step fails unless the active Buildx driver is `remote`. Two identical
steps then render Dynamo's existing Dockerfile template and perform cache-only
CUDA builds. The comparison step reports the repeated-build cache speedup and,
when supplied, the speedup over a matching GitHub Actions build. No image is
published by default.

Build parameters are supplied as Buildkite build environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `IMAGE_FRAMEWORK` | `dynamo` | `dynamo`, `vllm`, `sglang`, or `trtllm` |
| `IMAGE_TARGET` | `base` | Target used when rendering the Dockerfile |
| `IMAGE_BUILD_TARGET` | empty | Optional Dockerfile stage passed to `buildx --target` |
| `IMAGE_PLATFORM` | `linux/amd64` | `linux/amd64`, `linux/arm64`, or both |
| `CUDA_VERSION` | `13.0` | CUDA version accepted by `container/render.py` |
| `NO_CACHE` | `false` | Disable BuildKit cache when `true` |
| `PUSH_IMAGE` | `false` | Push instead of using a cache-only output |
| `IMAGE_TAG` | generated pilot tag | Required registry tag for a publishing test |
| `RENDERER_IMAGE` | `python:3.12-slim` | Disposable environment for `container/render.py` |
| `GITHUB_BASELINE_SECONDS` | empty | Matching GitHub Actions image-build duration |
| `GITHUB_BASELINE_URL` | empty | URL for the matching GitHub Actions run |

Publishing is opt-in because registry authentication must use short-lived
Buildkite OIDC credentials. Do not add static registry credentials to this
pipeline or repository.

## Staging results

The `nvidia-oss-ci-staging` pilot on August 20, 2026 produced these results:

| Test | Result |
| --- | --- |
| [Hosted builder verification](https://buildkite.com/nvidia-oss-ci-staging/dynamo-hosted-remote-builder-poc/builds/2) | Active Buildx driver was `remote` |
| AMD64 `base` | 50 seconds first build, 6 seconds repeated build |
| [AMD64 `runtime`](https://buildkite.com/nvidia-oss-ci-staging/dynamo-hosted-remote-builder-poc/builds/4) | 588 seconds first build, 9 seconds repeated build |
| [Cross-build cache](https://buildkite.com/nvidia-oss-ci-staging/dynamo-hosted-remote-builder-poc/builds/5) | 8 seconds in a separate, identical build |
| [Multi-architecture `runtime`](https://buildkite.com/nvidia-oss-ci-staging/dynamo-hosted-remote-builder-poc/builds/3) | ARM64 ran under QEMU and failed on unavailable ARM64 build packages |
| [GitHub Actions baseline](https://github.com/ai-dynamo/dynamo/actions/runs/32443989551/job/96660039875) | 377 seconds for the multi-architecture build-and-push step |

The managed cache is effective, but this configuration is not a production
replacement. The first AMD64-only runtime build was slower than GitHub's
multi-architecture build-and-push step, native ARM64 hosted capacity was not
enabled for the organization, and no short-lived registry or SCCache access
was configured. Keep the required image pipeline on GitHub Actions unless a
follow-up pilot passes native ARM64, registry push, and compiler-cache tests.
