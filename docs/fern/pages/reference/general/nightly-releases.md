---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Nightly Releases
subtitle: Nightly container images, Python wheels, install patterns, and current backend versions.
---

import { ReferenceStyles } from "@/components/ReferenceStyles";
import { NightlyBuilds } from "@/components/NightlyBuilds";

<ReferenceStyles />

Dynamo publishes nightly builds from `main`. Nightlies let you try the latest features and backend upgrades before they land in a stable release. This page covers what nightly publishes, how to install it, and which backend versions the current and recent nightlies ship.

<Warning>
**Nightly builds are experimental and are not QA-validated.** They are built from the tip of `main` and may contain bugs, breaking changes, or incomplete features. Use [stable releases](release-artifacts.mdx) for production workloads.
</Warning>

## Recent Nightlies

<NightlyBuilds />

## What Gets Published

Every night, the [Nightly CI pipeline](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/nightly-ci.yml) builds `main` and publishes:

- **Runtime container images (CUDA 13):** `vllm-runtime-nightly`, `sglang-runtime-nightly`, and `tensorrtllm-runtime-nightly` to NGC, each with an EFA variant under a `-efa` tag suffix.
- **Component container images:** `kubernetes-operator-nightly`, `dynamo-planner-nightly`, and `dynamo-frontend-nightly` to NGC.
- **Python wheels:** `ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/).
- **Helm chart:** `dynamo-platform` at a dated pre-release version, `X.Y.Z-dev.YYYYMMDD.g<shortsha>`.

The three runtime images and the wheels gate the release: if one of them fails to build, the whole nightly is held back. The component images and the Helm chart are staged fail-soft, so a failure there skips only that artifact for that night. Nightly does not publish Rust crates — for those, use a [stable or pre-release build](release-artifacts.mdx).

## Installing Nightly Containers

Nightly images live in their own `-nightly` NGC repositories so they cannot be pulled accidentally in place of a stable image. Every nightly build pushes an immutable `YYYYMMDD-<shortsha>` tag, and the `latest` and `nightly` tags both float to the most recent build.

```bash
# Most recent nightly
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest

# Pin one nightly build
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260826-27f09d5

# EFA variant, floating or pinned
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest-efa
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260826-27f09d5-efa
```

Pin the dated tag for anything you need to reproduce later: `latest` and `nightly` move every night, so the image behind them changes underneath you. The component repositories use the same tag scheme, without the EFA variants.

## Installing Nightly Wheels

Nightly wheels are published to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/), not the public PyPI. They are Linux manylinux builds for the Python versions in [Compatibility](compatibility.mdx); install on a supported Linux host or inside a Linux container. Nightly versions follow PEP 440 dev versioning, `X.Y.Z.devYYYYMMDD`.

```bash
# Latest nightly (uv)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Latest nightly (pip)
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Pin a specific nightly wheel
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ "ai-dynamo[vllm]==1.5.0.dev20260826"
```

Backend extras such as `ai-dynamo[vllm]` and `ai-dynamo[sglang]` use the same flags. For TensorRT-LLM, use the nightly container rather than a PyPI extra.

## Backend Versions

Nightlies track `main`, so the backend versions they ship change as `main` advances. To find which nightly or stable build ships a given backend version, and get the exact pull or install command, use the build selector in the [Kubernetes Quickstart](../../kubernetes/getting-started/quickstart.mdx#install-dynamo).

To confirm the exact versions a specific nightly shipped, read them from the pulled image:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest pip show vllm
```

## See Also

- [Release Artifacts](release-artifacts.mdx) — stable and pre-release artifact inventory
- [Compatibility](compatibility.mdx) — hardware, platform, CUDA, and driver support
- [Model Early Access Builds](model-early-access-builds.mdx) — model-specific pre-release container builds
