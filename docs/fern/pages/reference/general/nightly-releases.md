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

- **Container images (CUDA 13):** `vllm-runtime-nightly`, `sglang-runtime-nightly`, `tensorrtllm-runtime-nightly`, `dynamo-frontend-nightly`, `kubernetes-operator-nightly`, and `dynamo-planner-nightly` to NGC.
- **Helm chart:** `dynamo-platform-nightly` to NGC, published when the operator image was staged in the same run.
- **Python wheels:** `ai-dynamo`, `ai-dynamo-runtime`, and `kvbm` to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/).

Nightly deliberately does **not** publish `snapshot-agent` or Rust crates. For those, use a [stable or pre-release build](release-artifacts.mdx).

## Installing Nightly Containers

Nightly images live in their own `-nightly` NGC repositories so they cannot be pulled accidentally in place of a stable image. Every nightly image carries a floating `:latest` tag for the most recent build, and an immutable `:YYYYMMDD-<shortsha>` tag for the specific build.

```bash
# Latest nightly. This tag moves every night.
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/kubernetes-operator-nightly:latest
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-planner-nightly:latest

# A specific nightly, pinned. Use this for anything reproducible.
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260825-f7de8eb
```

Pin the immutable tag for CI jobs, benchmarks, and bug reports, so the build under test does not change overnight. Browse the **Tags** tab on a nightly repository to find the tag you want.

## Installing Nightly Wheels

Nightly wheels are published to the NVIDIA prerelease index at [pypi.nvidia.com](https://pypi.nvidia.com/), not the public PyPI. They are Linux manylinux builds for the Python versions in [Compatibility](compatibility.mdx); install on a supported Linux host or inside a Linux container. Nightly versions follow PEP 440 dev versioning, `X.Y.Z.devYYYYMMDD`.

```bash
# Latest nightly (uv)
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Latest nightly (pip)
pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Pin a specific nightly wheel
uv pip install --pre --extra-index-url https://pypi.nvidia.com/ "ai-dynamo[vllm]==1.4.0.dev20260803"
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
