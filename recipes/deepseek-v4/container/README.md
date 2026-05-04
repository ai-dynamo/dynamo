<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Reference Containers

Shared reference Dockerfiles for the DeepSeek-V4 family — used by both [`deepseek-v4-flash`](../deepseek-v4-flash/) and [`deepseek-v4-pro`](../deepseek-v4-pro/). Nothing in either image is recipe-specific; the model is selected at runtime via `--model-path` (SGLang).

| Backend | Dockerfile | Base image | Build flow |
|---------|-----------|-----------|------------|
| SGLang (B200)  | [`sglang/Dockerfile.dsv4.sglang.b200`](sglang/Dockerfile.dsv4.sglang.b200)   | `lmsysorg/sglang:deepseek-v4-blackwell` (digest-pinned, amd64)       | Two-stage; Dynamo runtime image as donor |
| SGLang (GB200) | [`sglang/Dockerfile.dsv4.sglang.gb200`](sglang/Dockerfile.dsv4.sglang.gb200) | `lmsysorg/sglang:deepseek-v4-grace-blackwell` (digest-pinned, arm64) | Two-stage; Dynamo runtime image as donor |

NVIDIA also publishes the prebuilt images for vLLM and SGLang which manifests pull directly:
- `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2` (multi-arch)
- `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.2` (arm64 only)
- `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.2` (amd64 only)

The `cudaXY` suffix encodes the CUDA major version baked into the image, not the hardware target.

> **Optional:** users may also build the standard Dynamo vLLM runtime image via `container/render.py`. See [`<repo_root>/container/README.md`](../../../container/README.md).

## SGLang (`sglang/Dockerfile.dsv4.sglang.b200`)

Two-stage build: a Dynamo SGLang runtime image as the donor (for nats / etcd / UCX / NIXL and the Dynamo wheels + Python source), layered onto the upstream SGLang dsv4 base.

### Step 1 — Build the Dynamo SGLang runtime

From the **repo root**:

```bash
container/render.py --framework sglang --target runtime --output-short-filename
docker build -t dynamo:latest-sglang-runtime -f container/rendered.Dockerfile .
```

This produces the local tag `dynamo:latest-sglang-runtime`, which Step 2 expects as `DYNAMO_SRC_IMAGE`. The donor must contain the V4 tool/reasoning parsers and the SGLang routed_experts fix; the build asserts on this with a post-install `assert 'deepseek_v4' in get_tool_parser_names()`.

See [`<repo_root>/container/README.md`](../../../container/README.md) for runtime-image build details and alternative tags.

### Step 2 — Build the dsv4 overlay

Still from the **repo root**:

```bash
docker build \
  -f recipes/deepseek-v4/container/sglang/Dockerfile.dsv4.sglang.b200 \
  -t <your-registry>/sglang-dsv4:<tag> \
  .
```

The Dockerfile takes nothing from the build context (everything comes from `FROM` / `COPY --from=`), so any context directory works.

### Build args

| Arg | Default | Purpose |
|-----|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-sglang-runtime` | Source for nats / etcd / UCX / NIXL and the V4-aware Dynamo wheels. Default matches Step 1; override with a published Dynamo SGLang runtime tag for reproducible builds without rebuilding locally. |
| `DSV4_BASE_IMAGE`  | `lmsysorg/sglang:deepseek-v4-blackwell@sha256:da2acdc8...` | The DeepSeek-V4 SGLang base. Digest-pinned for byte-stable rebuilds. |

### Wire into a recipe

Push:

```bash
docker push <your-registry>/sglang-dsv4:<tag>
```

Set the `image:` field (Frontend + decode worker) in the recipe's SGLang manifest, then follow the recipe's Quick Start:

- Flash → [`../deepseek-v4-flash/sglang/agg/deploy.yaml`](../deepseek-v4-flash/sglang/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-flash/README.md#quick-start).
- Pro → [`../deepseek-v4-pro/sglang/agg/deploy.yaml`](../deepseek-v4-pro/sglang/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-pro/README.md#quick-start).
