<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash Reference Container

This Dockerfile builds the SGLang runtime overlay for DeepSeek-V4.1-Flash on B200 (amd64) and GB200 (arm64). The model is selected at runtime with SGLang's `--model-path` option.

| Backend | Dockerfile | Base image | Build flow |
|---------|------------|------------|------------|
| SGLang (B200 / GB200) | [`sglang/Dockerfile.dsv41.sglang`](sglang/Dockerfile.dsv41.sglang) | `lmsysorg/sglang` (digest-pinned, multi-arch) | Two-stage; Dynamo runtime image as donor |

## SGLang (`sglang/Dockerfile.dsv41.sglang`)

The upstream SGLang preview image is the final base. A Dynamo SGLang runtime image donates nats, etcd, platform-matched UCX and NIXL, Dynamo wheels, and Python sources. `TARGETARCH` selects `x86_64-linux-gnu` for B200 and `aarch64-linux-gnu` for GB200.

### Step 1 — Build the Dynamo SGLang runtime

From the **repository root**:

```bash
container/render.py --framework sglang --target runtime --output-short-filename
docker build -t dynamo:latest-sglang-runtime -f container/rendered.Dockerfile .
```

This creates the default `DYNAMO_SRC_IMAGE` used in Step 2. The donor must contain a rebuilt Dynamo wheel with the V4.1 tool/reasoning parser. The overlay verifies this after installation with `assert 'deepseek_v41' in get_tool_parser_names()`.

> [!IMPORTANT]
> The parser is not in the currently published Dynamo crates. Until the V4.1 parser support lands in the donor runtime, this assertion intentionally fails and the image must not be used.

See [`<repo_root>/container/README.md`](../../../container/README.md) for runtime-image build details and alternative tags.

### Step 2 — Build the V4.1 overlay

Still from the **repository root**:

```bash
docker build \
  -f recipes/deepseek-v4.1/container/sglang/Dockerfile.dsv41.sglang \
  -t <your-registry>/sglang-dsv41:<tag> \
  .
```

The Dockerfile takes nothing from the build context; all artifacts come from `FROM` images and `COPY --from=` stages. Any context directory therefore works.

### Build arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-sglang-runtime` | Donor for nats, etcd, UCX, NIXL, and the V4.1-aware Dynamo wheels. Build this image in Step 1 or override it with an equivalent published image. |
| `DSV41_BASE_IMAGE` | `lmsysorg/sglang@sha256:3dbc3130...` | DeepSeek-V4.1-Flash SGLang preview base, digest-pinned as a multi-architecture image. |

### Wire into a recipe

Push the built overlay:

```bash
docker push <your-registry>/sglang-dsv41:<tag>
```

Set the `image:` field for the Frontend and SGLang worker in the recipe manifest, then follow that recipe's Quick Start.
