---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Model Early Access Builds
subtitle: Per-model early access container builds shipped ahead of stable releases
---

> **See also:** [Release Artifacts](release-artifacts.md) for the stable release inventory and platform previews | [Support Matrix](support-matrix.md) for hardware and platform compatibility

<Warning>
**Model early access builds do not go through QA validation.** They are experimental builds intended for early testing of a specific model. They may contain bugs, require pinned runtime flags, and receive no patch support. Use a stable release container for production workloads unless the build's GA path below says the recipe is promoted.
</Warning>

A **model early access build** packages a single model's recipe on one runtime container, tagged `X.Y.Z-<model>-dev.N` and cut from a side branch ahead of that model's launch — independently of the stable release cadence. When the model's backend patches land upstream in the versions a stable release ships, the recipe is **promoted** to the plain `:X.Y.Z` release container and the early access image is no longer needed.

Full-platform early access builds (`vX.Y.Z-dev.N`, covering all runtimes, wheels, crates, and Helm charts) are **platform previews** and are documented under [Early Access Artifacts](release-artifacts.md#early-access-artifacts) instead.

**GA path legend:**

- **Promoted** — the recipe runs on the stock `:X.Y.Z` stable release container; the early access image is superseded.
- **Dev-only** — the model still requires this early access image (its patches are not yet in a stable release).
- **Recipe in GA** — the model ships as a recipe on the standard release container.

Pull any image below with:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/<runtime>:<tag>
```

## v1.4.0 release

| Model | Tag | Runtime | Shipped | Recipe | GA path |
|-------|-----|---------|---------|--------|---------|
| Inkling | `1.4.0-inkling-dev.1` | sglang-runtime | Jul 17, 2026 | [Inkling recipe](../recipes/inkling.mdx) | Dev-only (v1.4.0 line) |

### v1.4.0-inkling-dev.1

- **GitHub Tag:** [v1.4.0-inkling-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.0-inkling-dev.1)
- **NGC Image:** `sglang-runtime:1.4.0-inkling-dev.1`
- **GA path:** Dev-only — first build on the v1.4.0 line; targets the next stable release.

## v1.3.0 release

| Model | Tag | Runtime | Shipped | Recipe | GA path |
|-------|-----|---------|---------|--------|---------|
| GLM-5.2 | `1.3.0-glm-5.2-dev.1` | sglang-runtime | Jul 20, 2026 (with v1.3.0) | [GLM-5 NVFP4 recipe](../recipes/glm-5-nvfp4.mdx) | **Dev-only** — SGLang cherry-picks not upstream |
| MiniMax-M3 | `1.3.0-minimax-m3-dev.1` | vllm + sglang + tensorrtllm | Jun 12, 2026 | [recipes/minimax-m3 (branch)](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-minimax-m3-dev.1/recipes/minimax-m3) | **Promoted → `:1.3.0`** |
| DeepSeek-V4 | `1.3.0-deepseek-v4-dev.1` | tensorrtllm-runtime | Jun 6, 2026 | [recipes/deepseek-v4](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4) | Recipe in v1.3.0 |
| Nemotron-3-Ultra | `1.3.0-nemotron-ultra-dev.1` | vllm-runtime | Jun 5, 2026 | [Nemotron-3-Ultra recipe](../recipes/nemotron-3-ultra.mdx) | **Dev-only** — patches not upstream; pinned flags required |
| Nemotron-3-Super | `1.3.0-nemotron-super-dev.1` | vllm-runtime | Jun 4, 2026 | [Nemotron-3-Super recipe](../recipes/nemotron-3-super.mdx) | **Promoted → `:1.3.0`** |
| Kimi-K2.6 | `1.3.0-kimi-k2.6-dev.1` | vllm-runtime | Jun 4, 2026 | [Kimi-K2.6 recipe](../recipes/kimi-k2-6.mdx) | **Promoted → `:1.3.0`** |
| Cosmos-3 | `1.3.0-cosmos3-dev.1` | vllm-runtime | Jun 1, 2026 | [examples (branch)](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-cosmos3-dev.1/examples/backends/vllm/launch) | **Dev-only** — Dynamo integration not merged |

### v1.3.0-glm-5.2-dev.1

- **GitHub Tag:** `v1.3.0-glm-5.2-dev.1` *(ships with the v1.3.0 GA release)*
- **NGC Image:** `sglang-runtime:1.3.0-glm-5.2-dev.1`
- **Upstream status:** the container carries SGLang cherry-picks (stability, config parsing, model support) opened upstream against [sgl-project/sglang](https://github.com/sgl-project/sglang) but not yet in a released SGLang.
- **GA path:** Dev-only until the SGLang fixes land in a release that a Dynamo stable release ships.

### v1.3.0-minimax-m3-dev.1

- **Branch:** [release/1.3.0-minimax-m3-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-minimax-m3-dev.1)
- **GitHub Tag:** [v1.3.0-minimax-m3-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-minimax-m3-dev.1)
- **NGC Images:** `vllm-runtime`, `sglang-runtime`, and `tensorrtllm-runtime` at `1.3.0-minimax-m3-dev.1`
- **Upstream status:** Dynamo [#10983](https://github.com/ai-dynamo/dynamo/pull/10983) (MiniMax-M3 Dynamo changes) merged to `main` 2026-06-29 and is in `release/1.3.0`; the M2 tool-calling fix ([#11554](https://github.com/ai-dynamo/dynamo/pull/11554)) is cherry-picked to `release/1.3.0` via [#11621](https://github.com/ai-dynamo/dynamo/pull/11621). No backend container patches.
- **GA path:** **Promoted** — the recipes run on the stock `:1.3.0` containers.

### v1.3.0-deepseek-v4-dev.1

- **GitHub Tag:** [v1.3.0-deepseek-v4-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-deepseek-v4-dev.1)
- **NGC Image:** `tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`
- **Backends:** TensorRT-LLM `1.3.0rc15post1` (TRT-LLM patches over the `main` base)
- **GA path:** Recipe in GA — DeepSeek-V4 Flash and Pro recipes ship in v1.3.0 on the standard TensorRT-LLM release container.

### v1.3.0-nemotron-ultra-dev.1

- **Branch:** [release/1.3.0-nemotron-ultra-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-nemotron-ultra-dev.1)
- **GitHub Tag:** [v1.3.0-nemotron-ultra-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-nemotron-ultra-dev.1)
- **NGC Image:** `vllm-runtime:1.3.0-nemotron-ultra-dev.1`
- **Backends:** vLLM `v0.22.0` with six container patches
- **Upstream status:**
  - vLLM [#40932](https://github.com/vllm-project/vllm/pull/40932) (Qwen3-VL deepstack boundary fix) — upstream, first released in vLLM `v0.22.0`; in `v0.23.0` ✓
  - vLLM [#42554](https://github.com/vllm-project/vllm/pull/42554) (Mamba prefix-cache PD runtime) — upstream, in vLLM `v0.23.0` ✓
  - Four Ultra-specific patches (hybrid Mamba/attention hash-block KV events, MTP DS conv-state layout, SSM NIXL tail transfer, DS-tail copy i64 offsets) — **not upstreamed**; no vLLM PRs opened
- **GA path:** **Dev-only** — the four un-upstreamed patches are not in the vLLM `v0.23.0` that v1.3.0 ships. Requires two pinned runtime flags: `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` and `--no-enable-flashinfer-autotune`.

### v1.3.0-nemotron-super-dev.1

- **Branch:** [release/1.3.0-nemotron-super-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-nemotron-super-dev.1)
- **GitHub Tag:** [v1.3.0-nemotron-super-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-nemotron-super-dev.1)
- **NGC Image:** `vllm-runtime:1.3.0-nemotron-super-dev.1`
- **Backends:** vLLM `v0.22.0` with two container patches
- **Upstream status:**
  - vLLM [#40984](https://github.com/vllm-project/vllm/pull/40984) (emit KV-cache metadata) — merged 2026-05-12, first released in vLLM `v0.22.0`; in `v0.23.0` ✓
  - vLLM [#40932](https://github.com/vllm-project/vllm/pull/40932) (Qwen3-VL deepstack boundary fix) — upstream, first released in vLLM `v0.22.0`; in `v0.23.0` ✓
- **GA path:** **Promoted** — both patches are in the vLLM `v0.23.0` that v1.3.0 ships; the recipe runs on the stock `vllm-runtime:1.3.0`.

### v1.3.0-kimi-k2.6-dev.1

- **Branch:** [release/1.3.0-kimi-k2.6-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-kimi-k2.6-dev.1)
- **GitHub Tag:** [v1.3.0-kimi-k2.6-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-kimi-k2.6-dev.1)
- **NGC Image:** `vllm-runtime:1.3.0-kimi-k2.6-dev.1`
- **Upstream status:** the build's only container patch, vLLM [#40932](https://github.com/vllm-project/vllm/pull/40932) (Qwen3-VL deepstack boundary fix), is upstream and first released in vLLM `v0.22.0`; in `v0.23.0` ✓; the K2.6 recipes themselves merged to `main` via Dynamo [#10187](https://github.com/ai-dynamo/dynamo/pull/10187).
- **GA path:** **Promoted** — the recipes run on the stock `vllm-runtime:1.3.0`.

### v1.3.0-cosmos3-dev.1

- **Branch:** [release/1.3.0-cosmos3-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-cosmos3-dev.1)
- **GitHub Tag:** [v1.3.0-cosmos3-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0-cosmos3-dev.1)
- **NGC Image:** `vllm-runtime:1.3.0-cosmos3-dev.1`
- **Backends:** vLLM-Omni pinned to the Cosmos3 support commit
- **Upstream status:**
  - vllm-omni [#3454](https://github.com/vllm-project/vllm-omni/pull/3454) (Cosmos3 support) — merged 2026-06-01, released in vLLM-Omni `v0.23.0rc1` ✓
  - Dynamo [#10132](https://github.com/ai-dynamo/dynamo/pull/10132) (Cosmos3 support in the vLLM-Omni backend) — **open, not merged**
- **GA path:** **Dev-only** — without #10132 the v1.3.0 containers cannot run Cosmos3, even though the vLLM-Omni side is released. Launch scripts and docs live on the build branch: [docs/backends/vllm/cosmos3.md](https://github.com/ai-dynamo/dynamo/blob/release/1.3.0-cosmos3-dev.1/docs/backends/vllm/cosmos3.md), [examples/backends/vllm/launch](https://github.com/ai-dynamo/dynamo/tree/release/1.3.0-cosmos3-dev.1/examples/backends/vllm/launch).

## v1.2.0 release

| Model | Tag | Runtime | Shipped | Recipe | GA path |
|-------|-----|---------|---------|--------|---------|
| DeepSeek-V4 preview | `1.2.0-deepseek-v4-dev.3` | vllm + sglang | May 9, 2026 | superseded | Superseded — recipe in v1.3.0 |
| DeepSeek-V4 preview | `1.2.0-deepseek-v4-dev.2` | vllm + sglang | May 1, 2026 | superseded | Superseded — recipe in v1.3.0 |
| DeepSeek-V4 preview | `1.2.0-sglang-deepseek-v4-dev.1` | sglang | Apr 25, 2026 | superseded | Superseded — recipe in v1.3.0 |

> The **Tag** column shows the GitHub release tag. Unlike the single-image builds above, these previews publish per-arch/CUDA container tags (for example `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`) — see each build's **Container Images** table below for the pullable tags.

### v1.2.0-deepseek-v4-dev.3

- **Branch:** [release/1.2.0-deepseek-v4-dev.3](https://github.com/ai-dynamo/dynamo/tree/release/1.2.0-deepseek-v4-dev.3)
- **GitHub Tag:** [v1.2.0-deepseek-v4-dev.3](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.3)
- **Backends:** vLLM `v0.20.1` (DSv4 stabilization patch over `v0.20.0` native DSv4 support) | SGLang upstream `lmsysorg/sglang:deepseek-v4-blackwell` preview (refreshed for dev.3) | NIXL `v0.10.1`
- **Coverage:** Partial -- DeepSeek-V4-Flash and V4-Pro only. vLLM and SGLang containers are published for Blackwell (B200 plus GB200); no TensorRT-LLM container, no other component containers, no Helm charts, no wheels. Model early access build for V4 model support; not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` | vLLM `v0.20.1` | `v13.0` | AMD64/ARM64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.3` | SGLang upstream DSv4 preview | `v12.9` | AMD64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.3` | SGLang upstream DSv4 preview | `v13.0` | ARM64 |

#### Python Wheels

Not published for this early access build. Use the `v1.1.1` wheels or `v1.1.0-dev.3` from [pypi.nvidia.com](https://pypi.nvidia.com/).

#### Helm Charts

Not published for this early access build. Use `v1.1.1` charts for platform install.

#### Rust Crates

Not shipped for early access builds.

### v1.2.0-sglang-deepseek-v4-dev.1

- **Branch:** [release/1.2.0-sglang-deepseek-v4-dev.1](https://github.com/ai-dynamo/dynamo/tree/release/1.2.0-sglang-deepseek-v4-dev.1)
- **GitHub Tag:** [v1.2.0-sglang-deepseek-v4-dev.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-sglang-deepseek-v4-dev.1)
- **Backends:** SGLang upstream `lmsysorg/sglang:deepseek-v4-blackwell` preview
- **Coverage:** Partial -- DeepSeek-V4-Flash and V4-Pro only. SGLang container only, published for Blackwell (B200). No vLLM or TensorRT-LLM containers, no other component containers, no Helm charts, no wheels. Earliest DSv4 preview build; superseded by dev.2/dev.3; not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` | SGLang (DSv4 Blackwell preview) | `v12.9` | AMD64 |

#### Python Wheels

Not published for this early access build. Use the `v1.1.1` wheels or `v1.1.0-dev.3` from [pypi.nvidia.com](https://pypi.nvidia.com/).

#### Helm Charts

Not published for this early access build. Use `v1.1.1` charts for platform install.

#### Rust Crates

Not shipped for early access builds.

### v1.2.0-deepseek-v4-dev.2

- **Branch:** [release/1.2.0-deepseek-v4-dev.2](https://github.com/ai-dynamo/dynamo/tree/release/1.2.0-deepseek-v4-dev.2)
- **GitHub Tag:** [v1.2.0-deepseek-v4-dev.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.2.0-deepseek-v4-dev.2)
- **Backends:** vLLM `v0.20.0` (native DeepSeek-V4 support) | SGLang upstream `lmsysorg/sglang:deepseek-v4-blackwell` preview | NIXL `v0.10.1`
- **Coverage:** DeepSeek-V4-Flash and V4-Pro only. vLLM and SGLang containers are published for Blackwell. TensorRT-LLM container, other component containers, Helm charts, and wheels are not published for this tag. Model early access build for V4 model support; not QA-gated.

#### Container Images

| Image:Tag | Backend | CUDA | Arch |
|-----------|---------|------|------|
| `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.2` | vLLM `v0.20.0` | `v13.0` | AMD64/ARM64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda12-dev.2` | SGLang upstream DSv4 preview | `v12.9` | AMD64 |
| `sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.2` | SGLang upstream DSv4 preview | `v13.0` | ARM64 |

#### Python Wheels

Not published for this early access build. Use the `v1.1.0` wheels or `v1.1.0-dev.3` from [pypi.nvidia.com](https://pypi.nvidia.com/).

#### Helm Charts

Not published for this early access build. Use `v1.1.0` charts for platform install.

#### Rust Crates

Not shipped for early access builds.
