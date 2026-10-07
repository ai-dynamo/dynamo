<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang v0.5.21 GLM-5.3-Flash backports

The CUDA SGLang image build applies these six unmodified upstream diffs to
`lmsysorg/sglang:v0.5.21-cu130-runtime`. The required SGLang source revision is
`e00930c5489053f26d86b179cee0d087f846acbb`.

| Order | Change | Upstream commit |
| --- | --- | --- |
| 1 | Prefill/decode latency accounting | [3c4653b6](https://github.com/sgl-project/sglang/commit/3c4653b6a30cba798448f7fc34ad9a92a7f8e68b) |
| 2 | Linear-attention hooks and cache accounting | [8055ccd2](https://github.com/sgl-project/sglang/commit/8055ccd2cd36964541b36b817e40f5a758e31746) |
| 3 | PTX KDA safety and workspace handling | [c0296186](https://github.com/sgl-project/sglang/commit/c0296186b75ef2ff7a376b16161bb8b2f09f1de6) |
| 4 | PTX KDA support for SM100 | [57c81258](https://github.com/sgl-project/sglang/commit/57c81258d493a08953407ceabf2cf73b1307ff8b) |
| 5 | Prefill/decode retraction queue timestamps | [28c5e7f5](https://github.com/sgl-project/sglang/commit/28c5e7f5cbac8bb1a4e645579dbf56d83f442cbc) |
| 6 | Mamba slot cleanup before decode preallocation | [aec4b799](https://github.com/sgl-project/sglang/commit/aec4b799e61af971701d960e19d9d3062a6eb5ee) |

`series` records application order. `SHA256SUMS` covers the patches and `series`.
Keep the upstream patch files byte-for-byte intact, including their regression
tests. These source changes do not upgrade FlashInfer, Mooncake, or sgl-kernel.

The build runs `apply.sh --apply` as root before Python bytecode compilation and
verifies that Python resolves SGLang to `/sgl-workspace/sglang/python`. No startup
wrapper or writable source mount is needed when serving the resulting image.
The patch step applies to CUDA runtime, dev, and local-dev images; XPU is excluded.

The CUDA build also pins `transformers==5.19.0` and `tokenizers==0.23.2` without
reinstalling their dependencies. The base image's Transformers 5.12.1 lacks
`Glm5NextProcessor` and silently treats image/video requests as text-only (see
[SGLang #39831](https://github.com/sgl-project/sglang/issues/39831)). A build-time
check imports the image/video processors and verifies AutoProcessor registration.

For a standalone applicability check against a clean release checkout:

```bash
bash container/deps/sglang/patches/v0.5.21/apply.sh --check /path/to/sglang
```

Check mode is the default and does not modify the checkout. Apply mode changes
the worktree without committing or staging. Both modes reject a different source
revision, dirty tracked files, checksum mismatches, and patch conflicts. Reapplying
the bundle also fails. When upgrading SGLang, remove backports already included
upstream and validate any remaining patches against the new revision; do not
bypass the revision or applicability checks.

Patch application and Python syntax checks do not establish GPU correctness.
Validate SM100 PTX compilation, graph replay, prefix/state tracking, and
prefill/decode transfer with the patched image. These patches do not add BF16
persistent-state support.
