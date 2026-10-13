<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 NIXL compatibility patch

`motif_nixl.patch` applies two changes to the pinned vLLM NIXL source:

- Validate KV cache rank using the attention backend's MLA flag: rank 3 for MLA and
  rank 4 for non-MLA, instead of using the model-wide MLA flag.
- Retain the NHD cache layout for Motif and reject tensor-parallel sizes
  other than 2. Both prefill and decode workers must use TP2.

The runtime Dockerfile template applies the diff with `patch` during builds
using the Motif CUDA base. A dry run checks all hunks before writing.
Forward-only application with zero fuzz fails the build if the patch no longer
applies cleanly or has already been applied.
