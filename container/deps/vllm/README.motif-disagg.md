<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 NIXL compatibility patch

`apply_motif_nixl_patch.py` applies two changes to the pinned vLLM NIXL source:

- Validate KV cache rank using the attention backend's MLA flag: rank 3 for MLA and
  rank 4 for non-MLA, instead of using the model-wide MLA flag.
- Retain the NHD cache layout for Motif and reject tensor-parallel sizes
  other than 2. Both prefill and decode workers must use TP2.

The patch checks both source hashes before writing, rejects unknown source
versions, and supports repeated application without further changes.

The runtime Dockerfile template applies it during builds using the Motif CUDA
base. `Dockerfile.motif-disagg` applies the same fixes to a pinned existing
runtime image.
