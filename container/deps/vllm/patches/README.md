<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek V4.1 Flash runtime patch stack

Base: `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`
(verified 2026-10-05, CUDA 13.0.2, Ubuntu 24.04, amd64 and arm64).
Both architectures' image config pins source commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, matching the release tag.

| Architecture | Child manifest digest |
| --- | --- |
| arm64 | `sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971` |
| amd64 | `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` |

PR [58215](https://github.com/vllm-project/vllm/pull/58215), commit
`9ef37771beac1c1ce6a8b7ceae1f7ebeb6f51800`, is an ancestor of the base and
its sparse top-k sentinel bound is present natively. No overlay is needed.

Apply the remaining requested patches in filename order:

| Patch | Purpose | Source |
| --- | --- | --- |
| `000-pr58038` | Optional telemetry must not fail a completed transfer | [58038](https://github.com/vllm-project/vllm/pull/58038) |
| `001-pr57662` | Deduplicate transfer regions by `(base_addr, block_len)` | [57662](https://github.com/vllm-project/vllm/pull/57662), `ebd21ca7462b0e495918dd508228cb383c327809` |
| `002-pr55374` | Piecewise-prefix loading and range-aware connector selection | [55374](https://github.com/vllm-project/vllm/pull/55374), `d3e956268db8682b43a059fd622128fc320e2762` and `877a3c671e69d628ecd868219ac10b554088bd73` |

These three PRs are open and their functionality is absent from the base as of
2026-10-05. All three patches apply to v0.31.0 with zero fuzz. Patch 002 has
line offsets against this release;
no additional runtime patches are introduced. The patches are ports of the
stack carried by Dynamo PR 15112.
The region-key port preserves the newer `route_packed_layers` handling. The
range-load port reuses the existing `cdiv` import and advances the native NIXL
connector version from 13 to 14 to keep the new protocol distinguishable.
All workers participating in a transfer must use the same patched protocol.
The CUDA dev and local-dev images retain the unpatched protocol 13. Do not pair
one of these workers with a patched runtime worker (protocol 14).

Validation: zero-fuzz application to the exact source commit, Python 3.12
syntax checks, CPU source probes for pull/push signatures, token-window slicing,
and successful-transfer behavior when telemetry throws. The runtime image build
also runs `../validate_patches_runtime.py` against the installed wheel.
