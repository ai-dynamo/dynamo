<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek V4.1 Flash runtime patch stack

Base: `vllm/vllm-openai:nightly-ac9126e58aa7bbab1856ba6593ba4d5003fea516@sha256:17d08dc42a7b7a6a071ce52fb060e58ae3a242bdc8341961155413bf124118e6`
(2026-10-01, CUDA 13.0.2, amd64 and arm64).

PR [58215](https://github.com/vllm-project/vllm/pull/58215), commit
`9ef37771beac1c1ce6a8b7ceae1f7ebeb6f51800`, is an ancestor of the base and
its sparse top-k sentinel bound is present natively. No overlay is needed.

Apply the remaining requested patches in filename order:

| Patch | Purpose | Source |
| --- | --- | --- |
| `0002-pr58038` | Optional telemetry must not fail a completed transfer | [58038](https://github.com/vllm-project/vllm/pull/58038) |
| `0003-pr57662` | Deduplicate transfer regions by `(base_addr, block_len)` | [57662](https://github.com/vllm-project/vllm/pull/57662), `ebd21ca7462b0e495918dd508228cb383c327809` |
| `0004-pr55374` | Piecewise-prefix loading and range-aware connector selection | [55374](https://github.com/vllm-project/vllm/pull/55374), `d3e956268db8682b43a059fd622128fc320e2762` and `877a3c671e69d628ecd868219ac10b554088bd73` |

These three PRs are open and their functionality is absent from the base as of
2026-10-01. The patches are ports of the stack carried by Dynamo PR 15112.
The region-key port preserves the newer `route_packed_layers` handling. The
range-load port reuses the existing `cdiv` import and advances the native NIXL
connector version from 13 to 14 to keep the new protocol distinguishable.
All workers participating in a transfer must use the same patched protocol.

Validation: zero-fuzz application to the exact source commit, Python 3.12
syntax checks, CPU source probes for pull/push signatures, token-window slicing,
and successful-transfer behavior when telemetry throws. The runtime image build
also runs `../validate_patches_runtime.py` against the installed wheel.
