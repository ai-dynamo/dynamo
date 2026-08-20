<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bulwark figure sources

## Architecture figures

`images/layout-overview.svg`, `images/fleet-overview.svg`, `images/failover-timeline.svg`,
`images/gpu-memory.svg`, and `images/aliased-kv-cache.svg` are hand-authored against the
Dynamo Dark token set (`#000000` canvas, `#76b900` accent). Edit the SVG directly.

## Benchmark figures

`images/bench-*.png` come from the 3-node Kimi-K2.6 cascade run described in the post. The
harness emits Matplotlib charts on a light canvas; `darkify.py` remaps them onto the dark
canvas the rest of the site uses. The transform inverts lightness in HLS space and leaves hue
and saturation alone, so no data point, axis, or label is altered — only its color.

```bash
python3 darkify.py <chart>.png    # writes <chart>_dark.png alongside the input
```

Requires `numpy` and `pillow`.
