# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move a light-canvas Matplotlib chart onto the dark canvas the docs site uses.

Lightness is inverted in HLS space while hue and saturation are preserved, so white
becomes black, black text becomes white, and the plotted series keep their identity.
Only pixel color changes; no data, axis, or label is touched.
"""

import sys

import numpy as np
from PIL import Image


def darkify(src, dst):
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    chroma = mx - mn
    lightness = (mx + mn) / 2.0

    saturated = chroma > 1e-6
    sat = np.zeros_like(lightness)
    sat[saturated] = np.where(
        lightness[saturated] < 0.5,
        chroma[saturated] / (mx + mn)[saturated],
        chroma[saturated] / (2.0 - mx - mn)[saturated],
    )

    hue = np.zeros_like(lightness)
    sector = saturated & (mx == r)
    hue[sector] = ((g - b)[sector] / chroma[sector]) % 6
    sector = saturated & (mx == g)
    hue[sector] = ((b - r)[sector] / chroma[sector]) + 2
    sector = saturated & (mx == b)
    hue[sector] = ((r - g)[sector] / chroma[sector]) + 4
    hue /= 6.0

    inverted = 1.0 - lightness
    c = (1 - np.abs(2 * inverted - 1)) * sat
    x = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = inverted - c / 2
    sextant = np.floor(hue * 6).astype(int) % 6
    z = np.zeros_like(c)
    conds = [sextant == i for i in range(6)]
    out = np.stack(
        [
            np.select(conds, [c, x, z, z, x, c]),
            np.select(conds, [x, c, c, x, z, z]),
            np.select(conds, [z, z, x, c, c, x]),
        ],
        axis=-1,
    ) + m[..., None]

    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        darkify(path, path.replace(".png", "_dark.png"))
