# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the moving-triangle source frames for the video serve fixtures.

Why a triangle. The current fixtures are ffmpeg `testsrc2` colour bars, and the
serve tests assert on whatever a vision model happens to call them. That is
unstable: the same clip drew "a test pattern or a color calibration screen" in CI
and "a digital or pixel art animation" on a GPU box. A single unmistakable object
gives a stable token to assert.

Why one source, three encodings. Emitting raw RGB here and letting the encoder
vary means VP9, H.264 and H.265 carry *identical* frames. Every codec path can
then assert the same word, and a difference in model output points at decode
rather than at phrasing. It also allows a direct cross-codec frame comparison,
which is a far stronger check than any word list.

Why motion. A moving triangle also exercises temporal sampling: a regression that
collapses frame selection to a single frame changes what the model can say about
movement, where a static image would hide it.

Regenerate all three fixtures (needs an ffmpeg with libx264/libx265/libvpx-vp9;
the Dynamo runtime images deliberately ship no such encoder):

    python3 make_triangle_fixture.py > frames.rgb
    for spec in "libvpx-vp9 -strict -2:triangle_240p_10.mp4" \\
                "libx264:triangle_240p_10_h264.mp4" \\
                "libx265 -tag:v hvc1:triangle_240p_10_h265.mp4"; do
        codec="${spec%%:*}"; out="${spec##*:}"
        ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 320x240 -r 10 -i frames.rgb \\
            -c:v $codec -g 1 -pix_fmt yuv420p "$out"
    done

Encoding all three from the same raw frames is the point: it keeps the decoded
content identical across codecs. Measured round-trip against the source is
1.59 / 1.68 / 1.63 mean absolute per-pixel difference.
"""

from __future__ import annotations

import sys

import numpy as np

WIDTH, HEIGHT, FRAMES = 320, 240, 10
BACKGROUND = (16, 16, 48)  # near-black blue; high contrast, not "colourful"
TRIANGLE = (250, 230, 40)  # saturated yellow


def _frame(index: int) -> np.ndarray:
    """One frame: a filled triangle translated horizontally across the clip."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :] = BACKGROUND

    # Sweep left to right so motion is unambiguous across the sampled frames.
    travel = WIDTH - 160
    x0 = 20 + int(travel * index / max(FRAMES - 1, 1))
    apex = (x0 + 60, 40)
    left = (x0, HEIGHT - 40)
    right = (x0 + 120, HEIGHT - 40)

    # Fill by half-plane tests: inside iff on the same side of all three edges.
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    def side(a, b):
        return (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])

    d1, d2, d3 = side(apex, left), side(left, right), side(right, apex)
    inside = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    img[inside] = TRIANGLE
    return img


def main() -> int:
    out = sys.stdout.buffer
    for i in range(FRAMES):
        out.write(_frame(i).tobytes())
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
