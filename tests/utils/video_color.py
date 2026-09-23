# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that VP9 encoded by the shipped ffmpeg keeps the colors it was given.

The in-tree FFmpeg 8.1.2, built with --disable-x86asm, converted RGB to YUV
wrongly on x86 and turned red frames magenta (ai-dynamo/dynamo#15198), while
every check passed because it only asked for a nonempty vp9 stream.

The oracle decodes with the shipped ffmpeg but reads the decoded YUV planes with
signalstats, so no libswscale conversion runs on the way back, and inverts
BT.601 here: a broken forward conversion cannot be cancelled by a matching
broken inverse.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

# Top-left, top-right, bottom-left, bottom-right.
QUADRANT_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
FRAME_SIZE = 128
# A clean round trip lands within about 1 of every channel; the swscale defect
# was off by more than 200.
TOLERANCE = 10.0
# Measure clear of the edges, where neighbouring quadrants bleed together.
_MARGIN = 8


def shipped_ffmpeg() -> str:
    """Resolve ffmpeg the way imageio, and so every encode path, does."""
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def quadrant_frames(num_frames: int = 8) -> np.ndarray:
    """``(num_frames, FRAME_SIZE, FRAME_SIZE, 3)`` uint8 RGB quadrant frames."""
    half = FRAME_SIZE // 2
    frame = np.empty((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for index, color in enumerate(QUADRANT_COLORS):
        row, col = divmod(index, 2)
        frame[row * half : (row + 1) * half, col * half : (col + 1) * half] = color
    return np.repeat(frame[None], num_frames, axis=0)


def _bt601_to_rgb(y: float, u: float, v: float) -> tuple[float, float, float]:
    """Invert swscale's default RGB-to-YUV conversion: BT.601, limited range."""
    luma = 1.16438356 * (y - 16.0)
    cb, cr = u - 128.0, v - 128.0
    r = luma + 1.59602678 * cr
    g = luma - 0.39176229 * cb - 0.81296764 * cr
    b = luma + 2.01723214 * cb
    return (min(255.0, max(0.0, r)), min(255.0, max(0.0, g)), min(255.0, max(0.0, b)))


def decoded_quadrant_colors(
    video: bytes, workdir: Path
) -> list[tuple[float, float, float]]:
    """Mean RGB of each quadrant's interior in the first decoded frame."""
    ffmpeg = shipped_ffmpeg()
    # Paths stay relative to workdir: a pytest tmp_path can contain brackets,
    # which filtergraph syntax reads as link labels.
    (workdir / "encoded").write_bytes(video)
    half = FRAME_SIZE // 2
    side = half - 2 * _MARGIN
    colors = []
    for index in range(len(QUADRANT_COLORS)):
        row, col = divmod(index, 2)
        x, y = col * half + _MARGIN, row * half + _MARGIN
        stats = f"quadrant{index}.stats"
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-y",
                "-i",
                "encoded",
                "-frames:v",
                "1",
                "-an",
                # crop and signalstats both take the decoder's yuv420p as-is, so
                # no libswscale conversion is negotiated ahead of the measurement.
                "-vf",
                f"crop={side}:{side}:{x}:{y},signalstats,"
                f"metadata=mode=print:file={stats}",
                # The CLI insists on an output, and VP9 is the only encoder it has.
                "-c:v",
                "libvpx-vp9",
                "-f",
                "webm",
                f"quadrant{index}.webm",
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"decoding with {ffmpeg} failed:\n{proc.stderr}"
        text = (workdir / stats).read_text()
        yuv = []
        for plane in "YUV":
            match = re.search(rf"lavfi\.signalstats\.{plane}AVG=(\S+)", text)
            assert match, f"signalstats reported no {plane}AVG:\n{text}"
            yuv.append(float(match.group(1)))
        colors.append(_bt601_to_rgb(*yuv))
    return colors


def assert_quadrant_colors(video: bytes, workdir: Path) -> None:
    """Assert each quadrant of ``quadrant_frames`` video decodes to its color."""
    decoded = decoded_quadrant_colors(video, workdir)
    error = max(
        abs(got - want)
        for rgb, expected in zip(decoded, QUADRANT_COLORS)
        for got, want in zip(rgb, expected)
    )
    assert error <= TOLERANCE, (
        f"the VP9 round trip changed the colors: encoded {QUADRANT_COLORS}, "
        f"decoded {[tuple(round(c) for c in rgb) for rgb in decoded]}, max channel "
        f"error {error:.1f} > {TOLERANCE} (see ai-dynamo/dynamo#15198)"
    )
