# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: synthetic video generation must not overflow its grids.

The per-frame gradients add ``frame_idx * speed`` to the index grids. Those
grids were ``np.uint16``, and under NEP 50 NumPy 2 raises ``OverflowError``
for a Python integer above 65535 rather than wrapping, so a longer or larger
clip aborted the run. The defaults stay far below the limit, which is why this
only showed up at larger sizes and durations.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from typing_extensions import Self

from benchmarks.multimodal.jsonl import generate_videos

# Leave these tests unmarked. The root conftest.py then adds ``pre_merge``,
# ``gpu_0`` and ``defaulted``, and the dynamo-runtime pipeline runs tests with
# ``defaulted`` in its CPU parallel job.


class _CollectingWriter:
    """Stands in for an imageio writer, so no ffmpeg is needed."""

    def __init__(self, keep_frames: bool) -> None:
        self.keep_frames = keep_frames
        self.frame_count = 0
        self.frames: list[np.ndarray] = []
        self.last_dtype: np.dtype | None = None
        self.last_shape: tuple[int, ...] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def append_data(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        self.last_dtype = frame.dtype
        self.last_shape = frame.shape
        if self.keep_frames:
            self.frames.append(frame.copy())


@pytest.fixture
def stub_imageio(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``imageio.v2`` and hand back the writer it returns."""

    def install(keep_frames: bool = False) -> _CollectingWriter:
        writer = _CollectingWriter(keep_frames)
        v2 = types.ModuleType("imageio.v2")
        v2.get_writer = lambda *_args, **_kwargs: writer  # type: ignore[attr-defined]
        root = types.ModuleType("imageio")
        root.v2 = v2  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "imageio", root)
        monkeypatch.setitem(sys.modules, "imageio.v2", v2)
        return writer

    return install


def test_long_clip_does_not_overflow_the_index_grids(stub_imageio, tmp_path) -> None:
    # The overflow needs only frame_idx * speed to pass 65535, so this is the
    # smallest clip that still crosses it: seed 4 draws speed_x = 150 at width
    # 1280 and frame_idx reaches 455, so the gradient adds 68250. The bug
    # report's 1280x720 for 30 s crosses the same bound but moves 71x more
    # pixel data for the same assertions.
    writer = stub_imageio()

    generate_videos._write_synthetic_video(
        tmp_path / "clip.mp4", seed=4, width=1280, height=16, fps=24, seconds=19
    )

    assert writer.frame_count == 24 * 19
    assert writer.last_dtype == np.uint8
    assert writer.last_shape == (16, 1280, 3)


def test_pixels_are_unchanged_for_a_clip_that_never_overflowed(
    stub_imageio, tmp_path
) -> None:
    """The wider grid must not alter any frame a uint16 grid could already build.

    Every use of the grids is reduced modulo 256 before the cast to uint8, so
    recomputing the same frames through uint16 grids has to agree exactly.
    """
    width, height, fps, seconds, seed = 320, 240, 8, 4, 7
    writer = stub_imageio(keep_frames=True)

    generate_videos._write_synthetic_video(
        tmp_path / "small.mp4",
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        seconds=seconds,
    )

    # Redraw the same random values in the same order the generator uses.
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=3, dtype=np.uint8)
    accent = rng.integers(0, 256, size=3, dtype=np.uint8)
    rect_w = max(4, width // 4)
    rect_h = max(4, height // 4)
    span_x = max(1, width - rect_w + 1)
    span_y = max(1, height - rect_h + 1)
    offset_x = int(rng.integers(0, span_x))
    offset_y = int(rng.integers(0, span_y))
    speed_x = int(rng.integers(1, max(2, width // 8)))
    speed_y = int(rng.integers(1, max(2, height // 8)))
    # Deliberately uint16, the pre-fix dtype; this size never exceeded it.
    xx = np.arange(width, dtype=np.uint16)[None, :]
    yy = np.arange(height, dtype=np.uint16)[:, None]

    assert (fps * seconds - 1) * max(speed_x, speed_y) <= np.iinfo(
        np.uint16
    ).max, "this configuration must stay inside uint16, or it is not a control"

    for frame_idx in range(fps * seconds):
        expected = np.empty((height, width, 3), dtype=np.uint8)
        expected[:, :, 0] = ((xx + int(base[0]) + frame_idx * speed_x) % 256).astype(
            np.uint8
        )
        expected[:, :, 1] = ((yy + int(base[1]) + frame_idx * speed_y) % 256).astype(
            np.uint8
        )
        expected[:, :, 2] = (
            (xx // 2 + yy // 2 + int(base[2]) + frame_idx * 7) % 256
        ).astype(np.uint8)
        x = (offset_x + frame_idx * speed_x) % span_x
        y = (offset_y + frame_idx * speed_y) % span_y
        expected[y : y + rect_h, x : x + rect_w, :] = accent

        np.testing.assert_array_equal(writer.frames[frame_idx], expected)
