# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for synthetic video index-grid overflow under NumPy 2."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from benchmarks.multimodal.jsonl import generate_videos

# Leave tests unmarked so bare `pytest benchmarks/...` passes under
# benchmarks/pyproject.toml (--strict-markers without custom marker entries),
# while repo-root collection applies pre_merge/gpu_0/defaulted via root conftest.py.


@dataclass(frozen=True)
class _VideoParams:
    """Deterministic parameters drawn by ``_write_synthetic_video`` for a seed."""

    base: np.ndarray
    accent: np.ndarray
    rect_w: int
    rect_h: int
    span_x: int
    span_y: int
    offset_x: int
    offset_y: int
    speed_x: int
    speed_y: int


def _draw_video_params(seed: int, width: int, height: int) -> _VideoParams:
    """Draw the same RNG sequence used by ``_write_synthetic_video``."""
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
    return _VideoParams(
        base=base,
        accent=accent,
        rect_w=rect_w,
        rect_h=rect_h,
        span_x=span_x,
        span_y=span_y,
        offset_x=offset_x,
        offset_y=offset_y,
        speed_x=speed_x,
        speed_y=speed_y,
    )


def _render_reference_frame(
    params: _VideoParams,
    width: int,
    height: int,
    frame_idx: int,
    grid_dtype: type[np.integer],
) -> np.ndarray:
    """Render one reference frame using index grids of ``grid_dtype``."""
    xx = np.arange(width, dtype=grid_dtype)[None, :]
    yy = np.arange(height, dtype=grid_dtype)[:, None]
    expected = np.empty((height, width, 3), dtype=np.uint8)
    expected[:, :, 0] = (
        (xx + int(params.base[0]) + frame_idx * params.speed_x) % 256
    ).astype(np.uint8)
    expected[:, :, 1] = (
        (yy + int(params.base[1]) + frame_idx * params.speed_y) % 256
    ).astype(np.uint8)
    expected[:, :, 2] = (
        (xx // 2 + yy // 2 + int(params.base[2]) + frame_idx * 7) % 256
    ).astype(np.uint8)
    x = (params.offset_x + frame_idx * params.speed_x) % params.span_x
    y = (params.offset_y + frame_idx * params.speed_y) % params.span_y
    expected[y : y + params.rect_h, x : x + params.rect_w, :] = params.accent
    return expected


class _CollectingWriter:
    """In-memory stand-in for ``imageio.v2`` writers so tests need no ffmpeg."""

    def __init__(self, keep_frames: bool) -> None:
        """Initialize frame counters and optional frame storage."""
        self.keep_frames = keep_frames
        self.frame_count = 0
        self.frames: list[np.ndarray] = []
        self.last_frame: np.ndarray | None = None

    def __enter__(self) -> _CollectingWriter:
        """Return self for context-manager protocol."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """No-op context-manager exit."""
        return None

    def append_data(self, frame: np.ndarray) -> None:
        """Record frame count, latest frame, and optionally every frame."""
        self.frame_count += 1
        copied = frame.copy()
        self.last_frame = copied
        if self.keep_frames:
            self.frames.append(copied)


@pytest.fixture
def stub_imageio(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``imageio.v2`` module and return a factory for its writer."""

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


@pytest.mark.parametrize(
    ("width", "height", "overflow_axis"),
    [
        (1280, 16, "x"),
        (16, 1280, "y"),
    ],
    ids=["x-grid-overflow", "y-grid-overflow"],
)
def test_index_grid_does_not_overflow_uint16(
    stub_imageio,
    tmp_path: Path,
    width: int,
    height: int,
    overflow_axis: str,
) -> None:
    """Crossing the uint16 limit on either ``xx`` or ``yy`` must not raise."""
    seed, fps, seconds = 4, 24, 19
    frame_count = fps * seconds
    params = _draw_video_params(seed, width, height)
    uint16_max = np.iinfo(np.uint16).max

    if overflow_axis == "x":
        assert (frame_count - 1) * params.speed_x > uint16_max
        assert (frame_count - 1) * params.speed_y <= uint16_max
    else:
        assert (frame_count - 1) * params.speed_y > uint16_max
        assert (frame_count - 1) * params.speed_x <= uint16_max

    writer = stub_imageio()
    generate_videos._write_synthetic_video(
        tmp_path / f"clip_{overflow_axis}.mp4",
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        seconds=seconds,
    )

    assert writer.frame_count == frame_count
    assert writer.last_frame is not None
    assert writer.last_frame.dtype == np.uint8
    assert writer.last_frame.shape == (height, width, 3)
    expected_last = _render_reference_frame(
        params,
        width=width,
        height=height,
        frame_idx=frame_count - 1,
        grid_dtype=np.int64,
    )
    np.testing.assert_array_equal(writer.last_frame, expected_last)


def test_pixels_match_uint16_reference_within_bounds(
    stub_imageio, tmp_path: Path
) -> None:
    """Widening the index grids to int64 must produce identical uint8 pixels."""
    width, height, fps, seconds, seed = 320, 240, 8, 4, 7
    frame_count = fps * seconds
    params = _draw_video_params(seed, width, height)

    assert (frame_count - 1) * max(params.speed_x, params.speed_y) <= np.iinfo(
        np.uint16
    ).max, "control clip must stay within uint16 bounds"

    writer = stub_imageio(keep_frames=True)
    generate_videos._write_synthetic_video(
        tmp_path / "small.mp4",
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        seconds=seconds,
    )

    assert writer.frame_count == frame_count
    for frame_idx in range(frame_count):
        expected = _render_reference_frame(
            params,
            width=width,
            height=height,
            frame_idx=frame_idx,
            grid_dtype=np.uint16,
        )
        np.testing.assert_array_equal(writer.frames[frame_idx], expected)
