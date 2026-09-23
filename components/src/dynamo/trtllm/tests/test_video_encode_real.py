# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real (un-mocked) video-encode regression test for the TRT-LLM output path.

Drives encode_to_video_bytes (the exact call TRT-LLM's VideoGenerationHandler
makes) through the ACTUAL ffmpeg baked into the shipped runtime image, and
asserts the produced stream is VP9 and keeps its colors -- the in-tree FFmpeg
8.1.2 turned red magenta (ai-dynamo/dynamo#15198). pre_merge guard for the
shipped-image codec gap. Encoding VP9 (libvpx-vp9) is CPU-only, so no GPU is
used (gpu_0).
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.utils.video_color import assert_quadrant_colors, quadrant_frames

try:
    from dynamo.common.utils.video_utils import encode_to_video_bytes
except ImportError:
    pytest.skip("video_utils not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(120),
]


def _probe_video_codec(video_bytes: bytes) -> str:
    """Return the video stream codec of encoded bytes, via the shipped ffmpeg."""
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = "ffmpeg"
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        stderr = subprocess.run(
            [exe, "-hide_banner", "-i", tmp.name],
            capture_output=True,
            text=True,
        ).stderr
    for line in stderr.splitlines():
        if "Video:" in line:
            return line.split("Video:", 1)[1].split(",")[0].split()[0]
    return "?"


def test_trtllm_video_output_is_vp9_in_shipped_image(tmp_path: Path):
    video_bytes = encode_to_video_bytes(quadrant_frames(), fps=8, output_format="mp4")
    assert video_bytes, "encoder produced no bytes"
    codec = _probe_video_codec(video_bytes)
    assert codec == "vp9", f"expected vp9-encoded output, got codec={codec!r}"
    assert_quadrant_colors(video_bytes, tmp_path)
