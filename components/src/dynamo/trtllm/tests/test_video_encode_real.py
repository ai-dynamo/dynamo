# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real (un-mocked) video-encode regression test for the TRT-LLM output path.

Drives encode_to_video_bytes (the exact call TRT-LLM's VideoGenerationHandler
makes) through the ACTUAL ffmpeg baked into the shipped runtime image, and
asserts the produced stream is VP9. pre_merge guard for the shipped-image codec
gap. Encoding VP9 (libvpx-vp9) is CPU-only, so no GPU is used (gpu_0).
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

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


def _synthetic_frames(n: int = 8, size: int = 64) -> np.ndarray:
    return np.stack(
        [np.full((size, size, 3), (i * 24) % 256, dtype=np.uint8) for i in range(n)]
    )


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


def test_trtllm_video_output_is_vp9_in_shipped_image():
    video_bytes = encode_to_video_bytes(_synthetic_frames(), fps=8, output_format="mp4")
    assert video_bytes, "encoder produced no bytes"
    codec = _probe_video_codec(video_bytes)
    assert codec == "vp9", f"expected vp9-encoded output, got codec={codec!r}"
