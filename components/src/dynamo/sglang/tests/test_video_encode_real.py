# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real (un-mocked) video-encode regression test for the SGLang output path.

Drives VideoGenerationWorkerHandler._frames_to_video through the ACTUAL ffmpeg
baked into the shipped runtime image. This is the pre_merge guard for the gap
that let SGLang ship with no ffmpeg at all (video generation would fail at the
encode step) go green through the PR pipeline. Runs on the CUDA image only
(gpu_0, no xpu) because the in-tree VP9 ffmpeg is copied into the CUDA runtime;
encoding VP9 (libvpx-vp9) is CPU-only, so no GPU is used.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

try:
    from PIL import Image

    from dynamo.sglang.request_handlers.video_generation.video_generation_handler import (
        VideoGenerationWorkerHandler,
    )
except ImportError:
    pytest.skip(
        "SGLang video-generation dependencies not available", allow_module_level=True
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(120),
]


def _synthetic_pil_frames(n: int = 8, size: int = 64) -> list:
    return [
        Image.fromarray(np.full((size, size, 3), (i * 24) % 256, dtype=np.uint8))
        for i in range(n)
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


@pytest.mark.asyncio
async def test_sglang_video_output_is_vp9_in_shipped_image():
    # _frames_to_video uses no instance state, so bypass the engine-bound
    # constructor and call it directly.
    handler = VideoGenerationWorkerHandler.__new__(VideoGenerationWorkerHandler)
    video_bytes = await handler._frames_to_video(_synthetic_pil_frames(), 8)
    assert video_bytes, "encoder produced no bytes"
    codec = _probe_video_codec(video_bytes)
    assert codec == "vp9", f"expected vp9-encoded output, got codec={codec!r}"
