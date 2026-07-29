# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real (un-mocked) video-encode regression test for the vLLM-Omni output path.

Drives DiffusionFormatter._encode_video through the ACTUAL ffmpeg baked into the
shipped runtime image (no imageio/encode mock, no pip install). This is the
pre_merge guard for the gap that let two failures ship green:
  - the image shipping no VP9 encoder (encode raises -> status "failed"), and
  - the handler defaulting to a removed H.264 encoder (would also fail).
Runs on the CUDA image only (gpu_0, no xpu) because the in-tree ffmpeg is copied
into the CUDA runtime; encoding VP9 (libvpx-vp9) is CPU-only, so no GPU is used.
"""

import base64
import os
import subprocess
import tempfile

import numpy as np
import pytest

try:
    from PIL import Image

    from dynamo.vllm.omni.output_formatter import DiffusionFormatter
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
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
async def test_omni_video_output_is_vp9_in_shipped_image():
    formatter = DiffusionFormatter(
        model_name="test", media_fs=None, media_http_url=None
    )
    result = await formatter._encode_video(
        _synthetic_pil_frames(),
        "req-vp9",
        fps=8,
        response_format="b64_json",
    )
    # A missing ffmpeg or a regression to the H.264 default surfaces here as a
    # "failed" status rather than a raise (the handler catches encode errors).
    assert result["status"] == "completed", result.get("error")
    video_bytes = base64.b64decode(result["data"][0]["b64_json"])
    assert video_bytes, "encoder produced no bytes"
    codec = _probe_video_codec(video_bytes)
    assert codec == "vp9", f"expected vp9-encoded output, got codec={codec!r}"
