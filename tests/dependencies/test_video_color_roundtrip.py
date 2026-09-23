# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB -> VP9 -> RGB through the shipped ffmpeg CLI and the shared encode helper.

Regression test for ai-dynamo/dynamo#15198: the in-tree FFmpeg 8.1.2, built with
--disable-x86asm, turned red quadrants magenta on x86 in both routes and both
containers. It reads the image itself and imports no framework module, so it
carries every framework marker to run in each backend's lane.
"""

import subprocess
from pathlib import Path

import pytest

from tests.utils.video_color import (
    assert_quadrant_colors,
    quadrant_frames,
    shipped_ffmpeg,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.vllm,
    pytest.mark.sglang,
    pytest.mark.trtllm,
    pytest.mark.framework_agnostic,
    pytest.mark.timeout(120),
]

_FPS = 8
_CONTAINERS = ("mp4", "webm")


@pytest.mark.parametrize("container", _CONTAINERS)
def test_cli_vp9_encode_preserves_rgb(container: str, tmp_path: Path) -> None:
    frames = quadrant_frames()
    _, height, width, _ = frames.shape
    output = tmp_path / f"cli.{container}"
    proc = subprocess.run(
        [
            shipped_ffmpeg(),
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(_FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        input=frames.tobytes(),
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert_quadrant_colors(output.read_bytes(), tmp_path)


@pytest.mark.parametrize("container", _CONTAINERS)
def test_shared_helper_vp9_encode_preserves_rgb(container: str, tmp_path: Path) -> None:
    # Imported here so collecting this module in an image without dynamo is harmless.
    from dynamo.common.utils.video_utils import encode_to_video_bytes

    video = encode_to_video_bytes(quadrant_frames(), fps=_FPS, output_format=container)
    assert_quadrant_colors(video, tmp_path)
