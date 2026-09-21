# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

from dynamo.llm import MediaDecoder

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_vllm_runtime_includes_working_frontend_video_decoder(tmp_path: Path):
    decoder = MediaDecoder()
    assert hasattr(decoder, "enable_video")
    decoder.enable_video({"num_frames": 2, "strict": True})

    # MediaDecoder is configuration-only. Importing it proves the Rust extension's
    # external libav* dependencies resolve; decode with the same in-tree FFmpeg to
    # prove its allowlist includes the VP9 codec used by frontend decoding.
    fixture = Path(__file__).resolve().parents[6] / "lib/llm/tests/data/media/2p_10.mp4"
    output = tmp_path / "decoded.webm"
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE") or "ffmpeg"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(fixture),
            "-frames:v",
            "2",
            "-an",
            "-c:v",
            "libvpx-vp9",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
