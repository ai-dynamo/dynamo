# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration test for NVDEC decode (needs a GPU + PyNvVideoCodec).

Decodes the committed H.264/H.265 fixtures through the real ``nvdec_decoder``
and asserts the frame contract, mirroring the hardware validation done on
gpu-ts. Skips cleanly where NVDEC is unavailable, so it is a no-op on CPU lanes
and images without PyNvVideoCodec.

The clips are committed fixtures rather than synthesized at run time on purpose:
generating them needs an H.264/H.265 *encoder*, which the codec-compliant image
deliberately does not ship (the in-tree ffmpeg is VP9-only, and imageio-ffmpeg
is installed with --no-binary). Synthesizing here made the test unable to run in
the very image it is meant to validate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynamo.common.multimodal import nvdec_decoder as nd

pytestmark = [
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
]


_MEDIA_DIR = (
    Path(__file__).resolve().parents[6] / "lib" / "llm" / "tests" / "data" / "media"
)

# Committed fixtures, re-encoded from the VP9 240p_10.mp4 clip.
_FIXTURES = {"h264": "240p_10_h264.mp4", "hevc": "240p_10_h265.mp4"}


@pytest.mark.parametrize("codec", sorted(_FIXTURES))
def test_nvdec_decodes_real_clip(codec):
    if not nd.nvdec_available():
        pytest.skip("PyNvVideoCodec/NVDEC not available (needs the video capability)")

    path = _MEDIA_DIR / _FIXTURES[codec]
    if not path.is_file():
        pytest.skip(f"fixture not available: {path}")
    data = path.read_bytes()
    # Guard against an unresolved Git LFS pointer masquerading as the clip.
    if data[:7] == b"version":
        pytest.skip(f"fixture is an unresolved LFS pointer: {path}")

    # Sanity: the probe classifies the clip as an NVDEC-routed codec.
    assert nd.probe_video_codec(data) in nd.HW_ROUTED_CODECS

    frames, metadata = nd.decode_video_nvdec(data, num_frames=8)

    assert frames.ndim == 4 and frames.shape[-1] == 3  # (T, H, W, 3)
    assert frames.dtype == np.uint8
    assert frames.flags["C_CONTIGUOUS"]
    assert frames.shape[0] == 8
    assert frames[:, :, :, :].max() > 0  # real pixels, not a black clip
    assert metadata["total_num_frames"] >= 8
    assert len(metadata["frames_indices"]) == 8
