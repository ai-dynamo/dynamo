# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration test for NVDEC decode (needs a GPU + PyNvVideoCodec).

Generates short H.264/H.265 clips with imageio's full ffmpeg (a test artifact
only -- the in-tree ffmpeg is VP9-only) and decodes them through the real
``nvdec_decoder``, asserting the frame contract. Mirrors the hardware validation
done on gpu-ts. Skips cleanly where NVDEC or a full ffmpeg is unavailable, so it
is a no-op on CPU lanes and images without PyNvVideoCodec.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dynamo.common.multimodal import nvdec_decoder as nd

pytestmark = [
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
]


def _synth_frames(n: int = 24, h: int = 256, w: int = 256) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, (i * 5) % w :, 0] = 220
        f[(i * 4) % h :, :, 1] = 140
        f[:, :, 2] = (i * 8) % 256
        frames.append(f)
    return frames


def _encode_sample(path: str, codec: str) -> None:
    """Write an H.264/H.265 clip using imageio's bundled full ffmpeg.

    The image points IMAGEIO_FFMPEG_EXE at the in-tree VP9-only ffmpeg, so unset
    it to reach imageio-ffmpeg's full build (has libx264/libx265). Skips the test
    if that build or its encoders are unavailable.
    """
    imageio = pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    saved = os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
    try:
        imageio.mimwrite(path, _synth_frames(), format="FFMPEG", codec=codec, fps=10)
    except Exception as exc:  # noqa: BLE001 - sample generator, not the code under test
        pytest.skip(f"could not generate a {codec} sample clip: {exc}")
    finally:
        if saved is not None:
            os.environ["IMAGEIO_FFMPEG_EXE"] = saved


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_nvdec_decodes_real_clip(tmp_path, codec):
    if not nd.nvdec_available():
        pytest.skip("PyNvVideoCodec/NVDEC not available (needs the video capability)")

    path = str(tmp_path / f"{codec}.mp4")
    _encode_sample(path, codec)
    with open(path, "rb") as fh:
        data = fh.read()

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
