# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.video_utils module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def make_frames(n=3, h=8, w=8) -> np.ndarray:
    """Return a small uint8 frame array (n, h, w, 3)."""
    return np.zeros((n, h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# encode_to_video_bytes
# ---------------------------------------------------------------------------


class TestEncodeToVideoBytes:
    """Tests for encode_to_video_bytes().

    encode_to_video_bytes pre-converts RGB->YUV420p in numpy and shells out to
    ffmpeg (feeding planar YUV on stdin) to sidestep the in-tree LGPL ffmpeg's
    broken libswscale RGB->YUV path. These tests mock subprocess.run + the temp
    file so no real ffmpeg is invoked.
    """

    def _patch_ffmpeg(self, read_bytes=b"video-bytes"):
        """Patch subprocess.run (success) and the output tempfile.

        Returns (run_patch, tempfile_patch); the run_patch's mock is what tests
        assert against.
        """
        run_patch = patch("subprocess.run", MagicMock())
        tmp = MagicMock()
        tmp.read.return_value = read_bytes
        ntf_cm = MagicMock()
        ntf_cm.__enter__.return_value = tmp
        tempfile_patch = patch(
            "tempfile.NamedTemporaryFile", MagicMock(return_value=ntf_cm)
        )
        return run_patch, tempfile_patch

    def test_mp4_uses_h264_nvenc(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        run_patch, tempfile_patch = self._patch_ffmpeg()
        with run_patch as mock_run, tempfile_patch:
            encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

            cmd = mock_run.call_args[0][0]
            assert "h264_nvenc" in cmd
            assert mock_run.call_args[1]["check"] is True

    def test_webm_uses_libvpx_vp9(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        run_patch, tempfile_patch = self._patch_ffmpeg()
        with run_patch as mock_run, tempfile_patch:
            encode_to_video_bytes(make_frames(), fps=16, output_format="webm")

            assert "libvpx-vp9" in mock_run.call_args[0][0]

    def test_unsupported_format_raises_value_error(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        with pytest.raises(ValueError, match="No codec"):
            encode_to_video_bytes(make_frames(), output_format="avi")

    def test_bad_shape_raises_value_error(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        with pytest.raises(ValueError, match="Expected frames of shape"):
            encode_to_video_bytes(
                np.zeros((3, 8, 8), dtype=np.uint8), output_format="mp4"
            )

    def test_subprocess_failure_raises_runtime_error(self):
        import subprocess

        from dynamo.common.utils.video_utils import encode_to_video_bytes

        err = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"boom")
        _, tempfile_patch = self._patch_ffmpeg()
        with patch("subprocess.run", MagicMock(side_effect=err)), tempfile_patch:
            with pytest.raises(RuntimeError, match="Video encoding to bytes failed"):
                encode_to_video_bytes(make_frames(), output_format="mp4")

    def test_returns_file_bytes(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        run_patch, tempfile_patch = self._patch_ffmpeg(read_bytes=b"\x00\x01\x02")
        with run_patch, tempfile_patch:
            result = encode_to_video_bytes(make_frames(), output_format="mp4")

        assert result == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# normalize_image_frames
# ---------------------------------------------------------------------------


class TestNormalizeImageFrames:
    """Tests for normalize_image_frames() — flattens DiffusionFormatter image
    inputs to PIL. Image pipelines usually emit PIL Images; the Cosmos3 native
    pipeline emits 5D numpy ``[B, F, H, W, C]``."""

    def test_pil_inputs_returned_by_identity(self):
        """PIL inputs must pass through without conversion or copy."""
        from PIL import Image

        from dynamo.common.utils.video_utils import normalize_image_frames

        a = Image.new("RGB", (4, 4), (255, 0, 0))
        b = Image.new("RGB", (4, 4), (0, 255, 0))
        out = normalize_image_frames([a, b])

        assert len(out) == 2
        assert out[0] is a and out[1] is b

    def test_uint8_hwc_numpy_preserves_pixels(self):
        from PIL import Image

        from dynamo.common.utils.video_utils import normalize_image_frames

        arr = np.full((4, 4, 3), 7, dtype=np.uint8)
        out = normalize_image_frames([arr])

        assert len(out) == 1
        assert isinstance(out[0], Image.Image)
        assert out[0].size == (4, 4)  # PIL is (W, H)
        assert np.asarray(out[0])[0, 0].tolist() == [7, 7, 7]

    def test_cosmos3_5d_strips_batch_and_preserves_frame_order(self):
        """[B, F, H, W, C] collapses to F PIL frames in order. Distinct
        per-frame content guards against wrong-axis indexing regressions."""
        from dynamo.common.utils.video_utils import normalize_image_frames

        arr = np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
        arr[0, 0] = 10  # frame 0 fill
        arr[0, 1] = 20  # frame 1 fill
        arr[0, 2] = 30  # frame 2 fill

        out = normalize_image_frames([arr])

        assert len(out) == 3
        assert np.asarray(out[0])[0, 0, 0] == 10
        assert np.asarray(out[1])[0, 0, 0] == 20
        assert np.asarray(out[2])[0, 0, 0] == 30

    def test_float_zero_to_one_scaled_to_uint8(self):
        """float32 [0, 1] inputs must be rescaled to uint8 [0, 255]."""
        from dynamo.common.utils.video_utils import normalize_image_frames

        arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = normalize_image_frames([arr])

        # 0.5 * 255 = 127.5; numpy's banker's rounding yields exactly 128.
        assert np.asarray(out[0])[0, 0, 0] == 128
