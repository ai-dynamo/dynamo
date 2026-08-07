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
    """Tests for encode_to_video_bytes()."""

    def _mock_iio_v3(self):
        """Return a mock that looks like imageio.v3 (has imwrite)."""
        iio = MagicMock()
        iio.imwrite = MagicMock()
        return iio

    def _mock_iio_v2(self):
        """Return a mock that looks like imageio v2 (no imwrite, has get_writer)."""
        iio = MagicMock(spec=[])  # no attributes by default
        writer = MagicMock()
        iio.get_writer = MagicMock(return_value=writer)
        return iio, writer

    def test_mp4_selects_vp9_codec(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            buf = MagicMock()
            buf.getvalue.return_value = b"fake-mp4"
            mock_io.BytesIO.return_value = buf

            encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

            iio.imwrite.assert_called_once()
            _, kwargs = iio.imwrite.call_args
            # Royalty-free: mp4 output uses VP9, not h264_nvenc.
            assert kwargs.get("codec") == "libvpx-vp9"
            assert kwargs.get("fps") == 8

    def test_webm_selects_libvpx_vp9_codec(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            buf = MagicMock()
            buf.getvalue.return_value = b"fake-webm"
            mock_io.BytesIO.return_value = buf

            encode_to_video_bytes(make_frames(), fps=16, output_format="webm")

            iio.imwrite.assert_called_once()
            _, kwargs = iio.imwrite.call_args
            assert kwargs.get("codec") == "libvpx-vp9"

    def test_mp4_passes_extension_to_imwrite(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            buf = MagicMock()
            buf.getvalue.return_value = b"bytes"
            mock_io.BytesIO.return_value = buf

            encode_to_video_bytes(make_frames(), output_format="mp4")

            _, kwargs = iio.imwrite.call_args
            assert kwargs.get("extension") == ".mp4"

    def test_webm_passes_extension_to_imwrite(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            buf = MagicMock()
            buf.getvalue.return_value = b"bytes"
            mock_io.BytesIO.return_value = buf

            encode_to_video_bytes(make_frames(), output_format="webm")

            _, kwargs = iio.imwrite.call_args
            assert kwargs.get("extension") == ".webm"

    def test_unsupported_format_raises_value_error(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            mock_io.BytesIO.return_value = MagicMock()

            # ValueError is wrapped into RuntimeError by the except block
            with pytest.raises(RuntimeError, match="Video encoding to bytes failed"):
                encode_to_video_bytes(make_frames(), output_format="avi")

    def test_returns_bytes_from_buffer(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        expected = b"\x00\x01\x02"
        iio = self._mock_iio_v3()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}):
            buf = MagicMock()
            buf.getvalue.return_value = expected
            mock_io.BytesIO.return_value = buf

            result = encode_to_video_bytes(make_frames(), output_format="mp4")

        assert result == expected

    def test_v2_api_fallback_writes_all_frames(self):
        """When imageio.v3.imwrite is absent, falls back to get_writer loop."""
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio_v2, writer = self._mock_iio_v2()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio_v2, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio_v2}):
            buf = MagicMock()
            buf.getvalue.return_value = b"v2-bytes"
            mock_io.BytesIO.return_value = buf

            frames = make_frames(n=4)
            encode_to_video_bytes(frames, output_format="mp4")

            assert writer.append_data.call_count == 4
            writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# frames_to_numpy
# ---------------------------------------------------------------------------


class TestFramesToNumpy:
    """Tests for frames_to_numpy() — must accept both PIL and numpy frames."""

    def test_numpy_float_frames_scale_to_uint8(self):
        """Regression: diffusion pipelines emit float [0, 1] numpy frames.

        The PIL-only implementation crashed with "'numpy.ndarray' object has no
        attribute 'convert'" on the Wan2.1 T2V serve path.
        """
        from dynamo.common.utils.video_utils import frames_to_numpy

        frames = [np.full((4, 4, 3), 0.5, dtype=np.float32) for _ in range(3)]
        out = frames_to_numpy(frames)

        assert out.shape == (3, 4, 4, 3)
        assert out.dtype == np.uint8
        # 0.5 * 255 -> 128 (rounded)
        assert np.all(out == 128)

    def test_numpy_uint8_frames_pass_through(self):
        from dynamo.common.utils.video_utils import frames_to_numpy

        frames = [np.full((4, 4, 3), 200, dtype=np.uint8) for _ in range(2)]
        out = frames_to_numpy(frames)

        assert out.shape == (2, 4, 4, 3)
        assert out.dtype == np.uint8
        assert np.all(out == 200)

    def test_pil_frames_still_supported(self):
        from PIL import Image

        from dynamo.common.utils.video_utils import frames_to_numpy

        frames = [
            Image.fromarray(np.full((4, 4, 3), 77, dtype=np.uint8)) for _ in range(2)
        ]
        out = frames_to_numpy(frames)

        assert out.shape == (2, 4, 4, 3)
        assert out.dtype == np.uint8
        assert np.all(out == 77)

    def test_empty_raises(self):
        from dynamo.common.utils.video_utils import frames_to_numpy

        with pytest.raises(ValueError, match="No images provided"):
            frames_to_numpy([])
