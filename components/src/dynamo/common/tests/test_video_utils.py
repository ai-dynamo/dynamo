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

    def test_mp4_selects_libx264_codec(self):
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
            assert kwargs.get("codec") == "libx264"
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
