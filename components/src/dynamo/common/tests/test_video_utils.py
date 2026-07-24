# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.video_utils module."""

import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dynamo.common.utils.video_utils import (
    drop_alpha,
    encode_video,
    ensure_uint8_rgb,
    pil_frames_to_array,
)

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

    def test_mp4_selects_h264_nvenc_codec(self):
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
            assert kwargs.get("codec") == "h264_nvenc"
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
# Canonical-domain primitives
# ---------------------------------------------------------------------------


class TestCanonicalPrimitives:
    """Tests for the shared canonical-domain primitives."""

    def test_drop_alpha_removes_fourth_channel(self):
        arr = np.zeros((2, 4, 4, 4), dtype=np.uint8)
        out = drop_alpha(arr)
        assert out.shape == (2, 4, 4, 3)

    def test_drop_alpha_passes_rgb_through(self):
        arr = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        assert drop_alpha(arr) is arr

    def test_ensure_uint8_rgb_scales_float_0_1(self):
        arr = np.ones((1, 2, 2, 3), dtype=np.float32)  # 1.0 -> 255
        out = ensure_uint8_rgb(arr)
        assert out.dtype == np.uint8
        assert np.all(out == 255)

    def test_ensure_uint8_rgb_drops_alpha_and_is_contiguous(self):
        arr = np.zeros((1, 2, 2, 4), dtype=np.uint8)
        out = ensure_uint8_rgb(arr)
        assert out.shape == (1, 2, 2, 3)
        assert out.flags["C_CONTIGUOUS"]

    def test_ensure_uint8_rgb_uint8_roundtrip_is_exact(self):
        truth = np.arange(1 * 2 * 2 * 3, dtype=np.uint8).reshape(1, 2, 2, 3)
        assert np.array_equal(ensure_uint8_rgb(truth), truth)

    def test_pil_frames_to_array_from_numpy(self):
        frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
        out = pil_frames_to_array(frames)
        assert out.shape == (2, 4, 4, 3)

    def test_pil_frames_to_array_from_pil_is_exact(self):
        Image = pytest.importorskip("PIL.Image")
        truth = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        imgs = [Image.fromarray(truth[i]) for i in range(truth.shape[0])]
        out = pil_frames_to_array(imgs)
        assert np.array_equal(out, truth)


# ---------------------------------------------------------------------------
# encode_video — canonical ABI validation
# ---------------------------------------------------------------------------


def canonical_frames(n=2, h=4, w=4) -> np.ndarray:
    """Return a valid canonical (n, h, w, 3) uint8 frame array."""
    return np.zeros((n, h, w, 3), dtype=np.uint8)


class TestEncodeVideoValidation:
    """encode_video rejects anything but canonical (T, H, W, 3) uint8."""

    def test_rejects_non_ndarray(self):
        with pytest.raises(ValueError, match="canonical"):
            encode_video([canonical_frames()], hw_accel="nvenc")

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="shape"):
            encode_video(np.zeros((4, 4, 3), np.uint8), hw_accel="nvenc")

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="shape"):
            encode_video(np.zeros((2, 4, 4, 4), np.uint8), hw_accel="nvenc")

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="uint8"):
            encode_video(np.zeros((2, 4, 4, 3), np.float32), hw_accel="nvenc")


class TestEncodeVideoContainerCodec:
    """encode_video rejects incompatible container/codec combinations."""

    def test_rejects_unknown_container(self):
        with pytest.raises(ValueError, match="Unsupported container"):
            encode_video(canonical_frames(), hw_accel="nvenc", container="avi")

    def test_rejects_h264_in_webm(self):
        with pytest.raises(ValueError, match="not compatible"):
            encode_video(
                canonical_frames(), hw_accel="nvenc", container="webm", codec="h264"
            )

    def test_rejects_vp9_in_mp4(self):
        with pytest.raises(ValueError, match="not compatible"):
            encode_video(
                canonical_frames(), hw_accel="nvenc", container="mp4", codec="vp9"
            )

    def test_accepts_compatible_non_default_codec(self):
        with patch(
            "dynamo.common.utils.video_utils._encode_video_imageio",
            return_value=b"x",
        ) as m_imageio:
            encode_video(
                canonical_frames(), hw_accel="nvenc", container="mp4", codec="hevc"
            )
        m_imageio.assert_called_once()

    def test_rejects_unknown_hw_accel(self):
        with pytest.raises(ValueError, match="Unsupported hw_accel"):
            encode_video(canonical_frames(), hw_accel="foo")


# ---------------------------------------------------------------------------
# encode_video — dispatch and control resolution
# ---------------------------------------------------------------------------


class TestEncodeVideoDispatch:
    """encode_video resolves controls (arg > env > auto) and dispatches paths."""

    def test_nvenc_uses_imageio_path(self):
        with patch(
            "dynamo.common.utils.video_utils._encode_video_imageio",
            return_value=b"nvenc-bytes",
        ) as m_imageio, patch(
            "dynamo.common.utils.video_utils._encode_video_ffmpeg_cli"
        ) as m_cli:
            out = encode_video(canonical_frames(), fps=8, hw_accel="nvenc")

        assert out == b"nvenc-bytes"
        m_imageio.assert_called_once()
        m_cli.assert_not_called()

    def test_cpu_uses_ffmpeg_cli_path(self):
        with patch(
            "dynamo.common.utils.video_utils._encode_video_imageio"
        ) as m_imageio, patch(
            "dynamo.common.utils.video_utils._encode_video_ffmpeg_cli",
            return_value=b"cli-bytes",
        ) as m_cli:
            out = encode_video(canonical_frames(), fps=8, hw_accel="cpu")

        assert out == b"cli-bytes"
        m_cli.assert_called_once()
        m_imageio.assert_not_called()

    def test_auto_selects_nvenc_when_not_xpu(self, monkeypatch):
        monkeypatch.delenv("DYN_VIDEO_HW_ACCEL", raising=False)
        with patch(
            "dynamo.common.utils.video_utils._running_on_xpu", return_value=False
        ), patch(
            "dynamo.common.utils.video_utils._encode_video_imageio",
            return_value=b"x",
        ) as m_imageio, patch(
            "dynamo.common.utils.video_utils._encode_video_ffmpeg_cli"
        ) as m_cli:
            encode_video(canonical_frames(), fps=8, hw_accel="auto")

        m_imageio.assert_called_once()
        m_cli.assert_not_called()

    def test_auto_selects_xpu_when_xpu_available(self, monkeypatch):
        monkeypatch.delenv("DYN_VIDEO_HW_ACCEL", raising=False)
        with patch(
            "dynamo.common.utils.video_utils._running_on_xpu", return_value=True
        ), patch(
            "dynamo.common.utils.video_utils._encode_video_imageio"
        ) as m_imageio, patch(
            "dynamo.common.utils.video_utils._encode_video_ffmpeg_cli",
            return_value=b"x",
        ) as m_cli:
            encode_video(canonical_frames(), fps=8, hw_accel="auto")

        m_cli.assert_called_once()
        m_imageio.assert_not_called()

    def test_explicit_container_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DYN_VIDEO_CONTAINER", "mp4")
        with patch(
            "dynamo.common.utils.video_utils._encode_video_imageio",
            return_value=b"x",
        ) as m_imageio:
            encode_video(canonical_frames(), fps=8, hw_accel="nvenc", container="webm")

        # _encode_video_imageio(frames, fps, container, codec)
        assert m_imageio.call_args[0][2] == "webm"

    def test_env_container_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("DYN_VIDEO_CONTAINER", "webm")
        with patch(
            "dynamo.common.utils.video_utils._encode_video_imageio",
            return_value=b"x",
        ) as m_imageio:
            encode_video(canonical_frames(), fps=8, hw_accel="nvenc")

        assert m_imageio.call_args[0][2] == "webm"


# ---------------------------------------------------------------------------
# ffmpeg-CLI path (XPU / VA-API)
# ---------------------------------------------------------------------------


class TestFfmpegCliPath:
    """The ffmpeg-CLI path builds the expected VA-API command line."""

    def test_xpu_path_uses_vaapi_device_and_hwupload(self):
        """The XPU path must select a VA-API encoder, pass the device, and
        upload frames to a HW surface -- the reason this path exists at all
        (imageio cannot express these). Exact argv layout is deliberately not
        asserted; real bitstream correctness belongs to a hardware-gated
        (xpu_1) encode/decode roundtrip.
        """
        frames = canonical_frames(n=3, h=8, w=8)
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = b""
        with patch(
            "dynamo.common.utils.video_utils.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ), patch(
            "dynamo.common.utils.video_utils.subprocess.run", return_value=proc
        ) as m_run:
            encode_video(
                frames,
                fps=12,
                container="mp4",
                codec="h264",
                hw_accel="xpu",
                device="/dev/dri/renderD129",
            )

        cmd = m_run.call_args[0][0]
        assert cmd[cmd.index("-vaapi_device") + 1] == "/dev/dri/renderD129"
        assert "format=nv12,hwupload" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "h264_vaapi"

    def test_missing_ffmpeg_raises(self):
        with patch("dynamo.common.utils.video_utils.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                encode_video(canonical_frames(), fps=8, hw_accel="xpu")

    def test_cpu_path_forces_yuv420p(self):
        """Software encoding must emit 4:2:0 so the output is broadly playable."""
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = b""
        with patch(
            "dynamo.common.utils.video_utils.shutil.which",
            return_value="/usr/bin/ffmpeg",
        ), patch(
            "dynamo.common.utils.video_utils.subprocess.run", return_value=proc
        ) as m_run:
            encode_video(canonical_frames(), fps=12, container="mp4", hw_accel="cpu")

        cmd = m_run.call_args[0][0]
        # The output pix_fmt (after -c:v), not the rawvideo input pix_fmt.
        out_pix_fmt = cmd.index("-pix_fmt", cmd.index("-c:v"))
        assert cmd[out_pix_fmt + 1] == "yuv420p"


# ---------------------------------------------------------------------------
# Real encode -> decode round-trip (the one bitstream-touching test)
# ---------------------------------------------------------------------------


def _synthetic_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """Deterministic, encoder-friendly synthetic clip (T, H, W, 3) uint8.

    A band-limited moving sinusoid ("plasma"): spatially low-frequency and
    temporally coherent, so a lossy DCT / motion-compensated codec can
    represent it well. Random pixels are the worst case for such a codec
    (unrepresentable high-frequency energy, defeats motion estimation) and
    would give a misleadingly low PSNR.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for t in range(num_frames):
        phase = t * 0.3  # animate -> real inter-frame motion
        r = 128 + 127 * np.sin(2 * np.pi * (xx / width) * 3 + phase)
        g = 128 + 127 * np.sin(2 * np.pi * (yy / height) * 3 + phase * 1.3)
        b = 128 + 127 * np.sin(
            2 * np.pi * ((xx + yy) / (width + height)) * 3 + phase * 0.7
        )
        frames[t] = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    return frames


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) between two uint8 clips of equal shape."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


class TestEncodeVideoRoundTrip:
    """Encode a realistic synthetic clip, decode it, and check fidelity."""

    @pytest.mark.timeout(30)
    def test_cpu_mp4_roundtrip(self):
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg CLI not available")
        iio = pytest.importorskip("imageio.v3")

        num_frames, height, width = 30, 144, 176  # QCIF
        frames = _synthetic_video(num_frames, height, width)
        try:
            data = encode_video(frames, fps=30, container="mp4", hw_accel="cpu")
        except RuntimeError as e:
            pytest.skip(f"CPU H.264 encoder unavailable: {e}")

        assert b"ftyp" in data[:64]  # mp4 container magic

        try:
            decoded = iio.imread(data, index=None, extension=".mp4")
        except Exception as e:  # decoder/plugin not available in this env
            pytest.skip(f"video decode unavailable: {e}")

        assert decoded.shape[0] == num_frames
        assert tuple(decoded.shape[1:3]) == (height, width)

        # Fidelity: a smooth clip through H.264 should reconstruct well.
        # 35 dB is a deliberately lenient floor (real content typically >40 dB).
        psnr = _psnr(frames, decoded)
        assert psnr >= 35.0, f"round-trip PSNR too low: {psnr:.1f} dB"
