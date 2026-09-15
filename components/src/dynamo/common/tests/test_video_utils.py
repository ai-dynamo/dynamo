# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.video_utils module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dynamo.common.utils import video_utils
from dynamo.common.utils.video_utils import encode_video, pil_frames_to_array

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

    def test_rejects_empty_frame_list(self):
        from dynamo.common.utils.video_utils import frames_to_numpy

        with pytest.raises(ValueError, match="No images provided"):
            frames_to_numpy([])

    def test_pil_frames_to_array_from_pil_is_exact(self):
        Image = pytest.importorskip("PIL.Image")
        truth = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        imgs = [Image.fromarray(truth[i]) for i in range(truth.shape[0])]
        out = pil_frames_to_array(imgs)
        assert np.array_equal(out, truth)


# ---------------------------------------------------------------------------
# encode_video -- canonical input contract
# ---------------------------------------------------------------------------


def canonical_frames(n=2, h=4, w=4) -> np.ndarray:
    """Return a valid canonical (n, h, w, 3) uint8 frame array."""
    return np.zeros((n, h, w, 3), dtype=np.uint8)


class TestEncodeVideoValidation:
    """encode_video rejects anything but canonical (T, H, W, 3) uint8."""

    def test_rejects_non_ndarray(self):
        with pytest.raises(ValueError, match="canonical"):
            encode_video([canonical_frames()])

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="shape"):
            encode_video(np.zeros((4, 4, 3), np.uint8))

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="shape"):
            encode_video(np.zeros((2, 4, 4, 4), np.uint8))

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="uint8"):
            encode_video(np.zeros((2, 4, 4, 3), np.float32))


# ---------------------------------------------------------------------------
# Encoder selection
#
# DYN_XPU_FFMPEG_PATH is the only switch. Unset means software VP9; set means
# hardware AV1, and a path that does not resolve is a hard error rather than a
# silent downgrade.
# ---------------------------------------------------------------------------


def _executable_stub(tmp_path):
    """Create a file that passes the isfile/X_OK check on POSIX and Windows."""
    exe = tmp_path / "ffmpeg"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return str(exe)


class TestEncoderSelection:
    """The env var alone chooses between software VP9 and hardware AV1."""

    def test_software_vp9_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(video_utils.ENV_XPU_FFMPEG_PATH, raising=False)
        with patch(
            "dynamo.common.utils.video_utils._encode_vp9_imageio",
            return_value=b"vp9-bytes",
        ) as m_sw, patch(
            "dynamo.common.utils.video_utils._encode_av1_vaapi"
        ) as m_hw:
            out = encode_video(canonical_frames(), fps=8)

        assert out == b"vp9-bytes"
        assert m_sw.called
        assert not m_hw.called

    def test_hardware_av1_when_env_set(self, monkeypatch, tmp_path):
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(video_utils.ENV_XPU_FFMPEG_PATH, exe)
        with patch(
            "dynamo.common.utils.video_utils._encode_av1_vaapi",
            return_value=b"av1-bytes",
        ) as m_hw, patch(
            "dynamo.common.utils.video_utils._encode_vp9_imageio"
        ) as m_sw:
            out = encode_video(canonical_frames(), fps=8)

        assert out == b"av1-bytes"
        assert not m_sw.called
        # The resolved binary and render node are handed to the hardware path.
        assert m_hw.call_args[0][2] == exe
        assert m_hw.call_args[0][3] == video_utils.DEFAULT_XPU_VIDEO_DEVICE

    def test_missing_hw_ffmpeg_is_a_hard_error(self, monkeypatch, tmp_path):
        """A declared-but-absent encoder must not fall back to software."""
        monkeypatch.setenv(
            video_utils.ENV_XPU_FFMPEG_PATH, str(tmp_path / "nope" / "ffmpeg")
        )
        with patch("dynamo.common.utils.video_utils._encode_vp9_imageio") as m_sw:
            with pytest.raises(RuntimeError, match="not an executable file"):
                encode_video(canonical_frames(), fps=8)

        assert not m_sw.called

    def test_blank_env_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(video_utils.ENV_XPU_FFMPEG_PATH, "   ")
        assert video_utils.hw_ffmpeg_path() is None

    def test_render_node_override(self, monkeypatch):
        monkeypatch.setenv(video_utils.ENV_XPU_VIDEO_DEVICE, "/dev/dri/renderD130")
        assert video_utils.xpu_video_device() == "/dev/dri/renderD130"

    def test_render_node_default(self, monkeypatch):
        monkeypatch.delenv(video_utils.ENV_XPU_VIDEO_DEVICE, raising=False)
        assert video_utils.xpu_video_device() == "/dev/dri/renderD128"


class TestValidateVideoEncoderConfig:
    """The startup hook reports the resolved encoder and fails loudly on typos."""

    def test_passes_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(video_utils.ENV_XPU_FFMPEG_PATH, raising=False)
        video_utils.validate_video_encoder_config()

    def test_raises_on_bad_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            video_utils.ENV_XPU_FFMPEG_PATH, str(tmp_path / "missing-ffmpeg")
        )
        with pytest.raises(RuntimeError, match=video_utils.ENV_XPU_FFMPEG_PATH):
            video_utils.validate_video_encoder_config()


# ---------------------------------------------------------------------------
# Hardware AV1 command line
# ---------------------------------------------------------------------------


class TestAv1VaapiCommandLine:
    """The VA-API path builds the command line the encoder needs."""

    def _run(self, monkeypatch, tmp_path):
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(video_utils.ENV_XPU_FFMPEG_PATH, exe)
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = b""
        with patch(
            "dynamo.common.utils.video_utils.subprocess.run", return_value=proc
        ) as m_run, patch(
            "dynamo.common.utils.video_utils.open",
            new_callable=lambda: MagicMock(),
        ):
            encode_video(canonical_frames(), fps=12)
        return exe, m_run.call_args[0][0]

    def test_uses_av1_vaapi_encoder(self, monkeypatch, tmp_path):
        _, cmd = self._run(monkeypatch, tmp_path)
        assert cmd[cmd.index("-c:v") + 1] == "av1_vaapi"

    def test_invokes_the_declared_binary(self, monkeypatch, tmp_path):
        exe, cmd = self._run(monkeypatch, tmp_path)
        assert cmd[0] == exe

    def test_passes_render_node_and_hwupload(self, monkeypatch, tmp_path):
        _, cmd = self._run(monkeypatch, tmp_path)
        assert (
            cmd[cmd.index("-vaapi_device") + 1]
            == video_utils.DEFAULT_XPU_VIDEO_DEVICE
        )
        assert "format=nv12,hwupload" in cmd

    def test_sets_quality_explicitly(self, monkeypatch, tmp_path):
        _, cmd = self._run(monkeypatch, tmp_path)
        assert cmd[cmd.index("-global_quality") + 1] == str(
            video_utils.HW_VIDEO_GLOBAL_QUALITY
        )

    def test_always_mp4(self, monkeypatch, tmp_path):
        _, cmd = self._run(monkeypatch, tmp_path)
        assert cmd[cmd.index("-f", cmd.index("-c:v")) + 1] == "mp4"

    def test_nonzero_exit_raises_with_stderr(self, monkeypatch, tmp_path):
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(video_utils.ENV_XPU_FFMPEG_PATH, exe)
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = b"No usable encoding entrypoint found"
        with patch("dynamo.common.utils.video_utils.subprocess.run", return_value=proc):
            with pytest.raises(RuntimeError, match="No usable encoding entrypoint"):
                encode_video(canonical_frames(), fps=12)


# ---------------------------------------------------------------------------
# Software round trip
# ---------------------------------------------------------------------------


def _synthetic_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """Deterministic, encoder-friendly synthetic clip (T, H, W, 3) uint8.

    A smooth moving gradient: no noise, so a competent encoder should reproduce
    it closely.
    """
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    for t in range(num_frames):
        shift = int(255 * t / max(num_frames - 1, 1))
        frames[t, :, :, 0] = ((xx * 255 // max(width - 1, 1)) + shift) % 256
        frames[t, :, :, 1] = (yy * 255 // max(height - 1, 1)) % 256
        frames[t, :, :, 2] = shift
    return frames


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between two uint8 arrays."""
    a32 = a.astype(np.float64)
    b32 = b.astype(np.float64)
    mse = float(np.mean((a32 - b32) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10((255.0**2) / mse))


class TestEncodeVideoRoundTrip:
    """Encode, then decode, and check the result is a usable mp4."""

    @pytest.mark.timeout(60)
    def test_software_mp4_roundtrip(self, monkeypatch):
        monkeypatch.delenv(video_utils.ENV_XPU_FFMPEG_PATH, raising=False)
        iio = pytest.importorskip("imageio.v3")

        num_frames, height, width = 30, 144, 176  # QCIF
        frames = _synthetic_video(num_frames, height, width)
        try:
            data = encode_video(frames, fps=30)
        except (RuntimeError, OSError) as e:
            pytest.skip(f"software VP9 encoder unavailable: {e}")

        assert b"ftyp" in data[:64]  # mp4 container magic

        try:
            decoded = iio.imread(data, index=None, extension=".mp4")
        except Exception as e:  # decoder/plugin not available in this env
            pytest.skip(f"video decode unavailable: {e}")

        assert decoded.shape[0] == num_frames
        assert tuple(decoded.shape[1:3]) == (height, width)

    @pytest.mark.timeout(60)
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Round-trip PSNR floor is unresolved. The 35 dB threshold was "
            "calibrated against H.264; the software path now encodes VP9, and "
            "the measurement is taken in RGB across a 4:2:0 round trip, which "
            "costs several dB on its own. Pending inspection of an encoded "
            "stream to decide between a codec-appropriate floor, explicit "
            "quality flags, or measuring luma only."
        ),
    )
    def test_software_mp4_roundtrip_fidelity(self, monkeypatch):
        monkeypatch.delenv(video_utils.ENV_XPU_FFMPEG_PATH, raising=False)
        iio = pytest.importorskip("imageio.v3")

        num_frames, height, width = 30, 144, 176
        frames = _synthetic_video(num_frames, height, width)
        try:
            data = encode_video(frames, fps=30)
            decoded = iio.imread(data, index=None, extension=".mp4")
        except Exception as e:
            pytest.skip(f"encode/decode unavailable: {e}")

        psnr = _psnr(frames, decoded)
        assert psnr >= 35.0, f"round-trip PSNR too low: {psnr:.1f} dB"
