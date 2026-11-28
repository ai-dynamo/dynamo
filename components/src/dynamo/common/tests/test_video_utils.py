# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.video_utils module."""

import subprocess
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
# compute_num_frames
# ---------------------------------------------------------------------------


def test_compute_num_frames_rounds_fractional_frame_rate():
    from dynamo.common.utils.video_utils import compute_num_frames

    result = compute_num_frames(seconds=10, fps=23.976)

    assert result == 240
    assert isinstance(result, int)


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


# ---------------------------------------------------------------------------
# Hardware AV1 (DYN_XPU_FFMPEG_PATH)
# ---------------------------------------------------------------------------


def _recording_ffmpeg(tmp_path, tail: str = "printf 'fake-av1' > \"$last\"") -> str:
    """A stand-in ffmpeg that records its argv and how many bytes it was fed."""
    path = tmp_path / "ffmpeg"
    path.write_text(
        "#!/bin/sh\n"
        f'printf \'%s\\n\' "$@" > "{tmp_path}/argv"\n'
        "for last; do :; done\n"
        f'wc -c > "{tmp_path}/stdin_bytes"\n'
        f"{tail}\n"
    )
    path.chmod(0o755)
    return str(path)


class TestEncodeToVideoBytesHardware:
    """With DYN_XPU_FFMPEG_PATH set, the clip goes to that ffmpeg, never to imageio."""

    @pytest.fixture(autouse=True)
    def _no_imageio(self):
        """A software encode would have to import imageio; make that impossible."""
        with patch.dict("sys.modules", {"imageio": None, "imageio.v3": None}):
            yield

    def test_pipes_raw_rgb_to_the_declared_ffmpeg(self, monkeypatch, tmp_path):
        from dynamo.common.utils.ffmpeg_command import (
            ENV_XPU_FFMPEG_PATH,
            ENV_XPU_VIDEO_DEVICE,
        )
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, _recording_ffmpeg(tmp_path))
        monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, "/dev/dri/renderD129")
        frames = make_frames(n=5, h=96, w=128)

        out = encode_to_video_bytes(frames, fps=24, output_format="mp4")

        assert out == b"fake-av1"
        argv = (tmp_path / "argv").read_text().splitlines()
        pairs = set(zip(argv, argv[1:]))
        assert ("-vaapi_device", "/dev/dri/renderD129") in pairs
        assert ("-c:v", "av1_vaapi") in pairs
        assert ("-vf", "format=nv12,hwupload") in pairs
        assert ("-s", "128x96") in pairs and ("-r", "24") in pairs
        assert ("-movflags", "+faststart") in pairs
        assert argv[-1].endswith(".mp4")
        assert int((tmp_path / "stdin_bytes").read_text()) == frames.nbytes

    def test_a_failed_encode_raises_with_stderr_and_does_not_fall_back(
        self, monkeypatch, tmp_path
    ):
        from dynamo.common.utils.ffmpeg_command import ENV_XPU_FFMPEG_PATH
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        exe = _recording_ffmpeg(tmp_path, 'echo "boom: no VA display" >&2\nexit 3')
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, exe)

        with pytest.raises(RuntimeError, match=r"exit 3\).*boom: no VA display"):
            encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

    def test_an_encode_that_writes_nothing_is_an_error(self, monkeypatch, tmp_path):
        from dynamo.common.utils.ffmpeg_command import ENV_XPU_FFMPEG_PATH
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, _recording_ffmpeg(tmp_path, "exit 0"))

        with pytest.raises(RuntimeError, match="produced no output"):
            encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

    def test_an_unusable_path_is_an_error_not_a_software_encode(
        self, monkeypatch, tmp_path
    ):
        from dynamo.common.utils.ffmpeg_command import ENV_XPU_FFMPEG_PATH
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, str(tmp_path / "nope"))

        with pytest.raises(RuntimeError, match=ENV_XPU_FFMPEG_PATH):
            encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

    def test_only_mp4_is_supported(self, monkeypatch, tmp_path):
        from dynamo.common.utils.ffmpeg_command import ENV_XPU_FFMPEG_PATH
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, _recording_ffmpeg(tmp_path))

        with pytest.raises(ValueError, match="supports only 'mp4'"):
            encode_to_video_bytes(make_frames(), fps=8, output_format="webm")
        assert not (tmp_path / "argv").exists(), "ffmpeg must not have been started"

    def test_frames_that_are_not_uint8_rgb_are_refused_before_encoding(
        self, monkeypatch, tmp_path
    ):
        """Raw bytes of the wrong type would encode as garbage, not fail."""
        from dynamo.common.utils.ffmpeg_command import ENV_XPU_FFMPEG_PATH
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, _recording_ffmpeg(tmp_path))

        with pytest.raises(ValueError, match="uint8"):
            encode_to_video_bytes(
                make_frames().astype(np.float32), fps=8, output_format="mp4"
            )
        assert not (tmp_path / "argv").exists(), "ffmpeg must not have been started"


class TestEncodeToVideoBytesSoftwareDefault:
    """With nothing declared, the hardware code is not reached."""

    def test_no_ffmpeg_process_is_started(self):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        iio = MagicMock()
        with patch("dynamo.common.utils.video_utils.io") as mock_io, patch(
            "imageio.v3", iio, create=True
        ), patch.dict("sys.modules", {"imageio.v3": iio}), patch(
            "dynamo.common.utils.video_utils.subprocess.run",
            side_effect=AssertionError("the ffmpeg CLI must not run"),
        ):
            mock_io.BytesIO.return_value.getvalue.return_value = b"fake-vp9"

            out = encode_to_video_bytes(make_frames(), fps=8, output_format="mp4")

        assert out == b"fake-vp9"
        iio.imwrite.assert_called_once()
        assert iio.imwrite.call_args.kwargs["codec"] == "libvpx-vp9"


def _top_level_boxes(data: bytes) -> list:
    """Types of the top-level MP4 boxes, in file order."""
    boxes, pos = [], 0
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        boxes.append(data[pos + 4 : pos + 8].decode("latin-1"))
        if size < 8:
            break
        pos += size
    return boxes


@pytest.mark.xpu_1
@pytest.mark.timeout(180)
class TestEncodeToVideoBytesHardwareReal:
    """A real VA-API encode. Runs only where DYN_XPU_FFMPEG_PATH is set."""

    def test_clip_is_av1_in_a_seekable_mp4_with_every_frame(
        self, hw_video_encoder, tmp_path
    ):
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        ffmpeg, _ = hw_video_encoder
        height, width = 96, 160
        yy, xx = np.mgrid[0:height, 0:width]
        frames = np.empty((17, height, width, 3), dtype=np.uint8)
        for t in range(len(frames)):
            frames[t, ..., 0] = xx * 255 // (width - 1)
            frames[t, ..., 1] = yy * 255 // (height - 1)
            frames[t, ..., 2] = (8 * t) % 256

        video = encode_to_video_bytes(frames, fps=16, output_format="mp4")

        boxes = _top_level_boxes(video)
        assert boxes[0] == "ftyp"
        # +faststart: the index precedes the media, so the file plays progressively.
        assert boxes.index("moov") < boxes.index("mdat"), boxes

        path = tmp_path / "clip.mp4"
        path.write_bytes(video)
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]
        info = next(
            line
            for line in proc.stderr.decode(errors="replace").splitlines()
            if "Video:" in line
        )
        assert "av1" in info and "yuv420p" in info, info
        decoded = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
            -1, height, width, 3
        )
        assert decoded.shape == frames.shape
        mse = np.mean((decoded.astype(np.float64) - frames.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse)
        assert psnr >= 25.0, f"decoded picture does not match the input: {psnr:.1f} dB"
