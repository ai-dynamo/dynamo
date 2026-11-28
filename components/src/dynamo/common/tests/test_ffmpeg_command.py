# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamo.common.utils.ffmpeg_command.

No process is started: encoder selection reads two environment variables and
command construction returns an argv. Both variables are cleared before each
test by ``conftest.py``.
"""

import numpy as np
import pytest

from dynamo.common.utils import ffmpeg_command
from dynamo.common.utils.ffmpeg_command import (
    ENV_XPU_FFMPEG_PATH,
    ENV_XPU_VIDEO_DEVICE,
    build_ffmpeg_command,
    hw_ffmpeg_path,
    validate_raw_rgb_frames,
    xpu_video_device,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _executable(tmp_path, name: str = "ffmpeg") -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


class TestHardwareSelection:
    """``DYN_XPU_FFMPEG_PATH`` is the only encoder switch."""

    def test_unset_means_software(self):
        assert hw_ffmpeg_path() is None

    def test_blank_means_software(self, monkeypatch):
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, "   ")
        assert hw_ffmpeg_path() is None

    def test_an_executable_selects_hardware(self, monkeypatch, tmp_path):
        exe = _executable(tmp_path)
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, f" {exe} ")
        assert hw_ffmpeg_path() == exe

    def test_a_missing_binary_is_an_error_not_software(self, monkeypatch, tmp_path):
        """The operator asked for hardware; a silent software encode would hide that."""
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match=ENV_XPU_FFMPEG_PATH):
            hw_ffmpeg_path()

    def test_a_non_executable_file_is_an_error(self, monkeypatch, tmp_path):
        path = tmp_path / "ffmpeg"
        path.write_text("not a program")
        path.chmod(0o644)
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, str(path))
        with pytest.raises(RuntimeError, match="not an executable file"):
            hw_ffmpeg_path()


class TestXpuVideoDevice:
    """The render node comes from ``DYN_XPU_VIDEO_DEVICE``."""

    def test_default_render_node(self):
        assert xpu_video_device() == ffmpeg_command.DEFAULT_XPU_VIDEO_DEVICE

    def test_override(self, monkeypatch):
        monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, "/dev/dri/renderD131")
        assert xpu_video_device() == "/dev/dri/renderD131"

    def test_blank_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, " ")
        assert xpu_video_device() == ffmpeg_command.DEFAULT_XPU_VIDEO_DEVICE


class TestValidateRawRgbFrames:
    """Raw bytes carry no shape or type, so the array is checked first."""

    def test_accepts_the_contract(self):
        validate_raw_rgb_frames(np.zeros((2, 8, 8, 3), dtype=np.uint8), what="X")

    @pytest.mark.parametrize(
        "frames, match",
        [
            ([np.zeros((8, 8, 3), dtype=np.uint8)], "np.ndarray"),
            (np.zeros((8, 8, 3), dtype=np.uint8), "shape"),
            (np.zeros((2, 8, 8, 4), dtype=np.uint8), "shape"),
            (np.zeros((2, 8, 8, 3), dtype=np.float32), "uint8"),
        ],
    )
    def test_rejects_everything_else_and_names_the_consumer(self, frames, match):
        with pytest.raises(ValueError, match=match) as err:
            validate_raw_rgb_frames(frames, what="Hardware video encoder")
        assert "Hardware video encoder" in str(err.value)


# The software streaming argv as it was before the hardware encoder existed.
# Adding hardware support must not change a byte of it.
_SOFTWARE_STREAMING_ARGV = [
    "/usr/local/bin/ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-s",
    "832x480",
    "-r",
    "16",
    "-i",
    "-",
    "-c:v",
    "libvpx-vp9",
    "-g",
    "8",
    "-keyint_min",
    "8",
    "-deadline",
    "realtime",
    "-cpu-used",
    "8",
    "-lag-in-frames",
    "0",
    "-auto-alt-ref",
    "0",
    "-row-mt",
    "1",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+frag_keyframe+empty_moov+default_base_moof",
    "-flush_packets",
    "1",
    "-f",
    "mp4",
    "pipe:1",
]


class TestBuildFfmpegCommand:
    """One builder for the batch hardware encode and both streaming encoders."""

    def test_software_streaming_argv_is_unchanged(self):
        cmd = build_ffmpeg_command(
            "/usr/local/bin/ffmpeg",
            width=832,
            height=480,
            fps=16,
            output="pipe:1",
            gop=8,
            streaming=True,
        )
        assert cmd == _SOFTWARE_STREAMING_ARGV

    def test_hardware_batch_argv(self):
        """Raw RGB in, uploaded to the VA-API surface, AV1 in a seekable mp4."""
        cmd = build_ffmpeg_command(
            "/opt/ffmpeg/bin/ffmpeg",
            width=832,
            height=480,
            fps=16,
            output="/tmp/out.mp4",
            hw_device="/dev/dri/renderD129",
        )
        assert cmd == [
            "/opt/ffmpeg/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-vaapi_device",
            "/dev/dri/renderD129",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "832x480",
            "-r",
            "16",
            "-i",
            "-",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "av1_vaapi",
            "-global_quality",
            "25",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            "/tmp/out.mp4",
        ]

    def test_hardware_streaming_argv(self):
        """Fragmented output, a pinned GOP, and the VA-API low-latency setting."""
        cmd = build_ffmpeg_command(
            "/opt/ffmpeg/bin/ffmpeg",
            width=832,
            height=480,
            fps=16,
            output="pipe:1",
            hw_device="/dev/dri/renderD129",
            gop=8,
            streaming=True,
        )
        assert cmd == [
            "/opt/ffmpeg/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-vaapi_device",
            "/dev/dri/renderD129",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "832x480",
            "-r",
            "16",
            "-i",
            "-",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "av1_vaapi",
            "-global_quality",
            "25",
            "-g",
            "8",
            "-keyint_min",
            "8",
            "-async_depth",
            "1",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "pipe:1",
        ]

    def test_hardware_takes_no_software_encoder_arguments(self):
        """libvpx options would be rejected by av1_vaapi."""
        cmd = build_ffmpeg_command(
            "ffmpeg",
            width=832,
            height=480,
            fps=16,
            output="pipe:1",
            hw_device="/dev/dri/renderD128",
            gop=8,
            streaming=True,
        )
        for libvpx_option in ("-deadline", "-cpu-used", "-lag-in-frames", "-row-mt"):
            assert libvpx_option not in cmd
