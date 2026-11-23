# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the video-encoder tests.

Which encoder runs is decided by the environment (``DYN_XPU_FFMPEG_PATH``), so
the tests must not inherit that decision from the shell they run in: every test
starts with software encoding, and the hardware tests opt in explicitly.
"""

import os
from typing import Tuple

import pytest

from dynamo.common.utils.ffmpeg_command import (
    DEFAULT_XPU_VIDEO_DEVICE,
    ENV_XPU_FFMPEG_PATH,
    ENV_XPU_VIDEO_DEVICE,
)

# What the operator declared for this run, read before any fixture clears it.
_DECLARED_HW_FFMPEG = (os.environ.get(ENV_XPU_FFMPEG_PATH) or "").strip()
_DECLARED_HW_DEVICE = (os.environ.get(ENV_XPU_VIDEO_DEVICE) or "").strip()


@pytest.fixture(autouse=True)
def software_video_encoder_by_default(monkeypatch):
    """Start every test with no hardware encoder declared."""
    monkeypatch.delenv(ENV_XPU_FFMPEG_PATH, raising=False)
    monkeypatch.delenv(ENV_XPU_VIDEO_DEVICE, raising=False)


@pytest.fixture
def hw_video_encoder(monkeypatch) -> Tuple[str, str]:
    """Select the hardware encoder declared for this run; skip if there is none.

    Returns ``(ffmpeg, render_node)``. Hardware is never probed for: the tests
    run only where ``DYN_XPU_FFMPEG_PATH`` was set on purpose.
    """
    if not _DECLARED_HW_FFMPEG:
        pytest.skip(f"{ENV_XPU_FFMPEG_PATH} is not set: no hardware encoder declared")
    device = _DECLARED_HW_DEVICE or DEFAULT_XPU_VIDEO_DEVICE
    if not os.access(_DECLARED_HW_FFMPEG, os.X_OK):
        pytest.skip(f"{ENV_XPU_FFMPEG_PATH}={_DECLARED_HW_FFMPEG!r} is not executable")
    if not os.path.exists(device):
        pytest.skip(f"render node {device} does not exist")
    monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, _DECLARED_HW_FFMPEG)
    monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, device)
    return _DECLARED_HW_FFMPEG, device
