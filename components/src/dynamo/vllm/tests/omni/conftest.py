# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by the vLLM-Omni tests."""

import pytest

from dynamo.common.utils.cmaf_video import CMAF_INIT_TAG, segment_tag


class FakeCmafEncoder:
    """Stands in for StreamingCmafEncoder: same call shape, no ffmpeg process.

    Mirrors how a fragmented muxer hands fragments back: the init segment comes
    with the first chunk, fragment *k* is only complete once chunk *k+1* has
    been written, and the last fragment appears when the input is closed.
    """

    def __init__(self, fps, width, height, *, gop_frames=None, **kwargs):
        """Record the session parameters."""
        self.fps = fps
        self.width = width
        self.height = height
        self.gop_frames = gop_frames
        self.chunks = []
        self.started = False
        self.closed = False

    async def start(self):
        """Mark the session started."""
        self.started = True

    async def aclose(self):
        """Mark the session closed."""
        self.closed = True

    def codec_string(self):
        """A fixed VP9 codec string."""
        return "vp09.00.10.08"

    async def push(self, frames):
        """Emit init with the first chunk, then the fragment the chunk completes."""
        self.chunks.append(frames)
        if len(self.chunks) == 1:
            yield CMAF_INIT_TAG, b"ftyp-moov"
        else:
            yield segment_tag(len(self.chunks) - 1), b"moof-mdat"

    async def finish(self):
        """Emit the fragment the muxer can only close at end of input."""
        yield segment_tag(len(self.chunks)), b"last-moof-mdat"


class _FakeCmafEncoderFactory:
    """What the formatter calls in place of the encoder class.

    ``cls`` is the class to instantiate; a test may replace it with a subclass
    before streaming. ``enc`` is the last instance built, ``None`` if none was.
    """

    def __init__(self):
        self.cls = FakeCmafEncoder
        self.enc = None

    def __call__(self, *args, **kwargs):
        self.enc = self.cls(*args, **kwargs)
        return self.enc


@pytest.fixture
def fake_cmaf_encoder(monkeypatch):
    """Replace the formatter's CMAF encoder with one that starts no process."""
    factory = _FakeCmafEncoderFactory()
    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.StreamingCmafEncoder", factory
    )
    return factory
