# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.cmaf_video module."""

import json

import numpy as np
import pytest

from dynamo.common.utils import cmaf_video

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestIterCmafChunks:
    """Segments are fixed-size except the last, which carries the remainder."""

    def test_model_default_leaves_a_single_frame_tail(self):
        """97 frames at 8 is twelve full segments and a 1-frame segment."""
        frames = np.zeros((97, 4, 4, 3), dtype=np.uint8)
        sizes = [len(c) for c in cmaf_video.iter_cmaf_chunks(frames, 8)]
        assert sizes == [8] * 12 + [1]

    def test_chunks_cover_every_frame_in_order(self):
        frames = np.arange(30, dtype=np.uint8).reshape(30, 1, 1, 1)
        joined = np.concatenate(list(cmaf_video.iter_cmaf_chunks(frames, 8)))
        assert np.array_equal(joined, frames)

    def test_exact_multiple_has_no_tail(self):
        frames = np.zeros((32, 2, 2, 3), dtype=np.uint8)
        assert [len(c) for c in cmaf_video.iter_cmaf_chunks(frames, 8)] == [8] * 4

    def test_rejects_a_non_positive_size(self):
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be positive"):
            list(cmaf_video.iter_cmaf_chunks(frames, 0))


# ---------------------------------------------------------------------------
# Segment length
# ---------------------------------------------------------------------------


class TestCmafGopFrames:
    """DYN_CMAF_GOP_FRAMES must be a positive multiple of four."""

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(cmaf_video.ENV_CMAF_GOP_FRAMES, raising=False)
        assert cmaf_video.cmaf_gop_frames() == cmaf_video.DEFAULT_CMAF_GOP_FRAMES

    def test_blank_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(cmaf_video.ENV_CMAF_GOP_FRAMES, "  ")
        assert cmaf_video.cmaf_gop_frames() == cmaf_video.DEFAULT_CMAF_GOP_FRAMES

    def test_override(self, monkeypatch):
        monkeypatch.setenv(cmaf_video.ENV_CMAF_GOP_FRAMES, "16")
        assert cmaf_video.cmaf_gop_frames() == 16

    @pytest.mark.parametrize("bad", ["6", "0", "-4", "eight", "8.0"])
    def test_rejects_anything_but_a_positive_multiple_of_four(self, monkeypatch, bad):
        """Off the VAE's temporal stride of 4 the carry oscillates (DEP 0017)."""
        monkeypatch.setenv(cmaf_video.ENV_CMAF_GOP_FRAMES, bad)
        with pytest.raises(ValueError, match=cmaf_video.ENV_CMAF_GOP_FRAMES):
            cmaf_video.cmaf_gop_frames()

    def test_segment_seconds_follows_the_frame_rate(self):
        assert cmaf_video.cmaf_segment_seconds(16, 8) == 0.5


# ---------------------------------------------------------------------------
# Wire vocabulary
# ---------------------------------------------------------------------------


class TestFrameTags:
    def test_segment_tags_are_one_based(self):
        assert cmaf_video.segment_tag(1) == "cmaf:segment:1"
        assert cmaf_video.segment_tag(13).startswith(cmaf_video.CMAF_SEGMENT_TAG_PREFIX)

    def test_metadata_mime_type_carries_the_codec_string(self):
        """The client hands mime_type straight to MediaSource.isTypeSupported()."""
        meta = json.loads(
            cmaf_video.metadata_bytes(
                video_codec="vp09.00.10.08",
                width=832,
                height=480,
                fps=16,
                target_duration=0.5,
                segment_count=4,
            )
        )
        assert meta["mime_type"] == 'video/mp4; codecs="vp09.00.10.08"'
        assert (meta["width"], meta["height"], meta["fps"]) == (832, 480, 16)
        assert meta["segment_count"] == 4

    def test_metadata_allows_an_unknown_segment_count(self):
        meta = json.loads(
            cmaf_video.metadata_bytes(
                video_codec="av01.0.05M.08",
                width=64,
                height=64,
                fps=16,
                target_duration=0.5,
            )
        )
        assert meta["segment_count"] is None
