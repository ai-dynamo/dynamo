# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multimodal routing utilities."""

from unittest.mock import MagicMock

import pytest

from dynamo.common.multimodal.routing_utils import (
    build_block_mm_infos,
    build_mm_routing_info_from_features,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestBuildBlockMmInfos:
    """Tests for build_block_mm_infos."""

    def test_single_image_single_block(self):
        """Image fits within one block."""
        result = build_block_mm_infos(
            num_tokens=16,
            block_size=16,
            mm_hashes=[12345],
            image_ranges=[(0, 16)],
        )
        assert result is not None
        assert len(result) == 1
        assert result[0] is not None
        assert result[0]["mm_objects"][0]["mm_hash"] == 12345

    def test_single_image_spans_multiple_blocks(self):
        """Image spans 3 blocks."""
        result = build_block_mm_infos(
            num_tokens=48,
            block_size=16,
            mm_hashes=[99999],
            image_ranges=[(0, 48)],
        )
        assert result is not None
        assert len(result) == 3
        for block in result:
            assert block is not None
            assert len(block["mm_objects"]) == 1
            assert block["mm_objects"][0]["mm_hash"] == 99999

    def test_text_block_before_image_is_none(self):
        """Block before the image range is None."""
        result = build_block_mm_infos(
            num_tokens=48,
            block_size=16,
            mm_hashes=[12345],
            image_ranges=[(16, 32)],  # middle block
        )
        assert result is not None
        assert len(result) == 3
        assert result[0] is None  # text only, before image
        assert result[1] is not None  # image block
        # Note: block 2 (start=32) overlaps with img_end=32 due to
        # the <= check (FIXME: https://github.com/ai-dynamo/dynamo/issues/6588)
        assert result[2] is not None

    def test_two_images_non_adjacent_blocks(self):
        """Two images with a gap between them."""
        result = build_block_mm_infos(
            num_tokens=64,
            block_size=16,
            mm_hashes=[111, 222],
            image_ranges=[(0, 14), (48, 64)],  # gap in blocks 1-2
        )
        assert result is not None
        assert len(result) == 4
        assert result[0]["mm_objects"][0]["mm_hash"] == 111
        # Blocks 1 and 2 are in the gap
        assert result[1] is None
        assert result[2] is None
        assert result[3]["mm_objects"][0]["mm_hash"] == 222

    def test_empty_inputs_returns_none(self):
        assert build_block_mm_infos(16, 16, None, None) is None
        assert build_block_mm_infos(16, 16, [], []) is None
        assert build_block_mm_infos(16, 16, [1], []) is None
        assert build_block_mm_infos(16, 16, [], [(0, 16)]) is None

    def test_mismatched_lengths_returns_none(self):
        assert build_block_mm_infos(16, 16, [1, 2], [(0, 16)]) is None


class TestBuildMmRoutingInfoFromFeatures:
    """Tests for build_mm_routing_info_from_features."""

    def _make_feature(self, mm_hash, offset, length):
        feat = MagicMock()
        feat.mm_hash = mm_hash
        feat.mm_position = MagicMock()
        feat.mm_position.offset = offset
        feat.mm_position.length = length
        return feat

    def test_single_feature(self):
        features = [self._make_feature("abcd1234" * 8, 0, 100)]
        token_ids = list(range(100))

        result = build_mm_routing_info_from_features(features, token_ids, block_size=16)

        assert result is not None
        assert result["routing_token_ids"] == token_ids
        assert result["block_mm_infos"] is not None
        # ceil(100 / 16) = 7 blocks
        assert len(result["block_mm_infos"]) == 7

    def test_no_features_returns_none(self):
        assert build_mm_routing_info_from_features([], [1, 2, 3], 16) is None

    def test_feature_with_none_hash_skipped(self):
        feat = self._make_feature(None, 0, 16)
        result = build_mm_routing_info_from_features([feat], list(range(16)), 16)
        assert result is None  # all features skipped

    def test_hash_truncated_to_u64(self):
        """mm_hash hex string is truncated to first 16 chars → u64."""
        features = [self._make_feature("abcdef0123456789" + "0" * 48, 0, 16)]
        result = build_mm_routing_info_from_features(
            features, list(range(16)), block_size=16
        )
        assert result is not None
        block = result["block_mm_infos"][0]
        assert block is not None
        expected_hash = int("abcdef0123456789", 16)
        assert block["mm_objects"][0]["mm_hash"] == expected_hash
