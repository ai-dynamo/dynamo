# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multimodal routing utilities."""

from unittest.mock import MagicMock

import pytest

from dynamo.common.multimodal.routing_utils import (
    build_mm_routing_info_from_features,
    pad_value_for_mm_hash,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestPadValueForMmHash:
    """Pin the pad_value formula against the Rust/sglang definition."""

    def test_formula(self):
        assert pad_value_for_mm_hash(0) == 1_000_000
        fits = (1 << 30) - 1
        assert pad_value_for_mm_hash(fits) == 1_000_000 + fits
        # high bits above the 30-bit mask are discarded
        overflow = (1 << 30) | 0xCAFE
        assert pad_value_for_mm_hash(overflow) == 1_000_000 + 0xCAFE


class TestBuildMmRoutingInfoFromFeatures:
    """Tests for build_mm_routing_info_from_features (pad_value scheme)."""

    def _make_feature(self, mm_hash, offset, length, is_embed=None):
        feat = MagicMock()
        feat.mm_hash = mm_hash
        feat.mm_position = MagicMock()
        feat.mm_position.offset = offset
        feat.mm_position.length = length
        feat.mm_position.is_embed = is_embed
        return feat

    def test_single_feature_substitutes_pad_value(self):
        mm_hash = int("abcdef0123456789", 16)
        features = [self._make_feature("abcdef0123456789" + "0" * 48, 2, 4)]
        token_ids = list(range(10))

        result = build_mm_routing_info_from_features(features, token_ids)

        assert result is not None
        assert result["block_mm_infos"] == []
        pad = pad_value_for_mm_hash(mm_hash)
        expected = [0, 1, pad, pad, pad, pad, 6, 7, 8, 9]
        assert result["routing_token_ids"] == expected

    def test_two_features_each_get_own_pad_value(self):
        h1 = int("1111111111111111", 16)
        h2 = int("2222222222222222", 16)
        features = [
            self._make_feature("1111111111111111" + "0" * 48, 0, 2),
            self._make_feature("2222222222222222" + "0" * 48, 4, 2),
        ]
        token_ids = list(range(6))

        result = build_mm_routing_info_from_features(features, token_ids)

        p1, p2 = pad_value_for_mm_hash(h1), pad_value_for_mm_hash(h2)
        assert result["routing_token_ids"] == [p1, p1, 2, 3, p2, p2]
        assert result["block_mm_infos"] == []

    def test_is_embed_mask_leaves_boundary_tokens_unchanged(self):
        mm_hash = int("abcdef0123456789", 16)
        features = [
            self._make_feature(
                "abcdef0123456789" + "0" * 48,
                2,
                4,
                is_embed=[False, True, True, False],
            )
        ]
        token_ids = list(range(8))

        result = build_mm_routing_info_from_features(features, token_ids)

        assert result is not None
        pad = pad_value_for_mm_hash(mm_hash)
        assert result["routing_token_ids"] == [0, 1, 2, pad, pad, 5, 6, 7]
        assert result["block_mm_infos"] == []

    def test_no_features_returns_none(self):
        assert build_mm_routing_info_from_features([], [1, 2, 3]) is None

    def test_feature_with_none_hash_skipped(self):
        feat = self._make_feature(None, 0, 16)
        result = build_mm_routing_info_from_features([feat], list(range(16)))
        assert result is None  # all features skipped

    def test_range_clamped_to_token_count(self):
        # length overruns the token list; substitution must not index past end.
        features = [self._make_feature("ff" * 32, 0, 100)]
        token_ids = list(range(4))
        result = build_mm_routing_info_from_features(features, token_ids)
        assert result is not None
        pad = pad_value_for_mm_hash(int("ff" * 8, 16))
        assert result["routing_token_ids"] == [pad, pad, pad, pad]
