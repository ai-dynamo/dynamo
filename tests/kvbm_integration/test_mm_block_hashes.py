# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for compute_mm_block_hashes and _hash_features_to_u64.

These validate that KVBM correctly produces per-block extra hashes for
multimodal (VLM) requests, ensuring different images produce different
block hashes and text-only blocks remain None (prefix-sharable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from .common import check_module_available

# kvbm.vllm_integration.__init__ transitively imports vllm connectors,
# so both kvbm and vllm must be importable.
HAS_DEPS = check_module_available("kvbm") and check_module_available("vllm")

if HAS_DEPS:
    from kvbm.vllm_integration.mm_block_hashes import (  # noqa: E402
        _hash_features_to_u64,
        compute_mm_block_hashes,
    )

pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.skipif(not HAS_DEPS, reason="requires kvbm and vllm"),
]


# ---------------------------------------------------------------------------
# Minimal stubs that mirror vLLM's PlaceholderRange / MultiModalFeatureSpec
# so tests don't depend on vLLM types at runtime.
# ---------------------------------------------------------------------------


@dataclass
class FakePlaceholderRange:
    offset: int
    length: int


@dataclass
class FakeFeatureSpec:
    """Mirrors vLLM MultiModalFeatureSpec with the fields used by hashing."""

    data: Any
    modality: str
    identifier: str
    mm_position: FakePlaceholderRange


# ===========================================================================
# _hash_features_to_u64 tests
# ===========================================================================


class TestHashFeaturesToU64:
    def test_different_identifiers_produce_different_hashes(self):
        """Core property: two images with different identifiers must hash differently."""
        feat_a = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="hash_aaa",
            mm_position=FakePlaceholderRange(0, 10),
        )
        feat_b = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="hash_bbb",
            mm_position=FakePlaceholderRange(0, 10),
        )
        assert _hash_features_to_u64(feat_a) != _hash_features_to_u64(feat_b)

    def test_same_identifier_same_hash(self):
        """Determinism: same feature spec must produce the same hash."""
        feat = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="hash_aaa",
            mm_position=FakePlaceholderRange(0, 10),
        )
        assert _hash_features_to_u64(feat) == _hash_features_to_u64(feat)

    def test_same_identifier_ignores_data(self):
        """Hash depends only on identifier, not on data (aligned with vLLM)."""
        feat_no_data = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="abc",
            mm_position=FakePlaceholderRange(0, 5),
        )
        feat_with_data = FakeFeatureSpec(
            data={"pixels": b"\x00\x01\x02"},
            modality="image",
            identifier="abc",
            mm_position=FakePlaceholderRange(0, 5),
        )
        # Same identifier → same hash regardless of data
        assert _hash_features_to_u64(feat_no_data) == _hash_features_to_u64(
            feat_with_data
        )

    def test_none_feature_returns_int(self):
        """None feature should still produce a valid u64 hash."""
        h = _hash_features_to_u64(None)
        assert isinstance(h, int)
        assert 0 <= h < (1 << 64)

    def test_plain_object_without_identifier(self):
        """Objects without 'identifier' attr fall back to str-based hashing."""
        h = _hash_features_to_u64({"key": "value"})
        assert isinstance(h, int)

    def test_data_none_identifier_present(self):
        """When data is None but identifier is set, identifier alone drives the hash."""
        feat_a = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="img1",
            mm_position=FakePlaceholderRange(0, 5),
        )
        feat_b = FakeFeatureSpec(
            data=None,
            modality="image",
            identifier="img2",
            mm_position=FakePlaceholderRange(0, 5),
        )
        # Same modality, position, but different identifier → different hash
        assert _hash_features_to_u64(feat_a) != _hash_features_to_u64(feat_b)


# ===========================================================================
# compute_mm_block_hashes tests
# ===========================================================================


def _pos(offset: int, length: int) -> FakePlaceholderRange:
    return FakePlaceholderRange(offset=offset, length=length)


def _feat(identifier: str, offset: int, length: int) -> FakeFeatureSpec:
    return FakeFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=_pos(offset, length),
    )


class TestComputeMmBlockHashes:
    """Tests for the top-level compute_mm_block_hashes function."""

    # -- text-only (no MM) --------------------------------------------------

    def test_text_only_all_none(self):
        """A text-only request (no mm_positions) returns all-None list."""
        result = compute_mm_block_hashes(
            mm_positions=[], mm_features=[], num_tokens=64, block_size=16
        )
        assert result == [None, None, None, None]

    def test_text_only_none_positions(self):
        """mm_positions=None should also return all None."""
        result = compute_mm_block_hashes(
            mm_positions=None, mm_features=None, num_tokens=32, block_size=16
        )
        assert result == [None, None]

    def test_zero_tokens(self):
        result = compute_mm_block_hashes(
            mm_positions=[], mm_features=[], num_tokens=0, block_size=16
        )
        assert result == []

    # -- single image -------------------------------------------------------

    def test_single_image_marks_correct_blocks(self):
        """Image spanning tokens 16..48 should mark blocks 1 and 2 (block_size=16)."""
        positions = [_pos(16, 32)]  # tokens [16, 48)
        features = [_feat("img1", 16, 32)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=64,
            block_size=16,
        )
        assert len(result) == 4
        assert result[0] is None  # text block
        assert result[1] is not None  # image block
        assert result[2] is not None  # image block
        assert result[3] is None  # text block

    def test_different_images_different_hashes(self):
        """Two requests with same text but different images must produce different hashes."""
        positions = [_pos(0, 16)]
        feat_bus = [_feat("bus_image_hash", 0, 16)]
        feat_duck = [_feat("duck_image_hash", 0, 16)]

        result_bus = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=feat_bus,
            num_tokens=16,
            block_size=16,
        )
        result_duck = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=feat_duck,
            num_tokens=16,
            block_size=16,
        )

        assert result_bus[0] is not None
        assert result_duck[0] is not None
        assert result_bus[0] != result_duck[0]

    def test_same_image_same_hashes(self):
        """Same image should produce identical block hashes (deterministic)."""
        positions = [_pos(0, 16)]
        features = [_feat("same_hash", 0, 16)]

        r1 = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=16,
            block_size=16,
        )
        r2 = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=16,
            block_size=16,
        )
        assert r1 == r2

    # -- image at block boundary --------------------------------------------

    def test_image_starts_mid_block(self):
        """Image starting at token 8 with block_size=16 should mark block 0."""
        positions = [_pos(8, 16)]  # tokens [8, 24)
        features = [_feat("img1", 8, 16)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=32,
            block_size=16,
        )
        # Block 0: tokens [0,16) overlaps with [8,24)
        # Block 1: tokens [16,32) overlaps with [8,24)
        assert result[0] is not None
        assert result[1] is not None

    def test_image_exactly_one_block(self):
        """Image filling exactly one block."""
        positions = [_pos(0, 16)]
        features = [_feat("img1", 0, 16)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=32,
            block_size=16,
        )
        assert result[0] is not None
        assert result[1] is None

    # -- multiple images ----------------------------------------------------

    def test_two_images_separate_blocks(self):
        """Two images in different blocks get independent hashes."""
        positions = [_pos(0, 16), _pos(32, 16)]
        features = [_feat("img_a", 0, 16), _feat("img_b", 32, 16)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=64,
            block_size=16,
        )
        assert result[0] is not None  # img_a
        assert result[1] is None  # text
        assert result[2] is not None  # img_b
        assert result[3] is None  # text
        assert result[0] != result[2]  # different images

    def test_two_images_same_block_combined(self):
        """Two images overlapping the same block get a combined hash."""
        # Both images overlap block 0 (tokens 0..16)
        positions = [_pos(0, 8), _pos(8, 8)]
        features = [_feat("img_a", 0, 8), _feat("img_b", 8, 8)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=16,
            block_size=16,
        )
        assert result[0] is not None

        # Compare with single-image hash — combined should differ
        single_a = compute_mm_block_hashes(
            mm_positions=[_pos(0, 8)],
            mm_features=[_feat("img_a", 0, 8)],
            num_tokens=16,
            block_size=16,
        )
        assert result[0] != single_a[0]

    # -- edge cases ---------------------------------------------------------

    def test_zero_length_position_skipped(self):
        """A position with length 0 should be skipped (no blocks marked)."""
        positions = [_pos(0, 0)]
        features = [_feat("img1", 0, 0)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=16,
            block_size=16,
        )
        assert result == [None]

    def test_partial_last_block(self):
        """num_tokens not a multiple of block_size → ceil produces extra block."""
        positions = [_pos(0, 20)]
        features = [_feat("img1", 0, 20)]
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=20,
            block_size=16,
        )
        # ceil(20/16) = 2 blocks
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is not None

    def test_features_shorter_than_positions(self):
        """If features list is shorter than positions, extra positions get hash 0."""
        positions = [_pos(0, 16), _pos(16, 16)]
        features = [_feat("img1", 0, 16)]  # only one feature for two positions
        result = compute_mm_block_hashes(
            mm_positions=positions,
            mm_features=features,
            num_tokens=32,
            block_size=16,
        )
        assert result[0] is not None  # from feature
        assert result[1] is not None  # from hash=0 fallback

    # -- derive positions from mm_features (vLLM 0.18+) --------------------

    def test_derive_positions_from_features(self):
        """When mm_positions is None/empty, positions should be derived from mm_features."""
        features = [_feat("img1", 16, 32)]
        result = compute_mm_block_hashes(
            mm_positions=None,
            mm_features=features,
            num_tokens=64,
            block_size=16,
        )
        assert len(result) == 4
        assert result[0] is None
        assert result[1] is not None
        assert result[2] is not None
        assert result[3] is None

    def test_derive_positions_from_features_empty_list(self):
        """Empty mm_positions list should also trigger derivation from features."""
        features = [_feat("img1", 0, 16)]
        result = compute_mm_block_hashes(
            mm_positions=[],
            mm_features=features,
            num_tokens=16,
            block_size=16,
        )
        assert result[0] is not None

    def test_derive_positions_skips_none_keeps_alignment(self):
        """When deriving positions, features with mm_position=None must be
        skipped without misaligning the remaining (position, feature) pairs."""

        @dataclass
        class _NoPosition:
            data: Any
            modality: str
            identifier: str
            # no mm_position attribute

        feat_a = _feat("img_a", 0, 16)
        feat_no_pos = _NoPosition(data=None, modality="image", identifier="img_orphan")
        feat_c = _feat("img_c", 32, 16)

        result = compute_mm_block_hashes(
            mm_positions=None,
            mm_features=[feat_a, feat_no_pos, feat_c],
            num_tokens=48,
            block_size=16,
        )
        assert len(result) == 3
        hash_a = result[0]
        hash_c = result[2]
        assert hash_a is not None
        assert result[1] is None  # text gap
        assert hash_c is not None

        # Verify block 0 got img_a's hash, not img_orphan's
        expected_a = compute_mm_block_hashes(
            mm_positions=None,
            mm_features=[_feat("img_a", 0, 16)],
            num_tokens=48,
            block_size=16,
        )[0]
        assert hash_a == expected_a

        # Verify block 2 got img_c's hash, not img_orphan's
        expected_c = compute_mm_block_hashes(
            mm_positions=None,
            mm_features=[_feat("img_c", 32, 16)],
            num_tokens=48,
            block_size=16,
        )[2]
        assert hash_c == expected_c
