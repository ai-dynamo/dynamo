# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
"""Unit tests for `_extend_match_via_staging` — Phase 2 of cross-node.

The function extends a local prefix-index match by walking the
remaining block hashes and counting contiguous hits in the staging
tier. A gap stops the walk. This module tests the pure-function
logic in isolation; the RPC layer is tested in test_staging_rpc.py."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")  # connector module pulls in vllm

from gpu_memory_service.integrations.vllm.gds_connector_v1 import (
    _extend_match_via_staging,
)


def _h(i: int) -> bytes:
    """Stable 32-byte content hash for a small int (for test readability)."""
    return i.to_bytes(32, "big")


def test_all_staging_hits_extend_full_remaining():
    block_hashes = [_h(i) for i in range(5)]
    local_match = 2
    staging_hits = {_h(2): {}, _h(3): {}, _h(4): {}}
    assert _extend_match_via_staging(block_hashes, local_match, staging_hits) == 3


def test_gap_stops_extension():
    block_hashes = [_h(i) for i in range(5)]
    local_match = 2
    # h[2] hit, h[3] miss, h[4] hit — only h[2] extends; h[4] unreachable
    staging_hits = {_h(2): {}, _h(4): {}}
    assert _extend_match_via_staging(block_hashes, local_match, staging_hits) == 1


def test_no_immediate_hit_returns_zero():
    block_hashes = [_h(i) for i in range(5)]
    staging_hits = {_h(4): {}}  # not the next-after-local block
    assert (
        _extend_match_via_staging(
            block_hashes, local_match_blocks=2, staging_hits=staging_hits
        )
        == 0
    )


def test_empty_staging_returns_zero():
    block_hashes = [_h(i) for i in range(5)]
    assert _extend_match_via_staging(block_hashes, 2, {}) == 0


def test_local_match_covers_everything():
    block_hashes = [_h(i) for i in range(3)]
    staging_hits = {_h(0): {}, _h(1): {}, _h(2): {}}
    assert _extend_match_via_staging(block_hashes, 3, staging_hits) == 0


def test_zero_local_match_with_full_staging_coverage():
    """A request with no local hit but full staging coverage."""
    block_hashes = [_h(i) for i in range(4)]
    staging_hits = {_h(i): {} for i in range(4)}
    assert _extend_match_via_staging(block_hashes, 0, staging_hits) == 4


def test_zero_local_match_with_no_staging_for_first_block():
    """Zero local + staging starts at block 2 → no contiguous coverage."""
    block_hashes = [_h(i) for i in range(4)]
    staging_hits = {_h(2): {}, _h(3): {}}
    assert _extend_match_via_staging(block_hashes, 0, staging_hits) == 0


def test_only_first_extends_then_gap():
    block_hashes = [_h(i) for i in range(10)]
    staging_hits = {_h(0): {}}  # one hit at the very start; gap immediately
    assert _extend_match_via_staging(block_hashes, 0, staging_hits) == 1


def test_extension_bounded_by_block_count():
    """Staging may contain unrelated hashes; only those in block_hashes
    contribute to the extension."""
    block_hashes = [_h(i) for i in range(3)]
    staging_hits = {_h(0): {}, _h(1): {}, _h(2): {}, _h(99): {}, _h(100): {}}
    assert _extend_match_via_staging(block_hashes, 0, staging_hits) == 3


def test_empty_block_hashes_returns_zero():
    """Edge case: no block hashes at all (no prefix to extend)."""
    assert _extend_match_via_staging([], 0, {_h(0): {}}) == 0
