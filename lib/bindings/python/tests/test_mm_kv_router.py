# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Multimodal KV Router functionality.

These tests verify that the KV router correctly handles multimodal content (images, videos)
by distinguishing between requests with identical token sequences but different MM objects.

Key Concepts:
- block_hash: External hash used to identify blocks uniquely (includes MM info)
- tokens_hash: Local hash based only on token content
- mm_hash: Hash of the multimodal object (image, video, etc.)

Test Strategy:
- Use RadixTree directly to avoid NATS/etcd infrastructure dependencies
- Simulate multiple workers caching same tokens with different MM content
- Verify that routing distinguishes between different MM objects
"""

import json
from typing import Any

import pytest

from dynamo.llm import RadixTree, compute_block_hash_for_seq_py

pytestmark = pytest.mark.pre_merge

# Constants for testing
DEFAULT_BLOCK_SIZE = 32
MM_HASH_1 = 0xDEADBEEF
MM_HASH_2 = 0xCAFEBABE
MM_HASH_3 = 0xFEEDFACE


def make_mm_info(mm_hash: int, offsets: list[list[int]] | None = None) -> dict:
    """Create a block's MM extra info structure."""
    if offsets is None:
        offsets = [[0, 10]]
    return {"mm_objects": [{"mm_hash": mm_hash, "offsets": offsets}]}


def make_store_event(
    event_id: int,
    blocks: list[dict],
    parent_hash: int | None = None,
) -> bytes:
    """Create a JSON-encoded store event for RadixTree."""
    event = {
        "event_id": event_id,
        "data": {
            "stored": {
                "parent_hash": parent_hash,
                "blocks": blocks,
            }
        },
    }
    return json.dumps(event).encode("utf-8")


def make_block(
    block_hash: int,
    tokens_hash: int | None = None,
    mm_info: dict | None = None,
) -> dict:
    """Create a block structure for store events."""
    block: dict[str, Any] = {
        "block_hash": block_hash,
        "tokens_hash": tokens_hash if tokens_hash is not None else block_hash,
    }
    if mm_info is not None:
        block["mm_extra_info"] = mm_info
    return block


# =============================================================================
# RadixTree MM Routing Tests
# =============================================================================


# # @pytest.mark.timeout(5)
def test_radix_tree_mm_routing_basic():
    """Test RadixTree correctly distinguishes blocks with same tokens but different MM content."""
    radix_tree = RadixTree()

    # Worker 0: Store block with MM Object 1
    worker_0, block_hash_w0 = 0, 1000
    event_w0 = make_store_event(
        event_id=1,
        blocks=[make_block(block_hash_w0, mm_info=make_mm_info(MM_HASH_1))],
    )
    radix_tree.apply_event(worker_0, event_w0)

    # Worker 1: Store block with DIFFERENT MM Object (same tokens)
    worker_1, block_hash_w1 = 1, 2000
    event_w1 = make_store_event(
        event_id=2,
        blocks=[make_block(block_hash_w1, mm_info=make_mm_info(MM_HASH_2))],
    )
    radix_tree.apply_event(worker_1, event_w1)

    # Verify both blocks are stored
    all_blocks = radix_tree.dump_tree_as_events()
    assert len(all_blocks) == 2

    # Query for worker 0's block
    scores_w0 = radix_tree.find_matches([block_hash_w0])
    assert (worker_0, 0) in scores_w0.scores
    assert scores_w0.scores[(worker_0, 0)] == 1

    # Query for worker 1's block
    scores_w1 = radix_tree.find_matches([block_hash_w1])
    assert (worker_1, 0) in scores_w1.scores
    assert scores_w1.scores[(worker_1, 0)] == 1

    # Query with non-existent hash should return no matches
    scores_none = radix_tree.find_matches([9999])
    assert len(scores_none.scores) == 0


# @pytest.mark.timeout(5)
def test_radix_tree_mm_block_chaining():
    """Test block chaining with parent_hash for multi-block sequences with MM content."""
    radix_tree = RadixTree()

    worker_id = 0
    parent_hash = 1000
    child_hash = 2000

    # Store parent block
    parent_event = make_store_event(
        event_id=1,
        blocks=[make_block(parent_hash, mm_info=make_mm_info(MM_HASH_1))],
    )
    radix_tree.apply_event(worker_id, parent_event)

    # Store child block that references parent
    child_event = make_store_event(
        event_id=2,
        blocks=[make_block(child_hash, mm_info=make_mm_info(MM_HASH_1))],
        parent_hash=parent_hash,
    )
    radix_tree.apply_event(worker_id, child_event)

    # Verify chain exists
    all_blocks = radix_tree.dump_tree_as_events()
    assert len(all_blocks) == 2

    # Query with both hashes should match the chain
    scores = radix_tree.find_matches([parent_hash, child_hash])
    assert (worker_id, 0) in scores.scores
    assert scores.scores[(worker_id, 0)] == 2


# @pytest.mark.timeout(5)
def test_radix_tree_worker_removal():
    """Test worker removal clears all its blocks."""
    radix_tree = RadixTree()

    worker_0, worker_1 = 0, 1

    # Add blocks for both workers
    radix_tree.apply_event(
        worker_0,
        make_store_event(1, [make_block(1000, mm_info=make_mm_info(MM_HASH_1))]),
    )
    radix_tree.apply_event(
        worker_1,
        make_store_event(2, [make_block(2000, mm_info=make_mm_info(MM_HASH_2))]),
    )

    assert len(radix_tree.dump_tree_as_events()) == 2

    # Remove worker 0
    radix_tree.remove_worker(worker_0)

    # Only worker 1's block should remain
    remaining = radix_tree.dump_tree_as_events()
    assert len(remaining) == 1

    scores = radix_tree.find_matches([2000])
    assert (worker_1, 0) in scores.scores


# @pytest.mark.timeout(5)
def test_radix_tree_clear_all_blocks():
    """Test clearing all blocks for a specific worker."""
    radix_tree = RadixTree()

    worker_id = 0

    # Add multiple blocks
    radix_tree.apply_event(
        worker_id,
        make_store_event(1, [make_block(1000), make_block(2000)]),
    )

    assert len(radix_tree.dump_tree_as_events()) == 2

    # Clear all blocks for worker
    radix_tree.clear_all_blocks(worker_id)

    assert len(radix_tree.dump_tree_as_events()) == 0


# =============================================================================
# Block Hash Computation Tests
# =============================================================================


# @pytest.mark.timeout(5)
def test_mm_block_hash_computation_basic():
    """Test that same tokens with different MM content produce different hashes."""
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Without MM info
    hashes_no_mm = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE)
    assert len(hashes_no_mm) == 1

    # With MM info 1
    hashes_mm1 = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(MM_HASH_1)]
    )
    assert len(hashes_mm1) == 1

    # With MM info 2
    hashes_mm2 = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(MM_HASH_2)]
    )
    assert len(hashes_mm2) == 1

    # All three should be different
    assert hashes_no_mm != hashes_mm1
    assert hashes_no_mm != hashes_mm2
    assert hashes_mm1 != hashes_mm2


# @pytest.mark.timeout(5)
def test_mm_block_hash_determinism():
    """Test that hash computation is deterministic."""
    tokens = [100] * DEFAULT_BLOCK_SIZE
    mm_info = [make_mm_info(MM_HASH_1)]

    hash1 = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, mm_info)
    hash2 = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, mm_info)

    assert hash1 == hash2


# @pytest.mark.timeout(5)
@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_mm_block_hash_multiple_blocks(block_size: int):
    """Test hash computation for sequences spanning multiple blocks."""
    num_blocks = 3
    # Use different tokens per block to get unique hashes
    tokens = []
    for i in range(num_blocks):
        tokens.extend([100 + i] * block_size)

    # One MM info per block
    mm_infos = [make_mm_info(MM_HASH_1) for _ in range(num_blocks)]

    hashes = compute_block_hash_for_seq_py(tokens, block_size, mm_infos)

    assert len(hashes) == num_blocks
    # Each block should have a unique hash (due to different tokens)
    assert len(set(hashes)) == num_blocks


# @pytest.mark.timeout(5)
def test_mm_block_hash_partial_block():
    """Test hash computation when tokens don't fill complete blocks."""
    # 1.5 blocks worth of tokens
    tokens = [100] * (DEFAULT_BLOCK_SIZE + DEFAULT_BLOCK_SIZE // 2)

    # MM info for each block
    mm_infos = [make_mm_info(MM_HASH_1), make_mm_info(MM_HASH_2)]

    hashes = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, mm_infos)

    # Only complete blocks get hashes - partial blocks are not hashed
    assert len(hashes) == 1


# @pytest.mark.timeout(5)
def test_mm_block_hash_none_mm_info():
    """Test that None MM info is handled correctly."""
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Pass None for some blocks' MM info
    mm_infos = [None]

    hashes_with_none = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, mm_infos
    )
    hashes_without = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE)

    # Both should produce the same result
    assert hashes_with_none == hashes_without


# @pytest.mark.timeout(5)
def test_mm_block_hash_different_offsets():
    """Test that same mm_hash with different offsets produces same hash."""
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Same MM hash, different offsets
    mm_info_1 = make_mm_info(MM_HASH_1, offsets=[[0, 10]])
    mm_info_2 = make_mm_info(MM_HASH_1, offsets=[[5, 15]])

    hash1 = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, [mm_info_1])
    hash2 = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, [mm_info_2])

    # Currently offsets are not included in hash computation - just mm_hash
    # This behavior may change - update test if needed
    assert hash1 == hash2


# @pytest.mark.timeout(5)
def test_mm_block_hash_multiple_mm_objects():
    """Test hash with multiple MM objects in a single block."""
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Multiple MM objects in one block
    mm_info = {
        "mm_objects": [
            {"mm_hash": MM_HASH_1, "offsets": [[0, 5]]},
            {"mm_hash": MM_HASH_2, "offsets": [[10, 15]]},
        ]
    }

    hashes = compute_block_hash_for_seq_py(tokens, DEFAULT_BLOCK_SIZE, [mm_info])

    assert len(hashes) == 1

    # Compare with single MM object
    single_mm_hashes = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(MM_HASH_1)]
    )

    # Should be different due to additional MM object
    assert hashes != single_mm_hashes


# @pytest.mark.timeout(5)
def test_mm_block_hash_error_zero_block_size():
    """Test that zero block size raises an error."""
    tokens = [100] * 32

    with pytest.raises(ValueError, match="kv_block_size cannot be 0"):
        compute_block_hash_for_seq_py(tokens, 0)


# =============================================================================
# Integration Tests: RadixTree + Hash Computation
# =============================================================================


# @pytest.mark.timeout(5)
def test_integration_mm_hash_to_routing():
    """Test end-to-end: compute hash -> store in tree -> query matches correctly."""
    radix_tree = RadixTree()
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Compute hashes for two different MM contents
    hash_mm1 = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(MM_HASH_1)]
    )[0]
    hash_mm2 = compute_block_hash_for_seq_py(
        tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(MM_HASH_2)]
    )[0]

    # Store each on different workers
    worker_0, worker_1 = 0, 1

    radix_tree.apply_event(
        worker_0,
        make_store_event(1, [make_block(hash_mm1, mm_info=make_mm_info(MM_HASH_1))]),
    )
    radix_tree.apply_event(
        worker_1,
        make_store_event(2, [make_block(hash_mm2, mm_info=make_mm_info(MM_HASH_2))]),
    )

    # Query with MM1's hash should match worker 0
    scores_mm1 = radix_tree.find_matches([hash_mm1])
    assert (worker_0, 0) in scores_mm1.scores
    assert (worker_1, 0) not in scores_mm1.scores

    # Query with MM2's hash should match worker 1
    scores_mm2 = radix_tree.find_matches([hash_mm2])
    assert (worker_1, 0) in scores_mm2.scores
    assert (worker_0, 0) not in scores_mm2.scores


# @pytest.mark.timeout(5)
@pytest.mark.parametrize("num_workers", [2, 3, 5])
def test_integration_multiple_workers_same_tokens(num_workers: int):
    """Test routing with multiple workers caching same tokens but different MM content."""
    radix_tree = RadixTree()
    tokens = [100] * DEFAULT_BLOCK_SIZE

    # Each worker has unique MM content
    mm_hashes = [0x1000 + i for i in range(num_workers)]

    # Store blocks for each worker
    for worker_id, mm_hash in enumerate(mm_hashes):
        block_hash = compute_block_hash_for_seq_py(
            tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(mm_hash)]
        )[0]

        radix_tree.apply_event(
            worker_id,
            make_store_event(
                event_id=worker_id + 1,
                blocks=[make_block(block_hash, mm_info=make_mm_info(mm_hash))],
            ),
        )

    # Verify all blocks stored
    assert len(radix_tree.dump_tree_as_events()) == num_workers

    # Query for each worker's block should match only that worker
    for worker_id, mm_hash in enumerate(mm_hashes):
        block_hash = compute_block_hash_for_seq_py(
            tokens, DEFAULT_BLOCK_SIZE, [make_mm_info(mm_hash)]
        )[0]

        scores = radix_tree.find_matches([block_hash])

        assert (worker_id, 0) in scores.scores
        assert scores.scores[(worker_id, 0)] == 1

        # No other workers should match
        for other_id in range(num_workers):
            if other_id != worker_id:
                assert (other_id, 0) not in scores.scores
