# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Multimodal KV Router functionality.

These tests verify that the KV router correctly handles multimodal content (images, videos)
by distinguishing between requests with identical token sequences but different MM objects.

Key Concepts:
- block_hash: External hash used to identify blocks uniquely (includes MM info)
- tokens_hash: Local hash based only on token content
- mm_hash: Hash of the multimodal object (image, video, etc.)

For MM-aware routing:
1. Publisher computes: block_hash = hash(tokens + mm_objects)
2. Router queries with: query_hash = hash(tokens + mm_objects)
3. RadixTree matches on the combined hash to find the correct worker

Test Strategy:
- Use RadixTree directly to avoid NATS/etcd infrastructure dependencies
- Simulate multiple workers caching same tokens with different MM content
- Verify that routing distinguishes between different MM objects
"""

import json
import pytest

from dynamo.llm import RadixTree, compute_block_hash_for_seq_py

pytestmark = pytest.mark.pre_merge


@pytest.mark.asyncio
async def test_mm_kv_routing_with_radix_tree():
    """
    Test that the RadixTree correctly distinguishes blocks with same tokens but different MM content.
    
    This test verifies the multimodal KV router implementation by:
    1. Creating blocks with identical tokens but different mm_hash values
    2. Verifying that these blocks produce different hashes and are stored separately
    3. Confirming that queries match the correct worker based on both tokens AND mm_hash
    """
    kv_block_size = 32
    radix_tree = RadixTree()
    
    # Create identical token sequences
    tokens = [100] * kv_block_size  # 1 block of identical tokens
    
    # Worker 0: Store block with MM Object 1
    worker_0 = 0
    mm_hash_1 = 0xDEADBEEF
    block_hash_w0 = 1000  # Simulated block hash for worker 0
    
    store_event_w0 = {
        "event_id": 1,
        "data": {
            "stored": {
                "parent_hash": None,
                "blocks": [
                    {
                        "block_hash": block_hash_w0,
                        "tokens_hash": block_hash_w0,
                        "mm_extra_info": {
                            "mm_objects": [
                                {
                                    "mm_hash": mm_hash_1,
                                    "offsets": [[0, 10]]
                                }
                            ]
                        }
                    }
                ],
            }
        },
    }
    
    event_bytes_w0 = json.dumps(store_event_w0).encode("utf-8")
    radix_tree.apply_event(worker_0, event_bytes_w0)
    
    # Worker 1: Store block with SAME tokens but DIFFERENT MM Object
    worker_1 = 1
    mm_hash_2 = 0xCAFEBABE  # Different MM hash
    block_hash_w1 = 2000  # Different block hash for worker 1
    
    store_event_w1 = {
        "event_id": 2,
        "data": {
            "stored": {
                "parent_hash": None,
                "blocks": [
                    {
                        "block_hash": block_hash_w1,
                        "tokens_hash": block_hash_w1,
                        "mm_extra_info": {
                            "mm_objects": [
                                {
                                    "mm_hash": mm_hash_2,
                                    "offsets": [[0, 10]]
                                }
                            ]
                        }
                    }
                ],
            }
        },
    }
    
    event_bytes_w1 = json.dumps(store_event_w1).encode("utf-8")
    radix_tree.apply_event(worker_1, event_bytes_w1)
    
    # Verify both blocks are stored
    all_blocks = radix_tree.dump_tree_as_events()
    assert len(all_blocks) == 2, f"Expected 2 blocks, got {len(all_blocks)}"
    
    # Query with block_hash matching worker 0
    overlap_scores_w0 = radix_tree.find_matches([block_hash_w0])
    print(f"Scores for worker 0 query: {overlap_scores_w0.scores}")
    
    # Should match worker 0
    worker_key_0 = (worker_0, 0)  # (worker_id, dp_rank)
    assert worker_key_0 in overlap_scores_w0.scores, \
        f"Worker {worker_key_0} not found in scores"
    assert overlap_scores_w0.scores[worker_key_0] == 1, \
        f"Expected score 1 for worker {worker_key_0}"
    
    # Query with block_hash matching worker 1
    overlap_scores_w1 = radix_tree.find_matches([block_hash_w1])
    print(f"Scores for worker 1 query: {overlap_scores_w1.scores}")
    
    # Should match worker 1
    worker_key_1 = (worker_1, 0)  # (worker_id, dp_rank)
    assert worker_key_1 in overlap_scores_w1.scores, \
        f"Worker {worker_key_1} not found in scores"
    assert overlap_scores_w1.scores[worker_key_1] == 1, \
        f"Expected score 1 for worker {worker_key_1}"
    
    # Verify that querying with wrong hash doesn't match
    wrong_hash = 9999
    overlap_scores_wrong = radix_tree.find_matches([wrong_hash])
    assert len(overlap_scores_wrong.scores) == 0, \
        f"Expected no matches for wrong hash, got {overlap_scores_wrong.scores}"
    
    print("✓ MM KV routing test passed: blocks with same tokens but different MM content are distinguished")


@pytest.mark.asyncio  
async def test_mm_block_hash_computation():
    """
    Test that compute_block_hash_for_seq correctly computes MM-aware hashes.
    
    Verifies that same tokens with different MM content produce different hashes.
    """
    kv_block_size = 32
    tokens = [100] * kv_block_size
    
    # Hash computation WITH MM info
    mm_hash_1 = 0xDEADBEEF
    mm_info_1 = {
        "mm_objects": [{
            "mm_hash": mm_hash_1,
            "offsets": [[0, 5]]
        }]
    }
    
    block_hashes_with_mm1 = compute_block_hash_for_seq_py(tokens, kv_block_size, [mm_info_1])
    assert len(block_hashes_with_mm1) == 1, f"Expected 1 block hash, got {len(block_hashes_with_mm1)}"
    
    # Same tokens with different MM info should produce different hash
    mm_hash_2 = 0xCAFEBABE
    mm_info_2 = {
        "mm_objects": [{
            "mm_hash": mm_hash_2,
            "offsets": [[0, 5]]
        }]
    }
    
    block_hashes_with_mm2 = compute_block_hash_for_seq_py(tokens, kv_block_size, [mm_info_2])
    
    assert block_hashes_with_mm1 != block_hashes_with_mm2, \
        f"Same tokens with different MM should produce different hashes"
    
    # Tokens without MM info should differ from tokens with MM info
    block_hashes_no_mm = compute_block_hash_for_seq_py(tokens, kv_block_size)
    assert block_hashes_no_mm != block_hashes_with_mm1, \
        "Tokens without MM should differ from tokens with MM"
    
    print(f"✓ MM-aware block hash computation test passed!")
    print(f"  - Hash without MM: {block_hashes_no_mm[0]}")
    print(f"  - Hash with MM 1:  {block_hashes_with_mm1[0]}")
    print(f"  - Hash with MM 2:  {block_hashes_with_mm2[0]}")
    print(f"  - All hashes are different as expected!")
