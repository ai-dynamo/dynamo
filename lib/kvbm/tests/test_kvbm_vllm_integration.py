# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
except ImportError:
    pass

try:
    from vllm.multimodal.inputs import MultiModalKwargs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import Request
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    VLLM_NOT_AVAILABLE = False
except ImportError:
    VLLM_NOT_AVAILABLE = True

try:
    from kvbm import BlockManager
    from kvbm.vllm_integration.kv_cache_manager import KvbmCacheManager

    KVBM_NOT_AVAILABLE = False
except ImportError:
    KVBM_NOT_AVAILABLE = True


def new_kv_cache_manager(num_blocks: int = 11, page_size: int = 16):
    """
    Creates a new KVBM cache manager.

    Returns:
        KvbmCacheManager: The KVBM cache manager.
    """

    return KvbmCacheManager(
        BlockManager(
            worker_id=0,
            leader=None,
            page_size=page_size,
            device_num_blocks=num_blocks,
        )
    )


def make_request(
    request_id,
    prompt_token_ids,
    mm_positions=None,
    mm_hashes=None,
    prompt_logprobs: Optional[int] = None,
    cache_salt: Optional[str] = None,
):
    if mm_positions is None:
        multi_modal_inputs = None
    else:
        multi_modal_inputs = [MultiModalKwargs({})] * len(mm_positions)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        multi_modal_inputs=multi_modal_inputs,
        multi_modal_hashes=mm_hashes,
        multi_modal_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17, prompt_logprobs=prompt_logprobs),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
        cache_salt=cache_salt,
    )


def make_kv_cache_config(block_size: int, num_blocks: int) -> "KVCacheConfig":
    return KVCacheConfig(
        num_blocks=num_blocks,
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size, 1, 1, torch.float32, False),
            )
        ],
    )


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_prefill():
    """
    Tests the KvbmCacheManager's prefill functionality.
    """
    manager = new_kv_cache_manager()

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids)

    # Step 1: Initial allocation - no computed blocks yet
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0

    # Step 2: Allocate slots for the request
    blocks_req0 = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )

    for block in blocks_req0.blocks:
        assert block._block_hash is None

    # Verify allocation was successful
    block_ids = manager.get_block_ids(req0.request_id)
    assert len(block_ids) == 1  # One sequence in the request
    assert len(block_ids[0]) == 4  # 4 blocks allocated (3 complete + 1 partial)

    # Step 3: Simulate model execution by updating the request's computed tokens
    req0.append_output_token_ids(100)
    req0.num_computed_tokens = 55

    _ = manager.allocate_slots(req0, num_new_tokens=1)

    # Step 5: Create a new request with the same prefix plus one token
    unique_token_ids = [3] * 4
    req1 = make_request("1", common_token_ids + unique_token_ids)

    # Step 8: Check for computed blocks - should find the common prefix
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks) == 3
    assert num_computed_tokens == len(computed_blocks.blocks) * 16

    for block in computed_blocks.blocks:
        assert block._block_hash is not None

    # Clean up
    del computed_blocks

    manager.free_block_hashes(req0)

    manager.free_block_hashes(req1)

    # Cache miss and eviction.
    req3 = make_request("3", [24] * (16 * 11))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks_req3 = manager.allocate_slots(
        req3, 16 * 11, len(computed_blocks.blocks) * 16, computed_blocks
    )

    assert len(blocks_req3.blocks) == 11
    for block, expected_block_id in zip(
        blocks_req3.blocks, [4, 5, 6, 7, 8, 9, 10, 3, 2, 1, 0]
    ):
        assert block._block_hash is None
        assert block.block_id == expected_block_id


@pytest.mark.skip(reason="KVBM needs to support reset_prefix_cache")
def test_prefill_plp():
    """Test prefill with APC and some prompt logprobs (plp) requests.

    1. Schedule plp request and validate APC block allocation
    2. Schedule non-plp request and validate blocks
    3. Schedule plp request; no hit should occur; validate blocks
    """
    manager = new_kv_cache_manager()

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Request #0 is a prompt logprobs request
    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    # assert len(manager.req_to_block_hashes[req0.request_id]) == 0
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )

    # assert blocks.get_block_ids() == [[1, 2, 3, 4]]
    assert blocks.get_block_ids() == [[0, 1, 2, 3]]
    req0_block_hashes = [b.block_hash for b in blocks.blocks]

    # Step 3: Simulate model execution by updating the request's computed tokens
    req0.append_output_token_ids(100)
    req0.num_computed_tokens = 55

    _ = manager.allocate_slots(req0, num_new_tokens=1)

    # Check full block metadata
    """
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16:block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash,
                                       block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value

    # Check partial block metadata
    for block_id in (4, ):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
    """

    # Request #1 is a non-prompt-logprobs request:
    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    # assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    assert computed_blocks.get_block_ids() == [[0, 1, 2]]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    # assert blocks.get_block_ids() == [[5]]
    assert blocks.get_block_ids() == [[4]]
    # for block in computed_blocks.blocks:
    #   assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    # assert manager.block_pool.free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

    """
    # All blocks should be available.
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (6, 7, 8, 9, 10)]
    # [unique_req0 (4)]
    # [unique_req1 (5)]
    # [common (3, 2, 1)]
    assert [
        b.block_id
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]
    """

    # Request #2 is a prompt-logprobs request:
    # NO cache hit in the common prefix; duplicates request #0 cached blocks
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    # assert len(manager.req_to_block_hashes[req2.request_id]) == 0
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req2, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )
    block_ids = blocks.get_block_ids()
    # Duplicate cached blocks have different ids but same hashes vs request #0
    assert [b.block_hash for b in blocks.blocks] == req0_block_hashes
    assert block_ids != [[1, 2, 3, 4]]

    # Request #2 block hashes are valid since request #0 hashes are.
    # Check block reference counts.
    for block_id in block_ids[0]:
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    manager.free(req2)


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_decode():
    manager = new_kv_cache_manager()

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    req0 = make_request("0", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )
    # assert blocks.get_block_ids() == [[1, 2, 3, 4]]
    assert blocks.get_block_ids() == [[0, 1, 2, 3]]
    # Append slots without allocating a new block.
    req0.num_computed_tokens = 55
    for _ in range(4):
        req0.append_output_token_ids(8)

    new_blocks = manager.allocate_slots(
        req0, 4, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks) == 0

    # NOTE(): There's no way to access the current active non-registered block
    # from the python bindings.
    # assert manager.single_type_manager.req_to_blocks[
    #    req0.request_id][-1].block_hash is None

    # Append slots with allocating a new block.
    req0.num_computed_tokens = 59
    # 9 tokens to fill the previous block, and 10 tokens to fill
    # the preallocated block.
    for _ in range(9 + 10):
        req0.append_output_token_ids(7)

    print(len(computed_blocks.blocks))
    new_blocks = manager.allocate_slots(
        req0, 19, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks) == 1
    assert new_blocks.blocks[-1].block_hash is None

    req0.num_computed_tokens = 78
    req0.append_output_token_ids(100)

    # The following is required for KVBM to register the block with id=3
    _ = manager.allocate_slots(
        req0, 1, len(computed_blocks.blocks) * 16, computed_blocks
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    # assert manager.single_type_manager.req_to_blocks[
    #    req0.request_id][-2].block_hash is not None
    # assert manager.single_type_manager.req_to_blocks[
    #    req0.request_id][-1].block_hash is None
    assert computed_blocks.blocks[-1].block_id == 3
    assert computed_blocks.blocks[-1].block_hash is not None

    # Clean up
    manager.free_block_hashes(req0)


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_evict():
    manager = new_kv_cache_manager()
    used_blocks = set()

    last_token_id = 5 * 16 + 7
    req0 = make_request("0", list(range(last_token_id)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 5 * 16 + 7, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert len(blocks.blocks) == 6  # 5 full + 1 partial
    used_blocks.update(blocks.get_block_ids()[0])

    req0.append_output_token_ids(100)
    req0.num_computed_tokens = 5 * 16 + 7
    manager.allocate_slots(req0, 1, len(computed_blocks.blocks) * 16, computed_blocks)

    req1 = make_request("1", list(range(last_token_id, last_token_id + 3 * 16 - 1)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req1, 3 * 16, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert (
        len(blocks.blocks) == 3
    )  # 2 full blocks and 1 partial (15 tokens) 1 more will be added during allocate_slots
    last_token_id += 3 * 16 - 1
    used_blocks.update(blocks.get_block_ids()[0])

    # 10 - (6 + 3) == 1
    assert len(used_blocks) == 6 + 3

    req1.append_output_token_ids(100)
    req1.num_computed_tokens = 3 * 16 - 1
    blocks = manager.allocate_slots(
        req1, 1, len(computed_blocks.blocks) * 16, computed_blocks
    )

    manager.free(req0)
    manager.free(req1)
    # Can't access the free blocks queue from the python bindings.
    # assert manager.block_pool.free_block_queue.num_free_blocks == 10
    # assert [
    #     b.block_id
    #     for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    # ] == [10, 6, 5, 4, 3, 2, 1, 9, 8, 7]

    # Touch the first 2 blocks.
    req2 = make_request("2", list(range(2 * 16 + 3)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    # assert computed_blocks.get_block_ids() == [[1, 2]]
    assert computed_blocks.get_block_ids() == [[0, 1]]
    assert num_computed_tokens == 2 * 16
    blocks = manager.allocate_slots(
        req2, 3, len(computed_blocks.blocks) * 16, computed_blocks
    )

    assert blocks.get_block_ids() == [[9]]
    # Can't access the free blocks queue from the python bindings.
    # assert manager.block_pool.free_block_queue.num_free_blocks == 7


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_hash_block_correct_reuse():
    """
    This tests when a previously cached block is reused as a new block,
    its hash metadata should be correctly reset.
    """
    block_size = 16
    manager = new_kv_cache_manager(num_blocks=2)

    # Allocate 1 block and cache it.
    num_tokens = block_size
    req = make_request("0", list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req, num_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert len(blocks.blocks) == 1
    for t in range(5):
        req.append_output_token_ids(100)
    req.num_computed_tokens = num_tokens
    blocks = manager.allocate_slots(
        req, 5, len(computed_blocks.blocks) * 16, computed_blocks
    )

    computed_blocks, _ = manager.get_computed_blocks(req)
    assert computed_blocks.blocks[0].block_hash is not None
    assert computed_blocks.blocks[0].block_id == 0

    # Deallocate the block.
    del computed_blocks
    manager.free(req)

    # Allocate new blocks, last one is partial not full, make sure hash info on the
    # blocks are cleared.
    # KVBM will allocate block 1 first, then block 0. Need to verify,
    # that block's 0 hash is cleared
    req = make_request("1", list(range(256, 256 + 2 * num_tokens - 1)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req, 2 * num_tokens - 1, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert len(blocks.blocks) == 2

    assert blocks.blocks[1].block_id == 0
    assert blocks.blocks[1].block_hash is None


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_computed_blocks_not_evicted():
    """
    Test that the computed blocks are not evicted when getting new blocks
    for a request if there are any other free blocks.
    """
    block_size = 16
    manager = new_kv_cache_manager(num_blocks=3)

    # Allocate a block and cache it.
    num_tokens = block_size * 1
    req0 = make_request("0", list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, num_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert len(blocks.blocks) == 1
    # assert blocks.blocks[0].block_id == 1
    assert blocks.blocks[0].block_id == 0

    # Allocate another block.
    req1 = make_request("1", list(range(num_tokens, num_tokens * 2)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req1, num_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert len(blocks.blocks) == 1
    # assert blocks.blocks[0].block_id == 2
    assert blocks.blocks[0].block_id == 1

    # Need to simulate the forward pass to get blocks registered
    req0.append_output_token_ids(100)
    req0.num_computed_tokens = num_tokens
    _ = manager.allocate_slots(
        req0, 1, len(computed_blocks.blocks) * 16, computed_blocks
    )

    req1.append_output_token_ids(100)
    req1.num_computed_tokens = num_tokens
    _ = manager.allocate_slots(
        req1, 1, len(computed_blocks.blocks) * 16, computed_blocks
    )

    # Free the blocks.
    manager.free(req0)
    manager.free(req1)
    del computed_blocks

    # Now if we have a cache hit on the block_id 0, we should evict the block_id 1
    # cached block rather than the first one.
    req2 = make_request("2", list(range(num_tokens * 3)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks) == 1
    # assert computed_blocks.blocks[0].block_id == 1
    assert computed_blocks.blocks[0].block_id == 0
    assert num_computed_tokens == block_size

    # Allocate should return a free block with id 2 first, and then block with id 1
    # which was evicted.
    blocks = manager.allocate_slots(
        req2,
        num_tokens * 3 - num_computed_tokens,
        len(computed_blocks.blocks) * 16,
        computed_blocks,
    )
    assert len(blocks.blocks) == 2
    assert blocks.blocks[0].block_id == 2
    assert blocks.blocks[1].block_id == 1


def _test_basic_prefix_caching_disabled():
    """
    Currently, KVBM does not support `enable_caching` or setting it to False to disable prefix caching.
    """
    pass


# @pytest.mark.parametrize("hash_fn", [sha256, hash])
def _test_cache_blocks(hash_fn):
    """
    Hashing is done by KVBM and tested by the core library.
    """
    pass


def _test_mm_prefix_caching():
    """
    KVBM currently does not support multi-modal prefix caching.
    This tests that the multi-modal prefix caching is correct.
    """
    pass


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_cache_key_salting():
    """
    This tests that cache salts are applied during hashing and the cache
    is separated cache as expected.

    The test is mostly the same as the one for vLLM's native KV cache manager.
    The only difference is for KVBM we don't need a `BlockHashType` object on python
    side, thus we don't check the value of the salt. We test the salt-ing
    functionality by validating cache miss and cache hit with different salts.
    """
    block_size = 16
    manager = new_kv_cache_manager()

    # 3 complete blocks and an incomplete block with 11 tokens.
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    token_ids = common_token_ids + [3] * 11
    req0 = make_request("0", token_ids, cache_salt="salt1")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    # Completed block should have hashes with extra keys.
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    """
    block_hashes = manager.req_to_block_hashes[req0.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ("salt1", )
    assert block_hashes[1].extra_keys is None
    assert block_hashes[2].extra_keys is None
    """

    blocks = manager.allocate_slots(
        req0, 59, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert blocks.get_block_ids() == [[0, 1, 2, 3]]  # [[1, 2, 3, 4]]
    req0.num_computed_tokens = 59

    # Append slots without allocating a new block.
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(
        req0, 5, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks) == 0
    print(new_blocks)
    """
    # Now one more block that should not have extra keys.
    assert len(block_hashes) == 4
    assert block_hashes[3].extra_keys is None
    """
    # Test cache hit with a new request that has the same salt.
    token_ids = common_token_ids + [4] * 11
    req1 = make_request("1", token_ids, cache_salt="salt1")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # Should match only a prefix of 3 blocks.
    assert len(computed_blocks.blocks) == 3
    assert num_computed_tokens == 3 * block_size

    # Test cache miss with same content but different salt.
    token_ids = common_token_ids + [4] * 11
    req2 = make_request("2", token_ids, cache_salt="salt2")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks) == 0
    assert num_computed_tokens == 0
    """
    block_hashes = manager.req_to_block_hashes[req2.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ("salt2", )
    """


@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
def test_prefill_not_enough_free_blocks_with_computed_blocks():
    """
    This is a unit test that tests the correctness of the allocate_slots
    when there is not enough free blocks. Specifically, when a request
    has computed blocks but cannot be allocated due to not enough free blocks,
    the computed blocks should not be touched.
    """
    block_size = 16
    manager = new_kv_cache_manager()

    # Complete 3 blocks (48 tokens)
    # | Common-0 | Common-1 | Common-2 | ... |
    common_token_ids = [i for i in range(3) for _ in range(16)]
    req0 = make_request("0", common_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    manager.allocate_slots(req0, 48, len(computed_blocks.blocks) * 16, computed_blocks)
    # block_part0 = manager.single_type_manager.req_to_blocks[req0.request_id]
    block_part0 = len(manager.get_block_ids(req0.request_id)[0])

    # Simulate model execution by updating the request's computed tokens
    req0.append_output_token_ids(100)
    req0.num_computed_tokens = 48
    _ = manager.allocate_slots(req0, num_new_tokens=1)

    # | Common-0 | Common-1 | Common-2 | Req1-3 | Req1-4 | Req1-5 | ... |
    req1 = make_request("1", common_token_ids * 2)  # Double the common tokens
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert (
        len(computed_blocks.blocks) == block_part0
    )  # First 3 blocks are computed from req0
    assert num_computed_tokens == 3 * 16  # 3 blocks * 16 tokens per block
    manager.allocate_slots(req1, 48, num_computed_tokens, computed_blocks)
    # block_part1 = manager.single_type_manager.req_to_blocks[req1.request_id]
    block_part1 = len(manager.get_block_ids(req1.request_id)[0])

    # Simulate forward pass for req1 to compute all 6 blocks
    req1.append_output_token_ids(100)
    req1.num_computed_tokens = 96
    _ = manager.allocate_slots(req1, num_new_tokens=1)

    # Free req1 to make its blocks available
    del computed_blocks
    manager.free(req1)

    # | Common-0 | Common-1 | Common-2 | Req1-3 (F) | Req1-4 (F) |
    # | Req1-5(F)| Req2-0   | Req2-1   | ... |
    req2 = make_request("2", [7] * block_size * 2)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    manager.allocate_slots(
        req2, block_size * 2, len(computed_blocks.blocks) * 16, computed_blocks
    )

    # Req3 is Req2 + 6 new blocks, so the first 6 blocks are computed,
    # but it cannot be allocated due to insufficient free blocks (2).
    # In this case, the ref_cnt of the computed blocks should not be changed.
    req3 = make_request("3", common_token_ids * 3)  # Use same tokens as req1
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)

    assert len(computed_blocks.blocks) == block_part1  # Should find 6 computed blocks
    assert num_computed_tokens == 6 * 16  # 6 blocks * 16 tokens per block

    # Req3 cannot be allocated due to insufficient free blocks
    # DYN LOG print:
    # DEBUG dynamo_llm::block_manager::pool::state: not enough blocks available, requested: 3, available: 2
    assert (
        manager.allocate_slots(
            req3, 48, len(computed_blocks.blocks) * 16, computed_blocks
        )
        is None
    )

    # Clean up
    manager.free_block_hashes(req0)
    manager.free_block_hashes(req2)
    manager.free_block_hashes(req3)


def _test_reset_prefix_cache():
    """
    `reset_prefix_cache` is currently not implemented.
    It returns False every time it is called
    """
    pass


def _test_prefix_cache_stats_disabled():
    """
    `reset_prefix_cache` is currently not implemented.
    It returns False every time it is called
    """
    pass


# @pytest.mark.parametrize("blocks_to_cache", [2, 3, 10])
def _test_kv_cache_events(blocks_to_cache: int):
    """
    KVBM's Event Manager is responsible for emitting events.
    Currently tested separately as a part of dynamo integration tests.
    """
    pass


def _test_eagle_enabled_removes_last_block():
    """NOTE: KVBM does not support spec decoding at the moment.
    Verify Eagle does NOT remove blocks when request
    length is divisible by block size."""
    pass


def _test_eagle_with_partial_blocks():
    """NOTE: KVBM does not support spec decoding at the moment.
    Test Eagle behavior with requests containing partial blocks."""
    pass


def _test_eagle_with_sliding_window():
    """NOTE: KVBM does not support spec decoding at the moment.
    Test Eagle behavior with sliding window."""
    pass


@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
def test_kvbm_wrong_blocks_provided():
    """
    Tests that providing wrong blocks to allocate_slots results in an error.
    Specifically, we test that using blocks from one request for another request
    with different tokens should fail.
    """
    manager = new_kv_cache_manager()

    # Create two requests with different token patterns
    req0 = make_request("0", [i for i in range(48)])  # 3 blocks of sequential tokens
    req1 = make_request("1", [i * 2 for i in range(48)])  # 3 blocks of even tokens

    # Allocate and compute blocks for req0
    computed_blocks_req0, _ = manager.get_computed_blocks(req0)
    _ = manager.allocate_slots(req0, 48, 0, computed_blocks_req0)

    # Simulate forward pass
    req0.append_output_token_ids(100)  # Add output token
    req0.num_computed_tokens = 48  # Mark all input tokens as computed
    _ = manager.allocate_slots(req0, num_new_tokens=1)  # Allocate slot for output token

    # Try to use req0's blocks for req1 - this should fail
    with pytest.raises(Exception) as exc_info:
        manager.allocate_slots(req1, 48, 48, computed_blocks_req0)
    assert (
        "slot error: Insufficient capacity: need 48 tokens but only 0 available in mutable blocks"
        in str(exc_info.value)
    )

    # Get computed blocks after forward pass
    computed_blocks_req0, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(computed_blocks_req0.blocks) == 3  # Should have 3 complete blocks
    assert num_computed_tokens == 48  # All input tokens should be computed

    # Try to use req0's blocks for req1 - this should fail
    with pytest.raises(Exception) as exc_info:
        manager.allocate_slots(req1, 48, 48, computed_blocks_req0)
    assert "slot error: computed block sequence hash mismatch" in str(exc_info.value)

    # Clean up
    manager.free_block_hashes(req0)
    manager.free_block_hashes(req1)


@pytest.mark.skipif(KVBM_NOT_AVAILABLE, reason="KVBM not available")
@pytest.mark.skipif(VLLM_NOT_AVAILABLE, reason="VLLM not available")
@patch("kvbm.vllm_integration.kv_cache_manager.KvbmCacheManager")
def test_kvbm_new_matched_tokens_edge_case(MockCacheManager):
    PAGE_SIZE = 4
    NUM_BLOCKS = 3
    SEQ_LEN = PAGE_SIZE * NUM_BLOCKS

    def create_list_mock(num_blocks: Optional[int]):
        if num_blocks is None:
            return None

        mock_list = MagicMock()
        mock_list.block_count.return_value = num_blocks
        mock_list.__len__.return_value = num_blocks
        return mock_list

    def create_mock(num_host_blocks: Optional[int], num_disk_blocks: Optional[int]):
        mock_instance = MagicMock()

        mock_instance.block_size = PAGE_SIZE

        mock_instance._create_slot.return_value = [0, 1, 2]

        host = create_list_mock(num_host_blocks)
        disk = create_list_mock(num_disk_blocks)

        mock_instance.cache_manager.get_num_offloaded_computed_blocks.return_value = (
            host,
            disk,
        )

        return mock_instance

    def get_pending_entry(mock, request_id):
        (id, entry) = mock.pending_onboard_blocks.__setitem__.call_args[0]
        assert id == request_id
        return entry

    def test_case(
        num_host_blocks: Optional[int],
        num_disk_blocks: Optional[int],
        expected_num_external_computed_tokens: int,
    ):
        request = make_request("0", [0] * SEQ_LEN)
        mock = create_mock(num_host_blocks, num_disk_blocks)
        (
            num_external_computed_tokens,
            async_load,
        ) = KvbmCacheManager.get_num_new_matched_tokens(mock, request, 0)
        assert num_external_computed_tokens == expected_num_external_computed_tokens
        assert not async_load

        entry = get_pending_entry(mock, request.request_id)

        assert (
            entry[0] is None
            if num_host_blocks is None
            else len(entry[0]) == num_host_blocks
        )
        assert (
            entry[1] is None
            if num_disk_blocks is None
            else len(entry[1]) == num_disk_blocks
        )

    # Case 1: Some blocks on host, no blocks on disk
    test_case(2, None, 2 * PAGE_SIZE)

    # Case 2: No blocks on host, some blocks on disk
    test_case(None, 2, 2 * PAGE_SIZE)

    # Case 3: All blocks on host.
    test_case(3, None, SEQ_LEN - 1)

    # Case 4: All blocks on disk.
    test_case(None, 3, SEQ_LEN - 1)
