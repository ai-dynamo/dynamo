# SPDX-License-Identifier: Apache-2.0
"""Compare the with and without prefix caching."""

from typing import Optional

import pytest
import torch
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import hash_block_tokens
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
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


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
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


@pytest.mark.parametrize("hash_algo", ["sha256", "hash"])
def test_prefill(hash_algo):
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
        caching_hash_algo=hash_algo,
    )

    # choose the hash function according to the parameter
    hash_fn = sha256 if hash_algo == "sha256" else hash

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert blocks.get_block_ids() == [[1, 2, 3, 4]]

    # Check full block metadata
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16 : block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value

    # Check partial block metadata
    for block_id in (4,):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    print(f"computed_blocks: {computed_blocks}")
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert blocks.get_block_ids() == [[5]]
    for block in computed_blocks.blocks:
        assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

    # All blocks should be available.
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (6, 7, 8, 9, 10)]
    # [unique_req0 (4)]
    # [unique_req1 (5)]
    # [common (3, 2, 1)]
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]

    # Cache hit in the common prefix when the original block is already free.
    # Incomplete 1 block (6 tokens)
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 3
    assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req2, num_new_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    assert blocks.get_block_ids() == [[6]]

    # Although we only have 6 free blocks, we have 8 blocks in
    # the free block queue due to lazy removal.
    assert manager.block_pool.free_block_queue.num_free_blocks == 6
    assert all(
        [
            b.ref_cnt == 0
            for b in manager.block_pool.free_block_queue.get_all_free_blocks()
        ]
    )
    assert (
        len([b for b in manager.block_pool.free_block_queue.get_all_free_blocks()]) == 6
    )

    manager.free(req2)

    # Cache miss and eviction.
    req3 = make_request("3", [99] * (16 * 10))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req3, 16 * 10, len(computed_blocks.blocks) * 16, computed_blocks
    )
    # This block ID order also checks the eviction order.
    assert blocks.get_block_ids() == [[7, 8, 9, 10, 4, 5, 6, 3, 2, 1]]
    assert manager.block_pool.free_block_queue.num_free_blocks == 0
    assert manager.block_pool.free_block_queue.free_list_head is None
    assert manager.block_pool.free_block_queue.free_list_tail is None


def test_kvbm_prefill():
    """
    set up
    """
    from dynamo.llm import BlockManager
    from dynamo.llm.vllm_integration.kv_cache_manager import KvbmCacheManager

    WORKER_ID = 0
    NUM_LAYER = 1
    OUTER_DIM = 1
    PAGE_SIZE = 16
    INNER_DIM = 1
    # DTYPE, TORCH_DTYPE = "FP32", torch.float32
    DTYPE = "FP32"
    HOST_NUM_BLOCKS = 11
    DEVICE_NUM_BLOCKS = 11
    DEVICE_ID = 0

    def new_kv_cache_manager():
        """
        Creates a new KVBM cache manager.

        Returns:
            KvbmCacheManager: The KVBM cache manager.
        """
        return KvbmCacheManager(
            BlockManager(
                WORKER_ID,
                NUM_LAYER,
                OUTER_DIM,
                PAGE_SIZE,
                INNER_DIM,
                DTYPE,
                HOST_NUM_BLOCKS,
                DEVICE_NUM_BLOCKS,
                DEVICE_ID,
            )
        )

    """ Test start  """
    manager = new_kv_cache_manager()

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    ## TODO: KVBM lacks `req_to_block_hashes``
    # assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    print(computed_blocks)

    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks) * 16, computed_blocks
    )
    # TODO(oandreeva): this check is irrelevant for KVBM
    # assert blocks.get_block_ids() == [[1, 2, 3, 4]]
    print("=>>>>>>>>>>>>>>>>>>>>>blocks", blocks)
    print(manager.get_block_ids(req0.request_id))
    assert len(manager.get_block_ids(req0.request_id)[0]) == len([1, 2, 3, 4])

    """
    TODO(oandreeva): `block_pool` is not a field of KVBM

    # Check full block metadata
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
    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    print("len(computed_blocks)", len(computed_blocks))
    # assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    # assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks) * 16, computed_blocks
    )
    # assert blocks.get_block_ids() == [[5]]
    for block in computed_blocks.blocks:
        assert block.ref_cnt == 2

    """


    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens,
                                    len(computed_blocks.blocks) * 16,
                                    computed_blocks)
    assert blocks.get_block_ids() == [[5]]
    for block in computed_blocks.blocks:
        assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

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

    # Cache hit in the common prefix when the original block is already free.
    # Incomplete 1 block (6 tokens)
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 3
    assert computed_blocks.get_block_ids() == [[1, 2, 3]]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req2, num_new_tokens,
                                    len(computed_blocks.blocks) * 16,
                                    computed_blocks)
    assert blocks.get_block_ids() == [[6]]

    # Although we only have 6 free blocks, we have 8 blocks in
    # the free block queue due to lazy removal.
    assert manager.block_pool.free_block_queue.num_free_blocks == 6
    assert all([
        b.ref_cnt == 0
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ])
    assert len([
        b for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]) == 6

    manager.free(req2)

    # Cache miss and eviction.
    req3 = make_request("3", [99] * (16 * 10))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req3, 16 * 10,
                                    len(computed_blocks.blocks) * 16,
                                    computed_blocks)
    # This block ID order also checks the eviction order.
    assert blocks.get_block_ids() == [[7, 8, 9, 10, 4, 5, 6, 3, 2, 1]]
    assert manager.block_pool.free_block_queue.num_free_blocks == 0
    assert manager.block_pool.free_block_queue.free_list_head is None
    assert manager.block_pool.free_block_queue.free_list_tail is None
    """
