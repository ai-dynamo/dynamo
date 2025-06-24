# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the KVBM cache manager with vLLM.
"""

import asyncio
import uuid

import pytest
import torch
from vllm.v1.request import Request, SamplingParams

from dynamo.llm import BlockManager
from dynamo.llm.vllm_integration.kv_cache_manager import KvbmCacheManager

pytestmark = pytest.mark.pre_merge


WORKER_ID = 0
NUM_LAYER = 5
OUTER_DIM = 2
PAGE_SIZE = 4
INNER_DIM = 13
DTYPE, TORCH_DTYPE = "FP32", torch.float32
HOST_NUM_BLOCKS = 16
DEVICE_NUM_BLOCKS = 16
DEVICE_ID = 0


def new_request():
    return Request(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        multi_modal_inputs=[],
        multi_modal_hashes=[],
        multi_modal_placeholders=[],
        eos_token_id=0,
        arrival_time=0.0,
        cache_salt="test",
        lora_request=None,
        sampling_params=SamplingParams(n=1),
    )


def new_kv_cache_manager():
    """
    Creates a new KVBM cache manager.

    Returns:
        KvbmCacheManager: The KVBM cache manager.
    """

    try:
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
    except Exception as e:
        print(f"Failed to create KvbmCacheManager: {e}")
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_kvbm(block_manager: KvbmCacheManager):
    """
    Tests the KVBM kv_cache_manager APIs.

    Args:
        block_manager: The KVBM cache manager.
    """
    request_1 = new_request()
    request_2 = new_request()
    request_3 = new_request()

    # test get_computed_blocks
    (blocks, count) = block_manager.get_computed_blocks(request_1)
    assert len(blocks) == count
    assert count == 0

    # test allocate_slots
    blocks = block_manager.allocate_slots(request_1, 6)
    assert blocks is not None
    assert len(blocks.blocks) == 2, "ceil(6/4) = 2"

    blocks = block_manager.allocate_slots(request_2, 12)
    assert blocks is not None
    assert len(blocks.blocks) == 3, "ceil(12/4) = 3"

    # test get_block_ids
    block_ids = block_manager.get_block_ids(request_1.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [0, 1]

    block_ids = block_manager.get_block_ids(request_2.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [2, 3, 4]

    # test free
    block_manager.free(request_1)
    block_ids = block_manager.get_block_ids(request_1.request_id)
    assert block_ids == [[]], "block_ids should be empty after freeing blocks"

    # test free_block_hashes
    block_manager.free_block_hashes(request_1)
    with pytest.raises(Exception):
        # would raise Exception: slot not found
        block_ids = block_manager.get_block_ids(request_1.request_id)

    # test allocate_slots again after freeing blocks
    # new blocks should not be allocated to [0, 1] even though they are free
    blocks = block_manager.allocate_slots(request_3, 6)
    assert blocks is not None
    assert len(blocks.blocks) == 2, "ceil(6/4) = 2"

    block_ids = block_manager.get_block_ids(request_3.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [5, 6]


async def main():
    """
    Main function to run the test.
    """
    await test_kvbm(new_kv_cache_manager())


if __name__ == "__main__":
    asyncio.run(main())
