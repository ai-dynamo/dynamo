# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the KVBM cache manager with vLLM.
"""

import asyncio

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


def new_request(request_id: str = "1"):
    return Request(
        request_id=request_id,
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
    request_1 = new_request("1")
    request_2 = new_request("2")
    request_3 = new_request("3")
    request_4 = new_request("4")

    (blocks, count) = block_manager.get_computed_blocks(request_1)
    assert len(blocks) == count
    assert count == 0

    blocks = block_manager.allocate_slots(request_1, 6)
    assert blocks is not None
    assert len(blocks.blocks) == 2, "ceil(6/4) = 2"

    blocks = block_manager.allocate_slots(request_2, 12)
    assert blocks is not None
    assert len(blocks.blocks) == 3, "ceil(12/4) = 3"

    block_ids = block_manager.get_block_ids(request_1.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [0, 1]

    block_ids = block_manager.get_block_ids(request_2.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [2, 3, 4]

    block_manager.free(request_1)
    block_ids = block_manager.get_block_ids(request_1.request_id)
    assert block_ids == [[]], "block_ids should be empty after freeing blocks"

    block_manager.free_block_hashes(request_1)
    with pytest.raises(Exception):
        # would raise Exception: slot not found
        block_ids = block_manager.get_block_ids(request_1.request_id)

    blocks = block_manager.allocate_slots(request_3, 18)
    assert blocks is not None
    assert len(blocks.blocks) == 5, "ceil(18/4) = 5"

    block_ids = block_manager.get_block_ids(request_3.request_id)
    assert len(block_ids) == 1
    assert block_ids[0] == [5, 6, 7, 8, 9]

    blocks = block_manager.allocate_slots(request_4, 6)
    assert blocks is not None
    assert len(blocks.blocks) == 2, "ceil(6/4) = 2"

    block_ids = block_manager.get_block_ids(request_4.request_id)
    assert len(block_ids) == 1
    print(f"block_ids: {block_ids}")
    assert block_ids[0] == [10, 11]


async def main():
    """
    Main function to run the test.
    """
    await test_kvbm(new_kv_cache_manager())


if __name__ == "__main__":
    asyncio.run(main())
