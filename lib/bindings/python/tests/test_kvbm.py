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


def new_request():
    return Request(
        request_id="1",
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
async def test_kvbm_get_computed_blocks(
    block_manager: KvbmCacheManager,
):
    """
    Tests the KVBM cache manager's get_computed_blocks method.

    Args:
        block_manager: The KVBM cache manager.
    """

    request = new_request()
    (blocks, count) = block_manager.get_computed_blocks(request)
    print(f"number of matched blocks: {count}")
    assert len(blocks) == count
    assert count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_kvbm_allocate_slots(
    block_manager: KvbmCacheManager,
):
    """
    Tests the KVBM cache manager's get_computed_blocks method.

    Args:
        block_manager: The KVBM cache manager.
    """
    request = new_request()
    blocks = block_manager.allocate_slots(request, 18)
    assert blocks is not None
    assert len(blocks.blocks) == 5


async def main():
    """
    Main function to run the test.
    """
    await test_kvbm_get_computed_blocks(new_kv_cache_manager())
    await test_kvbm_allocate_slots(new_kv_cache_manager())


if __name__ == "__main__":
    asyncio.run(main())
