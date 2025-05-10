# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio

import pytest
import torch

from dynamo.llm import BlockManager

pytestmark = pytest.mark.pre_merge


WORKER_ID = 0
NUM_LAYER = 5
PAGE_SIZE = 4
INNER_DIM = 13
HOST_NUM_BLOCKS = 16
DEVICE_NUM_BLOCKS = 16


async def test_tensor_access():
    block_manager = BlockManager(
        WORKER_ID, NUM_LAYER, PAGE_SIZE, INNER_DIM, HOST_NUM_BLOCKS, DEVICE_NUM_BLOCKS
    )
    block_list = block_manager.allocate_blocks(2)
    py_blocks = block_list.to_list()
    assert len(py_blocks) == 2
    tensors = [torch.from_dlpack(b) for b in py_blocks]
    for tensor in tensors:
        assert tensor.get_device() == -1  # CPU
        assert tensor.shape == (HOST_NUM_BLOCKS, NUM_LAYER, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == torch.float16  # DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0] = 1.0
        tensor[HOST_NUM_BLOCKS - 1][NUM_LAYER - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensors)
    py_blocks_ = block_list.to_list()
    assert py_blocks is not py_blocks_
    assert len(py_blocks) == len(py_blocks_)
    tensors_ = [torch.from_dlpack(b) for b in py_blocks_]
    for tensor, tensor_ in zip(tensors, tensors_):
        assert tensor is not tensor_
        assert tensor.shape == tensor_.shape
        assert tensor.dtype == tensor_.dtype
        assert torch.allclose(tensor, tensor_)


async def main():
    await test_tensor_access()


if __name__ == "__main__":
    asyncio.run(main())
