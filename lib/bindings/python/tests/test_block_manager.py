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
DTYPE, TORCH_DTYPE = "FP32", torch.float32
HOST_NUM_BLOCKS = 16
DEVICE_NUM_BLOCKS = 16
DEVICE_ID = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_manager_initialization():
    # Python should drop the BlockManager instance as soon as it goes out of scope, but
    # it may not be garbage collected immediately, depending on the garbage collector.
    BlockManager(WORKER_ID, NUM_LAYER, PAGE_SIZE, INNER_DIM)
    BlockManager(WORKER_ID, NUM_LAYER, PAGE_SIZE, INNER_DIM, DTYPE)
    BlockManager(WORKER_ID, NUM_LAYER, PAGE_SIZE, INNER_DIM, DTYPE, HOST_NUM_BLOCKS)
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        device_num_blocks=DEVICE_NUM_BLOCKS,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        device_num_blocks=DEVICE_NUM_BLOCKS,
        device_id=DEVICE_ID,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_cpu_block_access():
    block_manager = BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )
    block_count = 2
    block_list = block_manager.allocate_host_blocks_blocking(block_count)
    py_blocks = block_list.to_list()
    assert len(py_blocks) == block_count
    tensors = [torch.from_dlpack(b) for b in py_blocks]
    for tensor in tensors:
        assert tensor.get_device() == -1  # CPU
        assert tensor.shape == (1, NUM_LAYER, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0] = 1.0
        tensor[0][NUM_LAYER - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_gpu_block_access():
    block_manager = BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )
    block_count = 6
    block_list = block_manager.allocate_device_blocks_blocking(block_count)
    py_blocks = block_list.to_list()
    assert len(py_blocks) == block_count
    tensors = [torch.from_dlpack(b) for b in py_blocks]
    for tensor in tensors:
        assert tensor.get_device() == DEVICE_ID  # GPU
        assert tensor.shape == (1, NUM_LAYER, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0] = 1.0
        tensor[0][NUM_LAYER - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_list_iteration():
    block_manager = BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )
    block_count = 4
    block_list = block_manager.allocate_host_blocks_blocking(block_count)
    # Test __len__()
    assert len(block_list) == block_count
    # Test __getitem__()
    for i in range(block_count):
        block = block_list[i]
        tensor = torch.from_dlpack(block)
        tensor[0][0][0][0] = 1.0 + i
    # Test __iter__() and __next__()
    idx = 1.0
    for block in block_list:
        tensor = torch.from_dlpack(block)
        assert tensor[0][0][0][0] == idx
        tensor[0][0][0][0] += 0.5
        idx += 1.0
    assert idx == 1.0 + block_count
    # Test __iter__() should reset current index
    idx = 1.0
    for block in block_list:
        tensor = torch.from_dlpack(block)
        assert tensor[0][0][0][0] == idx + 0.5
        idx += 1.0
    assert idx == 1.0 + block_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_copy_g1_g2():
    block_manager = BlockManager(
        WORKER_ID,
        NUM_LAYER,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )
    # Allocate device (G1) and host (G2) block
    host_block_list = block_manager.allocate_host_blocks_blocking(1)
    device_block_list = block_manager.allocate_device_blocks_blocking(1)
    # Populate host block with unique values
    host_tensor = torch.from_dlpack(host_block_list[0])
    for i in range(NUM_LAYER):
        for j in range(PAGE_SIZE):
            for k in range(INNER_DIM):
                host_tensor[0][i][j][k] = i * PAGE_SIZE * INNER_DIM + j * INNER_DIM + k
    # Copy host block to device block after permuting
    permute_dims = (0, 2, 3, 1)
    device_tensor_ = torch.from_dlpack(device_block_list[0]).permute(*permute_dims)
    device_tensor_.copy_(host_tensor.permute(*permute_dims))
    # Assert device block is contiguous and updated in block manager
    device_tensor = torch.from_dlpack(device_block_list[0])
    for i in range(NUM_LAYER):
        for j in range(PAGE_SIZE):
            for k in range(INNER_DIM):
                assert (
                    device_tensor[0][i][j][k]
                    == i * PAGE_SIZE * INNER_DIM + j * INNER_DIM + k
                )
    # Set host block to zero and assert updated in block manager
    host_tensor_ = torch.from_dlpack(host_block_list[0]).permute(*permute_dims)
    host_tensor_.zero_()
    assert torch.all(host_tensor == 0)
    # Copy device block back to host block
    host_tensor_.copy_(device_tensor_)
    # Assert host block is updated in block manager
    for i in range(NUM_LAYER):
        for j in range(PAGE_SIZE):
            for k in range(INNER_DIM):
                assert (
                    host_tensor[0][i][j][k]
                    == i * PAGE_SIZE * INNER_DIM + j * INNER_DIM + k
                )


async def main():
    await test_block_manager_initialization()
    await test_cpu_block_access()
    await test_gpu_block_access()
    await test_block_list_iteration()
    await test_block_copy_g1_g2()


if __name__ == "__main__":
    asyncio.run(main())
