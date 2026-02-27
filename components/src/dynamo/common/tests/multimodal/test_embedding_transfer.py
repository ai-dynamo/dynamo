# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AsyncEncoderCache."""

import asyncio
import logging
import time
from random import randint

import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    NixlPersistentEmbeddingReceiver,
    NixlPersistentEmbeddingSender,
    NixlWriteEmbeddingReceiver,
    NixlWriteEmbeddingSender,
    RingBuffer,
)

logger = logging.getLogger(__name__)

EMBEDDING_SIZE = 8 * 1024


async def benchmark(sender, receiver, tensors=None, from_cuda=False):
    if tensors is None:
        tensors = [
            torch.randn(256, EMBEDDING_SIZE, device="cuda" if from_cuda else "cpu")
            for _ in range(30)
        ]

    # warmup
    request, send_future = await sender.send_embeddings(tensors[0])
    tensor_id, response = await receiver.receive_embeddings(request)
    receiver.release_tensor(tensor_id)
    await send_future

    # benchmark
    send_start = time.perf_counter()
    sender_tasks = [
        asyncio.create_task(sender.send_embeddings(tensor, stage_embeddings=True))
        for tensor in tensors
    ]
    requests = await asyncio.gather(*sender_tasks)
    send_end = time.perf_counter()
    logger.info(f"Total send time for 30 tensors: {send_end - send_start:.2f} seconds")
    receive_start = time.perf_counter()
    receive_tasks = [
        asyncio.create_task(receiver.receive_embeddings(request[0]))
        for request in requests
    ]

    responses = await asyncio.gather(*receive_tasks)
    receive_end = time.perf_counter()
    logger.info(
        f"Total receive time for 30 tensors: {receive_end - receive_start:.2f} seconds"
    )
    for tensor, request, response in zip(tensors, requests, responses):
        tensor_id, received_tensor = response
        assert torch.equal(received_tensor, tensor.cpu())
        receiver.release_tensor(tensor_id)
        await request[1]


async def correctness(sender, receiver, tensors=None):
    if tensors is None:
        tensors = [torch.randn(256, 8 * 1024) for _ in range(3)]
    sender_tasks = [
        asyncio.create_task(sender.send_embeddings(tensor)) for tensor in tensors
    ]
    requests = await asyncio.gather(*sender_tasks)
    for idx, request in enumerate(requests):
        tensor_id, received_tensor = await receiver.receive_embeddings(request[0])
        assert torch.equal(received_tensor, tensors[idx])
        receiver.release_tensor(tensor_id)
        await request[1]


class TestLocalEmbeddingTransfer:
    @pytest.mark.asyncio
    @pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
    async def test_correctness(self):
        sender = LocalEmbeddingSender()
        receiver = LocalEmbeddingReceiver()
        await correctness(sender, receiver)

    @pytest.mark.asyncio
    @pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
    async def test_benchmark(self):
        sender = LocalEmbeddingSender()
        receiver = LocalEmbeddingReceiver()
        await benchmark(sender, receiver)

    @pytest.mark.asyncio
    @pytest.mark.gpu_1
    async def test_gpu_benchmark(self):
        sender = LocalEmbeddingSender()
        receiver = LocalEmbeddingReceiver()
        await benchmark(sender, receiver, from_cuda=True)


@pytest.mark.asyncio
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
class TestNixlEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await benchmark(sender, receiver)

    @pytest.mark.asyncio
    @pytest.mark.gpu_1
    async def test_gpu_benchmark(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await benchmark(sender, receiver, from_cuda=True)


@pytest.mark.asyncio
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
class TestNixlPersistentEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlPersistentEmbeddingSender()
        receiver = NixlPersistentEmbeddingReceiver()
        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlPersistentEmbeddingSender()
        receiver = NixlPersistentEmbeddingReceiver(embedding_hidden_size=EMBEDDING_SIZE)
        await benchmark(sender, receiver)

    @pytest.mark.asyncio
    @pytest.mark.gpu_1
    async def test_gpu_benchmark(self):
        sender = NixlPersistentEmbeddingSender()
        receiver = NixlPersistentEmbeddingReceiver(embedding_hidden_size=EMBEDDING_SIZE)
        await benchmark(sender, receiver, from_cuda=True)


@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
class TestRingBuffer:
    def test_simple(self):
        buffer_size = 128
        ring_buffer = RingBuffer(buffer_size)
        # Fill buffer for debugging
        for idx in range(buffer_size):
            ring_buffer.buffer_tensor[idx] = idx

        for byte_size in [32, 64, 128]:
            id, tensor = ring_buffer.get_buffer(byte_size)
            assert id is not None, f"Failed to get buffer for size {byte_size}"
            assert tensor is not None, f"Failed to get tensor for size {byte_size}"
            assert (
                tensor.nbytes == byte_size
            ), f"Expected buffer of size {byte_size}, got {tensor.nbytes}"

            ring_buffer.release_buffer(id)
        # Test allocation that exceeds buffer size
        id, tensor = ring_buffer.get_buffer(buffer_size + 1)
        assert id is None, "Expected None when requesting buffer larger than capacity"
        assert (
            tensor is None
        ), "Expected None when requesting buffer larger than capacity"

    def test_release(self):
        buffer_size = 128
        ring_buffer = RingBuffer(buffer_size)
        # Fill buffer for debugging
        for idx in range(buffer_size):
            ring_buffer.buffer_tensor[idx] = idx

        allocated_ids = []
        for byte_size in [32, 32, 64]:
            id, tensor = ring_buffer.get_buffer(byte_size)
            assert id is not None, f"Failed to get buffer for size {byte_size}"
            assert tensor is not None, f"Failed to get tensor for size {byte_size}"
            assert (
                tensor.nbytes == byte_size
            ), f"Expected buffer of size {byte_size}, got {tensor.nbytes}"
            allocated_ids.append(id)

        # Release buffers except the first one, ring buffer will not actually reuse the released space
        # until the oldest allocated buffer is released, to maintain a simple implementation.
        # |-32-|*32*|*64*| (released but not claimed space marked with *)
        # | id1|    |    |
        for id in allocated_ids[1:2]:
            ring_buffer.release_buffer(id)

        failed_id, failed_tensor = ring_buffer.get_buffer(64)
        assert (
            failed_id is None
        ), "Expected None when requesting buffer larger than remaining capacity"
        assert (
            failed_tensor is None
        ), "Expected None when requesting buffer larger than remaining capacity"

        # Release the first allocated buffer to make sure the ring buffer can reuse the released space.
        ring_buffer.release_buffer(allocated_ids[0])

        # Now we should be able to allocate a buffer of size 64 again
        id, tensor = ring_buffer.get_buffer(64)
        assert id is not None, "Failed to get buffer after releasing space"
        assert tensor is not None, "Failed to get tensor after releasing space"
        assert tensor.nbytes == 64, f"Expected buffer of size 64, got {tensor.nbytes}"

    def test_wrap_around(self):
        buffer_size = 128
        ring_buffer = RingBuffer(buffer_size)
        # Fill buffer for debugging
        for idx in range(buffer_size):
            ring_buffer.buffer_tensor[idx] = idx

        # 32 bytes remaining after allocating 96 bytes, so this should succeed
        # |-32-|-32-|-32-| 32 |
        # | id1| id2| id3|    |
        allocated_id1, tensor1 = ring_buffer.get_buffer(32)
        allocated_id2, tensor2 = ring_buffer.get_buffer(32)
        allocated_id3, tensor3 = ring_buffer.get_buffer(32)
        assert (
            allocated_id1 is not None
            and allocated_id2 is not None
            and allocated_id3 is not None
        ), "Failed to allocate initial buffers"
        assert (
            tensor1.nbytes == 32 and tensor2.nbytes == 32 and tensor3.nbytes == 32
        ), "Expected buffers of size 32"

        # Out of space
        failed_allocation_id, failed_allocation_tensor = ring_buffer.get_buffer(64)
        assert (
            failed_allocation_id is None
        ), "Expected None when requesting buffer larger than remaining capacity"
        assert (
            failed_allocation_tensor is None
        ), "Expected None when requesting buffer larger than remaining capacity"

        # Release the first buffer to create free space at the beginning,
        # but the 64 bytes allocation will fail as we don't allocate
        # | 32 |-32-|-32-| 32 |
        # |    | id2| id3|    |
        ring_buffer.release_buffer(allocated_id1)

        # small allocation okay, and should occupy part of the last 32 bytes
        # | 32 |-32-|-32-|-16-| 16 |
        # |    | id2| id3| id4|    |
        allocated_id4, tensor4 = ring_buffer.get_buffer(16)
        assert (
            allocated_id4 is not None
        ), "Failed to allocate buffer after releasing space"
        assert tensor4.nbytes == 16, f"Expected buffer of size 16, got {tensor4.nbytes}"

        # Make room for large allocation
        # Implementation detail: after wrap around, the tailing free space is marked allocated
        # |-64-|-32-|-16-|*16*|
        # | id5| id3| id4|    |
        ring_buffer.release_buffer(allocated_id2)
        allocated_id5, tensor5 = ring_buffer.get_buffer(64)
        assert (
            allocated_id5 is not None
        ), "Failed to allocate buffer after releasing space"
        assert tensor5.nbytes == 64, f"Expected buffer of size 64, got {tensor5.nbytes}"

        failed_allocation_id, failed_allocation_tensor = ring_buffer.get_buffer(8)
        assert (
            failed_allocation_id is None
        ), "Expected None when requesting buffer larger than remaining capacity"
        assert (
            failed_allocation_tensor is None
        ), "Expected None when requesting buffer larger than remaining capacity"

        # Release all and make sure we have full capacity again
        ring_buffer.release_buffer(allocated_id3)
        ring_buffer.release_buffer(allocated_id4)
        ring_buffer.release_buffer(allocated_id5)
        print(ring_buffer)
        allocated_id6, tensor6 = ring_buffer.get_buffer(buffer_size)
        assert (
            allocated_id6 is not None
        ), "Failed to allocate buffer for full capacity after releasing all buffers"
        assert (
            tensor6.nbytes == buffer_size
        ), f"Expected buffer of size {buffer_size}, got {tensor6.nbytes}"

    def test_looping(self):
        buffer_size = 64 * 3
        ring_buffer = RingBuffer(buffer_size)
        # Fill buffer for debugging
        for idx in range(buffer_size):
            ring_buffer.buffer_tensor[idx] = idx % 128  # int8 max value

        allocated_batches = []
        for _ in range(10):
            # On each batch, allocate buffers with total size of 64, afterwards
            # release previous batch if any.
            # Implementation detail: Each batch takes 1/3 of the buffer to avoid not enough
            # space with possible waste of tailing free space after wrap around.
            current_batch_ids = []
            allocated_bytes = 0
            while allocated_bytes < 64:
                new_byte_size = min(randint(8, 64), 64 - allocated_bytes)
                allocated_id, tensor = ring_buffer.get_buffer(new_byte_size)
                assert (
                    allocated_id is not None
                ), "Failed to allocate buffer in looping test"
                assert (
                    tensor.nbytes == new_byte_size
                ), f"Expected buffer of size {new_byte_size} in looping test"
                allocated_bytes += new_byte_size
                current_batch_ids.append(allocated_id)
            # Release previous batch
            for allocated_id in allocated_batches:
                ring_buffer.release_buffer(allocated_id)
            allocated_batches = current_batch_ids
