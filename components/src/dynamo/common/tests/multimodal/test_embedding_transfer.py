# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AsyncEncoderCache."""

import asyncio
import logging
import time

import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    NixlEmbeddingReceiver,
    NixlEmbeddingSender,
    NixlPersistentEmbeddingReceiver,
    NixlPersistentEmbeddingSender,
)

logger = logging.getLogger(__name__)


async def benchmark(sender, receiver, tensors=None):
    if tensors is None:
        tensors = [torch.randn(256, 8 * 1024) for _ in range(30)]
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
        assert torch.equal(received_tensor, tensor)
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


@pytest.mark.xfail(run=False, reason="slow")
@pytest.mark.asyncio
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
class TestNixlEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlEmbeddingSender()
        receiver = NixlEmbeddingReceiver()

        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlEmbeddingSender()
        receiver = NixlEmbeddingReceiver()
        await benchmark(sender, receiver)


@pytest.mark.asyncio
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
class TestNixlPersistentEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlPersistentEmbeddingSender()
        receiver = NixlPersistentEmbeddingReceiver()
        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlPersistentEmbeddingSender()
        receiver = NixlPersistentEmbeddingReceiver()
        await benchmark(sender, receiver)
