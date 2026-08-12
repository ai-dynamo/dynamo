# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for embedding transfer (local, NIXL write, NIXL read, ring buffer)."""

import asyncio
import logging
import time
from collections import OrderedDict
from random import randint

import msgspec
import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    MonolithicCounter,
    NixlReadEmbeddingReceiver,
    NixlReadEmbeddingSender,
    NixlTransferRequest,
    NixlWriteEmbeddingReceiver,
    NixlWriteEmbeddingSender,
    RingBuffer,
    TransferRequest,
)

# GPU tier is set per-class/per-test below (gpu_0 for local/ring buffer, gpu_1
# for NIXL which requires CUDA).  Total runtime ~1.6s for gpu_0 subset — no
# need for parallel marker.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.multimodal,
]

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
@pytest.mark.gpu_1  # NIXL init requires proper CUDA environment
class TestNixlWriteEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await benchmark(sender, receiver)

    async def test_gpu_benchmark(self):
        sender = NixlWriteEmbeddingSender()
        receiver = NixlWriteEmbeddingReceiver()

        await benchmark(sender, receiver, from_cuda=True)


class _FakeWriteReceiverAgent:
    def __init__(self):
        self.notifs = {"sender": []}
        self.sent = []
        self.handshake_sent = asyncio.Event()

    def add_remote_agent(self, metadata):
        return metadata

    def send_notif(self, sender_agent_id, notif_msg):
        self.sent.append((sender_agent_id, notif_msg))
        decoded = msgspec.msgpack.decode(notif_msg)
        if not (len(decoded) == 2 and decoded[0] == "cancel"):
            self.handshake_sent.set()

    def update_notifs(self):
        return self.notifs


class _FakeWriteSenderAgent:
    def __init__(
        self,
        handshake,
        *,
        fail_descriptor=False,
        fail_notifications=0,
        fail_remote_agent_additions=0,
    ):
        if handshake is None:
            self.notifications = []
        elif isinstance(handshake, list):
            self.notifications = list(handshake)
        else:
            self.notifications = [handshake]
        self.fail_descriptor = fail_descriptor
        self.fail_notifications = fail_notifications
        self.fail_remote_agent_additions = fail_remote_agent_additions
        self.remote_agent_add_calls = 0
        self.sent = []
        self.terminal_sent = asyncio.Event()
        self.deregistered = []

    def get_new_notifs(self):
        if not self.notifications:
            return {}
        notifications, self.notifications = self.notifications, []
        return {"receiver": notifications}

    def queue_notification(self, notification):
        self.notifications.append(notification)

    def send_notif(self, remote_agent_id, notif_msg):
        if self.fail_notifications:
            self.fail_notifications -= 1
            raise RuntimeError("notification failed")
        self.sent.append((remote_agent_id, msgspec.msgpack.decode(notif_msg)))
        self.terminal_sent.set()

    def add_remote_agent(self, metadata):
        self.remote_agent_add_calls += 1
        if self.fail_remote_agent_additions:
            self.fail_remote_agent_additions -= 1
            raise RuntimeError("remote agent setup failed")
        return metadata

    def get_xfer_descs(self, *args, **kwargs):
        if self.fail_descriptor:
            raise RuntimeError("target descriptor failed")
        return object()

    def deregister_memory(self, descriptor):
        self.deregistered.append(descriptor)


def _write_handshake(tensor_id=41, write_done_id=7, metadata=b""):
    return msgspec.msgpack.encode(
        (tensor_id, (1234, 8, 0, "cpu"), write_done_id, metadata)
    )


def _write_sender_for_unit_test(agent):
    sender = object.__new__(NixlWriteEmbeddingSender)
    sender.nixl_agent = agent
    sender.remote_agents = {"receiver": object()}
    sender.transfer_tracker = {}
    sender.transfer_created_at = {}
    sender.transfer_failures = {}
    sender.retired_transfer_ids = OrderedDict()
    sender.retired_transfer_limit = 4
    sender.pending_terminal_notifications = OrderedDict()
    sender.responder_lease_expirations = {}
    sender.pending_write_requests = OrderedDict()
    sender.pending_write_retry_after = {}
    sender.pending_write_retry_attempts = {}
    sender.inflight_transfers = {}
    sender._closing = False
    sender.registered_descs = {}
    sender.transfer_queue = asyncio.Queue()
    sender.transfer_timeout = 60
    return sender


async def _run_sender_until_terminal(sender, agent, *, wake_sender=False):
    if wake_sender:
        sender.transfer_queue.put_nowait("task_indicator")
    task = asyncio.create_task(sender._state_update())
    try:
        await asyncio.wait_for(agent.terminal_sent.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _write_receiver_for_unit_test(buffer_size=8):
    receiver = object.__new__(NixlWriteEmbeddingReceiver)
    receiver.ring_buffer = RingBuffer(buffer_size)
    receiver.transfer_tensor = receiver.ring_buffer.buffer_tensor
    receiver.nixl_agent = _FakeWriteReceiverAgent()
    receiver.remote_agents = {"sender": object()}
    receiver.id_counter = MonolithicCounter()
    receiver.to_buffer_id = {}
    receiver._quarantine_tasks = set()
    return receiver


def _write_request(*, shape, tensor_size, expires_at_unix=None):
    return TransferRequest(
        embeddings_shape=shape,
        embedding_dtype_str="float16",
        serialized_request=NixlTransferRequest(
            sender_agent_id="sender",
            agent_metadata=None,
            tensor_id=41,
            tensor_size=tensor_size,
            expires_at_unix=expires_at_unix,
        ).model_dump_json(),
    )


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_cancel_after_handshake_quarantines_ring_slot():
    receiver = _write_receiver_for_unit_test()
    receive_task = asyncio.create_task(
        receiver.receive_embeddings(_write_request(shape=[4], tensor_size=8))
    )
    await asyncio.wait_for(receiver.nixl_agent.handshake_sent.wait(), timeout=1)

    receive_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await receive_task

    # A remote WRITE may still target the advertised address, so the whole
    # ring remains unavailable until the sender reports a terminal handle.
    unavailable_id, unavailable = receiver.ring_buffer.get_buffer(8)
    assert unavailable_id is None
    assert unavailable is None

    handshake = next(
        msgspec.msgpack.decode(payload)
        for _, payload in receiver.nixl_agent.sent
        if len(msgspec.msgpack.decode(payload)) == 4
    )
    write_done_id = handshake[2]
    receiver.nixl_agent.notifs["sender"].append(
        msgspec.msgpack.encode(("terminal", write_done_id, "DONE"))
    )
    await asyncio.wait_for(asyncio.gather(*receiver._quarantine_tasks), timeout=1)

    available_id, available = receiver.ring_buffer.get_buffer(8)
    assert available_id is not None
    assert available is not None


@pytest.mark.asyncio
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    ("shape", "tensor_size", "error"),
    [
        ([4], 7, "tensor_size does not match"),
        ([2, -2], 8, "positive dimensions"),
    ],
)
async def test_nixl_write_malformed_descriptor_preserves_ring_capacity(
    shape, tensor_size, error
):
    receiver = _write_receiver_for_unit_test()

    with pytest.raises(ValueError, match=error):
        await receiver.receive_embeddings(
            _write_request(shape=shape, tensor_size=tensor_size)
        )

    assert not receiver.ring_buffer.allocated_buffer_id_to_range
    available_id, available = receiver.ring_buffer.get_buffer(8)
    assert available_id is not None
    assert available is not None


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_expired_lease_never_allocates_ring_buffer():
    receiver = _write_receiver_for_unit_test()

    with pytest.raises(TimeoutError, match="lease expired"):
        await receiver.receive_embeddings(
            _write_request(
                shape=[4],
                tensor_size=8,
                expires_at_unix=time.time() - 1,
            )
        )

    assert not receiver.ring_buffer.allocated_buffer_id_to_range


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_late_handshake_for_retired_id_gets_terminal_error():
    agent = _FakeWriteSenderAgent(_write_handshake())
    sender = _write_sender_for_unit_test(agent)
    sender.retired_transfer_ids[41] = time.perf_counter()

    await _run_sender_until_terminal(sender, agent)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_terminal_ack_retries_while_sender_is_idle():
    agent = _FakeWriteSenderAgent(_write_handshake(), fail_notifications=2)
    sender = _write_sender_for_unit_test(agent)
    sender.retired_transfer_ids[41] = time.perf_counter()

    await _run_sender_until_terminal(sender, agent)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    assert not sender.pending_terminal_notifications


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_descriptor_init_failure_gets_terminal_error():
    agent = _FakeWriteSenderAgent(_write_handshake(), fail_descriptor=True)
    sender = _write_sender_for_unit_test(agent)
    source = torch.zeros(4, dtype=torch.float16)
    transfer_future = asyncio.get_running_loop().create_future()
    descriptor = object()
    sender.transfer_tracker[41] = (source, object(), transfer_future)
    sender.transfer_created_at[41] = time.perf_counter()
    sender.registered_descs[(source.data_ptr(), source.get_device())] = [
        descriptor,
        1,
    ]

    await _run_sender_until_terminal(sender, agent, wake_sender=True)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    with pytest.raises(RuntimeError, match="target descriptor failed"):
        await transfer_future
    assert agent.deregistered == [descriptor]
    assert 41 in sender.retired_transfer_ids


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_shutdown_waits_for_late_handshake():
    agent = _FakeWriteSenderAgent(None)
    sender = _write_sender_for_unit_test(agent)
    sender.transfer_timeout = 1
    source = torch.zeros(4, dtype=torch.float16)
    transfer_future = asyncio.get_running_loop().create_future()
    descriptor = object()
    sender.transfer_tracker[41] = (source, object(), transfer_future)
    sender.transfer_created_at[41] = time.perf_counter()
    sender.responder_lease_expirations[41] = time.time() + 1
    sender.registered_descs[(source.data_ptr(), source.get_device())] = [
        descriptor,
        1,
    ]
    sender._state_update_task = asyncio.create_task(sender._state_update())

    close_task = asyncio.create_task(sender.aclose())
    await asyncio.sleep(0.03)
    assert not close_task.done()
    assert transfer_future.done()
    with pytest.raises(RuntimeError, match="shutting down"):
        await transfer_future

    agent.queue_notification(_write_handshake())
    await asyncio.wait_for(agent.terminal_sent.wait(), timeout=1)
    await asyncio.wait_for(close_task, timeout=1)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    assert agent.deregistered == [descriptor]
    assert sender._state_update_task is None


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_boundary_handshake_survives_receiver_expiry():
    agent = _FakeWriteSenderAgent(_write_handshake())
    sender = _write_sender_for_unit_test(agent)
    sender.retired_transfer_ids[41] = time.perf_counter()

    # Model a handshake emitted immediately before the receiver-side cutoff.
    # One receiver-expiry boundary has passed, but the sender's guarded
    # responder lease must still be alive to reject it explicitly.
    receiver_expiry = time.time() - 0.01
    sender.responder_lease_expirations[41] = receiver_expiry + 1
    sender._expire_responder_leases()
    assert 41 in sender.responder_lease_expirations

    await _run_sender_until_terminal(sender, agent)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    assert 41 not in sender.responder_lease_expirations


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_shutdown_retries_first_contact_handshake():
    agent = _FakeWriteSenderAgent(
        _write_handshake(metadata=b"receiver-metadata"),
        fail_remote_agent_additions=1,
    )
    sender = _write_sender_for_unit_test(agent)
    sender.remote_agents = {}
    sender.retired_transfer_ids[41] = time.perf_counter()
    sender.responder_lease_expirations[41] = time.time() + 1
    sender._state_update_task = asyncio.create_task(sender._state_update())

    await asyncio.wait_for(sender.aclose(), timeout=1)

    assert agent.remote_agent_add_calls == 2
    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    assert not sender.pending_write_requests
    assert not sender.responder_lease_expirations


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_handshake_then_cancel_gets_terminal_error():
    agent = _FakeWriteSenderAgent(
        [_write_handshake(), msgspec.msgpack.encode(("cancel", 41))]
    )
    sender = _write_sender_for_unit_test(agent)
    source = torch.zeros(4, dtype=torch.float16)
    transfer_future = asyncio.get_running_loop().create_future()
    descriptor = object()
    sender.transfer_tracker[41] = (source, object(), transfer_future)
    sender.transfer_created_at[41] = time.perf_counter()
    sender.responder_lease_expirations[41] = time.time() + 1
    sender.registered_descs[(source.data_ptr(), source.get_device())] = [
        descriptor,
        1,
    ]

    await _run_sender_until_terminal(sender, agent, wake_sender=True)

    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    with pytest.raises(RuntimeError, match="cancelled by receiver"):
        await transfer_future
    assert agent.deregistered == [descriptor]
    assert not sender.pending_write_requests


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_cancel_during_first_contact_retry_gets_terminal_error():
    agent = _FakeWriteSenderAgent(
        _write_handshake(metadata=b"receiver-metadata"),
        fail_remote_agent_additions=1,
    )
    sender = _write_sender_for_unit_test(agent)
    sender.remote_agents = {}
    source = torch.zeros(4, dtype=torch.float16)
    transfer_future = asyncio.get_running_loop().create_future()
    descriptor = object()
    sender.transfer_tracker[41] = (source, object(), transfer_future)
    sender.transfer_created_at[41] = time.perf_counter()
    sender.responder_lease_expirations[41] = time.time() + 1
    sender.registered_descs[(source.data_ptr(), source.get_device())] = [
        descriptor,
        1,
    ]
    sender._state_update_task = asyncio.create_task(sender._state_update())

    while agent.remote_agent_add_calls < 1:
        await asyncio.sleep(0.001)
    agent.queue_notification(msgspec.msgpack.encode(("cancel", 41)))
    await asyncio.wait_for(agent.terminal_sent.wait(), timeout=1)
    await asyncio.wait_for(sender.aclose(), timeout=1)

    assert agent.remote_agent_add_calls == 2
    assert agent.sent == [("receiver", ["terminal", 7, "ERR"])]
    with pytest.raises(RuntimeError, match="cancelled by receiver"):
        await transfer_future
    assert agent.deregistered == [descriptor]


@pytest.mark.asyncio
@pytest.mark.gpu_0
async def test_nixl_write_first_contact_failures_back_off():
    agent = _FakeWriteSenderAgent(
        _write_handshake(metadata=b"receiver-metadata"),
        fail_remote_agent_additions=100,
    )
    sender = _write_sender_for_unit_test(agent)
    sender.remote_agents = {}
    sender.retired_transfer_ids[41] = time.perf_counter()
    sender.responder_lease_expirations[41] = time.time() + 1
    state_task = asyncio.create_task(sender._state_update())

    try:
        await asyncio.sleep(0.085)
        assert 2 <= agent.remote_agent_add_calls <= 4
        assert sender.pending_write_requests
    finally:
        state_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await state_task


@pytest.mark.asyncio
@pytest.mark.gpu_1  # NIXL init requires proper CUDA environment
class TestNixlReadEmbeddingTransfer:
    async def test_correctness(self):
        sender = NixlReadEmbeddingSender()
        receiver = NixlReadEmbeddingReceiver()
        await correctness(sender, receiver)

    async def test_benchmark(self):
        sender = NixlReadEmbeddingSender()
        receiver = NixlReadEmbeddingReceiver(embedding_hidden_size=EMBEDDING_SIZE)
        await benchmark(sender, receiver)

    async def test_gpu_benchmark(self):
        sender = NixlReadEmbeddingSender()
        receiver = NixlReadEmbeddingReceiver(embedding_hidden_size=EMBEDDING_SIZE)
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

        allocated_batches: list[int] = []
        for _ in range(10):
            # On each batch, allocate buffers with total size of 64, afterwards
            # release previous batch if any.
            # Implementation detail: Each batch takes 1/3 of the buffer to avoid not enough
            # space with possible waste of tailing free space after wrap around.
            current_batch_ids: list[int] = []
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
