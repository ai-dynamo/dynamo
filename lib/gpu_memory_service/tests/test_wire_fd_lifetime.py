# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FD passing over the async wire path, on the stdlib event loop and on uvloop
(the GMS server's loop)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import tempfile

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

import uvloop
from gpu_memory_service.common.protocol import wire
from gpu_memory_service.common.protocol.messages import (
    ErrorResponse,
    ExportAllocationResponse,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
    pytest.mark.skipif(
        not (os.path.isdir("/proc/self/fd") and hasattr(socket, "SCM_RIGHTS")),
        reason="requires Linux Unix-socket descriptor passing",
    ),
]

_SMALL_SNDBUF = 4096
_TIMEOUT = 5


@pytest.fixture(params=["asyncio", "uvloop"])
def run(request):
    """Run a coroutine function to completion on the parametrized event loop."""
    new_loop = (
        uvloop.new_event_loop if request.param == "uvloop" else asyncio.new_event_loop
    )

    def _run(coro_fn):
        loop = new_loop()
        try:
            return loop.run_until_complete(coro_fn())
        finally:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    return _run


@pytest.fixture
def payload():
    with tempfile.TemporaryFile() as f:
        f.write(b"ORIGINAL")
        f.flush()
        yield f


def _target(fd: int) -> str:
    return os.readlink(f"/proc/self/fd/{fd}")


def _descriptors_to(target: str) -> list[int]:
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            if _target(int(name)) == target:
                found.append(int(name))
        except OSError:
            pass
    return sorted(found)


@contextlib.asynccontextmanager
async def _stream_pair(sndbuf: int | None = None):
    """(writer, peer): a StreamWriter on one end, a blocking socket on the other."""
    local, peer = socket.socketpair()
    if sndbuf is not None:
        local.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
    peer.settimeout(_TIMEOUT)
    _, writer = await asyncio.open_connection(sock=local)
    try:
        yield writer, peer
    finally:
        peer.close()
        writer.transport.abort()
        await asyncio.sleep(0)
        local.close()


def _fill_send_buffer(writer) -> None:
    with writer.get_extra_info("socket").dup() as raw:
        with contextlib.suppress(BlockingIOError):
            while True:
                raw.send(b"\0" * 4096, socket.MSG_DONTWAIT)


def _large_export() -> ExportAllocationResponse:
    return ExportAllocationResponse(
        allocation_id="a", size=8, aligned_size=8, tag="z" * 262144, layout_slot=0
    )


def _receive(peer: socket.socket, size: int) -> tuple[bytes, list[int]]:
    """Read up to ``size`` bytes, or to EOF, with every descriptor that arrives."""
    data = bytearray()
    fds: list[int] = []
    while len(data) < size:
        chunk, received, _, _ = socket.recv_fds(peer, size - len(data), 8)
        fds += received
        if not chunk:
            break
        data += chunk
    return bytes(data), fds


async def _receive_in_thread(peer: socket.socket, size: int):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _receive, peer, size), _TIMEOUT
    )


def _start_send(writer, message, payload) -> asyncio.Task:
    return asyncio.create_task(wire.send_message(writer, message, payload.fileno()))


def test_fd_frames_leak_no_descriptor_and_keep_the_socket_non_blocking(run, payload):
    async def body():
        async with _stream_pair() as (writer, peer):
            sock_fd = writer.get_extra_info("socket").fileno()
            before = _descriptors_to(_target(sock_fd))
            for i in range(3):
                await wire.send_message(
                    writer, ErrorResponse(error=f"fd-{i}"), payload.fileno()
                )
                msg, fd, rest = wire.recv_message_sync(peer)
                try:
                    assert (msg.error, bytes(rest)) == (f"fd-{i}", b"")
                    assert os.pread(fd, 8, 0) == b"ORIGINAL"
                finally:
                    os.close(fd)
            assert _descriptors_to(_target(sock_fd)) == before
            assert not os.get_blocking(sock_fd)
            # The caller keeps ownership of the payload descriptor.
            os.fstat(payload.fileno())
            await wire.send_message(writer, ErrorResponse(error="plain"))
            msg, fd, _ = wire.recv_message_sync(peer)
            assert (msg.error, fd) == ("plain", -1)

    run(body)


def test_fd_frame_waits_for_buffered_transport_bytes(run, payload):
    earlier = wire._frame_message(ErrorResponse(error="A" * 300_000))
    fd_message = ErrorResponse(error="B")

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            writer.write(earlier)
            assert writer.transport.get_write_buffer_size() > 0
            expected = earlier + wire._frame_message(fd_message)
            receiver = asyncio.ensure_future(_receive_in_thread(peer, len(expected)))
            await asyncio.wait_for(
                wire.send_message(writer, fd_message, payload.fileno()), _TIMEOUT
            )
            data, fds = await receiver
            try:
                assert data == expected
                assert len(fds) == 1
                assert os.pread(fds[0], 8, 0) == b"ORIGINAL"
            finally:
                for fd in fds:
                    os.close(fd)

    run(body)


def test_partial_fd_frame_delivers_exact_bytes_and_one_fd(run, payload):
    message = _large_export()
    frame = wire._frame_message(message)

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            task = _start_send(writer, message, payload)
            await asyncio.sleep(0)
            # Nobody is reading, so the kernel took only part of the frame.
            assert not task.done()
            assert peer.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            receiver = asyncio.ensure_future(_receive_in_thread(peer, len(frame)))
            await asyncio.wait_for(task, _TIMEOUT)
            data, fds = await receiver
            try:
                assert data == frame
                assert len(fds) == 1
                assert os.pread(fds[0], 8, 0) == b"ORIGINAL"
            finally:
                for fd in fds:
                    os.close(fd)

    run(body)


def test_send_on_closing_transport_is_rejected(run, payload):
    async def body():
        async with _stream_pair() as (writer, _peer):
            writer.close()
            with pytest.raises(ConnectionResetError):
                await wire.send_message(
                    writer, ErrorResponse(error="x"), payload.fileno()
                )
            # Once closed, uvloop no longer exposes the socket at all.
            await writer.wait_closed()
            with pytest.raises(ConnectionResetError):
                await wire.send_message(
                    writer, ErrorResponse(error="x"), payload.fileno()
                )

    run(body)


@pytest.mark.parametrize("already_sent", ["nothing", "part of the frame"])
def test_close_while_waiting_for_the_peer_rejects_the_send(run, payload, already_sent):
    message = _large_export() if already_sent != "nothing" else ErrorResponse(error="x")
    frame = wire._frame_message(message)

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            if already_sent == "nothing":
                _fill_send_buffer(writer)
            task = _start_send(writer, message, payload)
            await asyncio.sleep(0)
            assert not task.done()
            writer.close()
            # The send wakes when the peer makes room, and must not use it.
            data, fds = await _receive_in_thread(peer, 1 << 30)
            for fd in fds:
                os.close(fd)
            with pytest.raises(ConnectionResetError):
                await asyncio.wait_for(task, _TIMEOUT)
            if already_sent == "nothing":
                assert not data.strip(b"\0")
                assert not fds
            else:
                # Only what went out before the close arrives.
                assert frame.startswith(data) and len(data) < len(frame)

    run(body)


def test_cancel_before_any_byte_leaves_the_stream_usable(run, payload):
    async def body():
        loop = asyncio.get_running_loop()
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, _peer):
            sock_target = _target(writer.get_extra_info("socket").fileno())
            before = _descriptors_to(sock_target)
            _fill_send_buffer(writer)
            task = _start_send(writer, ErrorResponse(error="x"), payload)
            await asyncio.sleep(0)
            (duplicate,) = set(_descriptors_to(sock_target)) - set(before)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert _descriptors_to(sock_target) == before
            assert not writer.is_closing()

            # The duplicate's number is free for reuse: the loop must not
            # still hold a registration for it.
            reused, other = socket.socketpair()
            try:
                if duplicate not in (reused.fileno(), other.fileno()):
                    pytest.skip("the descriptor number was not reused")
                writable = asyncio.Event()
                loop.add_writer(duplicate, writable.set)
                try:
                    await asyncio.wait_for(writable.wait(), _TIMEOUT)
                finally:
                    loop.remove_writer(duplicate)
            finally:
                reused.close()
                other.close()

    run(body)


def test_cancel_after_partial_frame_aborts_the_transport(run, payload):
    message = _large_export()
    frame_len = len(wire._frame_message(message))

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            payload_target = _target(payload.fileno())
            before = _descriptors_to(payload_target)
            task = _start_send(writer, message, payload)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # A half-sent frame cannot be resumed: the connection is torn down.
            assert writer.is_closing()
            data, fds = await _receive_in_thread(peer, frame_len)
            for fd in fds:
                os.close(fd)
            assert 0 < len(data) < frame_len
            assert _descriptors_to(payload_target) == before

    run(body)
