# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FD passing over the async wire path: ordering, partial sends, cancellation, and
descriptor ownership, on both the stdlib event loop and uvloop (the GMS server's
loop in production)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import tempfile
import threading
from unittest.mock import patch

import pytest
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


@pytest.fixture(params=["asyncio", "uvloop"])
def run(request):
    """Run a coroutine function to completion on the parametrized event loop."""
    if request.param == "uvloop":
        uvloop = pytest.importorskip("uvloop")
        loop_factory = uvloop.new_event_loop
    else:
        loop_factory = None

    def _run(coro_fn):
        with asyncio.Runner(loop_factory=loop_factory) as runner:
            return runner.run(coro_fn())

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
    local.setblocking(False)
    peer.settimeout(2)
    _, writer = await asyncio.open_connection(sock=local)
    try:
        yield writer, peer
    finally:
        peer.close()
        writer.transport.abort()
        await asyncio.sleep(0)
        local.close()


def _fill_send_buffer(writer) -> None:
    sock = writer.get_extra_info("socket")
    raw = socket.socket(fileno=os.dup(sock.fileno()))
    try:
        while True:
            try:
                raw.send(b"\0" * 4096, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return
    finally:
        raw.close()


def _large_export(tag_bytes: int = 256 * 1024) -> ExportAllocationResponse:
    return ExportAllocationResponse(
        allocation_id="a", size=8, aligned_size=8, tag="z" * tag_bytes, layout_slot=0
    )


def _tracing_sendmsg(observation: dict):
    """socket.sendmsg wrapper recording the duplicate's fd and the first result."""
    real_sendmsg = socket.socket.sendmsg

    def traced(sock, buffers, ancillary, *args, **kwargs):
        observation["sendmsg_calls"] = observation.get("sendmsg_calls", 0) + 1
        observation["socket_duplicate"] = sock.fileno()
        try:
            count = real_sendmsg(sock, buffers, ancillary, *args, **kwargs)
        except BlockingIOError:
            observation["first_send_bytes"] = 0
            raise
        observation["first_send_bytes"] = count
        return count

    return traced


async def _wait_for_sendmsg(observation: dict) -> None:
    while "first_send_bytes" not in observation:
        await asyncio.sleep(0.001)


def test_fd_frames_close_socket_duplicates_and_keep_transport_usable(run, payload):
    async def body():
        async with _stream_pair() as (writer, peer):
            sock_target = _target(writer.get_extra_info("socket").fileno())
            before = _descriptors_to(sock_target)
            for i in range(100):
                await wire.send_message(
                    writer, ErrorResponse(error=f"fd-{i}"), payload.fileno()
                )
                msg, fd, rest = wire.recv_message_sync(peer)
                try:
                    assert msg.error == f"fd-{i}"
                    assert not rest
                    assert os.pread(fd, 8, 0) == b"ORIGINAL"
                finally:
                    os.close(fd)
            assert _descriptors_to(sock_target) == before
            # The caller keeps ownership of the payload FD.
            os.fstat(payload.fileno())
            # The shared open file description stays non-blocking for the loop.
            assert not os.get_blocking(writer.get_extra_info("socket").fileno())
            await wire.send_message(writer, ErrorResponse(error="plain"))
            msg, fd, _ = wire.recv_message_sync(peer)
            assert (msg.error, fd) == ("plain", -1)

    run(body)


@pytest.mark.parametrize("fail_at", ["sendmsg", "socket"])
def test_send_setup_failure_closes_socket_duplicate(run, payload, fail_at):
    async def body():
        async with _stream_pair() as (writer, _peer):
            sock_target = _target(writer.get_extra_info("socket").fileno())
            before = _descriptors_to(sock_target)
            if fail_at == "sendmsg":
                patcher = patch.object(
                    socket.socket, "sendmsg", side_effect=BrokenPipeError("injected")
                )
                expected = BrokenPipeError
            else:
                patcher = patch.object(
                    wire.socket, "socket", side_effect=OSError("injected")
                )
                expected = OSError
            with patcher, pytest.raises(expected):
                await wire.send_message(
                    writer, ErrorResponse(error="x"), payload.fileno()
                )
            assert _descriptors_to(sock_target) == before
            # Nothing reached the wire, so the stream is still usable.
            assert not writer.is_closing()
            os.fstat(payload.fileno())

    run(body)


def test_partial_fd_frame_delivers_exact_message_and_one_fd(run, payload):
    message = _large_export()

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            observation: dict = {}
            result: dict = {}

            def receive():
                fd = -1
                try:
                    msg, fd, rest = wire.recv_message_sync(peer)
                    result["tag"] = msg.tag
                    result["payload"] = os.pread(fd, 8, 0)
                    result["rest"] = rest
                except Exception as exc:  # asserted below
                    result["error"] = exc
                finally:
                    if fd >= 0:
                        os.close(fd)

            with patch.object(socket.socket, "sendmsg", _tracing_sendmsg(observation)):
                task = asyncio.create_task(
                    wire.send_message(writer, message, payload.fileno())
                )
                await _wait_for_sendmsg(observation)
                # The kernel took only part of the frame before anyone read.
                frame_len = len(wire._frame_message(message))
                assert 0 < observation["first_send_bytes"] < frame_len
                receiver = threading.Thread(target=receive)
                receiver.start()
                await asyncio.wait_for(task, 5)
            await asyncio.get_running_loop().run_in_executor(None, receiver.join, 5)
            assert "error" not in result, result.get("error")
            assert result["tag"] == message.tag
            assert result["payload"] == b"ORIGINAL"
            assert not result["rest"]
            assert observation["sendmsg_calls"] == 1  # SCM_RIGHTS sent exactly once

    run(body)


def test_cancel_before_any_byte_keeps_stream_and_closes_duplicate(run, payload):
    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, _peer):
            sock_target = _target(writer.get_extra_info("socket").fileno())
            before = _descriptors_to(sock_target)
            _fill_send_buffer(writer)
            observation: dict = {}
            with patch.object(socket.socket, "sendmsg", _tracing_sendmsg(observation)):
                task = asyncio.create_task(
                    wire.send_message(
                        writer, ErrorResponse(error="x"), payload.fileno()
                    )
                )
                await _wait_for_sendmsg(observation)
                assert observation["first_send_bytes"] == 0  # blocked on writability
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert _descriptors_to(sock_target) == before
            assert not writer.is_closing()
            os.fstat(payload.fileno())

    run(body)


def test_cancelled_waiter_does_not_leave_a_stale_selector_entry(run, payload):
    """After cancellation the duplicate's fd number may be reused at once."""

    async def body():
        loop = asyncio.get_running_loop()
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, _peer):
            _fill_send_buffer(writer)
            observation: dict = {}
            with patch.object(socket.socket, "sendmsg", _tracing_sendmsg(observation)):
                task = asyncio.create_task(
                    wire.send_message(
                        writer, ErrorResponse(error="x"), payload.fileno()
                    )
                )
                await _wait_for_sendmsg(observation)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            old_fd = observation["socket_duplicate"]
            pair = socket.socketpair()
            reused = None
            try:
                if old_fd not in (pair[0].fileno(), pair[1].fileno()):
                    reused = os.dup2(pair[0].fileno(), old_fd)
                writable = asyncio.Event()
                loop.add_writer(old_fd, writable.set)
                try:
                    await asyncio.wait_for(writable.wait(), 2)
                finally:
                    loop.remove_writer(old_fd)
            finally:
                if reused is not None:
                    os.close(reused)
                pair[0].close()
                pair[1].close()

    run(body)


def test_cancel_after_partial_frame_aborts_and_peer_drops_fd(run, payload):
    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            payload_target = _target(payload.fileno())
            before = _descriptors_to(payload_target)
            observation: dict = {}
            with patch.object(socket.socket, "sendmsg", _tracing_sendmsg(observation)):
                task = asyncio.create_task(
                    wire.send_message(writer, _large_export(), payload.fileno())
                )
                await _wait_for_sendmsg(observation)
                assert observation["first_send_bytes"] > 0
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            # A half-sent frame cannot be resumed: the connection is torn down.
            assert writer.is_closing()

            def receive():
                fd = -1
                try:
                    _, fd, _ = wire.recv_message_sync(peer)
                    return "complete"
                except ConnectionResetError:
                    return "reset"
                finally:
                    if fd >= 0:
                        os.close(fd)

            outcome = await asyncio.get_running_loop().run_in_executor(None, receive)
            assert outcome == "reset"
            # The peer's received copy was closed by recv_message_sync.
            assert _descriptors_to(payload_target) == before

    run(body)


def test_fd_frame_waits_for_buffered_transport_bytes(run, payload):
    """An FD frame bypasses the transport, so it must not overtake its buffer."""
    earlier = ErrorResponse(error="A" * 300_000)

    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            writer.write(wire._frame_message(earlier))
            assert writer.transport.get_write_buffer_size() > 0
            result: dict = {"errors": [], "fds": []}

            def receive():
                buf = bytearray()
                try:
                    for _ in range(2):
                        msg, fd, buf = wire.recv_message_sync(peer, buf)
                        result["errors"].append(msg.error[:8])
                        result["fds"].append(fd)
                    result["payload"] = os.pread(result["fds"][1], 8, 0)
                except Exception as exc:  # asserted below
                    result["error"] = exc
                finally:
                    for fd in result["fds"]:
                        if fd >= 0:
                            os.close(fd)

            receiver = threading.Thread(target=receive)
            receiver.start()
            await asyncio.wait_for(
                wire.send_message(writer, ErrorResponse(error="B"), payload.fileno()),
                5,
            )
            await asyncio.get_running_loop().run_in_executor(None, receiver.join, 5)
            assert "error" not in result, result.get("error")
            assert result["errors"] == ["A" * 8, "B"]
            assert result["fds"][0] == -1 and result["fds"][1] >= 0
            assert result["payload"] == b"ORIGINAL"

    run(body)


def test_peer_close_while_waiting_raises_and_closes_duplicate(run, payload):
    async def body():
        async with _stream_pair(sndbuf=_SMALL_SNDBUF) as (writer, peer):
            sock_target = _target(writer.get_extra_info("socket").fileno())
            before = _descriptors_to(sock_target)
            _fill_send_buffer(writer)
            observation: dict = {}
            with patch.object(socket.socket, "sendmsg", _tracing_sendmsg(observation)):
                task = asyncio.create_task(
                    wire.send_message(writer, _large_export(), payload.fileno())
                )
                await _wait_for_sendmsg(observation)
                peer.close()  # dies with unread data: ECONNRESET / EPIPE
                with pytest.raises((BrokenPipeError, ConnectionResetError)):
                    await asyncio.wait_for(task, 5)
            # The transport may also have closed its own socket on the reset.
            assert set(_descriptors_to(sock_target)) <= set(before)
            os.fstat(payload.fileno())

    run(body)


def test_send_on_closing_transport_is_rejected(run, payload):
    async def body():
        async with _stream_pair() as (writer, _peer):
            writer.close()
            with pytest.raises(ConnectionResetError):
                await wire.send_message(
                    writer, ErrorResponse(error="x"), payload.fileno()
                )

    run(body)
