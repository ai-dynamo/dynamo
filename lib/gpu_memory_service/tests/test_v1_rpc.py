# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import socketserver
import struct
import threading

import pytest
from _v1_fakes import V1FakeVMM
from gpu_memory_service.v1.client.rpc import AllocationClient
from gpu_memory_service.v1.common.protocol import receive_message, send_message
from gpu_memory_service.v1.errors import FatalGMSError, GMSError
from gpu_memory_service.v1.server.allocations import AllocationStore
from gpu_memory_service.v1.server.rpc import AllocationRPCServer

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


class _DropAllocateResponse(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                request, received_fd = receive_message(self.request)
            except EOFError:
                return
            if received_fd >= 0:
                os.close(received_fd)
                return
            export_fd = -1
            try:
                result, export_fd = self.server.dispatch(request)
                if request[0] == self.server.drop_method:
                    self.server.drop_method = None
                    return
                response = [True, result]
            except Exception as exc:
                response = [False, type(exc).__name__, str(exc)]
            try:
                send_message(self.request, response, export_fd)
            finally:
                if export_fd >= 0:
                    os.close(export_fd)


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_uds_fd_lifecycle_and_response_loss_replay(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 0)
    server = AllocationRPCServer(path, store)
    server.RequestHandlerClass = _DropAllocateResponse
    server.drop_method = "allocate"
    thread = _start(server)
    client = AllocationClient(path)
    try:
        client.allocate("stable-id", 128)
        client.allocate("stable-id", 128)
        assert len(vmm.server_handles) == 1
        with pytest.raises(GMSError, match="another size"):
            client.allocate("stable-id", 64)

        fd = client.export("stable-id")
        os.fstat(fd)
        os.close(fd)
        server.drop_method = "free"
        client.free("stable-id")
        client.free("stable-id")
        assert not vmm.server_handles
    finally:
        client.close()
        _stop(server, thread)


def test_reconnect_rejects_changed_nonce_or_gpu_before_mutation(tmp_path) -> None:
    for changed in ("nonce", "gpu"):
        path = str(tmp_path / f"gms-v1-{changed}.sock")
        first_store = AllocationStore("GPU-0", V1FakeVMM(), 0)
        first = AllocationRPCServer(path, first_store)
        first_thread = _start(first)
        client = AllocationClient(path)
        original_nonce = first_store.server_nonce
        client.close()
        _stop(first, first_thread)

        restarted_vmm = V1FakeVMM()
        restarted_store = AllocationStore(
            "GPU-X" if changed == "gpu" else "GPU-0", restarted_vmm, 0
        )
        if changed == "gpu":
            restarted_store.server_nonce = original_nonce
        restarted = AllocationRPCServer(path, restarted_store)
        restarted_thread = _start(restarted)
        try:
            with pytest.raises(FatalGMSError, match="incarnation|physical GPU"):
                client.reconnect()
            assert not restarted_vmm.server_handles
        finally:
            client.close()
            _stop(restarted, restarted_thread)


def test_server_cleanup_failure_latches_and_retains_handle() -> None:
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 0)
    store.allocate("allocation", 64)
    handle = next(iter(vmm.server_handles))
    vmm.fail_release[handle] = 1

    with pytest.raises(FatalGMSError) as failure:
        store.free("allocation")
    with pytest.raises(FatalGMSError) as replay:
        store.hello()

    assert replay.value is failure.value
    assert handle in vmm.server_handles


def test_malformed_frames_close_all_received_fds(monkeypatch) -> None:
    sender, receiver = socket.socketpair()
    sender.sendall(b"\x00")
    sender.close()
    try:
        with pytest.raises(EOFError):
            receive_message(receiver)
    finally:
        receiver.close()

    real_close = os.close
    for fd_count, message in ((2, "multiple"), (32, "truncated")):
        sender, receiver = socket.socketpair()
        source = [os.open("/dev/null", os.O_RDONLY) for _ in range(fd_count)]
        closed: list[int] = []

        def record_close(fd):
            closed.append(fd)
            real_close(fd)

        monkeypatch.setattr(
            "gpu_memory_service.v1.common.protocol.os.close", record_close
        )
        try:
            frame = struct.pack("!I", 2) + b"[]"
            sender.sendmsg(
                [frame],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        struct.pack(f"{fd_count}i", *source),
                    )
                ],
            )
            with pytest.raises(GMSError, match=message):
                receive_message(receiver)
            assert closed
        finally:
            sender.close()
            receiver.close()
            for fd in source:
                real_close(fd)
