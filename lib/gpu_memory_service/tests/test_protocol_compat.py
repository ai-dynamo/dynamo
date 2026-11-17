# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched metadata reads and FD exports against a real in-process server:
capability gating across old/new peers, bounded reply pages, and descriptor
accounting on both the stdlib event loop and uvloop."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import typing

import msgspec
import pytest
from _deps import HAS_GMS, HAS_XPU

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

import gpu_memory_service.common.vmm as _vmm_module
import gpu_memory_service.server.gms as server_gms
import gpu_memory_service.server.rpc as server_rpc
from _fake_vmm import FakeVMM
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.protocol import messages, wire
from gpu_memory_service.common.protocol.messages import (
    CAPABILITY_METADATA_GET_ALL,
    HandshakeResponse,
    MetadataGetAllRequest,
    MetadataGetAllResponse,
)
from gpu_memory_service.common.vmm import VMMDeviceType
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]

# The frame cap v1 applies (v1/protocol.py MAX_FRAME) and #14259 brings to v0.
_MAX_FRAME = 1 << 20


class _ServerThread:
    def __init__(self, server: GMSRPCServer, loop_factory):
        self.server = server
        self._loop_factory = loop_factory
        self._loop = None
        self._task = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with asyncio.Runner(loop_factory=self._loop_factory) as runner:
            self._loop = runner.get_loop()
            self._task = self._loop.create_task(self.server.serve())
            try:
                runner.run(self._wait())
            except asyncio.CancelledError:
                pass

    async def _wait(self) -> None:
        await self._task

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5
        while not (
            self.server._server is not None and os.path.exists(self.server.socket_path)
        ):
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise TimeoutError("GMS server did not start")
            time.sleep(0.01)

    def stop(self) -> None:
        def cancel() -> None:
            self.server._server.close()
            self._task.cancel()

        self._loop.call_soon_threadsafe(cancel)
        self._thread.join(5)
        assert not self._thread.is_alive()


@pytest.fixture(params=["asyncio", "uvloop"])
def gms(request, monkeypatch, tmp_path):
    if request.param == "uvloop":
        loop_factory = pytest.importorskip("uvloop").new_event_loop
    else:
        loop_factory = None
    fake_vmm = FakeVMM()
    monkeypatch.setattr(_vmm_module, "_vmm_instance", fake_vmm)
    monkeypatch.setattr(
        _vmm_module,
        "_vmm_device_type",
        VMMDeviceType.XPU if HAS_XPU else VMMDeviceType.CUDA,
    )
    socket_path = str(tmp_path / "gms.sock")
    server = GMSRPCServer(socket_path, device=0, allocation_retry_interval=0.01)
    thread = _ServerThread(server, loop_factory)
    thread.start()
    try:
        yield server, socket_path
    finally:
        thread.stop()


def _publish(socket_path: str, entries: dict[str, bytes], allocations: int = 1):
    writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        ids = [
            writer.mappings[writer.create_mapping(size=4096)].allocation_id
            for _ in range(allocations)
        ]
        for i, (key, value) in enumerate(sorted(entries.items())):
            assert writer.metadata_put(key, ids[i % len(ids)], 0, value)
        assert writer.commit()
    finally:
        writer.close()
    return ids


def _expected(entries, prefix="", exclude_prefix=""):
    return [
        key
        for key in sorted(entries)
        if key.startswith(prefix)
        and not (exclude_prefix and key.startswith(exclude_prefix))
    ]


def _old_decoder():
    """The server's decoder before MetadataGetAll existed."""
    new_types = (MetadataGetAllRequest, MetadataGetAllResponse)
    old = typing.Union[
        tuple(t for t in typing.get_args(messages.Message) if t not in new_types)
    ]
    return msgspec.msgpack.Decoder(old)


def test_new_server_advertises_batched_metadata(gms):
    _, socket_path = gms
    _publish(socket_path, {"w.0": b"x"})
    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        assert reader.supports(CAPABILITY_METADATA_GET_ALL)


def test_old_client_ignores_new_handshake_field():
    class OldHandshakeResponse(msgspec.Struct, tag="handshake_response"):
        success: bool
        committed: bool
        granted_lock_type: typing.Optional[GrantedLockType] = None

    encoded = messages.encode_message(
        HandshakeResponse(
            success=True,
            committed=True,
            granted_lock_type=GrantedLockType.RO,
            capabilities=[CAPABILITY_METADATA_GET_ALL],
        )
    )
    decoded = msgspec.msgpack.decode(encoded, type=OldHandshakeResponse)
    assert decoded == OldHandshakeResponse(True, True, GrantedLockType.RO)


def test_new_client_on_old_server_falls_back_and_keeps_its_session(gms, monkeypatch):
    server, socket_path = gms
    entries = {"w.0": b"a", "w.1": b"b", "__gms__.private": b"c"}
    _publish(socket_path, entries)
    # An old server neither advertises the capability nor decodes the request.
    monkeypatch.setattr(server_rpc, "SERVER_CAPABILITIES", ())
    monkeypatch.setattr(messages, "_decoder", _old_decoder())

    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        assert not reader.supports(CAPABILITY_METADATA_GET_ALL)
        got = reader.metadata_get_all(exclude_prefix="__gms__.")
        assert [key for key, *_ in got] == ["w.0", "w.1"]
        assert [value for *_, value in got] == [b"a", b"b"]
        # The fallback is plain v0 traffic: the session and its lock survive.
        assert reader.is_connected
        assert server._gms._sessions.snapshot().ro_session_count == 1


def test_ungated_request_to_old_server_drops_the_session(gms, monkeypatch):
    """Why the capability check is needed: an old server cannot answer."""
    server, socket_path = gms
    _publish(socket_path, {"w.0": b"a"})
    monkeypatch.setattr(messages, "_decoder", _old_decoder())
    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        with pytest.raises(ConnectionError):
            reader._transport.request(MetadataGetAllRequest(), MetadataGetAllResponse)
        assert not reader.is_connected


@pytest.mark.parametrize(
    "prefix,exclude_prefix", [("", ""), ("", "__gms__."), ("layer.01", "")]
)
def test_metadata_get_all_pages_in_key_order(gms, monkeypatch, prefix, exclude_prefix):
    _, socket_path = gms
    entries = {f"layer.{i:04d}.weight": bytes([i % 251]) * 300 for i in range(400)}
    entries.update({"": b"empty-key", "__gms__.private": b"p"})
    _publish(socket_path, entries, allocations=3)
    page_budget = 8 * 1024
    monkeypatch.setattr(server_gms, "METADATA_GET_ALL_PAGE_BYTES", page_budget)
    frames: list[int] = []
    real_frame = wire._frame_message

    def recording_frame(msg):
        frame = real_frame(msg)
        if isinstance(msg, MetadataGetAllResponse):
            frames.append(len(frame))
        return frame

    monkeypatch.setattr(wire, "_frame_message", recording_frame)

    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        got = reader.metadata_get_all(prefix=prefix, exclude_prefix=exclude_prefix)
        assert [key for key, *_ in got] == _expected(entries, prefix, exclude_prefix)
        assert all(value == entries[key] for key, _, _, value in got)
    assert len(frames) > 1
    assert max(frames) <= page_budget


def test_default_page_stays_under_frame_cap_for_large_models(gms, monkeypatch):
    _, socket_path = gms
    # ~12k tensors of realistic key/value size: about 3 MiB of metadata.
    value = (
        b'{"dtype": "torch.float8_e4m3fn", "shape": [256, 4096, 7168], '
        b'"stride": [29360128, 7168, 1], "tensor_type": "parameter"}'
    )
    entries = {
        f"model.layers.{i // 150}.mlp.experts.{i % 150}.w13_weight_scale": value
        for i in range(12_000)
    }
    _publish(socket_path, entries, allocations=4)
    frames: list[int] = []
    real_frame = wire._frame_message

    def recording_frame(msg):
        frame = real_frame(msg)
        if isinstance(msg, MetadataGetAllResponse):
            frames.append(len(frame))
        return frame

    monkeypatch.setattr(wire, "_frame_message", recording_frame)
    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        got = reader.metadata_get_all()
    assert len(got) == len(entries)
    assert sum(frames) > _MAX_FRAME  # one unpaged reply would exceed the cap
    assert max(frames) <= server_gms.METADATA_GET_ALL_PAGE_BYTES < _MAX_FRAME


def test_oversized_entry_is_returned_alone(gms, monkeypatch):
    _, socket_path = gms
    entries = {"a": b"1", "b": b"x" * 4096, "c": b"3"}
    _publish(socket_path, entries)
    monkeypatch.setattr(server_gms, "METADATA_GET_ALL_PAGE_BYTES", 1024)
    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        assert [key for key, *_ in reader.metadata_get_all()] == ["a", "b", "c"]


def test_bulk_export_round_trips_fds_without_leaks(gms):
    """Many consecutive SCM_RIGHTS replies over one connection, both loops."""
    server, socket_path = gms
    ids = _publish(socket_path, {"w.0": b"x"}, allocations=64)
    # Everything the fake VMM exports is a pipe; count them across the process.
    before = len(os.listdir("/proc/self/fd"))
    with _GMSClientSession(socket_path, RequestedLockType.RO, None) as reader:
        for _ in range(4):
            fds = [reader.export(allocation_id) for allocation_id in ids]
            assert len(set(fds)) == len(fds)
            for fd in fds:
                os.fstat(fd)
                os.close(fd)
    deadline = time.monotonic() + 2
    while len(os.listdir("/proc/self/fd")) > before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(os.listdir("/proc/self/fd")) <= before
    assert server._gms._sessions.snapshot().ro_session_count == 0
