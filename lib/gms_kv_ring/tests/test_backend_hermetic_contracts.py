# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic adapter contracts for optional storage and transport runtimes.

These tests validate our descriptor, framing, integrity, and cleanup logic with
in-memory doubles. Hardware/service-backed tests remain separate live gates.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import types
import zlib

import pytest


class _MemoryMooncakeStore:
    def __init__(self):
        self.data = {}
        self.config = None
        self.closed = False

    def setup(self, config):
        self.config = config
        return 0

    def put_from(self, key, ptr, size):
        self.data[key] = bytes(ctypes.string_at(ptr, size))
        return 0

    def get_into(self, key, ptr, size):
        payload = self.data.get(key)
        if payload is None:
            return -1
        ctypes.memmove(ptr, payload, min(size, len(payload)))
        return len(payload)

    def remove(self, key):
        self.data.pop(key, None)

    def close(self):
        self.closed = True


@pytest.fixture
def mooncake_backend(monkeypatch):
    from gms_kv_ring.daemon.backends_mooncake import MooncakeBackend

    package = types.ModuleType("mooncake")
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = _MemoryMooncakeStore
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setattr(MooncakeBackend, "is_available", classmethod(lambda cls: True))
    backend = MooncakeBackend(
        local_hostname="contract-client",
        metadata_server="http://metadata.invalid",
        master_server_addr="master.invalid:50051",
        protocol="tcp",
    )
    yield backend
    backend.close()


def test_mooncake_adapter_round_trip_integrity_and_release(mooncake_backend):
    payload = bytes((i * 29) & 0xFF for i in range(2048))
    source = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    slot = mooncake_backend.demote(
        "engine", 3, 4096, ctypes.addressof(source), len(payload), crc
    )
    destination = (ctypes.c_ubyte * len(payload))()
    assert (
        mooncake_backend.promote(
            "engine", 3, 4096, ctypes.addressof(destination), len(payload)
        )
        == crc
    )
    assert bytes(destination) == payload

    stored = bytearray(mooncake_backend._mds.data[slot.mc_key])
    stored[-1] ^= 0xFF
    mooncake_backend._mds.data[slot.mc_key] = bytes(stored)
    assert (
        mooncake_backend.promote(
            "engine", 3, 4096, ctypes.addressof(destination), len(payload)
        )
        is None
    )
    assert mooncake_backend.release_slot("engine", 3, 4096)
    assert slot.mc_key not in mooncake_backend._mds.data


class _DescriptorAgent:
    def __init__(self, states=("DONE",)):
        self.states = list(states)
        self.registrations = []
        self.deregistered = []
        self.released = []
        self.initialized = []

    def get_reg_descs(self, descs, mem_type):
        return ("reg", mem_type, descs)

    def register_memory(self, desc):
        token = ("registered", len(self.registrations))
        self.registrations.append(desc)
        return token

    def get_xfer_descs(self, descs, mem_type):
        return ("xfer", mem_type, descs)

    def initialize_xfer(self, direction, local, remote, agent_name):
        handle = object()
        self.initialized.append((direction, local, remote, agent_name, handle))
        return handle

    def transfer(self, handle):
        return self.states.pop(0) if self.states else "DONE"

    def check_xfer_state(self, handle):
        return self.states.pop(0) if self.states else "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def deregister_memory(self, token):
        self.deregistered.append(token)


def _nixl_backend(agent):
    from gms_kv_ring.daemon.backends_nixl import NixlBackend

    backend = object.__new__(NixlBackend)
    backend._agent = agent
    backend.agent_name = "contract-agent"
    return backend


@pytest.mark.parametrize(
    ("method", "args", "host_type", "storage_type"),
    [
        ("_do_nixl_xfer_posix", ("WRITE", 0x1000, 64, 7, 24), "DRAM", "FILE"),
        ("_do_nixl_xfer_obj", ("WRITE", 0x1000, 64, "object-key"), "DRAM", "OBJ"),
        ("_do_nixl_xfer_gds", ("READ", 0x2000, 64, 7, 24), "VRAM", "FILE"),
    ],
)
def test_nixl_plugin_descriptor_contracts(method, args, host_type, storage_type):
    agent = _DescriptorAgent(states=("IN_PROGRESS", "DONE"))
    getattr(_nixl_backend(agent), method)(*args)

    _direction, local, remote, name, handle = agent.initialized[0]
    assert local[1] == host_type
    assert remote[1] == storage_type
    assert name == "contract-agent"
    assert agent.released == [handle]
    assert len(agent.deregistered) == 2


def test_nixl_transfer_error_still_releases_every_resource():
    agent = _DescriptorAgent(states=("ERR",))
    with pytest.raises(RuntimeError, match="returned ERR"):
        _nixl_backend(agent)._do_nixl_xfer_obj("WRITE", 0x1000, 64, "key")
    assert len(agent.released) == 1
    assert len(agent.deregistered) == 2


class _TransportAgent:
    def __init__(self, states=("IN_PROGRESS", "DONE"), metadata=True):
        self.states = list(states)
        self.metadata = metadata
        self.initialized = []
        self.released = []

    def get_xfer_descs(self, descs, mem_type):
        return (mem_type, descs)

    def check_remote_metadata(self, peer, descs):
        return self.metadata

    def initialize_xfer(self, direction, local, remote, peer, **kwargs):
        handle = object()
        self.initialized.append((direction, local, remote, peer, kwargs, handle))
        return handle

    def transfer(self, handle):
        return self.states.pop(0) if self.states else "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def _transport(agent):
    from gms_kv_ring.daemon.transport.nixl_transport import NixlTransport

    transport = object.__new__(NixlTransport)
    transport._closed = False
    transport._agent = agent
    transport._async_lock = threading.Lock()
    transport._async_pending = {}
    return transport


def test_ucx_transport_write_and_read_contracts():
    from gms_kv_ring.daemon.transport.nixl_transport import PeerHandle, _decode_notif

    agent = _TransportAgent()
    transport = _transport(agent)
    peer = PeerHandle("peer", "127.0.0.1", 1234)
    content_hash = b"h" * 32
    transport.send(peer, 0x1000, 64, 0x2000, "reservation", content_hash)
    direction, local, remote, name, kwargs, write_handle = agent.initialized[0]
    assert (direction, local[0], remote[0], name) == ("WRITE", "DRAM", "DRAM", "peer")
    assert _decode_notif(kwargs["notif_msg"]) == ("reservation", content_hash)
    assert agent.released == [write_handle]

    transport.read_batch("peer", [(0x3000, 32, 0x4000)])
    direction, _local, _remote, name, kwargs, read_handle = agent.initialized[1]
    assert (direction, name, kwargs) == ("READ", "peer", {})
    assert agent.released == [write_handle, read_handle]


def test_ucx_transport_rejects_uncovered_remote_memory():
    from gms_kv_ring.daemon.transport.nixl_transport import PeerHandle, TransportClosed

    transport = _transport(_TransportAgent(metadata=False))
    with pytest.raises(TransportClosed, match="doesn't cover"):
        transport.send(PeerHandle("peer", "host", 1), 1, 8, 2, "rid", b"x" * 32)
