# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

from __future__ import annotations

import threading
import time
from contextlib import ExitStack

import pytest

pytest.importorskip("msgspec")

from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.v1 import cli
from gpu_memory_service.v1.checkpoint import GMSCheckpointClient, GMSCheckpointLifecycle
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import PrepareCheckpointRequest
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _V1Owner:
    def __init__(self, tmp_path):
        self.lifecycle = GMSCheckpointLifecycle()
        self.vmm = FakeVMM(granularity=64)
        self.managers = {
            domain: GMSServerMemoryManager(
                "GPU-0",
                self.vmm,
                0,
                checkpoint_lifecycle=self.lifecycle,
            )
            for domain in ("weights", "kv_cache")
        }
        self.lifecycle.bind_domains(self.managers)
        self.paths = {
            domain: str(tmp_path / f"{domain}.sock") for domain in self.managers
        }
        self._stack = ExitStack()
        self.servers = {
            domain: self._stack.enter_context(
                GMSRPCServer(
                    self.paths[domain],
                    manager,
                )
            )
            for domain, manager in self.managers.items()
        }
        self.threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in self.servers.values()
        ]
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        for server in self.servers.values():
            server.shutdown()
        self._stack.close()
        for thread in self.threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    def publish_weights(self) -> None:
        writer = _GMSClientSession(self.paths["weights"], RequestedLockType.RW)
        writer.allocate("weight-0", 64)
        writer.commit()
        writer.close()
        self.wait_for_quiesced("weights")

    def wait_for_quiesced(self, domain: str) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            sessions = self.managers[domain].session_snapshot()
            if not sessions.rw_sessions and not sessions.ro_sessions:
                return
            time.sleep(0.001)
        raise TimeoutError(f"{domain} sessions did not quiesce")

    def control(self, domain: str = "weights") -> GMSCheckpointClient:
        return GMSCheckpointClient(self.paths[domain], timeout=2)


@pytest.fixture
def v1_owner(tmp_path):
    owner = _V1Owner(tmp_path)
    try:
        yield owner
    finally:
        owner.close()


@pytest.mark.timeout(10)
def test_prepare_fences_both_domains_and_abort_is_retry_safe(v1_owner) -> None:
    v1_owner.publish_weights()
    control = v1_owner.control()

    prepared = control.prepare()
    assert prepared.state == "checkpoint_ready"
    assert prepared.token

    assert control.prepare() == prepared
    for domain, lock_type in (
        ("weights", RequestedLockType.RO),
        ("kv_cache", RequestedLockType.RW),
    ):
        with pytest.raises(RuntimeError, match="admission is fenced"):
            _GMSClientSession(v1_owner.paths[domain], lock_type)
    assert v1_owner.control("kv_cache").state() == prepared
    with pytest.raises(RuntimeError, match="does not match"):
        control.abort("wrong-token")
    assert control.state() == prepared

    aborted = control.abort(prepared.token)
    assert aborted.state == "serving"
    assert aborted.token is None
    assert control.abort(prepared.token) == aborted
    with pytest.raises(RuntimeError, match="stale or already resolved"):
        control.complete(prepared.token)

    second = control.prepare()
    assert second.token and second.token != prepared.token
    with pytest.raises(RuntimeError, match="does not match"):
        control.abort(prepared.token)
    control.abort(second.token)


@pytest.mark.timeout(10)
def test_complete_is_retry_safe(v1_owner) -> None:
    v1_owner.publish_weights()
    control = v1_owner.control()
    prepared = control.prepare()
    assert prepared.token

    completed = control.complete(prepared.token)
    assert completed.state == "serving"
    assert control.complete(prepared.token) == completed
    with pytest.raises(RuntimeError, match="stale or already resolved"):
        control.abort(prepared.token)


@pytest.mark.timeout(10)
def test_prepare_rejects_invalid_domain_and_session_state(
    v1_owner, monkeypatch
) -> None:
    control = v1_owner.control()
    with pytest.raises(RuntimeError, match="weights must be committed"):
        control.prepare()

    v1_owner.publish_weights()
    reader = _GMSClientSession(v1_owner.paths["weights"], RequestedLockType.RO)
    with pytest.raises(RuntimeError, match="weights has active or waiting sessions"):
        control.prepare()

    waiting_writer: list[_GMSClientSession] = []
    writer_thread = threading.Thread(
        target=lambda: waiting_writer.append(
            _GMSClientSession(v1_owner.paths["weights"], RequestedLockType.RW)
        ),
        daemon=True,
    )
    writer_thread.start()
    deadline = time.monotonic() + 2
    while v1_owner.managers["weights"].session_snapshot().waiting_writers != 1:
        if time.monotonic() >= deadline:
            raise TimeoutError("writer did not enter the admission queue")
        time.sleep(0.001)
    with pytest.raises(RuntimeError, match="weights has active or waiting sessions"):
        control.prepare()
    reader.close()
    writer_thread.join(timeout=2)
    assert not writer_thread.is_alive()
    waiting_writer.pop().close()
    v1_owner.wait_for_quiesced("weights")
    v1_owner.publish_weights()

    kv_cache = v1_owner.managers["kv_cache"]
    monkeypatch.setattr(
        kv_cache,
        "allocation_snapshot",
        lambda: (("unexpected-kv", 64),),
    )
    with pytest.raises(RuntimeError, match="kv_cache must be empty"):
        control.prepare()


@pytest.mark.timeout(10)
def test_prepare_rejects_committed_or_active_kv(v1_owner) -> None:
    v1_owner.publish_weights()
    control = v1_owner.control()
    kv_writer = _GMSClientSession(v1_owner.paths["kv_cache"], RequestedLockType.RW)
    kv_writer.allocate("kv-0", 64)
    with pytest.raises(RuntimeError, match="kv_cache has active or waiting sessions"):
        control.prepare()
    kv_writer.commit()
    kv_writer.close()
    v1_owner.wait_for_quiesced("kv_cache")

    with pytest.raises(RuntimeError, match="kv_cache must not be committed"):
        control.prepare()


@pytest.mark.timeout(10)
def test_prepare_rejects_writer_reservation(v1_owner, monkeypatch) -> None:
    v1_owner.publish_weights()
    weights = v1_owner.managers["weights"]
    clear_started = threading.Event()
    clear_allowed = threading.Event()
    original_clear = weights._sessions._clear_epoch

    def blocked_clear() -> object:
        clear_started.set()
        assert clear_allowed.wait(5)
        return original_clear()

    monkeypatch.setattr(weights._sessions, "_clear_epoch", blocked_clear)
    writers: list[_GMSClientSession] = []
    writer_thread = threading.Thread(
        target=lambda: writers.append(
            _GMSClientSession(v1_owner.paths["weights"], RequestedLockType.RW)
        ),
        daemon=True,
    )
    writer_thread.start()
    try:
        assert clear_started.wait(5)
        assert weights.session_snapshot().writer_reserved
        with pytest.raises(
            RuntimeError, match="weights has active or waiting sessions"
        ):
            v1_owner.control().prepare()
    finally:
        clear_allowed.set()
    writer_thread.join(timeout=5)
    assert not writer_thread.is_alive()
    writers.pop().close()


@pytest.mark.timeout(10)
def test_handshake_racing_prepare_cannot_cross_the_fence(v1_owner) -> None:
    v1_owner.publish_weights()
    outcome: list[str] = []

    with v1_owner.lifecycle.condition:
        thread = threading.Thread(
            target=_record_handshake,
            args=(v1_owner, outcome),
            daemon=True,
        )
        thread.start()
        prepared = v1_owner.lifecycle.handle(PrepareCheckpointRequest())
        assert prepared.state == "checkpoint_ready"

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome == ["rejected"]


def _record_handshake(v1_owner: _V1Owner, outcome: list[str]) -> None:
    try:
        session = _GMSClientSession(v1_owner.paths["weights"], RequestedLockType.RO)
    except RuntimeError as exc:
        if "admission is fenced" not in str(exc):
            raise
        outcome.append("rejected")
        return
    session.close()
    outcome.append("admitted")


def test_cli_composes_one_lifecycle_across_both_domains(monkeypatch) -> None:
    servers = []

    class _Server:
        def __init__(self, path, manager):
            self.path = path
            self.manager = manager
            self.checkpoint_lifecycle = manager.checkpoint_lifecycle
            servers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(cli, "get_vmm", lambda: FakeVMM(granularity=64))
    monkeypatch.setattr(cli.device_identity, "get_device_uuid", lambda _device: "GPU-0")
    monkeypatch.setattr(
        cli,
        "get_socket_path",
        lambda device, domain: f"/{device}-{domain}.sock",
    )
    monkeypatch.setattr(cli, "GMSRPCServer", _Server)
    monkeypatch.setattr(cli, "run_servers", lambda _servers, _stop: None)
    monkeypatch.setattr(cli.signal, "signal", lambda *_args: None)

    cli.main(["--device", "0"])

    assert [server.path for server in servers] == [
        "/0-weights.sock",
        "/0-kv_cache.sock",
    ]
    lifecycle = servers[0].checkpoint_lifecycle
    assert servers[1].checkpoint_lifecycle is lifecycle
    assert all(server.manager.checkpoint_lifecycle is lifecycle for server in servers)


def test_bind_domains_rejects_a_different_lifecycle() -> None:
    lifecycle = GMSCheckpointLifecycle()
    other = GMSCheckpointLifecycle()
    managers = {
        domain: GMSServerMemoryManager(
            "GPU-0",
            FakeVMM(granularity=64),
            0,
            checkpoint_lifecycle=other,
        )
        for domain in ("weights", "kv_cache")
    }

    with pytest.raises(ValueError, match="must share their checkpoint lifecycle"):
        lifecycle.bind_domains(managers)
