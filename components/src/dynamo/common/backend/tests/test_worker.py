# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Python ``Worker`` shim."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
    exc_type=ImportError,
)

from dynamo.common.backend import worker as worker_module  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


async def test_run_keeps_host_logging_when_configure_logging_is_false(monkeypatch):
    rust_worker = MagicMock(run=AsyncMock())
    monkeypatch.setattr(
        worker_module._backend, "Worker", MagicMock(return_value=rust_worker)
    )
    # Registered so the variable run() exports is restored.
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "unset")
    monkeypatch.delenv("DYN_RESPONSE_PLANE")
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    host_handler = logging.NullHandler()
    root.addHandler(host_handler)
    root.setLevel(logging.WARNING)
    try:
        config = worker_module.WorkerConfig(namespace="test", configure_logging=False)
        await worker_module.Worker(MagicMock(), config).run()

        assert root.handlers == [*saved_handlers, host_handler]
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _clear_plane_env(monkeypatch):
    for name in ("DYN_REQUEST_PLANE", "DYN_RESPONSE_PLANE"):
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)


async def test_config_constructor_failure_preserves_host_response_plane(monkeypatch):
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    config = worker_module.WorkerConfig(
        namespace="test",
        configure_logging=False,
        discovery_backend="mem",
        request_plane="velo",
        response_plane="velo",
        structural_tag={"typo": True},
    )
    with pytest.raises(ValueError, match="Invalid structural_tag config"):
        await worker_module.Worker(MagicMock(), config).run()
    assert os.environ["DYN_RESPONSE_PLANE"] == "quic"


async def test_tls_refusal_preserves_the_host_response_plane(monkeypatch):
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    monkeypatch.setenv("DYN_TCP_TLS_INSECURE", "true")
    config = worker_module.WorkerConfig(
        namespace="test",
        configure_logging=False,
        discovery_backend="mem",
        request_plane="velo",
        response_plane="velo",
    )
    with pytest.raises(Exception, match="(?i)velo"):
        await worker_module.Worker(MagicMock(), config).run()
    assert os.environ["DYN_RESPONSE_PLANE"] == "quic"


async def test_run_rejects_velo_request_with_tcp_response_before_env_write(monkeypatch):
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "velo")
    rust_worker = MagicMock(run=AsyncMock())
    monkeypatch.setattr(
        worker_module._backend, "Worker", MagicMock(return_value=rust_worker)
    )

    config = worker_module.WorkerConfig(
        namespace="test",
        configure_logging=False,
        request_plane="velo",
        response_plane="tcp",
    )

    with pytest.raises(ValueError, match="requires"):
        await worker_module.Worker(MagicMock(), config).run()

    assert os.environ["DYN_RESPONSE_PLANE"] == "velo"
    rust_worker.run.assert_not_awaited()


async def test_run_exports_the_explicit_response_plane(monkeypatch):
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "velo")
    runtime_config = MagicMock()
    worker_config = MagicMock()
    rust_worker = MagicMock(run=AsyncMock())
    monkeypatch.setattr(worker_module._backend, "RuntimeConfig", runtime_config)
    monkeypatch.setattr(worker_module._backend, "WorkerConfig", worker_config)
    monkeypatch.setattr(
        worker_module._backend, "Worker", MagicMock(return_value=rust_worker)
    )

    config = worker_module.WorkerConfig(
        namespace="test",
        configure_logging=False,
        request_plane=" tcp ",
        response_plane=" TCP ",
    )

    await worker_module.Worker(MagicMock(), config).run()

    runtime_config.assert_called_once_with(
        discovery_backend="etcd",
        request_plane="tcp",
        event_plane=None,
        response_plane="tcp",
    )
    assert os.environ["DYN_RESPONSE_PLANE"] == "tcp"


async def test_start_failure_preserves_host_response_plane_set_during_run(monkeypatch):
    import asyncio

    _clear_plane_env(monkeypatch)
    entered = asyncio.Event()
    resume = asyncio.Event()

    class FailingEngine:
        async def start(self, worker_id):
            entered.set()
            await resume.wait()
            raise RuntimeError("engine startup failed")

        async def cleanup(self):
            pass

    config = worker_module.WorkerConfig(
        namespace="test", discovery_backend="mem", configure_logging=False
    )
    task = asyncio.create_task(worker_module.Worker(FailingEngine(), config).run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert "DYN_RESPONSE_PLANE" not in os.environ
        monkeypatch.setenv("DYN_RESPONSE_PLANE", "velo")
    finally:
        resume.set()
    with pytest.raises(Exception, match="engine startup failed"):
        await asyncio.wait_for(task, timeout=10)
    observed = os.environ.get("DYN_RESPONSE_PLANE")
    assert observed == "velo"


async def test_run_normalizes_the_velo_pair(monkeypatch):
    _clear_plane_env(monkeypatch)
    runtime_config = MagicMock()
    worker_config = MagicMock()
    monkeypatch.setattr(worker_module._backend, "RuntimeConfig", runtime_config)
    monkeypatch.setattr(worker_module._backend, "WorkerConfig", worker_config)
    monkeypatch.setattr(
        worker_module._backend,
        "Worker",
        MagicMock(return_value=MagicMock(run=AsyncMock())),
    )

    config = worker_module.WorkerConfig(
        namespace="test",
        configure_logging=False,
        request_plane=" VELO ",
        response_plane=" VELO ",
    )

    await worker_module.Worker(MagicMock(), config).run()

    runtime_config.assert_called_once_with(
        discovery_backend="etcd",
        request_plane="velo",
        event_plane=None,
        response_plane="velo",
    )
    assert os.environ["DYN_RESPONSE_PLANE"] == "velo"
