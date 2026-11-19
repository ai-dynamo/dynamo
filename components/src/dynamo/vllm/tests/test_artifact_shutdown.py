# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
]


@pytest.fixture
def vllm_main(vllm_cpu_platform_when_no_accelerator):
    from dynamo.vllm import main

    return main


@pytest.mark.timeout(60)
def test_worker_import_does_not_load_optional_artifact_runtime() -> None:
    probe = """
import importlib.abc
import os
import sys

class RejectArtifactRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "dynamo.artifacts.runtime":
            raise ModuleNotFoundError("optional artifact runtime was imported")

sys.meta_path.insert(0, RejectArtifactRuntime())
os.environ.pop("DYNAMO_ARTIFACT_CONFIG", None)
import dynamo.vllm.main
assert "dynamo.artifacts.runtime" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        timeout=50,
    )


def _patch_worker_startup(vllm_main, monkeypatch, tmp_path, *, factory_create):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = SimpleNamespace(
        embedding_worker_processes=1,
        dump_config_to=None,
        served_model_name="test-model",
        model=str(model_path),
        artifact_router_profile=None,
        engine_args=SimpleNamespace(
            model=str(model_path), served_model_name="test-model"
        ),
        headless=False,
        worker_start_method="spawn",
        discovery_backend=None,
        request_plane=None,
        event_plane=None,
        response_plane=None,
    )
    lifecycle = SimpleNamespace(close=AsyncMock())
    factory = SimpleNamespace(create=AsyncMock(side_effect=factory_create))
    signal_handlers = Mock()

    monkeypatch.setattr(
        vllm_main, "parse_args_with_model_fetch", AsyncMock(return_value=config)
    )
    monkeypatch.setattr(
        vllm_main, "is_embedding_process_child", Mock(return_value=False)
    )
    monkeypatch.setattr(vllm_main, "dump_config", Mock())
    monkeypatch.setattr(vllm_main, "configure_rl_logprobs_mode", Mock())
    monkeypatch.setattr(
        vllm_main, "prepare_snapshot_engine", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        vllm_main, "create_runtime", Mock(return_value=(Mock(), Mock()))
    )
    monkeypatch.setattr(vllm_main, "install_signal_handlers", signal_handlers)
    monkeypatch.setattr(vllm_main, "StateAgentLifecycle", Mock(return_value=lifecycle))
    monkeypatch.setattr(vllm_main, "WorkerFactory", Mock(return_value=factory))
    monkeypatch.setattr(
        vllm_main, "initialize_configured_artifact_runtime", AsyncMock()
    )
    return lifecycle, signal_handlers


@pytest.mark.asyncio
async def test_invalid_router_profile_fails_before_model_load(
    monkeypatch, tmp_path, vllm_main
) -> None:
    _patch_worker_startup(
        vllm_main, monkeypatch, tmp_path, factory_create=RuntimeError("unused")
    )
    config = vllm_main.parse_args_with_model_fetch.return_value
    config.artifact_router_profile = "invalid json"
    fetch_model = AsyncMock()
    monkeypatch.setattr(vllm_main, "fetch_model", fetch_model)

    with pytest.raises(ValueError, match="invalid --artifact-router-profile"):
        await vllm_main.worker([])

    fetch_model.assert_not_awaited()
    vllm_main.prepare_snapshot_engine.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, snapshot, want_order",
    [
        ("spawn", False, ["runtime"]),
        ("fork", False, ["engine", "runtime"]),
        # A snapshot controller already holds the engine; it is not built again.
        ("fork", True, ["runtime"]),
    ],
)
async def test_fork_builds_the_engine_before_the_runtime(
    monkeypatch, tmp_path, vllm_main, method, snapshot, want_order
) -> None:
    async def serve_noop(*_args, **_kwargs):
        return None

    _patch_worker_startup(vllm_main, monkeypatch, tmp_path, factory_create=serve_noop)
    config = vllm_main.parse_args_with_model_fetch.return_value
    config.worker_start_method = method
    order: list[str] = []
    engine_setup = Mock()
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    monkeypatch.setattr(
        vllm_main,
        "setup_vllm_engine",
        Mock(side_effect=lambda *_a, **_k: order.append("engine") or engine_setup),
    )
    vllm_main.create_runtime.side_effect = lambda **_k: order.append("runtime") or (
        Mock(),
        Mock(),
    )
    if snapshot:
        controller = SimpleNamespace(engine=Mock(), pause_controller=Mock())
        vllm_main.prepare_snapshot_engine.return_value = controller
        monkeypatch.setattr(vllm_main, "elect_and_wake", AsyncMock())
        monkeypatch.setattr(
            vllm_main, "refresh_snapshot_restore_config", AsyncMock(return_value=config)
        )
        monkeypatch.setattr(vllm_main, "apply_restore_served_model_name", Mock())
        monkeypatch.setattr(vllm_main, "env_bool", Mock(return_value=False))

    await vllm_main.worker([])

    assert order == want_order
    prebuilt = vllm_main.WorkerFactory.return_value.create.await_args.kwargs[
        "prebuilt_engine"
    ]
    if want_order[0] == "engine":
        assert prebuilt[0] is engine_setup
        assert isinstance(prebuilt[1], vllm_main.StatLoggerFactory)
    else:
        assert prebuilt is None


def _fork_engine_client(vllm_main, monkeypatch) -> Mock:
    """Select fork and stub the engine it builds ahead of the runtime."""
    config = vllm_main.parse_args_with_model_fetch.return_value
    config.worker_start_method = "fork"
    engine_client = Mock()
    engine_setup = (
        engine_client,
        SimpleNamespace(shutdown_timeout=5.0),
        Mock(),
        None,
        Mock(),
    )
    monkeypatch.setattr(vllm_main, "setup_vllm_engine", Mock(return_value=engine_setup))
    return engine_client


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["artifact_runtime", "runtime", "cancellation"])
async def test_fork_shuts_the_engine_down_when_startup_fails_before_a_worker_owns_it(
    monkeypatch, tmp_path, vllm_main, failure
) -> None:
    _patch_worker_startup(vllm_main, monkeypatch, tmp_path, factory_create=None)
    engine_client = _fork_engine_client(vllm_main, monkeypatch)
    # Cancellation, like Ctrl-C, is not an Exception.
    startup_error: BaseException = (
        asyncio.CancelledError()
        if failure == "cancellation"
        else RuntimeError(f"{failure} failed")
    )
    if failure == "artifact_runtime":
        monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(tmp_path / "artifact.json"))
        vllm_main.initialize_configured_artifact_runtime.side_effect = startup_error
    else:
        monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
        vllm_main.create_runtime.side_effect = startup_error

    with pytest.raises(type(startup_error)) as exc_info:
        await vllm_main.worker([])

    assert exc_info.value is startup_error
    engine_client.shutdown.assert_called_once_with(timeout=5.0)
    vllm_main.WorkerFactory.return_value.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_fork_startup_error_propagates_when_the_engine_shutdown_fails_too(
    monkeypatch, tmp_path, vllm_main, caplog
) -> None:
    _patch_worker_startup(vllm_main, monkeypatch, tmp_path, factory_create=None)
    engine_client = _fork_engine_client(vllm_main, monkeypatch)
    engine_client.shutdown.side_effect = RuntimeError("engine already dead")
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    vllm_main.create_runtime.side_effect = RuntimeError("runtime failed")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="runtime failed"):
            await vllm_main.worker([])

    engine_client.shutdown.assert_called_once_with(timeout=5.0)
    assert "Failed to shut down the engine after startup failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_fork_leaves_the_engine_to_the_worker_it_was_handed_to(
    monkeypatch, tmp_path, vllm_main, worker_fails
) -> None:
    """From factory.create() on, the worker's lifecycle shuts the engine down."""
    _patch_worker_startup(
        vllm_main,
        monkeypatch,
        tmp_path,
        factory_create=RuntimeError("worker failed") if worker_fails else None,
    )
    engine_client = _fork_engine_client(vllm_main, monkeypatch)
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)

    if worker_fails:
        with pytest.raises(RuntimeError, match="worker failed"):
            await vllm_main.worker([])
    else:
        await vllm_main.worker([])

    create = vllm_main.WorkerFactory.return_value.create
    create.assert_awaited_once()
    assert create.await_args.kwargs["prebuilt_engine"][0][0] is engine_client
    engine_client.shutdown.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method, in_subprocess", [("spawn", False), ("fork", True)])
async def test_fork_fetches_the_model_outside_the_forking_process(
    monkeypatch, tmp_path, vllm_main, method, in_subprocess
) -> None:
    async def serve_noop(*_args, **_kwargs):
        return None

    _patch_worker_startup(vllm_main, monkeypatch, tmp_path, factory_create=serve_noop)
    config = vllm_main.parse_args_with_model_fetch.return_value
    config.worker_start_method = method
    config.model = "org/model"
    fetch_here = AsyncMock(return_value="/models/here")
    fetch_in_subprocess = AsyncMock(return_value="/models/subprocess")
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    monkeypatch.setattr(vllm_main, "needs_local_model_path", Mock(return_value=True))
    monkeypatch.setattr(vllm_main, "fetch_model", fetch_here)
    monkeypatch.setattr(vllm_main, "fetch_model_in_subprocess", fetch_in_subprocess)
    monkeypatch.setattr(vllm_main, "setup_vllm_engine", Mock())

    await vllm_main.worker([])

    assert fetch_in_subprocess.await_count == int(in_subprocess)
    assert fetch_here.await_count == int(not in_subprocess)
    assert config.engine_args.model == (
        "/models/subprocess" if in_subprocess else "/models/here"
    )


@pytest.mark.asyncio
async def test_shutdown_without_artifact_config_does_not_touch_runtime(
    monkeypatch, tmp_path, vllm_main
) -> None:
    async def serve_noop(*_args, **_kwargs):
        return None

    lifecycle, _ = _patch_worker_startup(
        vllm_main, monkeypatch, tmp_path, factory_create=serve_noop
    )
    close_artifact_runtime = AsyncMock()
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )

    await vllm_main.worker([])

    lifecycle.close.assert_awaited_once_with()
    vllm_main.initialize_configured_artifact_runtime.assert_not_awaited()
    close_artifact_runtime.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_hook_keeps_artifact_runtime_open_for_accepted_work(
    monkeypatch, tmp_path, vllm_main
) -> None:
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(tmp_path / "artifact.json"))
    close_artifact_runtime = AsyncMock()
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )
    accepted = asyncio.Event()
    finished = asyncio.Event()

    async def serve_accepted_request(*_args, **_kwargs):
        accepted.set()
        await finished.wait()

    lifecycle, signal_handlers = _patch_worker_startup(
        vllm_main, monkeypatch, tmp_path, factory_create=serve_accepted_request
    )
    worker = asyncio.create_task(vllm_main.worker([]))
    try:
        await accepted.wait()
        callback = signal_handlers.call_args.kwargs["pre_shutdown_callback"]
        await callback()
        lifecycle.close.assert_awaited_once_with()
        close_artifact_runtime.assert_not_awaited()
    finally:
        finished.set()
        await worker

    assert lifecycle.close.await_count == 2
    close_artifact_runtime.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_factory_failure_closes_state_agent_and_artifact_runtime(
    monkeypatch, tmp_path, vllm_main
) -> None:
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(tmp_path / "artifact.json"))
    close_artifact_runtime = AsyncMock()
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )
    lifecycle, _ = _patch_worker_startup(
        vllm_main, monkeypatch, tmp_path, factory_create=RuntimeError("factory failed")
    )

    with pytest.raises(RuntimeError, match="factory failed"):
        await vllm_main.worker([])

    lifecycle.close.assert_awaited_once_with()
    close_artifact_runtime.assert_awaited_once_with()
