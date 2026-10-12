# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm import main as vllm_main

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
]


def _patch_worker_startup(monkeypatch, tmp_path, *, factory_create):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = SimpleNamespace(
        embedding_worker_processes=1,
        dump_config_to=None,
        served_model_name="test-model",
        model=str(model_path),
        engine_args=SimpleNamespace(
            model=str(model_path), served_model_name="test-model"
        ),
        headless=False,
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
    return lifecycle, signal_handlers


@pytest.mark.asyncio
async def test_shutdown_without_artifact_config_does_not_touch_runtime(
    monkeypatch,
) -> None:
    lifecycle = SimpleNamespace(close=AsyncMock())
    close_artifact_runtime = AsyncMock()
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )

    await vllm_main._close_worker_resources(lifecycle)

    lifecycle.close.assert_awaited_once_with()
    close_artifact_runtime.assert_not_awaited()


@pytest.mark.asyncio
async def test_signal_hook_closes_state_agent_and_artifact_runtime(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(tmp_path / "artifact.json"))
    close_artifact_runtime = AsyncMock()
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )
    lifecycle, signal_handlers = _patch_worker_startup(
        monkeypatch, tmp_path, factory_create=None
    )

    await vllm_main.worker([])
    callback = signal_handlers.call_args.kwargs["pre_shutdown_callback"]
    await callback()

    assert lifecycle.close.await_count == 2
    assert close_artifact_runtime.await_count == 2


@pytest.mark.asyncio
async def test_factory_failure_closes_state_agent_and_artifact_runtime(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(tmp_path / "artifact.json"))
    close_artifact_runtime = AsyncMock()
    monkeypatch.setattr(
        vllm_main, "close_configured_artifact_runtime", close_artifact_runtime
    )
    lifecycle, _ = _patch_worker_startup(
        monkeypatch, tmp_path, factory_create=RuntimeError("factory failed")
    )

    with pytest.raises(RuntimeError, match="factory failed"):
        await vllm_main.worker([])

    lifecycle.close.assert_awaited_once_with()
    close_artifact_runtime.assert_awaited_once_with()
