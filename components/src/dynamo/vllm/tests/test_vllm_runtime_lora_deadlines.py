# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import HttpError
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_cold_load_timeout_retains_engine_ownership_until_late_completion(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    coordinator._settings = replace(coordinator.settings, resolve_timeout_seconds=0.1)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def add_lora(_request):
        started.set()
        await finish.wait()
        return True

    handler.engine_client.add_lora.side_effect = add_lora
    request = _request("wandb-artifact:///entity/project/adapter:v1")
    waiter = asyncio.create_task(coordinator.ensure_from_request(request, "first"))
    await asyncio.wait_for(started.wait(), timeout=2)
    load_task = next(iter(handler._lora_state.runtime_load_tasks.values()))
    try:
        with pytest.raises(HttpError, match="runtime_lora_resolve_timeout") as error:
            await asyncio.wait_for(waiter, timeout=1)
        assert error.value.code == 504
        assert not load_task.done()
        assert handler._pause_lock.locked()
        assert handler._lora_state.runtime_reserved_ids
        assert not handler._lora_state.runtime_pending_admissions
        assert not coordinator._initial_waiters
        handler.engine_client.remove_lora.assert_not_awaited()
    finally:
        finish.set()
        await load_task

    result = await coordinator.ensure_from_request(request, "late-client")
    assert result.lora_name == request["routing"]["lora_name"]
    handler.engine_client.add_lora.assert_awaited_once()
    assert not handler._pause_lock.locked()
    assert not handler._lora_state.runtime_reserved_ids
    await coordinator.close()


@pytest.mark.asyncio
async def test_cold_load_wait_preserves_task_timeout_error(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    _handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    coordinator._resolve_validate_and_load = AsyncMock(
        side_effect=TimeoutError("task-owned timeout")
    )
    with pytest.raises(TimeoutError, match="task-owned timeout"):
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/adapter:v1"), "request"
        )
    assert not coordinator._initial_waiters
    await coordinator.close()
