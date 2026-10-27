# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
from dataclasses import replace

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
async def test_initial_waiters_share_pending_admission_limit(monkeypatch, tmp_path):
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def resolve(**_kwargs):
        started.set()
        await release.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    coordinator._settings = replace(coordinator.settings, max_pending_admissions=1)
    request = _request("wandb-artifact:///entity/project/adapter:v1")
    first = asyncio.create_task(coordinator.ensure_from_request(request, "first"))
    await started.wait()
    try:
        with pytest.raises(HttpError, match="lora_capacity_exceeded"):
            await coordinator.ensure_from_request(request, "second")
        assert sum(coordinator._initial_waiters.values()) == 1
    finally:
        release.set()
    result = await first
    assert handler._lora_state.runtime_pending_leases == {result.lora_name: 1}
    assert not coordinator._initial_waiters
    await coordinator.close()
