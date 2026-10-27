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


@pytest.mark.asyncio
async def test_duplicate_correlation_ids_have_independent_admissions(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    contexts = []

    async def resolve(**kwargs):
        contexts.append(kwargs["context"])
        return ResolvedLoRA(snapshot, "digest:immutable")

    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    request = _request("wandb-artifact:///entity/project/adapter:v1")
    first = await coordinator.ensure_from_request(
        request, "shared-client-id", admission_id="internal-first"
    )
    second = await coordinator.ensure_from_request(
        request, "shared-client-id", admission_id="internal-second"
    )
    assert set(handler._lora_state.runtime_pending_admissions) == {
        "internal-first",
        "internal-second",
    }
    assert handler._lora_state.runtime_pending_leases == {first.lora_name: 2}
    assert contexts[0].request_id == "shared-client-id"

    await coordinator.release_pending_admission("internal-first")
    admitted = await coordinator.activate_pending_admission("internal-second", second)
    assert admitted.lora_name == second.lora_name
    assert handler._lora_state.runtime_active_leases == {second.lora_name: 1}
    assert not handler._lora_state.runtime_pending_leases
    await coordinator.release_active_admission(second.lora_name)
    await coordinator.close()
