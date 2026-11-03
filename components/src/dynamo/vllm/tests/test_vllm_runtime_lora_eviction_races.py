# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import HttpError
from dynamo.vllm import runtime_lora as runtime_mod
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.asyncio
async def test_cold_admission_rechecks_eviction_and_waits_for_rollback(
    monkeypatch, tmp_path, capacity
):
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", str(capacity))
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    first_request = _request("wandb-artifact:///entity/project/first:v1")
    replacement_request = _request("wandb-artifact:///entity/project/replacement:v1")
    first_key = first_request["routing"]["lora_name"]
    replacement_key = replacement_request["routing"]["lora_name"]
    gated = asyncio.Event()
    resume = asyncio.Event()
    replacement_started = asyncio.Event()
    fail_add = asyncio.Event()
    registered = set()
    original_lease = coordinator._lease_resident
    first_late_call = True

    async def gated_lease(metadata, admission_id, **kwargs):
        nonlocal first_late_call
        result = await original_lease(metadata, admission_id, **kwargs)
        if admission_id == "late" and first_late_call:
            first_late_call = False
            assert result is None
            gated.set()
            await resume.wait()
        return result

    async def add_lora(request):
        if request.lora_name == replacement_key:
            replacement_started.set()
            await fail_add.wait()
            raise RuntimeError("replacement failed")
        registered.add(request.lora_int_id)
        return True

    async def remove_lora(lora_id):
        registered.discard(lora_id)
        return True

    coordinator._lease_resident = gated_lease
    handler.engine_client.add_lora.side_effect = add_lora
    handler.engine_client.remove_lora.side_effect = remove_lora
    late = asyncio.create_task(coordinator.ensure_from_request(first_request, "late"))
    await asyncio.wait_for(gated.wait(), timeout=2)
    first = await coordinator.ensure_from_request(first_request, "first")
    await coordinator.release_pending_admission("first")
    if capacity == 2:
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/spare:v1"), "spare"
        )
        await coordinator.release_pending_admission("spare")
    replacement = asyncio.create_task(
        coordinator.ensure_from_request(replacement_request, "replacement")
    )
    await asyncio.wait_for(replacement_started.wait(), timeout=2)
    assert first_key in handler._lora_state.runtime_eviction_events
    resume.set()
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not late.done()
        assert first_key not in handler._lora_state.runtime_load_tasks
        assert runtime_mod.get_lora_manager().resolve_runtime_lora.await_count == (
            capacity + 1
        )
    finally:
        fail_add.set()
        replacement_result, late_result = await asyncio.gather(
            replacement, late, return_exceptions=True
        )
        pending_leases = dict(handler._lora_state.runtime_pending_leases)
        await coordinator.close()
    assert isinstance(replacement_result, HttpError)
    assert replacement_result.code == 500
    assert replacement_result.message == "lora_load_failed"
    assert not isinstance(late_result, BaseException)
    assert late_result.lora_int_id == first.lora_int_id
    assert pending_leases == {first_key: 1}
    state = handler._lora_state
    assert registered == {record.id for record in state.runtime_loras.values()}
    assert state.loaded_loras[first_key].id == first.lora_int_id
    assert not state.runtime_reserved_ids
    assert not state.rollback_reserved_ids
    assert not state.uncertain_engine_lora_ids
    assert not state.runtime_eviction_events
