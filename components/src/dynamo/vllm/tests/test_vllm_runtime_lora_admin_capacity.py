# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
from dataclasses import replace

import pytest

pytest.importorskip("vllm.lora.request")

from dynamo.common.lora.manager import LoRAInfo
from dynamo.common.lora.runtime import ResolvedLoRA
from dynamo.llm import HttpError
from dynamo.vllm.tests.test_vllm_runtime_lora import _coordinator, _request, _snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


async def _full_runtime(monkeypatch, tmp_path):
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(_snapshot(tmp_path), "digest:immutable")
    )
    coordinator._settings = replace(
        coordinator.settings, max_registered_loras=1, max_resident_runtime_loras=1
    )
    handler._lora_capacity = 1
    request = _request("wandb-artifact:///entity/project/first:v1")
    resident = await coordinator.ensure_from_request(request, "first")
    return handler, coordinator, resident


@pytest.mark.asyncio
async def test_admin_admission_reclaims_idle_runtime_capacity(monkeypatch, tmp_path):
    handler, coordinator, resident = await _full_runtime(monkeypatch, tmp_path)
    try:
        await coordinator.release_pending_admission("first")

        reserved_id = await coordinator.reserve_admin_slot("administrative", 1)

        state = handler._lora_state
        assert reserved_id is not None and reserved_id != resident.lora_int_id
        assert state.loaded_loras == {"administrative": LoRAInfo(id=-1, path="")}
        assert state.admin_reserved_ids == {"administrative": reserved_id}
        assert not state.runtime_loras
        assert not state.runtime_eviction_events
        assert not state.rollback_reserved_ids
        assert resident.lora_name not in handler._engine_loaded_loras
        handler.engine_client.remove_lora.assert_awaited_once_with(resident.lora_int_id)
    finally:
        await coordinator.close()


@pytest.mark.parametrize("occupant", ["pending", "active", "initial", "administrative"])
@pytest.mark.asyncio
async def test_admin_admission_preserves_busy_or_administrative_residency(
    monkeypatch, tmp_path, occupant
):
    handler, coordinator, resident = await _full_runtime(monkeypatch, tmp_path)
    state = handler._lora_state
    try:
        if occupant == "active":
            await coordinator.activate_pending_admission("first", resident)
        elif occupant != "pending":
            await coordinator.release_pending_admission("first")
            if occupant == "initial":
                coordinator._initial_waiters[resident.lora_name] = 1
            else:
                state.runtime_loras.clear()

        assert await coordinator.reserve_admin_slot("administrative", 1) is None

        assert state.loaded_loras[resident.lora_name].id == resident.lora_int_id
        assert not state.admin_reserved_ids
        assert not state.runtime_eviction_events
        handler.engine_client.remove_lora.assert_not_awaited()
    finally:
        await coordinator.close()


@pytest.mark.parametrize("failure", ["cancel", "remove", "restore"])
@pytest.mark.asyncio
async def test_admin_eviction_failure_restores_or_quarantines_runtime(
    monkeypatch, tmp_path, failure
):
    handler, coordinator, resident = await _full_runtime(monkeypatch, tmp_path)
    await coordinator.release_pending_admission("first")
    removal_started = asyncio.Event()
    finish_removal = asyncio.Event()

    async def remove(_lora_id):
        removal_started.set()
        await finish_removal.wait()
        if failure != "cancel":
            raise RuntimeError("removal response lost")
        return True

    handler.engine_client.remove_lora.side_effect = remove
    if failure == "restore":
        handler.engine_client.add_lora.side_effect = RuntimeError("restore failed")
    admission = asyncio.create_task(coordinator.reserve_admin_slot("administrative", 1))
    try:
        await asyncio.wait_for(removal_started.wait(), timeout=2)
        assert "administrative" in handler._lora_state.loaded_loras
        if failure == "cancel":
            admission.cancel()
        finish_removal.set()
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await admission
        else:
            assert await admission is None

        state = handler._lora_state
        assert "administrative" not in state.loaded_loras
        assert not state.admin_reserved_ids
        assert not state.runtime_eviction_events
        if failure == "restore":
            assert not state.loaded_loras
            assert state.uncertain_engine_lora_ids == {resident.lora_int_id}
            assert state.rollback_reserved_ids == {resident.lora_int_id}
        else:
            assert state.loaded_loras[resident.lora_name].id == resident.lora_int_id
            assert resident.lora_name in state.runtime_loras
            assert not state.uncertain_engine_lora_ids
            assert not state.rollback_reserved_ids
    finally:
        finish_removal.set()
        await asyncio.gather(admission, return_exceptions=True)
        await coordinator.close()


@pytest.mark.asyncio
async def test_admin_eviction_reserves_capacity_before_engine_removal(
    monkeypatch, tmp_path
):
    handler, coordinator, _resident = await _full_runtime(monkeypatch, tmp_path)
    await coordinator.release_pending_admission("first")
    removal_started = asyncio.Event()
    finish_removal = asyncio.Event()

    async def remove(_lora_id):
        removal_started.set()
        await finish_removal.wait()
        return True

    handler.engine_client.remove_lora.side_effect = remove
    admission = asyncio.create_task(coordinator.reserve_admin_slot("administrative", 1))
    try:
        await asyncio.wait_for(removal_started.wait(), timeout=2)
        with pytest.raises(HttpError, match="lora_capacity_exceeded"):
            await coordinator.ensure_from_request(
                _request("wandb-artifact:///entity/project/second:v1"), "second"
            )
        assert await coordinator.reserve_admin_slot("another-admin", 1) is None
        finish_removal.set()
        assert await admission is not None
    finally:
        finish_removal.set()
        await asyncio.gather(admission, return_exceptions=True)
        await coordinator.close()


@pytest.mark.asyncio
async def test_admin_eviction_cancelled_before_commit_restores_runtime(
    monkeypatch, tmp_path
):
    handler, coordinator, resident = await _full_runtime(monkeypatch, tmp_path)
    await coordinator.release_pending_admission("first")
    removal_started = asyncio.Event()
    finish_removal = asyncio.Event()
    removed = asyncio.Event()
    original_remove = coordinator._remove_evicted_runtime_loras

    async def remove(_lora_id):
        removal_started.set()
        await finish_removal.wait()
        return True

    async def removed_victims(victims):
        result = await original_remove(victims)
        removed.set()
        return result

    handler.engine_client.remove_lora.side_effect = remove
    coordinator._remove_evicted_runtime_loras = removed_victims
    admission = asyncio.create_task(coordinator.reserve_admin_slot("administrative", 1))
    try:
        await asyncio.wait_for(removal_started.wait(), timeout=2)
        async with handler._lora_capacity_guard:
            finish_removal.set()
            await asyncio.wait_for(removed.wait(), timeout=2)
            admission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await admission

        state = handler._lora_state
        assert state.loaded_loras[resident.lora_name].id == resident.lora_int_id
        assert "administrative" not in state.loaded_loras
        assert not state.admin_reserved_ids
        assert not state.runtime_eviction_events
        assert not state.rollback_reserved_ids
    finally:
        finish_removal.set()
        await asyncio.gather(admission, return_exceptions=True)
        await coordinator.close()
