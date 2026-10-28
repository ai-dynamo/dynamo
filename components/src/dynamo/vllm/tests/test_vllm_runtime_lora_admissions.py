# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 - optional vLLM dependency is checked before Dynamo imports

import asyncio
from dataclasses import replace

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
async def test_active_release_finishes_through_repeated_cancellation(
    monkeypatch, tmp_path
):
    handler, coordinator = _coordinator(monkeypatch, tmp_path, None)
    key = "runtime-adapter"
    handler._lora_state.runtime_active_leases[key] = 1
    await handler._lora_capacity_guard.acquire()
    release = asyncio.create_task(coordinator.release_active_admission(key))
    await asyncio.sleep(0)
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    await asyncio.sleep(0)
    try:
        assert not release.done()
    finally:
        handler._lora_capacity_guard.release()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert not handler._lora_state.runtime_active_leases


@pytest.mark.asyncio
async def test_owned_pending_guard_disarms_expiry_before_activation(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    coordinator._settings = replace(
        coordinator.settings, pending_admission_timeout_seconds=0
    )
    async with coordinator.pending_admission_guard("owned"):
        result = await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/adapter:v1"), "owned"
        )
        for _ in range(3):
            await asyncio.sleep(0)
        assert handler._lora_state.runtime_pending_leases == {result.lora_name: 1}
        await coordinator.activate_pending_admission("owned", result)
        await coordinator.release_active_admission(result.lora_name)
    assert not handler._lora_state.runtime_pending_admissions
    await coordinator.close()


@pytest.mark.parametrize("failure", [TypeError, RecursionError, OSError])
@pytest.mark.asyncio
async def test_unexpected_validation_failures_are_redacted(
    monkeypatch, tmp_path, failure
):
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )

    def validate(*_args, **_kwargs):
        raise failure("provider-secret/cache/path")

    monkeypatch.setattr(runtime_mod, "validate_snapshot", validate)
    with pytest.raises(HttpError, match="invalid_lora_adapter") as error:
        await coordinator.ensure_from_request(
            _request("wandb-artifact:///entity/project/adapter:v1"), "request"
        )
    assert error.value.code == 422
    assert "provider-secret" not in str(error.value)
    handler.engine_client.add_lora.assert_not_awaited()
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


@pytest.mark.asyncio
async def test_warm_admission_preserves_reserved_initial_waiter_budget(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def resolve(**kwargs):
        if "cold" in kwargs["source_uri"]:
            started.set()
            await release.wait()
        return ResolvedLoRA(snapshot, "digest:immutable")

    handler, coordinator = _coordinator(monkeypatch, tmp_path, resolve)
    coordinator._settings = replace(coordinator.settings, max_pending_admissions=1)
    warm_request = _request("wandb-artifact:///entity/project/warm:v1")
    await coordinator.ensure_from_request(warm_request, "prime")
    await coordinator.release_pending_admission("prime")
    cold_request = _request("wandb-artifact:///entity/project/cold:v1")
    waiter = asyncio.create_task(coordinator.ensure_from_request(cold_request, "cold"))
    await started.wait()
    try:
        with pytest.raises(HttpError, match="lora_capacity_exceeded"):
            await coordinator.ensure_from_request(warm_request, "warm")
        assert not handler._lora_state.runtime_pending_admissions
        assert sum(coordinator._initial_waiters.values()) == 1
    finally:
        release.set()
    result = await waiter
    assert handler._lora_state.runtime_pending_leases == {result.lora_name: 1}
    assert not coordinator._initial_waiters
    await coordinator.close()


@pytest.mark.parametrize("routing", [None, {}])
@pytest.mark.asyncio
async def test_model_runtime_key_requires_source_metadata(
    monkeypatch, tmp_path, routing
):
    snapshot = _snapshot(tmp_path)
    handler, coordinator = _coordinator(
        monkeypatch, tmp_path, ResolvedLoRA(snapshot, "digest:immutable")
    )
    runtime = await coordinator.ensure_from_request(
        _request("wandb-artifact:///entity/project/adapter:v1"), "first"
    )
    await coordinator.release_pending_admission("first")
    request = {"model": runtime.lora_name}
    if routing is not None:
        request["routing"] = routing
    with pytest.raises(HttpError, match="invalid_lora_model_id") as error:
        await coordinator.ensure_from_request(request, "bypass")
    assert error.value.code == 400
    assert not handler._lora_state.runtime_pending_admissions
    assert not handler._lora_state.runtime_active_leases
    await coordinator.close()


@pytest.mark.asyncio
async def test_registered_base_with_runtime_prefix_keeps_exact_name_precedence(
    monkeypatch, tmp_path
):
    handler, coordinator = _coordinator(monkeypatch, tmp_path, None)
    handler._served_model_name = "dyn-lora-canonical-base"
    assert (
        await coordinator.ensure_from_request(
            {"model": handler._served_model_name}, "base"
        )
        is None
    )
    await coordinator.close()
