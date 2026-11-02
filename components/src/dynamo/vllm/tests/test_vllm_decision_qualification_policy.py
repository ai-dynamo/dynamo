# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup stage, eligibility and retained-liveness policy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.llm.exceptions import HttpError
from dynamo.vllm import decision_heads, decision_native_lifecycle
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_surface
from dynamo.vllm.decision_native_errors import NativeExecutionError, NativeRuntimeError
from dynamo.vllm.decision_native_registry import D3_PROFILE, PROFILES
from dynamo.vllm.tests.decision_head_test_support import Engine, NativeApi, monitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def qualification(monkeypatch):
    profile = PROFILES[D3_PROFILE]
    prepared = SimpleNamespace(reads=[object()])
    adapter = SimpleNamespace(
        kind="head",
        spec=profile,
        prepare_canary=AsyncMock(return_value=({}, prepared)),
        verify_liveness=AsyncMock(),
        execute=AsyncMock(),
    )
    create = AsyncMock(return_value=adapter)
    monkeypatch.setattr(decision_heads.HeadExecutionAdapter, "create", create)
    monkeypatch.setattr(native_registry, "decision_worker_eligible", lambda _: True)
    return profile, adapter, create


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["construct", "prepare", "prove", "canary"])
@pytest.mark.parametrize("error", ["incompatible", "transient", "timeout", "contract"])
async def test_qualification_classifies_errors_by_stage(qualification, stage, error):
    profile, adapter, create = qualification
    cause = {
        "incompatible": NativeRuntimeError("incompatible"),
        "transient": HttpError(503, "unavailable"),
        "timeout": TimeoutError(),
        "contract": NativeExecutionError("contract"),
    }[error]
    operation = {
        "construct": create,
        "prepare": adapter.prepare_canary,
        "prove": adapter.verify_liveness,
        "canary": adapter.execute,
    }[stage]
    operation.side_effect = cause
    surface = await native_registry.initialize_native_decisions(
        "head",
        object(),
        SimpleNamespace(architecture=profile.architecture),
        object(),
        abort_monitor=monitor,
    )
    assert surface.spec is None
    transient = error in {"transient", "timeout"}
    failed = not transient and not (error == "incompatible" and stage == "construct")
    assert surface.failed_profile == (profile if failed else None)
    assert (surface.pooling_probe is not None) == (stage == "canary")
    assert adapter.execute.await_count == (
        2 if stage == "canary" and transient else int(stage == "canary")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["construct", "prepare", "prove", "canary"])
async def test_qualification_deadline_retains_only_completed_liveness(
    monkeypatch, qualification, stage
):
    profile, adapter, create = qualification
    drained = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    {
        "construct": create,
        "prepare": adapter.prepare_canary,
        "prove": adapter.verify_liveness,
        "canary": adapter.execute,
    }[stage].side_effect = blocked
    monkeypatch.setattr(
        decision_surface.EngineHealthMonitorConfig,
        "from_env",
        lambda: SimpleNamespace(check_timeout=0.01),
    )
    surface = await native_registry.initialize_native_decisions(
        "head",
        object(),
        SimpleNamespace(architecture=profile.architecture),
        object(),
        abort_monitor=monitor,
    )
    assert drained.is_set()
    assert surface.spec is surface.failed_profile is None
    assert (surface.pooling_probe is not None) == (stage == "canary")
    assert adapter.execute.await_count == int(stage == "canary")


@pytest.mark.asyncio
async def test_read_canary_is_kind_neutral_and_calls_profile_check():
    prepared = object()
    checked = []
    adapter = SimpleNamespace(
        kind="read",
        spec=SimpleNamespace(name="read-profile"),
        engine=Engine(),
        _prepare=AsyncMock(return_value=prepared),
        _check_canary=checked.append,
    )
    wire, actual = await decision_native_lifecycle.NativeAdapter.prepare_canary(adapter)
    assert actual is prepared
    assert wire["decision_read"]["model"] == "dynamo-native-canary"
    assert checked == [prepared]


@pytest.mark.asyncio
async def test_profile_initialization_runs_after_all_base_fields(monkeypatch):
    initialized = []

    class CheckedHead(decision_heads.HeadExecutionAdapter):
        def _initialize_profile(self):
            assert self.preflight_tokenizer is not self.engine.renderer.get_tokenizer()
            assert self.preflight_lock is not None
            assert self.semaphore is not None
            assert self.spec.name == D3_PROFILE
            assert self.spec.max_candidates == 3
            super()._initialize_profile()
            initialized.append(self.decoder)

    monkeypatch.setattr(decision_native_lifecycle, "load_native", lambda: NativeApi())
    adapter = await CheckedHead.create(
        Engine(), PROFILES[D3_PROFILE], abort_monitor=monitor
    )
    assert initialized == [adapter.decoder]
