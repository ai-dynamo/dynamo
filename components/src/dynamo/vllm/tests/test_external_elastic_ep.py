# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the native external Elastic EP control boundary."""

import asyncio
import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.external_elastic_ep import (
    observe_external_elastic_ep,
    scale_external_elastic_ep,
    uses_external_elastic_ep,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
]


@dataclass(frozen=True)
class NativeStatus:
    instance_id: str = "api-0"
    operation_id: str | None = "grow-three"
    epoch: str | None = "native-epoch"
    phase: str = "completed"
    requested_data_parallel_size: int | None = 3
    effective_data_parallel_size: int = 3
    topology_known: bool = True
    error: str | None = None


@pytest.fixture
def engine():
    status = NativeStatus()
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_elastic_ep=True,
                data_parallel_external_lb=True,
            )
        ),
        scale_elastic_ep=AsyncMock(return_value=status),
        get_external_elastic_ep_status=AsyncMock(return_value=status),
    )


@pytest.fixture
def target():
    return {
        "new_data_parallel_size": 3,
        "drain_timeout": 120,
        "operation_id": "grow-three",
        "expected_instance_id": "api-0",
    }


@pytest.mark.asyncio
async def test_scale_forwards_absolute_target_and_native_correlation(engine, target):
    before = copy.deepcopy(target)

    result = await scale_external_elastic_ep(engine, target)

    engine.scale_elastic_ep.assert_awaited_once_with(
        3,
        drain_timeout=120,
        operation_id="grow-three",
        expected_instance_id="api-0",
    )
    assert target == before
    assert result == {
        "instance_id": "api-0",
        "operation_id": "grow-three",
        "epoch": "native-epoch",
        "phase": "completed",
        "requested_data_parallel_size": 3,
        "effective_data_parallel_size": 3,
        "topology_known": True,
        "error": None,
        "is_scaling_elastic_ep": False,
    }
    engine.get_external_elastic_ep_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_replays_same_request_without_local_operation_state(engine, target):
    await scale_external_elastic_ep(engine, target)
    await scale_external_elastic_ep(engine, target)

    assert engine.scale_elastic_ep.await_count == 2
    assert engine.scale_elastic_ep.await_args_list[0] == (
        engine.scale_elastic_ep.await_args_list[1]
    )


@pytest.mark.asyncio
async def test_noop_preserves_previous_native_operation(engine, target):
    engine.scale_elastic_ep.return_value = replace(
        NativeStatus(), operation_id="previous-operation"
    )

    result = await scale_external_elastic_ep(engine, target)

    assert result["operation_id"] == "previous-operation"
    assert "status" not in result  # Never invent a successful correlated commit.


@pytest.mark.parametrize(
    ("phase", "topology_known"),
    [
        ("idle", True),
        ("preparing", True),
        ("committing", False),
        ("completed", True),
        ("failed", False),
    ],
)
@pytest.mark.asyncio
async def test_observation_preserves_native_phase_and_certainty(
    engine, phase, topology_known
):
    native = replace(
        NativeStatus(),
        operation_id=None if phase == "idle" else "grow-three",
        epoch=None if phase == "idle" else "native-epoch",
        requested_data_parallel_size=None if phase == "idle" else 3,
        phase=phase,
        topology_known=topology_known,
        error="native failure" if phase == "failed" else None,
    )
    engine.get_external_elastic_ep_status.return_value = native

    result = await observe_external_elastic_ep(engine)

    assert result["phase"] == phase
    assert result["topology_known"] is topology_known
    assert result["error"] == native.error
    assert result["is_scaling_elastic_ep"] is (phase in {"preparing", "committing"})
    assert "status" not in result
    engine.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("exception", [TimeoutError, ValueError])
@pytest.mark.asyncio
async def test_scale_error_remains_ambiguous_and_status_is_still_observable(
    engine, target, exception
):
    engine.scale_elastic_ep.side_effect = exception("native outcome is inconclusive")

    with pytest.raises(exception, match="inconclusive"):
        await scale_external_elastic_ep(engine, target)

    result = await observe_external_elastic_ep(engine)
    assert result["phase"] == "completed"
    assert engine.scale_elastic_ep.await_count == 1


@pytest.mark.parametrize("value", [None, []])
@pytest.mark.asyncio
async def test_invalid_body_is_rejected_before_native_mutation(engine, value):
    with pytest.raises(TypeError, match="JSON object"):
        await scale_external_elastic_ep(engine, value)
    engine.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("new_data_parallel_size", 0),
        ("new_data_parallel_size", True),
        ("new_data_parallel_size", "3"),
        ("drain_timeout", 0),
        ("drain_timeout", 1.5),
    ],
)
@pytest.mark.asyncio
async def test_invalid_integer_is_rejected_before_native_mutation(
    engine, target, field, value
):
    target[field] = value
    with pytest.raises(ValueError, match=field):
        await scale_external_elastic_ep(engine, target)
    engine.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_id", None),
        ("operation_id", " \t"),
        ("operation_id", "x" * 257),
        ("expected_instance_id", ""),
        ("expected_instance_id", True),
    ],
)
@pytest.mark.asyncio
async def test_invalid_identity_is_rejected_before_native_mutation(
    engine, target, field, value
):
    target[field] = value
    with pytest.raises(ValueError, match=field):
        await scale_external_elastic_ep(engine, target)
    engine.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize(
    "field", ["new_data_parallel_size", "operation_id", "expected_instance_id"]
)
@pytest.mark.asyncio
async def test_missing_required_field_does_not_reach_engine(engine, target, field):
    del target[field]
    with pytest.raises(ValueError, match=field):
        await scale_external_elastic_ep(engine, target)
    engine.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_drain_timeout_matches_native_http_route(engine, target):
    del target["drain_timeout"]
    await scale_external_elastic_ep(engine, target)
    assert engine.scale_elastic_ep.await_args.kwargs["drain_timeout"] == 120


@pytest.mark.parametrize("legacy_status", [None, {"is_scaling_elastic_ep": False}])
@pytest.mark.asyncio
async def test_status_without_correlation_is_not_upgraded_to_evidence(
    engine, legacy_status
):
    engine.get_external_elastic_ep_status.return_value = legacy_status
    with pytest.raises(RuntimeError, match="62082"):
        await observe_external_elastic_ep(engine)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("external", [True, False])
def test_mode_selection_requires_both_external_ownership_and_elastic_ep(
    engine, enabled, external
):
    parallel = engine.vllm_config.parallel_config
    parallel.enable_elastic_ep = enabled
    parallel.data_parallel_external_lb = external
    assert uses_external_elastic_ep(engine) is (enabled and external)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_status_does_not_wait_for_inflight_native_scale(engine, target):
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def scale(*args, **kwargs):
        entered.set()
        await finish.wait()
        return NativeStatus()

    engine.scale_elastic_ep.side_effect = scale
    engine.get_external_elastic_ep_status.return_value = replace(
        NativeStatus(), phase="preparing", effective_data_parallel_size=2
    )
    task = asyncio.create_task(scale_external_elastic_ep(engine, target))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        result = await asyncio.wait_for(observe_external_elastic_ep(engine), timeout=1)
        assert result["phase"] == "preparing"
        assert not task.done()
    finally:
        finish.set()
        await asyncio.wait_for(task, timeout=1)
