# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native feature ownership is independent of mandatory worker handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.llm.exceptions import HttpError, WorkerUnavailable
from dynamo.runtime import Context
from dynamo.vllm import decision_native_lifecycle
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm.decision_dispatch import DECISION_DISPATCH_METADATA_KEY
from dynamo.vllm.decision_heads import HeadExecutionAdapter, PreparedHeadRead
from dynamo.vllm.decision_native_errors import NativeExecutionError
from dynamo.vllm.decision_surface import NativeDecisionSurface, PoolingProbe
from dynamo.vllm.tests.decision_head_test_support import Engine, NativeApi, monitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_unavailable_surface_rejects_native_wire_without_fallthrough(kind):
    surface = NativeDecisionSurface(kind)
    context = SimpleNamespace(metadata={DECISION_DISPATCH_METADATA_KEY: kind})
    handled = surface.maybe_handle({f"decision_{kind}": {}}, context)
    assert handled is not None
    with pytest.raises(HttpError) as error:
        [row async for row in handled]
    assert error.value.code == 503
    assert surface.maybe_handle({"input": "ordinary"}, object()) is None
    assert surface.spec is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["head", "read"])
async def test_native_surface_marks_its_result_complete_before_stream_eof(kind):
    result = {"protocol_version": 1, "outcomes": [], "usage": {}}
    adapter = SimpleNamespace(kind=kind, execute=AsyncMock(return_value=result))
    surface = NativeDecisionSurface(kind, adapter)
    context = SimpleNamespace(
        id=lambda: "native-terminal",
        is_stopped=lambda: False,
        is_killed=lambda: False,
        trace_headers=lambda: {},
    )
    stream = surface._execute({}, context)
    terminal = await anext(stream)
    assert terminal == {**result, "finish_reason": "stop"}
    assert "finish_reason" not in result
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1])
async def test_adapter_canary_owns_fixed_payload_and_single_head_read_contract(count):
    from dynamo.vllm.decision_native_lifecycle import NativeAdapter

    prepared = SimpleNamespace(reads=[object()] * count)
    instance = SimpleNamespace(
        kind="head",
        spec=native_registry.PROFILES[native_registry.D3_PROFILE],
        engine=Engine(),
        _prepare=AsyncMock(return_value=prepared),
        _check_canary=lambda prepared: HeadExecutionAdapter._check_canary(
            None, prepared
        ),
    )
    if not count:
        with pytest.raises(NativeExecutionError, match="one pooling read"):
            await NativeAdapter.prepare_canary(instance)
    else:
        wire, actual = await NativeAdapter.prepare_canary(instance)
        assert actual is prepared
        assert wire["decision_head"]["profile"] == instance.spec.name
        assert (
            wire["decision_head"]["cache_salt"]
            == (decision_native_lifecycle.NATIVE_HEALTH_CACHE_SALT)
            == "dynamo-native-head-health-v1"
        )
        assert wire["decision_head"]["questions"][0]["kind"] == "Predicate"
        instance._prepare.assert_awaited_once_with(wire)
    assert instance.engine.calls == []


@pytest.mark.parametrize("kind", ["head", "read"])
@pytest.mark.parametrize("metadata", [{}, {DECISION_DISPATCH_METADATA_KEY: "other"}])
def test_public_body_cannot_select_private_native_execution(kind, metadata):
    surface = NativeDecisionSurface(kind)
    context = SimpleNamespace(metadata=metadata)
    assert surface.maybe_handle({f"decision_{kind}": {}}, context) is None
    assert surface.maybe_handle({"native_pooling_health": True}, context) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        {
            "native_pooling_health": True,
            "model": "test",
            HEALTH_CHECK_KEY: True,
            "input": "unexpected",
        },
        {"native_pooling_health": False, "model": "test", HEALTH_CHECK_KEY: True},
        {"native_pooling_health": True, "model": 1, HEALTH_CHECK_KEY: True},
    ],
)
async def test_internal_probe_rejects_invalid_controls(wire):
    with pytest.raises(ValueError, match="invalid internal pooling health"):
        [
            row
            async for row in NativeDecisionSurface("head").maybe_handle(wire, object())
        ]


@pytest.mark.asyncio
async def test_uninitialized_internal_probe_is_unavailable():
    wire = {"native_pooling_health": True, "model": "test", HEALTH_CHECK_KEY: True}
    with pytest.raises(WorkerUnavailable):
        [
            row
            async for row in NativeDecisionSurface("head").maybe_handle(wire, object())
        ]


@pytest.mark.asyncio
async def test_raw_probe_copies_mutable_pooling_parameters_for_each_call(monkeypatch):
    params = SimpleNamespace(
        extra_kwargs={"joint_schema": {"option_counts": [2]}}, requires_token_ids=False
    )
    read = PreparedHeadRead((1, 2), params, (), (), ())
    engine = Engine()
    original = engine.encode

    async def encode(**kwargs):
        received = kwargs["pooling_params"]
        assert received is not params
        assert received.extra_kwargs["joint_schema"]["option_counts"] == [2]
        received.extra_kwargs["joint_schema"]["option_counts"].clear()
        received.requires_token_ids = True
        async for output in original(**kwargs):
            yield output

    monkeypatch.setattr(engine, "encode", encode)
    monkeypatch.setattr(decision_native_lifecycle, "load_native", lambda: NativeApi())
    adapter = await HeadExecutionAdapter.create(
        engine,
        native_registry.PROFILES[native_registry.D3_PROFILE],
        abort_monitor=monitor,
    )
    adapter.execute = AsyncMock(side_effect=AssertionError("not health"))
    adapter._decode = Mock()
    surface = NativeDecisionSurface("head", adapter, PoolingProbe(read, adapter))
    payload = {**surface.health_payload("model"), HEALTH_CHECK_KEY: True}
    for _ in range(2):
        assert [row async for row in surface.maybe_handle(payload, Context())] == [
            {"status": "ok"}
        ]
    assert params.requires_token_ids is False
    assert params.extra_kwargs["joint_schema"]["option_counts"] == [2]
    adapter.execute.assert_not_awaited()
    adapter._decode.assert_not_called()


@pytest.mark.parametrize(
    "kind,adapter",
    [("read", SimpleNamespace(kind="head")), ("head", SimpleNamespace(kind="read"))],
)
def test_surface_rejects_mismatched_readiness(kind, adapter):
    with pytest.raises(ValueError, match="does not match"):
        NativeDecisionSurface(kind, adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [[], [False], [True, True]])
async def test_pooling_liveness_still_requires_one_completed_engine_result(
    monkeypatch, finished
):
    engine = Engine()

    async def encode(**kwargs):
        for value in finished:
            yield SimpleNamespace(
                request_id=kwargs["request_id"], finished=value, error=None
            )

    monkeypatch.setattr(engine, "encode", encode)
    monkeypatch.setattr(decision_native_lifecycle, "load_native", lambda: NativeApi())
    adapter = await HeadExecutionAdapter.create(
        engine,
        native_registry.PROFILES[native_registry.D3_PROFILE],
        abort_monitor=monitor,
    )
    probe = PoolingProbe(PreparedHeadRead((1,), object(), (), (), ()), adapter)
    surface = NativeDecisionSurface("head", pooling_probe=probe)
    payload = {**surface.health_payload("model"), HEALTH_CHECK_KEY: True}
    with pytest.raises(NativeExecutionError):
        [row async for row in surface.maybe_handle(payload, Context())]


def test_read_surface_rejects_a_pooling_liveness_probe():
    with pytest.raises(ValueError, match="pooling health requires a head"):
        NativeDecisionSurface("read", pooling_probe=PoolingProbe(None, None))


@pytest.mark.parametrize(
    "kind,adapter", [("head", SimpleNamespace(kind="head")), ("read", None)]
)
def test_failed_head_profile_cannot_coexist_with_readiness_or_read_kind(kind, adapter):
    with pytest.raises(ValueError, match="failure conflicts"):
        NativeDecisionSurface(
            kind,
            adapter,
            failed_profile=native_registry.PROFILES[native_registry.D3_PROFILE],
        )
