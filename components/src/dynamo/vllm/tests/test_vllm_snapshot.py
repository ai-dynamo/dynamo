# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm import snapshot as snapshot_mod
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.snapshot import prepare_snapshot_engine, warmup_engine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_setup(engine, runner_type="generate"):
    config = SimpleNamespace(model_config=SimpleNamespace(runner_type=runner_type))
    return engine, config


def _config(**overrides):
    values = {
        "headless": False,
        "embedding_worker": False,
        "disaggregation_mode": DisaggregationMode.AGGREGATED,
        "engine_args": SimpleNamespace(enable_sleep_mode=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _output(finished=True):
    outputs = [SimpleNamespace(token_ids=[7, 8])] if finished else []
    return SimpleNamespace(finished=finished, outputs=outputs)


async def _prepare(engine):
    return await prepare_snapshot_engine(_config(), lambda _: _engine_setup(engine))


@pytest.fixture
def snapshot_enabled(monkeypatch):
    controller = Mock()
    monkeypatch.setattr(
        snapshot_mod.SnapshotConfig, "from_env", Mock(return_value=object())
    )
    monkeypatch.setattr(snapshot_mod, "configure_snapshot_capture_env", lambda: None)
    monkeypatch.setattr(snapshot_mod, "EngineSnapshotController", controller)
    monkeypatch.setattr(
        snapshot_mod, "get_dp_range_for_worker", Mock(return_value=(4, 2))
    )
    monkeypatch.setattr(snapshot_mod, "_WARMUP_TIMEOUT_SEC", 0.5)
    return controller


@pytest.mark.asyncio
async def test_prepare_snapshot_consumes_all_dp_warmups_before_readiness(
    snapshot_enabled,
):
    events = []

    async def generate(*args, data_parallel_rank):
        events.append(("first", data_parallel_rank))
        yield _output(False)
        await asyncio.sleep(0)
        events.append(("final", data_parallel_rank))
        yield _output()

    engine = SimpleNamespace(generate=Mock(side_effect=generate))
    snapshot_enabled.return_value.wait_for_restore = AsyncMock(
        side_effect=lambda: events.append("ready") or True
    )

    await _prepare(engine)

    assert events == [
        ("first", 0),
        ("first", 1),
        ("final", 0),
        ("final", 1),
        "ready",
    ]
    calls = engine.generate.call_args_list
    assert {call.kwargs["data_parallel_rank"] for call in calls} == {0, 1}
    identifiers = [call.args[2] for call in calls]
    identifiers += [call.args[0]["cache_salt"] for call in calls]
    assert len(set(identifiers)) == 4
    for call in calls:
        prompt, sampling_params, _ = call.args
        assert prompt["prompt_token_ids"] == [1, 2, 3]
        assert (
            sampling_params.max_tokens,
            sampling_params.temperature,
            sampling_params.ignore_eos,
            sampling_params.detokenize,
        ) == (2, 0.0, True, False)


@pytest.mark.asyncio
async def test_warmup_timeout_aborts_all_ranks_and_prevents_readiness(
    monkeypatch, snapshot_enabled
):
    cancelled_ranks = set()

    async def generate(*args, data_parallel_rank):
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            cancelled_ranks.add(data_parallel_rank)
            raise
        yield

    engine = SimpleNamespace(
        generate=Mock(side_effect=generate),
        abort=AsyncMock(side_effect=[RuntimeError("abort failed"), None]),
    )
    monkeypatch.setattr(snapshot_mod, "_WARMUP_TIMEOUT_SEC", 0.01)

    with pytest.raises(asyncio.TimeoutError):
        await _prepare(engine)

    assert cancelled_ranks == {0, 1}
    assert {call.args[0] for call in engine.abort.await_args_list} == {
        call.args[2] for call in engine.generate.call_args_list
    }
    snapshot_enabled.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_rank_failure_waits_for_peers_and_prevents_readiness(
    snapshot_enabled,
):
    completed_ranks = set()

    async def generate(*args, data_parallel_rank):
        if data_parallel_rank == 0:
            raise RuntimeError("generation failed")
        yield _output()
        completed_ranks.add(data_parallel_rank)

    engine = SimpleNamespace(generate=Mock(side_effect=generate))

    with pytest.raises(RuntimeError, match="generation failed"):
        await _prepare(engine)

    assert completed_ranks == {1}
    snapshot_enabled.assert_not_called()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"embedding_worker": True}, "--embedding-worker"),
        ({"disaggregation_mode": DisaggregationMode.ENCODE}, "encode workers"),
    ],
)
@pytest.mark.asyncio
async def test_prepare_snapshot_rejects_non_generation_worker_shapes(
    snapshot_enabled, override, expected
):
    setup_engine = Mock()

    with pytest.raises(ValueError, match=expected):
        await prepare_snapshot_engine(_config(**override), setup_engine)

    setup_engine.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_rejects_pooling_engine():
    with pytest.raises(ValueError, match="runner_type='pooling'"):
        await warmup_engine(_engine_setup(object(), runner_type="pooling"))
