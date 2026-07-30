# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
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
    return controller


@pytest.mark.asyncio
async def test_prepare_snapshot_consumes_all_dp_warmups_before_readiness(
    snapshot_enabled,
):
    events = []

    async def generate(*args, data_parallel_rank):
        events.append(("first", data_parallel_rank))
        yield object()
        await asyncio.sleep(0)
        events.append(("final", data_parallel_rank))
        yield object()

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
    assert len(set(identifiers)) == 2
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
async def test_warmup_generation_error_propagates_and_prevents_readiness(
    snapshot_enabled,
):
    async def generate(*args, data_parallel_rank):
        if data_parallel_rank == 0:
            raise RuntimeError("generation failed")
        yield object()

    engine = SimpleNamespace(generate=Mock(side_effect=generate))

    with pytest.raises(RuntimeError, match="generation failed"):
        await _prepare(engine)

    snapshot_enabled.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [
        {"embedding_worker": True},
        {"disaggregation_mode": DisaggregationMode.ENCODE},
    ],
)
@pytest.mark.asyncio
async def test_prepare_snapshot_skips_warmup_for_unwarmable_worker_shapes(
    snapshot_enabled, override
):
    engine = SimpleNamespace(generate=Mock())
    engine_setup = _engine_setup(engine, runner_type="generate")
    setup_engine = Mock(return_value=engine_setup)
    snapshot_enabled.return_value.wait_for_restore = AsyncMock(return_value=True)

    result = await prepare_snapshot_engine(_config(**override), setup_engine)

    setup_engine.assert_called_once()
    snapshot_enabled.assert_called_once()
    engine.generate.assert_not_called()
    assert result is snapshot_enabled.return_value


@pytest.mark.asyncio
async def test_warmup_skips_non_generation_engine(caplog):
    engine = SimpleNamespace(generate=Mock())

    with caplog.at_level(logging.INFO, logger=snapshot_mod.__name__):
        await warmup_engine(_engine_setup(engine, runner_type="pooling"))

    engine.generate.assert_not_called()
    assert "Skipping vLLM snapshot warmup for non-generation model" in caplog.text
