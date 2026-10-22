# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU readiness and dispatch contracts for model-native generative reads."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm import decision_reads as reads
from dynamo.vllm.args import Config
from dynamo.vllm.handlers import BaseWorkerHandler, DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def handler(monkeypatch):
    def base_init(self, runtime, config, engine, default_sampling_params, **kwargs):
        self.config = config
        self.engine_client = engine
        self.model_config = kwargs["model_config"]

    monkeypatch.setattr(BaseWorkerHandler, "__init__", base_init)
    config = Config.from_cli_args(
        Namespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_rl=False,
            route_to_encoder=False,
            custom_encoder_class=None,
            enable_multimodal=False,
            realtime=False,
        )
    )
    model = SimpleNamespace(architecture="Lfm2VlForConditionalGeneration")
    return DecodeWorkerHandler(None, config, object(), None, model_config=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("architecture", "profile"),
    [
        ("Lfm2VlForConditionalGeneration", reads.LIQUID_PROFILE),
        ("DiffusionGemmaForBlockDiffusion", reads.DIFFUSION_PROFILE),
    ],
)
async def test_native_read_ready_only_after_successful_initialization(
    monkeypatch, handler, architecture, profile
):
    handler.model_config.architecture = architecture
    adapter = SimpleNamespace(profile=profile)
    create = AsyncMock(return_value=adapter)
    monkeypatch.setattr(reads.ReadExecutionAdapter, "create", create)
    assert handler.decision_read_profile is None
    await asyncio.gather(
        handler.initialize_decision_read(), handler.initialize_decision_read()
    )
    assert handler.decision_read_profile == profile
    create.assert_awaited_once_with(
        handler.engine_client,
        profile,
        source_commit=reads.PINNED_VLLM_COMMIT,
        abort_monitor=handler._abort_monitor,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disaggregation_mode", DisaggregationMode.PREFILL),
        ("enable_rl", True),
        ("route_to_encoder", True),
        ("custom_encoder_class", "encoder.Custom"),
        ("enable_multimodal", True),
        ("realtime", True),
    ],
)
@pytest.mark.parametrize(
    "architecture",
    ["Lfm2VlForConditionalGeneration", "DiffusionGemmaForBlockDiffusion"],
)
async def test_native_read_unsupported_worker_does_not_advertise(
    monkeypatch, handler, field, value, architecture
):
    handler.model_config.architecture = architecture
    setattr(handler.config, field, value)
    create = AsyncMock()
    monkeypatch.setattr(reads.ReadExecutionAdapter, "create", create)
    await handler.initialize_decision_read()
    assert handler.decision_read_profile is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_read_unknown_or_unavailable_profile_keeps_normal_generation(
    monkeypatch, handler
):
    create = AsyncMock(side_effect=reads.ReadRuntimeError("unsupported native runtime"))
    monkeypatch.setattr(reads.ReadExecutionAdapter, "create", create)
    await handler.initialize_decision_read()
    assert handler.decision_read_profile is None
    create.assert_awaited_once()
    handler._decision_read_initialization_attempted = False
    handler.model_config.architecture = "Qwen3ForCausalLM"
    await handler.initialize_decision_read()
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_native_read_private_dispatch_bypasses_token_pipeline(handler):
    context = SimpleNamespace(
        id=lambda: "native",
        is_stopped=lambda: False,
        is_killed=lambda: False,
        trace_headers=lambda: {"traceparent": "test"},
    )
    wire = {"decision_read": {"protocol_version": 1}}
    execute = AsyncMock(return_value={"profile": reads.LIQUID_PROFILE})
    handler.decision_read_adapter = SimpleNamespace(
        profile=reads.LIQUID_PROFILE, execute=execute
    )
    results = [result async for result in handler.generate(wire, context)]
    assert results == [{"profile": reads.LIQUID_PROFILE}]
    execute.assert_awaited_once()
    assert execute.call_args.args == (wire, context)
    assert execute.call_args.kwargs["request_id"] == "native"
    assert execute.call_args.kwargs["cancelled"]() is False


@pytest.mark.asyncio
async def test_unready_native_read_is_error_not_generation(handler):
    with pytest.raises(reads.ReadValidationError, match="not available"):
        _ = [result async for result in handler.generate({"decision_read": {}}, Mock())]
