# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU readiness and dispatch contracts for model-native generative reads."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import HttpError
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_reads as reads
from dynamo.vllm.args import Config
from dynamo.vllm.decision_dispatch import DECISION_DISPATCH_METADATA_KEY
from dynamo.vllm.decision_surface import NativeDecisionSurface
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


async def initialize(handler, on_engine_dead=None):
    handler.decision = await native_registry.initialize_native_decisions(
        "read",
        handler.engine_client,
        handler.model_config,
        handler.config,
        abort_monitor=handler._abort_monitor,
        on_engine_dead=on_engine_dead,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_shutdown", [False, True])
@pytest.mark.parametrize(
    ("architecture", "profile"),
    [
        ("Lfm2VlForConditionalGeneration", native_registry.LIQUID_PROFILE),
        ("DiffusionGemmaForBlockDiffusion", native_registry.DIFFUSION_PROFILE),
    ],
)
async def test_native_read_ready_only_after_successful_initialization(
    monkeypatch, handler, architecture, profile, with_shutdown
):
    handler.model_config.architecture = architecture
    adapter = SimpleNamespace(
        spec=native_registry.PROFILES[profile],
        kind="read",
        prepare_canary=AsyncMock(return_value=({}, None)),
    )
    create = AsyncMock(return_value=adapter)
    monkeypatch.setattr(reads.ReadExecutionAdapter, "create", create)
    assert handler.decision.publication is None
    shutdown = Mock() if with_shutdown else None
    await initialize(handler, shutdown)
    assert handler.decision.spec == native_registry.PROFILES[profile]
    adapter.prepare_canary.assert_awaited_once_with()
    create.assert_awaited_once_with(
        handler.engine_client,
        native_registry.PROFILES[profile],
        abort_monitor=handler._abort_monitor,
        on_engine_dead=shutdown,
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
    await initialize(handler)
    assert handler.decision.publication is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_read_unknown_or_unavailable_profile_keeps_normal_generation(
    monkeypatch, handler
):
    create = AsyncMock(
        side_effect=reads.NativeRuntimeError("unsupported native runtime")
    )
    monkeypatch.setattr(reads.ReadExecutionAdapter, "create", create)
    await initialize(handler)
    assert handler.decision.spec is None
    assert handler.decision.failed_profile is None
    create.assert_awaited_once()
    handler.model_config.architecture = "Qwen3ForCausalLM"
    await initialize(handler)
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_native_read_private_dispatch_bypasses_token_pipeline(handler):
    context = SimpleNamespace(
        metadata={DECISION_DISPATCH_METADATA_KEY: "read"},
        id=lambda: "native",
        is_stopped=lambda: False,
        is_killed=lambda: False,
        trace_headers=lambda: {"traceparent": "test"},
    )
    wire = {"decision_read": {"protocol_version": 1}}
    execute = AsyncMock(return_value={"profile": native_registry.LIQUID_PROFILE})
    handler.decision = NativeDecisionSurface(
        "read",
        SimpleNamespace(
            spec=native_registry.PROFILES[native_registry.LIQUID_PROFILE],
            kind="read",
            execute=execute,
        ),
    )
    results = [result async for result in handler.generate(wire, context)]
    assert results == [
        {"profile": native_registry.LIQUID_PROFILE, "finish_reason": "stop"}
    ]
    execute.assert_awaited_once()
    assert execute.call_args.args == (wire, context)
    assert execute.call_args.kwargs["request_id"] == "native"
    assert execute.call_args.kwargs["cancelled"]() is False


@pytest.mark.asyncio
async def test_read_initialization_cancellation_propagates(monkeypatch, handler):
    monkeypatch.setattr(
        reads.ReadExecutionAdapter,
        "create",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    with pytest.raises(asyncio.CancelledError):
        await initialize(handler)
    assert handler.decision.publication is None


@pytest.mark.asyncio
async def test_unready_native_read_is_error_not_generation(handler):
    context = SimpleNamespace(metadata={DECISION_DISPATCH_METADATA_KEY: "read"})
    with pytest.raises(HttpError, match="not available") as failure:
        _ = [
            result async for result in handler.generate({"decision_read": {}}, context)
        ]
    assert failure.value.code == 503


@pytest.mark.asyncio
async def test_unexpected_read_initialization_failure_keeps_worker_available(
    monkeypatch, handler, caplog
):
    monkeypatch.setattr(
        reads.ReadExecutionAdapter,
        "create",
        AsyncMock(side_effect=ValueError("adapter bug")),
    )
    await initialize(handler)
    assert handler.decision.spec is None
    assert "adapter bug" in caplog.text
