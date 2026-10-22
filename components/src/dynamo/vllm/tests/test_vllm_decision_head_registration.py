# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for native-head admission and readiness advertisement."""

import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm import decision_heads as heads
from dynamo.vllm.args import Config
from dynamo.vllm.pooling_handlers import ClassifyWorkerHandler, EmbeddingWorkerHandler
from dynamo.vllm.tests.test_vllm_decision_heads import Engine, NativeApi, monitor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def config():
    return Config.from_cli_args(
        Namespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_rl=False,
            route_to_encoder=False,
            custom_encoder_class=None,
            enable_multimodal=False,
            realtime=False,
        )
    )


@pytest.fixture
def model():
    return SimpleNamespace(
        architecture="D3ForSequenceClassification",
        runner_type="pooling",
        is_encoder_decoder=False,
        max_model_len=8192,
    )


@pytest.mark.parametrize(
    ("architecture", "profile"),
    [
        ("D3ForSequenceClassification", heads.D3_PROFILE),
        ("ClefFlashForJointSchema", heads.CLEF_FLASH_PROFILE),
        ("ClefOmniForJointSchema", heads.CLEF_OMNI_PROFILE),
    ],
)
def test_recognized_head_selects_exact_profile(config, model, architecture, profile):
    model.architecture = architecture
    assert heads.select_decision_head_profile(model, config) == profile


def test_ordinary_pooling_does_not_select_native_head(config, model):
    model.architecture = "OrdinarySequenceClassification"
    assert heads.select_decision_head_profile(model, config) is None
    assert heads.select_decision_head_profile(None, config) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disaggregation_mode", "prefill"),
        ("enable_rl", True),
        ("route_to_encoder", True),
        ("custom_encoder_class", "encoder.Custom"),
        ("enable_multimodal", True),
        ("realtime", True),
    ],
)
def test_unsupported_worker_configuration_fails_closed(config, model, field, value):
    setattr(config, field, value)
    with pytest.raises(heads.HeadRuntimeError, match="worker configuration"):
        heads.select_decision_head_profile(model, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [("runner_type", "generate"), ("is_encoder_decoder", True)],
)
def test_unsupported_model_configuration_fails_closed(config, model, field, value):
    setattr(model, field, value)
    with pytest.raises(heads.HeadRuntimeError, match="pooling model"):
        heads.select_decision_head_profile(model, config)


@pytest.mark.parametrize(
    "profile", [heads.D3_PROFILE, heads.CLEF_FLASH_PROFILE, heads.CLEF_OMNI_PROFILE]
)
def test_ready_profile_publishes_boolean_metadata(profile):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert heads.publish_decision_head_capability(runtime, profile) is True
    runtime.set_engine_specific.assert_called_once_with(profile, "true")


@pytest.mark.parametrize("profile", [None, "unknown", "vllm_causal_decision_v1", True])
def test_unready_or_unknown_profiles_publish_nothing(profile):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert heads.publish_decision_head_capability(runtime, profile) is False
    runtime.set_engine_specific.assert_not_called()


@pytest.fixture
def handler(monkeypatch, config, model):
    def initialize_base(self, *, engine, config, **kwargs):
        self.engine_client = engine
        self.config = config

    monkeypatch.setattr(EmbeddingWorkerHandler, "__init__", initialize_base)
    return ClassifyWorkerHandler(None, object(), config, model)


def test_handler_starts_without_advertised_readiness(handler):
    assert handler.decision_head_adapter is None
    assert handler.decision_head_profile is None


@pytest.mark.asyncio
async def test_handler_initializes_once_before_exposing_profile(monkeypatch, handler):
    adapter = SimpleNamespace(profile=heads.D3_PROFILE)
    create = AsyncMock(return_value=adapter)
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)

    await asyncio.gather(
        handler.initialize_decision_head(), handler.initialize_decision_head()
    )

    assert handler.decision_head_adapter is adapter
    assert handler.decision_head_profile == heads.D3_PROFILE
    create.assert_awaited_once_with(
        handler.engine_client,
        heads.D3_PROFILE,
        source_commit=heads.PINNED_VLLM_COMMIT,
        abort_monitor=handler._abort_monitor,
    )


@pytest.mark.asyncio
async def test_unsupported_runtime_keeps_pooling_available(
    monkeypatch, handler, caplog
):
    create = AsyncMock(side_effect=heads.HeadRuntimeError("pinned source unavailable"))
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)
    caplog.set_level(logging.WARNING)

    await handler.initialize_decision_head()
    await handler.initialize_decision_head()

    assert handler.decision_head_adapter is None
    assert handler.decision_head_profile is None
    create.assert_awaited_once()
    assert "native head decision support disabled" in caplog.text
    assert "pinned source unavailable" in caplog.text


@pytest.mark.asyncio
async def test_unknown_architecture_never_probes_optional_runtime(monkeypatch, handler):
    handler.model_config.architecture = "OrdinarySequenceClassification"
    create = AsyncMock()
    monkeypatch.setattr(heads.HeadExecutionAdapter, "create", create)

    await handler.initialize_decision_head()

    create.assert_not_awaited()
    assert handler.decision_head_profile is None


@pytest.mark.asyncio
async def test_unexpected_initialization_failures_propagate(monkeypatch, handler):
    monkeypatch.setattr(
        heads.HeadExecutionAdapter, "create", AsyncMock(side_effect=ValueError("bug"))
    )
    with pytest.raises(ValueError, match="bug"):
        await handler.initialize_decision_head()
    assert handler.decision_head_profile is None


def test_unready_head_preserves_legacy_health_probe(handler):
    assert handler.decision_head_health_check_payload("model") is None


def test_invalid_native_canary_disables_only_optional_head(handler, caplog):
    adapter = SimpleNamespace(
        profile=heads.D3_PROFILE,
        engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        prepare=Mock(side_effect=heads.HeadValidationError("canary exceeds context")),
    )
    handler.decision_head_adapter = adapter
    caplog.set_level(logging.WARNING)

    assert handler.decision_head_health_check_payload("model") is None

    assert handler.decision_head_adapter is None
    assert handler.decision_head_profile is None
    assert "canary exceeds context" in caplog.text
    adapter.prepare.assert_called_once()


def test_unexpected_native_canary_failure_propagates(handler):
    adapter = SimpleNamespace(
        profile=heads.D3_PROFILE,
        engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192)),
        prepare=Mock(side_effect=ValueError("bug")),
    )
    handler.decision_head_adapter = adapter

    with pytest.raises(ValueError, match="bug"):
        handler.decision_head_health_check_payload("model")

    assert handler.decision_head_adapter is adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "architecture", "task"),
    [
        (heads.D3_PROFILE, "D3ForSequenceClassification", "classify"),
        (heads.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema", "token_classify"),
        (heads.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema", "token_classify"),
    ],
)
async def test_ready_head_health_probe_executes_native_schema(
    monkeypatch, handler, profile, architecture, task
):
    engine = Engine(architecture)
    engine.model_config.is_encoder_decoder = False
    if task == "token_classify":
        engine.values = [0.7, 0.3]
        engine.get_supported_tasks = AsyncMock(return_value=("token_classify",))
    handler.engine_client = engine
    handler.model_config = engine.model_config
    monkeypatch.setattr(handler, "_abort_monitor", monitor)
    monkeypatch.setattr(heads, "_load_native", lambda _: NativeApi())
    await handler.initialize_decision_head()

    payload = handler.decision_head_health_check_payload("served-model")
    body = payload["decision_head"]
    assert body["profile"] == profile
    assert body["model"] == "served-model"
    assert body["temperature"] == 1
    assert body["max_input_tokens"] <= engine.model_config.max_model_len
    assert body["max_tokenization_work"] <= 8192
    assert body["max_encoding_bytes"] <= 16384
    assert body["questions"][0]["ordinal"] == 0
    assert body["questions"][0]["kind"] == "Predicate"
    assert [c["value"] for c in body["questions"][0]["candidates"]] == [True, False]
    request_context = SimpleNamespace(
        id=lambda: "head-health-probe",
        is_stopped=lambda: False,
        is_killed=lambda: False,
        trace_headers=lambda: None,
        append=Mock(),
    )

    response = [item async for item in handler.generate(payload, request_context)]

    assert len(response) == 1
    assert response[0]["outcomes"][0]["ordinal"] == 0
    assert engine.calls[0]["pooling_params"].task == task
    if task == "token_classify":
        assert engine.calls[0]["pooling_params"].extra_kwargs["joint_schema"][
            "option_counts"
        ] == [2]
