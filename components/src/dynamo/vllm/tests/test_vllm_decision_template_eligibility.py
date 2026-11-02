# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction-time chat-template incompatibility is optional ineligibility."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jinja2 import TemplateError

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.llm.exceptions import EngineShutdown
from dynamo.vllm import decision_native_registry as registry
from dynamo.vllm.tests.decision_read_contract_support import monitor
from dynamo.vllm.tests.decision_read_test_support import Engine, Tokenizer

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def worker_config():
    return SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("diffusion", [False, True])
async def test_construction_template_error_leaves_no_read_capability_or_failure(
    monkeypatch, native_reads, diffusion, worker_config
):
    probes = []

    def incompatible_template(self, messages, **kwargs):
        probes.append(messages)
        raise TemplateError("template rejects the generation-prompt probe")

    monkeypatch.setattr(Tokenizer, "apply_chat_template", incompatible_template)
    engine = Engine(diffusion=diffusion)
    surface = await registry.initialize_native_decisions(
        "read",
        engine,
        engine.model_config,
        worker_config,
        abort_monitor=monitor,
    )
    assert len(probes) == 1
    assert surface.spec is surface.failed_profile is None
    assert surface.pooling_probe is None
    assert engine.calls == engine.admissions == []
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert not registry.publish_native_decision_capability(
        runtime,
        surface.publication,
        ModelInput.Tokens,
        ModelType.Chat,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize("error", [EngineShutdown, asyncio.CancelledError])
async def test_construction_template_boundary_preserves_shutdown_and_cancellation(
    monkeypatch, native_reads, diffusion, error, worker_config
):
    failure = error()

    def stopped(self, messages, **kwargs):
        raise failure

    monkeypatch.setattr(Tokenizer, "apply_chat_template", stopped)
    engine = Engine(diffusion=diffusion)
    with pytest.raises(error) as raised:
        await registry.initialize_native_decisions(
            "read",
            engine,
            engine.model_config,
            worker_config,
            abort_monitor=monitor,
        )
    assert raised.value is failure
    assert engine.calls == engine.admissions == []
