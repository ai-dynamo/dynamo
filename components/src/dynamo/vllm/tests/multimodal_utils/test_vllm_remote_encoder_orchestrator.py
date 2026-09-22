# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from examples.custom_encoder.remote.orchestrator import ExternalEncoderOrchestrator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Handoff:
    def __init__(self, prepared_request: Mapping[str, Any]) -> None:
        self.prepared_request = prepared_request
        self.request: Mapping[str, Any] | None = None
        self.target_model: str | None = None

    async def prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        target_model: str,
    ) -> Mapping[str, Any]:
        self.request = request
        self.target_model = target_model
        return self.prepared_request


class _Generator:
    def __init__(self, completion: dict[str, Any]) -> None:
        self.completion = completion
        self.request: Mapping[str, Any] | None = None
        self.context: Any = None

    async def complete(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        self.request = request
        self.context = context
        return self.completion


async def test_orchestrator_prepares_then_completes_request() -> None:
    prepared_request = {
        "model": "generator-model",
        "token_ids": [1, 99, 2],
        "encoder_result": {"opaque": "to-the-orchestrator"},
    }
    completion = {
        "token_ids": [7, 8],
        "index": 0,
        "finish_reason": "stop",
    }
    handoff = _Handoff(prepared_request)
    generator = _Generator(completion)
    orchestrator = ExternalEncoderOrchestrator(
        handoff,
        generator,
        "generator-model",
    )
    request = {
        "token_ids": [1, 99, 2],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/image.png"}]},
    }
    context = object()

    result = await orchestrator(request, context=context)

    assert result is completion
    assert handoff.request is request
    assert handoff.target_model == "generator-model"
    assert generator.request is prepared_request
    assert generator.context is context


def test_orchestrator_requires_generator_model_name() -> None:
    with pytest.raises(ValueError, match="generator_model_name"):
        ExternalEncoderOrchestrator(_Handoff({}), _Generator({}), "")
