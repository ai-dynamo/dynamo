# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="custom encoder tests require torch")

from dynamo.vllm.multimodal_utils.custom_encoder.handoff import (  # noqa: E402
    ExternalEncoderResult,
)
from examples.custom_encoder.remote.orchestrator import (  # noqa: E402
    DummyClassifier,
    ExternalEncoderOrchestrator,
)

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


class _Classifier:
    def __init__(self, label: str = "class_a") -> None:
        self.label = label
        self.encoder_result: Mapping[str, Any] | None = None

    def classify(self, encoder_result: Mapping[str, Any]) -> str:
        self.encoder_result = encoder_result
        return self.label


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
        "engine_data": {"generator": "remote-vllm"},
    }
    handoff = _Handoff(prepared_request)
    generator = _Generator(completion)
    classifier = _Classifier("class_b")
    orchestrator = ExternalEncoderOrchestrator(
        handoff,
        generator,
        classifier,
        "generator-model",
    )
    request = {
        "token_ids": [1, 99, 2],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/image.png"}]},
    }
    context = object()

    result = await orchestrator(request, context=context)

    assert result == {
        **completion,
        "engine_data": {
            "generator": "remote-vllm",
            "classifier_label": "class_b",
        },
    }
    assert result is not completion
    assert handoff.request is request
    assert handoff.target_model == "generator-model"
    assert classifier.encoder_result is prepared_request["encoder_result"]
    assert generator.request is prepared_request
    assert generator.context is context


def test_orchestrator_requires_generator_model_name() -> None:
    with pytest.raises(ValueError, match="generator_model_name"):
        ExternalEncoderOrchestrator(_Handoff({}), _Generator({}), _Classifier(), "")


@pytest.mark.parametrize(
    ("first_feature", "expected_label"),
    [(1.0, "class_a"), (-1.0, "class_b")],
)
def test_dummy_classifier_uses_encoder_features(
    first_feature: float,
    expected_label: str,
) -> None:
    encoder_result = ExternalEncoderResult.from_artifacts(
        [torch.tensor([[first_feature, 2.0]], dtype=torch.float32)],
        image_token_id=99,
    )

    assert DummyClassifier().classify(encoder_result.to_dict()) == expected_label


async def test_orchestrator_forwards_text_only_request_without_encoding() -> None:
    completion = {"token_ids": [5], "index": 0, "finish_reason": "stop"}
    handoff = _Handoff({"unused": True})
    generator = _Generator(completion)
    classifier = _Classifier()
    orchestrator = ExternalEncoderOrchestrator(
        handoff,
        generator,
        classifier,
        "generator-model",
    )
    request = {"token_ids": [1, 2, 3]}

    result = await orchestrator(request, context=object())

    assert result is completion
    assert handoff.request is None
    assert classifier.encoder_result is None
    assert generator.request == {"token_ids": [1, 2, 3], "model": "generator-model"}


async def test_orchestrator_keeps_non_image_media_on_the_handoff_path() -> None:
    prepared_request = {
        "model": "generator-model",
        "token_ids": [1],
        "encoder_result": {"opaque": "to-the-orchestrator"},
    }
    handoff = _Handoff(prepared_request)
    generator = _Generator({"token_ids": [2], "index": 0, "finish_reason": "stop"})
    classifier = _Classifier()
    orchestrator = ExternalEncoderOrchestrator(
        handoff,
        generator,
        classifier,
        "generator-model",
    )
    request = {"token_ids": [1], "multi_modal_data": {"video_url": [{"Url": "v"}]}}

    await orchestrator(request, context=object())

    assert handoff.request is request
    assert generator.request is prepared_request
