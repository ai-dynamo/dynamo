# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a local encoder with a remote generator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from dynamo.llm import with_engine_data
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import ExternalEncoderResult


class RequestPreparer(Protocol):
    async def prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        target_model: str,
    ) -> Mapping[str, Any]:
        ...


class GeneratorClient(Protocol):
    async def complete(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        ...


class Classifier(Protocol):
    def classify(self, encoder_result: Mapping[str, Any]) -> str:
        ...


class DummyClassifier:
    """Derive a toy label from the first external-encoder feature value."""

    def classify(self, encoder_result: Mapping[str, Any]) -> str:
        artifacts = ExternalEncoderResult.from_dict(encoder_result).to_artifacts()
        first_feature = float(artifacts[0][0, 0])
        return "class_a" if first_feature >= 0 else "class_b"


class ExternalEncoderOrchestrator:
    """Encode locally, classify the result, then invoke remote aggregated vLLM."""

    def __init__(
        self,
        handoff: RequestPreparer,
        generator: GeneratorClient,
        classifier: Classifier,
        generator_model_name: str,
    ) -> None:
        if not generator_model_name:
            raise ValueError("generator_model_name must not be empty")
        self._handoff = handoff
        self._generator = generator
        self._classifier = classifier
        self._generator_model_name = generator_model_name

    async def __call__(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        # Only media turns need the handoff; it rejects a request without any.
        classifier_label: str | None = None
        if _has_multimodal_inputs(request):
            generator_request = await self._handoff.prepare_request(
                request,
                target_model=self._generator_model_name,
            )
            encoder_result = generator_request.get("encoder_result")
            if not isinstance(encoder_result, Mapping):
                raise ValueError(
                    "prepared request must contain an encoder_result object"
                )
            classifier_label = self._classifier.classify(encoder_result)
        else:
            generator_request = {**request, "model": self._generator_model_name}
        completion = await self._generator.complete(generator_request, context=context)
        if classifier_label is None:
            return completion
        return with_engine_data(
            completion,
            {"classifier_label": classifier_label},
        )


def _has_multimodal_inputs(request: Mapping[str, Any]) -> bool:
    multi_modal_data = request.get("multi_modal_data")
    if not isinstance(multi_modal_data, Mapping):
        return False
    return any(multi_modal_data.values())
