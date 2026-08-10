# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-specific stages for the user ensemble example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dynamo.common.backend import GenerateRequest
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.decoder_stage import VllmDecoderStage
from dynamo.vllm.multimodal_utils.custom_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.workflow import StageContext, StageContract, ValueSpec

ARTIFACTS = ValueSpec(type="object", class_id="dynamo.vllm.CustomEncoderArtifacts")
REQUEST = VllmDecoderStage.contract.inputs["request"]
PROMPT = VllmDecoderStage.contract.inputs["prompt"]


class EncoderStage:
    """Adapt the existing async encoder to the common workflow interface."""

    contract = StageContract(
        id="custom-vision-encoder",
        inputs={"image_url": ValueSpec(type="text"), "request": REQUEST},
        outputs={"artifacts": ARTIFACTS, "prompt": PROMPT},
    )

    def __init__(
        self,
        encoder: AsyncVisionEncoder[Any, Any, Any],
        adapter: CustomEncoderAdapter[Any],
    ) -> None:
        self._encoder = encoder
        self._adapter = adapter

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        del context
        request = cast(GenerateRequest, inputs["request"])
        token_ids = request.get("token_ids")
        if not isinstance(token_ids, list):
            raise InvalidArgument("request must contain token_ids")
        artifacts = await self._encoder.encode([cast(str, inputs["image_url"])])
        prompt = self._adapter.prepare_prompt(list(token_ids), artifacts)
        return {"artifacts": artifacts, "prompt": prompt}


class DummyClassifier:
    """Replaceable classification worker used by the runnable example."""

    contract = StageContract(
        id="artifact-classifier",
        inputs={"artifacts": ARTIFACTS},
        outputs={"scores": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        del inputs, context
        return {"scores": {"dummy-classification": 1.0}}
