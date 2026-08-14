# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application stages and contracts for the remote user ensemble."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch

from dynamo.common.backend import GenerateRequest
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
)
from dynamo.workflow import StageContext, StageContract, ValueSpec

REQUEST = ValueSpec(type="json")
ENCODER_FEATURES = ValueSpec(type="tensor")
ENCODER_METADATA = ValueSpec(type="json")


class EncoderStage:
    """Encode request images into one dynamically shaped packed tensor."""

    contract = StageContract(
        id="custom-vision-encoder",
        inputs={"request": REQUEST},
        outputs={
            "encoder_features": ENCODER_FEATURES,
            "encoder_metadata": ENCODER_METADATA,
        },
    )

    def __init__(
        self,
        encoder: AsyncVisionEncoder[Any, Any, Any],
        image_token_id: int,
    ) -> None:
        self._encoder = encoder
        self._image_token_id = image_token_id

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        request = cast(GenerateRequest, inputs["request"])
        artifacts = await self._encoder.encode(self._image_urls(request))
        tensors = self._validate_artifacts(artifacts)

        row_splits = [0]
        for tensor in tensors:
            row_splits.append(row_splits[-1] + tensor.shape[0])
        return {
            "encoder_features": torch.cat(tensors, dim=0).contiguous(),
            "encoder_metadata": {
                "row_splits": row_splits,
                "image_token_id": self._image_token_id,
            },
        }

    @staticmethod
    def _image_urls(request: GenerateRequest) -> list[str]:
        multimodal = request.get("multi_modal_data") or {}
        unsupported = sorted(
            key for key, value in multimodal.items() if key != IMAGE_URL_KEY and value
        )
        if unsupported:
            raise InvalidArgument(
                "user ensemble supports image inputs only; got "
                f"unsupported multimodal data: {unsupported}"
            )

        image_items = multimodal.get(IMAGE_URL_KEY) or []
        if not image_items:
            raise InvalidArgument("user ensemble requires at least one image")
        image_urls = []
        for index, item in enumerate(image_items):
            if not isinstance(item, Mapping):
                raise InvalidArgument(f"image_url item {index} must be an object")
            image_url = item.get(URL_VARIANT_KEY)
            if not isinstance(image_url, str) or not image_url:
                raise InvalidArgument(
                    f"image_url item {index} must contain a non-empty 'Url' string"
                )
            image_urls.append(image_url)
        return image_urls

    @staticmethod
    def _validate_artifacts(artifacts: Any) -> list[torch.Tensor]:
        tensors = list(artifacts)
        if not tensors:
            raise InvalidArgument("external encoder returned no image artifacts")
        first = tensors[0]
        if not isinstance(first, torch.Tensor) or first.dim() != 2:
            raise InvalidArgument(
                "external encoder artifact 0 must be a 2D torch.Tensor"
            )
        hidden = first.shape[1]
        dtype = first.dtype
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise InvalidArgument(
                    f"external encoder artifact {index} must be a torch.Tensor"
                )
            if tensor.dim() != 2 or tensor.shape[1] != hidden:
                raise InvalidArgument(
                    f"external encoder artifact {index} must be 2D with hidden "
                    f"size {hidden}"
                )
            if tensor.shape[0] == 0:
                raise InvalidArgument(
                    f"external encoder artifact {index} has no feature rows"
                )
            if tensor.dtype != dtype:
                raise InvalidArgument(
                    f"external encoder artifact {index} has dtype {tensor.dtype}; "
                    f"expected {dtype}"
                )
            if tensor.device.type != "cpu":
                raise InvalidArgument(
                    f"external encoder artifact {index} is on {tensor.device}; "
                    "this example requires CPU output"
                )
        return tensors


class DummyClassifier:
    """Replaceable classifier that consumes the encoder's shared tensor."""

    contract = StageContract(
        id="embedding-classifier",
        inputs={"encoder_features": ENCODER_FEATURES},
        outputs={"scores": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        features = inputs["encoder_features"]
        if not isinstance(features, torch.Tensor):
            raise InvalidArgument("classifier features must be a torch.Tensor")
        mean = float(features.float().mean().item())
        if not math.isfinite(mean):
            raise InvalidArgument("classifier features must contain finite values")
        positive = (math.tanh(mean) + 1.0) / 2.0
        return {
            "scores": {
                "positive-mean": positive,
                "negative-mean": 1.0 - positive,
            }
        }


class StockVllmGenerator:
    """Contract implemented by a stock aggregated Dynamo vLLM worker."""

    contract = StageContract(
        id="stock-vllm-generator",
        inputs={
            "request": REQUEST,
            "encoder_features": ENCODER_FEATURES,
            "encoder_metadata": ENCODER_METADATA,
        },
        outputs={"chunk": ValueSpec(type="json")},
    )
