# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable encoder and stock-vLLM workflow stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from dynamo.common.backend import GenerateRequest
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)
from dynamo.workflow import StageContext, StageContract, StreamSpec, ValueSpec

_IMAGE_URL_PORT = "image_url"
_URL_VARIANT = "Url"

REQUEST = ValueSpec(type="json")
ENCODER_FEATURES = ValueSpec(type="tensor")
ENCODER_METADATA = ValueSpec(type="json")


class EncoderStage:
    """Encode request images into one dynamically shaped packed tensor."""

    contract = StageContract(
        id="dynamo-vision-encoder",
        inputs={"request": REQUEST},
        outputs={
            "encoder_features": ENCODER_FEATURES,
            "encoder_metadata": ENCODER_METADATA,
        },
    )

    def __init__(
        self,
        encoder: AsyncVisionEncoder[Any, Any, torch.Tensor],
        image_token_id: int,
    ) -> None:
        if (
            isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int)
            or image_token_id < 0
        ):
            raise ValueError("encoder backend requires a non-negative image_token_id")
        self._encoder = encoder
        self._image_token_id = image_token_id
        self._closed = False

    @classmethod
    def from_backend(
        cls,
        backend: VisionEncoderBackend[Any, Any, torch.Tensor],
        *,
        model: str,
        name: str = "workflow-vision-encoder",
    ) -> "EncoderStage":
        """Load an author-provided linear-embedding backend into this stage."""

        image_token_id = getattr(backend, "image_token_id", None)
        if (
            isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int)
            or image_token_id < 0
        ):
            raise ValueError(
                "encoder backend requires a non-negative integer image_token_id"
            )
        encoder: AsyncVisionEncoder[Any, Any, torch.Tensor] = AsyncVisionEncoder(
            backend, name=name
        )
        try:
            encoder.load(model)
        except BaseException:
            encoder.shutdown()
            raise
        return cls(encoder, image_token_id)

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        request_value = inputs["request"]
        if not isinstance(request_value, Mapping):
            raise InvalidArgument("encoder stage request must be an object")
        request = cast(GenerateRequest, request_value)
        artifacts = await self._encoder.encode(self._image_urls(request))
        context.raise_if_cancelled()
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

    def close(self) -> None:
        """Release the encoder driver and its author-owned backend resources."""

        if self._closed:
            return
        self._closed = True
        self._encoder.shutdown()

    @staticmethod
    def _image_urls(request: GenerateRequest) -> list[str]:
        multimodal = request.get("multi_modal_data") or {}
        if not isinstance(multimodal, Mapping):
            raise InvalidArgument("multi_modal_data must be an object")
        unsupported = sorted(
            key for key, value in multimodal.items() if key != _IMAGE_URL_PORT and value
        )
        if unsupported:
            raise InvalidArgument(
                "encoder stage supports image inputs only; got unsupported "
                f"multimodal data: {unsupported}"
            )

        image_items = multimodal.get(_IMAGE_URL_PORT) or []
        if not isinstance(image_items, list) or not image_items:
            raise InvalidArgument("encoder stage requires at least one image")
        image_urls = []
        for index, item in enumerate(image_items):
            if not isinstance(item, Mapping):
                raise InvalidArgument(f"image_url item {index} must be an object")
            image_url = item.get(_URL_VARIANT)
            if not isinstance(image_url, str) or not image_url:
                raise InvalidArgument(
                    f"image_url item {index} must contain a non-empty 'Url' string"
                )
            image_urls.append(image_url)
        return image_urls

    @staticmethod
    def _validate_artifacts(artifacts: Any) -> list[torch.Tensor]:
        try:
            tensors = list(artifacts)
        except TypeError as error:
            raise InvalidArgument(
                "external encoder artifacts must be an iterable of tensors"
            ) from error
        if not tensors:
            raise InvalidArgument("external encoder returned no image artifacts")
        first = tensors[0]
        if not isinstance(first, torch.Tensor) or first.dim() != 2:
            raise InvalidArgument(
                "external encoder artifact 0 must be a 2D torch.Tensor"
            )
        hidden = first.shape[1]
        if hidden == 0:
            raise InvalidArgument(
                "external encoder artifacts must have a non-zero hidden size"
            )
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
                    "workflow transfer requires CPU output"
                )
        return tensors


class DynamoVllmStage:
    """Contracts implemented by a stock aggregated Dynamo vLLM worker."""

    complete_contract = StageContract(
        id="dynamo-vllm-complete",
        inputs={
            "request": REQUEST,
            "encoder_features": ENCODER_FEATURES,
            "encoder_metadata": ENCODER_METADATA,
        },
        outputs={"completion": ValueSpec(type="json")},
    )

    # This declares the intended streaming ABI. The workflow compiler rejects
    # StreamSpec execution until scheduling and frontend propagation are added.
    stream_contract = StageContract(
        id="dynamo-vllm-stream",
        inputs={
            "request": REQUEST,
            "encoder_features": ENCODER_FEATURES,
            "encoder_metadata": ENCODER_METADATA,
        },
        outputs={"chunks": StreamSpec(item=ValueSpec(type="json"))},
    )
