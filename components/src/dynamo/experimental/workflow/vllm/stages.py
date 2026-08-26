# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable encoder and stock-vLLM workflow stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from dynamo.common.backend import GenerateRequest
from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    encode_request_plane_tensor,
)
from dynamo.experimental.workflow import StageContext, StageContract
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)

_IMAGE_URL_PORT = "image_url"
_URL_VARIANT = "Url"
_RAW_MULTIMODAL_FIELDS = (
    "multi_modal_data",
    "multi_modal_uuids",
    "mm_processor_kwargs",
    "mm_routing_info",
)


class EncoderStage:
    """Encode request images into one dynamically shaped packed tensor.

    The owner must call :meth:`close` during application shutdown.
    """

    contract = StageContract(
        id="dynamo-vision-encoder",
        inputs=frozenset({"request"}),
        outputs=frozenset({"encoder_features", "encoder_metadata"}),
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
        """Load an author-provided linear-embedding backend into this stage.

        The returned stage owns the encoder driver and must be closed by the
        application that binds it into a workflow.
        """

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
            backend,
            name=name,
        )
        try:
            encoder.load(model)
        except BaseException:
            encoder.shutdown()
            raise
        return cls(encoder, image_token_id)

    async def run(
        self,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        del context
        request_value = inputs["request"]
        if not isinstance(request_value, Mapping):
            raise InvalidArgument("encoder stage request must be an object")
        request = cast(GenerateRequest, request_value)
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


class ExternalEncoderRequestStage:
    """Attach packed encoder features to a standard ``GenerateRequest``.

    The request remains ordinary Dynamo Generate traffic, but its binary
    tensor payload requires a MsgPack-advertising destination endpoint.
    """

    contract = StageContract(
        id="dynamo-external-encoder-request",
        inputs=frozenset({"request", "encoder_features", "encoder_metadata"}),
        outputs=frozenset({"request"}),
    )

    async def run(
        self,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        del context
        request_value = inputs["request"]
        if not isinstance(request_value, Mapping):
            raise InvalidArgument("external encoder request must be an object")
        request = dict(request_value)
        if request.get("encoder_result") is not None:
            raise InvalidArgument("request already contains encoder_result")
        if request.get("prompt_embeds") is not None:
            raise InvalidArgument(
                "external encoder result cannot be combined with prompt_embeds"
            )

        try:
            features = encode_request_plane_tensor(inputs["encoder_features"])
            result = ExternalEncoderResult.from_parts(
                features,
                inputs["encoder_metadata"],
            )
        except (TypeError, ValueError) as error:
            raise InvalidArgument(str(error)) from error

        request["encoder_result"] = result.to_dict()
        for field_name in _RAW_MULTIMODAL_FIELDS:
            request.pop(field_name, None)
        extra_args = request.get("extra_args")
        if isinstance(extra_args, Mapping):
            copied_extra_args = dict(extra_args)
            copied_extra_args.pop("mm_kwargs_shm", None)
            copied_extra_args.pop("mm_kwargs_nixl", None)
            request["extra_args"] = copied_extra_args
        return {"request": request}


class DynamoVllmStage:
    """Contract implemented by a stock aggregated Dynamo vLLM worker."""

    request_complete_contract = StageContract(
        id="dynamo-vllm-request-complete",
        inputs=frozenset({"request"}),
        outputs=frozenset({"completion"}),
    )
