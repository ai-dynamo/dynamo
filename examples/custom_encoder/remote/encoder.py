# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline vision encoder that packages artifacts into an external encoder result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    encode_request_plane_tensor,
)
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
)

from .request import ImageUrlExtractor


class _EncoderDriver(Protocol):
    async def encode(self, raws: list[str]) -> list[Any]:
        ...

    def shutdown(self) -> None:
        ...


class InlineEncoder:
    """Drive a custom encoder and package its ordered linear embeddings."""

    def __init__(
        self,
        encoder: _EncoderDriver,
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
        self._url_extractor = ImageUrlExtractor()

    @classmethod
    def from_backend(
        cls,
        backend: VisionEncoderBackend[Any, Any, torch.Tensor],
        *,
        model: str,
    ) -> "InlineEncoder":
        """Load an author-provided backend through Dynamo's encoder driver."""

        image_token_id = getattr(backend, "image_token_id", None)
        encoder: AsyncVisionEncoder[Any, Any, torch.Tensor] = AsyncVisionEncoder(
            backend,
            name="remote-custom-encoder",
        )
        try:
            encoder.load(model)
            return cls(encoder, image_token_id)
        except BaseException:
            encoder.shutdown()
            raise

    async def encode(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return the versioned request-plane result for one Generate request."""

        image_urls = self._url_extractor.extract(request)
        artifacts = await self._encoder.encode(image_urls)
        tensors = _validate_artifacts(artifacts)
        packed = torch.cat(tensors, dim=0).contiguous()
        row_splits = _row_splits(tensors)
        return ExternalEncoderResult(
            features=encode_request_plane_tensor(packed),
            row_splits=row_splits,
            image_token_id=self._image_token_id,
        ).to_dict()

    def close(self) -> None:
        """Release the encoder driver and backend resources."""

        self._encoder.shutdown()


def _validate_artifacts(artifacts: list[Any]) -> list[torch.Tensor]:
    if not artifacts:
        raise InvalidArgument("external encoder returned no image artifacts")

    tensors: list[torch.Tensor] = []
    hidden_size: int | None = None
    dtype: torch.dtype | None = None
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, torch.Tensor):
            raise InvalidArgument(
                f"external encoder artifact {index} must be a torch.Tensor"
            )
        if artifact.dim() != 2 or any(size <= 0 for size in artifact.shape):
            raise InvalidArgument(
                f"external encoder artifact {index} must be a non-empty 2D tensor"
            )
        if artifact.device.type != "cpu":
            raise InvalidArgument(
                f"external encoder artifact {index} must be on CPU"
            )
        if hidden_size is None:
            hidden_size = artifact.shape[1]
            dtype = artifact.dtype
        elif artifact.shape[1] != hidden_size or artifact.dtype != dtype:
            raise InvalidArgument(
                "external encoder artifacts must have one hidden size and dtype"
            )
        tensors.append(artifact)
    return tensors


def _row_splits(tensors: list[torch.Tensor]) -> tuple[int, ...]:
    splits = [0]
    for tensor in tensors:
        splits.append(splits[-1] + tensor.shape[0])
    return tuple(splits)
