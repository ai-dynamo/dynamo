# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a local encoder with a remote generator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .request import EncoderResultRequestBuilder


class Encoder(Protocol):
    async def encode(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class GeneratorClient(Protocol):
    async def complete(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        ...


class ExternalEncoderOrchestrator:
    """Encode locally, then invoke a remote aggregated vLLM endpoint."""

    def __init__(
        self,
        encoder: Encoder,
        generator: GeneratorClient,
        generator_model_name: str,
        request_builder: EncoderResultRequestBuilder | None = None,
    ) -> None:
        if not generator_model_name:
            raise ValueError("generator_model_name must not be empty")
        self._encoder = encoder
        self._generator = generator
        self._generator_model_name = generator_model_name
        self._request_builder = request_builder or EncoderResultRequestBuilder()

    async def __call__(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        encoder_result = await self._encoder.encode(request)
        generator_request = self._request_builder.build(
            request,
            encoder_result,
            generator_model_name=self._generator_model_name,
        )
        return await self._generator.complete(generator_request, context=context)
