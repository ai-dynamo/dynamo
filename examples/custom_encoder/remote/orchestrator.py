# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a local encoder with a remote generator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


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


class ExternalEncoderOrchestrator:
    """Encode locally, then invoke a remote aggregated vLLM endpoint."""

    def __init__(
        self,
        handoff: RequestPreparer,
        generator: GeneratorClient,
        generator_model_name: str,
    ) -> None:
        if not generator_model_name:
            raise ValueError("generator_model_name must not be empty")
        self._handoff = handoff
        self._generator = generator
        self._generator_model_name = generator_model_name

    async def __call__(
        self,
        request: Mapping[str, Any],
        *,
        context: Any,
    ) -> dict[str, Any]:
        # Only media turns need the handoff; it rejects a request without any.
        if _has_multimodal_inputs(request):
            generator_request = await self._handoff.prepare_request(
                request,
                target_model=self._generator_model_name,
            )
        else:
            generator_request = {**request, "model": self._generator_model_name}
        return await self._generator.complete(generator_request, context=context)


def _has_multimodal_inputs(request: Mapping[str, Any]) -> bool:
    multi_modal_data = request.get("multi_modal_data")
    if not isinstance(multi_modal_data, Mapping):
        return False
    return any(multi_modal_data.values())
