# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for unary application code and Dynamo's streaming endpoint ABI."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from dynamo._core import Context


class _RoundRobinClient(Protocol):
    async def round_robin(
        self,
        request: Any,
        *,
        annotated: bool,
        context: Context | None = None,
    ) -> AsyncIterator[Any]:
        ...


class _Endpoint(Protocol):
    async def serve_endpoint(
        self,
        handler: Callable[..., AsyncIterator[Any]],
    ) -> None:
        ...


class _UnaryHandler(Protocol):
    async def __call__(self, request: Any, *, context: Context) -> Any:
        ...


class UnaryClient:
    """Return exactly one response from a round-robin Dynamo endpoint call."""

    def __init__(self, client: _RoundRobinClient) -> None:
        self._client = client

    async def complete(self, request: Any, *, context: Context | None = None) -> Any:
        """Return one response, linking the call to ``context`` when provided."""

        stream = await self._client.round_robin(
            request,
            annotated=False,
            context=context,
        )
        return await _collect_one(stream)


async def serve_unary_endpoint(
    endpoint: _Endpoint,
    handler: _UnaryHandler,
) -> None:
    """Serve a unary handler that accepts Dynamo's per-request ``context``."""

    async def generate(
        request: Any,
        *,
        context: Context,
    ) -> AsyncIterator[Any]:
        yield await handler(request, context=context)

    await endpoint.serve_endpoint(generate)


async def _collect_one(stream: AsyncIterator[Any]) -> Any:
    iterator = stream.__aiter__()
    try:
        try:
            response = await anext(iterator)
        except StopAsyncIteration as error:
            raise RuntimeError("unary endpoint returned no response") from error

        try:
            await anext(iterator)
        except StopAsyncIteration:
            return response
        raise RuntimeError("unary endpoint returned multiple responses")
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
