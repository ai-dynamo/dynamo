# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for unary application code and Dynamo's streaming endpoint ABI."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
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
        graceful_shutdown: bool = True,
        metrics_labels: list[tuple[str, str]] | None = None,
        health_check_payload: dict[str, Any] | None = None,
    ) -> None:
        ...


UnaryHandler = Callable[..., Awaitable[Any]]


class UnaryClient:
    """Return exactly one response from a round-robin Dynamo endpoint call."""

    def __init__(self, client: _RoundRobinClient) -> None:
        self._client = client

    async def complete(self, request: Any, *, context: Context | None = None) -> Any:
        """Invoke the endpoint and return its sole unannotated response."""

        stream = await self._client.round_robin(
            request,
            annotated=False,
            context=context,
        )
        return await _collect_one(stream)


async def serve_unary_endpoint(
    endpoint: _Endpoint,
    handler: UnaryHandler,
    *,
    graceful_shutdown: bool = True,
    metrics_labels: list[tuple[str, str]] | None = None,
    health_check_payload: dict[str, Any] | None = None,
) -> None:
    """Serve an async unary handler through Dynamo's streaming endpoint ABI."""

    has_context = _accepts_context(handler)

    async def generate(
        request: Any,
        *,
        context: Context | None = None,
    ) -> AsyncIterator[Any]:
        result = handler(request, context=context) if has_context else handler(request)
        if not inspect.isawaitable(result):
            raise TypeError("unary endpoint handler must return an awaitable")
        yield await result

    await endpoint.serve_endpoint(
        generate,
        graceful_shutdown=graceful_shutdown,
        metrics_labels=metrics_labels,
        health_check_payload=health_check_payload,
    )


def _accepts_context(handler: UnaryHandler) -> bool:
    parameters = inspect.signature(handler).parameters
    context = parameters.get("context")
    if context is not None and context.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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
