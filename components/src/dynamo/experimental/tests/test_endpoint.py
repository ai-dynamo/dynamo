# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest

from dynamo.experimental.endpoint import UnaryClient, serve_unary_endpoint

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.core,
]


class _Stream:
    def __init__(
        self, values: list[Any], *, terminal_error: BaseException | None = None
    ) -> None:
        self._values = iter(values)
        self._terminal_error = terminal_error
        self.closed = False

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._values)
        except StopIteration:
            if self._terminal_error is not None:
                raise self._terminal_error
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _Client:
    def __init__(self, stream: _Stream) -> None:
        self.stream = stream
        self.calls: list[tuple[Any, bool, Any]] = []

    async def round_robin(
        self, request: Any, *, annotated: bool, context: Any = None
    ) -> _Stream:
        self.calls.append((request, annotated, context))
        return self.stream


class _Endpoint:
    def __init__(self) -> None:
        self.handler: Callable[..., AsyncIterator[Any]] | None = None

    async def serve_endpoint(
        self,
        handler: Callable[..., AsyncIterator[Any]],
    ) -> None:
        self.handler = handler


async def _invoke_served(
    endpoint: _Endpoint, request: Any, *, context: Any = None
) -> list[Any]:
    if endpoint.handler is None:
        raise RuntimeError("endpoint has not been served")
    return [item async for item in endpoint.handler(request, context=context)]


@pytest.mark.parametrize("response", [{"result": "ok"}, [1, 2], "done"])
async def test_unary_client_returns_one_unannotated_response(response: Any) -> None:
    stream = _Stream([response])
    client = _Client(stream)
    context = cast(Any, object())

    result = await UnaryClient(client).complete({"input": "hello"}, context=context)

    assert result == response
    assert client.calls == [({"input": "hello"}, False, context)]
    assert stream.closed


@pytest.mark.parametrize(
    "responses, message",
    [([], "no response"), ([1, 2], "multiple responses")],
)
async def test_unary_client_enforces_response_cardinality(
    responses: list[Any], message: str
) -> None:
    stream = _Stream(responses)

    with pytest.raises(RuntimeError, match=message):
        await UnaryClient(_Client(stream)).complete({})

    assert stream.closed


@pytest.mark.parametrize(
    "error",
    [ValueError("failed"), asyncio.CancelledError()],
)
async def test_unary_client_propagates_stream_failure(error: BaseException) -> None:
    stream = _Stream([], terminal_error=error)

    with pytest.raises(type(error), match="failed" if str(error) else None):
        await UnaryClient(_Client(stream)).complete({})

    assert stream.closed


async def test_serve_unary_endpoint_propagates_context() -> None:
    endpoint = _Endpoint()
    seen: list[tuple[Any, Any]] = []

    async def handler(request: Any, *, context: Any) -> dict[str, Any]:
        seen.append((request, context))
        return {"result": request["value"]}

    await serve_unary_endpoint(endpoint, handler)
    context = object()

    assert await _invoke_served(endpoint, {"value": 7}, context=context) == [
        {"result": 7}
    ]
    assert seen == [({"value": 7}, context)]


async def test_serve_unary_endpoint_propagates_handler_failure() -> None:
    endpoint = _Endpoint()

    async def handler(request: Any, *, context: Any) -> Any:
        del request, context
        raise ValueError("handler failed")

    await serve_unary_endpoint(endpoint, handler)

    with pytest.raises(ValueError, match="handler failed"):
        await _invoke_served(endpoint, {})
