# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unary helpers for Dynamo's normalized token Generate endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
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


class LLMUnaryClient:
    """Collect one normalized Dynamo Generate stream into a terminal response."""

    def __init__(self, client: _RoundRobinClient) -> None:
        self._client = client

    async def complete(
        self,
        request: Mapping[str, Any],
        *,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Return one terminal response containing all generated token IDs."""

        _validate_request(request)
        stream = await self._client.round_robin(
            request,
            annotated=False,
            context=context,
        )
        return await _collect_completion(stream)


def with_engine_data(
    completion: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a completion copy with non-overlapping ``engine_data`` values."""

    if not isinstance(completion, Mapping):
        raise ValueError("LLM completion must be an object")
    if not isinstance(values, Mapping):
        raise ValueError("engine_data values must be an object")

    result = dict(completion)
    if not values:
        return result

    existing_value = completion.get("engine_data")
    if existing_value is None:
        existing: Mapping[str, Any] = {}
    elif isinstance(existing_value, Mapping):
        existing = existing_value
    else:
        raise ValueError("existing engine_data must be an object")

    duplicate_keys = sorted(set(existing).intersection(values))
    if duplicate_keys:
        keys = ", ".join(repr(key) for key in duplicate_keys)
        raise ValueError(f"engine_data already contains keys: {keys}")

    result["engine_data"] = {**existing, **values}
    return result


def _validate_request(request: Mapping[str, Any]) -> None:
    if not isinstance(request, Mapping):
        raise ValueError("LLM request must be an object")

    sampling_options = request.get("sampling_options", {})
    if not isinstance(sampling_options, Mapping):
        raise ValueError("sampling_options must be an object")
    if sampling_options.get("n") not in (None, 1):
        raise ValueError("LLMUnaryClient requires n=1")

    output_options = request.get("output_options", {})
    if not isinstance(output_options, Mapping):
        raise ValueError("output_options must be an object")
    if (
        output_options.get("logprobs") is not None
        or output_options.get("prompt_logprobs") is not None
    ):
        raise ValueError("LLMUnaryClient does not support logprobs")


async def _collect_completion(stream: AsyncIterator[Any]) -> dict[str, Any]:
    iterator = stream.__aiter__()
    token_ids: list[int] = []
    terminal: dict[str, Any] | None = None
    try:
        async for value in iterator:
            if terminal is not None:
                raise RuntimeError("LLM endpoint returned data after terminal")
            if not isinstance(value, Mapping):
                raise RuntimeError("LLM endpoint returned a non-object chunk")

            chunk = dict(value)
            if chunk.get("index") != 0:
                raise RuntimeError("LLM endpoint requires choice index 0")

            delta = chunk.get("token_ids")
            if not isinstance(delta, list) or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in delta
            ):
                raise RuntimeError("LLM endpoint returned invalid token_ids")
            if "log_probs" in chunk or "top_logprobs" in chunk:
                raise RuntimeError("LLM endpoint returned unsupported logprobs")
            token_ids.extend(delta)

            finish_reason = chunk.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str) or not finish_reason:
                    raise RuntimeError("LLM endpoint returned invalid finish_reason")
                terminal = chunk

        if terminal is None:
            raise RuntimeError("LLM endpoint returned no terminal chunk")
        terminal["token_ids"] = token_ids
        return terminal
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
