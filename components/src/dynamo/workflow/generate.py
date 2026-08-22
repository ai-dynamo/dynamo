# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter from a workflow stage invocation to Dynamo's Generate endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

from dynamo.workflow.runtime import StageContext, WorkflowExecutionError
from dynamo.workflow.types import StageContract

GENERATE_REQUEST_PORT = "request"
GENERATE_OUTPUT_PORT = "completion"


class _DynamoClient(Protocol):
    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> AsyncIterator[Any]:
        ...


class GenerateEndpointStream:
    """Own one Generate transport stream and its cancellation context."""

    def __init__(self, stream: AsyncIterator[Any], transport_context: Any) -> None:
        self._stream = stream.__aiter__()
        self._transport_context = transport_context
        self._closed = False

    def __aiter__(self) -> "GenerateEndpointStream":
        return self

    async def __anext__(self) -> Any:
        return await self._stream.__anext__()

    async def aclose(self, *, cancel: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if cancel and self._transport_context is not None:
            self._transport_context.stop_generating()
        close = getattr(self._stream, "aclose", None)
        if callable(close):
            await close()


class GenerateEndpointInvoker:
    """Open or collect a stock token Generate endpoint invocation."""

    def __init__(self, client: _DynamoClient) -> None:
        self._client = client

    async def open(
        self,
        stage_id: str,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> GenerateEndpointStream:
        """Prepare the Generate request and return its owned token stream."""

        context.raise_if_cancelled()
        request_value = inputs[GENERATE_REQUEST_PORT]
        if not isinstance(request_value, Mapping):
            raise WorkflowExecutionError("Generate endpoint request must be an object")
        request = dict(request_value)
        _validate_request_options(request)

        transport_context = None
        if context.request_context is not None:
            detach = getattr(context.request_context, "detached", None)
            if not callable(detach):
                raise WorkflowExecutionError(
                    "request context cannot create a detached child context"
                )
            transport_context = detach(context.invocation_id)

        try:
            stream = await self._client.round_robin(
                request, annotated=False, context=transport_context
            )
        except BaseException:
            if transport_context is not None:
                transport_context.stop_generating()
            raise
        return GenerateEndpointStream(stream, transport_context)

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        del contract
        stream = await self.open(stage_id, inputs, context)
        try:
            completion = await collect_generation(stream, stage_id)
        except BaseException:
            await stream.aclose(cancel=True)
            raise
        await stream.aclose()
        return {GENERATE_OUTPUT_PORT: completion}


def _validate_request_options(request: Mapping[str, Any]) -> None:
    sampling_options = request.get("sampling_options", {})
    if not isinstance(sampling_options, Mapping):
        raise WorkflowExecutionError("sampling_options must be an object")
    # The Rust frontend represents an omitted OpenAI ``n`` as JSON null.
    # Both omission forms mean the protocol default of one sequence.
    if sampling_options.get("n") not in (None, 1):
        raise WorkflowExecutionError("Generate endpoint workflow stage requires n=1")

    output_options = request.get("output_options", {})
    if not isinstance(output_options, Mapping):
        raise WorkflowExecutionError("output_options must be an object")
    if (
        output_options.get("logprobs") is not None
        or output_options.get("prompt_logprobs") is not None
    ):
        raise WorkflowExecutionError(
            "Generate endpoint workflow stage does not support logprobs"
        )


async def collect_generation(
    stream: AsyncIterator[Any], stage_id: str
) -> dict[str, Any]:
    """Fold one non-streaming workflow completion from Generate token deltas."""

    token_ids: list[int] = []
    terminal: dict[str, Any] | None = None
    async for value in stream:
        if terminal is not None:
            raise WorkflowExecutionError(
                f"Generate endpoint stage {stage_id!r} returned data after terminal"
            )
        if not isinstance(value, Mapping):
            raise WorkflowExecutionError(
                f"Generate endpoint stage {stage_id!r} returned a non-object chunk"
            )
        chunk = dict(value)
        if chunk.get("index") != 0:
            raise WorkflowExecutionError(
                f"Generate endpoint stage {stage_id!r} requires choice index 0"
            )
        delta = chunk.get("token_ids")
        if not isinstance(delta, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in delta
        ):
            raise WorkflowExecutionError(
                f"Generate endpoint stage {stage_id!r} returned invalid token_ids"
            )
        if "log_probs" in chunk or "top_logprobs" in chunk:
            raise WorkflowExecutionError(
                f"Generate endpoint stage {stage_id!r} returned unsupported logprobs"
            )
        token_ids.extend(delta)
        finish_reason = chunk.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or not finish_reason:
                raise WorkflowExecutionError(
                    f"Generate endpoint stage {stage_id!r} returned invalid finish_reason"
                )
            terminal = chunk

    if terminal is None:
        raise WorkflowExecutionError(
            f"Generate endpoint stage {stage_id!r} returned no terminal chunk"
        )
    terminal["token_ids"] = token_ids
    return terminal
