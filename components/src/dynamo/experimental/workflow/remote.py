# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters between unary workflow stages and streaming Dynamo endpoints."""

from __future__ import annotations

import asyncio
import uuid
from types import MappingProxyType
from typing import Any, AsyncIterator, Mapping, Protocol

from dynamo.experimental.workflow.runtime import StageContext, StageRunner, WorkflowExecutionError
from dynamo.experimental.workflow.types import StageContract, WorkflowValidationError, validate_name


class _DynamoClient(Protocol):
    async def round_robin(
        self,
        request: Mapping[str, Any],
        *,
        annotated: bool,
        context: Any = None,
    ) -> AsyncIterator[Any]:
        ...


class RemoteStageClient:
    """Adapt one unary workflow stage call to Dynamo's streaming client API.

    ``StageDispatcher`` owns this adapter for remote bindings. It sends the
    stage's contract-shaped input mapping directly and requires exactly one
    output mapping from the endpoint stream, keeping transport iteration opaque
    to ``StageRunner`` implementations.
    """

    def __init__(self, client: _DynamoClient) -> None:
        self._client = client

    async def run(
        self,
        stage_id: str,
        contract: StageContract,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        stage_label = f"remote stage {stage_id!r} with contract {contract.id!r}"
        transport_context = None
        if context.request_context is not None:
            detach = getattr(context.request_context, "detached", None)
            if not callable(detach):
                raise WorkflowExecutionError(
                    "request context cannot create a detached child context"
                )
            transport_context = detach(f"{context.attempt_id}:{context.stage_id}")

        try:
            stream = await self._client.round_robin(
                dict(inputs), annotated=False, context=transport_context
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise WorkflowExecutionError(
                f"{stage_label} request failed at the transport boundary"
            ) from error
        try:
            try:
                response = await stream.__anext__()
            except StopAsyncIteration as error:
                raise WorkflowExecutionError(
                    f"{stage_label} returned no response mapping"
                ) from error
            try:
                await stream.__anext__()
            except StopAsyncIteration:
                pass
            else:
                raise WorkflowExecutionError(
                    f"{stage_label} returned multiple response mappings"
                )
            if not isinstance(response, Mapping):
                raise WorkflowExecutionError(
                    f"{stage_label} returned a non-mapping response"
                )
        except BaseException as error:
            if transport_context is not None:
                transport_context.stop_generating()
            close = getattr(stream, "aclose", None)
            if callable(close):
                await close()
            if isinstance(error, (asyncio.CancelledError, WorkflowExecutionError)):
                raise
            if isinstance(error, Exception):
                raise WorkflowExecutionError(
                    f"{stage_label} response failed at the transport boundary"
                ) from error
            raise
        return dict(response)


class RemoteStageServer:
    """Adapt a unary ``StageRunner`` to Dynamo's streaming endpoint API.

    Dynamo ``serve_endpoint`` handlers must return an async iterator. This
    adapter validates the direct input mapping, awaits ``StageRunner.run`` once,
    validates its output mapping, and yields that mapping as one response frame
    so stage implementations never write transport-level ``yield``.
    """

    def __init__(self, stage_id: str, runner: StageRunner) -> None:
        validate_name(stage_id, "remote stage id")
        if not isinstance(runner, StageRunner):
            raise WorkflowValidationError("remote runner must implement StageRunner")
        self._stage_id = stage_id
        self._runner = runner

    async def generate(
        self, request: Mapping[str, Any], context: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        if not isinstance(request, Mapping):
            raise WorkflowExecutionError("remote stage request must be a mapping")
        inputs = MappingProxyType(dict(request))
        transport_context = context

        expected_inputs = set(self._runner.contract.inputs)
        actual_inputs = set(inputs)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        request_id = uuid.uuid4().hex
        if transport_context is not None:
            get_request_id = getattr(transport_context, "id", None)
            if callable(get_request_id):
                candidate = get_request_id()
                if isinstance(candidate, str) and candidate:
                    request_id = candidate
        stage_context = StageContext(
            workflow_name=None,
            stage_id=self._stage_id,
            attempt_id=request_id,
            request_context=transport_context,
        )

        async def invoke() -> Mapping[str, Any]:
            return await self._runner.run(inputs, stage_context)

        invoke_task = asyncio.create_task(
            invoke(), name=f"workflow-remote:{request_id}"
        )
        transport_task: asyncio.Future[Any] | None = None
        if transport_context is not None:
            wait_for_stop = getattr(transport_context, "async_killed_or_stopped", None)
            if callable(wait_for_stop):
                transport_task = asyncio.ensure_future(wait_for_stop())

        try:
            if transport_task is None:
                result = await invoke_task
            else:
                done, _ = await asyncio.wait(
                    {invoke_task, transport_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if invoke_task in done:
                    result = invoke_task.result()
                else:
                    raise asyncio.CancelledError()
        except BaseException:
            if not invoke_task.done():
                invoke_task.cancel()
            await asyncio.gather(invoke_task, return_exceptions=True)
            raise
        finally:
            if transport_task is not None and not transport_task.done():
                transport_task.cancel()
                await asyncio.gather(transport_task, return_exceptions=True)

        transport_stopped = transport_context is not None and bool(
            getattr(transport_context, "is_stopped", lambda: False)()
        )
        transport_killed = transport_context is not None and bool(
            getattr(transport_context, "is_killed", lambda: False)()
        )
        if transport_stopped or transport_killed:
            raise asyncio.CancelledError()
        if not isinstance(result, Mapping):
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} returned a non-mapping result"
            )
        expected_outputs = set(self._runner.contract.outputs)
        actual_outputs = set(result)
        if actual_outputs != expected_outputs:
            raise WorkflowExecutionError(
                f"remote stage {self._stage_id!r} outputs differ from its contract; "
                f"missing={sorted(expected_outputs - actual_outputs)}, "
                f"extra={sorted(actual_outputs - expected_outputs)}"
            )
        outputs = dict(result)
        yield outputs
