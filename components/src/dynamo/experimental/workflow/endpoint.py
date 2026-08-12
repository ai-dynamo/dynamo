# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a bound workflow through a Dynamo model endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from dynamo.experimental.workflow.orchestrator import WorkflowOrchestrator
from dynamo.experimental.workflow.types import WorkflowValidationError


class WorkflowEndpointHandler:
    """Adapt a request/chunk workflow to Dynamo's endpoint handler ABI."""

    def __init__(self, orchestrator: WorkflowOrchestrator) -> None:
        if not isinstance(orchestrator, WorkflowOrchestrator):
            raise TypeError("orchestrator must use WorkflowOrchestrator")
        _validate_endpoint_abi(orchestrator)
        self._orchestrator = orchestrator

    async def generate(
        self, request: Mapping[str, Any], context: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute one workflow attempt and yield its terminal chunk."""

        if not isinstance(request, Mapping):
            raise TypeError("workflow endpoint request must be a mapping")

        attempt_id = context.id()
        execution = asyncio.create_task(
            self._orchestrator.run(
                {"request": request},
                attempt_id=attempt_id,
                request_context=context,
            ),
            name=f"workflow-endpoint:{attempt_id}",
        )
        cancellation = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait(
                {execution, cancellation}, return_when=asyncio.FIRST_COMPLETED
            )
            if execution not in done:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                return
            result = await execution
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise
        finally:
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

        chunk = result["chunk"]
        if not isinstance(chunk, Mapping):
            raise TypeError("workflow 'chunk' output must be a mapping")
        yield dict(chunk)


def _validate_endpoint_abi(orchestrator: WorkflowOrchestrator) -> None:
    workflow = orchestrator.plan.workflow
    if workflow.inputs != frozenset({"request"}):
        raise WorkflowValidationError(
            "workflow endpoints require exactly one 'request' input"
        )
    if set(workflow.outputs) != {"chunk"}:
        raise WorkflowValidationError(
            "workflow endpoints require exactly one 'chunk' output"
        )
