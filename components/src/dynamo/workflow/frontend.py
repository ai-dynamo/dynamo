# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose a bound workflow through the Dynamo token frontend."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from dynamo.workflow.orchestrator import WorkflowOrchestrator
from dynamo.workflow.types import ValueSpec, WorkflowValidationError

_JSON = ValueSpec(type="json")


class WorkflowTokenEngine:
    """Run the fixed token-frontend workflow ABI as a typed token engine."""

    def __init__(self, orchestrator: WorkflowOrchestrator) -> None:
        if not isinstance(orchestrator, WorkflowOrchestrator):
            raise TypeError("orchestrator must use WorkflowOrchestrator")
        _validate_frontend_abi(orchestrator)
        self._orchestrator = orchestrator

    async def generate(self, request: Mapping[str, Any], context: Any):
        if not isinstance(request, Mapping):
            raise TypeError("workflow token engine request must be a mapping")

        attempt_id = context.id()
        execution = asyncio.create_task(
            self._orchestrator.run(
                {"request": request},
                attempt_id=attempt_id,
                request_context=context,
            ),
            name=f"workflow-frontend:{attempt_id}",
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


def _validate_frontend_abi(orchestrator: WorkflowOrchestrator) -> None:
    workflow = orchestrator.plan.workflow
    if dict(workflow.inputs) != {"request": _JSON}:
        raise WorkflowValidationError(
            "frontend workflows require exactly one 'request: json' input"
        )
    if set(workflow.outputs) != {"chunk"} or workflow.output_spec("chunk") != _JSON:
        raise WorkflowValidationError(
            "frontend workflows require exactly one 'chunk: json' output"
        )


def _resolve_provider(provider_path: str) -> Callable[..., Any]:
    if not isinstance(provider_path, str) or ":" not in provider_path:
        raise ValueError("workflow provider must use a 'module:callable' path")
    module_path, _, attribute_path = provider_path.partition(":")
    if not module_path or not attribute_path:
        raise ValueError("workflow provider must use a 'module:callable' path")
    try:
        provider: Any = importlib.import_module(module_path)
    except ImportError as error:
        raise ValueError(
            f"could not import workflow provider module {module_path!r}"
        ) from error
    try:
        for attribute in attribute_path.split("."):
            provider = getattr(provider, attribute)
    except AttributeError as error:
        raise ValueError(
            f"workflow provider module {module_path!r} has no attribute "
            f"{attribute_path!r}"
        ) from error
    if not callable(provider):
        raise TypeError(f"workflow provider {provider_path!r} must be callable")
    return provider


async def load_workflow_orchestrator(
    provider_path: str, runtime: Any
) -> WorkflowOrchestrator:
    """Load trusted provider code and validate its bound frontend workflow."""

    provided = _resolve_provider(provider_path)(runtime)
    if inspect.isawaitable(provided):
        provided = await provided
    if not isinstance(provided, WorkflowOrchestrator):
        raise TypeError(
            f"workflow provider {provider_path!r} must return WorkflowOrchestrator"
        )
    _validate_frontend_abi(provided)
    return provided
