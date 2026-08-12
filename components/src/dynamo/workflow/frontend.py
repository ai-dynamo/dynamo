# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that expose a workflow executor through the Dynamo token frontend."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from dynamo.workflow.runtime import WorkflowExecutor

RequestAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ResultAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _default_request_adapter(request: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"request": request}


def _default_result_adapter(result: Mapping[str, Any]) -> Mapping[str, Any]:
    chunk = result.get("chunk")
    if not isinstance(chunk, Mapping):
        raise TypeError("workflow result must contain a mapping-valued 'chunk' output")
    return chunk


@dataclass(frozen=True)
class WorkflowFrontendApplication:
    """A hydrated workflow plus its token-frontend boundary adapters."""

    executor: WorkflowExecutor
    model_path: str
    model_name: Optional[str] = None
    request_adapter: RequestAdapter = _default_request_adapter
    result_adapter: ResultAdapter = _default_result_adapter

    def __post_init__(self) -> None:
        if not isinstance(self.executor, WorkflowExecutor):
            raise TypeError("workflow frontend application requires WorkflowExecutor")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("workflow frontend application requires model_path")
        if self.model_name is not None and (
            not isinstance(self.model_name, str) or not self.model_name
        ):
            raise ValueError("workflow frontend model_name must be non-empty when set")
        if not callable(self.request_adapter) or not callable(self.result_adapter):
            raise TypeError("workflow frontend adapters must be callable")


class WorkflowTokenEngine:
    """Convert typed Dynamo token requests into workflow attempts."""

    def __init__(self, application: WorkflowFrontendApplication) -> None:
        if not isinstance(application, WorkflowFrontendApplication):
            raise TypeError("application must use WorkflowFrontendApplication")
        self._application = application

    async def generate(self, request: Mapping[str, Any], context: Any):
        if not isinstance(request, Mapping):
            raise TypeError("workflow token engine request must be a mapping")
        inputs = self._application.request_adapter(request)
        if not isinstance(inputs, Mapping):
            raise TypeError("workflow request adapter must return a mapping")

        attempt_id = context.id()
        execution = asyncio.create_task(
            self._application.executor.run(inputs, attempt_id=attempt_id),
            name=f"workflow-frontend:{attempt_id}",
        )
        cancellation = asyncio.create_task(
            context.async_killed_or_stopped(),
            name=f"workflow-frontend-cancellation:{attempt_id}",
        )
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

        chunk = self._application.result_adapter(result)
        if not isinstance(chunk, Mapping):
            raise TypeError("workflow result adapter must return a mapping")
        yield dict(chunk)


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


async def load_workflow_frontend_application(
    provider_path: str, runtime: Any, frontend_config: Any
) -> WorkflowFrontendApplication:
    """Load trusted provider code and build its frontend application."""

    provided = _resolve_provider(provider_path)(runtime, frontend_config)
    if inspect.isawaitable(provided):
        provided = await provided
    if not isinstance(provided, WorkflowFrontendApplication):
        raise TypeError(
            f"workflow provider {provider_path!r} must return "
            "WorkflowFrontendApplication"
        )
    return provided
