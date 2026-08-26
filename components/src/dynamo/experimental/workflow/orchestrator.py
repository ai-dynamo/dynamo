# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binding and execution lifecycle for declarative workflows."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Union

from dynamo.experimental.workflow.bindings import Binding
from dynamo.experimental.workflow.builder import Workflow
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.runtime import WorkflowExecutionError
from dynamo.experimental.workflow.scheduler import GraphScheduler


class WorkflowOrchestrator:
    """Bind and execute one declarative workflow."""

    # Declarative workflows execute through GraphScheduler.
    # TODO: Support imperative authoring through WorkflowHandler.

    def __init__(self, workflow_ir: WorkflowIR, dispatcher: StageDispatcher) -> None:
        self._workflow_ir = workflow_ir
        self._dispatcher = dispatcher

    @classmethod
    async def bind(
        cls,
        workflow: Union[Workflow, WorkflowIR],
        *,
        bindings: Mapping[str, Binding],
    ) -> "WorkflowOrchestrator":
        """Bind every authored stage to an initialized invocation target."""

        workflow_ir = workflow.build() if isinstance(workflow, Workflow) else workflow
        if not isinstance(workflow_ir, WorkflowIR):
            raise TypeError("workflow must be a Workflow or WorkflowIR")
        return cls(workflow_ir, StageDispatcher(workflow_ir, bindings))

    @property
    def workflow_ir(self) -> WorkflowIR:
        """Return the normalized logical graph executed by this orchestrator."""

        return self._workflow_ir

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        timeout: float | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one graph request and return its named results.

        A timeout starts cancellation but still waits for stage cleanup. A
        stage that suppresses cancellation can therefore delay this method.
        """

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        workflow = self._workflow_ir
        input_values = dict(inputs)
        expected_inputs = set(workflow.inputs)
        actual_inputs = set(input_values)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                "workflow inputs differ from its definition; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        resolved_attempt_id = attempt_id or uuid.uuid4().hex

        async def execute() -> dict[str, Any]:
            result = await GraphScheduler(workflow, self._dispatcher).run(
                MappingProxyType(input_values), resolved_attempt_id
            )
            return _validate_result(set(workflow.outputs), result)

        execution = asyncio.create_task(
            execute(), name=f"workflow-attempt:{resolved_attempt_id}"
        )
        try:
            if timeout is None:
                return await asyncio.shield(execution)
            return await asyncio.wait_for(asyncio.shield(execution), timeout=timeout)
        except BaseException:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            raise


def _validate_result(expected: set[str], result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise WorkflowExecutionError("workflow returned a non-mapping result")
    actual = set(result)
    if actual != expected:
        raise WorkflowExecutionError(
            "workflow outputs differ from its definition; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return dict(result)
