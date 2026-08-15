# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution lifecycle for compiled declarative workflows."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.plan import ExecutionPlan
from dynamo.experimental.workflow.runtime import (
    StageRunner,
    WorkflowAttempt,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.experimental.workflow.scheduler import GraphScheduler
from dynamo.experimental.workflow.types import ValueSpec


class WorkflowOrchestrator:
    """Own one compiled workflow's request and result lifecycle."""

    def __init__(self, plan: ExecutionPlan, dispatcher: StageDispatcher) -> None:
        self._plan = plan
        self._dispatcher = dispatcher

    @classmethod
    async def bind(
        cls,
        plan: ExecutionPlan,
        *,
        inline_runners: Mapping[str, StageRunner] = MappingProxyType({}),
    ) -> "WorkflowOrchestrator":
        """Bind initialized resources to an immutable execution plan."""

        return cls(plan, StageDispatcher(plan, inline_runners))

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        timeout: float | None = None,
        attempt_id: str | None = None,
        request_context: Any = None,
    ) -> dict[str, Any]:
        """Execute one graph request and return its named results."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        workflow = self._plan.workflow
        input_values = dict(inputs)
        expected_inputs = set(workflow.inputs)
        actual_inputs = set(input_values)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                "workflow inputs differ from its definition; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for name, spec in workflow.inputs.items():
            _validate_value(spec, input_values[name], f"workflow input {name!r}")

        loop = asyncio.get_running_loop()
        attempt = WorkflowAttempt(
            attempt_id=attempt_id or uuid.uuid4().hex,
            deadline=None if timeout is None else loop.time() + timeout,
            cancelled=asyncio.Event(),
            request_context=request_context,
        )

        async def execute() -> dict[str, Any]:
            result = await GraphScheduler(workflow, self._dispatcher).run(
                MappingProxyType(input_values), attempt
            )
            output_specs = {
                name: workflow.output_spec(name) for name in workflow.outputs
            }
            return _validate_result(output_specs, result)

        execution = asyncio.create_task(
            execute(), name=f"workflow-attempt:{attempt.attempt_id}"
        )
        try:
            if timeout is None:
                return await asyncio.shield(execution)
            return await asyncio.wait_for(asyncio.shield(execution), timeout=timeout)
        except BaseException:
            attempt.cancelled.set()
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            raise


def _validate_result(
    specs: Mapping[str, ValueSpec], result: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise WorkflowExecutionError("workflow returned a non-mapping result")
    expected = set(specs)
    actual = set(result)
    if actual != expected:
        raise WorkflowExecutionError(
            "workflow outputs differ from its definition; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    outputs = dict(result)
    for name, spec in specs.items():
        _validate_value(spec, outputs[name], f"workflow output {name!r}")
    return outputs
