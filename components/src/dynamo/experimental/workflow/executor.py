# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified execution for declarative and imperative workflows."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from dynamo.experimental.workflow.definition import StageRef, WorkflowHandler
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.plan import ExecutionPlan
from dynamo.experimental.workflow.runtime import (
    StageContext,
    StageRunner,
    WorkflowAttempt,
    WorkflowExecutionError,
    _validate_value,
)
from dynamo.experimental.workflow.scheduler import GraphScheduler
from dynamo.experimental.workflow.types import ValueSpec


class WorkflowContext:
    """Per-attempt imperative API and child-invocation lifetime owner."""

    def __init__(
        self,
        definition: WorkflowHandler,
        dispatcher: StageDispatcher,
        attempt: WorkflowAttempt,
    ) -> None:
        self._definition = definition
        self._dispatcher = dispatcher
        self._attempt = attempt
        self._next_invocation = 0
        self._active: set[asyncio.Task[dict[str, Any]]] = set()
        self._closed = False

    @property
    def attempt_id(self) -> str:
        return self._attempt.attempt_id

    @property
    def deadline(self) -> float | None:
        return self._attempt.deadline

    @property
    def cancelled(self) -> bool:
        return self._attempt.cancelled.is_set()

    def remaining_time(self) -> float | None:
        """Return seconds until the workflow deadline, when one exists."""

        if self.deadline is None:
            return None
        return max(0.0, self.deadline - asyncio.get_running_loop().time())

    def raise_if_cancelled(self) -> None:
        """Cooperatively stop handler work after cancellation or deadline."""

        if self.cancelled:
            raise asyncio.CancelledError
        if self.deadline is not None and self.remaining_time() == 0:
            raise asyncio.TimeoutError

    async def call(self, stage: StageRef, **inputs: Any) -> dict[str, Any]:
        """Invoke one catalog stage and return its complete output mapping."""

        if self._closed:
            raise WorkflowExecutionError("workflow context is closed")
        if (
            not isinstance(stage, StageRef)
            or stage._owner is not self._definition._owner
        ):
            raise WorkflowExecutionError(
                "stage reference belongs to a different workflow handler"
            )
        contract = self._definition.stages.get(stage.id)
        if contract is None or contract != stage.contract:
            raise WorkflowExecutionError(
                f"stage reference {stage.id!r} is not in the workflow catalog"
            )

        self._next_invocation += 1
        invocation_id = f"{self.attempt_id}:{self._next_invocation}"
        context = StageContext(
            workflow_name=self._definition.name,
            stage_id=stage.id,
            attempt_id=self.attempt_id,
            invocation_id=invocation_id,
            deadline=self.deadline,
            _cancelled=self._attempt.cancelled,
            request_context=self._attempt.request_context,
        )
        task = asyncio.create_task(
            self._dispatcher.call(stage.id, contract, inputs, context),
            name=f"workflow:{stage.id}:{invocation_id}",
        )
        self._active.add(task)
        try:
            return await task
        finally:
            if task.done():
                self._active.discard(task)

    async def close(self) -> None:
        """Reject new calls and cancel and await unfinished invocations."""

        self._closed = True
        active = tuple(task for task in self._active if not task.done())
        if active:
            self._attempt.cancelled.set()
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        self._active.clear()


class WorkflowExecutor:
    """Own one compiled workflow's request and result lifecycle."""

    def __init__(self, plan: ExecutionPlan, dispatcher: StageDispatcher) -> None:
        self._plan = plan
        self._dispatcher = dispatcher

    @classmethod
    async def bind(
        cls,
        plan: ExecutionPlan,
        *,
        local_runners: Mapping[str, StageRunner] = MappingProxyType({}),
    ) -> "WorkflowExecutor":
        """Bind initialized resources to an immutable execution plan."""

        return cls(plan, StageDispatcher(plan, local_runners))

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
        """Execute one graph or handler request and return one named result."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        definition = self._plan.definition
        input_values = dict(inputs)
        expected_inputs = set(definition.inputs)
        actual_inputs = set(input_values)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                "workflow inputs differ from its definition; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for name, spec in definition.inputs.items():
            _validate_value(spec, input_values[name], f"workflow input {name!r}")

        loop = asyncio.get_running_loop()
        attempt = WorkflowAttempt(
            attempt_id=attempt_id or uuid.uuid4().hex,
            deadline=None if timeout is None else loop.time() + timeout,
            cancelled=asyncio.Event(),
            request_context=request_context,
        )

        async def execute() -> dict[str, Any]:
            if isinstance(definition, WorkflowIR):
                result = await GraphScheduler(definition, self._dispatcher).run(
                    MappingProxyType(input_values), attempt
                )
                output_specs = {
                    name: definition.output_spec(name) for name in definition.outputs
                }
            else:
                context = WorkflowContext(definition, self._dispatcher, attempt)
                try:
                    result = await definition.callback(
                        MappingProxyType(input_values), context
                    )
                finally:
                    await context.close()
                output_specs = definition.outputs
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
