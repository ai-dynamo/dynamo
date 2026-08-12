# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydration and execution for compiled Dynamo workflows."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, cast, runtime_checkable

from dynamo.workflow.builder import StageDefinition
from dynamo.workflow.ir import StageIR
from dynamo.workflow.plan import ExecutionPlan, LocalBinding
from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    ValueSpec,
    WorkflowValidationError,
)


class WorkflowExecutionError(RuntimeError):
    """Raised when runtime values do not honor the authored workflow."""


@dataclass(frozen=True)
class StageContext:
    """Attempt metadata available to a running stage."""

    workflow_name: str
    stage_id: str
    attempt_id: str
    deadline: Optional[float]
    _cancelled: asyncio.Event

    @property
    def cancelled(self) -> bool:
        """Whether the workflow attempt is terminating."""

        return self._cancelled.is_set()

    def remaining_time(self) -> Optional[float]:
        """Return seconds until the attempt deadline, when one exists."""

        if self.deadline is None:
            return None
        return max(0.0, self.deadline - asyncio.get_running_loop().time())

    def raise_if_cancelled(self) -> None:
        """Cooperatively stop work after cancellation or deadline expiry."""

        if self.cancelled:
            raise asyncio.CancelledError
        if self.deadline is not None and self.remaining_time() == 0:
            raise asyncio.TimeoutError


@runtime_checkable
class StageRunner(StageDefinition, Protocol):
    """The small interface implemented by custom and Dynamo-provided workers."""

    contract: StageContract

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        """Run one stage attempt and return all declared outputs."""

        ...


@runtime_checkable
class _TensorValue(Protocol):
    shape: Any
    dtype: Any


@runtime_checkable
class _ImageValue(Protocol):
    mode: str
    size: Any


class WorkflowExecutor:
    """Hydrate and execute one compiled workflow plan."""

    def __init__(
        self,
        plan: ExecutionPlan,
        local_runners: Mapping[str, StageRunner],
    ) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must use ExecutionPlan")
        if not isinstance(local_runners, Mapping):
            raise TypeError("local_runners must be a mapping")

        expected_keys = {
            binding.runner_key
            for binding in plan.bindings.values()
            if isinstance(binding, LocalBinding)
        }
        actual_keys = set(local_runners)
        if actual_keys != expected_keys:
            raise WorkflowValidationError(
                "local runners differ from execution plan; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )

        runners = dict(local_runners)
        for stage in plan.workflow.stages:
            binding = plan.bindings[stage.id]
            if not isinstance(binding, LocalBinding):
                raise WorkflowValidationError(
                    f"executor does not support binding for stage {stage.id!r}"
                )
            runner = runners[binding.runner_key]
            if not isinstance(runner, StageRunner):
                raise WorkflowValidationError(
                    f"runner {binding.runner_key!r} must implement StageRunner"
                )
            if runner.contract != stage.contract:
                raise WorkflowValidationError(
                    f"runner {binding.runner_key!r} for stage {stage.id!r} "
                    "does not match its authored contract"
                )

        self._plan = plan
        self._local_runners = MappingProxyType(runners)

    @property
    def plan(self) -> ExecutionPlan:
        """Return the portable plan hydrated by this executor."""

        return self._plan

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
        attempt_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute one request, scheduling independent branches concurrently."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        input_values = dict(inputs)
        workflow = self._plan.workflow
        expected_inputs = set(workflow.inputs)
        actual_inputs = set(input_values)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                "workflow inputs differ from the authored graph; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for name, spec in workflow.inputs.items():
            _validate_value(spec, input_values[name], f"workflow input {name!r}")

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        cancelled = asyncio.Event()
        resolved_attempt_id = attempt_id or uuid.uuid4().hex

        async def execute() -> dict[str, Any]:
            tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

            async def run_stage(stage: StageIR) -> dict[str, Any]:
                stage_inputs: dict[str, Any] = {}
                for port_name, reference in stage.inputs.items():
                    stage_inputs[port_name] = await resolve(reference)
                context = StageContext(
                    workflow_name=workflow.name,
                    stage_id=stage.id,
                    attempt_id=resolved_attempt_id,
                    deadline=deadline,
                    _cancelled=cancelled,
                )
                binding = self._plan.bindings[stage.id]
                if not isinstance(binding, LocalBinding):
                    raise WorkflowExecutionError(
                        f"unsupported binding for stage {stage.id!r}"
                    )
                result = await self._local_runners[binding.runner_key].run(
                    MappingProxyType(stage_inputs), context
                )
                if not isinstance(result, Mapping):
                    raise WorkflowExecutionError(
                        f"stage {stage.id!r} returned a non-mapping result"
                    )
                expected_outputs = set(stage.contract.outputs)
                actual_outputs = set(result)
                if actual_outputs != expected_outputs:
                    raise WorkflowExecutionError(
                        f"stage {stage.id!r} outputs differ from its contract; "
                        f"missing={sorted(expected_outputs - actual_outputs)}, "
                        f"extra={sorted(actual_outputs - expected_outputs)}"
                    )
                outputs = dict(result)
                for output_name, spec in stage.contract.outputs.items():
                    _validate_value(
                        spec,
                        outputs[output_name],
                        f"stage {stage.id!r} output {output_name!r}",
                    )
                return outputs

            async def resolve(reference: ValueRef) -> Any:
                if reference.input_name is not None:
                    return input_values[reference.input_name]
                stage_id = cast(str, reference.stage_id)
                output_name = cast(str, reference.output_name)
                stage_outputs = await asyncio.shield(tasks[stage_id])
                return stage_outputs[output_name]

            for stage in workflow.stages:
                tasks[stage.id] = asyncio.create_task(
                    run_stage(stage), name=f"workflow:{stage.id}"
                )

            try:
                output_values = await asyncio.gather(
                    *(resolve(reference) for reference in workflow.outputs.values())
                )
                return dict(zip(workflow.outputs, output_values))
            except BaseException:
                cancelled.set()
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
                raise

        if timeout is None:
            return await execute()
        return await asyncio.wait_for(execute(), timeout=timeout)


def _validate_value(spec: ValueSpec, value: Any, location: str) -> None:
    if spec.type == "text" and not isinstance(value, str):
        raise WorkflowExecutionError(f"{location} must be text")
    if spec.type == "bytes" and not isinstance(value, (bytes, bytearray, memoryview)):
        raise WorkflowExecutionError(f"{location} must be bytes-like")
    if spec.type == "json" and not _is_json_value(value, set()):
        raise WorkflowExecutionError(f"{location} must use the JSON data model")
    if spec.type == "tensor":
        if not isinstance(value, _TensorValue):
            raise WorkflowExecutionError(f"{location} must be tensor-like")
        actual_dtype = str(value.dtype).rsplit(".", 1)[-1]
        if spec.dtype is not None and actual_dtype != spec.dtype:
            raise WorkflowExecutionError(
                f"{location} has dtype {actual_dtype!r}, expected {spec.dtype!r}"
            )
        actual_shape = tuple(value.shape)
        if spec.shape is not None:
            if len(actual_shape) != len(spec.shape) or any(
                expected != "dynamic" and expected != actual
                for actual, expected in zip(actual_shape, spec.shape)
            ):
                raise WorkflowExecutionError(
                    f"{location} has shape {actual_shape!r}, expected {spec.shape!r}"
                )
    if spec.type == "image":
        if not isinstance(value, _ImageValue):
            raise WorkflowExecutionError(f"{location} must be image-like")
        if spec.mode is not None and value.mode != spec.mode:
            raise WorkflowExecutionError(
                f"{location} has image mode {value.mode!r}, expected {spec.mode!r}"
            )


def _is_json_value(value: Any, active_containers: set[int]) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if not isinstance(value, (list, dict)):
        return False

    container_id = id(value)
    if container_id in active_containers:
        return False
    active_containers.add(container_id)
    try:
        if isinstance(value, list):
            return all(_is_json_value(item, active_containers) for item in value)
        return all(
            isinstance(key, str) and _is_json_value(item, active_containers)
            for key, item in value.items()
        )
    finally:
        active_containers.remove(container_id)
