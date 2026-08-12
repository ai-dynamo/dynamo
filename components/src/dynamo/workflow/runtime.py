# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated execution for authored workflows."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Union, cast, runtime_checkable

from dynamo.workflow.builder import StageDefinition, WorkflowBuilder
from dynamo.workflow.ir import StageIR, WorkflowIR
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


@dataclass(frozen=True)
class LocalBinding:
    """Bind one logical stage to an in-process worker."""

    runner: StageRunner


@runtime_checkable
class _TensorValue(Protocol):
    shape: Any
    dtype: Any


@runtime_checkable
class _ImageValue(Protocol):
    mode: str
    size: Any


class ExecutionPlan:
    """A compiled aggregated workflow ready to execute requests."""

    def __init__(
        self, workflow: WorkflowIR, bindings: Mapping[str, LocalBinding]
    ) -> None:
        self._workflow = workflow
        self._bindings = MappingProxyType(dict(bindings))

    @property
    def workflow(self) -> WorkflowIR:
        """Return the logical workflow compiled into this plan."""

        return self._workflow

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
        expected_inputs = set(self._workflow.inputs)
        actual_inputs = set(input_values)
        if actual_inputs != expected_inputs:
            raise WorkflowExecutionError(
                "workflow inputs differ from the authored graph; "
                f"missing={sorted(expected_inputs - actual_inputs)}, "
                f"extra={sorted(actual_inputs - expected_inputs)}"
            )
        for name, spec in self._workflow.inputs.items():
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
                    workflow_name=self._workflow.name,
                    stage_id=stage.id,
                    attempt_id=resolved_attempt_id,
                    deadline=deadline,
                    _cancelled=cancelled,
                )
                result = await self._bindings[stage.id].runner.run(
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

            for stage in self._workflow.stages:
                tasks[stage.id] = asyncio.create_task(
                    run_stage(stage), name=f"workflow:{stage.id}"
                )

            try:
                output_values = await asyncio.gather(
                    *(
                        resolve(reference)
                        for reference in self._workflow.outputs.values()
                    )
                )
                return dict(zip(self._workflow.outputs, output_values))
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


def compile_workflow(
    workflow: Union[WorkflowBuilder, WorkflowIR],
    bindings: Optional[Mapping[str, Union[StageRunner, LocalBinding]]] = None,
    **workers: StageRunner,
) -> ExecutionPlan:
    """Compile a workflow by binding each stage to a reusable local worker.

    Keyword workers are the concise path when stage IDs are valid Python names::

        plan = compile_workflow(workflow, encoder=encoder, decoder=decoder)

    The ``bindings`` mapping supports all portable stage names.
    """

    workflow_ir = (
        workflow.build() if isinstance(workflow, WorkflowBuilder) else workflow
    )
    if not isinstance(workflow_ir, WorkflowIR):
        raise TypeError("workflow must be a Workflow or WorkflowIR")

    combined: dict[str, Union[StageRunner, LocalBinding]] = dict(bindings or {})
    overlap = set(combined) & set(workers)
    if overlap:
        raise WorkflowValidationError(
            f"duplicate local bindings for stages {sorted(overlap)}"
        )
    combined.update(workers)

    expected = {stage.id for stage in workflow_ir.stages}
    actual = set(combined)
    if actual != expected:
        raise WorkflowValidationError(
            "local bindings differ from workflow stages; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    local_bindings: dict[str, LocalBinding] = {}
    for stage in workflow_ir.stages:
        candidate = combined[stage.id]
        binding = (
            candidate
            if isinstance(candidate, LocalBinding)
            else LocalBinding(candidate)
        )
        if not isinstance(binding.runner, StageRunner):
            raise WorkflowValidationError(
                f"binding for stage {stage.id!r} must implement StageRunner"
            )
        if binding.runner.contract != stage.contract:
            raise WorkflowValidationError(
                f"binding for stage {stage.id!r} does not match its authored contract"
            )
        local_bindings[stage.id] = binding

    return ExecutionPlan(workflow_ir, local_bindings)


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
