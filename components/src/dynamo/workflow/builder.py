# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readable Python authoring for declarative and imperative workflows."""

from __future__ import annotations

import keyword
from collections.abc import Callable
from typing import Mapping, Optional, Protocol, Tuple, Union, cast, runtime_checkable

from dynamo.workflow.definition import (
    StageRef,
    WorkflowDefinition,
    WorkflowHandler,
    WorkflowHandlerCallback,
    _freeze_specs,
)
from dynamo.workflow.ir import StageIR, WorkflowIR
from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    ValueSpec,
    WorkflowValidationError,
    compatibility_error,
    validate_name,
)

_RESERVED_OUTPUT_ATTRIBUTES = frozenset({"output", "output_names"})


@runtime_checkable
class StageDefinition(Protocol):
    """Anything that carries a reusable stage contract, including a worker."""

    contract: StageContract


class StageHandle:
    """References the outputs declared by one stage contract."""

    def __init__(self, stage_id: str, contract: StageContract, owner: object) -> None:
        self._stage_id = stage_id
        self._outputs = {
            name: ValueRef.for_stage_output(stage_id, name, owner)
            for name in contract.outputs
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the declared output names."""

        return tuple(self._outputs)

    def output(self, name: str) -> ValueRef:
        """Reference a declared output, including names unsafe as attributes."""

        try:
            return self._outputs[name]
        except KeyError as error:
            raise WorkflowValidationError(
                f"stage {self._stage_id!r} has no output {name!r}; "
                f"declared outputs are {list(self._outputs)}"
            ) from error

    def __getattr__(self, name: str) -> ValueRef:
        reference = self._outputs.get(name)
        if (
            reference is not None
            and name.isidentifier()
            and not keyword.iskeyword(name)
            and name not in _RESERVED_OUTPUT_ATTRIBUTES
        ):
            return reference
        raise AttributeError(
            f"stage {self._stage_id!r} has no attribute-safe output {name!r}; "
            "use stage.output(name) for the explicit form"
        )


class Workflow:
    """Author either a declarative graph or an imperative workflow handler."""

    def __init__(self, name: str) -> None:
        validate_name(name, "workflow name")
        self._name = name
        self._owner = object()
        self._inputs: dict[str, ValueSpec] = {}
        self._stages: dict[str, StageIR] = {}
        self._contracts: dict[str, StageContract] = {}
        self._outputs: dict[str, ValueRef] = {}
        self._mode: Optional[str] = None
        self._handler_inputs: Mapping[str, ValueSpec] = {}
        self._handler_outputs: Mapping[str, ValueSpec] = {}
        self._handler_stages: dict[str, StageContract] = {}
        self._handler_callback: Optional[WorkflowHandlerCallback] = None

    def add_input(self, name: str, spec: ValueSpec) -> ValueRef:
        """Declare and reference a workflow input."""

        self._select_mode("graph")
        validate_name(name, "workflow input")
        if name in self._inputs:
            raise WorkflowValidationError(f"duplicate workflow input {name!r}")
        if not isinstance(spec, ValueSpec):
            raise WorkflowValidationError("workflow inputs must use ValueSpec")
        self._inputs[name] = spec
        return ValueRef.for_input(name, self._owner)

    def input(
        self,
        name: str,
        *,
        type: str,
        dtype: Optional[str] = None,
        shape: Optional[Tuple[Union[int, str], ...]] = None,
        mode: Optional[str] = None,
        class_id: Optional[str] = None,
    ) -> ValueRef:
        """Declare an input without constructing ``ValueSpec`` explicitly."""

        return self.add_input(
            name,
            ValueSpec(
                type=type,
                dtype=dtype,
                shape=shape,
                mode=mode,
                class_id=class_id,
            ),
        )

    def add_stage(
        self,
        stage_id: str,
        contract: StageContract,
        *,
        inputs: Mapping[str, ValueRef],
    ) -> StageHandle:
        """Add a stage whose named inputs exactly match its contract."""

        self._select_mode("graph")
        validate_name(stage_id, "stage id")
        if stage_id in self._stages:
            raise WorkflowValidationError(f"duplicate stage id {stage_id!r}")
        if not isinstance(contract, StageContract):
            raise WorkflowValidationError("stage contract must use StageContract")
        prior = self._contracts.get(contract.id)
        if prior is not None and prior != contract:
            raise WorkflowValidationError(
                f"contract id {contract.id!r} has conflicting schemas"
            )
        expected = set(contract.inputs)
        actual = set(inputs)
        if actual != expected:
            raise WorkflowValidationError(
                f"stage {stage_id!r} inputs differ from its contract; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

        for port_name, reference in inputs.items():
            producer_spec = self._resolve_owned_reference(reference)
            error = compatibility_error(producer_spec, contract.inputs[port_name])
            if error is not None:
                raise WorkflowValidationError(
                    f"stage {stage_id!r} input {port_name!r} is incompatible: {error}"
                )

        self._contracts[contract.id] = contract
        self._stages[stage_id] = StageIR(
            id=stage_id, contract=contract, inputs=dict(inputs)
        )
        return StageHandle(stage_id, contract, self._owner)

    def stage(
        self,
        stage_id: str,
        stage: Union[StageContract, StageDefinition],
        **inputs: ValueRef,
    ) -> StageHandle:
        """Add a contracted worker or contract using keyword input ports."""

        contract = self._resolve_contract(stage)
        return self.add_stage(stage_id, contract, inputs=inputs)

    def use(
        self,
        stage_id: str,
        stage: Union[StageContract, StageDefinition],
    ) -> StageRef:
        """Declare one stage available to an imperative workflow handler."""

        self._select_mode("handler")
        validate_name(stage_id, "stage id")
        if stage_id in self._handler_stages:
            raise WorkflowValidationError(f"duplicate stage id {stage_id!r}")
        contract = self._resolve_contract(stage)
        prior = self._contracts.get(contract.id)
        if prior is not None and prior != contract:
            raise WorkflowValidationError(
                f"contract id {contract.id!r} has conflicting schemas"
            )
        self._contracts[contract.id] = contract
        self._handler_stages[stage_id] = contract
        return StageRef(stage_id, contract, self._owner)

    def handler(
        self,
        *,
        inputs: Mapping[str, ValueSpec],
        outputs: Mapping[str, ValueSpec],
    ) -> Callable[[WorkflowHandlerCallback], WorkflowHandlerCallback]:
        """Register the single async callback for an imperative workflow."""

        self._select_mode("handler")
        if self._handler_callback is not None:
            raise WorkflowValidationError("workflow already has a handler")
        normalized_inputs = _freeze_specs(inputs, "workflow input")
        normalized_outputs = _freeze_specs(outputs, "workflow output")
        if not normalized_outputs:
            raise WorkflowValidationError(
                "workflow handlers require at least one output"
            )

        def register(callback: WorkflowHandlerCallback) -> WorkflowHandlerCallback:
            if self._handler_callback is not None:
                raise WorkflowValidationError("workflow already has a handler")
            self._handler_inputs = normalized_inputs
            self._handler_outputs = normalized_outputs
            self._handler_callback = callback
            return callback

        return register

    def add_output(self, name: str, reference: ValueRef) -> None:
        """Expose a workflow input or stage output as a workflow output."""

        self._select_mode("graph")
        validate_name(name, "workflow output")
        if name in self._outputs:
            raise WorkflowValidationError(f"duplicate workflow output {name!r}")
        self._resolve_owned_reference(reference)
        self._outputs[name] = reference

    def output(self, name: str, reference: ValueRef) -> None:
        """Expose a workflow result."""

        self.add_output(name, reference)

    def build(self) -> WorkflowDefinition:
        """Return the selected immutable workflow definition."""

        if self._mode == "handler":
            if self._handler_callback is None:
                raise WorkflowValidationError("imperative workflow requires a handler")
            return WorkflowHandler(
                name=self._name,
                inputs=self._handler_inputs,
                outputs=self._handler_outputs,
                stages=self._handler_stages,
                callback=self._handler_callback,
                _owner=self._owner,
            )
        if self._mode is None:
            raise WorkflowValidationError("workflow has no graph or handler definition")

        return WorkflowIR(
            name=self._name,
            inputs=self._inputs,
            stages=tuple(self._stages.values()),
            outputs=self._outputs,
        )

    def _select_mode(self, mode: str) -> None:
        if self._mode is not None and self._mode != mode:
            raise WorkflowValidationError(
                "workflow cannot mix declarative graph and imperative handler authoring"
            )
        self._mode = mode

    @staticmethod
    def _resolve_contract(
        stage: Union[StageContract, StageDefinition],
    ) -> StageContract:
        if isinstance(stage, StageContract):
            return stage
        if isinstance(stage, StageDefinition):
            return stage.contract
        raise WorkflowValidationError(
            "stage must be a StageContract or carry a StageContract"
        )

    def _resolve_owned_reference(self, reference: ValueRef) -> ValueSpec:
        if not isinstance(reference, ValueRef) or reference._owner is not self._owner:
            raise WorkflowValidationError(
                "value reference belongs to a different workflow"
            )
        if reference.input_name is not None:
            if reference.input_name not in self._inputs:
                raise WorkflowValidationError(
                    f"unknown workflow input {reference.input_name!r}"
                )
            return self._inputs[reference.input_name]
        stage_id = cast(str, reference.stage_id)
        output_name = cast(str, reference.output_name)
        if stage_id not in self._stages:
            raise WorkflowValidationError(f"unknown stage {stage_id!r}")
        stage = self._stages[stage_id]
        if output_name not in stage.contract.outputs:
            raise WorkflowValidationError(
                f"unknown output {output_name!r} on stage {stage_id!r}"
            )
        return stage.contract.outputs[output_name]
