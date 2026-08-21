# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readable Python authoring helpers for canonical workflow IR."""

from __future__ import annotations

import keyword
from typing import cast

from dynamo.workflow.ir import StageIR, WorkflowIR
from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    WorkflowValidationError,
    validate_contract_consistency,
    validate_name,
)

_RESERVED_OUTPUT_ATTRIBUTES = frozenset({"output", "output_names"})


class StageHandle:
    """References the outputs declared by one stage contract."""

    def __init__(self, stage_id: str, contract: StageContract, owner: object) -> None:
        self._stage_id = stage_id
        self._outputs = {
            name: ValueRef.for_stage_output(stage_id, name, owner)
            for name in sorted(contract.outputs)
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
        if name.startswith("_"):
            raise AttributeError(name)
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
    """Author a static workflow with eager validation at each declaration.

    These checks provide local feedback; :class:`WorkflowIR` remains the
    canonical graph-validation boundary.
    """

    def __init__(self, name: str) -> None:
        validate_name(name, "workflow name")
        self._name = name
        self._owner = object()
        self._inputs: set[str] = set()
        self._stages: dict[str, StageIR] = {}
        self._contracts: dict[str, StageContract] = {}
        self._outputs: dict[str, ValueRef] = {}

    def input(self, name: str, /) -> ValueRef:
        """Declare and reference a workflow input."""

        validate_name(name, "workflow input")
        if name in self._inputs:
            raise WorkflowValidationError(f"duplicate workflow input {name!r}")
        self._inputs.add(name)
        return ValueRef.for_input(name, self._owner)

    def stage(
        self,
        stage_id: str,
        contract: StageContract,
        /,
        **inputs: ValueRef,
    ) -> StageHandle:
        """Add a stage whose named inputs exactly match its contract."""

        validate_name(stage_id, "stage id")
        if stage_id in self._stages:
            raise WorkflowValidationError(f"duplicate stage id {stage_id!r}")
        if not isinstance(contract, StageContract):
            raise WorkflowValidationError("stage contract must use StageContract")
        validate_contract_consistency(self._contracts.get(contract.id), contract)

        stage = StageIR(id=stage_id, contract=contract, inputs=dict(inputs))
        for reference in stage.inputs.values():
            self._validate_owned_reference(reference)

        self._contracts[contract.id] = contract
        self._stages[stage_id] = stage
        return StageHandle(stage_id, contract, self._owner)

    def output(self, name: str, reference: ValueRef, /) -> None:
        """Expose a workflow input or stage output as a workflow output."""

        validate_name(name, "workflow output")
        if name in self._outputs:
            raise WorkflowValidationError(f"duplicate workflow output {name!r}")
        self._validate_owned_reference(reference)
        self._outputs[name] = reference

    def build(self) -> WorkflowIR:
        """Return canonical, fully validated workflow IR."""

        return WorkflowIR(
            name=self._name,
            inputs=self._inputs,
            stages=tuple(self._stages.values()),
            outputs=self._outputs,
        )

    def _validate_owned_reference(self, reference: ValueRef) -> None:
        if not isinstance(reference, ValueRef) or reference._owner is not self._owner:
            raise WorkflowValidationError(
                "value reference belongs to a different workflow"
            )
        if reference.input_name is not None:
            if reference.input_name not in self._inputs:
                raise WorkflowValidationError(
                    f"unknown workflow input {reference.input_name!r}"
                )
            return
        stage_id = cast(str, reference.stage_id)
        output_name = cast(str, reference.output_name)
        if stage_id not in self._stages:
            raise WorkflowValidationError(f"unknown stage {stage_id!r}")
        stage = self._stages[stage_id]
        if output_name not in stage.contract.outputs:
            raise WorkflowValidationError(
                f"unknown output {output_name!r} on stage {stage_id!r}"
            )
