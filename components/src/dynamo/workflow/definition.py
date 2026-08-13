# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory workflow definitions that are not canonical graph IR."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Union

from dynamo.workflow.ir import WorkflowIR
from dynamo.workflow.types import (
    StageContract,
    ValueSpec,
    WorkflowValidationError,
    validate_name,
)

WorkflowHandlerCallback = Callable[
    [Mapping[str, Any], Any], Awaitable[Mapping[str, Any]]
]


def _freeze_specs(specs: Mapping[str, ValueSpec], kind: str) -> Mapping[str, ValueSpec]:
    if not isinstance(specs, Mapping):
        raise WorkflowValidationError(f"{kind}s must be a mapping")
    frozen: dict[str, ValueSpec] = {}
    for name, spec in sorted(specs.items()):
        validate_name(name, kind)
        if not isinstance(spec, ValueSpec):
            raise WorkflowValidationError(f"{kind} {name!r} must use ValueSpec")
        frozen[name] = spec
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class StageRef:
    """Reference one stage declared in an imperative workflow catalog."""

    id: str
    contract: StageContract
    _owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_name(self.id, "stage id")
        if not isinstance(self.contract, StageContract):
            raise WorkflowValidationError("stage reference must use StageContract")


@dataclass(frozen=True)
class WorkflowHandler:
    """An imperative Python handler with a fixed contracted stage catalog."""

    name: str
    inputs: Mapping[str, ValueSpec]
    outputs: Mapping[str, ValueSpec]
    stages: Mapping[str, StageContract]
    callback: WorkflowHandlerCallback = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_name(self.name, "workflow name")
        inputs = _freeze_specs(self.inputs, "workflow input")
        outputs = _freeze_specs(self.outputs, "workflow output")
        if not outputs:
            raise WorkflowValidationError(
                "workflow handlers require at least one output"
            )
        if not isinstance(self.stages, Mapping):
            raise WorkflowValidationError("workflow handler stages must be a mapping")

        stages: dict[str, StageContract] = {}
        contracts: dict[str, StageContract] = {}
        for stage_id, contract in sorted(self.stages.items()):
            validate_name(stage_id, "stage id")
            if not isinstance(contract, StageContract):
                raise WorkflowValidationError(
                    f"handler stage {stage_id!r} must use StageContract"
                )
            prior = contracts.get(contract.id)
            if prior is not None and prior != contract:
                raise WorkflowValidationError(
                    f"contract id {contract.id!r} has conflicting schemas"
                )
            contracts[contract.id] = contract
            stages[stage_id] = contract

        if not callable(self.callback) or not inspect.iscoroutinefunction(
            self.callback
        ):
            raise WorkflowValidationError("workflow handler must be an async function")

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "stages", MappingProxyType(stages))


WorkflowDefinition = Union[WorkflowIR, WorkflowHandler]
