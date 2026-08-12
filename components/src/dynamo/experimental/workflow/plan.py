# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory physical plans for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Union

from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.types import (
    StageContract,
    WorkflowValidationError,
    validate_name,
)


@dataclass(frozen=True)
class InlineBinding:
    """Resolve one logical stage to a named in-process runner at bind time."""

    runner_key: str

    def __post_init__(self) -> None:
        validate_name(self.runner_key, "inline runner key")


def _validate_endpoint_id(endpoint_id: str) -> None:
    if not isinstance(endpoint_id, str):
        raise WorkflowValidationError("remote endpoint id must be a string")
    parts = endpoint_id.split(".")
    if len(parts) != 3:
        raise WorkflowValidationError(
            "remote endpoint id must use 'namespace.component.endpoint'"
        )
    for kind, part in zip(("namespace", "component", "endpoint"), parts):
        validate_name(part, f"remote {kind}")


@dataclass(frozen=True)
class RemoteBinding:
    """Resolve one logical stage through a discovered Dynamo endpoint."""

    endpoint_id: str
    routing_policy: str = "round_robin"

    def __post_init__(self) -> None:
        _validate_endpoint_id(self.endpoint_id)
        if self.routing_policy != "round_robin":
            raise WorkflowValidationError(
                f"unsupported remote routing policy {self.routing_policy!r}"
            )

Binding = Union[InlineBinding, RemoteBinding]


@dataclass(frozen=True)
class ExecutionPlan:
    """A workflow plus immutable in-memory stage bindings."""

    workflow: WorkflowIR
    bindings: Mapping[str, Binding]

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, WorkflowIR):
            raise WorkflowValidationError("execution plan requires WorkflowIR")
        if not isinstance(self.bindings, Mapping):
            raise WorkflowValidationError("execution plan bindings must be a mapping")

        bindings: dict[str, Binding] = {}
        for stage_id, binding in sorted(self.bindings.items()):
            validate_name(stage_id, "binding stage id")
            if not isinstance(binding, (InlineBinding, RemoteBinding)):
                raise WorkflowValidationError(
                    f"binding for stage {stage_id!r} uses an unsupported type"
                )
            bindings[stage_id] = binding

        expected_stages = set(self.stage_contracts)
        actual_stages = set(bindings)
        if actual_stages != expected_stages:
            raise WorkflowValidationError(
                "execution plan bindings differ from workflow stages; "
                f"missing={sorted(expected_stages - actual_stages)}, "
                f"extra={sorted(actual_stages - expected_stages)}"
            )

        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    @property
    def stage_contracts(self) -> Mapping[str, StageContract]:
        """Return every stage contract keyed by its authored stage ID."""

        return MappingProxyType(
            {stage.id: stage.contract for stage in self.workflow.stages}
        )

    @property
    def remote(self) -> bool:
        """Whether every stage is bound to a remote endpoint."""

        return bool(self.bindings) and all(
            isinstance(binding, RemoteBinding) for binding in self.bindings.values()
        )
