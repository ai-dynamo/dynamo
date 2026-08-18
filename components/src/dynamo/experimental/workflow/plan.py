# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory physical plans for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.types import StageContract, WorkflowValidationError, validate_name


@dataclass(frozen=True)
class InlineBinding:
    """Resolve one logical stage to a named in-process runner at bind time."""

    runner_key: str

    def __post_init__(self) -> None:
        validate_name(self.runner_key, "inline runner key")


Binding = InlineBinding


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
            if not isinstance(binding, InlineBinding):
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
