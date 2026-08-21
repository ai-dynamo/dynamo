# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower placement-neutral workflows into physical execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union

from dynamo.workflow.builder import Workflow
from dynamo.workflow.ir import WorkflowIR
from dynamo.workflow.plan import Binding, ExecutionPlan, InlineBinding
from dynamo.workflow.types import WorkflowValidationError, validate_name


@dataclass(frozen=True)
class DeploymentSpec:
    """Stage placement requested when compiling one WorkflowIR."""

    bindings: Mapping[str, Binding]

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, Mapping):
            raise WorkflowValidationError("deployment bindings must be a mapping")
        bindings: dict[str, Binding] = {}
        for stage_id, binding in sorted(self.bindings.items()):
            validate_name(stage_id, "deployment stage id")
            if not isinstance(binding, InlineBinding):
                raise WorkflowValidationError(
                    f"binding for stage {stage_id!r} uses an unsupported type"
                )
            bindings[stage_id] = binding
        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    @classmethod
    def inline(cls, **runner_keys: str) -> "DeploymentSpec":
        """Build bindings to runners in the orchestrator process."""

        return cls(
            bindings={
                stage_id: InlineBinding(runner_key)
                for stage_id, runner_key in runner_keys.items()
            }
        )


def compile_workflow(
    workflow: Union[Workflow, WorkflowIR],
    deployment: Optional[DeploymentSpec] = None,
) -> ExecutionPlan:
    """Compile one logical workflow, defaulting every stage to inline placement."""

    workflow_ir = workflow.build() if isinstance(workflow, Workflow) else workflow
    if not isinstance(workflow_ir, WorkflowIR):
        raise TypeError("workflow must be a Workflow or WorkflowIR")
    stage_ids = tuple(stage.id for stage in workflow_ir.stages)
    if deployment is None:
        deployment = DeploymentSpec.inline(
            **{stage_id: stage_id for stage_id in stage_ids}
        )
    if not isinstance(deployment, DeploymentSpec):
        raise TypeError("deployment must use DeploymentSpec")

    expected = set(stage_ids)
    actual = set(deployment.bindings)
    if actual != expected:
        raise WorkflowValidationError(
            "deployment bindings differ from workflow stages; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    return ExecutionPlan(workflow_ir, deployment.bindings)
