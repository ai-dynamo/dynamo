# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower placement-neutral workflows into physical execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Union

from dynamo.experimental.workflow.builder import WorkflowBuilder
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.plan import (
    LOCAL_CARRIER,
    Binding,
    EdgePlan,
    ExecutionPlan,
    LocalBinding,
)
from dynamo.experimental.workflow.types import WorkflowValidationError, validate_name


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
            if not isinstance(binding, LocalBinding):
                raise WorkflowValidationError(
                    f"binding for stage {stage_id!r} uses an unsupported type"
                )
            bindings[stage_id] = binding
        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    @classmethod
    def local(cls, **runner_keys: str) -> "DeploymentSpec":
        """Build local bindings whose values name executor runner keys."""

        return cls(
            bindings={
                stage_id: LocalBinding(runner_key)
                for stage_id, runner_key in runner_keys.items()
            }
        )


def compile_workflow(
    workflow: Union[WorkflowBuilder, WorkflowIR],
    deployment: DeploymentSpec,
) -> ExecutionPlan:
    """Compile one logical workflow using explicit placement bindings."""

    workflow_ir = (
        workflow.build() if isinstance(workflow, WorkflowBuilder) else workflow
    )
    if not isinstance(workflow_ir, WorkflowIR):
        raise TypeError("workflow must be a Workflow or WorkflowIR")
    if not isinstance(deployment, DeploymentSpec):
        raise TypeError("deployment must use DeploymentSpec")

    expected = {stage.id for stage in workflow_ir.stages}
    actual = set(deployment.bindings)
    if actual != expected:
        raise WorkflowValidationError(
            "deployment bindings differ from workflow stages; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    edges = tuple(
        EdgePlan(
            source=source,
            target_stage=stage.id,
            target_port=port,
            carrier=LOCAL_CARRIER,
        )
        for stage in workflow_ir.stages
        for port, source in stage.inputs.items()
    )
    return ExecutionPlan(workflow_ir, deployment.bindings, edges)
