# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower placement-neutral workflows into physical execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union

from dynamo.workflow.builder import Workflow
from dynamo.workflow.ir import WorkflowIR
from dynamo.workflow.plan import (
    IN_PROCESS_CARRIER,
    INLINE_CARRIER,
    INLINE_VALUE_TYPES,
    Binding,
    EdgePlan,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
)
from dynamo.workflow.types import StreamSpec, WorkflowValidationError, validate_name


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
            if not isinstance(binding, (InlineBinding, RemoteBinding)):
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

    @classmethod
    def remote(cls, **endpoint_ids: str) -> "DeploymentSpec":
        """Build round-robin bindings to discovered Dynamo endpoints."""

        return cls(
            bindings={
                stage_id: RemoteBinding(endpoint_id)
                for stage_id, endpoint_id in endpoint_ids.items()
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
    port_specs = [
        *workflow_ir.inputs.values(),
        *(
            spec
            for stage in workflow_ir.stages
            for spec in (
                *stage.contract.inputs.values(),
                *stage.contract.outputs.values(),
            )
        ),
    ]
    if any(isinstance(spec, StreamSpec) for spec in port_specs):
        raise WorkflowValidationError(
            "stream ports are declarative only; workflow stream execution is not supported"
        )
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

    binding_types = {type(binding) for binding in deployment.bindings.values()}
    if len(binding_types) > 1:
        raise WorkflowValidationError(
            "workflow placement must be homogeneous; mixed inline and remote "
            "bindings are unsupported"
        )

    remote = bool(binding_types) and binding_types == {RemoteBinding}
    stages_by_id = {stage.id: stage for stage in workflow_ir.stages}
    if remote:
        for output_name, source in workflow_ir.outputs.items():
            if source.stage_id is None:
                continue
            source_port = source.output_name
            assert source_port is not None
            value_spec = stages_by_id[source.stage_id].contract.outputs[source_port]
            if value_spec.type not in INLINE_VALUE_TYPES:
                raise WorkflowValidationError(
                    f"remote workflow output {output_name!r} cannot carry value type "
                    f"{value_spec.type!r} inline"
                )

    edges = tuple(
        EdgePlan(
            source=source,
            target_stage=stage.id,
            target_port=port,
            carrier=INLINE_CARRIER if remote else IN_PROCESS_CARRIER,
        )
        for stage in workflow_ir.stages
        for port, source in stage.inputs.items()
    )
    if remote:
        for edge in edges:
            value_spec = stages_by_id[edge.target_stage].contract.inputs[
                edge.target_port
            ]
            if value_spec.type not in INLINE_VALUE_TYPES:
                raise WorkflowValidationError(
                    f"remote edge targeting stage {edge.target_stage!r} port "
                    f"{edge.target_port!r} cannot carry value type "
                    f"{value_spec.type!r} inline"
                )
    return ExecutionPlan(workflow_ir, deployment.bindings, edges)
