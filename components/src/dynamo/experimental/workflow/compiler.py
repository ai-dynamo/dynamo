# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower placement-neutral workflows into physical execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union

from dynamo.experimental.workflow.builder import Workflow
from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.plan import (
    INLINE_VALUE_TYPES,
    Binding,
    ExecutionPlan,
    InlineBinding,
    RemoteBinding,
    select_edge_carrier,
)
from dynamo.experimental.workflow.types import (
    StreamSpec,
    WorkflowValidationError,
    _require_value_spec,
    validate_name,
)


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
    def remote(
        cls, *, tensor_carrier: str | None = None, **endpoint_ids: str
    ) -> "DeploymentSpec":
        """Build round-robin bindings to discovered Dynamo endpoints."""

        return cls(
            bindings={
                stage_id: RemoteBinding(endpoint_id, tensor_carrier=tensor_carrier)
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

    stages_by_id = {stage.id: stage for stage in workflow_ir.stages}
    for output_name, source in workflow_ir.outputs.items():
        if source.stage_id is None or not isinstance(
            deployment.bindings[source.stage_id], RemoteBinding
        ):
            continue
        source_port = source.output_name
        assert source_port is not None
        value_spec = _require_value_spec(
            stages_by_id[source.stage_id].contract.outputs[source_port],
            f"workflow output {output_name!r}",
        )
        if value_spec.type not in INLINE_VALUE_TYPES:
            raise WorkflowValidationError(
                f"remote workflow output {output_name!r} cannot carry value type "
                f"{value_spec.type!r} inline"
            )

    edges = []
    for stage in workflow_ir.stages:
        target_binding = deployment.bindings[stage.id]
        for port, source in stage.inputs.items():
            if source.input_name is not None:
                source_binding = None
            else:
                stage_id = source.stage_id
                assert stage_id is not None
                source_binding = deployment.bindings[stage_id]
            value_spec = _require_value_spec(
                stage.contract.inputs[port],
                f"stage {stage.id!r} input {port!r}",
            )
            carrier = select_edge_carrier(
                source_binding, target_binding, value_spec.type
            )
            edges.append(
                EdgePlan(
                    source=source,
                    target_stage=stage.id,
                    target_port=port,
                    carrier=carrier,
                )
            )
    return ExecutionPlan(workflow_ir, deployment.bindings, tuple(edges))
