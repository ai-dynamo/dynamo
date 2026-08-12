# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory physical plans for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union

from dynamo.workflow.ir import WorkflowIR
from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    WorkflowValidationError,
    _require_value_spec,
    validate_name,
)

IN_PROCESS_CARRIER = "in_process"
INLINE_CARRIER = "inline"
NIXL_CARRIER = "nixl"
INLINE_VALUE_TYPES = frozenset({"bytes", "json", "text"})


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
    tensor_carrier: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_endpoint_id(self.endpoint_id)
        if self.routing_policy != "round_robin":
            raise WorkflowValidationError(
                f"unsupported remote routing policy {self.routing_policy!r}"
            )
        if self.tensor_carrier not in {None, NIXL_CARRIER}:
            raise WorkflowValidationError(
                f"unsupported remote tensor carrier {self.tensor_carrier!r}"
            )


Binding = Union[InlineBinding, RemoteBinding]


def select_edge_carrier(
    source_binding: Optional[Binding],
    target_binding: Binding,
    value_type: str,
) -> str:
    """Select one declared carrier for a physical producer-consumer edge."""

    if isinstance(target_binding, InlineBinding) and (
        source_binding is None or isinstance(source_binding, InlineBinding)
    ):
        return IN_PROCESS_CARRIER
    if value_type in INLINE_VALUE_TYPES:
        return INLINE_CARRIER
    if value_type == "tensor":
        if (
            isinstance(source_binding, RemoteBinding)
            and isinstance(target_binding, RemoteBinding)
            and source_binding.tensor_carrier == NIXL_CARRIER
            and target_binding.tensor_carrier == NIXL_CARRIER
        ):
            return NIXL_CARRIER
    raise WorkflowValidationError(
        f"cross-process value type {value_type!r} has no common declared carrier"
    )


@dataclass(frozen=True)
class EdgePlan:
    """One physical producer-to-consumer connection in an execution plan."""

    source: ValueRef
    target_stage: str
    target_port: str
    carrier: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, ValueRef):
            raise WorkflowValidationError("edge source must use ValueRef")
        validate_name(self.target_stage, "edge target stage")
        validate_name(self.target_port, "edge target port")
        if self.carrier not in {
            INLINE_CARRIER,
            IN_PROCESS_CARRIER,
            NIXL_CARRIER,
        }:
            raise WorkflowValidationError(f"unsupported edge carrier {self.carrier!r}")

    @property
    def transfer_id(self) -> str:
        """Return the stable per-consumer identity derived from the target port."""

        return f"{self.target_stage}.{self.target_port}"


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

        stages_by_id = {stage.id: stage for stage in self.workflow.stages}
        edges = tuple(self.edges)
        actual_edges: dict[tuple[str, str], EdgePlan] = {}
        for edge in edges:
            if not isinstance(edge, EdgePlan):
                raise WorkflowValidationError("execution plan edges must use EdgePlan")
            key = (edge.target_stage, edge.target_port)
            if key in actual_edges:
                raise WorkflowValidationError(
                    f"duplicate edge targeting stage {key[0]!r} port {key[1]!r}"
                )
            actual_edges[key] = edge

        expected_edges = {
            (stage.id, port): source
            for stage in self.workflow.stages
            for port, source in stage.inputs.items()
        }
        if set(actual_edges) != set(expected_edges):
            missing = sorted(set(expected_edges) - set(actual_edges))
            extra = sorted(set(actual_edges) - set(expected_edges))
            raise WorkflowValidationError(
                "execution plan edges differ from workflow inputs; "
                f"missing={missing}, extra={extra}"
            )
        for key, source in expected_edges.items():
            edge = actual_edges[key]
            if edge.source != source:
                raise WorkflowValidationError(
                    f"edge targeting stage {key[0]!r} port {key[1]!r} "
                    "does not match the workflow definition"
                )
            target_binding = bindings[key[0]]
            if source.input_name is not None:
                upstream_binding = None
            else:
                source_stage_id = source.stage_id
                assert source_stage_id is not None
                upstream_binding = bindings[source_stage_id]
            value_spec = _require_value_spec(
                stages_by_id[key[0]].contract.inputs[key[1]],
                f"stage {key[0]!r} input {key[1]!r}",
            )
            expected_carrier = select_edge_carrier(
                upstream_binding, target_binding, value_spec.type
            )
            if edge.carrier != expected_carrier:
                raise WorkflowValidationError(
                    f"stage {key[0]!r} port {key[1]!r} requires "
                    f"{expected_carrier!r} carrier"
                )
        outgoing_nixl = {
            (edge.source.stage_id, edge.source.output_name)
            for edge in edges
            if edge.carrier == NIXL_CARRIER and edge.source.stage_id is not None
        }
        for stage in self.workflow.stages:
            if not isinstance(bindings[stage.id], RemoteBinding):
                continue
            for output_name, port_spec in stage.contract.outputs.items():
                value_spec = _require_value_spec(
                    port_spec,
                    f"stage {stage.id!r} output {output_name!r}",
                )
                if (
                    value_spec.type == "tensor"
                    and (stage.id, output_name) not in outgoing_nixl
                ):
                    raise WorkflowValidationError(
                        f"remote tensor output {stage.id!r}.{output_name!r} has "
                        "no NIXL consumer edge"
                    )

        for output_name, source in self.workflow.outputs.items():
            if source.stage_id is None or not isinstance(
                bindings[source.stage_id], RemoteBinding
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
                    f"remote workflow output {output_name!r} cannot carry value "
                    f"type {value_spec.type!r} inline"
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
