# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory physical plans for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

from dynamo.experimental.workflow.ir import WorkflowIR
from dynamo.experimental.workflow.types import (
    StageContract,
    ValueRef,
    WorkflowValidationError,
    validate_name,
)

LOCAL_CARRIER = "local"


@dataclass(frozen=True)
class LocalBinding:
    """Resolve one logical stage to a named in-process runner at bind time."""

    runner_key: str

    def __post_init__(self) -> None:
        validate_name(self.runner_key, "local runner key")


Binding = LocalBinding


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
        if self.carrier != LOCAL_CARRIER:
            raise WorkflowValidationError(f"unsupported edge carrier {self.carrier!r}")


@dataclass(frozen=True)
class ExecutionPlan:
    """A workflow plus in-memory placement and carrier decisions."""

    workflow: WorkflowIR
    bindings: Mapping[str, Binding]
    edges: Tuple[EdgePlan, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, WorkflowIR):
            raise WorkflowValidationError("execution plan requires WorkflowIR")
        if not isinstance(self.bindings, Mapping):
            raise WorkflowValidationError("execution plan bindings must be a mapping")

        bindings: dict[str, Binding] = {}
        for stage_id, binding in sorted(self.bindings.items()):
            validate_name(stage_id, "binding stage id")
            if not isinstance(binding, LocalBinding):
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
            if edge.carrier != LOCAL_CARRIER:
                raise WorkflowValidationError(
                    f"local stage {key[0]!r} port {key[1]!r} requires local carrier"
                )

        object.__setattr__(self, "bindings", MappingProxyType(bindings))
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(edges, key=lambda edge: (edge.target_stage, edge.target_port))
            ),
        )

    @property
    def stage_contracts(self) -> Mapping[str, StageContract]:
        """Return every stage contract keyed by its authored stage ID."""

        return MappingProxyType(
            {stage.id: stage.contract for stage in self.workflow.stages}
        )
