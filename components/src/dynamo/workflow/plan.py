# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable physical plans for Dynamo workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from dynamo.workflow.ir import WorkflowIR
from dynamo.workflow.types import ValueRef, WorkflowValidationError, validate_name

EXECUTION_PLAN_SCHEMA = "dynamo.workflow.execution_plan"
EXECUTION_PLAN_VERSION = 0
LOCAL_CARRIER = "local"


def _check_keys(
    data: Mapping[str, Any], required: set[str], optional: frozenset[str] = frozenset()
) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise WorkflowValidationError(f"missing fields: {sorted(missing)}")
    if unknown:
        raise WorkflowValidationError(f"unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class LocalBinding:
    """Resolve one logical stage to a named in-process runner at hydration time."""

    runner_key: str

    def __post_init__(self) -> None:
        validate_name(self.runner_key, "local runner key")

    def to_dict(self) -> dict[str, str]:
        return {"kind": "local", "runner_key": self.runner_key}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalBinding":
        if not isinstance(data, Mapping):
            raise WorkflowValidationError("binding must be an object")
        _check_keys(data, {"kind", "runner_key"})
        if data["kind"] != "local":
            raise WorkflowValidationError(f"unsupported binding kind {data['kind']!r}")
        return cls(runner_key=data["runner_key"])


Binding = LocalBinding


def binding_from_dict(data: Mapping[str, Any]) -> Binding:
    if not isinstance(data, Mapping):
        raise WorkflowValidationError("binding must be an object")
    kind = data.get("kind")
    if kind == "local":
        return LocalBinding.from_dict(data)
    raise WorkflowValidationError(f"unsupported binding kind {kind!r}")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": {"stage": self.target_stage, "port": self.target_port},
            "carrier": self.carrier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EdgePlan":
        if not isinstance(data, Mapping):
            raise WorkflowValidationError("edge plan must be an object")
        _check_keys(data, {"source", "target", "carrier"})
        target = data["target"]
        if not isinstance(target, Mapping):
            raise WorkflowValidationError("edge target must be an object")
        _check_keys(target, {"stage", "port"})
        return cls(
            source=ValueRef.from_dict(data["source"]),
            target_stage=target["stage"],
            target_port=target["port"],
            carrier=data["carrier"],
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """A serializable workflow plus placement and carrier decisions."""

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

        expected_stages = {stage.id for stage in self.workflow.stages}
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
                    "does not match WorkflowIR"
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

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready representation."""

        return {
            "schema": EXECUTION_PLAN_SCHEMA,
            "version": EXECUTION_PLAN_VERSION,
            "workflow": self.workflow.to_dict(),
            "bindings": {
                stage_id: binding.to_dict()
                for stage_id, binding in self.bindings.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize deterministically as JSON."""

        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    def write_json(self, path: Path) -> None:
        """Write pretty, deterministic JSON to a file."""

        path.write_text(f"{self.to_json(indent=2)}\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        """Parse and validate a canonical execution plan object."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("execution plan must be an object")
        _check_keys(data, {"schema", "version", "workflow", "bindings", "edges"})
        if data["schema"] != EXECUTION_PLAN_SCHEMA:
            raise WorkflowValidationError(
                f"unsupported execution plan schema {data['schema']!r}"
            )
        version = data["version"]
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != EXECUTION_PLAN_VERSION
        ):
            raise WorkflowValidationError(
                f"unsupported execution plan version {version!r}"
            )
        bindings = data["bindings"]
        edges = data["edges"]
        if not isinstance(bindings, Mapping):
            raise WorkflowValidationError("execution plan bindings must be an object")
        if not isinstance(edges, list):
            raise WorkflowValidationError("execution plan edges must be an array")
        return cls(
            workflow=WorkflowIR.from_dict(data["workflow"]),
            bindings={
                stage_id: binding_from_dict(binding)
                for stage_id, binding in bindings.items()
            },
            edges=tuple(EdgePlan.from_dict(edge) for edge in edges),
        )

    @classmethod
    def from_json(cls, value: str) -> "ExecutionPlan":
        """Parse strict JSON while rejecting duplicate object keys."""

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise WorkflowValidationError(f"duplicate JSON key {key!r}")
                result[key] = item
            return result

        def reject_non_finite(constant: str) -> None:
            raise WorkflowValidationError(
                f"non-finite JSON constant {constant!r} is not supported"
            )

        try:
            data = json.loads(
                value,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_non_finite,
            )
        except json.JSONDecodeError as error:
            raise WorkflowValidationError(
                f"invalid execution plan JSON: {error.msg}"
            ) from error
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: Path) -> "ExecutionPlan":
        """Read and validate execution plan JSON from a file."""

        return cls.from_json(path.read_text(encoding="utf-8"))
