# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, placement-neutral workflow IR."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    ValueSpec,
    WorkflowValidationError,
    compatibility_error,
    validate_name,
)

WORKFLOW_SCHEMA = "dynamo.workflow.ir"
WORKFLOW_VERSION = 0


def _check_keys(data: Mapping[str, Any], required: set[str]) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing:
        raise WorkflowValidationError(f"missing fields: {sorted(missing)}")
    if unknown:
        raise WorkflowValidationError(f"unknown fields: {sorted(unknown)}")


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(sorted(values.items())))


@dataclass(frozen=True)
class StageIR:
    """One contracted stage instance in a workflow graph."""

    id: str
    contract: StageContract
    inputs: Mapping[str, ValueRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_name(self.id, "stage id")
        if not isinstance(self.contract, StageContract):
            raise WorkflowValidationError("stage contract must use StageContract")
        if not isinstance(self.inputs, Mapping):
            raise WorkflowValidationError("stage inputs must be a mapping")
        frozen: dict[str, ValueRef] = {}
        for name, reference in sorted(self.inputs.items()):
            validate_name(name, "stage input")
            if not isinstance(reference, ValueRef):
                raise WorkflowValidationError(f"stage input {name!r} must use ValueRef")
            frozen[name] = reference
        object.__setattr__(self, "inputs", MappingProxyType(frozen))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready representation."""

        return {
            "id": self.id,
            "contract": self.contract.to_dict(),
            "inputs": {
                name: reference.to_dict() for name, reference in self.inputs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageIR":
        """Parse one stage instance."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("stage must be an object")
        _check_keys(data, {"id", "contract", "inputs"})
        inputs = data["inputs"]
        if not isinstance(inputs, Mapping):
            raise WorkflowValidationError("stage inputs must be an object")
        return cls(
            id=data["id"],
            contract=StageContract.from_dict(data["contract"]),
            inputs={name: ValueRef.from_dict(ref) for name, ref in inputs.items()},
        )


@dataclass(frozen=True)
class WorkflowIR:
    """A validated, deterministic workflow graph without placement details."""

    name: str
    inputs: Mapping[str, ValueSpec]
    stages: Tuple[StageIR, ...]
    outputs: Mapping[str, ValueRef]

    def __post_init__(self) -> None:
        validate_name(self.name, "workflow name")
        if not isinstance(self.inputs, Mapping) or not isinstance(
            self.outputs, Mapping
        ):
            raise WorkflowValidationError(
                "workflow inputs and outputs must be mappings"
            )

        inputs: dict[str, ValueSpec] = {}
        for name, spec in sorted(self.inputs.items()):
            validate_name(name, "workflow input")
            if not isinstance(spec, ValueSpec):
                raise WorkflowValidationError(
                    f"workflow input {name!r} must use ValueSpec"
                )
            inputs[name] = spec

        outputs: dict[str, ValueRef] = {}
        for name, reference in sorted(self.outputs.items()):
            validate_name(name, "workflow output")
            if not isinstance(reference, ValueRef):
                raise WorkflowValidationError(
                    f"workflow output {name!r} must use ValueRef"
                )
            outputs[name] = reference
        if not outputs:
            raise WorkflowValidationError("workflow requires at least one output")

        stages = tuple(self.stages)
        ordered = self._validate_and_order(inputs, stages, outputs)
        object.__setattr__(self, "inputs", _freeze_mapping(inputs))
        object.__setattr__(self, "stages", ordered)
        object.__setattr__(self, "outputs", _freeze_mapping(outputs))

    @staticmethod
    def _validate_and_order(
        workflow_inputs: Mapping[str, ValueSpec],
        stages: Sequence[StageIR],
        workflow_outputs: Mapping[str, ValueRef],
    ) -> Tuple[StageIR, ...]:
        by_id: dict[str, StageIR] = {}
        contracts: dict[str, StageContract] = {}
        for stage in stages:
            if not isinstance(stage, StageIR):
                raise WorkflowValidationError("workflow stages must use StageIR")
            if stage.id in by_id:
                raise WorkflowValidationError(f"duplicate stage id {stage.id!r}")
            prior_contract = contracts.get(stage.contract.id)
            if prior_contract is not None and prior_contract != stage.contract:
                raise WorkflowValidationError(
                    f"contract id {stage.contract.id!r} has conflicting schemas"
                )
            contracts[stage.contract.id] = stage.contract
            by_id[stage.id] = stage

        dependencies: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        consumers: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        input_reachable: set[str] = set()

        for stage in stages:
            expected = set(stage.contract.inputs)
            actual = set(stage.inputs)
            if actual != expected:
                raise WorkflowValidationError(
                    f"stage {stage.id!r} inputs differ from its contract; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )
            for port_name, reference in stage.inputs.items():
                producer_spec, producer_stage = WorkflowIR._resolve_reference(
                    reference, workflow_inputs, by_id
                )
                error = compatibility_error(
                    producer_spec, stage.contract.inputs[port_name]
                )
                if error is not None:
                    raise WorkflowValidationError(
                        f"stage {stage.id!r} input {port_name!r} is incompatible: {error}"
                    )
                if producer_stage is None:
                    input_reachable.add(stage.id)
                else:
                    dependencies[stage.id].add(producer_stage)
                    consumers[producer_stage].add(stage.id)

        indegree = {
            stage_id: len(required) for stage_id, required in dependencies.items()
        }
        ready = [stage_id for stage_id, count in indegree.items() if count == 0]
        heapq.heapify(ready)
        ordered_ids: list[str] = []
        while ready:
            stage_id = heapq.heappop(ready)
            ordered_ids.append(stage_id)
            for consumer in sorted(consumers[stage_id]):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    heapq.heappush(ready, consumer)
        if len(ordered_ids) != len(stages):
            cyclic = sorted(
                stage_id for stage_id, count in indegree.items() if count > 0
            )
            raise WorkflowValidationError(
                f"workflow contains a cycle involving {cyclic}"
            )

        reachable = set(input_reachable)
        for stage_id in ordered_ids:
            if dependencies[stage_id] & reachable:
                reachable.add(stage_id)
        unreachable = set(by_id) - reachable
        if unreachable:
            raise WorkflowValidationError(
                f"stages are not reachable from workflow inputs: {sorted(unreachable)}"
            )

        live: set[str] = set()
        pending: list[str] = []
        for reference in workflow_outputs.values():
            _, producer_stage = WorkflowIR._resolve_reference(
                reference, workflow_inputs, by_id
            )
            if producer_stage is not None:
                pending.append(producer_stage)
        while pending:
            stage_id = pending.pop()
            if stage_id in live:
                continue
            live.add(stage_id)
            pending.extend(dependencies[stage_id])
        dead = set(by_id) - live
        if dead:
            raise WorkflowValidationError(
                f"stages do not contribute to workflow outputs: {sorted(dead)}"
            )

        return tuple(by_id[stage_id] for stage_id in ordered_ids)

    @staticmethod
    def _resolve_reference(
        reference: ValueRef,
        workflow_inputs: Mapping[str, ValueSpec],
        stages: Mapping[str, StageIR],
    ) -> tuple[ValueSpec, Optional[str]]:
        if reference.input_name is not None:
            if reference.input_name not in workflow_inputs:
                raise WorkflowValidationError(
                    f"unknown workflow input {reference.input_name!r}"
                )
            return workflow_inputs[reference.input_name], None

        stage_id = cast(str, reference.stage_id)
        output_name = cast(str, reference.output_name)
        if stage_id not in stages:
            raise WorkflowValidationError(f"unknown stage {stage_id!r}")
        stage = stages[stage_id]
        if output_name not in stage.contract.outputs:
            raise WorkflowValidationError(
                f"unknown output {output_name!r} on stage {stage_id!r}"
            )
        return stage.contract.outputs[output_name], stage_id

    def output_spec(self, name: str) -> ValueSpec:
        """Return the value description inferred for one workflow output."""

        if name not in self.outputs:
            raise KeyError(name)
        by_id = {stage.id: stage for stage in self.stages}
        spec, _ = self._resolve_reference(self.outputs[name], self.inputs, by_id)
        return spec

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready representation."""

        return {
            "schema": WORKFLOW_SCHEMA,
            "version": WORKFLOW_VERSION,
            "name": self.name,
            "inputs": {name: spec.to_dict() for name, spec in self.inputs.items()},
            "stages": [stage.to_dict() for stage in self.stages],
            "outputs": {
                name: reference.to_dict() for name, reference in self.outputs.items()
            },
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize deterministically as JSON."""

        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    def write_json(self, path: Path) -> None:
        """Write pretty, deterministic JSON to a file."""

        path.write_text(f"{self.to_json(indent=2)}\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowIR":
        """Parse and validate a canonical workflow object."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("workflow IR must be an object")
        _check_keys(data, {"schema", "version", "name", "inputs", "stages", "outputs"})
        if data["schema"] != WORKFLOW_SCHEMA:
            raise WorkflowValidationError(
                f"unsupported workflow schema {data['schema']!r}"
            )
        if data["version"] != WORKFLOW_VERSION:
            raise WorkflowValidationError(
                f"unsupported workflow version {data['version']!r}"
            )
        inputs = data["inputs"]
        stages = data["stages"]
        outputs = data["outputs"]
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise WorkflowValidationError("workflow inputs and outputs must be objects")
        if not isinstance(stages, list):
            raise WorkflowValidationError("workflow stages must be an array")
        return cls(
            name=data["name"],
            inputs={name: ValueSpec.from_dict(spec) for name, spec in inputs.items()},
            stages=tuple(StageIR.from_dict(stage) for stage in stages),
            outputs={name: ValueRef.from_dict(ref) for name, ref in outputs.items()},
        )

    @classmethod
    def from_json(cls, value: str) -> "WorkflowIR":
        """Parse JSON while rejecting duplicate object keys."""

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise WorkflowValidationError(f"duplicate JSON key {key!r}")
                result[key] = item
            return result

        try:
            data = json.loads(value, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as error:
            raise WorkflowValidationError(
                f"invalid workflow JSON: {error.msg}"
            ) from error
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: Path) -> "WorkflowIR":
        """Read and validate workflow JSON from a file."""

        return cls.from_json(path.read_text(encoding="utf-8"))
