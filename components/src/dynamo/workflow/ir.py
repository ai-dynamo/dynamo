# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, placement-neutral workflow IR."""

from __future__ import annotations

import heapq
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple, TypeVar, cast

from dynamo.workflow.types import (
    StageContract,
    ValueRef,
    WorkflowValidationError,
    validate_contract_consistency,
    validate_name,
)

_T = TypeVar("_T")


def _freeze_mapping(values: Mapping[str, _T]) -> Mapping[str, _T]:
    return MappingProxyType(dict(sorted(values.items())))


@dataclass(frozen=True)
class StageIR:
    """One contracted stage instance in a workflow graph."""

    id: str
    # Complete port-name interface for this stage kind. ``contract.inputs``
    # declares required input ports, while ``contract.outputs`` declares outputs.
    contract: StageContract
    # Per-instance wiring from every declared ``contract.inputs`` port to the
    # workflow input or upstream stage output that supplies its value.
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
        expected = set(self.contract.inputs)
        actual = set(frozen)
        if actual != expected:
            raise WorkflowValidationError(
                f"stage {self.id!r} inputs differ from its contract; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        object.__setattr__(self, "inputs", MappingProxyType(frozen))


@dataclass(frozen=True)
class WorkflowIR:
    """The canonical validation boundary for a placement-neutral workflow graph.

    Validation here does not assume construction through :class:`Workflow`, so
    direct IR construction and future parsers receive the same graph guarantees.
    """

    name: str
    inputs: FrozenSet[str]
    stages: Tuple[StageIR, ...]
    outputs: Mapping[str, ValueRef]

    def __post_init__(self) -> None:
        validate_name(self.name, "workflow name")
        if not isinstance(self.inputs, AbstractSet):
            raise WorkflowValidationError("workflow inputs must be a set of names")
        if not isinstance(self.outputs, Mapping):
            raise WorkflowValidationError("workflow outputs must be a mapping")

        inputs: set[str] = set()
        for name in self.inputs:
            validate_name(name, "workflow input")
            inputs.add(name)

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
        object.__setattr__(self, "inputs", frozenset(inputs))
        object.__setattr__(self, "stages", ordered)
        object.__setattr__(self, "outputs", _freeze_mapping(outputs))

    @staticmethod
    def _validate_and_order(
        workflow_inputs: AbstractSet[str],
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
            validate_contract_consistency(
                contracts.get(stage.contract.id), stage.contract
            )
            contracts[stage.contract.id] = stage.contract
            by_id[stage.id] = stage

        dependencies: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        consumers: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}

        for stage in stages:
            for reference in stage.inputs.values():
                producer_stage = WorkflowIR._resolve_reference(
                    reference, workflow_inputs, by_id
                )
                if producer_stage is not None:
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

        live: set[str] = set()
        pending: list[str] = []
        for reference in workflow_outputs.values():
            producer_stage = WorkflowIR._resolve_reference(
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
        workflow_inputs: AbstractSet[str],
        stages: Mapping[str, StageIR],
    ) -> Optional[str]:
        if reference.input_name is not None:
            if reference.input_name not in workflow_inputs:
                raise WorkflowValidationError(
                    f"unknown workflow input {reference.input_name!r}"
                )
            return None

        stage_id = cast(str, reference.stage_id)
        output_name = cast(str, reference.output_name)
        if stage_id not in stages:
            raise WorkflowValidationError(f"unknown stage {stage_id!r}")
        stage = stages[stage_id]
        if output_name not in stage.contract.outputs:
            raise WorkflowValidationError(
                f"unknown output {output_name!r} on stage {stage_id!r}"
            )
        return stage_id
