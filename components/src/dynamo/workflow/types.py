# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named stage contracts and value references for Dynamo workflows."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import FrozenSet, Optional


class WorkflowValidationError(ValueError):
    """Raised when a workflow contract or graph is invalid."""


def validate_name(name: str, kind: str) -> None:
    """Validate a portable workflow identifier."""

    if not isinstance(name, str) or not name:
        raise WorkflowValidationError(f"{kind} must be a non-empty string")
    _validate_utf8(name, kind)
    if not name[0].isalpha() or not all(
        char.isalnum() or char in "_-" for char in name
    ):
        raise WorkflowValidationError(
            f"{kind} {name!r} must start with a letter and contain only letters, "
            "digits, '_' or '-'"
        )


def _validate_utf8(value: str, kind: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise WorkflowValidationError(
            f"{kind} must contain valid Unicode scalar values"
        ) from error


@dataclass(frozen=True)
class StageContract:
    """The complete named input and output surface of a workflow stage."""

    id: str
    inputs: FrozenSet[str] = field(default_factory=frozenset)
    outputs: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        validate_name(self.id, "contract id")
        inputs = self._freeze_ports(self.inputs, "input port")
        outputs = self._freeze_ports(self.outputs, "output port")
        if not outputs:
            raise WorkflowValidationError("stage contracts require at least one output")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)

    @staticmethod
    def _freeze_ports(ports: AbstractSet[str], kind: str) -> FrozenSet[str]:
        if not isinstance(ports, AbstractSet):
            raise WorkflowValidationError(f"{kind}s must be a set of names")
        frozen: set[str] = set()
        for name in ports:
            validate_name(name, kind)
            frozen.add(name)
        return frozenset(frozen)


@dataclass(frozen=True)
class ValueRef:
    """A structured reference to a workflow input or stage output."""

    input_name: Optional[str] = None
    stage_id: Optional[str] = None
    output_name: Optional[str] = None
    _owner: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        is_input = self.input_name is not None
        is_stage = self.stage_id is not None or self.output_name is not None
        if is_input == is_stage:
            raise WorkflowValidationError(
                "a value reference must name exactly one workflow input or stage output"
            )
        if self.input_name is not None:
            validate_name(self.input_name, "workflow input reference")
            return
        if self.stage_id is None or self.output_name is None:
            raise WorkflowValidationError(
                "stage references require stage and output names"
            )
        validate_name(self.stage_id, "stage reference")
        validate_name(self.output_name, "stage output reference")

    @classmethod
    def for_input(cls, name: str, owner: Optional[object] = None) -> "ValueRef":
        """Create a workflow-input reference."""

        return cls(input_name=name, _owner=owner)

    @classmethod
    def for_stage_output(
        cls, stage_id: str, output_name: str, owner: Optional[object] = None
    ) -> "ValueRef":
        """Create a stage-output reference."""

        return cls(stage_id=stage_id, output_name=output_name, _owner=owner)
