# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable value and stage contracts for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Union


class WorkflowValidationError(ValueError):
    """Raised when a workflow contract or graph is invalid."""


ShapeDimension = Union[int, str]
_VALUE_TYPES = frozenset({"tensor", "text", "image", "bytes", "json", "object"})
_TENSOR_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    }
)


def _check_keys(
    data: Mapping[str, Any], required: set[str], optional: set[str]
) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise WorkflowValidationError(f"missing fields: {sorted(missing)}")
    if unknown:
        raise WorkflowValidationError(f"unknown fields: {sorted(unknown)}")


def validate_name(name: str, kind: str) -> None:
    """Validate a portable workflow identifier."""

    if not isinstance(name, str) or not name:
        raise WorkflowValidationError(f"{kind} must be a non-empty string")
    if not name[0].isalpha() or not all(
        char.isalnum() or char in "_-" for char in name
    ):
        raise WorkflowValidationError(
            f"{kind} {name!r} must start with a letter and contain only letters, "
            "digits, '_' or '-'"
        )


@dataclass(frozen=True)
class ValueSpec:
    """A small, serializable description of a value crossing a workflow port."""

    type: str
    dtype: Optional[str] = None
    shape: Optional[Tuple[ShapeDimension, ...]] = None
    mode: Optional[str] = None
    class_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in _VALUE_TYPES:
            raise WorkflowValidationError(
                f"unsupported value type {self.type!r}; expected one of {sorted(_VALUE_TYPES)}"
            )

        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))

        if self.type == "tensor":
            self._validate_tensor()
            self._reject("mode", self.mode)
            self._reject("class_id", self.class_id)
        elif self.type == "image":
            self._reject("dtype", self.dtype)
            self._reject("shape", self.shape)
            self._reject("class_id", self.class_id)
            if self.mode is not None and not self.mode:
                raise WorkflowValidationError("image mode must be non-empty when set")
        elif self.type == "object":
            self._reject("dtype", self.dtype)
            self._reject("shape", self.shape)
            self._reject("mode", self.mode)
            if not isinstance(self.class_id, str) or not self.class_id:
                raise WorkflowValidationError(
                    "object values require a non-empty class_id"
                )
        else:
            self._reject("dtype", self.dtype)
            self._reject("shape", self.shape)
            self._reject("mode", self.mode)
            self._reject("class_id", self.class_id)

    @staticmethod
    def _reject(field_name: str, value: object) -> None:
        if value is not None:
            raise WorkflowValidationError(
                f"{field_name} is not valid for this value type"
            )

    def _validate_tensor(self) -> None:
        if self.dtype is not None and self.dtype not in _TENSOR_DTYPES:
            raise WorkflowValidationError(
                f"unsupported tensor dtype {self.dtype!r}; expected one of "
                f"{sorted(_TENSOR_DTYPES)}"
            )
        if self.shape is None:
            return
        for dimension in self.shape:
            if dimension == "dynamic":
                continue
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise WorkflowValidationError(
                    "tensor shape dimensions must be non-negative integers or 'dynamic'"
                )
            if dimension < 0:
                raise WorkflowValidationError(
                    "tensor shape dimensions must be non-negative integers or 'dynamic'"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready representation."""

        data: dict[str, Any] = {"type": self.type}
        if self.dtype is not None:
            data["dtype"] = self.dtype
        if self.shape is not None:
            data["shape"] = list(self.shape)
        if self.mode is not None:
            data["mode"] = self.mode
        if self.class_id is not None:
            data["class_id"] = self.class_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueSpec":
        """Parse a value description while rejecting unknown fields."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("value spec must be an object")
        _check_keys(data, {"type"}, {"dtype", "shape", "mode", "class_id"})
        shape = data.get("shape")
        if shape is not None and (
            not isinstance(shape, list) or isinstance(shape, (str, bytes))
        ):
            raise WorkflowValidationError("tensor shape must be a JSON array")
        return cls(
            type=data["type"],
            dtype=data.get("dtype"),
            shape=None if shape is None else tuple(shape),
            mode=data.get("mode"),
            class_id=data.get("class_id"),
        )


def compatibility_error(producer: ValueSpec, consumer: ValueSpec) -> Optional[str]:
    """Explain why a producer cannot satisfy a consumer, or return ``None``."""

    if producer.type != consumer.type:
        return f"type {producer.type!r} does not satisfy {consumer.type!r}"

    if consumer.type == "tensor":
        if consumer.dtype is not None and producer.dtype != consumer.dtype:
            return (
                f"producer dtype {producer.dtype!r} does not guarantee "
                f"consumer dtype {consumer.dtype!r}"
            )
        if consumer.shape is not None:
            if producer.shape is None:
                return "producer does not guarantee the consumer tensor shape"
            if len(producer.shape) != len(consumer.shape):
                return "producer and consumer tensor ranks differ"
            for produced, consumed in zip(producer.shape, consumer.shape):
                if consumed != "dynamic" and produced != consumed:
                    return (
                        f"producer dimension {produced!r} does not guarantee "
                        f"consumer dimension {consumed!r}"
                    )

    if consumer.type == "image" and consumer.mode is not None:
        if producer.mode != consumer.mode:
            return (
                f"producer mode {producer.mode!r} does not guarantee "
                f"consumer mode {consumer.mode!r}"
            )

    if consumer.type == "object" and producer.class_id != consumer.class_id:
        return (
            f"producer class_id {producer.class_id!r} does not satisfy "
            f"consumer class_id {consumer.class_id!r}"
        )

    return None


@dataclass(frozen=True)
class StageContract:
    """The complete named input and output surface of a workflow stage."""

    id: str
    inputs: Mapping[str, ValueSpec] = field(default_factory=dict)
    outputs: Mapping[str, ValueSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_name(self.id, "contract id")
        inputs = self._freeze_ports(self.inputs, "input port")
        outputs = self._freeze_ports(self.outputs, "output port")
        if not outputs:
            raise WorkflowValidationError("stage contracts require at least one output")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)

    @staticmethod
    def _freeze_ports(
        ports: Mapping[str, ValueSpec], kind: str
    ) -> Mapping[str, ValueSpec]:
        if not isinstance(ports, Mapping):
            raise WorkflowValidationError(f"{kind}s must be a mapping")
        frozen: dict[str, ValueSpec] = {}
        for name, spec in sorted(ports.items()):
            validate_name(name, kind)
            if not isinstance(spec, ValueSpec):
                raise WorkflowValidationError(f"{kind} {name!r} must use ValueSpec")
            frozen[name] = spec
        return MappingProxyType(frozen)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready representation."""

        return {
            "id": self.id,
            "inputs": {name: spec.to_dict() for name, spec in self.inputs.items()},
            "outputs": {name: spec.to_dict() for name, spec in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageContract":
        """Parse a stage contract while rejecting unknown fields."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("stage contract must be an object")
        _check_keys(data, {"id", "inputs", "outputs"}, set())
        inputs = data["inputs"]
        outputs = data["outputs"]
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise WorkflowValidationError("contract inputs and outputs must be objects")
        return cls(
            id=data["id"],
            inputs={name: ValueSpec.from_dict(spec) for name, spec in inputs.items()},
            outputs={name: ValueSpec.from_dict(spec) for name, spec in outputs.items()},
        )


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

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-ready representation."""

        if self.input_name is not None:
            return {"input": self.input_name}
        return {"stage": self.stage_id, "output": self.output_name}  # type: ignore[dict-item]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueRef":
        """Parse a structured reference."""

        if not isinstance(data, Mapping):
            raise WorkflowValidationError("value reference must be an object")
        if set(data) == {"input"}:
            return cls.for_input(data["input"])
        if set(data) == {"stage", "output"}:
            return cls.for_stage_output(data["stage"], data["output"])
        raise WorkflowValidationError(
            "value reference must contain either 'input' or both 'stage' and 'output'"
        )
