# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical value and stage contracts for Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union


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
class ValueSpec:
    """A small logical description of a value crossing a workflow port."""

    type: str
    dtype: Optional[str] = None
    shape: Optional[Tuple[ShapeDimension, ...]] = None
    mode: Optional[str] = None
    class_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, str):
            raise WorkflowValidationError("value type must be a string")
        if self.type not in _VALUE_TYPES:
            raise WorkflowValidationError(
                f"unsupported value type {self.type!r}; expected one of {sorted(_VALUE_TYPES)}"
            )

        if self.shape is not None:
            if not isinstance(self.shape, (list, tuple)):
                raise WorkflowValidationError("tensor shape must be a list or tuple")
            object.__setattr__(self, "shape", tuple(self.shape))

        if self.type == "tensor":
            self._validate_tensor()
            self._reject("mode", self.mode)
            self._reject("class_id", self.class_id)
        elif self.type == "image":
            self._reject("dtype", self.dtype)
            self._reject("shape", self.shape)
            self._reject("class_id", self.class_id)
            if self.mode is not None and (
                not isinstance(self.mode, str) or not self.mode
            ):
                raise WorkflowValidationError(
                    "image mode must be a non-empty string when set"
                )
            if self.mode is not None:
                _validate_utf8(self.mode, "image mode")
        elif self.type == "object":
            self._reject("dtype", self.dtype)
            self._reject("shape", self.shape)
            self._reject("mode", self.mode)
            if not isinstance(self.class_id, str) or not self.class_id:
                raise WorkflowValidationError(
                    "object values require a non-empty class_id"
                )
            _validate_utf8(self.class_id, "object class_id")
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
        if self.dtype is not None:
            if not isinstance(self.dtype, str):
                raise WorkflowValidationError("tensor dtype must be a string")
            if self.dtype not in _TENSOR_DTYPES:
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


@dataclass(frozen=True)
class StreamSpec:
    """The item contract for a logical stream crossing a workflow port.

    Streaming execution is intentionally not implemented by the current compiler.
    """

    item: ValueSpec

    def __post_init__(self) -> None:
        if not isinstance(self.item, ValueSpec):
            raise WorkflowValidationError("stream items must use ValueSpec")


PortSpec = Union[ValueSpec, StreamSpec]


def _require_value_spec(spec: PortSpec, location: str) -> ValueSpec:
    """Narrow a port used by the current non-streaming execution path."""

    if isinstance(spec, StreamSpec):
        raise WorkflowValidationError(
            f"{location} uses a stream port, but stream execution is not supported"
        )
    return spec


def compatibility_error(producer: PortSpec, consumer: PortSpec) -> Optional[str]:
    """Explain why a producer cannot satisfy a consumer, or return ``None``."""

    if isinstance(producer, StreamSpec) or isinstance(consumer, StreamSpec):
        if not isinstance(producer, StreamSpec):
            return "value output does not satisfy a stream input"
        if not isinstance(consumer, StreamSpec):
            return "stream output does not satisfy a value input"
        item_error = compatibility_error(producer.item, consumer.item)
        return None if item_error is None else f"stream item {item_error}"

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
    inputs: Mapping[str, PortSpec] = field(default_factory=dict)
    outputs: Mapping[str, PortSpec] = field(default_factory=dict)

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
        ports: Mapping[str, PortSpec], kind: str
    ) -> Mapping[str, PortSpec]:
        if not isinstance(ports, Mapping):
            raise WorkflowValidationError(f"{kind}s must be a mapping")
        frozen: dict[str, PortSpec] = {}
        for name, spec in sorted(ports.items()):
            validate_name(name, kind)
            if not isinstance(spec, (ValueSpec, StreamSpec)):
                raise WorkflowValidationError(
                    f"{kind} {name!r} must use ValueSpec or StreamSpec"
                )
            frozen[name] = spec
        return MappingProxyType(frozen)


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
