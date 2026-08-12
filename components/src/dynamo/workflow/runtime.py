# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding and execution for compiled Dynamo workflows."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from dynamo.workflow.types import PortSpec, StageContract, StreamSpec


class WorkflowExecutionError(RuntimeError):
    """Raised when runtime values do not honor the authored workflow."""


@dataclass(frozen=True)
class WorkflowAttempt:
    """Shared attempt identity, deadline, and cancellation state."""

    attempt_id: str
    deadline: Optional[float]
    cancelled: asyncio.Event
    request_context: Any = None


@dataclass(frozen=True)
class StageContext:
    """Attempt metadata available to a running stage."""

    workflow_name: str
    stage_id: str
    attempt_id: str
    invocation_id: str
    deadline: Optional[float]
    _cancelled: asyncio.Event
    request_context: Any = None

    @property
    def cancelled(self) -> bool:
        """Whether the workflow attempt is terminating."""

        return self._cancelled.is_set()

    def remaining_time(self) -> Optional[float]:
        """Return seconds until the attempt deadline, when one exists."""

        if self.deadline is None:
            return None
        return max(0.0, self.deadline - asyncio.get_running_loop().time())

    def raise_if_cancelled(self) -> None:
        """Cooperatively stop work after cancellation or deadline expiry."""

        if self.cancelled:
            raise asyncio.CancelledError
        if self.deadline is not None and self.remaining_time() == 0:
            raise asyncio.TimeoutError


@runtime_checkable
class StageRunner(Protocol):
    """The small interface implemented by custom and Dynamo-provided workers."""

    contract: StageContract

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        """Run one stage attempt and return all declared outputs."""

        ...


@runtime_checkable
class TensorCarrier(Protocol):
    """Runtime-bound carrier used when a NIXL edge touches this process."""

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        ...

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        ...

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        ...


@runtime_checkable
class _TensorValue(Protocol):
    shape: Any
    dtype: Any


@runtime_checkable
class _ImageValue(Protocol):
    mode: str
    size: Any


def _validate_value(spec: PortSpec, value: Any, location: str) -> None:
    if isinstance(spec, StreamSpec):
        raise WorkflowExecutionError(
            f"{location} uses a stream port, but stream execution is not supported"
        )
    if spec.type == "text" and not isinstance(value, str):
        raise WorkflowExecutionError(f"{location} must be text")
    if spec.type == "bytes" and not isinstance(value, (bytes, bytearray, memoryview)):
        raise WorkflowExecutionError(f"{location} must be bytes-like")
    if spec.type == "json" and not _is_json_value(value, set()):
        raise WorkflowExecutionError(f"{location} must use the JSON data model")
    if spec.type == "tensor":
        if not isinstance(value, _TensorValue):
            raise WorkflowExecutionError(f"{location} must be tensor-like")
        actual_dtype = str(value.dtype).rsplit(".", 1)[-1]
        if spec.dtype is not None and actual_dtype != spec.dtype:
            raise WorkflowExecutionError(
                f"{location} has dtype {actual_dtype!r}, expected {spec.dtype!r}"
            )
        actual_shape = tuple(value.shape)
        if spec.shape is not None:
            if len(actual_shape) != len(spec.shape) or any(
                expected != "dynamic" and expected != actual
                for actual, expected in zip(actual_shape, spec.shape)
            ):
                raise WorkflowExecutionError(
                    f"{location} has shape {actual_shape!r}, expected {spec.shape!r}"
                )
    if spec.type == "image":
        if not isinstance(value, _ImageValue):
            raise WorkflowExecutionError(f"{location} must be image-like")
        if spec.mode is not None and value.mode != spec.mode:
            raise WorkflowExecutionError(
                f"{location} has image mode {value.mode!r}, expected {spec.mode!r}"
            )


def _is_json_value(value: Any, active_containers: set[int]) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if not isinstance(value, (list, dict)):
        return False

    container_id = id(value)
    if container_id in active_containers:
        return False
    active_containers.add(container_id)
    try:
        if isinstance(value, list):
            return all(_is_json_value(item, active_containers) for item in value)
        return all(
            isinstance(key, str) and _is_json_value(item, active_containers)
            for key, item in value.items()
        )
    finally:
        active_containers.remove(container_id)
