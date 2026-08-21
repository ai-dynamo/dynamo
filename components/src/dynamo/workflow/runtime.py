# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding and execution for compiled Dynamo workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from dynamo.workflow.types import StageContract


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

    workflow_name: Optional[str]
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
    """Runtime-bound carrier for optional out-of-band tensor transport."""

    def can_export(self, value: Any) -> bool:
        """Whether this carrier can export the complete port value."""

        ...

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        ...

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        ...

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        ...
