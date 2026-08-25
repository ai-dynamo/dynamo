# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding and execution for compiled Dynamo workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from dynamo.experimental.workflow.types import StageContract


class WorkflowExecutionError(RuntimeError):
    """Raised when runtime values do not honor the authored workflow."""


@dataclass(frozen=True)
class StageContext:
    """Logical request identity available to one stage call.

    ``workflow_name`` is unavailable for remote stages until the transport
    carries workflow lineage explicitly.
    """

    workflow_name: str | None
    stage_id: str
    attempt_id: str


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
