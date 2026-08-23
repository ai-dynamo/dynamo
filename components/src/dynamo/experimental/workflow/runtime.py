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
