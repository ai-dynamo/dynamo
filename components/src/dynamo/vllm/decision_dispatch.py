# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private decision transport without an optional executor dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.llm.exceptions import HttpError, WorkerUnavailable

if TYPE_CHECKING:
    from .decision_native_registry import NativePublication

DECISION_DISPATCH_METADATA_KEY = "dynamo.internal.decision"


@dataclass(frozen=True)
class DecisionDispatcher:
    """Own private decision transport without loading an optional executor."""

    kind: Literal["head", "read"]
    _wire_key: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_wire_key", f"decision_{self.kind}")

    @property
    def publication(self) -> NativePublication | None:
        return None

    def health_payload(self, model_name):
        return None

    def maybe_handle(self, request, context):
        if (
            self.kind == "head"
            and request.get(HEALTH_CHECK_KEY) is True
            and "native_pooling_health" in request
        ):
            return self._health(request, context)
        if self._wire_key not in request:
            return None
        metadata = getattr(context, "metadata", {})
        if metadata.get(DECISION_DISPATCH_METADATA_KEY) == self.kind:
            return self._execute(request, context)
        return None

    async def _execute(self, request, context):
        response = await self._run_decision(request, context)
        yield {**response, "finish_reason": "stop"}

    async def _run_decision(self, request, context):
        raise HttpError(
            503, "native decision execution is not available on this worker"
        )

    async def _health(self, request, context):
        if (
            set(request) != {"native_pooling_health", "model", HEALTH_CHECK_KEY}
            or request["native_pooling_health"] is not True
            or request[HEALTH_CHECK_KEY] is not True
            or not isinstance(request["model"], str)
        ):
            raise ValueError("invalid internal pooling health request")
        await self._probe(context)
        yield {"status": "ok"}

    async def _probe(self, context):
        raise WorkerUnavailable("native pooling health is not initialized")
