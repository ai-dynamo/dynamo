# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

from dynamo.llm import KvRouter


JsonDict = dict[str, Any]
RequestLike = Mapping[str, Any]


@dataclass(slots=True)
class WorkerSelection:
    """Decision returned by a custom routing strategy."""

    worker_id: int
    dp_rank: int
    metadata: JsonDict = field(default_factory=dict)


class BaseCustomRouter(ABC):
    """Base class for Python-defined routing strategies.

    A concrete strategy only needs to implement ``select_worker()``. The base class
    handles request ID extraction and forwarding the request through ``KvRouter``
    using explicit ``worker_id`` and ``dp_rank`` overrides.
    """

    def __init__(self, kv_router: KvRouter):
        self.kv_router = kv_router

    @abstractmethod
    async def select_worker(
        self,
        request: RequestLike,
        request_id: str,
    ) -> WorkerSelection:
        """Choose a worker for a request."""

    async def get_worker_loads(self, request: RequestLike) -> list[JsonDict]:
        """Return per-worker load signals exposed by ``KvRouter``."""

        return await self.kv_router.get_potential_loads(
            token_ids=self.token_ids_from_request(request),
            lora_name=self.lora_name_from_request(request),
        )

    async def get_best_overlap(self, request: RequestLike) -> tuple[int, int, int]:
        """Return the current best KV overlap candidate.

        This is query-only: it does not mutate router state because ``request_id`` is
        intentionally omitted.
        """

        return await self.kv_router.best_worker(
            token_ids=self.token_ids_from_request(request),
            router_config_override=request.get("router_config_override"),
            block_mm_infos=request.get("block_mm_infos"),
            lora_name=self.lora_name_from_request(request),
        )

    async def inspect_selection(self, request: RequestLike) -> JsonDict:
        """Return the routing decision without generating tokens."""

        request_id = self.request_id_from_request(request)
        selection = await self.select_worker(request, request_id)
        return {
            "request_id": request_id,
            "worker_id": selection.worker_id,
            "dp_rank": selection.dp_rank,
            **selection.metadata,
        }

    async def generate(self, request: RequestLike) -> AsyncIterator[JsonDict]:
        """Select a worker and forward the request through ``KvRouter.generate()``."""

        request_id = self.request_id_from_request(request)
        selection = await self.select_worker(request, request_id)

        async for output in await self.kv_router.generate(
            token_ids=self.token_ids_from_request(request),
            model=str(request.get("model", "unknown")),
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            router_config_override=request.get("router_config_override"),
            worker_id=selection.worker_id,
            dp_rank=selection.dp_rank,
            extra_args=request.get("extra_args"),
            block_mm_infos=request.get("block_mm_infos"),
            multi_modal_data=request.get("multi_modal_data"),
            mm_routing_info=request.get("mm_routing_info"),
        ):
            yield output

    @staticmethod
    def request_id_from_request(request: RequestLike) -> str:
        request_id = request.get("request_id")
        if request_id is None:
            return str(uuid.uuid4())
        return str(request_id)

    @staticmethod
    def token_ids_from_request(request: RequestLike) -> list[int]:
        token_ids = request.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("request must contain a non-empty token_ids list")
        return [int(token) for token in token_ids]

    @staticmethod
    def lora_name_from_request(request: RequestLike) -> str | None:
        lora_name = request.get("lora_name")
        if lora_name is None:
            return None
        return str(lora_name)

    @staticmethod
    def extract_agent_id(request: RequestLike) -> str | None:
        """Best-effort extraction of a sticky routing identity.

        Preferred fields:
        - ``agent_id``
        - ``prefix_id``
        - annotation entries like ``agent_id:<value>`` or ``prefix_id:<value>``
        """

        for key in ("agent_id", "prefix_id"):
            value = request.get(key)
            if value is not None:
                return str(value)

        annotations = request.get("annotations")
        if not isinstance(annotations, list):
            return None

        for annotation in annotations:
            if not isinstance(annotation, str):
                continue
            for prefix in ("agent_id:", "prefix_id:"):
                if annotation.startswith(prefix):
                    return annotation[len(prefix) :]
        return None
