# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker retention budget derived from worker model deployment cards.

Backends may publish an authoritative token capacity through common runtime
metadata. Legacy cards fall back to ``block_size * total_kv_blocks``. Native
offloading capacity is added to either device-capacity source. The backend
remains responsible for admission, spill, restore, and eviction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from dynamo.common.kv_cache_capacity import get_kv_cache_capacity_tokens
from dynamo.common.native_offloading import get_native_offloading_capacity_tokens
from dynamo.common.token_capacity import positive_int
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerCapacity:
    """A worker's retention budget and physical allocation granularity."""

    retention_tokens: int
    block_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("retention_tokens", self.retention_tokens),
            ("block_size", self.block_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class WorkerCapacityProvider:
    """Maps worker IDs to retention capacity from each worker's MDC."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._subscriber: Optional[FpmEventSubscriber] = None
        # Cache parsed cards keyed on the raw JSON string so a subsequent
        # snapshot() call avoids re-parsing on the request hot path.
        self._parsed: dict[str, Optional[WorkerCapacity]] = {}

    def start(self) -> None:
        if self._subscriber is not None:
            return
        self._subscriber = FpmEventSubscriber(self._endpoint)
        self._subscriber.start_tracking()
        logger.info("WorkerCapacityProvider: subscribed to MDC stream")

    def stop(self) -> None:
        if self._subscriber is None:
            return
        try:
            self._subscriber.shutdown()
        except Exception as exc:
            logger.warning("WorkerCapacityProvider shutdown error: %s", exc)
        self._subscriber = None

    def snapshot(self) -> dict[int, WorkerCapacity]:
        if self._subscriber is None:
            return {}
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[int, WorkerCapacity] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            capacity = self._parse_capacity(card_json)
            if capacity is not None:
                out[worker_id] = capacity
        return out

    def _parse_capacity(self, card_json: str) -> Optional[WorkerCapacity]:
        if card_json in self._parsed:
            return self._parsed[card_json]
        result: Optional[WorkerCapacity] = None
        try:
            card = json.loads(card_json)
        except (json.JSONDecodeError, TypeError):
            card = None
        if isinstance(card, dict):
            block_size = positive_int(card.get("kv_cache_block_size"))
            runtime_config = card.get("runtime_config") or {}
            if block_size is not None and isinstance(runtime_config, dict):
                runtime_data = runtime_config.get("runtime_data", {})
                retention_tokens = get_kv_cache_capacity_tokens(runtime_data)
                if retention_tokens is None:
                    total_blocks = positive_int(runtime_config.get("total_kv_blocks"))
                    if total_blocks is not None:
                        retention_tokens = block_size * total_blocks
                offloaded_tokens = get_native_offloading_capacity_tokens(runtime_data)
                if retention_tokens is not None:
                    if offloaded_tokens is not None:
                        retention_tokens += offloaded_tokens
                    result = WorkerCapacity(
                        retention_tokens=retention_tokens,
                        block_size=block_size,
                    )
        self._parsed[card_json] = result
        return result
