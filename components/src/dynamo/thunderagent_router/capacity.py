# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker retention budget derived from worker model deployment cards.

The device KV pool is ``block_size * total_kv_blocks``. Backends may publish
additional native offloading capacity through common runtime metadata. The
backend remains responsible for admission, spill, restore, and eviction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from dynamo.common.native_offloading import get_native_offloading_capacity_tokens
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


def _positive_int(value: object) -> int | None:
    """Return a positive integer without coercing worker metadata."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


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
        # Keep only the current card for each worker. This avoids repeat JSON
        # parsing without retaining every historical card body.
        self._parsed: dict[int, tuple[str, Optional[WorkerCapacity]]] = {}

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
        current: dict[int, tuple[str, Optional[WorkerCapacity]]] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            cached = self._parsed.get(worker_id)
            if cached is not None and cached[0] == card_json:
                capacity = cached[1]
            else:
                try:
                    capacity = self._parse_capacity(card_json)
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.debug(
                        "WorkerCapacityProvider invalid card for worker %s: %s",
                        worker_id,
                        exc,
                    )
                    capacity = None
            current[worker_id] = (card_json, capacity)
            if capacity is not None:
                out[worker_id] = capacity
        self._parsed = current
        return out

    def _parse_capacity(self, card_json: str) -> Optional[WorkerCapacity]:
        try:
            card = json.loads(card_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(card, dict):
            return None

        block_size = _positive_int(card.get("kv_cache_block_size"))
        runtime_config = card.get("runtime_config")
        if block_size is None or not isinstance(runtime_config, dict):
            return None

        total_blocks = _positive_int(runtime_config.get("total_kv_blocks"))
        if total_blocks is None:
            return None

        retention_tokens = block_size * total_blocks
        offloaded_tokens = get_native_offloading_capacity_tokens(
            runtime_config.get("runtime_data", {})
        )
        if offloaded_tokens is not None:
            retention_tokens += offloaded_tokens
        return WorkerCapacity(
            retention_tokens=retention_tokens,
            block_size=block_size,
        )
