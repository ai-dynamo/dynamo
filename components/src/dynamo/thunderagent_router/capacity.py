# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker retention budget derived from worker model deployment cards.

The device KV pool is always ``block_size * total_kv_blocks``. Backends may
publish additional native offloading capacity through common runtime metadata.
The backend remains responsible for admission, spill, restore, and eviction.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from dynamo.common.native_offloading import get_native_offloading_capacity_tokens
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


class WorkerCapacityProvider:
    """Maps ``worker_id -> program-retention tokens`` from each worker's MDC."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._subscriber: Optional[FpmEventSubscriber] = None
        # Cache parsed cards keyed on the raw JSON string so a subsequent
        # snapshot() call avoids re-parsing on the request hot path.
        self._parsed: dict[str, Optional[int]] = {}

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

    def snapshot(self) -> dict[int, int]:
        if self._subscriber is None:
            return {}
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[int, int] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            retention_tokens = self._parse_pool_tokens(card_json)
            if retention_tokens is not None:
                out[worker_id] = retention_tokens
        return out

    def _parse_pool_tokens(self, card_json: str) -> Optional[int]:
        if card_json in self._parsed:
            return self._parsed[card_json]
        result: Optional[int] = None
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            card = None
        if isinstance(card, dict):
            block_size = card.get("kv_cache_block_size")
            total_blocks = (card.get("runtime_config") or {}).get("total_kv_blocks")
            if (
                isinstance(block_size, (int, float))
                and block_size > 0
                and isinstance(total_blocks, (int, float))
                and total_blocks > 0
            ):
                result = int(block_size) * int(total_blocks)
                runtime_data = (card.get("runtime_config") or {}).get(
                    "runtime_data", {}
                )
                offloaded_tokens = get_native_offloading_capacity_tokens(runtime_data)
                if offloaded_tokens is not None:
                    result += offloaded_tokens
        self._parsed[card_json] = result
        return result
