# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker retention budget derived from worker model deployment cards.

The device KV pool is always published as ``block_size * total_kv_blocks``.
When SGLang HiCache is enabled, its exact host-pool capacity is published in
runtime metadata and added to the program-retention budget. SGLang still owns
device admission, eviction, and restore; this value only prevents the program
scheduler from pausing before native GPU-to-host spill can happen.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)

_SGLANG_HICACHE_CAPACITY_RUNTIME_KEY = "sglang_hicache_capacity"


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
                hicache = (
                    runtime_data.get(_SGLANG_HICACHE_CAPACITY_RUNTIME_KEY, {})
                    if isinstance(runtime_data, dict)
                    else {}
                )
                host_tokens = (
                    hicache.get("host_total_tokens")
                    if isinstance(hicache, dict)
                    else None
                )
                if isinstance(host_tokens, (int, float)) and host_tokens > 0:
                    result += int(host_tokens)
        self._parsed[card_json] = result
        return result
