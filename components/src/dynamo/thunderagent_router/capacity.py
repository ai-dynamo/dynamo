# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker capacity snapshot derived from worker model deployment cards.

``capacity_tokens`` (``block_size * total_kv_blocks``) is published once per
worker via its MDC. We piggyback on ``FpmEventSubscriber`` only as the
existing Python channel that already tracks per-worker MDCs; the
forward-pass-metric payloads themselves are not consumed.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


class WorkerCapacityProvider:
    """Maps ``worker_id -> kv_pool_tokens`` from each worker's MDC."""

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
            pool_tokens = self._parse_pool_tokens(card_json)
            if pool_tokens is not None:
                out[worker_id] = pool_tokens
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
        self._parsed[card_json] = result
        return result
