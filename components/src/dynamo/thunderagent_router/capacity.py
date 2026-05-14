# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker capacity snapshot via the FPM event-plane subscriber.

The scheduler treats ``capacity_tokens`` (``block_size * total_kv_blocks``
from each worker's published MDC) as a static pool size and uses it as the
pause-trigger denominator. We attach to ``FpmEventSubscriber`` purely for
its model-card stream; the per-iteration FPM payloads are not consumed
in v0.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


class FpmCapacityProvider:
    """Engine-true capacity snapshot, sourced from the FPM event plane."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._subscriber: Optional[FpmEventSubscriber] = None
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._subscriber = FpmEventSubscriber(self._endpoint)
            self._subscriber.start_tracking()
            self._started = True
            logger.info("FpmCapacityProvider: started tracking forward-pass-metrics")

    def stop(self) -> None:
        with self._lock:
            if self._stopped or self._subscriber is None:
                return
            try:
                self._subscriber.shutdown()
            except Exception as exc:
                logger.warning("FpmCapacityProvider shutdown error: %s", exc)
            self._stopped = True

    def snapshot(self) -> dict[int, int]:
        """worker_id -> capacity_tokens. Empty on cold start."""
        if self._subscriber is None:
            return {}

        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("FpmCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[int, int] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
                card = json.loads(card_json)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            block_size = card.get("kv_cache_block_size")
            total_blocks = (card.get("runtime_config") or {}).get("total_kv_blocks")
            if (
                isinstance(block_size, (int, float))
                and block_size > 0
                and isinstance(total_blocks, (int, float))
                and total_blocks > 0
            ):
                out[worker_id] = int(block_size) * int(total_blocks)
        return out
