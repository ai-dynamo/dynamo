# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker capacity snapshot via the FPM event-plane subscriber.

The scheduler treats ``kv_pool_tokens`` (block_size * total_kv_blocks from
each worker's published MDC) as a static pool size and uses it as the
pause-trigger denominator. We attach to ``FpmEventSubscriber`` purely for
its model-card stream; the per-iteration FPM payloads are not consumed.

If engine-runtime load fields (active_decode_tokens, num_running, etc) are
ever wired into the pause trigger, the FPM payload decode path will need to
be reintroduced. See DESIGN.md section 7 (open work).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerKey:
    """Composite key matching FPM ``(worker_id, dp_rank)``."""

    worker_id: int
    dp_rank: int


@dataclass
class WorkerCapacity:
    """Snapshot of one worker's capacity.

    ``kv_pool_tokens`` is the KV cache pool capacity in tokens
    (``block_size * total_kv_blocks`` from the worker's MDC). This is the
    pause-relevant denominator: pause when sum-of-program-tokens approaches
    the pool limit, so the engine doesn't preempt decode requests.

    We currently don't read engine-runtime load fields (active decode tokens,
    queued prefill, etc) -- the pause math is driven entirely by the
    scheduler's own program-table sum vs this static pool size. See DESIGN.md
    section 2.1 for why polled engine load isn't used here (mirrors upstream).
    """

    worker_id: int
    dp_rank: int
    kv_pool_tokens: int

    @property
    def capacity_tokens(self) -> int:
        """Pause-relevant capacity (KV pool size)."""
        return self.kv_pool_tokens


def _parse_worker_id(raw: str) -> Optional[int]:
    """FPM key worker_id is the runtime ``connection_id`` -- numeric."""
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _kv_pool_tokens_from_card(card_json: str) -> int:
    """Compute KV pool size in tokens from the MDC.

    ``kv_pool_tokens = kv_cache_block_size × runtime_config.total_kv_blocks``.
    Returns 0 if either field is missing so the caller can fall back to
    the batch budget rather than crash.
    """
    try:
        card = json.loads(card_json)
    except (TypeError, json.JSONDecodeError):
        return 0
    block_size = card.get("kv_cache_block_size")
    rc = card.get("runtime_config") or {}
    total_blocks = rc.get("total_kv_blocks")
    if (
        isinstance(block_size, (int, float))
        and block_size > 0
        and isinstance(total_blocks, (int, float))
        and total_blocks > 0
    ):
        return int(block_size) * int(total_blocks)
    return 0


class FpmCapacityProvider:
    """Engine-true capacity snapshot, sourced from the FPM event plane.

    Construction is cheap. ``start()`` spawns the FPM subscriber background
    tasks. ``snapshot()`` is lock-free; readers can call it from any context.
    ``stop()`` is idempotent and shuts the subscriber down cleanly.
    """

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
            logger.info(
                "FpmCapacityProvider: started tracking forward-pass-metrics"
            )

    def stop(self) -> None:
        with self._lock:
            if self._stopped or self._subscriber is None:
                return
            try:
                self._subscriber.shutdown()
            except Exception as exc:
                logger.warning("FpmCapacityProvider shutdown error: %s", exc)
            self._stopped = True

    def snapshot(self) -> dict[WorkerKey, WorkerCapacity]:
        """Return the latest per-worker capacity snapshot.

        Empty dict if the subscriber has not seen any FPM messages or model
        cards yet (e.g. cold start). Callers should treat that as "no opinion"
        and not pause.

        Only ``kv_pool_tokens`` (from the worker's MDC) is currently consumed
        by the scheduler. The FPM event-stream payload is subscribed-to but
        not decoded -- engine-runtime load fields would be a deliberate
        deviation from upstream and require pause-trigger logic changes; see
        DESIGN.md section 7 (open work).
        """
        if self._subscriber is None:
            return {}

        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("FpmCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[WorkerKey, WorkerCapacity] = {}
        for worker_id_str, card_json in cards.items():
            worker_id = _parse_worker_id(worker_id_str)
            if worker_id is None:
                continue
            kv_pool_tokens = _kv_pool_tokens_from_card(card_json)
            if kv_pool_tokens <= 0:
                continue
            out[WorkerKey(worker_id=worker_id, dp_rank=0)] = WorkerCapacity(
                worker_id=worker_id,
                dp_rank=0,
                kv_pool_tokens=kv_pool_tokens,
            )
        return out
