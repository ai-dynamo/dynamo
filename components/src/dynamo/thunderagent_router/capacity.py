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
from dynamo.runtime import Client, Endpoint
from dynamo.thunderagent_router.program_state import ReplicaKey

logger = logging.getLogger(__name__)


class WorkerCapacityProvider:
    """Tracks live workers and their MDC program-retention capacity."""

    def __init__(self, endpoint: Endpoint, client: Client) -> None:
        self._endpoint = endpoint
        self._client = client
        self._subscriber: Optional[FpmEventSubscriber] = None
        # Cache parsed cards keyed on the raw JSON string so a subsequent
        # snapshot() call avoids re-parsing on the request hot path.
        self._parsed: dict[str, Optional[int]] = {}
        # Same card, different field; separate cache to avoid evicting pool-token
        # entries when dp_size_for_worker is called and vice versa.
        self._parsed_dp_size: dict[str, Optional[int]] = {}

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

    def snapshot(self) -> dict[ReplicaKey, int]:
        """Program-retention budget in tokens, keyed by ``(worker_id, dp_rank)``.

        The card's ``total_kv_blocks`` is per rank, so a worker owning ``D`` ranks
        yields ``D`` entries with the same value -- filed under the key it describes,
        not rescaled.
        """
        if self._subscriber is None:
            return {}
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[ReplicaKey, int] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            retention_tokens = self._parse_pool_tokens(card_json)
            if retention_tokens is None:
                continue
            start_rank, dp_size = self._parse_dp_range(card_json)
            for dp_rank in range(start_rank, start_rank + dp_size):
                out[(worker_id, dp_rank)] = retention_tokens
        return out

    def _parse_dp_range(self, card_json: str) -> tuple[int, int]:
        """``(data_parallel_start_rank, data_parallel_size)``, or ``(0, 1)`` if absent.

        Ranks are global: the MDC advertises ``[0, dp_size)`` and only the leader node
        registers, so no per-node offset is needed.
        """
        dp_size = self._parse_dp_size(card_json) or 1
        start_rank = 0
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            card = None
        if isinstance(card, dict):
            declared = (card.get("runtime_config") or {}).get(
                "data_parallel_start_rank"
            )
            if isinstance(declared, int) and declared >= 0:
                start_rank = declared
        return start_rank, dp_size

    def live_worker_ids(self) -> set[int]:
        """Return workers currently registered for the generate endpoint.

        Worker-granular by design: liveness is a property of the instance.
        """
        try:
            return set(self._client.instance_ids())
        except Exception as exc:
            logger.debug("WorkerCapacityProvider liveness snapshot error: %s", exc)
            return set()

    def dp_size_for_worker(self, worker_id: int) -> Optional[int]:
        """Return the number of DP ranks owned by *worker_id*; None if unknown.

        A worker with exactly 1 rank can be pinned by worker id alone — the
        router will fill in the rank via ``unique_dp_rank_for_worker``. Any
        other value requires an explicit rank in the pin, so callers must treat
        an unknown result as "more than 1", not as "1".
        """
        if self._subscriber is None:
            return None
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider dp_size lookup error: %s", exc)
            return None
        card_json = cards.get(str(worker_id))
        if card_json is None:
            return None
        return self._parse_dp_size(card_json)

    def _parse_dp_size(self, card_json: str) -> Optional[int]:
        if card_json in self._parsed_dp_size:
            return self._parsed_dp_size[card_json]
        result: Optional[int] = None
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            card = None
        if isinstance(card, dict):
            dp_size = (card.get("runtime_config") or {}).get("data_parallel_size")
            if isinstance(dp_size, int) and dp_size > 0:
                result = dp_size
        self._parsed_dp_size[card_json] = result
        return result

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
