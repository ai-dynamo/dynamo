# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-replica admission budget derived from worker model deployment cards.

``block_size * total_kv_blocks`` is per DP rank, so the budget keys on
``(worker_id, dp_rank)``. The backend still owns admission, spill, restore and eviction.

The budget is GPU-resident KV capacity plus a *capped* credit for the worker's native
HiCache/host-offload capacity (``get_native_offloading_capacity_tokens``). Two bugs
motivate the cap:

1. #11185 added the two uncapped, on the theory that a 0.95 pause threshold against
   GPU+host means "95% of the memory that can retain the program locally." That assumes
   programs pushed past GPU residency get credited back out of ``used`` once SGLang
   spills them to host. Nothing does that: ``_ReplicaUsage.used`` (router.py) only ever
   tracks live ``program.token_total`` for programs the scheduler still considers
   admitted -- there's no signal here for how many of those tokens are actually
   GPU-resident versus already spilled to HiCache. So the numerator never grows with host
   capacity the way the uncapped denominator did, and pause/soft-demote/resume became
   correspondingly harder to trigger the larger ``hicache-ratio`` is. Confirmed live on a
   `hicache-ratio: 8` DP-attention deployment: zero pause events from c8 through c160
   concurrency while the router's own client logs showed 532 backend_connection_timeout
   warnings and climbing InvalidInferenceResultError counts at the high end
   (NVIDIA/InferenceMAX#271).
2. Dropping the credit to zero isn't right either: before the SGLang sidecar correctly
   published this field for DeepSeek V4 (#14313), the router only ever saw GPU-only
   capacity, and that empirically over-paused (300+ programs paused throughout a c64 run,
   worse throughput than an unthrottled baseline -- see the recipe history in
   NVIDIA/InferenceMAX#271). GPU-only is too little credit; GPU+full-host is too much.

Capping the credited host capacity at the GPU pool's own size (so the combined budget is
at most 2x GPU-only, independent of how large ``hicache-ratio`` is configured) is a
starting point between those two known-bad extremes, not a validated final constant --
until real GPU-vs-host residency is tracked, this is the best available proxy and needs
empirical re-validation across concurrencies.

Separately, the budget also carries ``max_programs``: SGLang's own ``max_running_requests``
(divided per DP rank), published into the card as ``runtime_config.max_num_seqs``
(``dynamo/sglang/register.py``) but never read here before now. Token-sum admission alone
is blind to the actual per-replica concurrent-request ceiling the engine enforces, which
matters independently of KV memory -- a replica can be nowhere near its token budget and
still not be able to schedule another concurrent request. Gating on both dimensions ties
admission to a real, engine-enforced signal rather than a token proxy alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from dynamo.common.native_offloading import get_native_offloading_capacity_tokens
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import Client, Endpoint
from dynamo.thunderagent_router.program_state import ReplicaKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplicaCapacity:
    """One replica's admission budget: a token budget plus an optional request-count
    ceiling (SGLang's ``max_running_requests``, per DP rank). ``max_programs=None`` means
    the card didn't publish one -- callers must treat that as unlimited, not zero."""

    tokens: int
    max_programs: Optional[int] = None


@dataclass(frozen=True)
class _Card:
    """Everything ``snapshot()`` needs from one card, parsed once.

    One record rather than a cache per field: ``snapshot()`` runs under the scheduler lock,
    on every tick and on every status scrape.
    """

    pool_tokens: Optional[int]
    max_programs: Optional[int]
    start_rank: int
    dp_size: int


class WorkerCapacityProvider:
    """Tracks live workers and their MDC program-retention capacity."""

    def __init__(self, endpoint: Endpoint, client: Client) -> None:
        self._endpoint = endpoint
        self._client = client
        self._subscriber: Optional[FpmEventSubscriber] = None
        # Keyed on the raw JSON body, which never changes for a given worker, so a
        # repeat snapshot() costs a dict lookup instead of a json.loads.
        self._parsed: dict[str, _Card] = {}

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

    def snapshot(self) -> dict[ReplicaKey, ReplicaCapacity]:
        """Per-replica admission budget, keyed by ``(worker_id, dp_rank)``.

        ``total_kv_blocks``/``max_num_seqs`` are both per rank, so a worker owning ``D``
        ranks yields ``D`` entries of each -- filed under the key it describes, not rescaled.
        """
        if self._subscriber is None:
            return {}
        try:
            cards = self._subscriber.get_model_cards()
        except Exception as exc:
            logger.debug("WorkerCapacityProvider snapshot error: %s", exc)
            return {}

        out: dict[ReplicaKey, ReplicaCapacity] = {}
        for worker_id_str, card_json in cards.items():
            try:
                worker_id = int(worker_id_str)
            except (ValueError, TypeError):
                continue
            card = self._parse_card(card_json)
            if card.pool_tokens is None:
                continue
            capacity = ReplicaCapacity(
                tokens=card.pool_tokens, max_programs=card.max_programs
            )
            for dp_rank in range(card.start_rank, card.start_rank + card.dp_size):
                out[(worker_id, dp_rank)] = capacity
        return out

    def live_worker_ids(self) -> set[int]:
        """Return workers currently registered for the generate endpoint.

        Worker-granular by design: liveness is a property of the instance.
        """
        try:
            return set(self._client.instance_ids())
        except Exception as exc:
            logger.debug("WorkerCapacityProvider liveness snapshot error: %s", exc)
            return set()

    def _parse_card(self, card_json: str) -> _Card:
        """Parse one card body, memoised on the body itself."""
        cached = self._parsed.get(card_json)
        if cached is not None:
            return cached
        card = self._build_card(card_json)
        self._parsed[card_json] = card
        return card

    def _build_card(self, card_json: str) -> _Card:
        try:
            body = json.loads(card_json)
        except json.JSONDecodeError:
            body = None
        if not isinstance(body, dict):
            return _Card(pool_tokens=None, max_programs=None, start_rank=0, dp_size=1)
        runtime_config = body.get("runtime_config") or {}
        return _Card(
            pool_tokens=self._pool_tokens(body, runtime_config),
            max_programs=self._max_programs(runtime_config),
            start_rank=self._start_rank(runtime_config),
            dp_size=self._dp_size(runtime_config),
        )

    @staticmethod
    def _pool_tokens(body: dict, runtime_config: dict) -> Optional[int]:
        block_size = body.get("kv_cache_block_size")
        total_blocks = runtime_config.get("total_kv_blocks")
        if not (
            isinstance(block_size, (int, float))
            and block_size > 0
            and isinstance(total_blocks, (int, float))
            and total_blocks > 0
        ):
            return None
        tokens = int(block_size) * int(total_blocks)
        offloaded = get_native_offloading_capacity_tokens(
            runtime_config.get("runtime_data", {})
        )
        if offloaded is None:
            return tokens
        # Cap the host-offload credit at the GPU pool's own size (module docstring):
        # admission budget is at most 2x GPU-only, regardless of hicache-ratio.
        return tokens + min(offloaded, tokens)

    @staticmethod
    def _max_programs(runtime_config: dict) -> Optional[int]:
        """SGLang's ``max_running_requests``, already divided per DP rank by the publisher
        (``dynamo/sglang/register.py``/``capacity.py::per_rank_max_running_requests``).
        ``None`` when the card doesn't say -- callers must treat that as unlimited."""
        declared = runtime_config.get("max_num_seqs")
        return declared if isinstance(declared, int) and declared > 0 else None

    @staticmethod
    def _start_rank(runtime_config: dict) -> int:
        """First global DP rank this worker owns.

        Not always 0, so the offset is load-bearing: vLLM publishes ``dp_range[0]`` here and
        rejects ranks outside ``[start, start + size)``.
        """
        declared = runtime_config.get("data_parallel_start_rank")
        return declared if isinstance(declared, int) and declared >= 0 else 0

    @staticmethod
    def _dp_size(runtime_config: dict) -> int:
        """Number of DP ranks this worker owns; 1 when the card does not say."""
        declared = runtime_config.get("data_parallel_size")
        return declared if isinstance(declared, int) and declared > 0 else 1
