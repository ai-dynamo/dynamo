# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KV Cache Indexer for the Thompson Sampling router.

Maintains an independent RadixTree fed by worker KV events to provide clean
per-request overlap scores without cumulative active-sequence noise.

Data flow::

    Worker (vLLM/SGLang)
      |  publishes KvCacheEvent via ZMQ or NATS
      v
    RadixTree.apply_event(worker_id, event_bytes)
      |  updates local radix tree with per-worker block state
      v
    compute_block_hash_for_seq(tokens, block_size)
      |  hashes request tokens into block-level hashes
      v
    RadixTree.find_matches(block_hashes) -> OverlapScores
      |  returns {(worker_id, dp_rank): matching_blocks}
      v
    KvIndexer.find_matches_for_request() -> per-worker overlap fractions
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_HAS_DYNAMO_KV = False
try:
    from dynamo.llm import RadixTree as _RadixTree

    try:
        from dynamo.llm import compute_block_hash_for_seq as _compute_block_hash_for_seq
    except ImportError:
        from dynamo.llm import compute_block_hash_for_seq_py as _compute_block_hash_for_seq

    _HAS_DYNAMO_KV = True
    logger.info("kv_indexer: dynamo.llm KV primitives imported successfully")
except ImportError as exc:
    logger.warning(
        "kv_indexer: dynamo.llm KV primitives not available (%s); "
        "KvIndexer will return empty overlap scores",
        exc,
    )

try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

try:
    import zmq
    import zmq.asyncio

    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False


@dataclass
class OverlapResult:
    """Per-request overlap scores from the KvIndexer.

    Attributes:
        scores: worker_id -> fraction of request blocks cached [0, 1]
        raw_block_counts: worker_id -> absolute matching block count
        total_blocks: total blocks in the request's token sequence
        tree_sizes: worker_id -> total blocks in worker's radix tree (memory pressure)
    """

    scores: dict[int, float] = field(default_factory=dict)
    raw_block_counts: dict[int, int] = field(default_factory=dict)
    total_blocks: int = 0
    tree_sizes: dict[int, int] = field(default_factory=dict)


class KvIndexer:
    """Maintains an independent RadixTree for clean per-request overlap queries.

    Uses ZMQ subscriptions to workers' KV event streams (same as the NAT
    KvIndexer) and optionally falls back to predict-from-routing-decisions
    mode when ZMQ events are not available.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._radix_tree: Any | None = None
        self._tree_lock = threading.Lock()
        self._event_counter: int = 0
        self._drain_task: asyncio.Task | None = None

        # vLLM ZMQ subscribers
        self._zmq_sockets: dict[int, Any] = {}
        self._zmq_ctx: Any | None = None
        self._events_applied: int = 0

        if _HAS_DYNAMO_KV:
            self._radix_tree = _RadixTree()
            logger.info(
                "KvIndexer initialized with RadixTree (block_size=%d)", block_size
            )
        else:
            logger.warning(
                "KvIndexer: no RadixTree available; overlap scores will be empty"
            )

    def add_worker(self, worker_id: int, zmq_endpoint: str) -> None:
        """Subscribe to a worker's ZMQ KV event stream."""
        if not _HAS_DYNAMO_KV or not _HAS_ZMQ or not _HAS_MSGPACK:
            return
        if worker_id in self._zmq_sockets:
            return
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.asyncio.Context()
        sock = self._zmq_ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.connect(zmq_endpoint)
        self._zmq_sockets[worker_id] = sock
        logger.info("KvIndexer: subscribed to worker %s at %s", worker_id, zmq_endpoint)

    def discover_workers(
        self,
        instance_ids: list[int],
        kv_event_base_port: int | None = None,
        worker_endpoints: dict[int, str] | None = None,
    ) -> None:
        """Subscribe to workers' KV event streams.

        If ``worker_endpoints`` is provided it takes precedence and is treated
        as an authoritative ``{worker_id: "tcp://host:port"}`` mapping. This
        is the cross-host path (e.g. k8s, where each worker pod binds the
        same port on its own pod IP).

        Otherwise the single-host fallback runs: each worker in sorted
        instance-id order is reached at ``tcp://127.0.0.1:{base + idx}``.
        This matches the local multi-worker layout where every worker
        shares a host and needs a distinct port offset.
        """
        if worker_endpoints:
            for wid, endpoint in worker_endpoints.items():
                self.add_worker(wid, endpoint)
            return

        if kv_event_base_port is None:
            kv_event_base_port = int(os.environ.get("KV_EVENT_BASE_PORT", "20080"))

        for idx, wid in enumerate(sorted(instance_ids)):
            endpoint = f"tcp://127.0.0.1:{kv_event_base_port + idx}"
            self.add_worker(wid, endpoint)

    def start_drain(self, interval: float = 0.1) -> None:
        """Start background asyncio task to drain ZMQ events into the radix tree."""
        if self._drain_task is not None:
            return
        self._drain_task = asyncio.create_task(self._drain_loop(interval))
        logger.info(
            "KvIndexer: started background drain (interval=%.2fs, workers=%d)",
            interval,
            len(self._zmq_sockets),
        )

    async def _drain_loop(self, interval: float) -> None:
        while True:
            try:
                await self._drain_events()
            except Exception:
                logger.exception("KvIndexer: error draining events")
            await asyncio.sleep(interval)

    async def _drain_events(self) -> int:
        if not _HAS_DYNAMO_KV or self._radix_tree is None:
            return 0

        total = 0
        for worker_id, sock in self._zmq_sockets.items():
            while True:
                try:
                    parts = await sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception:
                    break

                if len(parts) < 3:
                    continue
                try:
                    batch = msgpack.unpackb(parts[2], raw=False, strict_map_key=False)
                except Exception:
                    continue

                events = (
                    batch[1]
                    if isinstance(batch, (list, tuple)) and len(batch) >= 3
                    else []
                )
                if not isinstance(events, list):
                    continue

                for evt in events:
                    total += self._apply_event(worker_id, evt)

        self._events_applied += total
        return total

    def _apply_event(self, worker_id: int, evt: Any) -> int:
        if not isinstance(evt, (list, tuple)) or not evt:
            return 0

        evt_type = str(evt[0]).lower()
        self._event_counter += 1
        eid = self._event_counter

        with self._tree_lock:
            try:
                if "stored" in evt_type:
                    hashes = (
                        evt[1] if len(evt) > 1 and isinstance(evt[1], list) else []
                    )
                    if not hashes:
                        return 0
                    event = {
                        "event_id": eid,
                        "data": {
                            "stored": {
                                "blocks": [
                                    {"block_hash": h, "tokens_hash": h} for h in hashes
                                ]
                            }
                        },
                    }
                    self._radix_tree.apply_event(
                        worker_id, json.dumps(event).encode("utf-8")
                    )
                    return 1

                elif "removed" in evt_type:
                    hashes = (
                        evt[1] if len(evt) > 1 and isinstance(evt[1], list) else []
                    )
                    if not hashes:
                        return 0
                    event = {
                        "event_id": eid,
                        "data": {"removed": {"block_hashes": hashes}},
                    }
                    self._radix_tree.apply_event(
                        worker_id, json.dumps(event).encode("utf-8")
                    )
                    return 1

                elif "cleared" in evt_type:
                    self._radix_tree.clear_all_blocks(worker_id)
                    return 1

            except Exception:
                logger.debug(
                    "KvIndexer: failed to apply event type=%s for worker %s",
                    evt_type,
                    worker_id,
                )
        return 0

    async def find_matches_for_request(
        self, tokens: list[int]
    ) -> OverlapResult:
        """Compute per-worker overlap for a token sequence.

        Returns clean per-request overlap (no cumulative active-sequence noise).
        """
        if not _HAS_DYNAMO_KV or self._radix_tree is None:
            return OverlapResult()

        if self._drain_task is None:
            await self._drain_events()

        block_hashes = _compute_block_hash_for_seq(tokens, self.block_size)
        if not block_hashes:
            return OverlapResult()

        total_blocks = len(block_hashes)

        with self._tree_lock:
            raw_scores = self._radix_tree.find_matches(block_hashes)

        scores: dict[int, float] = {}
        raw_counts: dict[int, int] = {}
        for key, count in raw_scores.scores.items():
            wid = int(key[0]) if isinstance(key, tuple) else int(key)
            frac = float(count) / float(total_blocks)
            if frac > scores.get(wid, 0.0):
                scores[wid] = frac
                raw_counts[wid] = int(count)

        worker_tree_sizes: dict[int, int] = {}
        if hasattr(raw_scores, "tree_sizes"):
            for key, size in raw_scores.tree_sizes.items():
                wid = int(key[0]) if isinstance(key, tuple) else int(key)
                worker_tree_sizes[wid] = max(
                    worker_tree_sizes.get(wid, 0), int(size)
                )

        return OverlapResult(
            scores=scores,
            raw_block_counts=raw_counts,
            total_blocks=total_blocks,
            tree_sizes=worker_tree_sizes,
        )

    def record_routing_decision(self, worker_id: int, tokens: list[int]) -> None:
        """Predict-from-decision: synthesize a stored event for routed tokens.

        Keeps the radix tree approximately up-to-date even when ZMQ events
        are delayed or unavailable.
        """
        if not _HAS_DYNAMO_KV or self._radix_tree is None:
            return

        block_hashes = _compute_block_hash_for_seq(tokens, self.block_size)
        if not block_hashes:
            return

        self._event_counter += 1
        event = {
            "event_id": self._event_counter,
            "data": {
                "stored": {
                    "blocks": [
                        {"block_hash": h, "tokens_hash": h} for h in block_hashes
                    ]
                }
            },
        }

        with self._tree_lock:
            try:
                self._radix_tree.apply_event(
                    worker_id, json.dumps(event).encode("utf-8")
                )
            except Exception:
                logger.debug(
                    "record_routing_decision: apply_event failed for worker %s",
                    worker_id,
                )

    @property
    def events_applied(self) -> int:
        return self._events_applied

    def shutdown(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None
        for sock in self._zmq_sockets.values():
            sock.close()
        self._zmq_sockets.clear()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        logger.info(
            "KvIndexer: shutdown (events_applied=%d)", self._events_applied
        )
