# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from dynamo._core import RadixTree, ZmqKvEventListener

logger = logging.getLogger(__name__)

DEBUG_ENABLED = os.environ.get("DYNAMO_DEBUG", "0") == "1"


def dump_kv_event(worker_id: int, event: dict):
    """Dump KV event to file for debugging (only when DYNAMO_DEBUG=1)."""
    if not DEBUG_ENABLED:
        return
    import datetime

    with open("/tmp/debug_kv_events.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Worker ID: {worker_id}\n")
        f.write(f"Event: {json.dumps(event, indent=2)}\n")


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class RouterRequest(BaseModel):
    local_hashes: list[int]
    num_tokens: int


class RouterResponse(BaseModel):
    worker_id: int
    overlap: float = 0.0
    matched_blocks: int = 0


class InjectEventRequest(BaseModel):
    """For testing: inject a KV event directly into RadixTree."""

    worker_id: int
    tokens_hash: int
    block_hash: int | None = None
    mm_extra_info: dict | None = None


class LoadMetrics(BaseModel):
    kv_cache_usage: float
    num_waiting_reqs: int


# -----------------------------------------------------------------------------
# ZMQ Helpers
# -----------------------------------------------------------------------------


def create_zmq_subscriber(context: zmq.Context, endpoint: str) -> zmq.Socket[bytes]:
    """Create a ZMQ SUB socket with standard settings."""
    socket = context.socket(zmq.SUB)
    socket.connect(endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 1)
    return socket


# -----------------------------------------------------------------------------
# KvRouter Core
# -----------------------------------------------------------------------------


class KvRouter:
    """Router that uses RadixTree for KV cache-aware worker selection."""

    def __init__(
        self,
        block_size: int = 64,
        num_workers: int = 4,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
    ):
        self.num_workers = num_workers
        self.block_size = block_size
        self.radix_tree = RadixTree()

        # Per-worker metrics
        self.kv_usages = [0.0] * num_workers
        self.waitings = [0] * num_workers

        # ZMQ setup
        self.context = zmq.Context()
        self.load_listeners = [
            create_zmq_subscriber(
                self.context, f"tcp://localhost:{base_metrics_port + i}"
            )
            for i in range(num_workers)
        ]
        self.kv_listeners = [
            ZmqKvEventListener(
                f"tcp://localhost:{base_kv_events_port + i}", "", block_size
            )
            for i in range(num_workers)
        ]

        self.background_tasks: list[asyncio.Task] = []
        logger.info("Router initialized")

    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------

    async def start_background_tasks(self):
        """Start background tasks for load and tree updates."""
        logger.info("Starting router background tasks...")
        for worker_id in range(self.num_workers):
            self.background_tasks.append(
                asyncio.create_task(self._poll_worker_load(worker_id))
            )
            self.background_tasks.append(
                asyncio.create_task(self._poll_worker_kv_events(worker_id))
            )

    async def _poll_worker_load(self, worker_id: int):
        """Poll load metrics for a single worker."""
        while True:
            try:
                data = self.load_listeners[worker_id].recv_json(zmq.NOBLOCK)
                metrics = LoadMetrics.model_validate(data)
                self.kv_usages[worker_id] = metrics.kv_cache_usage
                self.waitings[worker_id] = metrics.num_waiting_reqs
            except zmq.Again:
                pass
            except (zmq.ZMQError, ValidationError) as e:
                logger.warning(f"Worker {worker_id} metrics error: {e}")
            except Exception:
                logger.exception(f"Worker {worker_id} unexpected metrics error")
            await asyncio.sleep(0.1)

    async def _poll_worker_kv_events(self, worker_id: int):
        """Poll KV events for a single worker and update RadixTree."""
        while True:
            try:
                events: list[str] = await self.kv_listeners[worker_id].get_events()
                for event_str in events:
                    event = json.loads(event_str)
                    dump_kv_event(worker_id, event)
                    self.radix_tree.apply_event(
                        worker_id, json.dumps(event).encode("utf-8")
                    )
            except zmq.Again:
                pass
            except (zmq.ZMQError, json.JSONDecodeError) as e:
                logger.warning(f"Worker {worker_id} KV events error: {e}")
            except Exception:
                logger.exception(f"Worker {worker_id} unexpected KV events error")
            await asyncio.sleep(0.1)

    # -------------------------------------------------------------------------
    # Worker Selection
    # -------------------------------------------------------------------------

    async def get_best_worker(
        self, local_hashes: list[int], num_tokens: int
    ) -> tuple[int, float, int]:
        """
        Find best worker for request.

        Returns: (worker_id, overlap_ratio, matched_blocks)
        """
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")

        # Get cache matches from RadixTree
        matched_blocks = self._get_matched_blocks(local_hashes)

        # Compute overlap scores
        overlap_scores = {
            wid: matched_blocks[wid] * self.block_size / num_tokens
            for wid in range(self.num_workers)
        }

        # Compute routing logits
        logits = self._compute_logits(overlap_scores)

        # Select best worker (random tie-breaking)
        best_id = self._select_best_worker(logits)

        # Predictive update for burst handling
        self.waitings[best_id] += 1

        return best_id, overlap_scores[best_id], matched_blocks[best_id]

    def _get_matched_blocks(self, local_hashes: list[int]) -> dict[int, int]:
        """Get matched block count per worker from RadixTree."""
        result = self.radix_tree.find_matches(local_hashes)
        raw_scores = result.scores
        logger.info(f"Router: raw_scores={raw_scores}")

        # raw_scores is keyed by (worker_id, dp_rank); assume dp_rank=0
        return {wid: raw_scores.get((wid, 0), 0) for wid in range(self.num_workers)}

    def _compute_logits(self, overlap_scores: dict[int, float]) -> list[float]:
        """Compute routing logits for each worker."""
        max_waiting = max(self.waitings) if self.waitings else 0

        logits = []
        for wid in range(self.num_workers):
            overlap = overlap_scores[wid]
            usage = self.kv_usages[wid]
            waiting_norm = self.waitings[wid] / max_waiting if max_waiting else 0.0
            logit = 2 * overlap - usage - waiting_norm
            logits.append(logit)
            logger.info(
                f"worker_id: {wid}, logit = 2 * {overlap:.3f} - {usage:.3f} - {waiting_norm:.3f} = {logit:.3f}"
            )
        return logits

    def _select_best_worker(self, logits: list[float]) -> int:
        """Select worker with highest logit (random tie-breaking)."""
        arr = np.array(logits)
        return int(np.random.choice(np.flatnonzero(arr == arr.max())))

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(self):
        """Shutdown ZMQ listeners and background tasks."""
        logger.info("Shutting down KvRouter...")

        for task in self.background_tasks:
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        for listener in self.load_listeners:
            listener.close()

        self.context.term()
        logger.info("KvRouter shutdown completed")


# -----------------------------------------------------------------------------
# Router API Server
# -----------------------------------------------------------------------------


class RouterAPI:
    """FastAPI wrapper for KvRouter."""

    def __init__(
        self,
        block_size: int = 64,
        num_workers: int = 4,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        port: int = 7000,
    ):
        self.port = port
        self.router_config = {
            "block_size": block_size,
            "num_workers": num_workers,
            "base_kv_events_port": base_kv_events_port,
            "base_metrics_port": base_metrics_port,
        }
        self.router: KvRouter | None = None
        self.app = FastAPI(
            title="KV Router API", version="0.0.1", lifespan=self.lifespan
        )
        self._setup_routes()

    def _require_router(self) -> KvRouter:
        """Get router or raise 503 if not initialized."""
        if self.router is None:
            raise HTTPException(status_code=503, detail="Router not initialized")
        return self.router

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        self.router = KvRouter(**self.router_config)
        await self.router.start_background_tasks()
        logger.info("Router API started")
        yield
        if self.router:
            await self.router.shutdown()

    def _setup_routes(self):
        @self.app.post("/find_best_worker", response_model=RouterResponse)
        async def find_best_worker(request: RouterRequest):
            router = self._require_router()
            try:
                wid, overlap, matched = await router.get_best_worker(
                    request.local_hashes, request.num_tokens
                )
                return RouterResponse(
                    worker_id=wid, overlap=overlap, matched_blocks=matched
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/debug/tree_info")
        async def get_tree_info():
            router = self._require_router()
            events = router.radix_tree.dump_tree_as_events()
            return {"num_blocks": len(events), "events": events[:20]}

        @self.app.post("/debug/inject_event")
        async def inject_event(request: InjectEventRequest):
            router = self._require_router()
            block_hash = request.block_hash or request.tokens_hash
            event = {
                "event_id": 99999,
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [
                            {
                                "block_hash": block_hash,
                                "tokens_hash": request.tokens_hash,
                                "mm_extra_info": request.mm_extra_info,
                            }
                        ],
                    }
                },
            }
            router.radix_tree.apply_event(
                request.worker_id, json.dumps(event).encode("utf-8")
            )
            return {
                "status": "ok",
                "tokens_hash": request.tokens_hash,
                "worker_id": request.worker_id,
            }

    async def start(self):
        """Start the router API server."""
        logger.info(f"Starting Router API on port {self.port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )
        await uvicorn.Server(config).serve()


def main():
    parser = argparse.ArgumentParser(description="KV Router API Server")
    parser.add_argument(
        "--block-size", type=int, default=32, help="Block size (default: 32)"
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers")
    parser.add_argument(
        "--base-kv-events-port", type=int, default=5557, help="Base KV events port"
    )
    parser.add_argument(
        "--base-metrics-port", type=int, default=5657, help="Base metrics port"
    )
    parser.add_argument("--port", type=int, default=7000, help="Router API port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    api = RouterAPI(
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        port=args.port,
    )

    asyncio.run(api.start())


if __name__ == "__main__":
    main()
