# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo._core import RadixTree, ZmqKvEventListener

logger = logging.getLogger(__name__)

# Debug flag: set DYNAMO_DEBUG=1 to enable debug file dumps
DEBUG_ENABLED = os.environ.get("DYNAMO_DEBUG", "0") == "1"
DEBUG_KV_EVENT_FILE = "/tmp/debug_kv_events.txt"


def dump_kv_event(worker_id: int, event: dict):
    """Dump KV event to file for debugging."""
    if not DEBUG_ENABLED:
        return
    import datetime

    with open(DEBUG_KV_EVENT_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Worker ID: {worker_id}\n")
        f.write(f"Event: {json.dumps(event, indent=2)}\n")
        f.write(f"{'='*60}\n")


class RouterRequest(BaseModel):
    local_hashes: List[int]
    num_tokens: int


class RouterResponse(BaseModel):
    worker_id: int
    overlap: float = 0.0  # Overlap ratio for the selected worker
    matched_blocks: int = 0  # Number of matched blocks


class InjectEventRequest(BaseModel):
    """Request to inject a KV event directly into the RadixTree for testing."""
    worker_id: int
    tokens_hash: int
    block_hash: int | None = None  # If None, use tokens_hash
    mm_extra_info: dict | None = None


class LoadMetrics(BaseModel):
    kv_cache_usage: float
    num_waiting_reqs: int


def setup_zmq_subscriber(context: zmq.Context, endpoint: str) -> zmq.Socket[bytes]:
    socket = context.socket(zmq.SUB)
    socket.connect(endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
    socket.setsockopt(zmq.RCVTIMEO, 1)  # 1ms timeout (very short)
    return socket


class KvRouter:
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

        self.kv_usages = [0.0] * num_workers
        self.waitings = [0] * num_workers

        self.context = zmq.Context()
        self.load_listeners = [
            setup_zmq_subscriber(
                self.context, f"tcp://localhost:{base_metrics_port + worker_id}"
            )
            for worker_id in range(num_workers)
        ]
        self.kv_listeners = [
            ZmqKvEventListener(
                f"tcp://localhost:{base_kv_events_port + worker_id}", "", block_size
            )
            for worker_id in range(num_workers)
        ]

        self.background_tasks: list[asyncio.Task] = []
        logger.info("Router initialized")

    async def start_background_tasks(self):
        """Start background tasks for load and indexer updates"""
        logger.info("Starting router background tasks...")
        self.background_tasks.append(asyncio.create_task(self.periodic_update_load()))
        self.background_tasks.append(
            asyncio.create_task(self.periodic_update_indexer())
        )

    async def periodic_update_load(self):
        async def update_load(worker_id: int):
            while True:
                try:
                    metrics_dict = self.load_listeners[worker_id].recv_json(zmq.NOBLOCK)
                    metrics = LoadMetrics.model_validate(metrics_dict)
                    self.kv_usages[worker_id] = metrics.kv_cache_usage
                    self.waitings[worker_id] = metrics.num_waiting_reqs
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.warning(
                        f"Error receiving metrics for worker {worker_id}: {e}"
                    )

                await asyncio.sleep(0.1)

        for worker_id in range(self.num_workers):
            asyncio.create_task(update_load(worker_id))

    async def periodic_update_indexer(self):
        async def update_tree(worker_id: int):
            while True:
                try:
                    kv_events: list[str] = await self.kv_listeners[
                        worker_id
                    ].get_events()
                    for event_str in kv_events:
                        event = json.loads(event_str)
                        # Dump event to file for debugging
                        dump_kv_event(worker_id, event)
                        # Log the event data to debug hash issues
                        if "data" in event and "stored" in event.get("data", {}):
                            stored_data = event["data"]["stored"]
                            blocks = stored_data.get("blocks", [])
                            for blk in blocks:
                                logger.info(
                                    f"Router: Worker {worker_id} storing block: "
                                    f"tokens_hash={blk.get('tokens_hash')}, "
                                    f"block_hash={blk.get('block_hash')}"
                                )
                        self.radix_tree.apply_event(
                            worker_id, json.dumps(event).encode("utf-8")
                        )
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.warning(
                        f"Error receiving KV events for worker {worker_id}: {e}"
                    )

                await asyncio.sleep(0.1)

        for worker_id in range(self.num_workers):
            asyncio.create_task(update_tree(worker_id))

    async def get_best_worker(
        self, local_hashes: list[int], num_tokens: int
    ) -> tuple[int, float, int]:
        """
        Find best worker for request.

        Returns:
            tuple of (worker_id, overlap_ratio, matched_blocks)
        """
        try:
            if num_tokens <= 0:
                raise ValueError("num_tokens must be positive")

            # local_hashes can be empty
            logger.info(f"Router: find_matches called with local_hashes={local_hashes}")
            result = self.radix_tree.find_matches(local_hashes)
            raw_scores = result.scores
            logger.info(f"Router: find_matches returned raw_scores={raw_scores}")

            # raw_scores is keyed by (worker_id, dp_rank) tuples, not just worker_id
            # For now, assume dp_rank=0 for all workers
            matched_blocks_per_worker = {
                worker_id: raw_scores.get((worker_id, 0), 0)
                for worker_id in range(self.num_workers)
            }
            overlap_scores = {
                worker_id: matched_blocks_per_worker[worker_id] * self.block_size / num_tokens
                for worker_id in range(self.num_workers)
            }

            kv_usages = self.kv_usages[:]
            waitings = self.waitings[:]

            max_waiting = max(waitings) if waitings else 0
            waitings_normalized = [
                waiting / max_waiting if max_waiting else 0.0 for waiting in waitings
            ]

            logits = []
            for worker_id in range(self.num_workers):
                overlap = overlap_scores[worker_id]
                usage = kv_usages[worker_id]
                waiting = waitings_normalized[worker_id]
                logit = 2 * overlap - usage - waiting
                logits.append(logit)
                logger.info(
                    f"worker_id: {worker_id}, logit = 2 * {overlap:.3f} - {usage:.3f} - {waiting:.3f} = {logit:.3f}"
                )

            logits_array = np.array(logits)
            best_worker_id = int(
                np.random.choice(np.flatnonzero(logits_array == logits_array.max()))
            )

            # this is a predictive update which will be reset as new metrics are polled
            # but it is helpful for handling short bursts of highly concurrent requests
            # we omit updating the gpu_usage_perc as done in the rusty router for simplicity
            # as this requires obtaining num_gpu_blocks from the engines and can be intrusive
            # no need for async lock here, as the state is intended to be continuously overwritten
            self.waitings[best_worker_id] += 1

            best_overlap = overlap_scores[best_worker_id]
            best_matched = matched_blocks_per_worker[best_worker_id]
            return best_worker_id, best_overlap, best_matched

        except Exception as e:
            logger.error(f"Error in get_best_worker: {e}")
            raise

    async def shutdown(self):
        """Shutdown ZMQ listeners, context, and background tasks"""
        logger.info("Shutting down KvRouter...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Close load listeners (ZMQ sockets)
        for listener in self.load_listeners:
            try:
                listener.close()
            except Exception as e:
                logger.error(f"Error closing load listener: {e}")

        # Terminate ZMQ context
        try:
            self.context.term()
            logger.info("ZMQ context terminated successfully")
        except Exception as e:
            logger.error(f"Error terminating ZMQ context: {e}")

        logger.info("KvRouter shutdown completed")


class RouterAPI:
    def __init__(
        self,
        block_size: int = 64,
        num_workers: int = 4,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        port: int = 7000,
    ):
        self.port = port
        self.block_size = block_size
        self.num_workers = num_workers
        self.base_kv_events_port = base_kv_events_port
        self.base_metrics_port = base_metrics_port
        self.router = None
        self.app = FastAPI(
            title="KV Router API", version="0.0.1", lifespan=self.lifespan
        )
        self.setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Startup
        self.router = KvRouter(
            block_size=self.block_size,
            num_workers=self.num_workers,
            base_kv_events_port=self.base_kv_events_port,
            base_metrics_port=self.base_metrics_port,
        )
        await self.router.start_background_tasks()
        logger.info("Router API started successfully")

        yield

        # Shutdown
        if self.router:
            await self.router.shutdown()

    def setup_routes(self):
        @self.app.post("/find_best_worker", response_model=RouterResponse)
        async def find_best_worker(request: RouterRequest):
            if self.router is None:
                raise HTTPException(status_code=503, detail="Router not initialized")

            try:
                worker_id, overlap, matched_blocks = await self.router.get_best_worker(
                    request.local_hashes, request.num_tokens
                )
                return RouterResponse(
                    worker_id=worker_id,
                    overlap=overlap,
                    matched_blocks=matched_blocks,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error finding best worker: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.get("/debug/tree_info")
        async def get_tree_info():
            if self.router is None:
                raise HTTPException(status_code=503, detail="Router not initialized")

            events = self.router.radix_tree.dump_tree_as_events()
            return {
                "num_blocks": len(events),
                "events": events[:20],  # Show first 20 events to avoid huge response
            }

        @self.app.post("/debug/inject_event")
        async def inject_event(request: InjectEventRequest):
            """Inject a KV event directly into RadixTree for testing."""
            if self.router is None:
                raise HTTPException(status_code=503, detail="Router not initialized")

            block_hash = request.block_hash if request.block_hash else request.tokens_hash
            event = {
                "event_id": 99999,  # Test event ID
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [{
                            "block_hash": block_hash,
                            "tokens_hash": request.tokens_hash,
                            "mm_extra_info": request.mm_extra_info,
                        }]
                    }
                }
            }
            self.router.radix_tree.apply_event(
                request.worker_id, json.dumps(event).encode("utf-8")
            )
            return {"status": "ok", "tokens_hash": request.tokens_hash, "worker_id": request.worker_id}

    async def start(self):
        """Start the router API server"""
        logger.info(f"Starting Router API server on port {self.port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    parser = argparse.ArgumentParser(description="KV Router API Server")

    parser.add_argument(
        "--block-size", type=int, default=32, help="Block size for caching (TensorRT-LLM uses 32)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--base-kv-events-port", type=int, default=5557, help="Base port for KV events"
    )
    parser.add_argument(
        "--base-metrics-port", type=int, default=5657, help="Base port for metrics"
    )
    parser.add_argument(
        "--port", type=int, default=7000, help="Port to serve the Router API on"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    api = RouterAPI(
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        port=args.port,
    )

    async def run_with_shutdown():
        try:
            await api.start()
        except KeyboardInterrupt:
            logger.info(
                "Received KeyboardInterrupt, shutting down Router API server..."
            )
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")

    try:
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        logger.info("Force shutdown via KeyboardInterrupt.")


if __name__ == "__main__":
    main()
