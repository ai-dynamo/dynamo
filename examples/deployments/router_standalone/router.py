# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo._core import RadixTree, ZmqKvEventListener

logger = logging.getLogger(__name__)


class RouterRequest(BaseModel):
    local_hashes: List[int]
    num_tokens: int


class RouterResponse(BaseModel):
    worker_id: int


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
                    for event in kv_events:
                        event = json.loads(event)
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

    async def get_best_worker(self, local_hashes: list[int], num_tokens: int) -> int:
        try:
            if num_tokens <= 0:
                raise ValueError("num_tokens must be positive")

            # local_hashes can be empty
            raw_scores = self.radix_tree.find_matches(local_hashes).scores

            overlap_scores = {
                worker_id: raw_scores.get(worker_id, 0) * self.block_size / num_tokens
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

            return best_worker_id

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
                worker_id = await self.router.get_best_worker(
                    request.local_hashes, request.num_tokens
                )
                return RouterResponse(worker_id=worker_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error finding best worker: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

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
        "--block-size", type=int, default=64, help="Block size for caching"
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
