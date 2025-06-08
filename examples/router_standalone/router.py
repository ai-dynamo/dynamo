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


import asyncio
import json
import logging
from typing import Any

import numpy as np
import zmq

from dynamo._core import RadixTree, ZmqKvEventListener

logger = logging.getLogger(__name__)


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

        self._load_locks: list[asyncio.Lock] = [
            asyncio.Lock() for _ in range(num_workers)
        ]
        self._radix_lock = asyncio.Lock()

        self.kv_usages = [0.0] * num_workers
        self.waitings = [0] * num_workers

        self.context = zmq.Context()  # Store context as instance variable
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

        logger.info("Router initialized")

    async def periodic_update_load(self):
        async def update_load(worker_id: int):
            while True:
                try:
                    metrics: dict[str, Any] = self.load_listeners[worker_id].recv_json(
                        zmq.NOBLOCK
                    )
                    async with self._load_locks[worker_id]:
                        self.kv_usages[worker_id] = metrics["gpu_cache_usage"]
                        self.waitings[worker_id] = int(metrics["num_waiting_reqs"])
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving metrics: {e}")

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
                        event: Any = json.loads(event)
                        async with self._radix_lock:
                            self.radix_tree.apply_event(
                                worker_id, json.dumps(event).encode("utf-8")
                            )
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving metrics: {e}")

                await asyncio.sleep(0.1)

        for worker_id in range(self.num_workers):
            asyncio.create_task(update_tree(worker_id))

    async def get_best_worker(self, local_hashes: list[int], num_tokens: int) -> int:
        # Run tokenization in a separate thread to avoid blocking the event loop
        async with self._radix_lock:
            raw_scores = self.radix_tree.find_matches(local_hashes).scores
        overlap_scores = {
            worker_id: raw_scores.get(worker_id, 0) * self.block_size / num_tokens
            for worker_id in range(self.num_workers)
        }

        kv_usages = self.kv_usages[:]
        waitings = self.waitings[:]

        max_waiting = max(waitings)
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

        logits = np.array(logits)
        best_worker_id = int(np.random.choice(np.flatnonzero(logits == logits.max())))

        # this is a predictive update which will be reset as new metrics are polled
        # but it is helpful for handling short bursts of highly concurrent requests
        # we omit updating the gpu_usage_perc as done in the rusty router for simplicity
        # as this requires obtaining num_gpu_blocks from the engines and can be intrusive
        async with self._load_locks[best_worker_id]:
            self.waitings[best_worker_id] += 1

        return best_worker_id

    def shutdown(self):
        """Shutdown ZMQ listeners and context"""
        logger.info("Shutting down KvRouter...")

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
