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
from contextlib import asynccontextmanager
from typing import ClassVar, Optional

from dynamo._core import NatsQueue


class NATSQueue:
    _instance: ClassVar[Optional["NATSQueue"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(
        self,
        stream_name: str = "default",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        self.nats_q = NatsQueue(stream_name, nats_server, dequeue_timeout)

    @classmethod
    @asynccontextmanager
    async def get_instance(
        cls,
        *,
        stream_name: str = "default",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        """Get or create a singleton instance of NATSq"""
        # TODO: check if this _lock is needed with GIL
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(
                    stream_name=stream_name,
                    nats_server=nats_server,
                    dequeue_timeout=dequeue_timeout,
                )
                await cls._instance.connect()
            try:
                yield cls._instance
            except Exception:
                if cls._instance:
                    await cls._instance.close()
                cls._instance = None
                raise

    # TODO: check to see if this can be replaced by something like get_instance().close()
    @classmethod
    async def shutdown(cls):
        """Explicitly close the singleton instance if it exists"""
        async with cls._lock:
            if cls._instance:
                await cls._instance.close()
                cls._instance = None

    async def connect(self):
        await self.nats_q.connect()

    async def ensure_connection(self):
        await self.nats_q.ensure_connection()

    async def close(self):
        await self.nats_q.close()

    # TODO: is enqueue/dequeue_object a better name for a general queue?
    async def enqueue_task(self, task_data: bytes) -> None:
        await self.nats_q.enqueue_task(task_data)

    async def dequeue_task(self, timeout: Optional[float] = None) -> Optional[bytes]:
        return await self.nats_q.dequeue_task(timeout)

    async def get_queue_size(self) -> int:
        return await self.nats_q.get_queue_size()

    async def clear_queue(self) -> int:
        try:
            cleared_count = 0
            # Continue until we can't dequeue any more messages
            while True:
                # use a small timeout
                message = await self.dequeue_task(timeout=0.1)
                if message is None:
                    break
                cleared_count += 1
            return cleared_count
        except Exception as e:
            raise RuntimeError(f"Failed to clear queue: {e}")
