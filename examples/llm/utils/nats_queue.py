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

from nats.aio.client import Client as NATS
from nats.errors import Error as NatsError
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError


class NATSQueue:
    _instance: ClassVar[Optional["NATSQueue"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(
        self,
        stream_name: str = "default",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        self.nats_url = nats_server
        self._nc: Optional[NATS] = None
        self._js: Optional[JetStreamContext] = None
        # TODO: check if this is needed
        # Sanitize stream_name to remove path separators
        self._stream_name = stream_name.replace("/", "_").replace("\\", "_")
        self._subject = f"{self._stream_name}.*"
        self.dequeue_timeout = dequeue_timeout
        self._subscriber: Optional[JetStreamContext.PullSubscription] = None

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
        """Establish connection and create stream if needed"""
        try:
            if self._nc is None:
                self._nc = NATS()
                await self._nc.connect(self.nats_url)
                self._js = self._nc.jetstream()
                # Check if stream exists, if not create it
                try:
                    await self._js.stream_info(self._stream_name)
                except NotFoundError:
                    await self._js.add_stream(
                        name=self._stream_name, subjects=[self._subject]
                    )
                # Create persistent subscriber
                self._subscriber = await self._js.pull_subscribe(
                    f"{self._stream_name}.queue", durable="worker-group"
                )
        except NatsError as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to NATS: {e}")

    async def ensure_connection(self):
        """Ensure we have an active connection"""
        if self._nc is None or self._nc.is_closed:
            await self.connect()

    async def close(self):
        """Close the connection when done"""
        if self._nc:
            await self._nc.close()
            self._nc = None
            self._js = None
            self._subscriber = None

    # TODO: is enqueue/dequeue_object a better name for a general queue?
    async def enqueue_task(self, task_data: bytes) -> None:
        """
        Enqueue a task using msgspec-encoded data
        """
        await self.ensure_connection()
        try:
            await self._js.publish(f"{self._stream_name}.queue", task_data)  # type: ignore
        except NatsError as e:
            raise RuntimeError(f"Failed to enqueue task: {e}")

    async def dequeue_task(self) -> Optional[bytes]:
        """Dequeue and return a task as raw bytes, to be decoded with msgspec"""
        await self.ensure_connection()
        try:
            msgs = await self._subscriber.fetch(1, timeout=self.dequeue_timeout)  # type: ignore
            if msgs:
                msg = msgs[0]
                await msg.ack()
                return msg.data
            return None
        except asyncio.TimeoutError:
            return None
        except NatsError as e:
            raise RuntimeError(f"Failed to dequeue task: {e}")

    async def get_queue_size(self) -> int:
        """Get the number of messages currently in the queue"""
        await self.ensure_connection()
        try:
            # Get consumer info to get pending messages count
            consumer_info = await self._js.consumer_info(  # type: ignore
                self._stream_name, "worker-group"
            )
            # Return number of pending messages (real-time queue size)
            return consumer_info.num_pending
        except NatsError as e:
            raise RuntimeError(f"Failed to get queue size: {e}")
