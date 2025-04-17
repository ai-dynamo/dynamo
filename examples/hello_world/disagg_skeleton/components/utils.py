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
import logging
from contextlib import asynccontextmanager
from typing import ClassVar, Optional

import msgspec
from nats.aio.client import Client as NATS
from nats.errors import Error as NatsError
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError
from pydantic import BaseModel
from vllm.distributed.device_communicators.nixl import NixlMetadata

from dynamo._core import Client
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class GeneralRequest(BaseModel):
    prompt: str = "user input"
    request_id: str = "id_string"


class GeneralResponse(BaseModel):
    worker_output: str = "generated output"
    request_id: str = "id_string"


class RemotePrefillRequest(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str = "Engine ID"
    request_id: str = "id_string"


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


class PrefillQueue(NATSQueue):
    """
    A wrapper of NATSQueue for PrefillRequest.
    The stream name is forced to be "prefill_queue".
    """

    def __init__(
        self,
        stream_name="prefill_queue",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        super().__init__(
            stream_name=stream_name,
            nats_server=nats_server,
            dequeue_timeout=dequeue_timeout,
        )

    async def enqueue_prefill_request(
        self, prefill_request: RemotePrefillRequest
    ) -> None:
        encoded_request = msgspec.json.encode(prefill_request)
        await self.enqueue_task(encoded_request)

    async def dequeue_prefill_request(self) -> Optional[RemotePrefillRequest]:
        encoded_request = await self.dequeue_task()
        if encoded_request is not None:
            prefill_request = msgspec.json.decode(
                encoded_request, type=RemotePrefillRequest
            )
            return prefill_request
        else:
            return None


class NixlMetadataStore:
    NIXL_METADATA_KEY = "nixl_metadata"

    def __init__(self, namespace: str, runtime: DistributedRuntime) -> None:
        self._namespace = namespace

        # TODO Remove metadata from etcd on delete
        self._stored: set[str] = set()

        self._cached: dict[str, NixlMetadata] = {}
        self._client = runtime.etcd_client()
        if self._client is None:
            raise Exception("Cannot be used with static workers")
        self._key_prefix = f"{self._namespace}/{NixlMetadataStore.NIXL_METADATA_KEY}"

    async def put(self, engine_id, metadata: NixlMetadata):
        serialized_metadata = msgspec.msgpack.encode(metadata)
        key = "/".join([self._key_prefix, engine_id])
        await self._client.kv_put(key, serialized_metadata, None)
        self._stored.add(engine_id)

    async def get(self, engine_id) -> NixlMetadata:
        try:
            if engine_id in self._cached:
                return self._cached[engine_id]

            key = "/".join([self._key_prefix, engine_id])
            key_values = await self._client.kv_get_prefix(key)
            deserialized_metadata = None

            for item in key_values:
                deserialized_metadata = msgspec.msgpack.decode(
                    item["value"], type=NixlMetadata
                )
                break

            if deserialized_metadata is None:
                raise Exception("metadata not found in etcd")

            self._cached[engine_id] = deserialized_metadata

            # TODO watch for changes and update cache

            # self._client.add_watch_callback(
            #     key,
            #     self._watch_callback,
            # )

        except Exception as e:
            raise Exception(f"Error retrieving metadata for engine {engine_id}") from e

        return deserialized_metadata


async def check_required_workers(
    workers_client: Client,
    required_workers: int,
    on_change=True,
    poll_interval=5,
    tag="",
):
    """Wait until the minimum number of workers are ready."""
    worker_ids = workers_client.endpoint_ids()
    num_workers = len(worker_ids)
    new_count = -1  # Force to print "waiting for worker" once
    while num_workers < required_workers:
        if (not on_change) or new_count != num_workers:
            num_workers = new_count if new_count >= 0 else num_workers
            print(
                f" {tag} Waiting for more workers to be ready.\n"
                f" Current: {num_workers},"
                f" Required: {required_workers}"
            )
        await asyncio.sleep(poll_interval)
        worker_ids = workers_client.endpoint_ids()
        new_count = len(worker_ids)

    print(f"Workers ready: {worker_ids}")
    return worker_ids
