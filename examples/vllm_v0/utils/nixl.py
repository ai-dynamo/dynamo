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

import logging
import os
from contextlib import contextmanager

import msgspec
from vllm.distributed.device_communicators.nixl import NixlMetadata

from dynamo.runtime import DistributedRuntime

METADATA_DIR = "/tmp/nixl"

logger = logging.getLogger(__name__)


@contextmanager
def temp_metadata_file(engine_id, metadata: NixlMetadata):
    os.makedirs(METADATA_DIR, exist_ok=True)
    path = f"{METADATA_DIR}/{engine_id}.nixl_meta"
    with open(path, "wb") as f:
        encoded = msgspec.msgpack.encode(metadata)
        logger.info(f"Size of encoded metadata: {len(encoded)}")
        f.write(encoded)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def find_remote_metadata(engine_id):
    # find and load metadata from METADATA_DIR that do not match engine_id
    remote_metadata = []
    for file in os.listdir(METADATA_DIR):
        if file.endswith(".nixl_meta"):
            if file.split(".")[0] != engine_id:
                with open(os.path.join(METADATA_DIR, file), "rb") as f:
                    remote_metadata.append(
                        msgspec.msgpack.decode(f.read(), type=NixlMetadata)
                    )
    return remote_metadata


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
        # create with primary lease so that the kv entry will be deleted when the worker shutdowns
        try:
            # TODO: should we create a series of function in etcd client to use primary lease?
            await self._client.kv_create_or_validate(
                key, serialized_metadata, self._client.primary_lease_id()
            )
        except Exception as e:
            logger.warning(f"A different metadata exists for engine {engine_id}: {e}")
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
            raise Exception("Error retrieving metadata for engine {engine_id}") from e

        return deserialized_metadata
