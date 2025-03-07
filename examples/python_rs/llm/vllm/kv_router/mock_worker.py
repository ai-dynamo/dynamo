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
import ctypes
from ctypes import c_char_p, c_int64, c_uint32

import uvloop
from common.protocol import Request, Response
from vllm.logger import logger as vllm_logger

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker


class DynamoResult:
    OK = 0
    ERR = 1


class MockEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, metrics_publisher, worker_id):
        self.worker_id = worker_id
        # KV events
        self.lib = ctypes.CDLL("/opt/dynamo/llm_binding/lib/libdynamo.llm_capi.so")
        self.lib.dynamo.llm_init.argtypes = [c_char_p, c_char_p, c_int64]
        self.lib.dynamo.llm_init.restype = c_uint32
        result = self.lib.dynamo.llm_init("dynamo".encode(), "vllm".encode(), worker_id)
        if result == DynamoResult.OK:
            vllm_logger.info(
                "KVCacheEventManager initialized successfully. Ready to publish KV Cache Events"
            )
        else:
            vllm_logger.info("KVCacheEventManager initialization failed!")
        self.lib.dynamo_kv_event_publish_stored.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint32),  # token_ids
            ctypes.POINTER(ctypes.c_size_t),  # num_block_tokens
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
            ctypes.POINTER(ctypes.c_uint64),  # parent_hash
            ctypes.c_uint64,  # lora_id
        ]
        self.lib.dynamo_kv_event_publish_stored.restype = (
            ctypes.c_uint32
        )  # dynamo.llm_result_t

        self.lib.dynamo_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
        ]
        self.lib.dynamo_kv_event_publish_removed.restype = (
            ctypes.c_uint32
        )  # dynamo.llm_result_t

        # KV metrics
        self.metrics_publisher = metrics_publisher

        self.request_active_slots = 0
        self.request_total_slots = 4
        self.kv_active_block = 0
        self.kv_total_blocks = 4
        # [NOTE] Now that the component must has proper metrics reported
        # to be properly selected by the router
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_block,
            self.kv_total_blocks,
        )
        self.event_id_counter = 0
        self.tokens = [3] * 64

    @dynamo_endpoint(Request, Response)
    async def generate(self, request):
        print(f"Received request: {request}")
        self.request_active_slots = min(
            self.request_active_slots + 1, self.request_total_slots
        )
        self.kv_active_block = min(self.kv_active_block + 1, self.kv_total_blocks)
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_block,
            self.kv_total_blocks,
        )
        self.store_event()
        yield "Hello, World!"

    def store_event(self):
        parent_hash = (
            (ctypes.c_uint64 * 1)(self.event_id_counter)
            if self.event_id_counter > 0
            else None
        )
        result = self.lib.dynamo_kv_event_publish_stored(
            self.event_id_counter,  # uint64_t event_id
            (ctypes.c_uint32 * len(self.tokens))(
                *self.tokens
            ),  # const uint32_t *token_ids
            (ctypes.c_size_t * 1)(
                len(self.tokens)
            ),  # const uintptr_t *num_block_tokens
            (ctypes.c_uint64 * 1)(self.event_id_counter),  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
            parent_hash,  # const uint64_t *parent_hash
            0,  # uint64_t lora_id
        )
        self.event_id_counter += 1

        if result == DynamoResult.OK:
            vllm_logger.debug(f"Store - Published KV Event: {self.event_id_counter}")
        else:
            vllm_logger.debug(
                f"Store - Failed to Publish KV Event: {self.event_id_counter}"
            )

    async def cooldown(self):
        while True:
            await asyncio.sleep(5)
            self.request_active_slots = max(0, self.request_active_slots - 1)
            self.kv_active_block = max(0, self.kv_active_block - 1)
            self.metrics_publisher.publish(
                self.request_active_slots,
                self.request_total_slots,
                self.kv_active_block,
                self.kv_total_blocks,
            )


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("vllm")
    metrics_publisher = KvMetricsPublisher()
    await metrics_publisher.create_service(component)

    endpoint = component.endpoint("generate")
    engine = MockEngine(metrics_publisher, endpoint.lease_id())
    await asyncio.gather(
        engine.cooldown(),
        endpoint.serve_endpoint(engine.generate),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
