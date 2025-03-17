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
import os
import subprocess
from ctypes import c_char_p, c_int64, c_uint32
from time import sleep
from typing import List

import pytest

from dynamo.llm import (
    KvEventPublisher,
    KvIndexer,
    KvMetricsAggregator,
    KvMetricsPublisher,
)
from dynamo.runtime import Component, DistributedRuntime

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    sleep(5)  # wait for nats-server and etcd to start
    yield

    # Teardown code
    print("Tearing down resources")
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


@pytest.fixture(scope="module")
async def distributed_runtime():
    loop = asyncio.get_running_loop()
    return DistributedRuntime(loop)


# TODO Figure out how to test with different kv_block_size
# Right now I get an error in EventPublisher init when I run this test
# back to back. It occurs when calling dynamo_llm_init and I think is related to the
# OnceCell initializations not being reset.
# The test works individually if I run it with 32, then 11, then 64.
# @pytest.mark.parametrize("kv_block_size", [11, 32, 64])
async def test_event_handler(distributed_runtime):
    kv_block_size = 32
    namespace = "kv_test"
    component = "event"
    kv_listener = distributed_runtime.namespace(namespace).component(component)
    await kv_listener.create_service()

    # publisher
    worker_id = 233
    event_publisher = EventPublisher(kv_listener, worker_id, kv_block_size)

    # indexer
    indexer = KvIndexer(kv_listener, kv_block_size)

    test_token = [3] * kv_block_size
    lora_id = 0  # lora_id is not used in the indexer
    scores = await indexer.find_matches_for_request(test_token, lora_id)
    assert not scores.scores

    event_publisher.store_event(test_token, lora_id)
    # wait for the event to be processed as it is sent asynchronously
    await asyncio.sleep(1)
    scores = await indexer.find_matches_for_request(test_token, lora_id)
    assert scores.scores
    assert worker_id in scores.scores
    assert scores.scores[worker_id] == 1

    # remove event
    event_publisher.remove_event()
    await asyncio.sleep(1)
    scores = await indexer.find_matches_for_request(test_token, lora_id)
    assert not scores.scores


class EventPublisher:
    def __init__(self, component: Component, worker_id: int, kv_block_size: int):
        self.publisher = KvEventPublisher(component, worker_id, kv_block_size)
        self.event_id_counter = 0
        self.block_hashes: List[int] = []

    def store_event(self, tokens, lora_id):
        parent_hash = self.event_id_counter if self.event_id_counter > 0 else None
        self.publisher.publish_stored(
            self.event_id_counter,  # event_id
            tokens,  # token_ids
            [
                len(tokens),
            ],  # num_block_tokens
            [
                self.event_id_counter,
            ],  # block_hashes
            lora_id,  # lora_id
            parent_hash,  # parent_hash
        )
        self.block_hashes.append(self.event_id_counter)
        self.event_id_counter += 1

    def remove_event(self):
        self.publisher.publish_removed(
            self.event_id_counter,  # event_id
            [
                self.block_hashes[-1],
            ],  # block_hashes
        )
        self.event_id_counter += 1


# [TODO] to be deprecated
# KV events
class DynamoResult:
    OK = 0
    ERR = 1


class CtypesEventPublisher:
    def __init__(
        self, namespace: str, component: str, worker_id: int, kv_block_size: int
    ):
        self.event_id_counter = 0
        self.block_ids: List[int] = []

        # load event publisher library
        self.lib = ctypes.CDLL(os.environ["VLLM_KV_CAPI_PATH"])
        self.lib.dynamo_llm_init.argtypes = [c_char_p, c_char_p, c_int64, c_uint32]
        self.lib.dynamo_llm_init.restype = c_uint32
        result = self.lib.dynamo_llm_init(
            namespace.encode(), component.encode(), worker_id, kv_block_size
        )
        assert result == DynamoResult.OK

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
        )  # dynamo_llm_result_t

        self.lib.dynamo_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
        ]
        self.lib.dynamo_kv_event_publish_removed.restype = (
            ctypes.c_uint32
        )  # dynamo_llm_result_t

    def store_event(self, tokens, lora_id):
        parent_hash = (
            (ctypes.c_uint64 * 1)(self.event_id_counter)
            if self.event_id_counter > 0
            else None
        )
        result = self.lib.dynamo_kv_event_publish_stored(
            self.event_id_counter,  # uint64_t event_id
            (ctypes.c_uint32 * len(tokens))(*tokens),  # const uint32_t *token_ids
            (ctypes.c_size_t * 1)(len(tokens)),  # const uintptr_t *num_block_tokens
            (ctypes.c_uint64 * 1)(self.event_id_counter),  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
            parent_hash,  # const uint64_t *parent_hash
            lora_id,  # uint64_t lora_id
        )
        self.block_ids.append(self.event_id_counter)
        self.event_id_counter += 1

        assert result == DynamoResult.OK

    def remove_event(self):
        result = self.lib.dynamo_kv_event_publish_removed(
            self.event_id_counter,  # uint64_t event_id
            (ctypes.c_uint64 * 1)(self.block_ids[-1]),  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
        )
        self.event_id_counter += 1

        assert result == DynamoResult.OK

    def shutdown(self):
        result = self.lib.dynamo_llm_shutdown()
        assert result == DynamoResult.OK


async def test_metrics_aggregator(distributed_runtime):
    namespace = "kv_test"
    component = "metrics"
    kv_listener = distributed_runtime.namespace(namespace).component(component)
    await kv_listener.create_service()

    # aggregator
    metrics_aggregator = KvMetricsAggregator(kv_listener)

    # has nothing to aggregate as worker has not started
    metrics = await metrics_aggregator.get_metrics()
    assert not metrics.endpoints

    expected_metrics = {
        "request_active_slots": 0,
        "request_total_slots": 1024,
        "kv_active_blocks": 523,
        "kv_total_blocks": 777,
        "num_requests_waiting": 10,
        "gpu_cache_usage_perc": 0.5,
        "gpu_prefix_cache_hit_rate": 0.75,
    }

    # need 'create_task' to put publisher task in the background
    asyncio.create_task(metrics_publisher_task(kv_listener, expected_metrics))

    # needs time for publisher to spawn up
    for i in range(10):
        await asyncio.sleep(1)
        metrics = await metrics_aggregator.get_metrics()
        if metrics.endpoints:
            break
    assert metrics.endpoints
    for endpoint in metrics.endpoints:
        # [TODO] not really checking id for now, can't get it as create_endpoint()
        # create and serve the endpoint internally
        assert endpoint.worker_id != 0
        assert endpoint.request_active_slots == expected_metrics["request_active_slots"]
        assert endpoint.request_total_slots == expected_metrics["request_total_slots"]
        assert endpoint.kv_active_blocks == expected_metrics["kv_active_blocks"]
        assert endpoint.kv_total_blocks == expected_metrics["kv_total_blocks"]


async def metrics_publisher_task(kv_listener, expected_metrics):
    metrics_publisher = KvMetricsPublisher()
    metrics_publisher.publish(
        expected_metrics["request_active_slots"],
        expected_metrics["request_total_slots"],
        expected_metrics["kv_active_blocks"],
        expected_metrics["kv_total_blocks"],
        expected_metrics["num_requests_waiting"],
        expected_metrics["gpu_cache_usage_perc"],
        expected_metrics["gpu_prefix_cache_hit_rate"],
    )
    await metrics_publisher.create_endpoint(kv_listener)
