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

from dynemo.llm import KvIndexer, KvMetricsAggregator, KvMetricsPublisher
from dynemo.runtime import DistributedRuntime

pytestmark = pytest.mark.pre_merge

runtime = None


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


async def test_event_handler():
    global runtime
    if runtime is None:
        loop = asyncio.get_running_loop()
        runtime = DistributedRuntime(loop)

    namespace = "kv_test"
    component = "event"

    # publisher
    worker_id = 233
    event_publisher = EventPublisher(namespace, component, worker_id)

    # indexer
    kv_listener = runtime.namespace(namespace).component(component)
    await kv_listener.create_service()
    indexer = KvIndexer(kv_listener)

    test_token = [3] * 64
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


# KV events
class DynemoResult:
    OK = 0
    ERR = 1


class EventPublisher:
    def __init__(self, namespace: str, component: str, worker_id: int):
        self.event_id_counter = 0
        self.block_ids: List[int] = []

        # load event publisher library
        self.lib = ctypes.CDLL(os.environ["VLLM_KV_CAPI_PATH"])
        self.lib.dynemo_llm_init.argtypes = [c_char_p, c_char_p, c_int64]
        self.lib.dynemo_llm_init.restype = c_uint32
        result = self.lib.dynemo_llm_init(
            namespace.encode(), component.encode(), worker_id
        )
        assert result == DynemoResult.OK
        self.lib.dynemo_kv_event_publish_stored.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint32),  # token_ids
            ctypes.POINTER(ctypes.c_size_t),  # num_block_tokens
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
            ctypes.POINTER(ctypes.c_uint64),  # parent_hash
            ctypes.c_uint64,  # lora_id
        ]
        self.lib.dynemo_kv_event_publish_stored.restype = (
            ctypes.c_uint32
        )  # dynemo_llm_result_t

        self.lib.dynemo_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
        ]
        self.lib.dynemo_kv_event_publish_removed.restype = (
            ctypes.c_uint32
        )  # dynemo_llm_result_t

    def store_event(self, tokens, lora_id):
        parent_hash = (
            (ctypes.c_uint64 * 1)(self.event_id_counter)
            if self.event_id_counter > 0
            else None
        )
        result = self.lib.dynemo_kv_event_publish_stored(
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

        assert result == DynemoResult.OK

    def remove_event(self):
        result = self.lib.dynemo_kv_event_publish_removed(
            self.event_id_counter,  # uint64_t event_id
            (ctypes.c_uint64 * 1)(self.block_ids[-1]),  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
        )
        self.event_id_counter += 1

        assert result == DynemoResult.OK


async def test_metrics_aggregator():
    global runtime
    if runtime is None:
        loop = asyncio.get_running_loop()
        runtime = DistributedRuntime(loop)

    namespace = "kv_test"
    component = "metrics"
    kv_listener = runtime.namespace(namespace).component(component)
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
    }

    # need 'create_taskk' to put publisher task in the background
    asyncio.create_task(metrics_publisher(kv_listener, expected_metrics))

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


async def metrics_publisher(kv_listener, expected_metrics):
    metrics_publisher = KvMetricsPublisher()
    metrics_publisher.publish(
        expected_metrics["request_active_slots"],
        expected_metrics["request_total_slots"],
        expected_metrics["kv_active_blocks"],
        expected_metrics["kv_total_blocks"],
    )
    await metrics_publisher.create_endpoint(kv_listener)
