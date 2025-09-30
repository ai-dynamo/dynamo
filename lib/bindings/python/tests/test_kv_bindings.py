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
from typing import List

import pytest

from dynamo.llm import (
    ApproxKvIndexer,
    ForwardPassMetrics,
    KvEventPublisher,
    KvIndexer,
    KvMetricsAggregator,
    KvStats,
    RadixTree,
    WorkerMetricsPublisher,
    WorkerStats,
)
from dynamo.runtime import Component, DistributedRuntime

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module")
async def distributed_runtime():
    """TODO: This should not use scope='module' as DistributedRuntime has singleton requirements.
    and blocks any tests with DistributedRuntime(loop, True) from running in the same process, or any forked process.
    """
    loop = asyncio.get_running_loop()
    return DistributedRuntime(loop, False)


# TODO: enable pytest.mark.forked + scope='function' runtime.
async def test_radix_tree_binding(distributed_runtime):
    """Test RadixTree binding directly with store event and find matches"""
    import json

    # Create RadixTree instance
    radix_tree = RadixTree()

    # Create a store event with parent_hash=None, block_hash=0
    # Following the KvCacheEvent format from the Rust protocols
    store_event = {
        "event_id": 1,
        "data": {
            "stored": {
                "parent_hash": None,
                "blocks": [
                    {
                        "block_hash": 0,
                        "tokens_hash": 0,  # Using 0 for both hashes to match tokens [0]
                    }
                ],
            }
        },
    }

    # Convert to JSON bytes
    event_bytes = json.dumps(store_event).encode("utf-8")

    # Apply the event to worker_id 0
    worker_id = 0
    radix_tree.apply_event(worker_id, event_bytes)

    # Find matches for tokens [0]
    # The sequence parameter expects token hashes, so we use [0] to match tokens_hash=0
    overlap_scores = radix_tree.find_matches([0])

    # Verify the results
    assert overlap_scores.scores is not None
    assert (
        len(overlap_scores.scores) == 1
    ), f"Expected 1 worker in scores, got {len(overlap_scores.scores)}"
    assert worker_id in overlap_scores.scores, f"Worker {worker_id} not found in scores"
    assert (
        overlap_scores.scores[worker_id] == 1
    ), f"Expected score 1 for worker {worker_id}, got {overlap_scores.scores[worker_id]}"

    print(
        f"✓ RadixTree test passed: worker {worker_id} has score {overlap_scores.scores[worker_id]}"
    )


# TODO Figure out how to test with different kv_block_size
# Right now I get an error in EventPublisher init when I run this test
# back to back. It occurs when calling dynamo_llm_init and I think is related to the
# OnceCell initializations not being reset.
# The test works individually if I run it with 32, then 11, then 64.
# @pytest.mark.parametrize("kv_block_size", [11, 32, 64])
@pytest.mark.skip(reason="Flakey in CI. Likely race condition going on.")
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
    # Retry loop for CI environments where processing may take longer
    for retry in range(10):  # Try up to 10 times
        await asyncio.sleep(0.5)  # Wait 500ms between retries
        scores = await indexer.find_matches_for_request(test_token, lora_id)
        if (
            scores.scores
            and worker_id in scores.scores
            and scores.scores[worker_id] == 1
        ):
            break
        if retry == 9:  # Last iteration
            # Provide detailed error message for debugging
            assert scores.scores, f"No scores found after {(retry+1)*0.5}s"
            assert (
                worker_id in scores.scores
            ), f"Worker {worker_id} not in scores after {(retry+1)*0.5}s"
            assert (
                scores.scores[worker_id] == 1
            ), f"Expected score 1, got {scores.scores.get(worker_id)} after {(retry+1)*0.5}s"

    # remove event
    event_publisher.remove_event()
    # Retry loop for event removal verification
    for retry in range(10):  # Try up to 10 times
        await asyncio.sleep(0.5)  # Wait 500ms between retries
        scores = await indexer.find_matches_for_request(test_token, lora_id)
        if not scores.scores:
            break
        if retry == 9:  # Last iteration
            assert (
                not scores.scores
            ), f"Scores still present after {(retry+1)*0.5}s: {scores.scores}"


# TODO: enable pytest.mark.forked + scope='function' runtime.
async def test_approx_kv_indexer(distributed_runtime):
    kv_block_size = 32
    namespace = "kv_test"
    component = "approx_kv"
    kv_listener = distributed_runtime.namespace(namespace).component(component)
    await kv_listener.create_service()

    indexer = ApproxKvIndexer(kv_listener, kv_block_size, 30.0)

    tokens = [0] * (kv_block_size * 2)

    scores = await indexer.find_matches_for_request(tokens)
    assert not scores.scores

    worker_id = 0

    await indexer.process_routing_decision_for_request(tokens, worker_id)

    scores = await indexer.find_matches_for_request(tokens)
    assert scores.scores
    assert worker_id in scores.scores
    assert scores.scores[worker_id] == 2


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


# TODO: enable pytest.mark.forked + scope='function' runtime.
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
    # Using shorter intervals for faster detection in normal cases
    for i in range(20):  # Try up to 20 times (10 seconds total)
        await asyncio.sleep(0.5)  # Wait 500ms between retries
        metrics = await metrics_aggregator.get_metrics()
        if metrics.endpoints:
            break
    assert metrics.endpoints, f"No metrics endpoints found after {(i+1)*0.5}s"
    for endpoint in metrics.endpoints:
        # [TODO] not really checking id for now, can't get it as create_endpoint()
        # create and serve the endpoint internally
        assert endpoint.worker_id != 0
        assert endpoint.request_active_slots == expected_metrics["request_active_slots"]
        assert endpoint.request_total_slots == expected_metrics["request_total_slots"]
        assert endpoint.kv_active_blocks == expected_metrics["kv_active_blocks"]
        assert endpoint.kv_total_blocks == expected_metrics["kv_total_blocks"]


async def metrics_publisher_task(kv_listener, expected_metrics):
    # Construct the structured ForwardPassMetrics payload expected by the
    # current Rust bindings instead of passing the individual scalar values
    # directly. The API for `WorkerMetricsPublisher.publish`
    # changed from a list of positional scalars to a single
    # `ForwardPassMetrics` object.

    metrics_publisher = WorkerMetricsPublisher()

    worker_stats = WorkerStats(
        expected_metrics["request_active_slots"],
        expected_metrics["request_total_slots"],
        expected_metrics["num_requests_waiting"],
        None,
    )

    kv_stats = KvStats(
        expected_metrics["kv_active_blocks"],
        expected_metrics["kv_total_blocks"],
        expected_metrics["gpu_cache_usage_perc"],
        expected_metrics["gpu_prefix_cache_hit_rate"],
    )

    metrics = ForwardPassMetrics(worker_stats, kv_stats, None)

    # Publish and expose the metrics via the endpoint so that the aggregator
    # test can discover them.
    metrics_publisher.publish(metrics)
    await metrics_publisher.create_endpoint(kv_listener)
