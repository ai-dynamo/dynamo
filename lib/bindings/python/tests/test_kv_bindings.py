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
import threading
from typing import List

import pytest

from dynamo.llm import ApproxKvIndexer, KvEventPublisher, KvIndexer, RadixTree
from dynamo.runtime import Component, DistributedRuntime

pytestmark = pytest.mark.pre_merge


@pytest.fixture
async def distributed_runtime():
    """Function-scoped runtime fixture for use with @pytest.mark.forked tests.

    Each test gets its own runtime in a forked process to avoid singleton conflicts.
    """
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "etcd", "nats")
    yield runtime
    runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.forked
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
    # Note: scores is now Dict[(worker_id, dp_rank), score]
    assert overlap_scores.scores is not None
    assert (
        len(overlap_scores.scores) == 1
    ), f"Expected 1 worker in scores, got {len(overlap_scores.scores)}"
    worker_key = (worker_id, 0)  # (worker_id, dp_rank)
    assert (
        worker_key in overlap_scores.scores
    ), f"Worker {worker_key} not found in scores"
    assert (
        overlap_scores.scores[worker_key] == 1
    ), f"Expected score 1 for worker {worker_key}, got {overlap_scores.scores[worker_key]}"

    blocks = radix_tree.dump_tree_as_events()
    assert len(blocks) == 1, f"Expected 1 block event, got {len(blocks)}"
    json.loads(blocks[0])  # check valid json

    # cleanup
    radix_tree.remove_worker(worker_id)
    blocks_empty = radix_tree.dump_tree_as_events()
    assert (
        len(blocks_empty) == 0
    ), f"Expected 0 block events after removal, got {len(blocks_empty)}"

    print(
        f"✓ RadixTree test passed: worker {worker_key} has score {overlap_scores.scores[worker_key]}"
    )


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.parametrize("num_threads", [2, 3, 5, 128])
@pytest.mark.parametrize("prepopulate_worker_ids", [True, False])
@pytest.mark.parametrize("expiration_duration_secs", [None])
@pytest.mark.parametrize("is_threaded", [True, False])
async def test_radix_tree_thread_safety(
    distributed_runtime,
    num_threads,
    prepopulate_worker_ids,
    expiration_duration_secs,
    is_threaded,
):
    """Test RadixTree thread safety by applying events from multiple threads."""
    radix_tree = RadixTree(expiration_duration_secs=expiration_duration_secs)
    threads = []
    done_counter = 0
    exception_counter = 0

    def worker(worker_id, prepopulate_worker_ids: bool = False):
        try:
            nonlocal done_counter
            worker_id = worker_id
            hash = worker_id
            if prepopulate_worker_ids:
                hash = (
                    2**32 - worker_id
                )  # use different hash for prepopulate_worker_ids
            assert 0 <= hash < 2**64  # needs to be valid u64
            store_event = {
                "event_id": worker_id,
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [
                            {
                                "block_hash": hash,
                                "tokens_hash": hash,
                            }
                        ],
                    }
                },
            }
            event_bytes = json.dumps(store_event).encode("utf-8")
            radix_tree.apply_event(worker_id, event_bytes)
            if not prepopulate_worker_ids:
                done_counter += 1
        except Exception as e:
            print(f"Exception in worker {worker_id}: {e}")
            nonlocal exception_counter
            exception_counter += 1

    if prepopulate_worker_ids:
        for i in range(num_threads):
            worker(i, prepopulate_worker_ids=True)
        assert (
            exception_counter == 0
        ), f"Warmup: expected 0 exceptions, got {exception_counter}"

    for i in range(num_threads):
        if is_threaded:
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        else:
            worker(i)
    if is_threaded:
        timeout = 10  # seconds
        for t in threads:
            t.join(timeout)
            assert not t.is_alive(), "Thread timed out"
    assert exception_counter == 0, f"Expected 0 exceptions, got {exception_counter}"
    assert (
        done_counter == num_threads
    ), f"Expected {num_threads} done, got {done_counter}"

    for i in range(num_threads):
        overlap_scores = radix_tree.find_matches([i])
        assert overlap_scores.scores is not None
        worker_key = (i, 0)
        assert (
            worker_key in overlap_scores.scores
        ), f"Worker {worker_key} not found in scores"
        assert (
            overlap_scores.scores[worker_key] == 1
        ), f"Expected score 1 for worker {worker_key}, got {overlap_scores.scores[worker_key]}"
    # get all blocks
    blocks = radix_tree.dump_tree_as_events()
    expected_blocks = num_threads + (prepopulate_worker_ids * num_threads)
    assert (
        len(blocks) == expected_blocks
    ), f"Expected {expected_blocks} block events, got {len(blocks)}"
    # remove single worker
    radix_tree.remove_worker(0)
    expected_blocks_after_removal = expected_blocks - (
        2 if prepopulate_worker_ids else 1
    )
    blocks_after_removal = radix_tree.dump_tree_as_events()
    assert (
        len(blocks_after_removal) == expected_blocks_after_removal
    ), f"Expected {expected_blocks_after_removal} block events after removal, got {len(blocks_after_removal)}"


# TODO Figure out how to test with different kv_block_size
# Right now I get an error in EventPublisher init when I run this test
# back to back. It occurs when calling dynamo_llm_init and I think is related to the
# OnceCell initializations not being reset.
# The test works individually if I run it with 32, then 11, then 64.
# @pytest.mark.parametrize("kv_block_size", [11, 32, 64])
@pytest.mark.asyncio
@pytest.mark.forked
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
    worker_key = (worker_id, 0)  # (worker_id, dp_rank)
    for retry in range(10):  # Try up to 10 times
        await asyncio.sleep(0.5)  # Wait 500ms between retries
        scores = await indexer.find_matches_for_request(test_token, lora_id)
        if (
            scores.scores
            and worker_key in scores.scores
            and scores.scores[worker_key] == 1
        ):
            break
        if retry == 9:  # Last iteration
            # Provide detailed error message for debugging
            assert scores.scores, f"No scores found after {(retry+1)*0.5}s"
            assert (
                worker_key in scores.scores
            ), f"Worker {worker_key} not in scores after {(retry+1)*0.5}s"
            assert (
                scores.scores[worker_key] == 1
            ), f"Expected score 1, got {scores.scores.get(worker_key)} after {(retry+1)*0.5}s"

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


@pytest.mark.asyncio
@pytest.mark.forked
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
    worker_key = (worker_id, 0)  # (worker_id, dp_rank)
    assert worker_key in scores.scores
    assert scores.scores[worker_key] == 2


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
