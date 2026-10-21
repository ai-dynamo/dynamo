# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import json
import subprocess
import sys
import threading

import pytest

from dynamo.llm import RadixTree

try:
    from dynamo.llm import SelectionService
except ImportError:
    SelectionService = None

from dynamo.llm.exceptions import SelectionServiceError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


@pytest.mark.timeout(5)  # Expected: ~1s, timeout set to 5x for safety
def test_radix_tree_binding():
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


@pytest.mark.timeout(5)  # Expected: ~1s, timeout set to 5x for safety
@pytest.mark.parametrize("num_threads", [2, 3, 5, 128])
@pytest.mark.parametrize("prepopulate_worker_ids", [True, False])
@pytest.mark.parametrize("is_threaded", [True, False])
def test_radix_tree_thread_safety(
    num_threads,
    prepopulate_worker_ids,
    is_threaded,
):
    """Test RadixTree thread safety by applying events from multiple threads."""
    radix_tree = RadixTree()
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


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize(
    "ttl", [-1, 0, 0.5, float("nan"), float("inf"), 31536001, 1e300]
)
@pytest.mark.timeout(5)
def test_selection_service_rejects_invalid_affinity_ttl(ttl):
    with pytest.raises(ValueError, match="session_affinity_ttl_secs must be between"):
        SelectionService(session_affinity_ttl_secs=ttl)


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize("ttl", ["-1", "NaN", "31536001"])
@pytest.mark.timeout(60)  # subprocess wait is capped at 15 s per case
def test_selection_service_cli_rejects_invalid_affinity_ttl(ttl):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.select_service",
            f"--session-affinity-ttl-secs={ttl}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "session_affinity_ttl_secs must be between" in result.stderr
    assert "panicked" not in result.stderr


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize("ttl", [1.0, 1.5, 31536000.0])
@pytest.mark.timeout(5)
def test_selection_service_accepts_valid_affinity_ttl(ttl):
    service = SelectionService(indexer_threads=1, session_affinity_ttl_secs=ttl)
    service.shutdown()


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_selects_registered_worker(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    service = SelectionService(indexer_threads=1)

    try:
        record = await service.upsert_worker(
            {
                "worker_id": 1,
                "model_name": "model",
                "endpoint": "http://worker-1:8000",
                "block_size": 4,
                "max_num_batched_tokens": 1024,
            }
        )
        assert record["lifecycle"] == "schedulable"

        ready = service.ready()
        assert ready["ready"] is True
        assert ready["schedulable_workers"] == 1

        selected = await service.select(
            {
                "model_name": "model",
                "token_ids": [1, 2, 3, 4],
                "selection_id": "sel-a",
            }
        )
        assert selected["selection_id"] == "sel-a"
        assert selected["worker_id"] == 1
        assert selected["dp_rank"] == 0
        assert selected["endpoint"] == "http://worker-1:8000"
    finally:
        await service.shutdown_async()


@pytest.fixture
async def session_binding_services(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    source = SelectionService(indexer_threads=1, session_affinity_ttl_secs=10)
    peer = SelectionService(indexer_threads=1, session_affinity_ttl_secs=10)
    try:
        # Give both services the same routing partition and worker catalog.
        for service in (source, peer):
            for worker_id in (1, 2):
                await service.upsert_worker(
                    {
                        "worker_id": worker_id,
                        "model_name": "model",
                        "routing_group": "host",
                        "endpoint": f"http://worker-{worker_id}",
                        "block_size": 4,
                        "max_num_batched_tokens": 1024,
                    }
                )
        yield source, peer
    finally:
        await source.shutdown_async()
        await peer.shutdown_async()


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize("competing_score", ["load", "overlap"])
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_affinity_scores(
    session_binding_services, competing_score
):
    source, peer = session_binding_services
    partition = {"model_name": "model", "routing_group": "host"}
    # Bind the session to worker 1 and replicate its assignment to the peer.
    origin = await source.select_and_reserve(
        {
            **partition,
            "token_ids": [1, 2, 3, 4],
            "selection_id": "origin",
            "session_id": "s",
            "allowed_worker_ids": [1],
        }
    )
    await source.free_reservation("origin")
    origin = json.loads(json.dumps(origin))
    await peer.create_reservation({**origin, "skip_load_booking": True})

    # Make worker 2 preferable by load or cached-prefix overlap.
    warm_tokens = list(range(100, 132))
    await peer.select_and_reserve(
        {
            **partition,
            "token_ids": warm_tokens,
            "selection_id": "background",
            "allowed_worker_ids": [1 if competing_score == "load" else 2],
        }
    )
    if competing_score == "overlap":
        await peer.free_reservation("background")
        await peer.indexer_snapshot()

    # Unpinned requests prefer worker 2; session affinity keeps worker 1.
    tokens = [5, 6, 7, 8] if competing_score == "load" else warm_tokens
    request = {**partition, "token_ids": tokens}
    control = await peer.select(request)
    assert control["worker_id"] == 2
    selected = await peer.select_and_reserve(
        {**request, "selection_id": "next-turn", "session_id": "s"}
    )
    assert selected["worker_id"] == 1
    assert selected["session_binding"] == origin["session_binding"]
    if competing_score == "overlap":
        assert control["overlap"]["longest_matched"] == len(tokens)
        assert selected["overlap"]["longest_matched"] == 0


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize("booking", ["book", "skip", "fail"])
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_peer_affinity(session_binding_services, booking):
    source, peer = session_binding_services
    partition = {"model_name": "model", "routing_group": "host"}
    # Export the session assignment chosen by the source.
    origin = await source.select_and_reserve(
        {
            **partition,
            "token_ids": [1, 2, 3, 4],
            "selection_id": "origin",
            "session_id": "s",
            "allowed_worker_ids": [1],
        }
    )

    # Import affinity with successful, skipped, or failed load booking.
    request = {**origin, "skip_load_booking": booking == "skip"}
    if booking == "fail":
        request["dp_rank"] = 99
        with pytest.raises(SelectionServiceError):
            await peer.create_reservation(request)
    else:
        result = await peer.create_reservation(request)
        if booking == "skip":
            assert result is None
        else:
            assert result["worker_id"] == 1

    # Load reflects the booking outcome; the next turn keeps session affinity.
    loads = peer.loads(**partition)[0]["loads"]
    assert sum(row["active_requests"] for row in loads) == int(booking == "book")
    selected = await peer.select_and_reserve(
        {
            **partition,
            "selection_id": "next-turn",
            "token_ids": [5, 6, 7, 8],
            "session_id": "s",
        }
    )
    assert selected["worker_id"] == 1
    assert selected["session_binding"] == origin["session_binding"]


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_skip_after_shutdown(
    session_binding_services,
):
    _, peer = session_binding_services
    await peer.shutdown_async()

    # Skipping load booking must still enforce shutdown.
    with pytest.raises(SelectionServiceError):
        await peer.create_reservation(
            {"selection_id": "after-shutdown", "skip_load_booking": True}
        )


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize(
    "method,payload",
    [
        pytest.param("upsert_worker", {"model_name": "model"}, id="worker"),
        pytest.param(
            "create_reservation",
            {"selection_id": "malformed", "session_binding": {"session_id": "s"}},
            id="session-binding",
        ),
    ],
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_malformed_payload_raises_value_error(
    monkeypatch, method, payload
):
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    service = SelectionService(indexer_threads=1)

    try:
        # Malformed input surfaces as a plain ValueError, not a SelectionServiceError.
        with pytest.raises(ValueError):
            await getattr(service, method)(payload)
    finally:
        service.shutdown()


@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_not_ready_carries_status(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    service = SelectionService(indexer_threads=1)

    try:
        # Selecting before any worker is schedulable is a typed selector failure.
        with pytest.raises(SelectionServiceError) as exc_info:
            await service.select({"model_name": "model", "token_ids": [1, 2, 3, 4]})
        assert exc_info.value.kind == "not_ready"
        assert exc_info.value.status_code == 503
    finally:
        service.shutdown()
