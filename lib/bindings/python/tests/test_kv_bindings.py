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


import asyncio
import inspect
import json
import subprocess
import sys
import threading
import types
import uuid

import pytest

from dynamo._core import _convert_preprocessed_request
from dynamo.llm import (
    KvRouter,
    KvRouterConfig,
    ModelInput,
    ModelType,
    RadixTree,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime

try:
    from dynamo.llm import SelectionService
except ImportError:
    SelectionService = None

from dynamo.llm.exceptions import SelectionServiceError

# `unit` and `parallel` are set per test: the KvRouter tests below start a runtime.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.unit
@pytest.mark.parallel
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


@pytest.mark.unit
@pytest.mark.parallel
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


@pytest.mark.unit
@pytest.mark.parallel
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


@pytest.mark.unit
@pytest.mark.parallel
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


@pytest.mark.unit
@pytest.mark.parallel
@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize("ttl", [1.0, 1.5, 31536000.0])
@pytest.mark.timeout(5)
def test_selection_service_accepts_valid_affinity_ttl(ttl):
    service = SelectionService(indexer_threads=1, session_affinity_ttl_secs=ttl)
    service.shutdown()


@pytest.mark.unit
@pytest.mark.parallel
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

        # Sequences other than an exact list take the generic conversion.
        selected = await service.select(
            {"model_name": "model", "token_ids": (1, 2, 3, 4)}
        )
        assert selected["worker_id"] == 1
    finally:
        await service.shutdown_async()


@pytest.fixture
async def session_binding_services(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    source = SelectionService(
        indexer_threads=1,
        session_affinity_ttl_secs=10,
        session_replication_writer_id=11,
    )
    peer = SelectionService(
        indexer_threads=1,
        session_affinity_ttl_secs=10,
        session_affinity_mode="soft",
        session_replication_writer_id=22,
    )
    try:
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


@pytest.mark.unit
@pytest.mark.parallel
@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_session_binding_stream(session_binding_services):
    source, peer = session_binding_services
    stream = source.session_binding_updates()
    peer_stream = peer.session_binding_updates()
    with pytest.raises(SelectionServiceError):
        source.session_binding_updates()
    partition = {"model_name": "model", "routing_group": "host"}
    selected = await source.select_and_reserve(
        {
            **partition,
            "token_ids": [1, 2, 3, 4],
            "selection_id": "origin",
            "session_id": "s",
            "allowed_worker_ids": [1],
        }
    )
    assert "session_binding" not in selected
    (binding,) = json.loads(json.dumps(await anext(stream)))
    assert binding["writer_id"] == 11
    peer.apply_session_binding(binding)
    await source.free_reservation("origin")
    assert await anext(stream) == [binding]  # A repeated version refreshes TTL.
    selected = await peer.select_and_reserve(
        {
            **partition,
            "token_ids": [5, 6, 7, 8],
            "selection_id": "next",
            "session_id": "s",
        }
    )
    assert selected["worker_id"] == 1
    assert await anext(peer_stream) == [
        binding
    ]  # Local commits preserve imported versions.
    # The default hard mode rejects an excluded binding; soft mode rebinds.
    request = {
        **partition,
        "token_ids": [9, 10, 11, 12],
        "selection_id": "excluded",
        "session_id": "s",
        "allowed_worker_ids": [2],
    }
    with pytest.raises(ValueError, match="session s is bound to worker 1, not 2"):
        await source.select_and_reserve(request)
    assert (await peer.select_and_reserve(request))["worker_id"] == 2
    assert (await anext(peer_stream))[0]["worker_id"] == 2
    await source.shutdown_async()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    with pytest.raises(SelectionServiceError):
        source.apply_session_binding(binding)


@pytest.mark.unit
@pytest.mark.parallel
@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_session_binding_stream_drains_queue(
    session_binding_services,
):
    source, _ = session_binding_services
    stream = source.session_binding_updates()
    for session_id in ("a", "b", "c"):
        await source.select_and_reserve(
            {
                "model_name": "model",
                "routing_group": "host",
                "token_ids": [1, 2, 3, 4],
                "selection_id": session_id,
                "session_id": session_id,
            }
        )
    # Bindings queued while the consumer was away arrive as one item.
    batch = await anext(stream)
    assert [binding["session_id"] for binding in batch] == ["a", "b", "c"]


@pytest.mark.unit
@pytest.mark.parallel
@pytest.mark.skipif(
    SelectionService is None,
    reason="SelectionService requires the select-service Cargo feature",
)
@pytest.mark.parametrize(
    "method,payload",
    [
        pytest.param("upsert_worker", {"model_name": "model"}, id="worker"),
        pytest.param(
            "apply_session_binding",
            {"session_id": "s"},
            id="session-binding",
        ),
    ],
)
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_selection_service_malformed_payload_raises_value_error(
    monkeypatch, method, payload
):
    """Malformed binding inputs raise ValueError without truthy boolean coercion."""
    monkeypatch.setenv("DYN_ROUTER_USE_KV_EVENTS", "false")
    service = SelectionService(indexer_threads=1)

    try:
        # Malformed input surfaces as a plain ValueError, not a SelectionServiceError.
        with pytest.raises(ValueError):
            result = getattr(service, method)(payload)
            if inspect.isawaitable(result):
                await result
        # Prompt booleans are not coerced by truthiness.
        with pytest.raises(ValueError, match="boolean"):
            await service.select({"token_ids": [1, 2, 3, 4], "is_eagle": "false"})

        for field in ("is_eagle", "engine_owned_mm_identity"):
            with pytest.raises(ValueError, match="boolean"):
                await service.upsert_worker({"worker_id": 1, field: "false"})
            with pytest.raises(ValueError, match="boolean"):
                await service.patch_worker(1, {field: "false"})

        class IteratesNegative(list):
            def __iter__(self):
                return iter([-1])

        # Out-of-range or non-int ids keep the conversion's ValueError; a list subclass
        # is converted through its own iterator.
        for token_ids in [[1, -1], [1, 2**32], [1, "2"], IteratesNegative([1, 2])]:
            with pytest.raises(ValueError, match="token_ids"):
                await service.select({"token_ids": token_ids})
    finally:
        service.shutdown()


@pytest.mark.unit
@pytest.mark.parallel
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


# KvRouter reads exact builtin `token_ids` and `extra_args` directly, else uses pythonize;
# the worker must get pythonize's result, or the call must raise pythonize's error.


def _preprocessed(token_ids=(1, 2, 3), **fields):
    return {"model": "m", "token_ids": list(token_ids), **fields}


class _TokenList(list):
    pass


class _ExtraArgs(dict):
    pass


_EXTRA_ARGS_DIRECT = [
    None,
    {
        "nemo_rl": {
            "template_token_ids": [7, 8, 9],
            "request": {"max_tokens": None, "tools": [{"type": "function"}]},
            "verify": False,
        }
    },
    {"ints": [0, -1, -(2**63), 2**63 - 1, 2**63, 2**64 - 1]},
    {"floats": [1.5, -0.0, float("nan"), float("inf")]},
    {"z": (1, "x"), "a": [], "m": {}, "s": "héllo", "b": True},
]

_EXTRA_ARGS_FALLBACK = [
    {"s": {1, 2}},
    {1: "x"},
    {"big": 2**64},
    _ExtraArgs(ints=[1, 2]),
]

_REQUESTS_DIRECT = [
    _preprocessed(),
    _preprocessed(token_ids=[]),
    _preprocessed(
        stop_conditions={"max_tokens": 8},
        sampling_options={"temperature": 0.5},
        annotations=["nemo_gym:rollout"],
    ),
    *(_preprocessed(extra_args=extra_args) for extra_args in _EXTRA_ARGS_DIRECT),
]

_REQUESTS_FALLBACK = [
    {"model": "m", "token_ids": _TokenList([1, 2])},
    _preprocessed(token_ids=[1, -1]),
    {"model": "m"},
    types.MappingProxyType(_preprocessed()),
    *(_preprocessed(extra_args=extra_args) for extra_args in _EXTRA_ARGS_FALLBACK),
]


def _pythonized(preprocessed):
    """pythonize's request, or the exception it raises."""
    try:
        return _convert_preprocessed_request(preprocessed, fast_path=False)
    except Exception as e:
        return e


@pytest.fixture
async def echo_router(temp_file_store, monkeypatch):
    """A KvRouter in front of one worker that records each request it receives."""
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
    received = []

    async def echo(request, _context=None):
        received.append(request)
        yield {"token_ids": [], "finish_reason": "stop"}

    loop = asyncio.get_running_loop()
    endpoint_path = f"echo-router-{uuid.uuid4().hex}.worker.generate"
    worker_runtime = DistributedRuntime(loop, "file", "tcp")
    router_runtime = DistributedRuntime(loop, "file", "tcp")
    server_task = None
    try:
        endpoint = worker_runtime.endpoint(endpoint_path)
        await register_model(
            ModelInput.Tensor,
            ModelType.TensorBased,
            endpoint,
            "echo-router-worker",
            worker_type=WorkerType.Aggregated,
            tensor_model_config={
                "name": "echo-router-worker",
                "inputs": [],
                "outputs": [],
            },
        )
        server_task = asyncio.ensure_future(endpoint.serve_endpoint(echo))
        router_endpoint = router_runtime.endpoint(endpoint_path)
        client = await router_endpoint.client()
        assert len(await client.wait_for_instances()) == 1
        router = KvRouter(router_endpoint, 4, KvRouterConfig(use_kv_events=False))
        yield router, received
    finally:
        router_runtime.shutdown()
        worker_runtime.shutdown()
        if server_task is not None:
            await asyncio.wait_for(server_task, timeout=10)


async def _received_or_error(received, call):
    """The request the worker received for `call`, or the exception `call` raised."""
    try:
        stream = await call()
    except Exception as e:
        return e
    responses = [response async for response in stream]
    assert [response["finish_reason"] for response in responses] == ["stop"]
    return received[-1]


def _assert_same(actual, expected):
    if isinstance(expected, Exception):
        assert isinstance(actual, Exception), actual
        assert (type(actual), str(actual)) == (type(expected), str(expected))
    else:
        assert not isinstance(actual, Exception), actual
        # JSON text also pins key order and NaN -> null.
        assert json.dumps(actual) == json.dumps(expected)


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("preprocessed", _REQUESTS_DIRECT + _REQUESTS_FALLBACK)
async def test_kv_router_generate_from_request_matches_pythonize(
    echo_router, preprocessed
):
    router, received = echo_router
    expected = _pythonized(preprocessed)
    actual = await _received_or_error(
        received, lambda: router.generate_from_request(preprocessed)
    )
    if isinstance(actual, dict):
        # The router adds its routing hints on the way to the worker.
        actual.pop("routing")
    _assert_same(actual, expected)


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("extra_args", _EXTRA_ARGS_DIRECT + _EXTRA_ARGS_FALLBACK)
async def test_kv_router_generate_extra_args_matches_pythonize(echo_router, extra_args):
    router, received = echo_router
    expected = _pythonized(_preprocessed(extra_args=extra_args))
    if not isinstance(expected, Exception):
        expected = expected.get("extra_args")
    actual = await _received_or_error(
        received, lambda: router.generate([1, 2, 3], "m", extra_args=extra_args)
    )
    if isinstance(actual, dict):
        actual = actual.get("extra_args")
    _assert_same(actual, expected)
