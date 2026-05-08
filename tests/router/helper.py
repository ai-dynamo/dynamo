# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import random
import string
import sys
from typing import Any, Optional

import aiohttp
import nats

from dynamo.llm import KvRouter
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

NUM_REQUESTS = 100
BLOCK_SIZE = 16


def _nats_server() -> str:
    # Prefer dynamically-started NATS from per-test fixtures when present.
    return os.environ.get("NATS_SERVER", "nats://localhost:4222")


def generate_random_suffix() -> str:
    """Generate a 10-character random alphabetic suffix for namespace isolation."""
    return "".join(random.choices(string.ascii_lowercase, k=10))  # noqa: S311


def get_kv_indexer_command() -> list[str]:
    """Return the preferred standalone indexer command for the current Python env."""
    return [sys.executable, "-m", "dynamo.indexer"]


def assert_event_dumps_equal(
    expected: list[dict],
    actual: list[dict],
    expected_label: str,
    actual_label: str,
) -> None:
    """Assert two sorted event dump lists are equal, ignoring event_id fields."""
    assert len(expected) == len(actual), (
        f"{expected_label} has {len(expected)} events, "
        f"{actual_label} has {len(actual)} events"
    )

    differences = []
    for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
        exp_compare = exp_item.copy()
        act_compare = act_item.copy()
        if "event" in exp_compare and "event_id" in exp_compare["event"]:
            del exp_compare["event"]["event_id"]
        if "event" in act_compare and "event_id" in act_compare["event"]:
            del act_compare["event"]["event_id"]
        if exp_compare != act_compare:
            differences.append(
                {"index": i, expected_label: exp_item, actual_label: act_item}
            )

    if differences:
        error_msg = (
            f"{expected_label} and {actual_label} differ. "
            f"Found {len(differences)} differences:\n"
        )
        for diff in differences:
            error_msg += f"\nDifference at index {diff['index']}:\n"
            error_msg += (
                f"{expected_label}: {json.dumps(diff[expected_label], indent=2)}\n"
            )
            error_msg += f"{actual_label}: {json.dumps(diff[actual_label], indent=2)}\n"
            error_msg += "-" * 80 + "\n"
        assert False, error_msg


def verify_response_worker_ids(
    response_worker_ids: list[dict[str, Optional[int]]],
    key: str,
    expected_worker_id: int,
) -> None:
    """Verify that all responses have the same worker ID for a given key.

    Args:
        response_worker_ids: List of dicts with worker ID info from responses.
        key: The key to check (e.g., "decode_worker_id" or "prefill_worker_id").
        expected_worker_id: The expected worker ID value.

    Raises:
        AssertionError: If any response is missing the key, values differ, or don't match expected.
    """
    worker_ids = [r.get(key) for r in response_worker_ids]
    logger.info(f"Response {key}s: {worker_ids}")

    # All responses should have the key
    assert all(
        wid is not None for wid in worker_ids
    ), f"Expected all {len(response_worker_ids)} responses to have {key}, got: {worker_ids}"

    # All values should be the same (due to prefix reuse routing)
    unique_ids = set(worker_ids)
    assert len(unique_ids) == 1, (
        f"Expected all responses to have the same {key} (due to prefix reuse), "
        f"but found {len(unique_ids)} unique values: {unique_ids}"
    )

    # The value should match the expected worker ID
    actual_worker_id = worker_ids[0]
    assert actual_worker_id == expected_worker_id, (
        f"Expected {key}={expected_worker_id} (forced in first request), "
        f"but got {key}={actual_worker_id}"
    )
    logger.info(
        f"✓ Verified all {len(response_worker_ids)} responses have {key}={actual_worker_id}"
    )


def verify_response_timing(timing_info: dict[str, Any], disagg: bool = False) -> None:
    """Verify timing info has valid values (ttft_ms > 0, total_time_ms > 0).

    Args:
        timing_info: Dict of timing fields from nvext.timing in the response.
        disagg: If True, also verify kv_transfer_estimated_latency_ms > 0 (disaggregated mode only).
    """
    ttft_ms = timing_info.get("ttft_ms")
    total_time_ms = timing_info.get("total_time_ms")

    assert ttft_ms is not None and ttft_ms > 0, f"Expected ttft_ms > 0, got: {ttft_ms}"
    assert (
        total_time_ms is not None and total_time_ms > 0
    ), f"Expected total_time_ms > 0, got: {total_time_ms}"
    assert (
        total_time_ms >= ttft_ms
    ), f"Expected total_time_ms >= ttft_ms, got {total_time_ms} < {ttft_ms}"
    logger.info(
        f"✓ Verified timing: ttft_ms={ttft_ms:.2f}, total_time_ms={total_time_ms:.2f}"
    )

    if disagg:
        kv_transfer_estimated_latency_ms = timing_info.get(
            "kv_transfer_estimated_latency_ms"
        )
        assert (
            kv_transfer_estimated_latency_ms is not None
            and kv_transfer_estimated_latency_ms > 0
        ), (
            f"Expected kv_transfer_estimated_latency_ms > 0 in disaggregated mode, "
            f"got: {kv_transfer_estimated_latency_ms}"
        )
        logger.info(
            f"✓ Verified kv_transfer_estimated_latency_ms={kv_transfer_estimated_latency_ms:.2f}"
        )


########################################################
# Utility functions
########################################################


async def wait_for_frontend_ready(
    frontend_url: str, expected_num_workers: int = 2, timeout: int = 120
):
    """Wait for backend worker(s) to be ready via the HTTP frontend (OpenAI API).

    This function performs a two-phase readiness check through the frontend HTTP server:
        1. Polls GET /v1/models until at least one model is registered (workers connected)
        2. Sends a test POST to /v1/chat/completions to verify the request pipeline is functional

    Use this when testing through the HTTP frontend server (dynamo.frontend).
    For direct Python API testing with KvRouter, use wait_for_workers_ready() instead.

    Args:
        frontend_url: Base URL of the frontend HTTP server (e.g., "http://localhost:8000")
        expected_num_workers: Number of workers to wait for (currently logs but doesn't enforce)
        timeout: Maximum time to wait in seconds for both phases combined

    Raises:
        TimeoutError: If workers don't register or pipeline doesn't become ready within timeout
        aiohttp.ClientError: If HTTP requests fail unexpectedly
    """

    models_url = f"{frontend_url}/v1/models"
    chat_url = f"{frontend_url}/v1/chat/completions"
    start_time = asyncio.get_event_loop().time()

    logger.info(
        f"Waiting for {expected_num_workers} workers to register on HTTP frontend (timeout={timeout}s)..."
    )

    # Phase 1: Wait for models to appear in /v1/models
    model_name = None
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time

        if elapsed > timeout:
            raise TimeoutError(
                f"Timeout waiting for vLLM workers. Waited {elapsed:.1f}s, no workers registered."
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        if len(models) > 0:
                            model_name = models[0].get("id")
                            logger.info(
                                f"Workers registered. Found {len(models)} model(s): {[m.get('id') for m in models]}"
                            )
                            break
                        else:
                            logger.debug(
                                f"No models registered yet (elapsed: {elapsed:.1f}s)"
                            )
        except Exception as e:
            logger.debug(f"Error checking models endpoint: {e}")

        # Wait before next poll
        await asyncio.sleep(1)

    # Phase 2: Wait for chat completions pipeline to be ready
    logger.info("Waiting for chat completions pipeline to be built...")
    test_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
        "stream": False,
    }

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time

        if elapsed > timeout:
            raise TimeoutError(
                f"Timeout waiting for chat completions pipeline. Waited {elapsed:.1f}s."
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(chat_url, json=test_payload) as response:
                    if response.status == 200:
                        logger.info("Chat completions pipeline ready!")
                        return
                    else:
                        logger.debug(
                            f"Chat completions not ready yet, status {response.status} (elapsed: {elapsed:.1f}s)"
                        )
        except Exception as e:
            logger.debug(f"Error testing chat completions: {e}")

        # Wait before next poll
        await asyncio.sleep(1)


async def poll_for_worker_instances(
    endpoint,
    expected_num_workers: int,
    max_wait_time: int = 60,
) -> list[int]:
    """Poll the endpoint's discovery client until the expected number of worker instances appear.

    Args:
        endpoint: The endpoint object to get the client from
        expected_num_workers: Number of worker instances to wait for
        max_wait_time: Timeout in seconds

    Returns:
        List of discovered instance IDs (unsorted).

    Raises:
        AssertionError: If the expected number of workers don't appear within max_wait_time.
    """
    logger.info(f"Waiting for {expected_num_workers} worker instance(s) to register...")
    client = await endpoint.client()
    instance_ids: list[int] = []
    start_time = asyncio.get_running_loop().time()

    while len(instance_ids) < expected_num_workers:
        instance_ids = client.instance_ids()
        logger.info(f"Found {len(instance_ids)} instance(s): {instance_ids}")

        if len(instance_ids) >= expected_num_workers:
            break

        if asyncio.get_running_loop().time() - start_time > max_wait_time:
            raise AssertionError(
                f"Timeout waiting for workers. Found {len(instance_ids)} instance(s), expected {expected_num_workers}"
            )

        await asyncio.sleep(1.0)

    return instance_ids


async def wait_for_workers_ready(
    endpoint,
    router: KvRouter,
    expected_num_workers: int,
    model_name: str,
) -> list[int]:
    """Wait for workers to be ready and return their instance IDs.
    Supports mocker and vLLM workers.

    This function polls the endpoint's client for instance IDs until the expected
    number of workers are available, then sends a warmup request to verify they
    can handle requests.

    Args:
        endpoint: The endpoint object to get the client from
        router: The KvRouter to use for sending warmup requests
        expected_num_workers: Number of workers to wait for

    Returns:
        Sorted list of unique instance IDs (ints).

    Raises:
        AssertionError: If workers don't become ready or warmup request fails.
    """
    instance_ids = await poll_for_worker_instances(endpoint, expected_num_workers)

    # Send a warmup request to verify workers can handle requests
    test_token_ids = [random.randint(1, 10000) for _ in range(4)]
    logger.info(f"Sending warmup request with {len(test_token_ids)} tokens")

    try:
        await send_request_via_python_kv_router(
            kv_python_router=router,
            model_name=model_name,
            token_ids=test_token_ids,
            initial_wait=1.0,
            max_retries=8,
            stop_conditions={
                "ignore_eos": True,
                "max_tokens": 2,
            },
        )
    except Exception as e:
        raise AssertionError(f"Warmup request failed: {e}")

    logger.info(f"All {len(instance_ids)} workers are ready")
    return sorted(instance_ids)


async def wait_for_indexer_workers_active(
    indexer_url: str,
    expected_workers: dict[int, dict[int, str]],
    timeout_s: float = 30.0,
) -> None:
    """Wait until the standalone indexer reports all ZMQ listeners as active."""
    if not expected_workers:
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    workers_url = f"{indexer_url}/workers"

    async with aiohttp.ClientSession() as session:
        while loop.time() < deadline:
            remaining_s = deadline - loop.time()
            if remaining_s <= 0:
                break

            try:
                request_timeout = aiohttp.ClientTimeout(total=min(2.0, remaining_s))
                async with session.get(workers_url, timeout=request_timeout) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(0.5)
                        continue
                    workers = await resp.json()
            except aiohttp.ClientError:
                await asyncio.sleep(0.5)
                continue

            workers_by_id = {
                worker["instance_id"]: worker
                for worker in workers
                if worker.get("source") == "zmq"
            }

            all_active = True
            for worker_id, endpoints in expected_workers.items():
                worker = workers_by_id.get(worker_id)
                if worker is None:
                    all_active = False
                    break

                listeners = worker.get("listeners", {})
                for dp_rank, endpoint in endpoints.items():
                    listener = listeners.get(str(dp_rank))
                    if listener is None:
                        all_active = False
                        break
                    if listener.get("endpoint") != endpoint:
                        all_active = False
                        break
                    if listener.get("status") != "active":
                        all_active = False
                        break

                if not all_active:
                    break

            if all_active:
                return

            await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Timed out waiting for indexer listeners to become active at {workers_url}"
    )


async def send_request_with_retry(url: str, payload: dict, max_retries: int = 8):
    """Send a single request with exponential backoff retry"""
    wait_time = 1  # Start with 1 second

    for attempt in range(max_retries + 1):
        await asyncio.sleep(wait_time)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        # Read the response to ensure it's valid
                        async for _ in response.content:
                            pass
                        logger.debug(
                            f"First request succeeded on attempt {attempt + 1}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed with status {response.status}"
                        )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < max_retries:
            wait_time *= 2  # Double the wait time

    return False


def get_runtime(store_backend="etcd", request_plane="tcp"):
    """Create a DistributedRuntime instance for testing.

    Args:
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: How frontend talks to backend ("tcp", "http" or "nats"). Defaults to "tcp".
    """
    try:
        # Try to get running loop (works in async context)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one (sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return DistributedRuntime(loop, store_backend, request_plane)


async def check_nats_consumers(namespace: str, expected_count: Optional[int] = None):
    """Check NATS consumers for the KV events stream.

    Args:
        namespace: The namespace to check consumers for
        expected_count: Optional expected number of consumers. If provided, asserts if count doesn't match.

    Returns:
        List of consumer names
    """
    component_subject = f"namespace.{namespace}.component.mocker"
    slugified = component_subject.lower().replace(".", "-").replace("_", "-")
    stream_name = f"{slugified}-kv-events"
    logger.info(f"Checking consumers for stream: {stream_name}")

    nc = await nats.connect(servers=_nats_server())
    try:
        js = nc.jetstream()
        consumer_infos = await js.consumers_info(stream_name)
        consumer_names = [info.name for info in consumer_infos]
        logger.info(f"Found {len(consumer_names)} consumers: {consumer_names}")

        # Log detailed consumer info
        for info in consumer_infos:
            logger.info(
                f"Consumer {info.name}: "
                f"num_pending={info.num_pending}, "
                f"num_ack_pending={info.num_ack_pending}, "
                f"ack_floor={info.ack_floor}, "
                f"delivered={info.delivered}"
            )

        if expected_count is not None:
            assert (
                len(consumer_names) == expected_count
            ), f"Expected {expected_count} durable consumers, found {len(consumer_names)}: {consumer_names}"
            logger.info(f"✓ Verified {expected_count} durable consumers exist")

        return consumer_names
    finally:
        await nc.close()


async def send_inflight_requests(urls: list, payload: dict, num_requests: int):
    """Send multiple requests concurrently, alternating between URLs if multiple provided"""

    # First, send test requests with retry to ensure all systems are ready
    for i, url in enumerate(urls):
        logger.info(f"Sending initial test request to URL {i} ({url}) with retry...")
        if not await send_request_with_retry(url, payload):
            raise RuntimeError(f"Failed to connect to URL {i} after multiple retries")

    async def send_single_request(session: aiohttp.ClientSession, request_id: int):
        # Alternate between URLs based on request_id
        url = urls[request_id % len(urls)]
        url_index = request_id % len(urls)

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(
                        f"Request {request_id} to URL {url_index} failed with status {response.status}"
                    )
                    return False

                # For streaming responses, read the entire stream
                chunks = []
                async for line in response.content:
                    if line:
                        chunks.append(line)

                logger.debug(
                    f"Request {request_id} to URL {url_index} completed with {len(chunks)} chunks"
                )
                return True

        except Exception as e:
            logger.error(
                f"Request {request_id} to URL {url_index} failed with error: {e}"
            )
            return False

    # Send all requests at once
    async with aiohttp.ClientSession() as session:
        tasks = [send_single_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r if r is True)
        failed = num_requests - successful

        logger.info(f"Completed all requests: {successful} successful, {failed} failed")

    assert (
        successful == num_requests
    ), f"Expected {num_requests} successful requests, got {successful}"
    logger.info(f"All {num_requests} requests completed successfully")


async def send_request_via_python_kv_router(
    kv_python_router: KvRouter,
    model_name: str,
    token_ids: list,
    initial_wait: float,
    max_retries: int,
    stop_conditions: Optional[dict] = None,
    sampling_options: Optional[dict] = None,
    output_options: Optional[dict] = None,
    router_config_override: Optional[dict] = None,
    worker_id: Optional[
        int
    ] = None,  # If None, Router will select the best available worker
    dp_rank: Optional[int] = None,  # Data parallel rank (defaults to 0)
    return_worker_ids: bool = False,  # If True, return worker IDs from response
) -> bool | dict[str, Optional[int]]:
    """Send a request to the specified worker instance.

    Args:
        return_worker_ids: If True, returns a dict with prefill_worker_id and decode_worker_id.
                          If False, returns True on success or False on failure.

    Returns:
        If return_worker_ids=False: True if workers respond, otherwise raises or returns False.
        If return_worker_ids=True: Dict with 'prefill_worker_id' and 'decode_worker_id' keys.
    """

    wait_time = initial_wait

    log_message = (
        f"worker with worker_id={worker_id}"
        if worker_id is not None
        else "the best available worker"
    )

    # Retry loop sending request to worker with exponential backoff
    stream = None
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Sending request to {log_message} (attempt {attempt + 1})")

            stream = await kv_python_router.generate(
                token_ids=token_ids,
                model=model_name,
                stop_conditions=stop_conditions,  # type: ignore[arg-type]
                sampling_options=sampling_options,  # type: ignore[arg-type]
                output_options=output_options,  # type: ignore[arg-type]
                router_config_override=router_config_override,  # type: ignore[arg-type]
                worker_id=worker_id,
                dp_rank=dp_rank,
            )

            if stream is not None:
                logger.debug(f"Request succeeded on attempt {attempt + 1}")
                break

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries:
                await asyncio.sleep(wait_time)
                wait_time *= 2
            else:
                raise RuntimeError(
                    f"Failed to connect to workers after {max_retries + 1} attempts"
                ) from e

    if stream is None:
        raise RuntimeError(
            f"Failed to get a valid stream from workers after {max_retries + 1} attempts"
        )

    # Collect tokens and worker IDs from the SSE stream
    generated_tokens = []
    prefill_worker_id: Optional[int] = None
    decode_worker_id: Optional[int] = None
    prefill_dp_rank: Optional[int] = None
    decode_dp_rank: Optional[int] = None

    async for response in stream:
        if isinstance(response, dict):
            # Check if response has token_ids
            if "token_ids" in response:
                tokens = response["token_ids"]
                if isinstance(tokens, list):
                    generated_tokens.extend(tokens)
                    logger.debug(f"Received {len(tokens)} tokens: {tokens}")

            # Check for finish reason
            if "finish_reason" in response:
                logger.debug(
                    f"Stream finished with reason: {response['finish_reason']}"
                )

            # Extract worker IDs and dp_ranks from disaggregated_params if present
            if return_worker_ids and "disaggregated_params" in response:
                disagg_params = response["disaggregated_params"]
                if isinstance(disagg_params, dict) and "worker_id" in disagg_params:
                    worker_id_info = disagg_params["worker_id"]
                    if isinstance(worker_id_info, dict):
                        if "prefill_worker_id" in worker_id_info:
                            prefill_worker_id = worker_id_info["prefill_worker_id"]
                        if "decode_worker_id" in worker_id_info:
                            decode_worker_id = worker_id_info["decode_worker_id"]
                        if "prefill_dp_rank" in worker_id_info:
                            prefill_dp_rank = worker_id_info["prefill_dp_rank"]
                        if "decode_dp_rank" in worker_id_info:
                            decode_dp_rank = worker_id_info["decode_dp_rank"]

    # Verify if expected number of tokens are generated if max_tokens specified and ignore_eos is True
    logger.debug(f"Total generated tokens: {len(generated_tokens)}")
    if (
        stop_conditions
        and "max_tokens" in stop_conditions
        and "ignore_eos" in stop_conditions
        and stop_conditions["ignore_eos"]
    ):
        max_tokens = int(stop_conditions["max_tokens"])
        assert len(generated_tokens) == max_tokens, (
            f"Expected exactly {max_tokens} tokens but got {len(generated_tokens)}. "
            f"Tokens: {generated_tokens}"
        )

        logger.debug(
            f"Successfully verified {max_tokens} tokens generated as expected via KvRouter with ignore_eos=True"
        )

    if return_worker_ids:
        return {
            "prefill_worker_id": prefill_worker_id,
            "decode_worker_id": decode_worker_id,
            "prefill_dp_rank": prefill_dp_rank,
            "decode_dp_rank": decode_dp_rank,
        }

    return True
