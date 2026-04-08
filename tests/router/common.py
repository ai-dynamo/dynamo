# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import logging
import os
import random
from typing import TYPE_CHECKING, Any, Optional

import aiohttp
import nats
import requests

from dynamo.llm import AicPerfConfig, KvRouter, KvRouterConfig
from dynamo.prometheus_names import frontend_service, name_prefix
from tests.router.helper import (
    _nats_server,
    assert_event_dumps_equal,
    get_runtime,
    send_inflight_requests,
    send_request_via_python_kv_router,
    send_request_with_retry,
    verify_response_timing,
    wait_for_frontend_ready,
    wait_for_indexer_workers_active,
    wait_for_workers_ready,
)
from tests.router.router_process import (
    DirectRouterProcess,
    FrontendRouterProcess,
    KVRouterProcess,
)

if TYPE_CHECKING:
    from tests.conftest import NatsServer

logger = logging.getLogger(__name__)

NUM_REQUESTS = 100
BLOCK_SIZE = 16
MIN_INITIAL_WORKERS_ENV = "DYN_ROUTER_MIN_INITIAL_WORKERS"


@contextlib.contextmanager
def min_initial_workers_env(min_initial_workers: int):
    previous = os.environ.get(MIN_INITIAL_WORKERS_ENV)
    os.environ[MIN_INITIAL_WORKERS_ENV] = str(min_initial_workers)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MIN_INITIAL_WORKERS_ENV, None)
        else:
            os.environ[MIN_INITIAL_WORKERS_ENV] = previous


########################################################
# Test templates
########################################################


def _test_router_basic(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    num_requests: int,
    frontend_timeout: int = 120,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    router_mode: str = "kv",
    enforce_disagg: bool = False,
    min_initial_workers: int | None = None,
):
    """Basic router test: start router, wait for workers and send concurrent requests via HTTP frontend.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This is a shared test implementation for both mocker and vLLM workers.
    Always waits for workers to be properly registered before sending requests to avoid flakiness.

    Supports any router_mode (defaults to "kv" for existing callers).
    block_size is only sent to the frontend CLI when router_mode is "kv".

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port to start the frontend HTTP server on
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        frontend_timeout: Timeout for frontend readiness check (default: 120s)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats", "tcp", or "http"). Defaults to "nats".
        router_mode: Router mode ("kv", "round-robin", "random", "power-of-two", "direct"). Defaults to "kv".
        enforce_disagg: Whether to pass --enforce-disagg to the frontend. Defaults to False.
        min_initial_workers: Optional frontend startup worker gate. Defaults to None.

    Raises:
        AssertionError: If requests fail or frontend doesn't become ready
        TimeoutError: If frontend doesn't become ready within timeout
    """
    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        engine_workers.namespace,
        store_backend,
        enforce_disagg=enforce_disagg,
        request_plane=request_plane,
        router_mode=router_mode,
        min_initial_workers=min_initial_workers,
    ):
        # Start router frontend
        logger.info(
            f"Starting frontend --router-mode {router_mode} on port {frontend_port}"
        )

        frontend_url = f"http://localhost:{frontend_port}"

        # Always wait for workers to register with frontend to avoid flakiness
        logger.info("Waiting for workers to register with frontend...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=frontend_timeout,
            )
        )

        # Send concurrent requests to the frontend
        logger.info(f"Sending {num_requests} concurrent requests to frontend...")
        asyncio.run(
            send_inflight_requests(
                [f"{frontend_url}/v1/chat/completions"],
                test_payload,
                num_requests,
            )
        )

        logger.info(f"Successfully completed {num_requests} requests")


def _test_router_two_routers(
    engine_workers,
    block_size: int,
    request,
    router_ports: list[int],
    test_payload: dict,
    num_requests: int,
    store_backend: str = "etcd",
    skip_consumer_verification: bool = False,
):
    """Test two KV routers with alternating requests and consumer lifecycle verification.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This test:
    1. Starts two KV routers on different ports
    2. Sends requests alternating between the two routers
    3. Verifies that both routers create durable consumers (unless skipped)
    4. Verifies consumers are cleaned up when routers exit (unless skipped)

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        router_ports: List of two port numbers for the routers (e.g., [8091, 8092])
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        skip_consumer_verification: Skip JetStream consumer verification (for NATS Core mode).

    Raises:
        AssertionError: If consumer lifecycle verification fails
    """
    kv_routers = []

    try:
        # Start two KV routers on different ports
        for i, port in enumerate(router_ports):
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(
                request,
                block_size,
                port,
                engine_workers.namespace,
                store_backend,
                min_initial_workers=engine_workers.num_workers,
            )
            kv_router.__enter__()
            kv_routers.append(kv_router)

        # Wait for workers to be ready on both routers
        logger.info("Waiting for workers to register with both routers...")
        for i, port in enumerate(router_ports):
            frontend_url = f"http://localhost:{port}"
            logger.info(f"Waiting for router {i} on port {port} to discover workers...")
            asyncio.run(
                wait_for_frontend_ready(
                    frontend_url=frontend_url,
                    expected_num_workers=engine_workers.num_workers,
                    timeout=120,
                )
            )
        logger.info("Both routers have discovered workers")

        # Build URLs for both routers
        router_urls = [
            f"http://localhost:{port}/v1/chat/completions" for port in router_ports
        ]

        # Send requests concurrently, alternating between routers
        asyncio.run(
            send_inflight_requests(
                router_urls,
                test_payload,
                num_requests,
            )
        )

        logger.info(
            f"Successfully completed {num_requests} requests across {len(router_ports)} routers"
        )

        # Verify durable consumers lifecycle
        async def verify_consumer_lifecycle():
            logger.info("Verifying durable consumers lifecycle")

            # Construct the stream name from the workers namespace
            component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
            slugified = component_subject.lower().replace(".", "-").replace("_", "-")
            stream_name = f"{slugified}-kv-events"

            logger.info(f"Checking consumers for stream: {stream_name}")

            # Connect to NATS and list consumers
            nc = await nats.connect(servers=_nats_server())
            try:
                js = nc.jetstream()

                # List consumers - should have 2 (one for each router process)
                consumer_infos = await js.consumers_info(stream_name)
                consumer_names = [info.name for info in consumer_infos]
                logger.info(f"Found {len(consumer_names)} consumers: {consumer_names}")

                assert (
                    len(consumer_names) == 2
                ), f"Expected 2 durable consumers (one per router), found {len(consumer_names)}: {consumer_names}"
                logger.info("✓ Verified 2 durable consumers exist (one per router)")

                # Kill the first router process
                logger.info(f"Killing first router on port {router_ports[0]}")
                kv_routers[0].__exit__(None, None, None)

                # Poll until one consumer remains (up to 5s)
                for _ in range(25):
                    consumer_infos = await js.consumers_info(stream_name)
                    if len(list(consumer_infos)) == 1:
                        break
                    await asyncio.sleep(0.2)

                # Verify only 1 consumer remains
                consumer_names = [info.name for info in consumer_infos]
                logger.info(
                    f"After killing router1, found {len(consumer_names)} consumers: {consumer_names}"
                )

                assert (
                    len(consumer_names) == 1
                ), f"Expected 1 durable consumer after killing router1, found {len(consumer_names)}: {consumer_names}"
                logger.info(
                    "✓ Verified 1 durable consumer remains after killing first router"
                )

                # Kill the second router process
                logger.info(f"Killing second router on port {router_ports[1]}")
                kv_routers[1].__exit__(None, None, None)

                # Poll until no consumers remain (up to 5s)
                for _ in range(25):
                    consumer_infos = await js.consumers_info(stream_name)
                    if len(list(consumer_infos)) == 0:
                        break
                    await asyncio.sleep(0.2)

                consumer_names = [info.name for info in consumer_infos]
                logger.info(
                    f"After killing router2, found {len(consumer_names)} consumers: {consumer_names}"
                )

                assert (
                    len(consumer_names) == 0
                ), f"Expected 0 durable consumers after killing both routers, found {len(consumer_names)}: {consumer_names}"
                logger.info(
                    "✓ Verified 0 durable consumers remain after killing both routers"
                )

            finally:
                await nc.close()

        # Run consumer lifecycle verification (skip for NATS Core mode)
        if skip_consumer_verification:
            logger.info("Skipping JetStream consumer verification (NATS Core mode)")
            # Clean up routers manually since we're not doing consumer verification
            for kv_router in kv_routers:
                kv_router.__exit__(None, None, None)
        else:
            asyncio.run(verify_consumer_lifecycle())

        # Clear the kv_routers list since we've already cleaned them up
        kv_routers = []

    finally:
        # Clean up any remaining routers (in case of error before consumer verification)
        for kv_router in kv_routers:
            kv_router.__exit__(None, None, None)


def _test_remote_indexer_decisions(
    engine_workers,
    model_name: str,
    block_size: int = 8,
    use_kv_events: bool = True,
    test_dp_rank: bool = True,
    request_plane: str = "nats",
    store_backend: str = "etcd",
):
    """Validate remote-indexer-backed routing decisions using direct KvRouter instances."""

    async def wait_for_worker_ids(endpoint, expected_num_workers: int) -> list[int]:
        client = await endpoint.client()

        for _ in range(120):
            worker_ids = sorted(set(client.instance_ids()))
            if len(worker_ids) >= expected_num_workers:
                return worker_ids
            await asyncio.sleep(1)

        raise TimeoutError("Timed out waiting for backend worker IDs")

    async def wait_for_served_indexer(
        runtime,
        expected_query_instances: int,
        expected_record_instances: int,
    ) -> None:
        query_endpoint = runtime.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.kv_indexer_query"
        )
        query_client = await query_endpoint.client()
        record_endpoint = runtime.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.kv_indexer_record_routing_decision"
        )
        record_client = await record_endpoint.client()

        for _ in range(120):
            query_ids = set(query_client.instance_ids())
            record_ids = set(record_client.instance_ids())

            if use_kv_events:
                if len(query_ids) >= expected_query_instances and len(record_ids) == 0:
                    return
            elif (
                len(query_ids) == expected_query_instances
                and len(record_ids) == expected_record_instances
                and query_ids == record_ids
            ):
                return

            await asyncio.sleep(0.5)

        raise TimeoutError("Timed out waiting for served indexer endpoints to register")

    async def test_sync():
        endpoint_path = (
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )
        expected_num_instances = engine_workers.num_workers

        async def make_router(*, serve_indexer: bool, use_remote_indexer: bool):
            kv_router_config = KvRouterConfig(
                router_snapshot_threshold=20,
                use_kv_events=use_kv_events,
                router_track_prefill_tokens=True,
                serve_indexer=serve_indexer,
                use_remote_indexer=use_remote_indexer,
            )
            last_error: Exception | None = None
            for _ in range(60):
                runtime = get_runtime(
                    store_backend=store_backend, request_plane=request_plane
                )
                endpoint = runtime.endpoint(endpoint_path)
                try:
                    with min_initial_workers_env(expected_num_instances):
                        kv_router = KvRouter(
                            endpoint=endpoint,
                            block_size=block_size,
                            kv_router_config=kv_router_config,
                        )
                    return runtime, endpoint, kv_router
                except Exception as error:
                    last_error = error
                    if not (serve_indexer or use_remote_indexer):
                        raise
                    del endpoint
                    del runtime
                    await asyncio.sleep(1.0)

            raise AssertionError(
                "Timed out waiting for model discovery before creating remote-indexer router"
            ) from last_error

        serving_runtimes = []
        serving_endpoints = []
        serving_routers = []

        runtime_a, endpoint_a, router_a = await make_router(
            serve_indexer=True, use_remote_indexer=False
        )
        serving_runtimes.append(runtime_a)
        serving_endpoints.append(endpoint_a)
        serving_routers.append(router_a)

        if use_kv_events:
            runtime_b, endpoint_b, router_b = await make_router(
                serve_indexer=True, use_remote_indexer=False
            )
            serving_runtimes.append(runtime_b)
            serving_endpoints.append(endpoint_b)
            serving_routers.append(router_b)

        await wait_for_served_indexer(
            serving_runtimes[0],
            expected_query_instances=len(serving_routers),
            expected_record_instances=0 if use_kv_events else 1,
        )

        _, consumer_endpoint, consumer_router = await make_router(
            serve_indexer=False, use_remote_indexer=True
        )

        worker_ids = await wait_for_worker_ids(
            serving_endpoints[0], expected_num_instances
        )
        if len(worker_ids) >= 2:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[1]
        elif len(worker_ids) == 1 and test_dp_rank:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[0]
        else:
            raise AssertionError(
                f"Need at least 2 routing targets but got {len(worker_ids)} worker(s) "
                f"with test_dp_rank={test_dp_rank}"
            )

        dp_rank_a = 0 if test_dp_rank else None
        dp_rank_b = 1 if test_dp_rank else None
        logger.info(
            "Remote-indexer routing targets: worker_a=%s/%s worker_b=%s/%s",
            worker_a_id,
            dp_rank_a,
            worker_b_id,
            dp_rank_b,
        )

        blocks = [
            [random.randint(1, 10000) for _ in range(block_size)] for _ in range(7)
        ]
        A, B, C, D, E, F, G = blocks
        request_specs = [
            (serving_routers[0], A + B, worker_a_id, dp_rank_a, 0.1),
            (serving_routers[0], A + C + D, worker_a_id, dp_rank_a, 0.1),
            (serving_routers[-1], A + C + E, worker_b_id, dp_rank_b, 2.0),
            (consumer_router, A + C + D + F, None, None, 2.0),
            (consumer_router, A + C + G, None, None, 2.0),
        ]

        responses: list[dict[str, Optional[int]]] = []
        for i, (
            kv_router,
            token_ids,
            forced_worker_id,
            forced_dp_rank,
            sleep_after,
        ) in enumerate(request_specs, start=1):
            logger.info(
                "Sending remote-indexer request %s/5%s%s",
                i,
                (
                    f" forced_worker_id={forced_worker_id}"
                    if forced_worker_id is not None
                    else ""
                ),
                (
                    f" forced_dp_rank={forced_dp_rank}"
                    if forced_dp_rank is not None
                    else ""
                ),
            )
            result = await send_request_via_python_kv_router(
                kv_python_router=kv_router,
                model_name=model_name,
                token_ids=token_ids,
                initial_wait=1.0,
                max_retries=8,
                stop_conditions={
                    "ignore_eos": True,
                    "max_tokens": 2,
                },
                worker_id=forced_worker_id,
                dp_rank=forced_dp_rank,
                return_worker_ids=True,
            )
            assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
            responses.append(result)
            if sleep_after > 0:
                await asyncio.sleep(sleep_after)

        req4 = responses[3]
        assert req4["prefill_worker_id"] == worker_a_id, (
            f"Request 4: expected prefill_worker_id={worker_a_id} (longest prefix match), "
            f"got {req4['prefill_worker_id']}"
        )
        if test_dp_rank:
            assert req4["prefill_dp_rank"] == dp_rank_a, (
                f"Request 4: expected prefill_dp_rank={dp_rank_a} "
                f"(longest prefix match), got {req4['prefill_dp_rank']}"
            )

        req5 = responses[4]
        assert req5["prefill_worker_id"] == worker_b_id, (
            f"Request 5: expected prefill_worker_id={worker_b_id} (tiebreak by smaller tree), "
            f"got {req5['prefill_worker_id']}"
        )
        if test_dp_rank:
            assert req5["prefill_dp_rank"] == dp_rank_b, (
                f"Request 5: expected prefill_dp_rank={dp_rank_b} "
                f"(tiebreak by smaller tree), got {req5['prefill_dp_rank']}"
            )

        await wait_for_worker_ids(consumer_endpoint, expected_num_instances)

    asyncio.run(test_sync())


def _test_python_router_bindings(
    engine_workers,
    endpoint,
    block_size: int,
    model_name: str,
    num_workers: int,
):
    """Test KvRouter Python bindings with token streaming and config overrides.

    Assumes engine_workers are already initialized. This test creates a KvRouter
    Python object and sends three test requests to verify:
    1. Token streaming with full router config overrides (overlap_score_weight, router_temperature)
    2. Token streaming without any overrides (uses default config)
    3. Token streaming with partial override (only router_temperature)

    All requests use ignore_eos=True with varying max_tokens to test token generation control.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        endpoint: Dynamo endpoint for the workers
        block_size: Block size for KV cache
        model_name: Model name to use for requests
        num_workers: Expected number of workers

    Raises:
        AssertionError: If requests fail or router doesn't work correctly
    """
    # Create KvRouterConfig with default settings
    kv_router_config = KvRouterConfig()

    # Create KvRouter Python object
    with min_initial_workers_env(num_workers):
        kv_router = KvRouter(
            endpoint=endpoint,
            block_size=block_size,
            kv_router_config=kv_router_config,
        )

    logger.info("Created KvRouter Python object")

    # Wait for workers to be ready
    asyncio.run(wait_for_workers_ready(endpoint, kv_router, num_workers, model_name))

    # Generate random token IDs (100 to 200 tokens)
    num_input_tokens = random.randint(100, 200)
    token_ids = [random.randint(1, 10000) for _ in range(num_input_tokens)]

    # Set up override parameters
    router_config_override = {
        "overlap_score_weight": 0.5,  # Override the default weight
        "router_temperature": 0.5,  # Override the default temperature
    }

    logger.info(f"Generated {num_input_tokens} random token IDs")

    # Test with full overrides
    logger.info(f"Testing with full router config overrides: {router_config_override}")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids,
            initial_wait=1.0,
            max_retries=8,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 20,  # Generate exactly 20 tokens
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            router_config_override=router_config_override,
        )
    )

    # Test without overrides
    logger.info("Testing without router config overrides")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids[:50],  # Use fewer tokens for second test,
            initial_wait=1.0,
            max_retries=8,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 10,  # Generate exactly 10 tokens for the second test
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            # No router_config_override this time
        )
    )

    # Test with partial override (only temperature)
    partial_override = {"router_temperature": 0.1}
    logger.info(f"Testing with partial router config overrides: {partial_override}")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids[:30],  # Use fewer tokens for third test,
            initial_wait=1.0,
            max_retries=8,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 5,  # Generate exactly 5 tokens for the third test
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            router_config_override=partial_override,
        )
    )

    logger.info("KvRouter bindings test completed successfully")


def _test_router_query_instance_id(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
):
    """Test query_instance_id annotation returns worker_instance_id and token_data without routing.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This tests the early return optimization where a request with 'nvext.annotations': ['query_instance_id']
    receives metadata without waiting for model generation. The router should:
    1. NOT route the request to a worker for generation
    2. Return worker_instance_id as an SSE event (which worker would handle it)
    3. Return token_data as an SSE event (the tokenized input)
    4. Terminate the stream with [DONE]

    This is useful for clients that want to know which worker will handle a request before
    committing to the full generation (e.g., for request routing decisions).

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If annotation response structure is incorrect or contains generation content
    """

    with KVRouterProcess(
        request, block_size, frontend_port, engine_workers.namespace, store_backend
    ):
        # Start KV router (frontend)
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Send a warming request first to ensure system is ready
        logger.info("Sending warming request without annotations...")
        asyncio.run(send_request_with_retry(url, test_payload))

        # Test payload with query_instance_id annotation
        # Format: "query_instance_id:" (colon with empty value) for GAIE aggregated mode
        annotated_payload = {
            **test_payload,
            "nvext": {"annotations": ["query_instance_id:"]},
        }

        async def test_annotation_response():
            """Send request with query_instance_id and validate response structure"""
            async with aiohttp.ClientSession() as session:
                logger.info("Sending request with query_instance_id annotation...")

                async with session.post(url, json=annotated_payload) as response:
                    assert (
                        response.status == 200
                    ), f"Expected 200 but got {response.status}"

                    # Collect all response chunks
                    response_chunks = []
                    async for chunk in response.content:
                        if chunk:
                            chunk_str = chunk.decode("utf-8", errors="replace")
                            response_chunks.append(chunk_str)

                    full_response = "".join(response_chunks)
                    logger.info(
                        f"Full SSE response ({len(full_response)} bytes):\n{full_response}"
                    )

                    # Parse the SSE response to extract the first chunk with nvext data
                    # New format: nvext contains worker_id and token_ids
                    sse_parts = full_response.split("\n\n")
                    worker_id_info = None
                    token_list = None

                    for part in sse_parts:
                        part = part.strip()
                        if not part or not part.startswith("data:"):
                            continue

                        data_str = part.split("data:", 1)[1].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                            logger.info(f"Parsed chunk: {json.dumps(chunk, indent=2)}")

                            # Extract nvext data containing worker_id and token_ids
                            nvext = chunk.get("nvext", {})
                            if nvext:
                                if "worker_id" in nvext:
                                    worker_id_info = nvext["worker_id"]
                                    logger.info(
                                        f"Found worker_id info: {worker_id_info}"
                                    )
                                if "token_ids" in nvext:
                                    token_list = nvext["token_ids"]
                                    logger.info(
                                        f"Found token_ids: {len(token_list)} tokens"
                                    )
                        except json.JSONDecodeError:
                            continue

                    # Validate worker_id info
                    assert (
                        worker_id_info is not None
                    ), f"Missing worker_id in nvext. Response: {full_response}"

                    # For aggregated mode, both prefill and decode should be the same
                    prefill_worker_id = worker_id_info.get("prefill_worker_id")
                    decode_worker_id = worker_id_info.get("decode_worker_id")
                    assert (
                        prefill_worker_id is not None
                    ), f"Missing prefill_worker_id in worker_id: {worker_id_info}"
                    assert (
                        decode_worker_id is not None
                    ), f"Missing decode_worker_id in worker_id: {worker_id_info}"
                    assert (
                        prefill_worker_id == decode_worker_id
                    ), f"For aggregated mode, prefill and decode worker should be same: {worker_id_info}"

                    # Validate token_ids
                    assert (
                        token_list is not None
                    ), f"Missing token_ids in nvext. Response: {full_response}"
                    assert isinstance(
                        token_list, list
                    ), f"token_ids should be a list, got: {type(token_list)}"
                    assert (
                        len(token_list) > 0
                    ), f"token_ids should not be empty: {token_list}"
                    assert all(
                        isinstance(token, int) for token in token_list
                    ), f"All tokens should be integers: {token_list}"

                    logger.info(
                        f"Valid token_ids with {len(token_list)} tokens: {token_list[:10]}{'...' if len(token_list) > 10 else ''}"
                    )

                    return {
                        "prefill_worker_id": prefill_worker_id,
                        "decode_worker_id": decode_worker_id,
                        "token_count": len(token_list),
                        "tokens": token_list,
                    }

        result = asyncio.run(test_annotation_response())

        logger.info("Successfully validated query_instance_id annotation response:")
        logger.info(f"Prefill Worker ID: {result['prefill_worker_id']}")
        logger.info(f"Decode Worker ID: {result['decode_worker_id']}")
        logger.info(f"Token count: {result['token_count']}")


def _parse_frontend_rejection_metric(
    metrics_text: str, model_name: str, endpoint: str
) -> int:
    """Parse frontend model_rejection_total from Prometheus metrics text.

    Args:
        metrics_text: Raw Prometheus metrics text
        model_name: The model name label value
        endpoint: The endpoint label value (e.g. "chat_completions")

    Returns:
        The metric count, or 0 if not found
    """
    metric_name = f"{name_prefix.FRONTEND}_{frontend_service.MODEL_REJECTION_TOTAL}"
    for line in metrics_text.splitlines():
        if not line.startswith(f"{metric_name}{{"):
            continue
        if f'model="{model_name}"' in line and f'endpoint="{endpoint}"' in line:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    return int(float(parts[1]))
                except ValueError:
                    pass
    return 0


def _verify_frontend_rejection_metrics(
    frontend_port: int,
    model_name: str,
    endpoint: str,
    expected_count: int,
) -> None:
    """Verify frontend rejection metrics by scraping the /metrics endpoint.

    Args:
        frontend_port: Port where the frontend /metrics is served
        model_name: The model name label value
        endpoint: The endpoint label value (e.g. "chat_completions")
        expected_count: Expected rejection count to match exactly
    """
    metrics_url = f"http://localhost:{frontend_port}/metrics"
    try:
        metrics_response = requests.get(metrics_url, timeout=5)
        metrics_response.raise_for_status()
    except requests.RequestException as e:
        raise AssertionError(
            f"Failed to fetch frontend metrics from {metrics_url}: {e}"
        ) from e

    metric_count = _parse_frontend_rejection_metric(
        metrics_response.text, model_name, endpoint
    )
    logger.info(f"Frontend rejection metric: model_rejection_total={metric_count}")
    assert metric_count == expected_count, (
        f"Frontend model_rejection_total ({metric_count}) does not match "
        f"expected count ({expected_count})"
    )


def _test_router_overload_503(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    blocks_threshold: float = 0.2,
):
    """Test that 503 is returned when all workers are busy, and verify rejection metrics.

    Assumes engine_workers are already initialized. This function manages router lifecycle.
    Uses limited resources to intentionally trigger the overload condition.

    Sends staggered requests (0.1s apart) to exhaust worker resources, then verifies:
    1. At least one request succeeds (routed before busy state propagates)
    2. At least one request is rejected with 503 (worker busy)
    3. The frontend model_rejection_total metric matches the observed 503 count

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache (should be small to exhaust quickly, e.g. 4)
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        blocks_threshold: Active decode blocks threshold for the router (default 0.2)

    Raises:
        AssertionError: If success/rejection counts or metrics don't meet expectations
    """
    logger.info(
        f"Starting KV router frontend on port {frontend_port} with limited resources"
    )

    with KVRouterProcess(
        request=request,
        block_size=block_size,
        frontend_port=frontend_port,
        namespace=engine_workers.namespace,
        blocks_threshold=blocks_threshold,
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Custom payload for 503 test with more tokens to consume resources
        test_payload_503 = {
            **test_payload,
            "max_tokens": 50,  # Longer output to consume more blocks
        }

        logger.info("Waiting for frontend readiness before overload test...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=1,
                timeout=60,
            )
        )

        logger.info("Launching streaming requests until the router returns 503...")

        async def exhaust_resources_and_verify_503():
            stop_event = asyncio.Event()

            async with aiohttp.ClientSession() as session:
                tasks = []

                async def send_request(req_id, payload):
                    try:
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                logger.info(f"Request {req_id} accepted")
                                await stop_event.wait()
                                return response.status

                            if response.status == 503:
                                body = await response.json()
                                logger.info(
                                    f"Request {req_id} got expected 503: {body}"
                                )
                                stop_event.set()
                                error_msg = body.get("message", "")
                                assert (
                                    "Service temporarily unavailable" in error_msg
                                    or "All workers are busy" in error_msg
                                ), f"Expected service overload error message, got: {body}"
                                return response.status

                            body = await response.text()
                            logger.info(
                                f"Request {req_id} got unexpected status {response.status}: {body}"
                            )
                            return response.status
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.info(f"Request {req_id} failed: {e}")
                        raise

                try:
                    for i in range(50):
                        if stop_event.is_set():
                            break

                        content_words = test_payload["messages"][0]["content"].split()
                        random.shuffle(content_words)
                        shuffled_content = " ".join(content_words)
                        unique_payload = {
                            **test_payload_503,
                            "messages": [
                                {
                                    **test_payload["messages"][0],
                                    "content": shuffled_content,
                                }
                            ],
                        }
                        tasks.append(
                            asyncio.create_task(send_request(i, unique_payload))
                        )
                        await asyncio.sleep(0.1)

                    if not stop_event.is_set():
                        try:
                            await asyncio.wait_for(stop_event.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            logger.error("Timed out waiting for overload 503")
                finally:
                    stop_event.set()
                    done, pending = await asyncio.wait(tasks, timeout=3)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                return [t.result() for t in done]

        results = asyncio.run(exhaust_resources_and_verify_503())

        # Count outcomes
        num_succeeded = sum(1 for s in results if s == 200)
        num_rejected = sum(1 for s in results if s == 503)
        num_other = sum(1 for s in results if s not in (200, 503))

        logger.info(
            f"Results: {num_succeeded} succeeded, {num_rejected} rejected (503), "
            f"{num_other} other"
        )

        # Assert minimum thresholds
        assert (
            num_other == 0
        ), f"Expected only 200 or 503 responses, but got {num_other} other"
        assert (
            num_rejected > 0
        ), f"Expected at least 1 rejection, but got {num_rejected}"
        assert (
            num_succeeded > 0
        ), f"Expected at least 1 success, but got {num_succeeded}"

        # Verify rejection metrics from frontend /metrics endpoint
        model_name = test_payload.get("model", "")
        _verify_frontend_rejection_metrics(
            frontend_port, model_name, "chat_completions", num_rejected
        )

        logger.info(
            f"Successfully verified overload 503: {num_rejected} rejected, "
            f"{num_succeeded} succeeded, metrics match"
        )


async def _zmq_replay_cycle(
    phase: int,
    router,
    router_name: str,
    endpoint,
    indexer_url: str,
    engine_workers,
    send_requests_to_router,
):
    """Pause indexer listeners → send gap requests → resume → send to trigger replay."""
    await asyncio.sleep(1)
    worker_ids = list(engine_workers.worker_id_to_zmq_ports.keys())
    dp_size = getattr(engine_workers, "dp_size", None) or 1

    logger.info(f"=== ZMQ REPLAY TEST: Phase {phase} ({router_name}) ===")
    async with aiohttp.ClientSession() as session:
        for wid in worker_ids:
            for dp_rank in range(dp_size):
                async with session.post(
                    f"{indexer_url}/test/pause_listener",
                    json={"instance_id": wid, "dp_rank": dp_rank},
                ) as resp:
                    assert (
                        resp.status == 200
                    ), f"Pause {wid}:{dp_rank} failed: {await resp.text()}"

    logger.info("Sending 10 requests while indexer listeners are paused")
    successful_gap = await send_requests_to_router(
        router, 10, f"{router_name} (indexer paused)", endpoint
    )
    assert (
        successful_gap == 10
    ), f"Expected 10 requests while paused, got {successful_gap}"

    async with aiohttp.ClientSession() as session:
        for wid in worker_ids:
            for dp_rank in range(dp_size):
                async with session.post(
                    f"{indexer_url}/test/resume_listener",
                    json={"instance_id": wid, "dp_rank": dp_rank},
                ) as resp:
                    assert (
                        resp.status == 200
                    ), f"Resume {wid}:{dp_rank} failed: {await resp.text()}"

    logger.info("Sending 5 requests after resume (triggers gap detection + replay)")
    successful_post = await send_requests_to_router(
        router, 5, f"{router_name} (post-resume)", endpoint
    )
    assert (
        successful_post == 5
    ), f"Expected 5 requests post-resume, got {successful_post}"
    await asyncio.sleep(2)


def _test_router_indexers_sync(
    engine_workers,
    block_size: int,
    model_name: str,
    num_workers: int,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    test_nats_interruption: bool = False,
    nats_server: Optional["NatsServer"] = None,
    durable_kv_events: bool = False,
    router_event_threads: int = 4,
    standalone_indexer_url: Optional[str] = None,
    standalone_indexer_b_url: Optional[str] = None,
    test_zmq_replay: bool = False,
):
    """Test that two KV routers have synchronized indexer states after processing requests.

    Assumes engine_workers are already initialized. This test:
    1. Creates first KvRouter (with its own runtime) and sends 25 requests (triggers snapshot at threshold=20)
    2. Creates second KvRouter (with its own runtime, should sync from NATS snapshot)
    3. Sends 25 requests to second router
    4. Verifies NATS object store contains the snapshot
    5. Dumps states from both routers and compares them (should be identical)

    This validates that the snapshot mechanism works and routers can sync state from NATS.

    When test_nats_interruption=True (requires nats_server and request_plane="tcp"):
    - After first router sends 25 requests, NATS is stopped
    - 10 more requests sent while NATS is down (stored locally by local indexer)
    - NATS restarted (fresh state), recovery mechanism re-syncs
    - Second router starts and sends 25 requests
    - NATS stopped again, 10 more requests sent
    - NATS restarted, 5 more requests sent
    - Verify both routers converge to same state

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        block_size: Block size for KV cache
        model_name: Model name to use for requests
        num_workers: Expected number of workers
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats" or "tcp"). Defaults to "nats".
        test_nats_interruption: If True, test NATS interruption recovery. Defaults to False.
        nats_server: NatsServer instance for stop/start (required if test_nats_interruption=True).
        durable_kv_events: If True, use durable KV events (JetStream). Defaults to False.

    Raises:
        AssertionError: If router states don't synchronize correctly or snapshot is missing
    """
    if test_nats_interruption and nats_server is None:
        raise ValueError("nats_server is required when test_nats_interruption=True")

    # Use async to manage the test flow
    async def test_sync():
        # Create KvRouterConfig with lower snapshot threshold for testing
        kv_router_config = KvRouterConfig(
            router_snapshot_threshold=20,
            durable_kv_events=durable_kv_events,
            router_event_threads=router_event_threads,
        )

        # If standalone indexer mode, launch workers one-by-one and register.
        # We need to create a temporary endpoint just to discover worker IDs.
        if standalone_indexer_url:
            tmp_runtime = get_runtime(store_backend, request_plane)
            tmp_endpoint = tmp_runtime.endpoint(
                f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
            )
            await engine_workers.launch_workers_with_indexer(tmp_endpoint)

        async def send_requests_to_router(router, num_requests, router_name, endpoint):
            # Now send the actual requests
            tasks = []
            for i in range(num_requests):
                # Generate random token IDs for each request
                logger.debug(f"Sending request {i + 1}/{num_requests} to {router_name}")

                # Generate 30 random tokens
                request_tokens = [random.randint(1, 10000) for _ in range(30)]

                # Send request to mocker via the router
                tasks.append(
                    asyncio.create_task(
                        send_request_via_python_kv_router(
                            kv_python_router=router,
                            model_name=model_name,
                            token_ids=request_tokens,
                            initial_wait=1.0,
                            max_retries=8,
                            stop_conditions={
                                "ignore_eos": True,  # Don't stop on EOS token
                                "max_tokens": 10,  # Generate exactly 10 tokens
                            },
                        )
                    )
                )

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            successful = sum(1 for r in results if r)
            logger.info(
                f"Completed {successful}/{num_requests} requests for {router_name}"
            )
            return successful

        # Create first runtime and endpoint for router 1
        logger.info("Creating first KV router with its own runtime")
        runtime1 = get_runtime(store_backend, request_plane)
        endpoint1 = runtime1.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )

        with min_initial_workers_env(num_workers):
            kv_router1 = KvRouter(
                endpoint=endpoint1,
                block_size=block_size,
                kv_router_config=kv_router_config,
            )

        # Wait for workers to be ready
        await wait_for_workers_ready(endpoint1, kv_router1, num_workers, model_name)

        # Send 25 requests to first router
        logger.info("Sending 25 requests to first router")

        # Send requests to first router
        successful1 = await send_requests_to_router(
            kv_router1, 25, "Router 1", endpoint1
        )
        assert (
            successful1 == 25
        ), f"Expected 25 successful requests to router 1, got {successful1}"

        # NATS interruption test: stop NATS, send requests, restart
        if test_nats_interruption:
            await asyncio.sleep(1)

            assert nats_server is not None  # Validated at function entry
            logger.info("=== NATS INTERRUPTION TEST: Phase 1 ===")
            logger.info("Stopping NATS server")
            nats_server.stop()

            logger.info("Sending 10 requests while NATS is down (via TCP)")
            successful_offline1 = await send_requests_to_router(
                kv_router1, 10, "Router 1 (NATS down)", endpoint1
            )
            assert (
                successful_offline1 == 10
            ), f"Expected 10 successful requests while NATS down, got {successful_offline1}"

            logger.info("Restarting NATS server (fresh state)")
            nats_server.start()

            await asyncio.sleep(5)

        if test_zmq_replay and standalone_indexer_url:
            await _zmq_replay_cycle(
                1,
                kv_router1,
                "Router 1",
                endpoint1,
                standalone_indexer_url,
                engine_workers,
                send_requests_to_router,
            )

        # Wait for snapshot to be available before creating second router.
        # In JetStream mode, the background task may purge acknowledged messages
        # from the stream before the snapshot upload completes. Poll the object
        # store so Router 2 can reliably download the snapshot on startup.
        if durable_kv_events:
            component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
            slugified = component_subject.lower().replace(".", "-").replace("_", "-")
            bucket_name = f"{slugified}-radix-bucket"
            nc = await nats.connect(servers=_nats_server())
            try:
                js = nc.jetstream()
                for attempt in range(50):
                    try:
                        obj_store = await js.object_store(bucket_name)
                        await obj_store.get("radix-state")
                        logger.info(
                            f"Snapshot available in object store (attempt {attempt + 1})"
                        )
                        break
                    except Exception:
                        await asyncio.sleep(0.1)
                else:
                    assert False, (
                        f"Snapshot not found in bucket '{bucket_name}' after 50 attempts (5s). "
                        f"Router 1 sent 25 requests with snapshot_threshold=20, snapshot should exist."
                    )
            finally:
                await nc.close()
        else:
            await asyncio.sleep(1)

        # Create second runtime and endpoint for router 2
        logger.info("Creating second KV router with its own runtime")
        runtime2 = get_runtime(store_backend, request_plane)
        endpoint2 = runtime2.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )

        with min_initial_workers_env(num_workers):
            kv_router2 = KvRouter(
                endpoint=endpoint2,
                block_size=block_size,
                kv_router_config=kv_router_config,
            )

        # Launch Indexer B alongside Router 2. Workers are passed via --workers
        # so ZMQ sockets connect before recovery, avoiding the slow-joiner problem.
        if standalone_indexer_b_url:
            engine_workers.launch_indexer()
            await wait_for_indexer_workers_active(
                standalone_indexer_b_url, engine_workers.worker_id_to_zmq_ports
            )
            logger.info(
                f"Launched Indexer B at {standalone_indexer_b_url} "
                f"(P2P recovery from Indexer A)"
            )

        # Send 25 requests to second router with initial retry loop
        logger.info("Sending 25 requests to second router")
        successful2 = await send_requests_to_router(
            kv_router2, 25, "Router 2", endpoint2
        )
        assert (
            successful2 == 25
        ), f"Expected 25 successful requests to router 2, got {successful2}"

        # NATS interruption test: stop NATS again, send requests, restart, send more
        if test_nats_interruption:
            await asyncio.sleep(1)

            assert nats_server is not None  # Validated at function entry
            logger.info("=== NATS INTERRUPTION TEST: Phase 2 ===")
            logger.info("Stopping NATS server")
            nats_server.stop()

            logger.info("Sending 10 requests while NATS is down (via TCP)")
            successful_offline2 = await send_requests_to_router(
                kv_router2, 10, "Router 2 (NATS down)", endpoint2
            )
            assert (
                successful_offline2 == 10
            ), f"Expected 10 successful requests while NATS down, got {successful_offline2}"

            logger.info("Restarting NATS server (fresh state)")
            nats_server.start()
            await asyncio.sleep(5)

            logger.info("Sending 5 more requests after NATS recovery")
            successful_recovery = await send_requests_to_router(
                kv_router1, 5, "Router 1 (post-recovery)", endpoint1
            )
            assert (
                successful_recovery == 5
            ), f"Expected 5 successful requests post-recovery, got {successful_recovery}"

        if test_zmq_replay and standalone_indexer_url:
            await _zmq_replay_cycle(
                2,
                kv_router2,
                "Router 2",
                endpoint2,
                standalone_indexer_url,
                engine_workers,
                send_requests_to_router,
            )

        # Wait for internal synchronization and ZMQ event propagation
        logger.info("Waiting for final synchronization")
        await asyncio.sleep(2)

        # Verify NATS object store bucket was created with snapshot
        # Skip for NATS interruption test (restarts fresh) and non-durable modes
        if not test_nats_interruption and durable_kv_events:
            # Mirror the Rust bucket naming logic from subscriber.rs:
            # component.subject() -> "namespace.{ns}.component.{comp}"
            # then slugify (convert dots to dashes, lowercase, etc) and append "-radix-bucket"
            component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
            slugified = component_subject.lower().replace(".", "-").replace("_", "-")
            expected_bucket = f"{slugified}-radix-bucket"
            expected_file = "radix-state"

            logger.info(f"Verifying NATS object store bucket exists: {expected_bucket}")
            snapshot_verified = False

            # Connect to NATS and check object store. This honors per-test NATS instances
            # started by fixtures (xdist-safe) instead of assuming localhost:4222.
            nc = await nats.connect(servers=_nats_server())
            try:
                js = nc.jetstream()
                obj_store = await js.object_store(expected_bucket)

                # Try to get the expected file
                try:
                    result = await obj_store.get(expected_file)
                    logger.info(
                        f"✓ Snapshot file '{expected_file}' found in bucket '{expected_bucket}' "
                        f"(size: {len(result.data) if result.data else 0} bytes)"
                    )
                    snapshot_verified = True
                except Exception as e:
                    logger.error(
                        f"Snapshot file '{expected_file}' not found in bucket '{expected_bucket}': {e}"
                    )
            except Exception as e:
                logger.error(f"Error checking NATS object store: {e}")
            finally:
                await nc.close()

            # Assert that snapshot was created (threshold=20, sent 25 requests)
            if not snapshot_verified:
                assert False, (
                    f"Expected snapshot to be created in bucket '{expected_bucket}' with file '{expected_file}'. "
                    f"Router sent 25 requests with snapshot_threshold=20, so snapshot should have been triggered."
                )
        else:
            logger.info(
                "Skipping NATS object store verification (NATS was restarted fresh for interruption test)"
            )

        # Dump states from all sources
        logger.info("Dumping states from all sources")
        state1_json = await kv_router1.dump_events()
        state2_json = await kv_router2.dump_events()

        state1 = json.loads(state1_json)
        state2 = json.loads(state2_json)

        def sort_key(event):
            data = event["event"]["data"]["stored"]
            blocks = data["blocks"]
            first_block = blocks[0]
            return (
                event["worker_id"],
                first_block["tokens_hash"],
                data["parent_hash"],
            )

        sorted_state1 = sorted(state1, key=sort_key)
        sorted_state2 = sorted(state2, key=sort_key)

        logger.info(f"Router 1 has {len(sorted_state1)} events")
        logger.info(f"Router 2 has {len(sorted_state2)} events")

        assert_event_dumps_equal(sorted_state1, sorted_state2, "Router 1", "Router 2")
        logger.info("Successfully verified Router 1 and Router 2 states are equal")

        # Verify standalone HTTP indexers build the same tree (via ZMQ)
        if standalone_indexer_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{standalone_indexer_url}/dump") as resp:
                    assert resp.status == 200, f"GET /dump failed: {resp.status}"
                    dump_a = await resp.json()

            # /dump returns {model:tenant -> {"block_size": N, "events": [...]}}
            expected_key = f"{model_name}:default"
            assert expected_key in dump_a, (
                f"Expected dump key '{expected_key}', "
                f"got keys={list(dump_a.keys())}"
            )
            for k, v in dump_a.items():
                assert (
                    isinstance(v, dict) and "events" in v
                ), f"Dump key '{k}' returned unexpected format: {v}"
            sorted_standalone_a = sorted(dump_a[expected_key]["events"], key=sort_key)
            logger.info(f"Standalone Indexer A has {len(sorted_standalone_a)} events")

            assert_event_dumps_equal(
                sorted_state1, sorted_standalone_a, "Router 1", "Standalone A"
            )
            logger.info("Standalone A matches Router 1")

            if standalone_indexer_b_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{standalone_indexer_b_url}/dump") as resp:
                        assert (
                            resp.status == 200
                        ), f"GET /dump from Indexer B failed: {resp.status}"
                        dump_b = await resp.json()

                assert expected_key in dump_b, (
                    f"Indexer B missing dump key '{expected_key}', "
                    f"got keys={list(dump_b.keys())}"
                )
                sorted_standalone_b = sorted(
                    dump_b[expected_key]["events"], key=sort_key
                )
                logger.info(
                    f"Standalone Indexer B has {len(sorted_standalone_b)} events"
                )

                assert_event_dumps_equal(
                    sorted_standalone_a,
                    sorted_standalone_b,
                    "Standalone A",
                    "Standalone B",
                )
                logger.info(
                    "All 4 dumps match: Router 1, Router 2, "
                    "Standalone A, Standalone B"
                )

        # Verify NATS consumers are created (while routers are still alive)
        # Skip for NATS interruption test (restarts fresh) and non-durable modes
        if not test_nats_interruption and durable_kv_events:
            logger.info("Verifying NATS consumers exist for both routers")
            component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
            slugified = component_subject.lower().replace(".", "-").replace("_", "-")
            stream_name = f"{slugified}-kv-events"

            nc = await nats.connect(servers=_nats_server())
            try:
                js = nc.jetstream()
                consumer_infos = await js.consumers_info(stream_name)
                consumer_names = [info.name for info in consumer_infos]
                logger.info(f"Found {len(consumer_names)} consumers: {consumer_names}")

                assert len(consumer_names) == 2, (
                    f"Expected 2 durable consumers (one per router), "
                    f"found {len(consumer_names)}: {consumer_names}"
                )
                logger.info("✓ Verified 2 durable consumers exist (one per router)")
            finally:
                await nc.close()
        else:
            logger.info(
                "Skipping NATS consumers verification (local indexer uses NATS Core, not JetStream)"
            )

    # Run the async test
    asyncio.run(test_sync())

    logger.info("Indexers sync test completed successfully")


def _test_router_decisions_disagg(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    durable_kv_events: bool = False,
    router_aic_config: Optional[dict[str, Any]] = None,
):
    """Validate KV cache prefix reuse in disaggregated prefill-decode setup via HTTP frontend.

    Assumes prefill_workers and decode_workers are already initialized. This function manages
    router lifecycle and sends progressive requests with overlapping prefixes.

    This test:
    1. Starts the KV router frontend with disagg support
    2. Sends 4 progressive requests where each extends the previous tokens by block_size
    3. Extracts prefill_worker_id and decode_worker_id from response nvext
    4. Verifies all prefill_worker_ids are the same (due to prefix reuse routing)
    5. Verifies prefill_worker_id is NOT in the set of decode_worker_ids (true disagg)

    Args:
        prefill_workers: Prefill workers already initialized with __enter__()
        decode_workers: Decode workers already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        durable_kv_events: If True, use durable KV events (JetStream). Defaults to False.
        router_aic_config: Optional AIC router perf-model config for frontend KV routing.

    Raises:
        AssertionError: If prefill_worker_ids differ across requests (prefix reuse failure)
        AssertionError: If prefill_worker_id is in decode_worker_ids (not true disagg)
    """
    with KVRouterProcess(
        request,
        block_size,
        frontend_port,
        decode_workers.namespace,
        store_backend,
        enforce_disagg=True,
        request_plane=request_plane,
        durable_kv_events=durable_kv_events,
        min_initial_workers=decode_workers.num_workers,
        router_aic_config=router_aic_config,
    ):
        # Start KV router frontend - uses decode_workers namespace for discovery
        # The frontend will auto-discover both prefill and decode workers
        logger.info(
            f"Starting KV router frontend on port {frontend_port} for disagg test"
        )

        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        # Wait for workers to register with frontend
        logger.info(
            "Waiting for prefill and decode workers to register with frontend..."
        )
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=decode_workers.num_workers,
                timeout=120,
            )
        )

        async def send_progressive_requests():
            """Send 4 progressive requests with overlapping prefixes and collect worker IDs."""
            prefill_worker_ids = []
            decode_worker_ids = []

            # Generate base tokens for progressive prefix extension
            base_content = test_payload["messages"][0]["content"]

            async with aiohttp.ClientSession() as session:
                for i in range(4):
                    # Build progressive content by repeating base content
                    # Each iteration adds more content to extend the prefix
                    progressive_content = " ".join([base_content] * (i + 1))

                    # Create payload with worker_id and timing in extra_fields
                    payload = {
                        **test_payload,
                        "messages": [
                            {
                                "role": "user",
                                "content": progressive_content,
                            }
                        ],
                        "nvext": {"extra_fields": ["worker_id", "timing"]},
                        "stream": True,
                    }

                    logger.info(
                        f"Sending request {i + 1}/4 with progressive prefix "
                        f"(~{len(progressive_content)} chars)"
                    )

                    async with session.post(chat_url, json=payload) as response:
                        assert (
                            response.status == 200
                        ), f"Request {i + 1} failed with status {response.status}"

                        # Collect all chunks and look for nvext with worker_id and timing
                        prefill_wid = None
                        decode_wid = None
                        timing_info = None

                        async for line in response.content:
                            if not line:
                                continue

                            line_str = line.decode("utf-8", errors="replace").strip()
                            if not line_str.startswith("data:"):
                                continue

                            data_str = line_str[5:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                # Check for nvext in the response
                                nvext = data.get("nvext", {})
                                if nvext:
                                    worker_id_info = nvext.get("worker_id", {})
                                    if worker_id_info:
                                        if "prefill_worker_id" in worker_id_info:
                                            prefill_wid = worker_id_info[
                                                "prefill_worker_id"
                                            ]
                                        if "decode_worker_id" in worker_id_info:
                                            decode_wid = worker_id_info[
                                                "decode_worker_id"
                                            ]
                                    # Timing info appears in final chunk
                                    if "timing" in nvext:
                                        timing_info = nvext["timing"]

                            except json.JSONDecodeError:
                                continue

                        logger.info(
                            f"Request {i + 1}: prefill_worker_id={prefill_wid}, "
                            f"decode_worker_id={decode_wid}, timing={timing_info}"
                        )

                        if prefill_wid is not None:
                            prefill_worker_ids.append(prefill_wid)
                        if decode_wid is not None:
                            decode_worker_ids.append(decode_wid)

                        # Verify timing info is present and valid
                        assert (
                            timing_info is not None
                        ), f"Request {i + 1}: Expected timing info in final chunk, got None"
                        verify_response_timing(timing_info)

                    # Small delay between requests
                    await asyncio.sleep(1)

            return prefill_worker_ids, decode_worker_ids

        # Run the progressive requests
        prefill_ids, decode_ids = asyncio.run(send_progressive_requests())

        logger.info(f"Collected prefill_worker_ids: {prefill_ids}")
        logger.info(f"Collected decode_worker_ids: {decode_ids}")

        # Verify we got worker IDs from all requests
        assert len(prefill_ids) == 4, (
            f"Expected 4 prefill_worker_ids, got {len(prefill_ids)}. "
            f"Make sure nvext.extra_fields=['worker_id'] is being processed."
        )

        # Verify prefix reuse behavior.
        #
        # In JetStream (KV events enabled) mode, the router learns cache state from KV events.
        # With the TCP request plane, we can observe a transient on the *first* request where
        # the second request is routed before the first request's KV "stored" events have been
        # fully ingested. After ingestion, routing stabilizes.
        #
        # So for TCP we assert that requests 2-4 converge to the same prefill worker; for NATS
        # request plane we keep the stronger assertion that all 4 match.
        if request_plane == "tcp":
            unique_prefill_ids = set(prefill_ids[1:])
            assert len(unique_prefill_ids) == 1, (
                f"Expected prefill requests 2-4 to route to the same worker due to prefix reuse, "
                f"but found {len(unique_prefill_ids)} unique prefill_worker_ids: {unique_prefill_ids}. "
                f"Full list: {prefill_ids}"
            )
        else:
            unique_prefill_ids = set(prefill_ids)
            assert len(unique_prefill_ids) == 1, (
                f"Expected all prefill requests to route to the same worker due to prefix reuse, "
                f"but found {len(unique_prefill_ids)} unique prefill_worker_ids: {unique_prefill_ids}. "
                f"Full list: {prefill_ids}"
            )

        # Verify prefill_worker_id is NOT in decode_worker_ids (true disagg)
        unique_decode_ids = set(decode_ids)
        prefill_id = prefill_ids[0]
        assert prefill_id not in unique_decode_ids, (
            f"Prefill worker {prefill_id} should NOT be in decode workers {unique_decode_ids}. "
            f"This suggests disaggregated mode is not working correctly - "
            f"prefill and decode should use separate worker pools."
        )

        logger.info(
            f"Successfully verified disaggregated routing:\n"
            f"  - All 4 requests routed to same prefill_worker_id={prefill_id} (prefix reuse)\n"
            f"  - Prefill worker is NOT in decode worker set {unique_decode_ids} (true disagg)"
        )


def _test_router_decisions(
    engine_workers,
    endpoint,
    model_name: str,
    request,
    test_dp_rank: bool = False,
    block_size: int = 8,
    use_kv_events: bool = True,
    durable_kv_events: bool = False,
    router_event_threads: int = 4,
    standalone_indexer_url: Optional[str] = None,
    router_aic_config: Optional[dict[str, Any]] = None,
):
    """Validate cross-worker routing decisions based on longest prefix match and tree-size tiebreaking.

    Assumes engine workers are already initialized.
    Seeds two routing targets (worker a and worker b) with different prefix trees,
    then verifies the router picks the correct worker for subsequent requests.

    Test sequence (7 blocks A-G, each block_size tokens, 5 requests):
    1. [A, B]       → force worker a        (seed worker a's tree)
    2. [A, C, D]    → force worker a        (branch under A on worker a)
    3. [A, C, E]    → force worker b        (seed worker b's tree)
    4. [A, C, D, F] → router picks          (worker a wins: prefix [A,C,D]=3 vs worker b [A,C]=2)
    5. [A, C, G]    → router picks          (tie on [A,C], worker b wins by smaller tree: 3 vs 5)

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        endpoint: Endpoint of the engine workers
        model_name: Name of the model
        request: Pytest request fixture
        test_dp_rank: If True, also forces and validates dp_rank routing (for data parallel setups)
        block_size: KV cache block size. Defaults to 8.
        use_kv_events: If True (default), uses KV events from workers. If False, uses
            approximate routing with TTL-based expiration (--no-kv-events mode).
        durable_kv_events: If True, use durable KV events (JetStream). Defaults to False.
        router_aic_config: Optional AIC router perf-model config for direct KvRouter tests.

    Raises:
        AssertionError: If routing decisions don't match expected prefix/tiebreak logic
    """

    # Create KvRouterConfig with lower snapshot threshold for testing
    # Use async to manage the test flow
    async def test_sync():
        # If standalone indexer mode, launch workers one-by-one and register.
        # Must happen before KvRouter creation since KvRouter blocks until workers appear.
        if standalone_indexer_url:
            await engine_workers.launch_workers_with_indexer(endpoint)

        # Workers register one instance per process (not per dp_rank)
        expected_num_instances = engine_workers.num_workers

        kv_router_config = KvRouterConfig(
            router_snapshot_threshold=20,
            use_kv_events=use_kv_events,
            durable_kv_events=durable_kv_events,
            router_event_threads=router_event_threads,
            router_track_prefill_tokens=True,
            router_prefill_load_model=(
                "aic" if router_aic_config is not None else "none"
            ),
        )
        aic_perf_config = (
            AicPerfConfig(**router_aic_config)
            if router_aic_config is not None
            else None
        )
        with min_initial_workers_env(expected_num_instances):
            kv_router = KvRouter(
                endpoint=endpoint,
                block_size=block_size,
                kv_router_config=kv_router_config,
                aic_perf_config=aic_perf_config,
            )

        # Wait for workers to be ready and get their instance IDs
        worker_ids = await wait_for_workers_ready(
            endpoint,
            kv_router,
            expected_num_workers=expected_num_instances,
            model_name=model_name,
        )
        logger.info(f"Workers ready: {worker_ids}")

        # Determine worker a / worker b routing targets
        if len(worker_ids) >= 2:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[1]
        elif len(worker_ids) == 1 and test_dp_rank:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[0]
        else:
            raise AssertionError(
                f"Need at least 2 routing targets but got {len(worker_ids)} worker(s) "
                f"with test_dp_rank={test_dp_rank}"
            )

        dp_rank_a = 0 if test_dp_rank else None
        dp_rank_b = 1 if test_dp_rank else None

        logger.info(
            f"Routing targets: worker_a=(id={worker_a_id}, dp_rank={dp_rank_a}), "
            f"worker_b=(id={worker_b_id}, dp_rank={dp_rank_b})"
        )

        # Generate 7 random blocks (A-G)
        num_blocks = 7
        blocks = [
            [random.randint(1, 10000) for _ in range(block_size)]
            for _ in range(num_blocks)
        ]
        A, B, C, D, E, F, G = blocks

        # 5 requests with specific prefix structure
        request_specs = [
            # (token_ids, forced_worker_id, forced_dp_rank, sleep_after)
            (A + B, worker_a_id, dp_rank_a, 0.1),  # req1: seed worker a
            (
                A + C + D,
                worker_a_id,
                dp_rank_a,
                0.1,
            ),  # req2: branch under A on worker a
            (A + C + E, worker_b_id, dp_rank_b, 2.0),  # req3: seed worker b
            (
                A + C + D + F,
                None,
                None,
                2.0,
            ),  # req4: router picks (worker a should win)
            (A + C + G, None, None, 2.0),  # req5: router picks (worker b should win)
        ]

        response_worker_ids: list[dict[str, Optional[int]]] = []

        for i, (token_ids, wid_override, dp_override, sleep_after) in enumerate(
            request_specs
        ):
            log_msg = f"Sending request {i + 1}/5 with {len(token_ids)} tokens"
            if wid_override is not None:
                log_msg += f" - FORCING worker_id={wid_override}"
                if dp_override is not None:
                    log_msg += f", dp_rank={dp_override}"
            logger.info(log_msg)

            result = await send_request_via_python_kv_router(
                kv_python_router=kv_router,
                model_name=model_name,
                token_ids=token_ids,
                initial_wait=1.0,
                max_retries=8,
                stop_conditions={
                    "ignore_eos": True,
                    "max_tokens": 2,
                },
                worker_id=wid_override,
                dp_rank=dp_override,
                return_worker_ids=True,
            )
            assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
            response_worker_ids.append(result)
            logger.info(
                f"Request {i + 1} response: prefill_worker_id={result.get('prefill_worker_id')}, "
                f"decode_worker_id={result.get('decode_worker_id')}, "
                f"prefill_dp_rank={result.get('prefill_dp_rank')}, "
                f"decode_dp_rank={result.get('decode_dp_rank')}"
            )

            if sleep_after > 0:
                await asyncio.sleep(sleep_after)

        events_json = await kv_router.dump_events()
        return (
            events_json,
            worker_a_id,
            worker_b_id,
            dp_rank_a,
            dp_rank_b,
            response_worker_ids,
            A + C + D + F,  # req4 tokens for standalone indexer /score verification
        )

    # Run the async test
    (
        events_json,
        worker_a_id,
        worker_b_id,
        dp_rank_a,
        dp_rank_b,
        response_worker_ids,
        req4_tokens,
    ) = asyncio.run(test_sync())

    # Verify request 4 routed to worker a (longest prefix match)
    req4 = response_worker_ids[3]
    assert req4["prefill_worker_id"] == worker_a_id, (
        f"Request 4: expected prefill_worker_id={worker_a_id} (longest prefix match), "
        f"got {req4['prefill_worker_id']}"
    )
    if test_dp_rank:
        assert (
            req4["prefill_dp_rank"] == dp_rank_a
        ), f"Request 4: expected prefill_dp_rank={dp_rank_a}, got {req4['prefill_dp_rank']}"

    # Verify request 5 routed to worker b (tiebreak by smaller tree)
    req5 = response_worker_ids[4]
    assert req5["prefill_worker_id"] == worker_b_id, (
        f"Request 5: expected prefill_worker_id={worker_b_id} (tiebreak by smaller tree), "
        f"got {req5['prefill_worker_id']}"
    )
    if test_dp_rank:
        assert (
            req5["prefill_dp_rank"] == dp_rank_b
        ), f"Request 5: expected prefill_dp_rank={dp_rank_b}, got {req5['prefill_dp_rank']}"

    logger.info(
        f"Response routing verified: req4 → worker_a (id={worker_a_id}, dp_rank={dp_rank_a}), "
        f"req5 → worker_b (id={worker_b_id}, dp_rank={dp_rank_b})"
    )

    # Parse events and verify event counts per routing target
    events = json.loads(events_json)

    # Always group by (worker_id, dp_rank)
    events_by_key: dict[tuple[int, int], list[Any]] = {}
    for event in events:
        worker_id = event.get("worker_id")
        dp_rank = event.get("event", {}).get("dp_rank", 0)
        key = (worker_id, dp_rank)
        if key not in events_by_key:
            events_by_key[key] = []
        events_by_key[key].append(event)

    def count_stored_blocks(events: list[Any]) -> int:
        total = 0
        for event in events:
            stored = event.get("event", {}).get("data", {}).get("stored")
            if stored is None:
                continue
            total += len(stored.get("blocks", []))
        return total

    logger.info(
        "Stored blocks by (worker_id, dp_rank): "
        f"{[(key, count_stored_blocks(evts)) for key, evts in events_by_key.items()]}"
    )

    # Worker a key: 5 stored blocks (A, B from req1; C, D from req2; F from req4)
    worker_a_key = (worker_a_id, dp_rank_a if dp_rank_a is not None else 0)
    worker_a_blocks = count_stored_blocks(events_by_key.get(worker_a_key, []))
    assert worker_a_blocks == 5, (
        f"Expected worker_a {worker_a_key} to have 5 stored blocks (A,B + C,D + F), "
        f"but found {worker_a_blocks}"
    )

    # Worker b key: 4 stored blocks (A, C, E from req3; G from req5)
    worker_b_key = (worker_b_id, dp_rank_b if dp_rank_b is not None else 0)
    worker_b_blocks = count_stored_blocks(events_by_key.get(worker_b_key, []))
    assert worker_b_blocks == 4, (
        f"Expected worker_b {worker_b_key} to have 4 stored blocks (A,C,E + G), "
        f"but found {worker_b_blocks}"
    )

    logger.info(
        f"Successfully verified cross-worker routing: "
        f"worker_a {worker_a_key} has {worker_a_blocks} stored blocks, "
        f"worker_b {worker_b_key} has {worker_b_blocks} stored blocks"
    )

    # Verify standalone indexer scores via HTTP POST /query
    if standalone_indexer_url:
        _dp_a = dp_rank_a if dp_rank_a is not None else 0
        _dp_b = dp_rank_b if dp_rank_b is not None else 0

        async def _verify_scores():
            # Wait for ZMQ events to propagate to the indexer
            await asyncio.sleep(3)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{standalone_indexer_url}/query",
                    json={"token_ids": req4_tokens, "model_name": model_name},
                ) as resp:
                    assert resp.status == 200, f"POST /query failed: {resp.status}"
                    scores = (await resp.json())["scores"]

                    id_a = str(worker_a_id)
                    id_b = str(worker_b_id)
                    dp_a = str(_dp_a)
                    dp_b = str(_dp_b)
                    score_a = scores[id_a][dp_a]
                    score_b = scores[id_b][dp_b]

                    logger.info(
                        f"Standalone indexer /query: {id_a}[{dp_a}]={score_a}, "
                        f"{id_b}[{dp_b}]={score_b}"
                    )
                    assert score_a > score_b, (
                        f"Expected instance {id_a} dp_rank {dp_a} score {score_a} > "
                        f"instance {id_b} dp_rank {dp_b} score {score_b} for req4 tokens"
                    )

        asyncio.run(_verify_scores())


def _test_busy_threshold_endpoint(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 503

    For now, this test only verifies the endpoint is accessible and returns valid responses.

    Args:
        engine_workers: MockerProcess instance (already initialized with __enter__())
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload (used to extract model name)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats" or "tcp"). Defaults to "nats".

    Raises:
        AssertionError: If endpoint responses are incorrect
    """
    # Initial thresholds - we need to start with these so the monitor is created
    initial_active_decode_blocks_threshold = 0.9
    initial_active_prefill_tokens_threshold = 1000  # Literal token count threshold

    with KVRouterProcess(
        request,
        block_size,
        frontend_port,
        engine_workers.namespace,
        store_backend,
        blocks_threshold=initial_active_decode_blocks_threshold,
        tokens_threshold=initial_active_prefill_tokens_threshold,
        request_plane=request_plane,
    ):
        # Start KV router frontend with initial thresholds to create monitor
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        frontend_url = f"http://localhost:{frontend_port}"
        busy_threshold_url = f"{frontend_url}/busy_threshold"

        # Wait for workers to register with frontend
        logger.info("Waiting for workers to register with frontend...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=120,
            )
        )

        model_name = test_payload.get("model", "test-model")

        async def test_busy_threshold_api():
            async with aiohttp.ClientSession() as session:
                # Test 1: GET /busy_threshold - list all thresholds
                logger.info("Testing GET /busy_threshold (list all)")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    assert (
                        "thresholds" in data
                    ), f"Expected 'thresholds' key in response: {data}"
                    logger.info(f"GET /busy_threshold response: {data}")

                # Test 2: POST /busy_threshold with model only (get thresholds)
                logger.info(
                    f"Testing POST /busy_threshold to get thresholds for model '{model_name}'"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (get) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == initial_active_decode_blocks_threshold
                    ), f"Expected initial active_decode_blocks_threshold={initial_active_decode_blocks_threshold}: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == initial_active_prefill_tokens_threshold
                    ), f"Expected initial active_prefill_tokens_threshold={initial_active_prefill_tokens_threshold}: {data}"
                    logger.info(
                        f"POST /busy_threshold (get) response: status={response.status}, data={data}"
                    )

                # Test 3: POST /busy_threshold to set active_decode_blocks_threshold only
                test_active_decode_blocks_threshold = 0.75
                logger.info(
                    f"Testing POST /busy_threshold to set active_decode_blocks_threshold={test_active_decode_blocks_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_decode_blocks_threshold": test_active_decode_blocks_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set blocks) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("model") == model_name
                    ), f"Expected model={model_name}: {data}"
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == test_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={test_active_decode_blocks_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set blocks) response: {data}")

                # Test 4: POST /busy_threshold to set active_prefill_tokens_threshold only
                test_active_prefill_tokens_threshold = (
                    2000  # Literal token count threshold
                )
                logger.info(
                    f"Testing POST /busy_threshold to set active_prefill_tokens_threshold={test_active_prefill_tokens_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_prefill_tokens_threshold": test_active_prefill_tokens_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set tokens) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == test_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={test_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set tokens) response: {data}")

                # Test 5: POST /busy_threshold to set both thresholds
                new_active_decode_blocks_threshold = 0.5
                new_active_prefill_tokens_threshold = (
                    1200  # Literal token count threshold
                )
                logger.info(
                    f"Testing POST /busy_threshold to set both thresholds: "
                    f"active_decode_blocks={new_active_decode_blocks_threshold}, active_prefill_tokens={new_active_prefill_tokens_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_decode_blocks_threshold": new_active_decode_blocks_threshold,
                        "active_prefill_tokens_threshold": new_active_prefill_tokens_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set both) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == new_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={new_active_decode_blocks_threshold}: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == new_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={new_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set both) response: {data}")

                # Test 6: GET /busy_threshold - verify thresholds appear in list
                logger.info("Testing GET /busy_threshold to verify thresholds in list")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    thresholds = data.get("thresholds", [])
                    model_entry = next(
                        (t for t in thresholds if t["model"] == model_name), None
                    )
                    assert (
                        model_entry is not None
                    ), f"Expected model '{model_name}' in thresholds: {data}"
                    assert (
                        model_entry.get("active_decode_blocks_threshold")
                        == new_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={new_active_decode_blocks_threshold}: {data}"
                    assert (
                        model_entry.get("active_prefill_tokens_threshold")
                        == new_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={new_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"GET /busy_threshold (after set) response: {data}")

                # Test 7: Invalid active_decode_blocks_threshold value (should fail validation)
                logger.info(
                    "Testing POST /busy_threshold with invalid active_decode_blocks_threshold (>1.0)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_decode_blocks_threshold": 1.5},
                ) as response:
                    assert (
                        response.status == 400
                    ), f"Expected 400 for invalid active_decode_blocks_threshold, got {response.status}"
                    data = await response.json()
                    logger.info(
                        f"POST /busy_threshold (invalid blocks) response: {data}"
                    )

                # Test 8: active_prefill_tokens_threshold accepts large values (should be valid)
                logger.info(
                    "Testing POST /busy_threshold with large active_prefill_tokens_threshold (valid)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_prefill_tokens_threshold": 5000},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"Expected 200 for large active_prefill_tokens_threshold, got {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold") == 5000
                    ), f"Expected active_prefill_tokens_threshold=5000: {data}"
                    logger.info(
                        f"POST /busy_threshold (large tokens threshold) response: {data}"
                    )

                # Test 9: Invalid active_prefill_tokens_threshold value (should fail validation for < 0)
                # Note: Returns 422 because -1.0 can't be deserialized into u64 (type validation)
                # vs Test 7 which returns 400 because 1.5 is a valid f64 but fails range validation
                logger.info(
                    "Testing POST /busy_threshold with invalid active_prefill_tokens_threshold (< 0)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_prefill_tokens_threshold": -1.0},
                ) as response:
                    assert (
                        response.status == 422
                    ), f"Expected 422 for negative active_prefill_tokens_threshold, got {response.status}"
                    data = await response.json()
                    logger.info(
                        f"POST /busy_threshold (invalid tokens) response: {data}"
                    )

                # Test 10: Set active_prefill_tokens_threshold_frac (fraction of max_num_batched_tokens)
                test_frac_threshold = 0.8
                logger.info(
                    f"Testing POST /busy_threshold to set active_prefill_tokens_threshold_frac={test_frac_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_prefill_tokens_threshold_frac": test_frac_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set frac) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold_frac")
                        == test_frac_threshold
                    ), f"Expected active_prefill_tokens_threshold_frac={test_frac_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set frac) response: {data}")

                # Test 11: Verify frac threshold appears in GET /busy_threshold list
                logger.info(
                    "Testing GET /busy_threshold to verify frac threshold in list"
                )
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    thresholds = data.get("thresholds", [])
                    model_entry = next(
                        (t for t in thresholds if t["model"] == model_name), None
                    )
                    assert (
                        model_entry is not None
                    ), f"Expected model '{model_name}' in thresholds: {data}"
                    assert (
                        model_entry.get("active_prefill_tokens_threshold_frac")
                        == test_frac_threshold
                    ), f"Expected active_prefill_tokens_threshold_frac={test_frac_threshold}: {data}"
                    logger.info(
                        f"GET /busy_threshold (after set frac) response: {data}"
                    )

                logger.info("All busy_threshold endpoint tests passed!")

        asyncio.run(test_busy_threshold_api())


def _test_disagg_direct_mode(
    prefill_workers,
    decode_workers,
    request,
    frontend_port: int,
    test_payload: dict,
    request_plane: str = "nats",
):
    """E2E test for disaggregated Direct routing mode (simulating GAIE EPP).

    In Direct mode, the router does not select workers itself.
    Worker IDs must be provided via x-worker-instance-id and x-prefill-instance-id
    HTTP headers. The test verifies:
      1. Requests with explicit worker ID headers succeed and return a valid response.
      2. Requests without headers fail (Direct mode rejects unaddressed requests).

    Args:
        prefill_workers: Prefill mocker workers (already started).
        decode_workers: Decode mocker workers (already started).
        request: Pytest request fixture.
        frontend_port: Port for the Direct-mode frontend HTTP server.
        test_payload: Base test payload for /v1/chat/completions.
        request_plane: Transport for request plane ("nats" or "tcp").
    """
    with DirectRouterProcess(
        request,
        frontend_port,
        decode_workers.namespace,
        enforce_disagg=True,
        request_plane=request_plane,
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        logger.info("Waiting for models to appear in Direct-mode frontend...")

        async def wait_for_models():
            models_url = f"{frontend_url}/v1/models"
            for _ in range(120):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(models_url) as response:
                            if response.status == 200:
                                data = await response.json()
                                models = data.get("data", [])
                                if models:
                                    logger.info(
                                        f"Models registered: {[m.get('id') for m in models]}"
                                    )
                                    return
                except Exception as e:
                    logger.debug(f"Error checking models endpoint: {e}")
                await asyncio.sleep(1)
            raise TimeoutError("Timeout waiting for models in Direct-mode frontend")

        asyncio.run(wait_for_models())

        # Phase 2: Discover worker IDs via the runtime
        runtime = get_runtime(request_plane=request_plane)
        prefill_endpoint = runtime.endpoint(
            f"{decode_workers.namespace}.prefill.generate"
        )
        decode_endpoint = runtime.endpoint(
            f"{decode_workers.namespace}.backend.generate"
        )

        async def discover_workers():
            prefill_client = await prefill_endpoint.client()
            decode_client = await decode_endpoint.client()

            for _ in range(60):
                p_ids = prefill_client.instance_ids()
                d_ids = decode_client.instance_ids()
                if p_ids and d_ids:
                    return p_ids, d_ids
                await asyncio.sleep(0.5)
            raise TimeoutError(
                f"Timeout discovering workers: prefill={p_ids}, decode={d_ids}"
            )

        prefill_ids, decode_ids = asyncio.run(discover_workers())
        logger.info(f"Discovered prefill workers: {prefill_ids}")
        logger.info(f"Discovered decode workers: {decode_ids}")

        target_prefill = prefill_ids[0]
        target_decode = decode_ids[0]

        async def run_direct_mode_tests():
            # Test 1: Request WITH correct headers should succeed.
            # In direct mode the router is a passthrough — it does not have a
            # KvRouter and does not record worker IDs on the RequestTracker, so
            # the response's nvext will not contain worker_id info.  We only
            # verify that the request is routed successfully (HTTP 200) and
            # produces a valid chat completion response.
            payload = {
                **test_payload,
                "stream": False,
            }
            headers = {
                "x-worker-instance-id": str(target_decode),
                "x-prefill-instance-id": str(target_prefill),
                "x-dp-rank": "0",
                "x-prefill-dp-rank": "0",
            }

            async with aiohttp.ClientSession() as session:
                # Retry a few times to allow the pipeline to warm up
                for attempt in range(10):
                    async with session.post(
                        chat_url, json=payload, headers=headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(
                                f"Direct-mode response (attempt {attempt + 1}): "
                                f"status=200, model={data.get('model')}"
                            )
                            assert (
                                "choices" in data
                            ), "Expected 'choices' in response data"
                            assert (
                                len(data["choices"]) > 0
                            ), "Expected at least one choice in response"
                            break
                        else:
                            logger.info(
                                f"Direct-mode attempt {attempt + 1} returned "
                                f"status {response.status}, retrying..."
                            )
                            await asyncio.sleep(2)
                else:
                    raise AssertionError(
                        "Direct-mode request with headers never returned 200"
                    )

                # Test 2: Request WITHOUT headers should fail (Direct mode
                # rejects requests that have no worker ID)
                logger.info(
                    "Sending request without headers (should fail in Direct mode)..."
                )
                no_header_payload = {**test_payload, "stream": False}
                async with session.post(chat_url, json=no_header_payload) as response:
                    assert response.status != 200, (
                        f"Expected non-200 status without routing headers in Direct mode, "
                        f"got {response.status}. Direct mode must reject unaddressed requests."
                    )
                    logger.info(
                        f"Correctly rejected headerless request: status={response.status}"
                    )

        asyncio.run(run_direct_mode_tests())
        logger.info("Direct-mode disagg E2E test passed")
