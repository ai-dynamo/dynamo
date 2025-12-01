# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import random
import string
import time
from typing import Any, Optional

import aiohttp
import nats

from dynamo._core import DistributedRuntime, KvPushRouter, KvRouterConfig
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

NUM_REQUESTS = 100
BLOCK_SIZE = 16


########################################################
# Helper Classes
########################################################


class KVRouterProcess(ManagedProcess):
    """Manages the KV router process using dynamo.frontend"""

    def __init__(
        self,
        request,
        block_size: int,
        frontend_port: int,
        namespace: str,
        store_backend: str = "etcd",
        enforce_disagg: bool = False,
        busy_threshold: float | None = None,
    ):
        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--kv-cache-block-size",
            str(block_size),
            "--router-mode",
            "kv",
            "--http-port",
            str(frontend_port),
            "--store-kv",
            store_backend,
            "--namespace",
            namespace,
        ]

        if enforce_disagg:
            command.append("--enforce-disagg")

        if busy_threshold is not None:
            command.extend(["--busy-threshold", str(busy_threshold)])

        super().__init__(
            command=command,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_ready)
            ],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        self.port = frontend_port

    def _check_ready(self, response):
        """Check if KV router is ready"""
        return response.status_code == 200

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)


def generate_random_suffix() -> str:
    """Generate a 10-character random alphabetic suffix for namespace isolation."""
    return "".join(random.choices(string.ascii_lowercase, k=10))  # noqa: S311


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
    For direct Python API testing with KvPushRouter, use wait_for_workers_ready() instead.

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
        await asyncio.sleep(2)

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
        await asyncio.sleep(2)


async def wait_for_workers_ready(
    endpoint,
    router: KvPushRouter,
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
        router: The KvPushRouter to use for sending warmup requests
        expected_num_workers: Number of workers to wait for

    Returns:
        Sorted list of unique instance IDs (ints).

    Raises:
        AssertionError: If workers don't become ready or warmup request fails.
    """
    logger.info("Waiting for workers to be ready")

    # Get the client from the endpoint
    client = await endpoint.client()

    # Poll for instance IDs until we have the expected number
    instance_ids: list[int] = []
    max_wait_time = 60  # seconds
    start_time = asyncio.get_running_loop().time()

    while len(instance_ids) < expected_num_workers:
        instance_ids = client.instance_ids()
        logger.info(f"Found {len(instance_ids)} instance(s): {instance_ids}")

        if len(instance_ids) >= expected_num_workers:
            break

        # Check timeout
        if asyncio.get_running_loop().time() - start_time > max_wait_time:
            raise AssertionError(
                f"Timeout waiting for workers. Found {len(instance_ids)} instance(s), expected {expected_num_workers}"
            )

        # Wait 1 second before polling again
        await asyncio.sleep(1.0)

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


def get_runtime(store_backend="etcd", request_plane="nats"):
    """Create a DistributedRuntime instance for testing.

    Args:
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: How frontend talks to backend ("tcp", "http" or "nats"). Defaults to "nats".
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

    nc = await nats.connect("nats://localhost:4222")
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
    kv_python_router: KvPushRouter,
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
) -> bool:
    """Send a request to the specified worker instance.
    Returns True if workers respond, otherwise raises or returns False.
    """

    wait_time = initial_wait

    log_message = (
        f"worker with worker_id={worker_id}"
        if worker_id is not None
        else "the best available worker"
    )

    # Retry loop sending request to worker with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Sending request to {log_message} (attempt {attempt + 1})")

            stream = await kv_python_router.generate(
                token_ids=token_ids,
                model=model_name,
                stop_conditions=stop_conditions,
                sampling_options=sampling_options,
                output_options=output_options,
                router_config_override=router_config_override,
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

    # Collect tokens from the SSE stream
    generated_tokens = []
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
            f"Successfully verified {max_tokens} tokens generated as expected via KvPushRouter with ignore_eos=True"
        )
        return True

    return False


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
):
    """Basic router test: start router, wait for workers and send concurrent requests via HTTP frontend.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This is a shared test implementation for both mocker and vLLM workers.
    Always waits for workers to be properly registered before sending requests to avoid flakiness.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port to start the frontend HTTP server on
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        frontend_timeout: Timeout for frontend readiness check (default: 120s)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If requests fail or frontend doesn't become ready
        TimeoutError: If frontend doesn't become ready within timeout
    """
    try:
        # Start KV router frontend
        logger.info(f"Starting KV router frontend on port {frontend_port}")
        kv_router = KVRouterProcess(
            request, block_size, frontend_port, engine_workers.namespace, store_backend
        )
        kv_router.__enter__()

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

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)


def _test_router_two_routers(
    engine_workers,
    block_size: int,
    request,
    router_ports: list[int],
    test_payload: dict,
    num_requests: int,
    store_backend: str = "etcd",
):
    """Test two KV routers with alternating requests and consumer lifecycle verification.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This test:
    1. Starts two KV routers on different ports
    2. Sends requests alternating between the two routers
    3. Verifies that both routers create durable consumers
    4. Verifies consumers are cleaned up when routers exit

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        router_ports: List of two port numbers for the routers (e.g., [8091, 8092])
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If consumer lifecycle verification fails
    """
    import nats

    kv_routers = []

    try:
        # Start two KV routers on different ports
        for i, port in enumerate(router_ports):
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(
                request, block_size, port, engine_workers.namespace, store_backend
            )
            kv_router.__enter__()
            kv_routers.append(kv_router)

            # Add delay between routers for file backend to ensure first router's
            # registration is visible before second router starts its cleanup
            if i == 0 and store_backend == "file":
                logger.info(
                    "Waiting 0.5s for first router to fully register (file backend)"
                )
                time.sleep(0.5)

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
            nc = await nats.connect("nats://localhost:4222")
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

        # Run consumer lifecycle verification
        asyncio.run(verify_consumer_lifecycle())

        # Clear the kv_routers list since we've already cleaned them up
        kv_routers = []

    finally:
        # Clean up any remaining routers (in case of error before consumer verification)
        for kv_router in kv_routers:
            kv_router.__exit__(None, None, None)


def _test_python_router_bindings(
    engine_workers,
    endpoint,
    block_size: int,
    model_name: str,
    num_workers: int,
):
    """Test KvPushRouter Python bindings with token streaming and config overrides.

    Assumes engine_workers are already initialized. This test creates a KvPushRouter
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

    # Create KvPushRouter Python object
    kv_push_router = KvPushRouter(
        endpoint=endpoint,
        block_size=block_size,
        kv_router_config=kv_router_config,
    )

    logger.info("Created KvPushRouter Python object")

    # Wait for workers to be ready
    asyncio.run(
        wait_for_workers_ready(endpoint, kv_push_router, num_workers, model_name)
    )

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
            kv_python_router=kv_push_router,
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
            kv_python_router=kv_push_router,
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
            kv_python_router=kv_push_router,
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

    logger.info("KvPushRouter bindings test completed successfully")


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
    import aiohttp

    try:
        # Start KV router (frontend)
        logger.info(f"Starting KV router frontend on port {frontend_port}")
        kv_router = KVRouterProcess(
            request, block_size, frontend_port, engine_workers.namespace, store_backend
        )
        kv_router.__enter__()

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Send a warming request first to ensure system is ready
        logger.info("Sending warming request without annotations...")
        asyncio.run(send_request_with_retry(url, test_payload))

        # Test payload with query_instance_id annotation
        annotated_payload = {
            **test_payload,
            "nvext": {"annotations": ["query_instance_id"]},
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

                    # Parse and validate the response structure
                    events = []

                    sse_parts = full_response.split("\n\n")

                    for part in sse_parts:
                        part = part.strip()
                        if not part:
                            continue

                        if part.startswith("event:"):
                            lines = part.split("\n")
                            event_line = next(
                                (line for line in lines if line.startswith("event:")),
                                None,
                            )
                            data_line = next(
                                (
                                    line
                                    for line in lines
                                    if line.startswith("data:") or line.startswith(":")
                                ),
                                None,
                            )

                            if event_line and data_line:
                                event_type = event_line.split(":", 1)[1].strip()
                                if data_line.startswith("data:"):
                                    data_value = data_line.split(":", 1)[1].strip()
                                else:
                                    data_value = data_line.split(":", 1)[1].strip()
                                events.append((event_type, data_value))
                        elif part.startswith("data:"):
                            data_value = part.split(":", 1)[1].strip()

                    logger.info(f"Parsed events: {events}")

                    # Validate worker_instance_id event
                    worker_event = next(
                        (e for e in events if e[0] == "worker_instance_id"), None
                    )
                    assert (
                        worker_event is not None
                    ), f"Missing worker_instance_id event in: {events}"

                    # Validate token_data event
                    token_event = next(
                        (e for e in events if e[0] == "token_data"), None
                    )
                    assert (
                        token_event is not None
                    ), f"Missing token_data event in: {events}"

                    token_data_str = token_event[1].strip('"')
                    try:
                        token_list = json.loads(token_data_str)
                    except json.JSONDecodeError as e:
                        raise AssertionError(
                            f"token_data is not valid JSON: {token_data_str}, error: {e}"
                        )

                    assert isinstance(
                        token_list, list
                    ), f"token_data should be a list, got: {type(token_list)}"
                    assert (
                        len(token_list) > 0
                    ), f"token_data should not be empty: {token_list}"
                    assert all(
                        isinstance(token, int) for token in token_list
                    ), f"All tokens should be integers: {token_list}"

                    logger.info(
                        f"Valid token_data with {len(token_list)} tokens: {token_list[:10]}{'...' if len(token_list) > 10 else ''}"
                    )

                    # Validate that no actual generation happened (should only be metadata)
                    # This proves the early return worked correctly
                    generation_indicators = [
                        "choices",
                        "content",
                        "delta",
                        "finish_reason",
                    ]
                    for indicator in generation_indicators:
                        assert (
                            indicator not in full_response.lower()
                        ), f"Found generation indicator '{indicator}' - request should not have been routed to worker"

                    logger.info(
                        "No generation content found - early return worked correctly"
                    )

                    return {
                        "worker_instance_id": worker_event[1].strip('"'),
                        "token_count": len(token_list),
                        "tokens": token_list,
                    }

        result = asyncio.run(test_annotation_response())

        logger.info("Successfully validated query_instance_id annotation response:")
        logger.info(f"Worker ID: {result['worker_instance_id']}")
        logger.info(f"Token count: {result['token_count']}")

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)


def _test_router_overload_503(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    busy_threshold: float = 0.2,
):
    """Test that KV router returns 503 when all workers are busy.

    Assumes engine_workers are already initialized. This function manages router lifecycle.
    Uses limited resources to intentionally trigger the overload condition.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache (should be small to exhaust quickly, e.g. 4)
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        busy_threshold: Busy threshold for the router (default 0.2)

    Raises:
        AssertionError: If 503 response is not received when expected
    """
    import aiohttp

    from tests.utils.managed_process import ManagedProcess

    try:
        logger.info(
            f"Starting KV router frontend on port {frontend_port} with limited resources"
        )

        # Custom command for router with limited block size
        command = [
            "python",
            "-m",
            "dynamo.frontend",
            "--busy-threshold",
            str(busy_threshold),
            "--kv-cache-block-size",
            str(block_size),
            "--router-mode",
            "kv",
            "--http-port",
            str(frontend_port),
        ]

        kv_router = ManagedProcess(
            command=command,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    lambda r: r.status_code == 200,
                )
            ],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        kv_router.__enter__()

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Custom payload for 503 test with more tokens to consume resources
        test_payload_503 = {
            **test_payload,
            "max_tokens": 50,  # Longer output to consume more blocks
        }

        # First, send one request with retry to ensure system is ready
        logger.info("Sending initial request to ensure system is ready...")
        asyncio.run(send_inflight_requests([url], test_payload_503, 1))

        # Now send 50 concurrent requests to exhaust resources, then verify 503
        logger.info("Sending 50 concurrent requests to exhaust resources...")

        async def exhaust_resources_and_verify_503():
            async with aiohttp.ClientSession() as session:
                # Start 50 long-running requests concurrently
                tasks = []
                for i in range(50):
                    # Create unique shuffled content for each request
                    content_words = test_payload["messages"][0]["content"].split()
                    random.shuffle(content_words)
                    shuffled_content = " ".join(content_words)

                    # Create unique payload for this request
                    unique_payload = {
                        **test_payload,
                        "max_tokens": 50,
                        "messages": [
                            {**test_payload["messages"][0], "content": shuffled_content}
                        ],
                    }

                    async def send_long_request(req_id, payload):
                        try:
                            async with session.post(url, json=payload) as response:
                                if response.status == 200:
                                    # Don't read the response fully, just hold the connection
                                    await asyncio.sleep(
                                        10
                                    )  # Hold connection for 10 seconds
                                    return True
                                else:
                                    logger.info(
                                        f"Request {req_id} got status {response.status}"
                                    )
                                    return False
                        except Exception as e:
                            logger.info(f"Request {req_id} failed: {e}")
                            return False

                    tasks.append(
                        asyncio.create_task(send_long_request(i, unique_payload))
                    )

                # Wait briefly to ensure requests are in-flight
                await asyncio.sleep(0.2)

                # Now send one more request that should get 503
                logger.info("Sending additional request that should receive 503...")
                try:
                    async with session.post(url, json=test_payload_503) as response:
                        status_code = response.status
                        if status_code == 503:
                            body = await response.json()
                            logger.info(f"Got expected 503 response: {body}")
                            assert "Service temporarily unavailable" in body.get(
                                "error", ""
                            ) or "All workers are busy" in body.get(
                                "error", ""
                            ), f"Expected service overload error message, got: {body}"
                            return True
                        else:
                            logger.error(f"Expected 503 but got {status_code}")
                            if status_code == 200:
                                logger.error(
                                    "Request unexpectedly succeeded when it should have been rejected"
                                )
                            return False
                except Exception as e:
                    logger.error(f"Failed to send overload test request: {e}")
                    return False
                finally:
                    # Cancel all background tasks
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        # Run the test
        success = asyncio.run(exhaust_resources_and_verify_503())
        assert success, "Failed to verify 503 response when resources are exhausted"

        logger.info("Successfully verified 503 response when all workers are busy")

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)


def _test_router_indexers_sync(
    engine_workers,
    block_size: int,
    model_name: str,
    num_workers: int,
    store_backend: str = "etcd",
):
    """Test that two KV routers have synchronized indexer states after processing requests.

    Assumes engine_workers are already initialized. This test:
    1. Creates first KvPushRouter (with its own runtime) and sends 25 requests (triggers snapshot at threshold=20)
    2. Creates second KvPushRouter (with its own runtime, should sync from NATS snapshot)
    3. Sends 25 requests to second router
    4. Verifies NATS object store contains the snapshot
    5. Dumps states from both routers and compares them (should be identical)

    This validates that the snapshot mechanism works and routers can sync state from NATS.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        model_name: Model name to use for requests
        num_workers: Expected number of workers
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If router states don't synchronize correctly or snapshot is missing
    """
    import nats

    # Use async to manage the test flow
    async def test_sync():
        # Create KvRouterConfig with lower snapshot threshold for testing
        kv_router_config = KvRouterConfig(router_snapshot_threshold=20)

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
        runtime1 = get_runtime(store_backend)
        namespace1 = runtime1.namespace(engine_workers.namespace)
        component1 = namespace1.component(engine_workers.component_name)
        endpoint1 = component1.endpoint("generate")

        kv_push_router1 = KvPushRouter(
            endpoint=endpoint1,
            block_size=block_size,
            kv_router_config=kv_router_config,
        )

        # Wait for workers to be ready
        await wait_for_workers_ready(
            endpoint1, kv_push_router1, num_workers, model_name
        )

        # Send 25 requests to first router
        logger.info("Sending 25 requests to first router")

        # Send requests to first router
        successful1 = await send_requests_to_router(
            kv_push_router1, 25, "Router 1", endpoint1
        )
        assert (
            successful1 == 25
        ), f"Expected 25 successful requests to router 1, got {successful1}"

        # Wait for a second before creating the second router
        logger.info("Waiting for 1 second before creating second router")
        await asyncio.sleep(1)

        # Create second runtime and endpoint for router 2
        logger.info("Creating second KV router with its own runtime")
        runtime2 = get_runtime(store_backend)
        namespace2 = runtime2.namespace(engine_workers.namespace)
        component2 = namespace2.component(engine_workers.component_name)
        endpoint2 = component2.endpoint("generate")

        kv_push_router2 = KvPushRouter(
            endpoint=endpoint2,
            block_size=block_size,
            kv_router_config=kv_router_config,
        )

        # Send 25 requests to second router with initial retry loop
        logger.info("Sending 25 requests to second router")
        successful2 = await send_requests_to_router(
            kv_push_router2, 25, "Router 2", endpoint2
        )
        assert (
            successful2 == 25
        ), f"Expected 25 successful requests to router 2, got {successful2}"

        # Wait for all requests to complete (they should already be complete from gather)
        # Wait another 1 second for internal synchronization
        logger.info("Waiting for final synchronization")
        await asyncio.sleep(1)

        # Verify NATS object store bucket was created with snapshot
        # Mirror the Rust bucket naming logic from subscriber.rs:
        # component.subject() -> "namespace.{ns}.component.{comp}"
        # then slugify (convert dots to dashes, lowercase, etc) and append "-radix-bucket"
        component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
        slugified = component_subject.lower().replace(".", "-").replace("_", "-")
        expected_bucket = f"{slugified}-radix-bucket"
        expected_file = "radix-state"

        logger.info(f"Verifying NATS object store bucket exists: {expected_bucket}")
        snapshot_verified = False
        try:
            # Connect to NATS and check object store
            nc = await nats.connect("nats://localhost:4222")
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
            finally:
                await nc.close()
        except Exception as e:
            logger.error(f"Error checking NATS object store: {e}")

        # Assert that snapshot was created (threshold=20, sent 25 requests)
        if not snapshot_verified:
            assert False, (
                f"Expected snapshot to be created in bucket '{expected_bucket}' with file '{expected_file}'. "
                f"Router sent 25 requests with snapshot_threshold=20, so snapshot should have been triggered."
            )

        # Dump states from both routers
        logger.info("Dumping states from both routers")
        state1_json = await kv_push_router1.dump_events()
        state2_json = await kv_push_router2.dump_events()

        # Parse JSON strings for comparison
        state1 = json.loads(state1_json)
        state2 = json.loads(state2_json)

        # Sort both states for comparison (order might differ due to HashMap iteration and sharding)
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

        # Verify they are equal
        logger.info(f"Router 1 has {len(sorted_state1)} events")
        logger.info(f"Router 2 has {len(sorted_state2)} events")

        # Compare states one by one and only show differences
        if len(sorted_state1) != len(sorted_state2):
            logger.error(
                f"Router 1 has {len(sorted_state1)} events, Router 2 has {len(sorted_state2)} events"
            )
            assert False, "Router states have different numbers of events"

        differences = []
        for i, (state1_item, state2_item) in enumerate(
            zip(sorted_state1, sorted_state2)
        ):
            # Create copies without event_id for comparison
            item1_compare = state1_item.copy()
            item2_compare = state2_item.copy()

            # Remove event_id from the nested event structure
            if "event" in item1_compare and "event_id" in item1_compare["event"]:
                del item1_compare["event"]["event_id"]
            if "event" in item2_compare and "event_id" in item2_compare["event"]:
                del item2_compare["event"]["event_id"]

            if item1_compare != item2_compare:
                differences.append(
                    {
                        "index": i,
                        "router1_state": state1_item,
                        "router2_state": state2_item,
                    }
                )
        # If there are differences, format them for easier debugging
        if differences:
            error_msg = (
                f"Router states are not equal. Found {len(differences)} differences:\n"
            )
            for diff in differences:
                error_msg += f"\nDifference at index {diff['index']}:\n"
                error_msg += (
                    f"Router 1: {json.dumps(diff['router1_state'], indent=2)}\n"
                )
                error_msg += (
                    f"Router 2: {json.dumps(diff['router2_state'], indent=2)}\n"
                )
                error_msg += "-" * 80 + "\n"

            assert False, error_msg

        logger.info("Successfully verified that both router states are equal")

        # Verify NATS consumers are created (while routers are still alive)
        logger.info("Verifying NATS consumers exist for both routers")
        component_subject = f"namespace.{engine_workers.namespace}.component.{engine_workers.component_name}"
        slugified = component_subject.lower().replace(".", "-").replace("_", "-")
        stream_name = f"{slugified}-kv-events"

        nc = await nats.connect("nats://localhost:4222")
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

    # Run the async test
    asyncio.run(test_sync())

    logger.info("Indexers sync test completed successfully")


def _test_router_disagg_decisions(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
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

    Raises:
        AssertionError: If prefill_worker_ids differ across requests (prefix reuse failure)
        AssertionError: If prefill_worker_id is in decode_worker_ids (not true disagg)
    """
    try:
        # Start KV router frontend - uses decode_workers namespace for discovery
        # The frontend will auto-discover both prefill and decode workers
        logger.info(
            f"Starting KV router frontend on port {frontend_port} for disagg test"
        )
        kv_router = KVRouterProcess(
            request,
            block_size,
            frontend_port,
            decode_workers.namespace,
            store_backend,
            enforce_disagg=True,
        )
        kv_router.__enter__()

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

                    # Create payload with worker_id in extra_fields to get prefill/decode worker IDs
                    payload = {
                        **test_payload,
                        "messages": [
                            {
                                "role": "user",
                                "content": progressive_content,
                            }
                        ],
                        "nvext": {"extra_fields": ["worker_id"]},
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

                        # Collect all chunks and look for nvext with worker_id
                        prefill_wid = None
                        decode_wid = None

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
                                # Check for nvext.worker_id in the response
                                nvext = data.get("nvext", {})
                                worker_id_info = nvext.get("worker_id", {})

                                if worker_id_info:
                                    if "prefill_worker_id" in worker_id_info:
                                        prefill_wid = worker_id_info[
                                            "prefill_worker_id"
                                        ]
                                    if "decode_worker_id" in worker_id_info:
                                        decode_wid = worker_id_info["decode_worker_id"]

                            except json.JSONDecodeError:
                                continue

                        logger.info(
                            f"Request {i + 1}: prefill_worker_id={prefill_wid}, "
                            f"decode_worker_id={decode_wid}"
                        )

                        if prefill_wid is not None:
                            prefill_worker_ids.append(prefill_wid)
                        if decode_wid is not None:
                            decode_worker_ids.append(decode_wid)

                    # Small delay between requests
                    await asyncio.sleep(0.5)

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

        # Verify all prefill_worker_ids are the same (prefix reuse)
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

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)


def _test_router_decisions(
    engine_workers,
    endpoint,
    model_name: str,
    request,
    test_dp_rank: bool = False,
):
    """Validate KV cache prefix reuse and worker routing by sending progressive requests with overlapping prefixes.

    Assumes engine workers are already initialized. Sends 4 progressive requests where each extends
    the previous tokens by BLOCK_SIZE. The first request is forced to a specific worker (and optionally
    dp_rank), and subsequent requests should naturally route to the same worker due to prefix reuse.

    Args:
        engine_workers: MockerProcess or VLLMProcess instance (already initialized with __enter__())
        endpoint: Endpoint of the engine workers
        model_name: Name of the model
        request: Pytest request fixture
        test_dp_rank: If True, also forces and validates dp_rank routing (for data parallel setups)

    Raises:
        AssertionError: If routing decisions don't follow KV cache prefix reuse as expected
    """
    # Create KvRouterConfig with lower snapshot threshold for testing
    kv_router_config = KvRouterConfig(router_snapshot_threshold=20)
    kv_push_router = KvPushRouter(
        endpoint=endpoint,
        block_size=BLOCK_SIZE,
        kv_router_config=kv_router_config,
    )

    # Use async to manage the test flow
    async def test_sync():
        # Wait for workers to be ready and get their instance IDs
        worker_ids = await wait_for_workers_ready(
            endpoint,
            kv_push_router,
            expected_num_workers=engine_workers.num_workers,
            model_name=model_name,
        )
        logger.info(f"Workers ready: {worker_ids}")

        # Use the first worker_id for forced routing
        forced_worker_id = worker_ids[0]
        forced_dp_rank = 1 if test_dp_rank else None

        if test_dp_rank:
            logger.info(
                f"Will force first request to worker_id={forced_worker_id}, dp_rank={forced_dp_rank}"
            )
        else:
            logger.info(f"Will force first request to worker_id={forced_worker_id}")

        # Send 4 progressive requests with overlapping prefixes
        cumulative_tokens = []

        for i in range(4):
            # Add BLOCK_SIZE new random tokens
            new_tokens = [random.randint(1, 10000) for _ in range(BLOCK_SIZE)]
            cumulative_tokens.extend(new_tokens)

            # Force first request to specific worker_id (and dp_rank if testing DP), let subsequent requests follow naturally
            worker_id_override = forced_worker_id if i == 0 else None
            dp_rank_override = forced_dp_rank if i == 0 and test_dp_rank else None

            log_msg = (
                f"Sending request {i + 1}/4 with {len(cumulative_tokens)} tokens "
                f"(added {len(new_tokens)} new tokens)"
            )
            if worker_id_override is not None:
                if test_dp_rank:
                    log_msg += f" - FORCING worker_id={worker_id_override}, dp_rank={dp_rank_override}"
                else:
                    log_msg += f" - FORCING worker_id={worker_id_override}"
            logger.info(log_msg)

            await send_request_via_python_kv_router(
                kv_python_router=kv_push_router,
                model_name=model_name,
                token_ids=cumulative_tokens.copy(),
                initial_wait=1.0,
                max_retries=8,
                stop_conditions={
                    "ignore_eos": True,  # Don't stop on EOS token
                    "max_tokens": 2,  # Generate exactly 2 tokens
                },
                worker_id=worker_id_override,
                dp_rank=dp_rank_override,
            )

            # Wait a bit between requests
            await asyncio.sleep(0.5)

        # Wait for final synchronization (especially important for DP)
        if test_dp_rank:
            await asyncio.sleep(1)

        # Dump events from the router
        events_json = await kv_push_router.dump_events()
        return events_json, forced_worker_id, forced_dp_rank

    # Run the async test
    events_json, expected_worker_id, expected_dp_rank = asyncio.run(test_sync())

    # Parse events and count by worker routing key (worker_id or (worker_id, dp_rank))
    events = json.loads(events_json)

    if test_dp_rank:
        # Group by (worker_id, dp_rank) tuple for DP testing
        events_by_key_dp: dict[tuple[int, int], list[Any]] = {}
        for event in events:
            worker_id = event.get("worker_id")
            dp_rank = event.get("event", {}).get("dp_rank", 0)
            key = (worker_id, dp_rank)
            if key not in events_by_key_dp:
                events_by_key_dp[key] = []
            events_by_key_dp[key].append(event)

        logger.info(
            f"Events by (worker_id, dp_rank): {[(key, len(evts)) for key, evts in events_by_key_dp.items()]}"
        )

        # Verify: All but one routing key should have no events (due to prefix reuse)
        keys_with_events_dp = [
            key for key, evts in events_by_key_dp.items() if len(evts) > 0
        ]

        assert len(keys_with_events_dp) == 1, (
            f"Expected exactly 1 (worker_id, dp_rank) to have events (due to prefix reuse), "
            f"but found {len(keys_with_events_dp)} with events: {keys_with_events_dp}"
        )

        # Verify: The routing key with events should have exactly 4 events (one per request)
        active_key_dp = keys_with_events_dp[0]
        num_events = len(events_by_key_dp[active_key_dp])

        assert num_events == 4, (
            f"Expected (worker_id, dp_rank) {active_key_dp} to have exactly 4 events, "
            f"but found {num_events} events"
        )

        # Verify: Routing should match the forced values
        active_worker_id, active_dp_rank = active_key_dp
        assert active_worker_id == expected_worker_id, (
            f"Expected all events to have worker_id={expected_worker_id} (forced in first request), "
            f"but found worker_id={active_worker_id}"
        )
        assert active_dp_rank == expected_dp_rank, (
            f"Expected all events to have dp_rank={expected_dp_rank} (forced in first request), "
            f"but found dp_rank={active_dp_rank}"
        )
        logger.info(
            f"Successfully verified: Worker {active_worker_id} dp_rank {active_dp_rank} handled all 4 requests with prefix reuse. "
            f"All events correctly routed to worker_id={expected_worker_id}, dp_rank={expected_dp_rank} as expected. "
            f"KV events synchronized correctly."
        )
    else:
        # Group by worker_id only for multiple workers testing
        events_by_key_single: dict[int, list] = {}
        for event in events:
            worker_id = event.get("worker_id")
            if worker_id not in events_by_key_single:
                events_by_key_single[worker_id] = []
            events_by_key_single[worker_id].append(event)

        logger.info(
            f"Events by worker_id: {[(key, len(evts)) for key, evts in events_by_key_single.items()]}"
        )

        # Verify: All but one routing key should have no events (due to prefix reuse)
        keys_with_events_single = [
            key for key, evts in events_by_key_single.items() if len(evts) > 0
        ]

        assert len(keys_with_events_single) == 1, (
            f"Expected exactly 1 worker_id to have events (due to prefix reuse), "
            f"but found {len(keys_with_events_single)} with events: {keys_with_events_single}"
        )

        # Verify: The routing key with events should have exactly 4 events (one per request)
        active_worker_id = keys_with_events_single[0]
        num_events = len(events_by_key_single[active_worker_id])

        assert num_events == 4, (
            f"Expected worker_id {active_worker_id} to have exactly 4 events, "
            f"but found {num_events} events"
        )

        # Verify: Routing should match the forced values
        assert active_worker_id == expected_worker_id, (
            f"Expected all events to have worker_id={expected_worker_id} (forced in first request), "
            f"but found worker_id={active_worker_id}"
        )
        logger.info(
            f"Successfully verified: Worker {active_worker_id} handled all 4 requests with prefix reuse. "
            f"All events correctly routed to worker_id={expected_worker_id} as expected. "
            f"KV events synchronized correctly."
        )


def _test_busy_threshold_endpoint(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 503

    For now, this test only verifies the endpoint is accessible and returns valid responses.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload (used to extract model name)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If endpoint responses are incorrect
    """
    # Initial threshold - we need to start with one so the monitor is created
    initial_threshold = 0.9

    try:
        # Start KV router frontend with initial busy_threshold to create monitor
        logger.info(f"Starting KV router frontend on port {frontend_port}")
        kv_router = KVRouterProcess(
            request,
            block_size,
            frontend_port,
            engine_workers.namespace,
            store_backend,
            busy_threshold=initial_threshold,
        )
        kv_router.__enter__()

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
                # Should have the initial threshold since we started with --busy-threshold
                logger.info("Testing GET /busy_threshold (list all)")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    assert (
                        "thresholds" in data
                    ), f"Expected 'thresholds' key in response: {data}"
                    thresholds = data.get("thresholds", [])
                    # Should have at least the model with initial_threshold
                    logger.info(f"GET /busy_threshold response: {data}")

                # Test 2: POST /busy_threshold with model only (get threshold)
                # Should return the initial threshold since we started with --busy-threshold
                logger.info(
                    f"Testing POST /busy_threshold to get threshold for model '{model_name}'"
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
                        data.get("threshold") == initial_threshold
                    ), f"Expected initial threshold={initial_threshold}: {data}"
                    logger.info(
                        f"POST /busy_threshold (get) response: status={response.status}, data={data}"
                    )

                # Test 3: POST /busy_threshold to set a threshold
                test_threshold = 0.75
                logger.info(
                    f"Testing POST /busy_threshold to set threshold={test_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "threshold": test_threshold},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("model") == model_name
                    ), f"Expected model={model_name}: {data}"
                    assert (
                        data.get("threshold") == test_threshold
                    ), f"Expected threshold={test_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set) response: {data}")

                # Test 4: POST /busy_threshold to get the threshold we just set
                logger.info("Testing POST /busy_threshold to verify threshold was set")
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (get after set) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("threshold") == test_threshold
                    ), f"Expected threshold={test_threshold}: {data}"
                    logger.info(
                        f"POST /busy_threshold (get after set) response: {data}"
                    )

                # Test 5: POST /busy_threshold to update the threshold
                new_threshold = 0.5
                logger.info(
                    f"Testing POST /busy_threshold to update threshold={new_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "threshold": new_threshold},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (update) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("threshold") == new_threshold
                    ), f"Expected threshold={new_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (update) response: {data}")

                # Test 6: GET /busy_threshold - verify threshold appears in list
                logger.info("Testing GET /busy_threshold to verify threshold in list")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    thresholds = data.get("thresholds", [])
                    # thresholds is an array of {model, threshold} objects
                    model_thresholds = {t["model"]: t["threshold"] for t in thresholds}
                    assert (
                        model_name in model_thresholds
                    ), f"Expected model '{model_name}' in thresholds: {data}"
                    assert (
                        model_thresholds[model_name] == new_threshold
                    ), f"Expected threshold={new_threshold} for model '{model_name}': {data}"
                    logger.info(f"GET /busy_threshold (after set) response: {data}")

                # Test 7: Invalid threshold value (should fail validation)
                logger.info(
                    "Testing POST /busy_threshold with invalid threshold (>1.0)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "threshold": 1.5},
                ) as response:
                    assert (
                        response.status == 400
                    ), f"Expected 400 for invalid threshold, got {response.status}"
                    data = await response.json()
                    logger.info(f"POST /busy_threshold (invalid) response: {data}")

                logger.info("All busy_threshold endpoint tests passed!")

        asyncio.run(test_busy_threshold_api())

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)
