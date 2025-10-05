# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import random
import string
from typing import Any, Dict, Optional

import aiohttp
import pytest

from dynamo._core import DistributedRuntime, KvPushRouter, KvRouterConfig
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
NUM_MOCKERS = 2
BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
NUM_REQUESTS = 100
PORT = 8090  # Starting port for mocker instances


def generate_random_suffix() -> str:
    """Generate a 10-character random alphabetic suffix for namespace isolation."""
    return "".join(random.choices(string.ascii_lowercase, k=10))


# Shared test payload for all tests
TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "In a quiet meadow tucked between rolling hills, a plump gray rabbit nibbled on clover beneath the shade of a gnarled oak tree. Its ears twitched at the faint rustle of leaves, but it remained calm, confident in the safety of its burrow just a few hops away. The late afternoon sun warmed its fur, and tiny dust motes danced in the golden light as bees hummed lazily nearby. Though the rabbit lived a simple life, every day was an adventure of scents, shadows, and snacks—an endless search for the tastiest patch of greens and the softest spot to nap.",
        }
    ],
    "stream": True,
    "max_tokens": 10,
}


class MockerProcess:
    """Manages multiple mocker engine instances with the same namespace"""

    def __init__(
        self,
        request,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
    ):
        # Generate a unique namespace suffix shared by all mockers
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.endpoint = f"dyn://{self.namespace}.mocker.generate"
        self.num_mockers = num_mockers
        self.mocker_processes = []

        # Default mocker args if not provided
        if mocker_args is None:
            mocker_args = {}

        # Create multiple mocker processes with the same namespace
        for i in range(num_mockers):
            command = [
                "python",
                "-m",
                "dynamo.mocker",
                "--model-path",
                MODEL_NAME,
                "--endpoint",
                self.endpoint,
            ]

            # Add individual CLI arguments from mocker_args
            if "speedup_ratio" in mocker_args:
                command.extend(["--speedup-ratio", str(mocker_args["speedup_ratio"])])
            if "block_size" in mocker_args:
                command.extend(["--block-size", str(mocker_args["block_size"])])
            if "num_gpu_blocks" in mocker_args:
                command.extend(
                    ["--num-gpu-blocks-override", str(mocker_args["num_gpu_blocks"])]
                )
            if "max_num_seqs" in mocker_args:
                command.extend(["--max-num-seqs", str(mocker_args["max_num_seqs"])])
            if "max_num_batched_tokens" in mocker_args:
                command.extend(
                    [
                        "--max-num-batched-tokens",
                        str(mocker_args["max_num_batched_tokens"]),
                    ]
                )
            if "enable_prefix_caching" in mocker_args:
                if mocker_args["enable_prefix_caching"]:
                    command.append("--enable-prefix-caching")
                else:
                    command.append("--no-enable-prefix-caching")
            if "enable_chunked_prefill" in mocker_args:
                if mocker_args["enable_chunked_prefill"]:
                    command.append("--enable-chunked-prefill")
                else:
                    command.append("--no-enable-chunked-prefill")
            if "watermark" in mocker_args:
                command.extend(["--watermark", str(mocker_args["watermark"])])
            if "dp_size" in mocker_args:
                command.extend(["--data-parallel-size", str(mocker_args["dp_size"])])

            process = ManagedProcess(
                command=command,
                timeout=60,
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_existing=False,
            )
            self.mocker_processes.append(process)
            logger.info(f"Created mocker instance {i} with endpoint: {self.endpoint}")

    def __enter__(self):
        """Start all mocker processes"""
        for i, process in enumerate(self.mocker_processes):
            logger.info(f"Starting mocker instance {i}")
            process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all mocker processes"""
        for i, process in enumerate(self.mocker_processes):
            logger.info(f"Stopping mocker instance {i}")
            process.__exit__(exc_type, exc_val, exc_tb)


class KVRouterProcess(ManagedProcess):
    """Manages the KV router process using dynamo.frontend"""

    def __init__(self, request, frontend_port: int):
        command = [
            "python",
            "-m",
            "dynamo.frontend",
            "--kv-cache-block-size",
            str(BLOCK_SIZE),
            "--router-mode",
            "kv",
            "--http-port",
            str(frontend_port),
        ]

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
                        logger.info(f"First request succeeded on attempt {attempt + 1}")
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


def get_runtime():
    """Get or create a DistributedRuntime instance.

    This handles the case where a worker is already initialized (common in CI)
    by using the detached() method to reuse the existing runtime.
    """
    try:
        # Try to use existing runtime (common in CI where tests run in same process)
        _runtime_instance = DistributedRuntime.detached()
        logger.info("Using detached runtime (worker already initialized)")
    except Exception as e:
        # If no existing runtime, create a new one
        logger.info(f"Creating new runtime (detached failed: {e})")
        loop = asyncio.get_running_loop()
        _runtime_instance = DistributedRuntime(loop, False)

    return _runtime_instance


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
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        failed = sum(1 for r in results if not r)

        logger.info(f"Completed all requests: {successful} successful, {failed} failed")

    assert (
        successful == num_requests
    ), f"Expected {num_requests} successful requests, got {successful}"
    logger.info(f"All {num_requests} requests completed successfully")


async def send_request_via_python_kv_router(
    kv_python_router: KvPushRouter,
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
):
    """Send a request to the specified mocker instance.
    Returns True if mockers respond, otherwise raises or returns False.
    """

    wait_time = initial_wait

    log_message = (
        f"the mocker with worker_id={worker_id}"
        if worker_id is not None
        else "the best available mocker"
    )

    # Retry loop sending reuqest to mocker worker with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Sending request to {log_message} (attempt {attempt + 1})")

            stream = await kv_python_router.generate(
                token_ids=token_ids,
                model=MODEL_NAME,
                stop_conditions=stop_conditions,
                sampling_options=sampling_options,
                output_options=output_options,
                router_config_override=router_config_override,
                worker_id=worker_id,
            )

            if stream is not None:
                logger.info(f"Request succeeded on attempt {attempt + 1}")
                break

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries:
                await asyncio.sleep(wait_time)
                wait_time *= 2
            else:
                raise RuntimeError(
                    f"Failed to connect to mockers after {max_retries + 1} attempts: {e}"
                )

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
                logger.info(f"Stream finished with reason: {response['finish_reason']}")

    # Verify if expected number of tokens are generated if max_tokens specified and ignore_eos is True
    logger.info(f"Total generated tokens: {len(generated_tokens)}")
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

        logger.info(
            f"Successfully verified {max_tokens} tokens generated as expected via KvPushRouter with ignore_eos=True"
        )
        return True

    return False


async def initialize_mockers_and_get_ids(
    router: KvPushRouter, retries: int = 10
) -> list[int]:
    """Warm up mocker workers and collect their worker IDs.

    Why this is needed:
      For tests that rely on KV state, we require confirmation that each worker has actually
      been initialized and produced at least one event the router indexed.

    Returns:
        Sorted list of unique worker IDs (ints) discovered via KV events.

    Raises:
        AssertionError: If the maximum number of retries is exceeded before all NUM_MOCKERS workers are observed.
    """

    logger.info("Initializing mockers and getting worker ids")
    retries_count: int = 0
    event_worker_ids: list[int] = []
    # Loop until we have all expected worker IDs or we hit timeout
    # To do: by the routing logic, the second request shouldn't go to the same worker as the first,
    #        so we shouldn't need more than NUM_MOCKERS requests to discover all workers.
    while len(event_worker_ids) < NUM_MOCKERS and retries_count < retries:
        # Generate small test token IDs
        test_token_ids = [random.randint(1, 10000) for _ in range(10)]

        # Send request without specifying worker_id to allow routing to uninitialized workers
        logger.info(
            f"Sending warm-up request {retries_count + 1} with token_ids: {test_token_ids}"
        )
        await send_request_via_python_kv_router(
            kv_python_router=router,
            token_ids=test_token_ids,
            initial_wait=1.0,
            max_retries=8,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 10,  # Generate exactly 10 token (To be verified: There seems to be a mimimum of # tokens required to trigger the kv event generation)
            },
        )

        # Small delay to allow events to be processed
        await asyncio.sleep(0.5)

        # Dump KV events from the router
        events_json = await router.dump_events()

        # Parse the JSON events and extract worker IDs
        events = json.loads(events_json)
        event_worker_ids = sorted({e["worker_id"] for e in events})
        logger.info(f"Discovered worker IDs: {event_worker_ids}")
        retries_count += 1

    if len(event_worker_ids) < NUM_MOCKERS:
        raise AssertionError(
            f"Discovered {len(event_worker_ids)} worker ids {event_worker_ids}, expected {NUM_MOCKERS}"
        )

    return event_worker_ids


def get_worker_status_from_events_by_ids(
    dumped_events: str, worker_ids: list[int]
) -> list[Dict[str, Any]]:
    """Extract per-worker KV event summary for the specified worker IDs.

    Args:
        dumped_events: JSON string returned by KvPushRouter.dump_events().
        worker_ids: List of worker IDs to summarize.

    Returns:
        A list of dictionaries, each containing:
            - worker_id: The worker ID
            - kv_block_hashes: Sorted list of block hashes present for that worker
            - kv_event_ids: Sorted list of event IDs for that worker
    """
    events = json.loads(dumped_events)
    worker_statuses: list[Dict[str, Any]] = []
    for worker_id in worker_ids:
        logger.info(f"Extracting status for worker_id={worker_id} from events")
        # Extract block_hash and event_id info.
        block_hashes: set[int] = set()
        event_ids: set[int] = set()
        for e in events:
            if e.get("worker_id") != worker_id:
                continue
            # Collect event_id if present
            event_id = (
                e.get("event").get("event_id")
                if "event" in e and isinstance(e["event"], dict)
                else None
            )
            if event_id is not None:
                event_ids.add(event_id)
            # Prefer a direct block_hash if available
            if "block_hash" in e:
                bh = e.get("block_hash")
                if isinstance(bh, int):
                    block_hashes.add(bh)
            else:
                # Attempt to traverse nested structure: event -> data -> stored -> blocks -> tokens_hash / block_hash
                stored = e.get("event", {}).get("data", {}).get("stored", {})
                blocks = stored.get("blocks", []) if isinstance(stored, dict) else []
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    # Use tokens_hash as a surrogate if block_hash absent
                    bh = b.get("block_hash", b.get("tokens_hash"))
                    if isinstance(bh, int):
                        block_hashes.add(bh)
        worker_statuses.append(
            {
                "worker_id": worker_id,
                "kv_block_hashes": sorted(block_hashes),
                "kv_event_ids": sorted(event_ids),
            }
        )
    return worker_statuses


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_mocker_kv_router(request, runtime_services, predownload_tokenizers):
    """
    Test KV router with multiple mocker engine instances.
    This test doesn't require GPUs and runs quickly for pre-merge validation.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker KV router test")

    # Create mocker args dictiona: FixtureRequestry: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start KV router (frontend)
        frontend_port = PORT
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        kv_router = KVRouterProcess(request, frontend_port)
        kv_router.__enter__()

        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Use async to send requests concurrently for better performance
        asyncio.run(
            send_inflight_requests(
                [
                    f"http://localhost:{frontend_port}/v1/chat/completions"
                ],  # Pass as list
                TEST_PAYLOAD,
                NUM_REQUESTS,
            )
        )

        logger.info(f"Successfully completed {NUM_REQUESTS} requests")

    finally:
        # Clean up
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)

        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_mocker_two_kv_router(request, runtime_services, predownload_tokenizers):
    """
    Test with two KV routers and multiple mocker engine instances.
    Alternates requests between the two routers to test load distribution.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker two KV router test")

    # Create mocker args dictionary: FixtureRequest: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    kv_routers = []

    try:
        # Start two KV routers (frontend) on ports 8091 and 8092
        router_ports = [PORT + 1, PORT + 2]  # 8091 and 8092

        for port in router_ports:
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(request, port)
            kv_router.__enter__()
            kv_routers.append(kv_router)

        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Build URLs for both routers
        router_urls = [
            f"http://localhost:{port}/v1/chat/completions" for port in router_ports
        ]

        # Use async to send requests concurrently, alternating between routers
        asyncio.run(
            send_inflight_requests(
                router_urls,
                TEST_PAYLOAD,
                NUM_REQUESTS,
            )
        )

        logger.info(
            f"Successfully completed {NUM_REQUESTS} requests across {len(router_ports)} routers"
        )

    finally:
        # Clean up routers
        for kv_router in kv_routers:
            kv_router.__exit__(None, None, None)

        # Clean up mockers
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
@pytest.mark.skip(reason="Flaky, temporarily disabled")
def test_mocker_kv_router_overload_503(
    request, runtime_services, predownload_tokenizers
):
    """
    Test that KV router returns 503 when all workers are busy.
    This test uses limited resources to intentionally trigger the overload condition.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker KV router overload test for 503 status")
    # Create mocker args dictionary with limited resources
    mocker_args = {
        "speedup_ratio": 10,
        "block_size": 4,  # Smaller block size
        "num_gpu_blocks": 64,  # Limited GPU blocks to exhaust quickly
    }

    try:
        # Start KV router (frontend) with limited block size
        frontend_port = PORT + 10  # Use different port to avoid conflicts
        logger.info(
            f"Starting KV router frontend on port {frontend_port} with limited resources"
        )

        # Custom command for router with limited block size
        command = [
            "python",
            "-m",
            "dynamo.frontend",
            "--busy-threshold",
            "0.2",
            "--kv-cache-block-size",
            "4",  # Match the mocker's block size
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

        # Start single mocker instance with limited resources using the new CLI interface
        logger.info("Starting single mocker instance with limited resources")
        mockers = MockerProcess(request, mocker_args=mocker_args, num_mockers=1)
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Custom payload for 503 test with more tokens to consume resources
        test_payload_503 = {
            **TEST_PAYLOAD,
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
                    content_words = TEST_PAYLOAD["messages"][0]["content"].split()
                    random.shuffle(content_words)
                    shuffled_content = " ".join(content_words)

                    # Create unique payload for this request
                    unique_payload = {
                        **TEST_PAYLOAD,
                        "max_tokens": 50,
                        "messages": [
                            {**TEST_PAYLOAD["messages"][0], "content": shuffled_content}
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
        # Clean up
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)

        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_kv_push_router_bindings(request, runtime_services, predownload_tokenizers):
    """
    Test KvPushRouter Python bindings with mocker engines.
    This test creates KvPushRouter as a Python object and verifies
    token streaming with ignore_eos=True and max_tokens=20.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting KvPushRouter bindings test")

    # Create mocker args dictionary
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        # Use the namespace from the mockers
        namespace = runtime.namespace(mockers.namespace)
        component = namespace.component("mocker")
        endpoint = component.endpoint("generate")

        # Create KvRouterConfig with default settings
        kv_router_config = KvRouterConfig()

        # Create KvPushRouter Python object
        kv_push_router = KvPushRouter(
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            kv_router_config=kv_router_config,
        )

        logger.info("Created KvPushRouter Python object")

        # Initialize and check the readiness of the mockers by sending dummy request
        asyncio.run(
            send_request_via_python_kv_router(
                kv_python_router=kv_push_router,
                token_ids=[1, 2, 3],
                initial_wait=1.0,
                max_retries=8,
                stop_conditions={"max_tokens": 1},  # Generate just 1 token
                sampling_options={"temperature": 0.7},
                output_options={
                    "include_input_tokens": False,
                    "return_full_text": False,
                },
            )
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
        logger.info(
            f"Testing with full router config overrides: {router_config_override}"
        )
        asyncio.run(
            send_request_via_python_kv_router(
                kv_python_router=kv_push_router,
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

    finally:
        # Clean up mockers
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_indexers_sync(request, runtime_services, predownload_tokenizers):
    """
    Test that two KV routers have synchronized indexer states after processing requests.
    This test verifies that both routers converge to the same internal state.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting indexers sync test")

    # Create mocker args dicti: FixtureRequestonary: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        # Initialize mockers
        mockers.__enter__()

        # Use async to manage the test flow
        async def test_sync():
            # Get runtime and create endpoint
            runtime = get_runtime()
            # Use the namespace from the mockers
            namespace = runtime.namespace(mockers.namespace)
            component = namespace.component("mocker")
            endpoint = component.endpoint("generate")

            # Create KvRouterConfig with lower snapshot threshold for testing
            kv_router_config = KvRouterConfig(router_snapshot_threshold=20)

            async def send_requests_to_router(router, num_requests, router_name):
                # Generate small test token IDs
                test_token_ids = [random.randint(1, 10000) for _ in range(10)]

                # Initialize and check the readiness of the mockers by sending dummy request
                logger.info(f"Initializing {router_name} and mocker instances")
                await send_request_via_python_kv_router(
                    kv_python_router=router,
                    token_ids=test_token_ids,
                    initial_wait=1.0,
                    max_retries=8,
                    stop_conditions={"max_tokens": 1},  # Generate just 1 token
                )

                # Now send the actual requests
                tasks = []
                for i in range(num_requests):
                    # Generate random token IDs for each request
                    logger.info(
                        f"Sending request {i + 1}/{num_requests} to {router_name}"
                    )

                    # Generate 30 random tokens
                    request_tokens = [random.randint(1, 10000) for _ in range(30)]

                    # Send request to mocker via the router
                    tasks.append(
                        asyncio.create_task(
                            send_request_via_python_kv_router(
                                kv_python_router=router,
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

            # Launch first router
            logger.info("Creating first KV router")
            kv_push_router1 = KvPushRouter(
                endpoint=endpoint,
                block_size=BLOCK_SIZE,
                kv_router_config=kv_router_config,
            )

            # Send 25 requests to first router with initial retry loop
            logger.info("Sending 25 requests to first router")

            # Send requests to first router
            successful1 = await send_requests_to_router(kv_push_router1, 25, "Router 1")
            assert (
                successful1 == 25
            ), f"Expected 25 successful requests to router 1, got {successful1}"

            # Wait for a second before creating the second router
            logger.info("Waiting for 1 second before creating second router")
            await asyncio.sleep(1)

            # Launch second router - will automatically sync with the first router's state
            logger.info("Creating second KV router")
            kv_push_router2 = KvPushRouter(
                endpoint=endpoint,
                block_size=BLOCK_SIZE,
                kv_router_config=kv_router_config,
            )

            # Send 25 requests to second router with initial retry loop
            logger.info("Sending 25 requests to second router")
            successful2 = await send_requests_to_router(kv_push_router2, 25, "Router 2")
            assert (
                successful2 == 25
            ), f"Expected 25 successful requests to router 2, got {successful2}"

            # Wait for all requests to complete (they should already be complete from gather)
            # Wait another 1 second for internal synchronization
            logger.info("Waiting for final synchronization")
            await asyncio.sleep(1)

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
                error_msg = f"Router states are not equal. Found {len(differences)} differences:\n"
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

        # Run the async test
        asyncio.run(test_sync())

        logger.info("Indexers sync test completed successfully")

    finally:
        # Clean up mockers
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_query_instance_id_returns_worker_and_tokens(
    request, runtime_services, predownload_tokenizers
):
    """
    Test that the KV router correctly handles query_instance_id annotation.

    When a request includes 'nvext.annotations': ['query_instance_id'], the router should:
    1. NOT route the request to a worker immediately
    2. Return worker_instance_id as an SSE event
    3. Return token_data as an SSE event containing the request tokens
    4. Term: FixtureRequestinate the stream w: tuple[NatsServer, EtcdServer]ith [DONE]: NoneType

    This tests the specific code block:
        if query_instance_id {
            let instance_id_str = instance_id.to_string();
            let response = Annotated::from_annotation("worker_instance_id", &instance_id_str)?;
            let response_tokens = Annotated::from_annotation("token_data", &request.token_ids)?;
            let stream = stream::iter(vec![response, response_tokens]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }
    """

    logger.info("Starting KV router query_instance_id annotation test")

    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    os.makedirs(request.node.name, exist_ok=True)

    try:
        # Start KV router (frontend)
        frontend_port = PORT + 30  # Use unique port to avoid conflicts
        logger.info(f"Starting KV router frontend on port {frontend_port}")
        kv_router = KVRouterProcess(request, frontend_port)
        kv_router.__enter__()

        # Start multiple mocker engines to ensure worker selection logic
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Send a warming request first to ensure system is ready
        logger.info("Sending warming request without annotations...")
        asyncio.run(send_request_with_retry(url, TEST_PAYLOAD))

        # Test payload with query_instance_id annotation
        annotated_payload = {
            **TEST_PAYLOAD,
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
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_router_by_workerid_and_check_kvevents_sync(
    request, runtime_services, predownload_tokenizers
):
    """Validate per-worker routing and KV event synchronization when explicitly targeting workers.

    Flow:
      - Start two mocker workers sharing a namespace.
      - Initialize all workers and ensure they are ready for serving by checking their kv events synchronization to the router.
      - Snapshot initial per-worker block hashes.
      - For each worker, send a request pinned to it with incrementing block spans (2, 3, ...).
      - Check the post-test events to ensure each request was routed to the expected worker
        and processed correctly by comparing block hashes dumped from the route events.
      - Ensure the number of new blocks per worker matches the expected count based on the requests sent.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting test router by worker id and check KV events synchronization")

    # Create mocker args dicti: FixtureRequestonary: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        # Initialize mockers
        mockers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        # Use the namespace from the mockers
        namespace = runtime.namespace(mockers.namespace)
        component = namespace.component("mocker")
        endpoint = component.endpoint("generate")

        # Create KvRouterConfig with lower snapshot threshold for testing
        kv_router_config = KvRouterConfig(router_snapshot_threshold=20)
        kv_push_router = KvPushRouter(
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            kv_router_config=kv_router_config,
        )

        # Use async to manage the test flow
        async def test_sync():
            # Initialize workers via small warmup requests and derive worker IDs from router events
            mocker_worker_ids = await initialize_mockers_and_get_ids(kv_push_router)

            # Get the initial worker statuses from the events (await dump_events!)
            initial_events_json = await kv_push_router.dump_events()
            initial_worker_statuses = get_worker_status_from_events_by_ids(
                initial_events_json, mocker_worker_ids
            )
            logger.info(f"Initial worker statuses: {initial_worker_statuses}")
            assert (
                len(initial_worker_statuses) == NUM_MOCKERS
            ), "Not all workers have been initialized successfully with registered kv events"

            # Test each worker with specific number of blocks
            block_count = 2
            expected_added_blocks: dict[int, int] = {}
            for worker_id in mocker_worker_ids:
                # Generate random test tokens
                test_token_ids = [
                    random.randint(1, 10000) for _ in range(block_count * BLOCK_SIZE)
                ]
                logger.info(
                    f"Sending request to worker {worker_id} with {block_count} blocks ({len(test_token_ids)} tokens)"
                )
                expected_added_blocks[worker_id] = block_count
                # Send request to worker with the specific worker_id
                await send_request_via_python_kv_router(
                    kv_python_router=kv_push_router,
                    token_ids=test_token_ids,
                    initial_wait=1.0,
                    max_retries=8,
                    stop_conditions={
                        "ignore_eos": True,  # Don't stop on EOS token
                        "max_tokens": 10,  # Generate exactly 10 tokens
                    },
                    worker_id=worker_id,
                )
                block_count += 1

            # Wait for all requests to complete and for internal synchronization
            await asyncio.sleep(1)

            # Extract states from the router (await dump_events!)
            final_events_json = await kv_push_router.dump_events()
            final_worker_statuses = get_worker_status_from_events_by_ids(
                final_events_json, mocker_worker_ids
            )
            logger.info(
                f"Worker statuses after the first test: {final_worker_statuses}"
            )

            return (
                mocker_worker_ids,
                initial_worker_statuses,
                final_worker_statuses,
                expected_added_blocks,
            )

        # Run the async test
        (
            worker_ids,
            initial_worker_statuses,
            final_worker_statuses,
            expected_added_blocks,
        ) = asyncio.run(test_sync())

        # Compare initial vs final block hashes per worker to ensure growth matches expectations
        initial_by_worker = {
            s["worker_id"]: set(s["kv_block_hashes"]) for s in initial_worker_statuses
        }
        final_by_worker = {
            s["worker_id"]: set(s["kv_block_hashes"]) for s in final_worker_statuses
        }

        for wid in worker_ids:
            init_blocks = initial_by_worker.get(wid, set())
            final_blocks = final_by_worker.get(wid, set())
            new_blocks = final_blocks - init_blocks
            expected = expected_added_blocks.get(wid, 0)

            if len(new_blocks) == expected:
                logger.info(
                    f"The request with {len(new_blocks)} blocks of tokens was routed to {wid} as expected with events synchronized successfully."
                    f"Worker {wid}: initial_blocks={len(init_blocks)} final_blocks={len(final_blocks)} new_blocks={len(new_blocks)} expected_added={expected}"
                    f"Initial hashes: {sorted(init_blocks)} Final hashes: {sorted(final_blocks)} New hashes: {sorted(new_blocks)}"
                )
            else:
                raise AssertionError(
                    f"Worker {wid} expected exactly {expected} new blocks but saw {len(new_blocks)}. "
                    f"Initial hashes: {sorted(init_blocks)} Final hashes: {sorted(final_blocks)} New hashes: {sorted(new_blocks)}"
                )

        logger.info(
            "Test router by worker id and check KV events synchronization completed successfully"
        )

    finally:
        # Clean up mockers
        if "mockers" in locals():
            mockers.__exit__(None, None, None)
