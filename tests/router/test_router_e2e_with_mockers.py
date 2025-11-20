# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import random
import tempfile
from typing import Any, Dict, Optional

import aiohttp
import nats
import pytest

from dynamo._core import DistributedRuntime, KvPushRouter, KvRouterConfig
from tests.router.common import (  # utilities
    BLOCK_SIZE,
    KVRouterProcess,
    _test_python_router_bindings,
    _test_router_basic,
    _test_router_decisions,
    _test_router_overload_503,
    _test_router_query_instance_id,
    _test_router_two_routers,
    check_nats_consumers,
    generate_random_suffix,
    send_request_via_python_kv_router,
    send_request_with_retry,
)
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge


logger = logging.getLogger(__name__)


@pytest.fixture
def file_storage_backend():
    """Fixture that sets up and tears down file storage backend.

    Creates a temporary directory for file-based KV storage and sets
    the DYN_FILE_KV environment variable. Cleans up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        old_env = os.environ.get("DYN_FILE_KV")
        os.environ["DYN_FILE_KV"] = tmpdir
        logger.info(f"Set up file storage backend in: {tmpdir}")
        yield tmpdir
        # Cleanup
        if old_env is not None:
            os.environ["DYN_FILE_KV"] = old_env
        else:
            os.environ.pop("DYN_FILE_KV", None)


MODEL_NAME = ROUTER_MODEL_NAME
NUM_MOCKERS = 2
SPEEDUP_RATIO = 10.0
PORT = 8090  # Starting port for mocker instances
NUM_REQUESTS = 100

TEST_PAYLOAD = {
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
        store_backend: str = "etcd",
    ):
        # Generate a unique namespace suffix shared by all mockers
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "mocker"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_mockers = num_mockers
        self.num_workers = self.num_mockers  # for compatibility with common.py
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
                "--store-kv",
                store_backend,
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


def get_runtime(store_backend="etcd", request_plane="nats"):
    """Create a DistributedRuntime instance for testing.

    Args:
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: How frontend talks to backend ("tcp", "http" or "nats). Defaults to "nats".
    """
    try:
        # Try to get running loop (works in async context)
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one (sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return DistributedRuntime(loop, store_backend, request_plane)


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


async def wait_for_mockers_ready(
    endpoint, router: KvPushRouter, expected_num_workers: int = NUM_MOCKERS
) -> list[int]:
    """Wait for mocker workers to be ready and return their instance IDs.

    This function polls the endpoint's client for instance IDs until the expected
    number of workers are available, then sends a warmup request to verify they
    can handle requests.

    Args:
        endpoint: The endpoint object to get the client from
        router: The KvPushRouter to use for sending warmup requests
        expected_num_workers: Number of workers to wait for (default: NUM_MOCKERS)

    Returns:
        Sorted list of unique instance IDs (ints).

    Raises:
        AssertionError: If workers don't become ready or warmup request fails.
    """
    logger.info("Waiting for mockers to be ready")

    # Get the client from the endpoint
    client = await endpoint.client()

    # Poll for instance IDs until we have the expected number
    instance_ids: list[int] = []
    max_wait_time = 60  # seconds
    start_time = asyncio.get_event_loop().time()

    while len(instance_ids) < expected_num_workers:
        instance_ids = client.instance_ids()
        logger.info(f"Found {len(instance_ids)} instance(s): {instance_ids}")

        if len(instance_ids) >= expected_num_workers:
            break

        # Check timeout
        if asyncio.get_event_loop().time() - start_time > max_wait_time:
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
            model_name=MODEL_NAME,
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

        kv_router = KVRouterProcess(request, frontend_port, store_backend="etcd")
        kv_router.__enter__()

        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Run basic router test (mocker workers don't need frontend readiness check)
        _test_router_basic(
            frontend_port=frontend_port,
            num_workers=NUM_MOCKERS,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            wait_for_frontend=False,  # Mocker workers are fast, no need to wait
        )

    finally:
        # Clean up
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)

        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
@pytest.mark.parametrize("store_backend", ["etcd", "file"])
def test_mocker_two_kv_router(
    request,
    runtime_services,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
):
    """
    Test with two KV routers and multiple mocker engine instances.
    Alternates requests between the two routers to test load distribution.
    Tests with both etcd and file storage backends.
    """

    # runtime_services starts etcd and nats
    logger.info(
        f"Starting mocker two KV router test with {store_backend} storage backend"
    )

    # Create mocker args dictionary: FixtureRequest: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    kv_routers = []

    try:
        # Start two KV routers (frontend) on ports 8091 and 8092
        router_ports = [PORT + 1, PORT + 2]  # 8091 and 8092

        for port in router_ports:
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(request, port, store_backend=store_backend)
            kv_router.__enter__()
            kv_routers.append(kv_router)

        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            store_backend=store_backend,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Run two router test with consumer lifecycle verification
        _test_router_two_routers(
            engine_workers=mockers,
            kv_routers=kv_routers,
            block_size=BLOCK_SIZE,
            request=request,
            router_ports=router_ports,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
        )

        # Clear the kv_routers list since _test_router_two_routers already cleaned them up
        kv_routers = []

    finally:
        # Clean up mockers
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
@pytest.mark.skip(reason="Flaky, temporarily disabled")
def test_mocker_kv_router_overload_503(
    request, runtime_services, predownload_tokenizers
):
    """Test that KV router returns 503 when mocker workers are overloaded."""
    logger.info("Starting mocker KV router overload test for 503 status")
    # Create mocker args dictionary with limited resources
    mocker_args = {
        "speedup_ratio": 10,
        "block_size": 4,  # Smaller block size
        "num_gpu_blocks": 64,  # Limited GPU blocks to exhaust quickly
    }

    try:
        # Start single mocker instance with limited resources
        logger.info("Starting single mocker instance with limited resources")
        mockers = MockerProcess(request, mocker_args=mocker_args, num_mockers=1)
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Run overload 503 test
        frontend_port = PORT + 10  # Use different port to avoid conflicts
        _test_router_overload_503(
            engine_workers=mockers,
            block_size=4,  # Match the mocker's block size
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            busy_threshold=0.2,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_kv_push_router_bindings(request, runtime_services, predownload_tokenizers):
    """Test KvPushRouter Python bindings with mocker engines."""
    logger.info("Starting KvPushRouter bindings test")
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        namespace = runtime.namespace(mockers.namespace)
        component = namespace.component(mockers.component_name)
        endpoint = component.endpoint("generate")

        # Run Python router bindings test
        _test_python_router_bindings(
            engine_workers=mockers,
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=NUM_MOCKERS,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
@pytest.mark.parametrize("store_backend", ["etcd", "file"])
def test_indexers_sync(
    request,
    runtime_services,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
):
    """
    Test that two KV routers have synchronized indexer states after processing requests.
    This test verifies that both routers converge to the same internal state.
    Tests with both etcd and file storage backends.
    """

    # runtime_services starts etcd and nats
    logger.info(f"Starting indexers sync test with {store_backend} storage backend")

    # Create mocker args dicti: FixtureRequestonary: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            store_backend=store_backend,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        # Initialize mockers
        mockers.__enter__()

        # Use async to manage the test flow
        async def test_sync():
            # Create SEPARATE runtimes for each router to ensure independence
            # This is especially important for file storage backend where connection_id
            # would otherwise be shared between routers
            runtime1 = get_runtime(store_backend)
            runtime2 = get_runtime(store_backend)

            # Use the namespace from the mockers for both runtimes
            namespace1 = runtime1.namespace(mockers.namespace)
            component1 = namespace1.component("mocker")
            endpoint1 = component1.endpoint("generate")

            namespace2 = runtime2.namespace(mockers.namespace)
            component2 = namespace2.component("mocker")
            endpoint2 = component2.endpoint("generate")

            # Create KvRouterConfig with lower snapshot threshold for testing
            kv_router_config = KvRouterConfig(router_snapshot_threshold=20)

            async def send_requests_to_router(router, num_requests, router_name):
                # Now send the actual requests
                tasks = []
                for i in range(num_requests):
                    # Generate random token IDs for each request
                    logger.debug(
                        f"Sending request {i + 1}/{num_requests} to {router_name}"
                    )

                    # Generate 30 random tokens
                    request_tokens = [random.randint(1, 10000) for _ in range(30)]

                    # Send request to mocker via the router
                    tasks.append(
                        asyncio.create_task(
                            send_request_via_python_kv_router(
                                kv_python_router=router,
                                model_name=MODEL_NAME,
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
                endpoint=endpoint1,
                block_size=BLOCK_SIZE,
                kv_router_config=kv_router_config,
            )

            # Wait for mockers to be ready
            await wait_for_mockers_ready(endpoint1, kv_push_router1)

            # Send 25 requests to first router
            logger.info("Sending 25 requests to first router")

            # Send requests to first router
            successful1 = await send_requests_to_router(kv_push_router1, 25, "Router 1")
            assert (
                successful1 == 25
            ), f"Expected 25 successful requests to router 1, got {successful1}"

            # Wait for a second before creating the second router
            logger.info("Waiting for 1 second before creating second router")
            await asyncio.sleep(2)

            # Launch second router - will automatically sync with the first router's state
            logger.info("Creating second KV router")
            kv_push_router2 = KvPushRouter(
                endpoint=endpoint2,
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

            # Check NATS consumers to verify both routers have separate consumers
            await check_nats_consumers(mockers.namespace, expected_count=2)

            # Verify NATS object store bucket was created with snapshot
            # Mirror the Rust bucket naming logic from subscriber.rs:
            # component.subject() -> "namespace.{ns}.component.{comp}"
            # then slugify (convert dots to dashes, lowercase, etc) and append "-radix-bucket"
            component_subject = f"namespace.{mockers.namespace}.component.mocker"
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
        kv_router = KVRouterProcess(request, frontend_port, store_backend="etcd")
        kv_router.__enter__()

        # Start multiple mocker engines to ensure worker selection logic
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        _test_router_query_instance_id(
            engine_workers=mockers,
            kv_router=kv_router,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
        )

    finally:
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.model(MODEL_NAME)
def test_router_decisions(request, runtime_services, predownload_tokenizers):
    """Validate KV cache prefix reuse and dp_rank routing by sending progressive requests with overlapping prefixes."""

    # runtime_services starts etcd and nats
    logger.info("Starting test router prefix reuse and KV events synchronization")

    # Create mocker args dictionary with dp_size=4
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "dp_size": 4,
    }

    try:
        logger.info(
            "Starting 2 mocker instances with dp_size=4 each (8 total dp ranks)"
        )
        mockers = MockerProcess(request, mocker_args=mocker_args, num_mockers=2)
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Initialize mockers
        mockers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        # Use the namespace from the mockers
        namespace = runtime.namespace(mockers.namespace)
        component = namespace.component("mocker")
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            mockers, endpoint, MODEL_NAME, request, test_dp_rank=True
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)
