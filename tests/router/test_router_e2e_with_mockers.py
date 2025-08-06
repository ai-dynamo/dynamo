# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os

import aiohttp
import pytest

from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_MOCKERS = 2
BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
NUM_REQUESTS = 100
PORT = 8090  # Starting port for mocker instances


class MockerProcess(ManagedProcess):
    """Manages a single mocker engine instance"""

    def __init__(self, request, endpoint: str, mocker_args_file: str):
        command = [
            "python",
            "-m",
            "dynamo.mocker",
            "--model-path",
            MODEL_NAME,
            "--extra-engine-args",
            mocker_args_file,
            "--endpoint",
            endpoint,
        ]

        super().__init__(
            command=command,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        self.endpoint = endpoint


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


@pytest.mark.pre_merge
def test_mocker_kv_router(request, runtime_services):
    """
    Test KV router with multiple mocker engine instances.
    This test doesn't require GPUs and runs quickly for pre-merge validation.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker KV router test")

    # Create mocker args file
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    # Start mocker instances
    mocker_processes = []

    try:
        # Start KV router (frontend)
        frontend_port = PORT
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        kv_router = KVRouterProcess(request, frontend_port)
        kv_router.__enter__()

        for i in range(NUM_MOCKERS):
            # Use unique endpoints for each mocker
            endpoint = "dyn://test-namespace.mocker.generate"
            logger.info(f"Starting mocker instance {i} on endpoint {endpoint}")

            mocker = MockerProcess(request, endpoint, mocker_args_file)
            mocker_processes.append(mocker)

        # Start all mockers
        for mocker in mocker_processes:
            mocker.__enter__()

        # Send test requests
        test_payload = {
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

        # Use async to send requests concurrently for better performance
        asyncio.run(
            send_concurrent_requests(
                [
                    f"http://localhost:{frontend_port}/v1/chat/completions"
                ],  # Pass as list
                test_payload,
                NUM_REQUESTS,
            )
        )

        logger.info(f"Successfully completed {NUM_REQUESTS} requests")

    finally:
        # Clean up
        if "kv_router" in locals():
            kv_router.__exit__(None, None, None)

        for mocker in mocker_processes:
            mocker.__exit__(None, None, None)

        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)


@pytest.mark.pre_merge
def test_mocker_two_kv_router(request, runtime_services):
    """
    Test with two KV routers and multiple mocker engine instances.
    Alternates requests between the two routers to test load distribution.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker two KV router test")

    # Create mocker args file
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    # Start mocker instances
    mocker_processes = []
    kv_routers = []

    try:
        # Start two KV routers (frontend) on ports 8091 and 8092
        router_ports = [PORT + 1, PORT + 2]  # 8091 and 8092

        for port in router_ports:
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(request, port)
            kv_router.__enter__()
            kv_routers.append(kv_router)

        for i in range(NUM_MOCKERS):
            # Use unique endpoints for each mocker
            endpoint = "dyn://test-namespace.mocker.generate"
            logger.info(f"Starting mocker instance {i} on endpoint {endpoint}")

            mocker = MockerProcess(request, endpoint, mocker_args_file)
            mocker_processes.append(mocker)

        # Start all mockers
        for mocker in mocker_processes:
            mocker.__enter__()

        # Send test requests
        test_payload = {
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

        # Build URLs for both routers
        router_urls = [
            f"http://localhost:{port}/v1/chat/completions" for port in router_ports
        ]

        # Use async to send requests concurrently, alternating between routers
        asyncio.run(
            send_concurrent_requests(
                router_urls,
                test_payload,
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
        for mocker in mocker_processes:
            mocker.__exit__(None, None, None)

        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)


async def send_request_with_retry(url: str, payload: dict, max_retries: int = 4):
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


async def send_concurrent_requests(urls: list, payload: dict, num_requests: int):
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
