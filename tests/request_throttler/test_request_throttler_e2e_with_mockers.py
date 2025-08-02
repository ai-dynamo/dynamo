# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time

import aiohttp
import nats
import pytest

from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_MOCKERS = 2
BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
FRONTEND_PORT = 8091
REQUEST_THROTTLE_DURATION_MS = 2000  # 2 seconds for faster testing, this is the duration of the request throttling window
MAX_QUEUE_DEPTH = 1  # Low threshold for easier testing


class RequestThrottlerTestFrontend(ManagedProcess):
    """Frontend with request throttling enabled for testing"""

    def __init__(
        self,
        request,
        frontend_port: int,
        request_throttle_duration_ms: int,
        max_queue_depth: int,
    ):
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
            "--all-workers-busy-rejection-time-window",
            str(request_throttle_duration_ms // 1000),
            "--max-workers-busy-queue-depth",
            str(max_queue_depth),
        ]

        # Force fixed namespace for coordination
        env = os.environ.copy()
        env["DYN_NAMESPACE"] = "test-namespace"

        super().__init__(
            command=command,
            env=env,
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
        """Check if frontend is ready"""
        return response.status_code == 200


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

        env = os.environ.copy()
        env["DYN_NAMESPACE"] = "test-namespace"

        super().__init__(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        self.endpoint = endpoint


async def publish_all_workers_busy_event(nats_url: str = "nats://localhost:4222"):
    """Manually publish KVAllWorkersBusyEvent to trigger request throttling"""
    nc = await nats.connect(nats_url)

    try:
        # Create the event payload
        event_data = {"max_queue_depth": 1, "timestamp": int(time.time())}

        # Publish to the KV all workers busy subject
        subject = "namespace.test-namespace.kv-all-workers-busy"

        payload = json.dumps(event_data).encode()
        await nc.publish(subject, payload)

        logger.info(f"Published all workers busy event to {subject}: {event_data}")

    finally:
        await nc.close()


@pytest.mark.pre_merge
async def test_request_throttler_e2e(request, runtime_services):
    """
    End-to-end test for request throttler functionality:
    1. Start frontend with request throttling enabled
    2. Start mocker backends
    3. Verify normal requests work
    4. Manually publish all-workers-busy event
    5. Verify requests get 503 responses during request throttling
    6. Verify requests return to normal after request throttling expires
    """

    logger.info("Starting request throttler e2e test")

    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    # Start frontend with request throttling enabled
    frontend = RequestThrottlerTestFrontend(
        request, FRONTEND_PORT, REQUEST_THROTTLE_DURATION_MS, MAX_QUEUE_DEPTH
    )
    mocker_processes = []

    try:
        # Start frontend
        logger.info(
            f"Starting request throttling enabled frontend on port {FRONTEND_PORT}"
        )
        frontend.__enter__()

        # Start mocker processes
        for i in range(NUM_MOCKERS):
            endpoint = "dyn://test-namespace.mocker.generate"
            logger.info(f"Starting mocker instance {i} on endpoint {endpoint}")

            mocker = MockerProcess(request, endpoint, mocker_args_file)
            mocker_processes.append(mocker)
            mocker.__enter__()

        # Give system time to fully initialize
        await asyncio.sleep(5)

        base_url = f"http://localhost:{FRONTEND_PORT}/v1/chat/completions"

        # Step 1: Verify normal requests work
        logger.info("=== Testing normal operation ===")
        status, response = await send_test_request(base_url)
        assert status == 200, f"Expected 200, got {status}. Response: {response}"
        logger.info("✓ Normal request successful")

        # Step 2: Publish all-workers-busy event to trigger request throttling
        logger.info("=== Triggering request throttling ===")
        await publish_all_workers_busy_event()

        # Give the request throttler time to process the event
        await asyncio.sleep(0.2)

        # Step 3: Verify requests now get 503 responses
        logger.info("=== Testing request throttling active ===")
        status, response = await send_test_request(base_url, expect_success=False)
        assert (
            status == 503
        ), f"Expected 503 (request throttled), got {status}. Response: {response}"
        logger.info("✓ Request correctly request throttled (503)")

        # Step 4: Send multiple requests to ensure consistent request throttling
        for i in range(3):
            status, _ = await send_test_request(base_url, expect_success=False)
            assert status == 503, f"Request {i+1}: Expected 503, got {status}"
            await asyncio.sleep(0.1)  # Small delay between requests
        logger.info("✓ Multiple requests consistently request throttled")

        # Step 5: Wait for request throttling to expire
        logger.info("=== Waiting for request throttling to expire ===")
        await wait_for_request_throttle_to_clear(
            2 * REQUEST_THROTTLE_DURATION_MS / 1000.0
        )

        # Step 6: Verify requests work again
        logger.info("=== Testing normal operation after request throttling expires ===")
        status, response = await send_test_request(base_url)
        assert (
            status == 200
        ), f"Expected 200 after request throttling expired, got {status}. Response: {response}"
        logger.info("✓ Request successful after request throttling expired")

        # Step 7: Send multiple requests to ensure consistent normal operation
        for i in range(3):
            status, _ = await send_test_request(base_url)
            assert (
                status == 200
            ), f"Request {i+1} after expiry: Expected 200, got {status}"
            await asyncio.sleep(0.1)
        logger.info("✓ Multiple requests successful after request throttling expired")

        logger.info("🎉 Request throttler e2e test completed successfully!")

    finally:
        # Cleanup
        if "frontend" in locals():
            frontend.__exit__(None, None, None)

        for mocker in mocker_processes:
            mocker.__exit__(None, None, None)

        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)


async def send_test_request(
    url: str, expect_success: bool = True, max_retries: int = 4
) -> tuple[int, str]:
    """Send a request with exponential backoff retry for system readiness"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 5,
    }

    wait_time = 1  # Start with 1 second

    for attempt in range(max_retries + 1):
        await asyncio.sleep(wait_time)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    text = await response.text()
                    if response.status == 200:
                        logger.info(f"Request succeeded on attempt {attempt + 1}")
                        return response.status, text
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed with status {response.status}: {text}"
                        )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < max_retries:
            wait_time *= 2  # Double the wait time (exponential backoff)

    # If we get here, all retries failed
    return 404, "Failed after all retries"


async def wait_for_request_throttle_to_clear(duration_seconds: float):
    """Wait for request throttling to clear with some buffer"""
    wait_time = duration_seconds + 0.5  # Add 500ms buffer
    logger.info(f"Waiting {wait_time} seconds for request throttling to clear...")
    await asyncio.sleep(wait_time)
