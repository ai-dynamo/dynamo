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

from tests.conftest import download_models
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_MOCKERS = 2
BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
FRONTEND_PORT = 8091
RATE_LIMIT_DURATION_MS = 2000  # 2 seconds for faster testing
MAX_QUEUE_DEPTH = 1  # Low threshold for easier testing


class RateLimitTestFrontend(ManagedProcess):
    """Frontend with rate limiting enabled for testing"""

    def __init__(
        self,
        request,
        frontend_port: int,
        rate_limit_duration_ms: int,
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
            str(rate_limit_duration_ms / 1000),
            "--max-workers-busy-queue-depth",
            str(max_queue_depth),
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
        """Check if frontend is ready"""
        return response.status_code == 200


class MockerProcess(ManagedProcess):
    """Manages a single mocker engine instance"""

    def __init__(self, request, endpoint: str, mocker_args_file: str):
        command = [
            "python",
            "-m",
            "dynamo.mocker",
            "--kv-cache-block-size",
            str(BLOCK_SIZE),
            "--router-mode",
            "kv",
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


async def publish_all_workers_busy_event(nats_url: str = "nats://localhost:4222"):
    """Manually publish KVAllWorkersBusyEvent to trigger rate limiting"""
    nc = await nats.connect(nats_url)

    try:
        # Create the event payload
        event_data = {"max_queue_depth": 1, "timestamp": int(time.time())}

        # Publish to the KV all workers busy subject
        # Based on scheduler.rs, the full subject would be: namespace.{namespace_name}.kv-all-workers-busy
        subject = "namespace.test-namespace.kv-all-workers-busy"

        payload = json.dumps(event_data).encode()
        await nc.publish(subject, payload)

        logger.info(f"Published all workers busy event to {subject}: {event_data}")

    finally:
        await nc.close()


async def send_test_request(url: str, expect_success: bool = True) -> tuple[int, str]:
    """Send a test HTTP request and return status code and response text"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "max_tokens": 5,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                text = await response.text()
                logger.info(f"Request returned status {response.status}")
                return response.status, text

    except Exception as e:
        logger.error(f"Request failed with error: {e}")
        return 500, str(e)


async def wait_for_rate_limit_to_clear(duration_seconds: float):
    """Wait for rate limit to clear with some buffer"""
    wait_time = duration_seconds + 0.5  # Add 500ms buffer
    logger.info(f"Waiting {wait_time} seconds for rate limit to clear...")
    await asyncio.sleep(wait_time)


@pytest.mark.pre_merge
async def test_rate_limiter_e2e(request, runtime_services):
    """
    End-to-end test for rate limiter functionality:
    1. Start frontend with rate limiting enabled
    2. Start mocker backends
    3. Verify normal requests work
    4. Manually publish all-workers-busy event
    5. Verify requests get 429 responses during rate limiting
    6. Verify requests return to normal after rate limiting expires
    """

    # Download model for this test
    download_models([MODEL_NAME])

    # runtime_services provides NATS and etcd
    nats_process, etcd_process = runtime_services
    logger.info("Starting rate limiter e2e test")

    # Create mocker args file
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    # Start frontend with rate limiting enabled
    frontend = RateLimitTestFrontend(
        request, FRONTEND_PORT, RATE_LIMIT_DURATION_MS, MAX_QUEUE_DEPTH
    )
    mocker_processes = []

    try:
        # Start frontend
        logger.info(f"Starting rate-limited frontend on port {FRONTEND_PORT}")
        frontend.__enter__()

        # Start mocker processes
        for i in range(NUM_MOCKERS):
            endpoint = f"dyn://test-namespace.mocker-{i}.generate"
            logger.info(f"Starting mocker instance {i} on endpoint {endpoint}")

            mocker = MockerProcess(request, endpoint, mocker_args_file)
            mocker_processes.append(mocker)
            mocker.__enter__()

        # Give system time to fully initialize
        await asyncio.sleep(2)

        base_url = f"http://localhost:{FRONTEND_PORT}/v1/chat/completions"

        # Step 1: Verify normal requests work
        logger.info("=== Testing normal operation ===")
        status, response = await send_test_request(base_url)
        assert status == 200, f"Expected 200, got {status}. Response: {response}"
        logger.info("✓ Normal request successful")

        # Step 2: Publish all-workers-busy event to trigger rate limiting
        logger.info("=== Triggering rate limiting ===")
        await publish_all_workers_busy_event()

        # Give the rate limiter time to process the event
        await asyncio.sleep(0.2)

        # Step 3: Verify requests now get 429 responses
        logger.info("=== Testing rate limiting active ===")
        status, response = await send_test_request(base_url, expect_success=False)
        assert (
            status == 429
        ), f"Expected 429 (rate limited), got {status}. Response: {response}"
        logger.info("✓ Request correctly rate limited (429)")

        # Step 4: Send multiple requests to ensure consistent rate limiting
        for i in range(3):
            status, _ = await send_test_request(base_url, expect_success=False)
            assert status == 429, f"Request {i+1}: Expected 429, got {status}"
            await asyncio.sleep(0.1)  # Small delay between requests
        logger.info("✓ Multiple requests consistently rate limited")

        # Step 5: Wait for rate limiting to expire
        logger.info("=== Waiting for rate limit to expire ===")
        await wait_for_rate_limit_to_clear(RATE_LIMIT_DURATION_MS / 1000.0)

        # Step 6: Verify requests work again
        logger.info("=== Testing normal operation after rate limit expires ===")
        status, response = await send_test_request(base_url)
        assert (
            status == 200
        ), f"Expected 200 after rate limit expired, got {status}. Response: {response}"
        logger.info("✓ Request successful after rate limit expired")

        # Step 7: Send multiple requests to ensure consistent normal operation
        for i in range(3):
            status, _ = await send_test_request(base_url)
            assert (
                status == 200
            ), f"Request {i+1} after expiry: Expected 200, got {status}"
            await asyncio.sleep(0.1)
        logger.info("✓ Multiple requests successful after rate limit expired")

        logger.info("🎉 Rate limiter e2e test completed successfully!")

    finally:
        # Cleanup
        if "frontend" in locals():
            frontend.__exit__(None, None, None)

        for mocker in mocker_processes:
            mocker.__exit__(None, None, None)

        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)


@pytest.mark.pre_merge
async def test_rate_limiter_multiple_events(request, runtime_services):
    """
    Test that multiple all-workers-busy events extend the rate limiting period
    """
    download_models([MODEL_NAME])
    nats_process, etcd_process = runtime_services

    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    frontend = RateLimitTestFrontend(
        request, FRONTEND_PORT + 1, RATE_LIMIT_DURATION_MS, MAX_QUEUE_DEPTH
    )
    mocker_processes = []

    try:
        frontend.__enter__()

        # Start one mocker for minimal functionality
        mocker = MockerProcess(
            request, "dyn://test-namespace.mocker.generate", mocker_args_file
        )
        mocker_processes.append(mocker)
        mocker.__enter__()

        await asyncio.sleep(2)

        base_url = f"http://localhost:{FRONTEND_PORT + 1}/v1/chat/completions"

        # Trigger first rate limiting event
        logger.info("=== Triggering first rate limiting event ===")
        await publish_all_workers_busy_event()
        await asyncio.sleep(0.2)

        # Verify rate limiting is active
        status, _ = await send_test_request(base_url, expect_success=False)
        assert status == 429, f"Expected 429 after first event, got {status}"

        # Wait halfway through the rate limit period, then send another event
        await asyncio.sleep(RATE_LIMIT_DURATION_MS / 2000.0)

        logger.info(
            "=== Triggering second rate limiting event (should extend period) ==="
        )
        await publish_all_workers_busy_event()
        await asyncio.sleep(0.2)

        # Should still be rate limited
        status, _ = await send_test_request(base_url, expect_success=False)
        assert status == 429, f"Expected 429 after second event, got {status}"

        # Wait for the original duration from the first event - should still be limited
        # because the second event reset the timer
        await asyncio.sleep(RATE_LIMIT_DURATION_MS / 2000.0)
        status, _ = await send_test_request(base_url, expect_success=False)
        assert status == 429, f"Expected 429 (period should be extended), got {status}"

        # Wait for the full duration from the second event
        await wait_for_rate_limit_to_clear(RATE_LIMIT_DURATION_MS / 1000.0)

        # Should now work
        status, _ = await send_test_request(base_url)
        assert status == 200, f"Expected 200 after extended period, got {status}"

        logger.info("✓ Multiple events correctly extend rate limiting period")

    finally:
        if "frontend" in locals():
            frontend.__exit__(None, None, None)
        for mocker in mocker_processes:
            mocker.__exit__(None, None, None)
        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)


@pytest.mark.pre_merge
async def test_rate_limiter_disabled(request, runtime_services):
    """
    Test that when rate limiting is disabled, all-workers-busy events don't affect requests
    """
    download_models([MODEL_NAME])
    nats_process, etcd_process = runtime_services

    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    mocker_args_file = os.path.join(request.node.name, "mocker_args.json")
    with open(mocker_args_file, "w") as f:
        json.dump(mocker_args, f)

    # Create frontend WITHOUT rate limiting environment variable
    command = [
        "python",
        "-m",
        "dynamo.frontend",
        "--kv-cache-block-size",
        str(BLOCK_SIZE),
        "--router-mode",
        "kv",
        "--http-port",
        str(FRONTEND_PORT + 2),
    ]

    frontend = ManagedProcess(
        command=command,
        timeout=60,
        display_output=True,
        health_check_ports=[FRONTEND_PORT + 2],
        health_check_urls=[
            (
                f"http://localhost:{FRONTEND_PORT + 2}/v1/models",
                lambda r: r.status_code == 200,
            )
        ],
        log_dir=request.node.name,
        terminate_existing=False,
    )

    try:
        frontend.__enter__()

        mocker = MockerProcess(
            request, "dyn://test-namespace.mocker.generate", mocker_args_file
        )
        mocker.__enter__()

        await asyncio.sleep(2)

        base_url = f"http://localhost:{FRONTEND_PORT + 2}/v1/chat/completions"

        # Verify normal operation
        status, _ = await send_test_request(base_url)
        assert status == 200, f"Expected 200, got {status}"

        # Publish all-workers-busy event
        logger.info("=== Publishing event with rate limiting disabled ===")
        await publish_all_workers_busy_event()
        await asyncio.sleep(0.5)

        # Should still work (not rate limited)
        status, _ = await send_test_request(base_url)
        assert status == 200, f"Expected 200 (rate limiting disabled), got {status}"

        logger.info("✓ Rate limiting correctly disabled")

    finally:
        if "frontend" in locals():
            frontend.__exit__(None, None, None)
        if "mocker" in locals():
            mocker.__exit__(None, None, None)
        if os.path.exists(mocker_args_file):
            os.unlink(mocker_args_file)
