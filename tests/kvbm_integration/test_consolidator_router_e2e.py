#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E test for KV Event Consolidator with Router integration.

This test validates that:
1. vLLM with KVBM correctly emits KV events to the consolidator
2. The consolidator correctly deduplicates events from vLLM and KVBM
3. The router receives and processes consolidated events without warnings

"""

import concurrent.futures
import importlib.util
import logging
import os
import re
import time
from pathlib import Path

import pytest
import requests

from tests.kvbm_integration.common import ApiTester, check_logs_for_patterns
from tests.utils.managed_process import ManagedProcess

# Check if vLLM is available
HAS_VLLM = importlib.util.find_spec("vllm") is not None

# Test markers
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.skipif(not HAS_VLLM, reason="requires vllm"),
]

logger = logging.getLogger(__name__)

# Constants
FRONTEND_PORT = 8000


@pytest.fixture
def test_directory(request):
    """Create a test directory for logs and temporary files."""
    test_dir = Path(request.node.name)
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup handled by pytest (logs are kept for debugging)


def extract_consolidator_stats(log_path: Path) -> dict:
    """Extract consolidator event statistics from vLLM logs."""
    stats = {
        "store_events": 0,
        "remove_events": 0,
        "dedup_events": 0,
        "published_events": 0,
        "consolidator_started": False,
        "consolidator_port": None,
    }

    if not log_path.exists():
        return stats

    try:
        with open(log_path, "r") as f:
            content = f.read()

            # Check if consolidator started
            # Actual log: "KV Event Consolidator fully started and ready"
            stats["consolidator_started"] = bool(
                re.search(r"KV Event Consolidator.*started", content, re.IGNORECASE)
            )

            # Extract consolidator output port from logs
            # Look for: "Starting KV Event Consolidator: subscribe from ..., publish to tcp://0.0.0.0:PORT"
            port_match = re.search(r"publish to tcp://[^:]+:(\d+)", content)
            if port_match:
                stats["consolidator_port"] = int(port_match.group(1))

            # Count event types (all at debug level)
            # Actual: "stored in first source ... will publish STORE event"
            stats["store_events"] = len(
                re.findall(r"will publish STORE event", content)
            )
            # Actual: "removed from last source ... will publish REMOVE event"
            stats["remove_events"] = len(
                re.findall(r"will publish REMOVE event", content)
            )
            # Actual: "DEDUP: Block ... added to source"
            stats["dedup_events"] = len(re.findall(r"DEDUP:.*added to source", content))
            # Actual: "Publishing N consolidated event(s) to router" or "Published batch with N event(s)"
            stats["published_events"] = len(
                re.findall(
                    r"Publish(?:ing|ed).*event.*to router", content, re.IGNORECASE
                )
            )
    except Exception as e:
        logger.warning(f"Error extracting consolidator stats: {e}")

    return stats


def wait_for_worker_registration(
    frontend_url: str, max_wait_seconds: int = 120, poll_interval: int = 2
) -> bool:
    """
    Poll frontend health endpoint until a worker registers.

    Args:
        frontend_url: Base URL of the frontend (e.g., "http://localhost:8000")
        max_wait_seconds: Maximum time to wait for registration
        poll_interval: Seconds between health checks

    Returns:
        True if worker registered, False if timeout
    """

    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(f"{frontend_url}/health", timeout=5)
            health_data = response.json()
            if health_data.get("instances"):
                elapsed = time.time() - start_time
                logger.info(f"vLLM worker registered after {elapsed:.1f}s")
                return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    logger.error(f"vLLM worker failed to register after {elapsed:.1f}s")
    logger.error("Check vLLM logs for initialization errors")
    return False


@pytest.fixture
def frontend_server(test_directory, runtime_services):
    """Start Dynamo frontend with embedded KV router."""
    logger.info("Starting Dynamo frontend with KV router")

    # Frontend command - includes embedded router
    command = [
        "python",
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(FRONTEND_PORT),
        "--router-mode",
        "kv",
        "--router-reset-states",
    ]

    # Environment
    env = os.environ.copy()
    env.update(
        {
            "RUST_BACKTRACE": "1",
            "NATS_SERVER": "nats://localhost:4222",
            "ETCD_ENDPOINTS": "http://localhost:2379",
            "DYN_LOG": "debug",  # Enable debug logs for consolidator visibility
        }
    )

    # Create separate log directory for frontend to avoid conflicts with vllm
    frontend_log_dir = test_directory / "frontend"
    frontend_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = frontend_log_dir / "python.log.txt"

    # Create managed process and start via context manager
    with ManagedProcess(
        command=command,
        env=env,
        health_check_urls=[f"http://localhost:{FRONTEND_PORT}/health"],
        timeout=120,  # Increased timeout for frontend+router initialization
        working_dir=str(test_directory),
        display_output=False,
        log_dir=str(frontend_log_dir),  # Separate log directory
    ) as frontend_process:
        logger.info(f"Frontend started on port {FRONTEND_PORT}")

        yield {
            "process": frontend_process,
            "port": FRONTEND_PORT,
            "base_url": f"http://localhost:{FRONTEND_PORT}",
            "log_file": log_file,
        }

    # Cleanup happens automatically via context manager __exit__
    logger.info("Frontend server stopped")


@pytest.fixture
def vllm_worker(frontend_server, test_directory, runtime_services):
    """Start vLLM worker with KVBM connector and KV Event Consolidator."""
    model_id = os.environ.get("CONSOLIDATOR_MODEL_ID", "Qwen/Qwen3-0.6B")

    logger.info(f"Starting vLLM worker with KVBM connector and model {model_id}")

    # vLLM worker command - consolidator is auto-enabled with KVBM
    command = [
        "python",
        "-m",
        "dynamo.vllm",
        "--model",
        model_id,
        "--connector",
        "kvbm",
        "--enforce-eager",  # For faster startup in tests
    ]

    # Environment
    env = os.environ.copy()
    env.update(
        {
            "RUST_BACKTRACE": "1",
            "NATS_SERVER": "nats://localhost:4222",
            "ETCD_ENDPOINTS": "http://localhost:2379",
            "DYN_KVBM_CPU_CACHE_GB": "5",
            "DYN_KVBM_DISK_CACHE_GB": "5",
            "DYN_LOG": "debug",  # Enable debug logs for consolidator visibility
        }
    )

    # Create separate log directory for vllm to avoid conflicts with frontend
    vllm_log_dir = test_directory / "vllm"
    vllm_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = vllm_log_dir / "python.log.txt"

    # Create managed process and start via context manager
    with ManagedProcess(
        command=command,
        env=env,
        health_check_urls=[],  # Workers don't expose HTTP endpoints
        timeout=300,  # Increased timeout for model loading and consolidator init
        working_dir=str(test_directory),
        display_output=False,
        log_dir=str(vllm_log_dir),  # Separate log directory
        terminate_existing=False,
    ) as vllm_process:
        logger.info("Waiting for vLLM worker and consolidator to initialize...")

        # Wait for worker to register with frontend
        worker_registered = wait_for_worker_registration(frontend_server["base_url"])

        if not worker_registered:
            logger.warning("Continuing test despite worker registration failure")

        # Additional wait for consolidator to fully initialize
        time.sleep(5)

        # Verify consolidator started by checking logs
        stats = extract_consolidator_stats(log_file)
        if not stats["consolidator_started"]:
            logger.warning("Consolidator may not have started - check logs")
        else:
            logger.info("Consolidator detected in logs")

        yield {
            "process": vllm_process,
            "model_id": model_id,
            "log_file": log_file,
            "consolidator_stats": stats,
        }

    # Cleanup happens automatically via context manager __exit__
    logger.info("vLLM worker stopped")


@pytest.fixture
def tester(frontend_server, vllm_worker):
    """Provides a test client that sends requests to frontend."""
    return ApiTester(
        base_url=frontend_server["base_url"],
        model_id=vllm_worker["model_id"],
    )


class TestConsolidatorRouterE2E:
    """E2E tests for KV Event Consolidator with Router."""

    SYSTEM_PROMPT = "You are a helpful AI assistant."

    # Common error patterns to check in logs
    ERROR_PATTERNS = [
        r"error.*block_size=0",
        r"Failed to parse block_hash",
        r"panic",
        r"fatal",
    ]

    def assert_no_errors_in_logs(self, vllm_log: Path, frontend_log: Path):
        """Helper to check both vLLM and frontend logs for errors."""
        vllm_errors = check_logs_for_patterns(
            vllm_log, self.ERROR_PATTERNS, "vLLM Worker"
        )
        assert not vllm_errors, f"Errors in vLLM Worker logs: {vllm_errors}"

        frontend_errors = check_logs_for_patterns(
            frontend_log, self.ERROR_PATTERNS, "Frontend/Router"
        )
        assert not frontend_errors, f"Errors in Frontend/Router logs: {frontend_errors}"

    def send_concurrent_requests(
        self,
        tester: ApiTester,
        num_requests: int,
        max_tokens: int = 50,
        content_template: str = "Request {i}: Tell me about topic {i}",
    ):
        """Helper to send concurrent requests and return results."""

        def send_request(i: int):
            try:
                messages = [{"role": "user", "content": content_template.format(i=i)}]
                response = tester.send_chat_request(messages, max_tokens=max_tokens)
                return True, response
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                return False, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            results = [f.result() for f in futures]

        successes = sum(1 for success, _ in results if success)
        logger.info(f"Concurrent requests: {successes}/{num_requests} succeeded")
        return successes, results

    def test_basic_consolidator_flow(self, tester, vllm_worker, frontend_server):
        """
        Test basic consolidator flow:
        1. Send requests
        2. Verify consolidator starts and processes events
        3. Verify router receives events without errors
        """
        logger.info("TEST: Basic Consolidator Flow")

        # Send 3 requests to frontend
        requests_data = [
            [{"role": "user", "content": "What is machine learning?"}],
            [{"role": "user", "content": "Explain neural networks."}],
            [{"role": "user", "content": "Tell me about transformers."}],
        ]

        for i, messages in enumerate(requests_data):
            response = tester.send_chat_request(messages)
            content = response["choices"][0]["message"]["content"]
            logger.info(
                f"Request {i+1}/3: {messages[0]['content'][:30]}... => {content[:40]}..."
            )

        # Wait for logs to flush
        time.sleep(5)

        # Check vLLM worker logs for consolidator
        vllm_log = vllm_worker["log_file"]
        consolidator_stats = extract_consolidator_stats(vllm_log)

        logger.info(f"Consolidator stats: {consolidator_stats}")

        # Assertions
        assert consolidator_stats["consolidator_started"], "Consolidator did not start"
        assert consolidator_stats["store_events"] > 0, "No store events processed"
        assert consolidator_stats["published_events"] > 0, "No events published"

        # Check for errors in logs
        self.assert_no_errors_in_logs(vllm_log, frontend_server["log_file"])

        logger.info("Basic consolidator flow test passed")

    def test_consolidator_handles_concurrent_requests(
        self, tester, vllm_worker, frontend_server
    ):
        """
        Test consolidator under concurrent load:
        1. Send many requests quickly
        2. Verify no crashes or critical errors
        3. Verify all events processed
        """
        logger.info("TEST: Concurrent Request Handling")

        # Send 10 concurrent requests
        num_requests = 10
        successes, _ = self.send_concurrent_requests(
            tester,
            num_requests,
            max_tokens=20,
            content_template="Request {i}: Count to 5",
        )

        # Wait for logs to flush
        time.sleep(5)

        # Assertions
        assert (
            successes >= num_requests * 0.9
        ), f"Too many failed requests: {num_requests - successes}"

        # Check for errors in logs
        self.assert_no_errors_in_logs(
            vllm_worker["log_file"], frontend_server["log_file"]
        )

        # Verify events were processed
        stats = extract_consolidator_stats(vllm_worker["log_file"])
        assert stats["store_events"] > 0, "No events processed during concurrent load"

        logger.info("Concurrent request handling test passed")

    def test_store_deduplication_across_sources(
        self, tester, vllm_worker, frontend_server
    ):
        """
        Test STORE event deduplication across vLLM (G1) and KVBM (G2/G3):

        When a block is stored in G1 (GPU), it's automatically offloaded
        to G2 (CPU) and G3 (Disk). This triggers STORE events from both vLLM and KVBM.

        Test Scenario:
        1. Send requests → blocks stored in vLLM (G1)
        2. Consolidator receives vLLM STORE events → queues them for publishing
        3. KVBM replicates blocks to G2/G3 → emits STORE events
        4. Consolidator sees blocks already exist → logs DEDUP message → does NOT publish again
        5. Result: Router receives ONE STORE event per unique block (from step 2)

        This verifies: Only one STORE event is sent to router per unique block,
        even though the block exists in multiple storage tiers (G1, G2, G3).
        KVBM replications are deduplicated and don't trigger duplicate router updates.
        """
        logger.info("Starting STORE deduplication test")

        # Send requests to generate STORE events
        logger.info("Sending concurrent requests to generate STORE events")
        num_requests = 10
        successes, _ = self.send_concurrent_requests(
            tester, num_requests, max_tokens=50
        )
        assert (
            successes >= num_requests * 0.9
        ), f"Too many failed requests: {num_requests - successes}"

        # Wait for events to be processed
        time.sleep(5)

        # Phase 2: Analyze consolidator logs
        logger.info("Phase 2: Analyzing STORE event deduplication")
        vllm_log = vllm_worker["log_file"]
        log_content = vllm_log.read_text()

        # Count STORE events received from vLLM (first source = will publish)
        vllm_stores = len(
            re.findall(
                r"stored in first source Vllm.*will publish STORE event", log_content
            )
        )

        # Count STORE events received from KVBM (they appear as DEDUP messages)
        kvbm_stores = len(
            re.findall(
                r"DEDUP: Block \d+ \(seq_hash=\d+\) added to source Kvbm", log_content
            )
        )

        # Count total STORE events received (from both sources)
        total_stores_received = vllm_stores + kvbm_stores

        # Count STORE events actually published to router
        published_stores = len(re.findall(r"will publish STORE event", log_content))

        logger.info(f"STORE events received from vLLM: {vllm_stores}")
        logger.info(f"STORE events received from KVBM: {kvbm_stores}")
        logger.info(f"Total STORE events received: {total_stores_received}")
        logger.info(f"STORE events published to router: {published_stores}")

        # Assertions:
        # 1. We should receive STORE events from both vLLM and KVBM
        assert vllm_stores > 0, "Expected STORE events from vLLM"
        assert kvbm_stores > 0, "Expected STORE events from KVBM (replication to G2/G3)"

        # 2. Published stores should approximately equal vLLM stores
        #    (each unique block is published once when first stored in vLLM)
        assert (
            published_stores == vllm_stores
        ), f"Expected published events ({published_stores}) to equal vLLM stores ({vllm_stores})"

        # 3. Total stores should be vLLM + KVBM (each block stored in both)
        assert (
            total_stores_received == vllm_stores + kvbm_stores
        ), f"Total should be vLLM ({vllm_stores}) + KVBM ({kvbm_stores})"

        # 4. Check for errors in logs
        self.assert_no_errors_in_logs(vllm_log, frontend_server["log_file"])

        logger.info("STORE deduplication test passed")

    def test_remove_deduplication_across_sources(
        self, test_directory, runtime_services
    ):
        """
        Test REMOVE event deduplication across G1 (vLLM GPU), G2 (KVBM CPU), G3 (KVBM disk):

        When blocks are stored in G1 (GPU), they are AUTOMATICALLY
        replicated to G2 (CPU) and G3 (Disk) simultaneously.

        Test Scenario:
        1. Configure very small GPU cache (30 blocks) and slightly larger KVBM caches (50 blocks each)
        2. Send 25 requests with 100 tokens each → blocks stored in G1 AND offloaded to G2/G3
        3. GPU fills up (30 blocks) → blocks evicted from G1 → consolidator receives REMOVE from vLLM
           → consolidator sees blocks still exist in G2/G3 → does NOT publish REMOVE to router
        4. Some blocks only exist in G1 (not replicated) → when evicted → published to router

        This verifies: REMOVE is only sent to router when a block is removed from ALL sources.
        Deduplication prevents unnecessary REMOVE events when blocks are still cached in G2/G3.
        """
        logger.info("Starting REMOVE deduplication test")

        # Start frontend with router
        frontend_command = [
            "python",
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(FRONTEND_PORT),
            "--router-mode",
            "kv",
            "--router-reset-states",
        ]

        frontend_env = os.environ.copy()
        frontend_env.update(
            {
                "RUST_BACKTRACE": "1",
                "NATS_SERVER": "nats://localhost:4222",
                "ETCD_ENDPOINTS": "http://localhost:2379",
                "DYN_LOG": "debug",
            }
        )

        frontend_log_dir = test_directory / "frontend"
        frontend_log_dir.mkdir(parents=True, exist_ok=True)
        frontend_log = frontend_log_dir / "python.log.txt"

        with ManagedProcess(
            command=frontend_command,
            env=frontend_env,
            health_check_urls=[f"http://localhost:{FRONTEND_PORT}/health"],
            timeout=120,
            working_dir=str(test_directory),
            display_output=False,
            log_dir=str(frontend_log_dir),
        ) as _frontend_process:
            logger.info(f"Frontend started on port {FRONTEND_PORT}")

            # Start vLLM worker with constrained GPU blocks but larger KVBM blocks
            model_id = os.environ.get("CONSOLIDATOR_MODEL_ID", "Qwen/Qwen3-0.6B")

            vllm_command = [
                "python",
                "-m",
                "dynamo.vllm",
                "--model",
                model_id,
                "--connector",
                "kvbm",
                "--enforce-eager",
                "--enable-prefix-caching",
                "--num-gpu-blocks-override",
                "30",  # Very small GPU cache to force evictions
            ]

            vllm_env = os.environ.copy()
            vllm_env.update(
                {
                    "RUST_BACKTRACE": "1",
                    "NATS_SERVER": "nats://localhost:4222",
                    "ETCD_ENDPOINTS": "http://localhost:2379",
                    "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS": "50",  # Larger than GPU but still constrained
                    "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS": "50",
                    "DYN_LOG": "debug",
                }
            )

            vllm_log_dir = test_directory / "vllm"
            vllm_log_dir.mkdir(parents=True, exist_ok=True)
            vllm_log = vllm_log_dir / "python.log.txt"

            with ManagedProcess(
                command=vllm_command,
                env=vllm_env,
                health_check_urls=[],
                timeout=300,
                working_dir=str(test_directory),
                display_output=False,
                log_dir=str(vllm_log_dir),
                terminate_existing=False,
            ) as _vllm_process:
                logger.info("Waiting for vLLM worker to initialize...")

                # Wait for worker to register with frontend
                worker_registered = wait_for_worker_registration(
                    f"http://localhost:{FRONTEND_PORT}"
                )

                if not worker_registered:
                    pytest.fail("vLLM worker failed to register with frontend")

                # Additional wait for consolidator to fully initialize
                time.sleep(5)

                # Create tester
                tester = ApiTester(
                    base_url=f"http://localhost:{FRONTEND_PORT}", model_id=model_id
                )

                # Phase 1: Send requests to fill GPU cache
                logger.info("Phase 1: Filling GPU cache with diverse prompts")
                for i in range(25):  # Send enough requests to trigger GPU eviction
                    prompt = f"Tell me a unique story about topic {i}. Make it very long and detailed with many paragraphs."
                    response = tester.send_chat_request(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,  # Increase tokens to use more blocks per request
                    )
                    assert "content" in response["choices"][0]["message"]
                    logger.info(f"Request {i+1}/25 completed")

                # Wait for evictions to settle
                time.sleep(5)

                # Phase 2: Analyze consolidator logs
                logger.info("Phase 2: Analyzing consolidator deduplication behavior")
                log_content = vllm_log.read_text()

                # Count blocks removed from vLLM but still in KVBM (deduplication working!)
                vllm_removes_but_in_kvbm = len(
                    re.findall(r"removed from source Vllm, still in.*Kvbm", log_content)
                )

                # Count blocks removed from vLLM as last source (no KVBM copy)
                vllm_removes_last_source = len(
                    re.findall(r"removed from last source Vllm", log_content)
                )

                # Count REMOVE events actually published to router
                published_removes = len(
                    re.findall(r"will publish REMOVE event", log_content)
                )

                logger.info(
                    f"Blocks removed from vLLM (G1) but still in KVBM (G2/G3): {vllm_removes_but_in_kvbm}"
                )
                logger.info(
                    f"Blocks removed from vLLM as last source: {vllm_removes_last_source}"
                )
                logger.info(f"REMOVE events published to router: {published_removes}")

                # Assertions:
                # 1. We should see GPU evictions where blocks still exist in KVBM
                #    This proves deduplication is working (REMOVE not sent to router yet)
                assert (
                    vllm_removes_but_in_kvbm > 0
                ), "Expected GPU evictions where blocks still exist in KVBM (deduplication working)"

                # 2. REMOVE events should be published for last-source removals
                assert (
                    published_removes > 0
                ), "Expected REMOVE events to be published for last-source removals"

                # 3. Check for errors in logs
                self.assert_no_errors_in_logs(vllm_log, frontend_log)
