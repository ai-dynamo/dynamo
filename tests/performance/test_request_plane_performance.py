# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
import random
import string
import sys
import time
from io import BytesIO
from typing import Dict, List, Tuple

import aiohttp
import pytest
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Add workspace to Python path for direct execution
if __name__ == "__main__":
    workspace_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

import tempfile

# Import EtcdServer from conftest since we need both NATS and ETCD
from tests.conftest import EtcdServer
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
BLOCK_SIZE = 16
SPEEDUP_RATIO = (
    1000000.0  # Near-instantaneous response for transport performance testing
)
PORT = 8090

# Default performance test configuration
DEFAULT_MIN_PROMPT_SIZE = 1000  # Start with 1K tokens
DEFAULT_MAX_PROMPT_SIZE = 400000  # Go up to 400K tokens
DEFAULT_STEP_SIZE = 50000  # Increase by 50K each step
DEFAULT_NUM_RUNS_PER_SIZE = 3  # Number of runs to average
DEFAULT_TRANSPORTS = ["tcp", "nats"]  # Default transports to test


# Global configuration (will be set by command line args or pytest)
class PerformanceConfig:
    def __init__(self):
        self.min_prompt_size = DEFAULT_MIN_PROMPT_SIZE
        self.max_prompt_size = DEFAULT_MAX_PROMPT_SIZE
        self.step_size = DEFAULT_STEP_SIZE
        self.num_runs_per_size = DEFAULT_NUM_RUNS_PER_SIZE
        self.transports = DEFAULT_TRANSPORTS.copy()
        self.custom_sizes = None  # List of specific sizes to test
        self.nats_max_payload_mb = 70  # Default NATS max payload in MB


# Function to create config with environment variable override
def create_config():
    """Create configuration, checking environment variables for CLI overrides"""
    cfg = PerformanceConfig()

    # Check for environment variables set by main() function
    if "PERF_TEST_TRANSPORTS" in os.environ:
        cfg.transports = os.environ["PERF_TEST_TRANSPORTS"].split(",")
    if "PERF_TEST_RUNS" in os.environ:
        cfg.num_runs_per_size = int(os.environ["PERF_TEST_RUNS"])
    if "PERF_TEST_NATS_PAYLOAD_MB" in os.environ:
        cfg.nats_max_payload_mb = int(os.environ["PERF_TEST_NATS_PAYLOAD_MB"])
    if "PERF_TEST_SIZES" in os.environ:
        cfg.custom_sizes = [int(x) for x in os.environ["PERF_TEST_SIZES"].split(",")]
    else:
        if "PERF_TEST_MIN_SIZE" in os.environ:
            cfg.min_prompt_size = int(os.environ["PERF_TEST_MIN_SIZE"])
        if "PERF_TEST_MAX_SIZE" in os.environ:
            cfg.max_prompt_size = int(os.environ["PERF_TEST_MAX_SIZE"])
        if "PERF_TEST_STEP_SIZE" in os.environ:
            cfg.step_size = int(os.environ["PERF_TEST_STEP_SIZE"])

    return cfg


# Global config instance - will be configured when running as __main__
config = create_config()


class HighCapacityNatsServer(ManagedProcess):
    """NATS server configured for large payloads (up to 70MB)"""

    def __init__(self, request, port=4222, timeout=300, max_payload_mb=70):
        data_dir = tempfile.mkdtemp(prefix="nats_perf_")

        # Create NATS configuration file with large payload support
        config_file = os.path.join(data_dir, "nats-server.conf")
        max_payload_bytes = max_payload_mb * 1024 * 1024

        config_content = f"""
# NATS Server Configuration for Performance Testing
port: {port}
jetstream: {{
    store_dir: "{data_dir}"
}}

# Set maximum payload size (recommended max: 8MB)
max_payload: {max_payload_bytes}

# Increase max_pending to be >= max_payload (required)
max_pending: {max_payload_bytes * 2}

# Disable trace logging for cleaner output
trace: false
"""

        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Create the config file and verify it exists
        with open(config_file, "w") as f:
            f.write(config_content)

        # Verify the file was created successfully
        if not os.path.exists(config_file):
            raise RuntimeError(f"Failed to create NATS config file: {config_file}")

        logger.info(f"Created NATS config file: {config_file}")
        logger.info(
            f"Config contains max_payload: {max_payload_bytes} bytes ({max_payload_mb}MB)"
        )
        logger.debug(f"NATS config content:\n{config_content}")

        # Double-check file exists and is readable
        try:
            with open(config_file, "r") as f:
                content = f.read()
                logger.debug(
                    f"Verified config file contents: {len(content)} characters"
                )
        except Exception as e:
            logger.error(f"Cannot read config file {config_file}: {e}")
            raise

        # Use configuration file instead of command line flags
        command = ["nats-server", "-c", config_file]

        # Store references to prevent cleanup
        self.max_payload_mb = max_payload_mb
        self.config_file = config_file
        self.data_dir = data_dir

        super().__init__(
            command=command,
            timeout=timeout,
            display_output=True,  # Show NATS logs for debugging
            data_dir=None,  # Don't let ManagedProcess clean up our config directory
            health_check_ports=[port],
            log_dir=f"{request.node.name}_nats_server",
        )

    def __enter__(self):
        # Final verification before starting the server
        if not os.path.exists(self.config_file):
            logger.error(f"Config file missing at startup: {self.config_file}")
            logger.error(
                f"Data directory contents: {os.listdir(self.data_dir) if os.path.exists(self.data_dir) else 'DIR MISSING'}"
            )
            raise RuntimeError(
                f"NATS config file disappeared before server startup: {self.config_file}"
            )

        logger.info(f"Starting NATS server with config file: {self.config_file}")
        result = super().__enter__()
        logger.info(
            f"Started high-capacity NATS server with {self.max_payload_mb}MB max payload (config: {self.config_file})"
        )
        return result


def generate_random_suffix() -> str:
    """Generate a 10-character random alphabetic suffix for namespace isolation."""
    return "".join(random.choices(string.ascii_lowercase, k=10))


def tokens_to_bytes_estimate(num_tokens: int) -> int:
    """Estimate bytes from token count. Average ~4 bytes per token for English text."""
    return num_tokens * 4


def generate_test_content(num_tokens: int) -> str:
    """Generate test content with approximately the specified number of tokens.

    Uses a word-based approximation where 1 token ≈ 0.75 words on average.
    """
    base_words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "in",
        "a",
        "meadow",
        "beneath",
        "golden",
        "sunlight",
        "where",
        "flowers",
        "bloom",
        "and",
        "birds",
        "sing",
        "their",
        "melodious",
        "songs",
        "throughout",
        "peaceful",
        "day",
        "while",
        "gentle",
        "breeze",
        "carries",
        "sweet",
        "fragrance",
        "across",
        "rolling",
        "hills",
        "that",
        "stretch",
        "far",
        "into",
        "distance",
        "under",
        "vast",
        "blue",
        "sky",
        "dotted",
        "with",
        "fluffy",
        "white",
        "clouds",
        "drifting",
        "slowly",
    ]

    # Approximate tokens-to-words ratio (1 token ≈ 0.75 words)
    estimated_words = int(num_tokens * 0.75)

    content_words = []
    for i in range(estimated_words):
        content_words.append(base_words[i % len(base_words)])

    return " ".join(content_words)


def send_large_payload_request_sync(url: str, payload: dict) -> Tuple[float, bool]:
    """Send large payload using requests library as fallback for aiohttp limitations.
    Returns:
        Tuple of (time_taken_seconds, success)
    """
    start_time = time.time()
    try:
        # Configure requests session for large payloads
        session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=[
                "HEAD",
                "GET",
                "OPTIONS",
                "POST",
            ],  # Updated parameter name
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Send request with streaming
        response = session.post(
            url,
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,  # Use streaming to handle large responses
            timeout=(120, 600),  # Connect timeout, read timeout
        )

        end_time = time.time()

        if response.status_code == 200:
            # For streaming responses, consume the stream
            for _ in response.iter_lines():
                pass  # Consume the stream
            return (end_time - start_time, True)
        else:
            logger.warning(f"Request failed with status {response.status_code}")
            return (end_time - start_time, False)

    except Exception as e:
        end_time = time.time()
        logger.error(
            f"Request failed with error after {end_time - start_time:.3f}s: {e}"
        )
        return (end_time - start_time, False)


class MockerProcess:
    """Manages mocker engine instances for performance testing"""

    def __init__(self, request, request_plane: str = "nats", num_mockers: int = 1):
        # Generate a unique namespace suffix
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-perf-{request_plane}-{namespace_suffix}"
        self.endpoint = f"dyn://{self.namespace}.mocker.generate"
        self.num_mockers = num_mockers
        self.request_plane = request_plane
        self.mocker_processes = []

        # Create mocker processes
        for i in range(num_mockers):
            command = [
                "python",
                "-m",
                "dynamo.mocker",
                "--model-path",
                MODEL_NAME,
                "--endpoint",
                self.endpoint,
                "--speedup-ratio",
                str(SPEEDUP_RATIO),
                "--block-size",
                str(BLOCK_SIZE),
            ]

            env = os.environ.copy()
            env["DYN_REQUEST_PLANE"] = request_plane
            env["DYN_LOG"] = "debug"  # Enable trace logging for mocker

            process = ManagedProcess(
                command=command,
                timeout=900,  # 15 minute timeout to match HTTP timeout
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=f"{request.node.name}_{request_plane}_worker_{i}",
                terminate_existing=False,
                env=env,
            )
            self.mocker_processes.append(process)

    def __enter__(self):
        """Start all mocker processes"""
        for i, process in enumerate(self.mocker_processes):
            logger.info(f"Starting {self.request_plane} mocker instance {i}")
            process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all mocker processes"""
        for i, process in enumerate(self.mocker_processes):
            logger.info(f"Stopping {self.request_plane} mocker instance {i}")
            process.__exit__(exc_type, exc_val, exc_tb)


class FrontendProcess(ManagedProcess):
    """Manages the frontend process for performance testing"""

    def __init__(self, request, frontend_port: int, request_plane: str = "nats"):
        command = [
            "python",
            "-m",
            "dynamo.frontend",
            #            "--kv-cache-block-size", str(BLOCK_SIZE),
            #           "--router-mode", "kv",
            "--http-port",
            str(frontend_port),
        ]

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane
        env["DYN_LOG"] = "debug"  # Enable debug logging for frontend
        env[
            "DYN_HTTP_BODY_LIMIT_MB"
        ] = "100"  # Set HTTP body limit to 100MB for large payload testing

        super().__init__(
            command=command,
            timeout=900,  # 15 minute timeout to match HTTP timeout
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_ready)
            ],
            log_dir=f"{request.node.name}_{request_plane}_router",
            terminate_existing=False,
            env=env,
        )
        self.port = frontend_port
        self.request_plane = request_plane

    def _check_ready(self, response):
        """Check if KV router is ready"""
        return response.status_code == 200


async def _process_response(response, start_time: float) -> Tuple[float, bool]:
    """Helper function to process HTTP response for performance measurement."""
    if response.status == 200:
        # Read the streaming response
        chunk_count = 0
        async for chunk in response.content.iter_chunked(8192):  # Read in 8KB chunks
            chunk_count += 1
            if chunk_count % 100 == 0:  # Log progress for very large responses
                logger.debug(f"Processed {chunk_count} response chunks")

        end_time = time.time()
        logger.debug(
            f"Completed request in {end_time - start_time:.3f}s, processed {chunk_count} chunks"
        )
        return (end_time - start_time, True)
    else:
        end_time = time.time()
        try:
            response_text = (
                await response.text()
                if response.content_length and response.content_length < 10000
                else "Response too large to log"
            )
        except Exception:
            response_text = f"Could not read response body (status: {response.status})"
        logger.warning(f"Request failed with status {response.status}: {response_text}")
        return (end_time - start_time, False)


async def send_performance_request(url: str, payload: dict) -> Tuple[float, bool]:
    """Send a single request and measure the time taken.

    Uses optimized settings for large payload handling to avoid HTTP client issues.

    Returns:
        Tuple of (time_taken_seconds, success)
    """
    start_time = time.time()

    # Check payload size - if very large, use synchronous requests library
    json_bytes = json.dumps(payload).encode("utf-8")
    payload_size_mb = len(json_bytes) / (1024 * 1024)

    # Use requests library for payloads > 1MB as aiohttp has buffering issues
    if payload_size_mb > 1.0:
        logger.debug(
            f"Using requests library for large payload ({payload_size_mb:.2f} MB)"
        )
        return send_large_payload_request_sync(url, payload)

    try:
        # Debug: Validate payload structure before sending
        logger.debug(f"Payload keys: {list(payload.keys())}")
        logger.debug(f"Payload model: {payload.get('model', 'MISSING')}")
        logger.debug(
            f"Payload messages type: {type(payload.get('messages', 'MISSING'))}"
        )

        if "messages" in payload and payload["messages"]:
            logger.debug(f"First message keys: {list(payload['messages'][0].keys())}")
            content_preview = payload["messages"][0].get("content", "")[:100]
            logger.debug(f"Content preview: {content_preview}...")

        # Convert payload to JSON bytes
        json_bytes = json.dumps(payload).encode("utf-8")
        payload_size_bytes = len(json_bytes)
        payload_size_mb = payload_size_bytes / (1024 * 1024)

        logger.debug(
            f"Sending payload of {payload_size_bytes:,} bytes ({payload_size_mb:.2f} MB)"
        )

        # Debug: Check first few bytes to ensure JSON is well-formed
        json_preview = json_bytes[:200].decode("utf-8", errors="replace")
        logger.debug(f"JSON preview: {json_preview}...")

        # Create session with optimized settings for large payloads
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=30,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=900,  # 15 minute total timeout (> 10 min as requested)
            connect=120,  # 2 minute connect timeout
            sock_connect=120,  # 2 minute socket connect timeout
            sock_read=600,  # 10 minute socket read timeout
        )

        # Configure session with larger buffer limits for large payloads
        async with aiohttp.ClientSession(
            connector=connector,
            # Increase read buffer size for large responses and requests
            read_bufsize=1024 * 1024 * 100,  # 100MB read buffer
            # Increase request limits
            request_class=aiohttp.ClientRequest,
        ) as session:
            # Use BytesIO for payloads larger than ~500KB to avoid event loop blocking
            payload_size_threshold = 500 * 1024  # 500KB

            if len(json_bytes) > payload_size_threshold:
                reason = "large payload"
                logger.debug(f"Using BytesIO for {reason} ({payload_size_mb:.2f} MB)")

                # Use BytesIO for large payloads - create a custom stream
                data_stream = BytesIO(json_bytes)
                headers = {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(json_bytes)),
                    "Accept": "text/event-stream",
                }

                # Try to force no buffering by setting expectation headers
                headers["Expect"] = ""  # Disable 100-continue expectation

                async with session.post(
                    url,
                    data=data_stream,
                    headers=headers,
                    timeout=timeout,
                    # Disable SSL verification and compression to reduce overhead
                    ssl=False,
                    compress=False,
                ) as response:
                    logger.debug(f"Received response with status {response.status}")
                    return await _process_response(response, start_time)
            else:
                logger.debug(
                    f"Using json parameter for standard payload ({payload_size_mb:.2f} MB)"
                )

                # Use standard json parameter for smaller payloads
                headers = {
                    "Accept": "text/event-stream",
                }

                async with session.post(
                    url,
                    json=payload,  # Let aiohttp handle JSON serialization
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    logger.debug(f"Received response with status {response.status}")
                    return await _process_response(response, start_time)

    except asyncio.TimeoutError as e:
        end_time = time.time()
        logger.error(f"Request timed out after {end_time - start_time:.3f}s: {e}")
        return (end_time - start_time, False)
    except Exception as e:
        end_time = time.time()
        logger.error(
            f"Request failed with error after {end_time - start_time:.3f}s: {e}"
        )
        return (end_time - start_time, False)


async def benchmark_request_plane(
    request_plane: str, payload_sizes: List[int], request, runtime_services
) -> Dict[int, List[float]]:
    """Benchmark a specific request plane with various payload sizes.

    Returns:
        Dict mapping payload size to list of timing measurements
    """
    results = {}

    # Use different ports for TCP vs NATS to avoid conflicts
    frontend_port = PORT + (1 if request_plane == "tcp" else 2)

    try:
        logger.info(f"Starting {request_plane.upper()} performance test")

        # Start frontend
        frontend = FrontendProcess(request, frontend_port, request_plane)
        frontend.__enter__()

        # Start mocker instance
        mockers = MockerProcess(request, request_plane, num_mockers=1)
        mockers.__enter__()

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        # Wait for system to be ready with a small test request
        logger.info(f"Warming up {request_plane.upper()} system...")
        warmup_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 1,
        }

        # Try warmup request with retries
        for attempt in range(10):
            warmup_time, warmup_success = await send_performance_request(
                url, warmup_payload
            )
            if warmup_success:
                logger.info(
                    f"{request_plane.upper()} system ready after {attempt + 1} attempts"
                )
                break
            await asyncio.sleep(2)
        else:
            raise RuntimeError(f"Failed to warm up {request_plane.upper()} system")

        # Test each payload size
        for size in payload_sizes:
            estimated_bytes = tokens_to_bytes_estimate(size)

            logger.info(
                f"Testing {request_plane.upper()} with ~{size:,} tokens (~{estimated_bytes:,} bytes, ~{estimated_bytes/(1024*1024):.1f}MB)..."
            )

            test_content = generate_test_content(size)
            test_payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": test_content}],
                "stream": True,
                "max_tokens": 10,
            }

            size_results = []

            # Run multiple times for this size
            for run in range(config.num_runs_per_size):
                logger.info(f"  Run {run + 1}/{config.num_runs_per_size}")

                timing, success = await send_performance_request(url, test_payload)
                if success:
                    size_results.append(timing)
                    logger.info(f"    Completed in {timing:.3f}s")
                else:
                    logger.warning(f"    Run {run + 1} failed")

                # Small delay between runs
                await asyncio.sleep(1)

            if size_results:
                results[size] = size_results
                avg_time = sum(size_results) / len(size_results)
                logger.info(
                    f"  {request_plane.upper()} ~{size:,} tokens: avg {avg_time:.3f}s ({len(size_results)}/{config.num_runs_per_size} successful)"
                )
            else:
                logger.warning(
                    f"  All runs failed for {request_plane.upper()} with ~{size:,} tokens"
                )
                results[size] = []

    finally:
        # Clean up
        if "frontend" in locals():
            frontend.__exit__(None, None, None)
        if "mockers" in locals():
            mockers.__exit__(None, None, None)

    return results


def print_performance_comparison(
    tcp_results: Dict[int, List[float]], nats_results: Dict[int, List[float]]
):
    """Print a formatted comparison of TCP vs NATS performance"""
    print("\n" + "=" * 95)
    print("REQUEST PLANE PERFORMANCE COMPARISON")
    print("=" * 95)
    print(
        f"{'Payload Size (Tokens & Bytes)':<25} {'TCP Avg (s)':<12} {'NATS Avg (s)':<12} {'Speedup':<10} {'Winner':<8}"
    )
    print("-" * 95)

    for size in sorted(set(tcp_results.keys()) | set(nats_results.keys())):
        tcp_times = tcp_results.get(size, [])
        nats_times = nats_results.get(size, [])

        # Calculate averages for successful runs
        tcp_avg = sum(tcp_times) / len(tcp_times) if tcp_times else None
        nats_avg = sum(nats_times) / len(nats_times) if nats_times else None

        # Format display values
        if tcp_avg is not None:
            tcp_display = f"{tcp_avg:.3f}"
        elif size in tcp_results:  # Test was attempted but failed
            tcp_display = "FAILED"
        else:  # Test not run yet
            tcp_display = "NOT RUN"

        if nats_avg is not None:
            nats_display = f"{nats_avg:.3f}"
        elif size in nats_results:  # Test was attempted but failed
            nats_display = "FAILED"
        else:  # Test not run yet
            nats_display = "NOT RUN"

        # Calculate speedup and winner if both have valid results
        if tcp_avg is not None and nats_avg is not None:
            if tcp_avg < nats_avg:
                speedup = nats_avg / tcp_avg
                winner = "TCP"
            else:
                speedup = tcp_avg / nats_avg
                winner = "NATS"
            speedup_display = f"{speedup:.2f}x"
        elif tcp_avg is not None and nats_avg is None:
            # Only TCP succeeded
            speedup_display = "N/A"
            winner = "TCP*"
        elif tcp_avg is None and nats_avg is not None:
            # Only NATS succeeded
            speedup_display = "N/A"
            winner = "NATS*"
        else:
            # Neither succeeded or both not run
            speedup_display = "N/A"
            winner = "N/A"

        bytes_est = tokens_to_bytes_estimate(size)
        size_display = f"~{size//1000:,}K tokens (~{bytes_est//1024}KB)"
        print(
            f"{size_display:<25} {tcp_display:<12} {nats_display:<12} {speedup_display:<10} {winner:<8}"
        )

    print("=" * 95)
    print("Notes:")
    print("- Payload sizes are approximate (1 token ≈ 0.75 words, ~4 bytes)")
    print(f"- Each test run {config.num_runs_per_size} times and averaged")
    print("- Speedup shows how many times faster the winner is")
    print("- Times include full request/response cycle")
    print("- * indicates only one protocol had successful results")
    print("- FAILED means test was attempted but all runs failed")
    print("- NOT RUN means test was not attempted yet")
    print("=" * 95)


@pytest.mark.performance
@pytest.mark.model(MODEL_NAME)
def test_request_plane_performance_comparison(request, predownload_tokenizers):
    """
    Performance comparison test between TCP and NATS request planes with increasing payload sizes.

    This test:
    1. Sets up identical mocker and router configurations for selected transports
    2. Tests with configurable payload sizes
    3. Measures request/response times for each payload size
    4. Prints a detailed performance comparison table
    5. Helps determine optimal request plane for different use cases

    Configuration can be set via command-line arguments when running directly,
    or will use defaults when run via pytest.
    """
    transports_str = ", ".join(config.transports).upper()
    logger.info(f"Starting {transports_str} performance comparison test")

    # Generate test payload sizes based on configuration
    if config.custom_sizes:
        payload_sizes = sorted(config.custom_sizes)
        size_descriptions = [
            f"~{s//1000}K tokens (~{tokens_to_bytes_estimate(s)//1024}KB)"
            for s in payload_sizes
        ]
        logger.info(f"Testing custom payload sizes: {size_descriptions}")
    else:
        payload_sizes = list(
            range(config.min_prompt_size, config.max_prompt_size + 1, config.step_size)
        )
        size_descriptions = [
            f"~{s//1000}K tokens (~{tokens_to_bytes_estimate(s)//1024}KB)"
            for s in payload_sizes
        ]
        logger.info(f"Testing payload sizes: {size_descriptions}")

    # Start high-capacity NATS server and ETCD server for performance testing
    with HighCapacityNatsServer(
        request, max_payload_mb=config.nats_max_payload_mb
    ) as nats_server:
        with EtcdServer(request) as etcd_server:
            runtime_services = (nats_server, etcd_server)

            # Run benchmarks for selected request planes
            async def run_benchmarks():
                results = {}

                for transport in config.transports:
                    logger.info(f"Benchmarking {transport.upper()} request plane...")
                    results[transport] = await benchmark_request_plane(
                        transport, payload_sizes, request, runtime_services
                    )

                    # Small delay between tests to ensure clean separation
                    if len(config.transports) > 1:
                        await asyncio.sleep(5)

                return results

            all_results = asyncio.run(run_benchmarks())

    # Print detailed comparison (support both single and dual transport testing)
    tcp_results = all_results.get("tcp", {})
    nats_results = all_results.get("nats", {})
    print_performance_comparison(tcp_results, nats_results)

    # Basic assertions to ensure the test actually ran
    assert all_results, "At least one transport test should have produced results"

    # Count successful tests for each transport
    for transport in config.transports:
        transport_results = all_results.get(transport, {})
        if transport_results:
            successful = sum(1 for times in transport_results.values() if times)
            logger.info(
                f"{transport.upper()} successful tests: {successful}/{len(payload_sizes)}"
            )
            assert (
                successful > 0
            ), f"At least one {transport.upper()} test should have succeeded"
        else:
            logger.warning(f"No results found for {transport.upper()} transport")

    logger.info("Performance comparison test completed successfully")


def parse_size_ranges(size_str: str) -> List[int]:
    """Parse size range string like '1000-5000:1000' or '1000,2000,5000'"""
    sizes = []

    for part in size_str.split(","):
        part = part.strip()
        if ":" in part and "-" in part:
            # Range format: start-end:step
            range_part, step_str = part.split(":")
            start_str, end_str = range_part.split("-")
            start, end, step = int(start_str), int(end_str), int(step_str)
            sizes.extend(range(start, end + 1, step))
        elif "-" in part:
            # Range format: start-end (default step of 10000)
            start_str, end_str = part.split("-")
            start, end = int(start_str), int(end_str)
            step = 10000 if end - start > 10000 else max(1000, (end - start) // 5)
            sizes.extend(range(start, end + 1, step))
        else:
            # Single size
            sizes.append(int(part))

    return sorted(list(set(sizes)))  # Remove duplicates and sort


def parse_arguments():
    """Parse command line arguments for the performance test"""
    parser = argparse.ArgumentParser(
        description="TCP vs NATS Performance Comparison Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test only TCP with default sizes
  python test_request_plane_performance.py --transport tcp

  # Test both transports with custom size range
  python test_request_plane_performance.py --sizes 1000-100000:10000

  # Test specific sizes with NATS only
  python test_request_plane_performance.py --transport nats --sizes 1000,5000,10000,50000

  # Quick test with fewer runs
  python test_request_plane_performance.py --runs 1 --sizes 1000,10000

  # Test large payloads only
  python test_request_plane_performance.py --min-size 200000 --max-size 400000 --step 50000
        """,
    )

    # Transport selection
    parser.add_argument(
        "--transport",
        "-t",
        choices=["tcp", "nats", "both"],
        default="both",
        help="Transport(s) to test (default: both)",
    )

    # Payload size configuration
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--sizes",
        "-s",
        type=str,
        help='Custom payload sizes. Format: "1000,5000,10000" or "1000-50000:5000" or "1000-10000"',
    )

    size_group.add_argument(
        "--size-range",
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        type=int,
        help="Size range: min_size max_size step_size",
    )

    # Individual range components (when not using --size-range)
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_PROMPT_SIZE,
        help=f"Minimum payload size in tokens (default: {DEFAULT_MIN_PROMPT_SIZE})",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_PROMPT_SIZE,
        help=f"Maximum payload size in tokens (default: {DEFAULT_MAX_PROMPT_SIZE})",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP_SIZE,
        help=f"Step size between payload sizes (default: {DEFAULT_STEP_SIZE})",
    )

    # Test execution parameters
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=DEFAULT_NUM_RUNS_PER_SIZE,
        help=f"Number of runs per payload size (default: {DEFAULT_NUM_RUNS_PER_SIZE})",
    )

    parser.add_argument(
        "--nats-max-payload-mb",
        type=int,
        default=70,
        help="NATS server max payload size in MB (default: 70)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def configure_from_args(args):
    """Configure the global test configuration from parsed arguments"""
    global config

    # Transport selection
    if args.transport == "both":
        config.transports = ["tcp", "nats"]
    elif args.transport == "tcp":
        config.transports = ["tcp"]
    elif args.transport == "nats":
        config.transports = ["nats"]

    # Payload size configuration
    if args.sizes:
        config.custom_sizes = parse_size_ranges(args.sizes)
    elif args.size_range:
        min_size, max_size, step = args.size_range
        config.custom_sizes = list(range(min_size, max_size + 1, step))
    else:
        # Use individual parameters
        config.min_prompt_size = args.min_size
        config.max_prompt_size = args.max_size
        config.step_size = args.step

    # Test execution parameters
    config.num_runs_per_size = args.runs

    # NATS configuration
    config.nats_max_payload_mb = args.nats_max_payload_mb

    # Logging configuration
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def main():
    """Main execution function that configures and runs the test"""
    global config

    # Parse command line arguments and configure
    args = parse_arguments()
    configure_from_args(args)

    # Set environment variables so pytest subprocess can access the configuration
    os.environ["PERF_TEST_TRANSPORTS"] = ",".join(config.transports)
    os.environ["PERF_TEST_RUNS"] = str(config.num_runs_per_size)
    os.environ["PERF_TEST_NATS_PAYLOAD_MB"] = str(config.nats_max_payload_mb)
    if config.custom_sizes:
        os.environ["PERF_TEST_SIZES"] = ",".join(map(str, config.custom_sizes))
    else:
        os.environ["PERF_TEST_MIN_SIZE"] = str(config.min_prompt_size)
        os.environ["PERF_TEST_MAX_SIZE"] = str(config.max_prompt_size)
        os.environ["PERF_TEST_STEP_SIZE"] = str(config.step_size)

    # Print configuration summary
    print("Performance Test Configuration:")
    print(f"  Transports: {', '.join(config.transports).upper()}")
    print(f"  Runs per size: {config.num_runs_per_size}")
    print(f"  NATS max payload: {config.nats_max_payload_mb}MB")

    if config.custom_sizes:
        size_descriptions = []
        for s in config.custom_sizes:
            bytes_est = tokens_to_bytes_estimate(s)
            size_descriptions.append(f"~{s//1000}K tokens (~{bytes_est//1024}KB)")
        print(f"  Custom sizes: {size_descriptions}")
    else:
        min_bytes = tokens_to_bytes_estimate(config.min_prompt_size)
        max_bytes = tokens_to_bytes_estimate(config.max_prompt_size)
        print(
            f"  Size range: {config.min_prompt_size:,} to {config.max_prompt_size:,} tokens (~{min_bytes//1024}KB to ~{max_bytes//(1024*1024)}MB) (step: {config.step_size:,})"
        )

    print(f"  Parsed args: transport={args.transport}, runs={args.runs}")
    print()

    # Run the specific test via pytest
    test_args = [
        __file__ + "::test_request_plane_performance_comparison",
        "-v",
        "-s",
        "--tb=short",
    ]
    sys.exit(pytest.main(test_args))


if __name__ == "__main__":
    main()
