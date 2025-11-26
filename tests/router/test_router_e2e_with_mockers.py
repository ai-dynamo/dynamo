# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Router E2E tests with mock backend engines.

These tests validate KV router functionality using mocker engines (no GPU required).
All tests are marked as @pytest.mark.pre_merge and @pytest.mark.parallel.

Parallel Execution:
-------------------
Install pytest-xdist for parallel test execution:
    pip install pytest-xdist

All tests in this file are designed for parallel execution with isolated ports.

Usage Examples:
    # Run all tests serially
    pytest tests/router/test_router_e2e_with_mockers.py -v

    # Run all tests in parallel with 4 workers
    pytest tests/router/test_router_e2e_with_mockers.py -v -n 4

    # Run all tests in parallel with auto workers
    pytest tests/router/test_router_e2e_with_mockers.py -v -n auto

    # Run specific test
    pytest tests/router/test_router_e2e_with_mockers.py::test_mocker_kv_router -v

    # Run only pre-merge tests (all tests in this file)
    pytest tests/router/test_router_e2e_with_mockers.py -v -m pre_merge

    # Run parametrized test with specific backend
    pytest tests/router/test_router_e2e_with_mockers.py::test_mocker_two_kv_router[etcd] -v
    pytest tests/router/test_router_e2e_with_mockers.py::test_mocker_two_kv_router[file] -v

Port Allocation:
----------------
Tests use dynamic port allocation with file-based locking to prevent race conditions:
    - Starting port: 8100 (for frontend services)
    - Ports are dynamically allocated using allocate_free_port() and allocate_free_ports()
    - File-based locking ensures no conflicts between parallel test workers
    - Allocated ports are tracked in a registry file with automatic cleanup of stale allocations

This ensures no port conflicts when running tests in parallel.

Test Requirements:
------------------
- No GPU required (uses mocker engines with speedup_ratio)
- Requires NATS and Etcd (provided by runtime_services fixture)
- Tokenizers must be downloaded (predownload_tokenizers fixture)
- Model: Uses ROUTER_MODEL_NAME from tests.utils.constants

Performance:
------------
Parallel execution significantly reduces test time (24 tests, 32-core machine):
    Serial:              ~360s (6 minutes)
    Parallel (-n auto):  ~64s (5.6x faster, 24 workers)
"""

import logging
import os
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (  # utilities
    _test_busy_threshold_endpoint,
    _test_python_router_bindings,
    _test_router_basic,
    _test_router_decisions,
    _test_router_disagg_decisions,
    _test_router_indexers_sync,
    _test_router_overload_503,
    _test_router_query_instance_id,
    _test_router_two_routers,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_free_port, allocate_free_ports

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.model(MODEL_NAME),
]
NUM_MOCKERS = 2
SPEEDUP_RATIO = 10.0
NUM_REQUESTS = 100
BLOCK_SIZE = 16


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


def _build_mocker_command(
    endpoint: str,
    store_backend: str,
    num_workers: int,
    mocker_args: Dict[str, Any],
    worker_type: Optional[str] = None,
) -> list[str]:
    """Build the mocker CLI command with all arguments.

    Args:
        endpoint: The dynamo endpoint string
        store_backend: Storage backend ("etcd" or "file")
        num_workers: Number of workers to spawn (uses --num-workers flag)
        mocker_args: Dictionary of mocker arguments
        worker_type: Optional worker type ("prefill" or "decode") for disagg mode

    Returns:
        List of command arguments for subprocess
    """
    command = [
        "python",
        "-m",
        "dynamo.mocker",
        "--model-path",
        MODEL_NAME,
        "--endpoint",
        endpoint,
        "--store-kv",
        store_backend,
        "--num-workers",
        str(num_workers),
    ]

    # Add worker type flag for disaggregated mode
    if worker_type == "prefill":
        command.append("--is-prefill-worker")
    elif worker_type == "decode":
        command.append("--is-decode-worker")

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
            ["--max-num-batched-tokens", str(mocker_args["max_num_batched_tokens"])]
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

    return command


class MockerProcess:
    """Manages mocker engine instances with shared tokio runtime via --num-workers."""

    def __init__(
        self,
        request,
        nats_port: int,
        etcd_port: int,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
    ):
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "mocker"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_mockers

        mocker_args = mocker_args or {}

        command = _build_mocker_command(
            endpoint=self.endpoint,
            store_backend=store_backend,
            num_workers=num_mockers,
            mocker_args=mocker_args,
        )

        # Set environment variables for NATS and Etcd connections
        env = os.environ.copy()
        env["NATS_SERVER"] = f"nats://localhost:{nats_port}"
        env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_port}"

        self._process = ManagedProcess(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        logger.info(
            f"Created mocker process with {num_mockers} worker(s), endpoint: {self.endpoint}"
        )

    def __enter__(self):
        logger.info(f"Starting mocker process with {self.num_workers} worker(s)")
        self._process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping mocker process")
        self._process.__exit__(exc_type, exc_val, exc_tb)


class DisaggMockerProcess:
    """Manages prefill or decode mocker instances for disaggregated serving.

    Uses --num-workers for shared tokio runtime. For disaggregated serving:
    - Prefill workers: worker_type="prefill", endpoint is namespace.prefill.generate
    - Decode workers: worker_type="decode", endpoint is namespace.backend.generate

    Both prefill and decode workers should share the same namespace for proper discovery.
    """

    def __init__(
        self,
        request,
        nats_port: int,
        etcd_port: int,
        namespace: str,
        worker_type: str,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
    ):
        if worker_type not in ("prefill", "decode"):
            raise ValueError(
                f"worker_type must be 'prefill' or 'decode', got {worker_type}"
            )

        self.namespace = namespace
        self.worker_type = worker_type
        self.num_workers = num_mockers

        # Set component name and endpoint based on worker type
        if worker_type == "prefill":
            self.component_name = "prefill"
            self.endpoint = f"dyn://{self.namespace}.prefill.generate"
        else:
            self.component_name = "backend"
            self.endpoint = f"dyn://{self.namespace}.backend.generate"

        mocker_args = mocker_args or {}

        command = _build_mocker_command(
            endpoint=self.endpoint,
            store_backend=store_backend,
            num_workers=num_mockers,
            mocker_args=mocker_args,
            worker_type=worker_type,
        )

        # Set environment variables for NATS and Etcd connections
        env = os.environ.copy()
        env["NATS_SERVER"] = f"nats://localhost:{nats_port}"
        env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_port}"

        self._process = ManagedProcess(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_existing=False,
        )
        logger.info(
            f"Created {worker_type} mocker process with {num_mockers} worker(s), "
            f"endpoint: {self.endpoint}"
        )

    def __enter__(self):
        logger.info(
            f"Starting {self.worker_type} mocker process with {self.num_workers} worker(s)"
        )
        self._process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Stopping {self.worker_type} mocker process")
        self._process.__exit__(exc_type, exc_val, exc_tb)


@pytest.mark.parallel
def test_mocker_kv_router(request, runtime_services, predownload_tokenizers):
    """
    Test KV router with multiple mocker engine instances.
    This test doesn't require GPUs and runs quickly for pre-merge validation.
    """
    nats_process, etcd_process = runtime_services

    # runtime_services starts etcd and nats
    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info("Starting mocker KV router test")

    # Create mocker args dictionary: FixtureRequestry: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = allocate_free_port(8100)

        # Run basic router test (starts router internally and waits for workers to be ready)
        _test_router_basic(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.parallel
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
    nats_process, etcd_process = runtime_services

    # runtime_services starts etcd and nats
    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info(
        f"Starting mocker two KV router test with {store_backend} storage backend"
    )

    # Create mocker args dictionary
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            store_backend=store_backend,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique ports for this test (2 ports for two routers)
        router_ports = allocate_free_ports(2, 8100)

        # Run two-router test (starts KV routers internally and manages their lifecycle)
        _test_router_two_routers(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            router_ports=router_ports,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            store_backend=store_backend,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.parallel
@pytest.mark.skip(reason="Flaky, temporarily disabled")
def test_mocker_kv_router_overload_503(
    request, runtime_services, predownload_tokenizers
):
    """Test that KV router returns 503 when mocker workers are overloaded."""
    nats_process, etcd_process = runtime_services

    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

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
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=1,
        )
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = allocate_free_port(8100)

        # Run overload 503 test
        _test_router_overload_503(
            engine_workers=mockers,
            block_size=4,  # Match the mocker's block size
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            busy_threshold=0.2,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


def test_kv_push_router_bindings(request, runtime_services, predownload_tokenizers):
    """Test KvPushRouter Python bindings with mocker engines."""
    nats_process, etcd_process = runtime_services

    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info("Starting KvPushRouter bindings test")
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # pytest-xdist safe: workers are separate processes with isolated os.environ.
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

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

    Uses runtime_services (function-scoped) to provide isolated NATS/Etcd instances
    with dynamically allocated ports for parallel test safety. The two KvPushRouter
    objects created in this test share the same NATS/Etcd instances for snapshot
    synchronization within the test scope.
    """
    nats_process, etcd_process = runtime_services

    # runtime_services starts etcd and nats
    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info(f"Starting indexers sync test with {store_backend} storage backend")

    # Create mocker args dictionary
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            store_backend=store_backend,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # pytest-xdist safe: workers are separate processes with isolated os.environ.
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

        # Use the common test implementation (creates its own runtimes for each router)
        # Note: Consumer verification is done inside _test_router_indexers_sync while routers are alive
        _test_router_indexers_sync(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=NUM_MOCKERS,
            store_backend=store_backend,
            nats_port=nats_process.port,
        )

        logger.info("Indexers sync test completed successfully")

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.parallel
def test_query_instance_id_returns_worker_and_tokens(
    request, runtime_services, predownload_tokenizers
):
    """Test query_instance_id annotation with mocker engines."""
    nats_process, etcd_process = runtime_services

    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info("Starting KV router query_instance_id annotation test")
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    os.makedirs(request.node.name, exist_ok=True)

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = allocate_free_port(8100)

        # Run query_instance_id annotation test
        _test_router_query_instance_id(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.parallel
def test_router_decisions(request, runtime_services, predownload_tokenizers):
    """Validate KV cache prefix reuse and dp_rank routing by sending progressive requests with overlapping prefixes."""
    nats_process, etcd_process = runtime_services

    # runtime_services starts etcd and nats
    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

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
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=2,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Initialize mockers
        mockers.__enter__()

        # pytest-xdist safe: workers are separate processes with isolated os.environ.
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

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


@pytest.mark.parallel
def test_router_disagg_decisions(request, runtime_services, predownload_tokenizers):
    """Validate KV cache prefix reuse in disaggregated prefill-decode setup.

    Tests that progressive requests with overlapping prefixes are routed to the
    same prefill worker due to KV cache reuse.
    """
    nats_process, etcd_process = runtime_services

    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info("Starting disaggregated router prefix reuse test")

    # Generate shared namespace for prefill and decode workers
    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"

    # Create mocker args
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    prefill_workers = None
    decode_workers = None

    try:
        # Start prefill workers (4 instances)
        logger.info("Starting 4 prefill mocker instances")
        prefill_workers = DisaggMockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            namespace=shared_namespace,
            worker_type="prefill",
            mocker_args=mocker_args,
            num_mockers=4,
        )
        prefill_workers.__enter__()
        logger.info(f"Prefill workers using endpoint: {prefill_workers.endpoint}")

        # Start decode workers (4 instances)
        logger.info("Starting 4 decode mocker instances")
        decode_workers = DisaggMockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            namespace=shared_namespace,
            worker_type="decode",
            mocker_args=mocker_args,
            num_mockers=4,
        )
        decode_workers.__enter__()
        logger.info(f"Decode workers using endpoint: {decode_workers.endpoint}")

        # Get unique port for this test
        frontend_port = allocate_free_port(8100)

        # Run disagg routing test
        _test_router_disagg_decisions(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if decode_workers is not None:
            decode_workers.__exit__(None, None, None)
        if prefill_workers is not None:
            prefill_workers.__exit__(None, None, None)


@pytest.mark.parallel
def test_busy_threshold_endpoint(request, runtime_services, predownload_tokenizers):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 503

    For now, this test only verifies the endpoint is accessible and returns valid responses.
    """
    nats_process, etcd_process = runtime_services

    logger.info(
        f"Using runtime services - NATS port: {nats_process.port}, Etcd port: {etcd_process.port}"
    )

    # Set environment variables for runtime services before starting mocker workers
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

    logger.info("Starting busy_threshold endpoint test")

    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            nats_process.port,
            etcd_process.port,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = allocate_free_port(8100)

        _test_busy_threshold_endpoint(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            nats_port=nats_process.port,
            etcd_port=etcd_process.port,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)
