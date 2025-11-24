# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (  # utilities
    _test_python_router_bindings,
    _test_router_basic,
    _test_router_decisions,
    _test_router_indexers_sync,
    _test_router_overload_503,
    _test_router_query_instance_id,
    _test_router_two_routers,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = pytest.mark.pre_merge


logger = logging.getLogger(__name__)


MODEL_NAME = ROUTER_MODEL_NAME
NUM_MOCKERS = 2
SPEEDUP_RATIO = 10.0
BASE_PORT = 9100  # Base port for all tests (high port to avoid conflicts)
NUM_REQUESTS = 100
BLOCK_SIZE = 16


def get_unique_ports(
    request, num_ports: int = 1, store_backend: str = "etcd"
) -> list[int]:
    """Generate unique ports for parallel test execution.

    Ports are unique based on:
    - Test function name (each test gets a base offset)
    - Parametrization value (etcd=0, file=50)
    - Port index (for multi-port tests)

    Args:
        request: Pytest request fixture
        num_ports: Number of ports needed (1 for single router, 2 for two routers)
        store_backend: Storage backend parameter ("etcd" or "file")

    Returns:
        List of unique port numbers
    """
    # Get test name without parametrization suffix
    test_name = request.node.name.split("[")[0]

    # Base offsets per test function (ensures each test gets unique range)
    test_offsets = {
        "test_mocker_kv_router": 0,
        "test_mocker_two_kv_router": 100,
        "test_mocker_kv_router_overload_503": 200,
        "test_query_instance_id_returns_worker_and_tokens": 300,
    }

    base_offset = test_offsets.get(test_name, 0)

    # Parametrization offset (etcd=0, file=50)
    param_offset = 0 if store_backend == "etcd" else 50

    # Generate ports
    ports = [BASE_PORT + base_offset + param_offset + i for i in range(num_ports)]
    return ports


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


@pytest.mark.pre_merge
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
def test_mocker_kv_router(request, runtime_services_session, predownload_tokenizers):
    """
    Test KV router with multiple mocker engine instances.
    This test doesn't require GPUs and runs quickly for pre-merge validation.
    """

    # runtime_services starts etcd and nats
    logger.info("Starting mocker KV router test")

    # Create mocker args dictiona: FixtureRequestry: tuple[NatsServer, EtcdServer]: NoneType
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = get_unique_ports(request, num_ports=1)[0]

        # Run basic router test (starts router internally and waits for workers to be ready)
        _test_router_basic(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
@pytest.mark.parametrize("store_backend", ["etcd", "file"])
def test_mocker_two_kv_router(
    request,
    runtime_services_session,
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

    # Create mocker args dictionary
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
        mockers.__enter__()

        # Get unique ports for this test (2 ports for two routers)
        router_ports = get_unique_ports(
            request, num_ports=2, store_backend=store_backend
        )

        # Run two-router test (starts KV routers internally and manages their lifecycle)
        _test_router_two_routers(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            router_ports=router_ports,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            store_backend=store_backend,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
@pytest.mark.skip(reason="Flaky, temporarily disabled")
def test_mocker_kv_router_overload_503(
    request, runtime_services_session, predownload_tokenizers
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

        # Get unique port for this test
        frontend_port = get_unique_ports(request, num_ports=1)[0]

        # Run overload 503 test
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
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
def test_kv_push_router_bindings(
    request, runtime_services_session, predownload_tokenizers
):
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
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
@pytest.mark.parametrize("store_backend", ["etcd", "file"])
def test_indexers_sync(
    request,
    runtime_services_session,
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

    # Create mocker args dictionary
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            store_backend=store_backend,
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Use the common test implementation (creates its own runtimes for each router)
        # Note: Consumer verification is done inside _test_router_indexers_sync while routers are alive
        _test_router_indexers_sync(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=NUM_MOCKERS,
            store_backend=store_backend,
        )

        logger.info("Indexers sync test completed successfully")

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
def test_query_instance_id_returns_worker_and_tokens(
    request, runtime_services_session, predownload_tokenizers
):
    """Test query_instance_id annotation with mocker engines."""
    logger.info("Starting KV router query_instance_id annotation test")
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    os.makedirs(request.node.name, exist_ok=True)

    try:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        mockers = MockerProcess(
            request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
        )
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")
        mockers.__enter__()

        # Get unique port for this test
        frontend_port = get_unique_ports(request, num_ports=1)[0]

        # Run query_instance_id annotation test
        _test_router_query_instance_id(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
        )

    finally:
        if "mockers" in locals():
            mockers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.parallel
@pytest.mark.model(MODEL_NAME)
def test_router_decisions(request, runtime_services_session, predownload_tokenizers):
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
