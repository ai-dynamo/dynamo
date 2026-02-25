# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic tests (xdist-safe via dynamic ports + per-test namespaces).
# Tested on: Linux container.
# Combined pre_merge wall time (this file):
# - Serialized: 304.01s.
# - Parallel (-n auto): 34.55s (269.46s saved, 8.80x).
#
# NOTE: TCP request plane is NOT tested here. These tests use --num-workers > 1 which spawns
# multiple workers in a single process sharing one TCP server. The shared TCP server uses
# endpoint_path (e.g., "generate") as the routing key, causing handler collisions when multiple
# workers register the same endpoint. This is a test-only limitation; production deployments
# with separate processes per worker work correctly with TCP.
import asyncio
import logging
import os
from typing import Any, Dict, Optional

import aiohttp
import pytest

from tests.router.common import (  # utilities
    _test_busy_threshold_endpoint,
    _test_python_router_bindings,
    _test_router_basic,
    _test_router_decisions,
    _test_router_decisions_disagg,
    _test_router_indexers_sync,
    _test_router_overload_503,
    _test_router_query_instance_id,
    _test_router_two_routers,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

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
BASE_PORT = 9100  # Base port for general test allocations (frontend, system, etc.)
BASE_PORT_BOOTSTRAP = 10100  # Base port for disagg bootstrap rendezvous
BASE_PORT_ZMQ = 11100  # Base port for ZMQ KV event publishing
NUM_REQUESTS = 100
BLOCK_SIZE = 16


def get_unique_ports(
    request,
    num_ports: int = 1,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    registration_order: str = "prefill_first",
) -> list[int]:
    """Allocate random free ports for xdist-safe router tests.

    This replaces the previous "test-name offset" scheme with the shared flock-backed
    allocator from `tests.utils.port_utils`, which avoids collisions across pytest-xdist
    worker processes.

    Notes:
    - The extra parameters are kept for call-site compatibility (they no longer affect
      the chosen ports).
    - Ports are released at the end of the test via a pytest finalizer.
    """
    _ = (store_backend, request_plane, registration_order)
    ports = allocate_ports(num_ports, BASE_PORT)
    request.addfinalizer(lambda: deallocate_ports(ports))
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
        "--discovery-backend",
        store_backend,
        "--num-workers",
        str(num_workers),
    ]

    # Add worker type flag for disaggregated mode
    if worker_type == "prefill":
        command.extend(["--disaggregation-mode", "prefill"])
    elif worker_type == "decode":
        command.extend(["--disaggregation-mode", "decode"])

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
    # Use --durable-kv-events to enable JetStream mode (local indexer disabled)
    if mocker_args.get("durable_kv_events") is True:
        command.append("--durable-kv-events")
    if "bootstrap_ports" in mocker_args:
        command.extend(["--bootstrap-ports", mocker_args["bootstrap_ports"]])
    if "zmq_kv_events_ports" in mocker_args:
        command.extend(["--zmq-kv-events-ports", mocker_args["zmq_kv_events_ports"]])

    return command


class MockerProcess:
    """Manages mocker engine instances with shared tokio runtime via --num-workers.

    When standalone_indexer=True, launches mockers one-by-one (each as --num-workers 1)
    and runs a standalone HTTP KV indexer binary alongside them. Call launch_mockers_with_indexer()
    in async context to start mockers and register their ZMQ ports with the indexer.
    """

    def __init__(
        self,
        request,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        zmq_kv_events: bool = False,
        standalone_indexer: bool = False,
    ):
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "mocker"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_mockers
        self._zmq_kv_events_ports: list[int] = []
        self._standalone_indexer = standalone_indexer
        self._standalone_indexer_port: Optional[int] = None
        self._indexer_process: Optional[ManagedProcess] = None
        self._mocker_processes: list[ManagedProcess] = []
        self._request = request
        self._store_backend = store_backend
        self._request_plane = request_plane
        self._mocker_args_orig: Dict[str, Any] = (mocker_args or {}).copy()
        self.worker_id_to_zmq_ports: dict[int, dict[int, str]] = {}

        mocker_args = self._mocker_args_orig.copy()
        # Store dp_size for DP-aware test functions
        self.dp_size = mocker_args.get("dp_size")
        # Alias for consistency with vLLM/SGLang workers
        self.data_parallel_size = self.dp_size

        # Allocate ZMQ base ports for KV event publishing.
        # Each worker's DP ranks bind on base_port + dp_rank, so we need bases
        # spaced dp_size apart. Allocate num_mockers * dp_size ports total,
        # then pick every dp_size'th port as a base.
        if zmq_kv_events:
            dp_size = mocker_args.get("dp_size", 1)
            self._zmq_kv_events_ports = allocate_ports(
                num_mockers * dp_size, BASE_PORT_ZMQ
            )
            bases = [self._zmq_kv_events_ports[i * dp_size] for i in range(num_mockers)]
            if not standalone_indexer:
                mocker_args["zmq_kv_events_ports"] = ",".join(str(p) for p in bases)
            logger.info(
                f"Allocated ZMQ KV event ports {self._zmq_kv_events_ports} "
                f"(bases: {bases}) for {num_mockers} workers"
            )

        if standalone_indexer:
            # Allocate a port for the standalone indexer HTTP server
            self._standalone_indexer_port = allocate_ports(1, BASE_PORT)[0]
            request.addfinalizer(
                lambda: deallocate_ports([self._standalone_indexer_port])
                if self._standalone_indexer_port
                else None
            )
            # Don't build a single mocker command — we'll launch per-mocker in launch_mockers_with_indexer
            self._process = None
        else:
            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=store_backend,
                num_workers=num_mockers,
                mocker_args=mocker_args,
            )

            env = os.environ.copy()
            env["DYN_REQUEST_PLANE"] = request_plane

            self._process = ManagedProcess(
                command=command,
                env=env,
                timeout=60,
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_all_matching_process_names=False,
            )
        logger.info(
            f"Created mocker process with {num_mockers} worker(s), endpoint: {self.endpoint}"
            f"{', standalone_indexer=True' if standalone_indexer else ''}"
        )

    @property
    def standalone_indexer_url(self) -> Optional[str]:
        if self._standalone_indexer_port is not None:
            return f"http://localhost:{self._standalone_indexer_port}"
        return None

    def __enter__(self):
        if self._standalone_indexer:
            # Launch the standalone indexer binary
            block_size = self._mocker_args_orig.get("block_size", BLOCK_SIZE)
            indexer_cmd = [
                "cargo",
                "run",
                "-p",
                "dynamo-kv-router",
                "--features",
                "indexer-bin",
                "--bin",
                "dynamo-kv-indexer",
                "--",
                "--block-size",
                str(block_size),
                "--port",
                str(self._standalone_indexer_port),
            ]
            self._indexer_process = ManagedProcess(
                command=indexer_cmd,
                timeout=120,
                display_output=True,
                health_check_ports=[self._standalone_indexer_port],
                health_check_urls=[],
                log_dir=self._request.node.name,
                terminate_all_matching_process_names=False,
                display_name="dynamo-kv-indexer",
            )
            logger.info(
                f"Starting standalone indexer on port {self._standalone_indexer_port}"
            )
            self._indexer_process.__enter__()
            # Don't start mocker processes yet — launch_mockers_with_indexer will do it
        else:
            logger.info(f"Starting mocker process with {self.num_workers} worker(s)")
            self._process.__enter__()
        return self

    async def launch_mockers_with_indexer(self, endpoint):
        """Launch mockers one-by-one and register each with the standalone indexer.

        For each mocker:
        1. Launch a mocker process with --num-workers 1
        2. Poll endpoint.client().instance_ids() until a new worker_id appears
        3. POST /workers to the indexer with the worker_id and its ZMQ addresses

        Args:
            endpoint: The dynamo endpoint object to discover worker IDs.
        """
        client = await endpoint.client()
        known_ids: set[int] = set()
        dp_size = self._mocker_args_orig.get("dp_size", 1)

        for i in range(self.num_workers):
            # Build per-mocker args with its own ZMQ base port
            mocker_args = self._mocker_args_orig.copy()
            base_port = self._zmq_kv_events_ports[i * dp_size]
            mocker_args["zmq_kv_events_ports"] = str(base_port)

            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=self._store_backend,
                num_workers=1,
                mocker_args=mocker_args,
            )

            env = os.environ.copy()
            env["DYN_REQUEST_PLANE"] = self._request_plane

            proc = ManagedProcess(
                command=command,
                env=env,
                timeout=60,
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=self._request.node.name,
                terminate_all_matching_process_names=False,
                display_name=f"mocker-{i}",
            )
            proc.__enter__()
            self._mocker_processes.append(proc)

            # Poll for the new worker_id
            new_worker_id = None
            for _ in range(120):
                ids = set(client.instance_ids())
                new = ids - known_ids
                if new:
                    new_worker_id = new.pop()
                    known_ids.add(new_worker_id)
                    break
                await asyncio.sleep(0.5)

            if new_worker_id is None:
                raise RuntimeError(
                    f"Timed out waiting for mocker {i} to register "
                    f"(known_ids={known_ids})"
                )

            # Build ZMQ address map for this worker
            zmq_addresses = {}
            for dp_rank in range(dp_size):
                port = self._zmq_kv_events_ports[i * dp_size + dp_rank]
                zmq_addresses[str(dp_rank)] = f"tcp://127.0.0.1:{port}"

            self.worker_id_to_zmq_ports[new_worker_id] = zmq_addresses

            # Register with the standalone indexer
            register_url = f"{self.standalone_indexer_url}/workers"
            payload = {
                "worker_id": new_worker_id,
                "zmq_addresses": zmq_addresses,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(register_url, json=payload) as resp:
                    if resp.status != 201:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Failed to register worker {new_worker_id}: "
                            f"{resp.status} {body}"
                        )

            logger.info(
                f"Mocker {i}: worker_id={new_worker_id}, "
                f"zmq_addresses={zmq_addresses}"
            )

        logger.info(
            f"All {self.num_workers} mockers launched and registered with indexer"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping mocker process(es)")
        # Stop individual mocker processes (standalone_indexer mode)
        for proc in self._mocker_processes:
            try:
                proc.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error stopping mocker process: {e}")
        self._mocker_processes.clear()
        # Stop standalone indexer
        if self._indexer_process is not None:
            try:
                self._indexer_process.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error stopping indexer process: {e}")
            self._indexer_process = None
        # Stop single mocker process (non-standalone mode)
        if self._process is not None:
            self._process.__exit__(exc_type, exc_val, exc_tb)
        if self._zmq_kv_events_ports:
            deallocate_ports(self._zmq_kv_events_ports)
            logger.info(f"Deallocated ZMQ KV event ports {self._zmq_kv_events_ports}")
            self._zmq_kv_events_ports = []


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
        namespace: str,
        worker_type: str,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        enable_bootstrap: bool = False,
    ):
        if worker_type not in ("prefill", "decode"):
            raise ValueError(
                f"worker_type must be 'prefill' or 'decode', got {worker_type}"
            )

        self.namespace = namespace
        self.worker_type = worker_type
        self.num_workers = num_mockers
        self._bootstrap_ports: list[int] = []

        # Set component name and endpoint based on worker type
        if worker_type == "prefill":
            self.component_name = "prefill"
            self.endpoint = f"dyn://{self.namespace}.prefill.generate"
        else:
            self.component_name = "backend"
            self.endpoint = f"dyn://{self.namespace}.backend.generate"

        mocker_args = (mocker_args or {}).copy()

        # Allocate bootstrap ports for prefill workers if enabled (one per worker)
        if enable_bootstrap and worker_type == "prefill":
            self._bootstrap_ports = allocate_ports(num_mockers, BASE_PORT_BOOTSTRAP)
            mocker_args["bootstrap_ports"] = ",".join(
                str(p) for p in self._bootstrap_ports
            )
            logger.info(
                f"Allocated bootstrap ports {self._bootstrap_ports} for {num_mockers} prefill workers"
            )

        command = _build_mocker_command(
            endpoint=self.endpoint,
            store_backend=store_backend,
            num_workers=num_mockers,
            mocker_args=mocker_args,
            worker_type=worker_type,
        )

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane

        self._process = ManagedProcess(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
        )
        logger.info(
            f"Created {worker_type} mocker process with {num_mockers} worker(s), "
            f"endpoint: {self.endpoint}"
        )

    @property
    def bootstrap_ports(self) -> list[int]:
        """Return the allocated bootstrap ports, if any."""
        return self._bootstrap_ports

    def __enter__(self):
        logger.info(
            f"Starting {self.worker_type} mocker process with {self.num_workers} worker(s)"
        )
        self._process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Stopping {self.worker_type} mocker process")
        self._process.__exit__(exc_type, exc_val, exc_tb)
        # Deallocate bootstrap ports if we allocated any
        if self._bootstrap_ports:
            deallocate_ports(self._bootstrap_ports)
            logger.info(f"Deallocated bootstrap ports {self._bootstrap_ports}")
            self._bootstrap_ports = []


@pytest.mark.timeout(120)  # bumped for xdist contention (was 42s; ~13.80s serial avg)
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
def test_mocker_kv_router(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
    durable_kv_events,
):
    """
    Test KV router with multiple mocker engine instances.
    This test doesn't require GPUs and runs quickly for pre-merge validation.
    Tests both NATS and TCP request planes.
    """

    # runtime_services starts etcd and optionally nats based on request_plane
    logger.info(f"Starting mocker KV router test with request_plane={request_plane}")

    # Create mocker args dictionary - use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        request_plane=request_plane,
    ) as mockers:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Get unique port for this test
        frontend_port = get_unique_ports(
            request, num_ports=1, request_plane=request_plane
        )[0]

        # Run basic router test (starts router internally and waits for workers to be ready)
        _test_router_basic(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            request_plane=request_plane,
        )


@pytest.mark.parametrize("store_backend", ["etcd", "file"])
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
@pytest.mark.timeout(180)  # bumped for xdist contention (was 60s; ~19.86s serial avg)
def test_mocker_two_kv_router(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
    durable_kv_events,
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

    # Create mocker args dictionary - use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        store_backend=store_backend,
    ) as mockers:
        # Start mocker instances with the new CLI interface
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

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
            skip_consumer_verification=not durable_kv_events,  # Skip JetStream checks in NATS Core mode
        )


@pytest.mark.skip(reason="Flaky, temporarily disabled")
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
@pytest.mark.timeout(60)  # ~3x average (~19.86s), rounded up (when enabled)
def test_mocker_kv_router_overload_503(
    request, runtime_services_dynamic_ports, predownload_tokenizers, durable_kv_events
):
    """Test that KV router returns 503 when mocker workers are overloaded."""
    logger.info("Starting mocker KV router overload test for 503 status")
    # Create mocker args dictionary with limited resources - use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": 10,
        "block_size": 4,  # Smaller block size
        "num_gpu_blocks": 64,  # Limited GPU blocks to exhaust quickly
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(request, mocker_args=mocker_args, num_mockers=1) as mockers:
        # Start single mocker instance with limited resources
        logger.info("Starting single mocker instance with limited resources")
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")

        # Get unique port for this test
        frontend_port = get_unique_ports(request, num_ports=1)[0]

        # Run overload 503 test
        _test_router_overload_503(
            engine_workers=mockers,
            block_size=4,  # Match the mocker's block size
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            blocks_threshold=0.2,
        )


@pytest.mark.timeout(90)  # bumped for xdist contention (was 22s; ~7.10s serial avg)
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
def test_kv_router_bindings(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
    durable_kv_events,
):
    """Test KvRouter Python bindings with mocker engines."""
    logger.info("Starting KvRouter bindings test")
    # Use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        request_plane=request_plane,
    ) as mockers:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Get runtime and create endpoint
        runtime = get_runtime(request_plane=request_plane)
        endpoint = runtime.endpoint(
            f"{mockers.namespace}.{mockers.component_name}.generate"
        )

        # Run Python router bindings test
        _test_python_router_bindings(
            engine_workers=mockers,
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=NUM_MOCKERS,
        )


@pytest.mark.parametrize(
    "store_backend,durable_kv_events,request_plane,router_event_threads,zmq_kv_events",
    [
        ("etcd", True, "nats", 1, False),  # JetStream mode - uses JetStream
        ("etcd", False, "tcp", 1, True),  # NATS core mode + standalone indexer
        ("file", True, "nats", 1, False),  # File backend - uses JetStream
        (
            "etcd",
            False,
            "tcp",
            2,
            True,
        ),  # NATS core mode - multi-threaded + standalone indexer
    ],
    ids=[
        "jetstream",
        "nats_core",
        "file",
        "nats_core_multi_thread",
    ],
    indirect=["request_plane", "durable_kv_events"],
)
@pytest.mark.timeout(180)
def test_indexers_sync(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
    durable_kv_events,
    request_plane,
    router_event_threads,
    zmq_kv_events,
):
    """
    Test that two KV routers have synchronized indexer states after processing requests.
    This test verifies that both routers converge to the same internal state.

    Tests with four configurations:
    - jetstream: etcd backend, JetStream for KV events, NATS request plane
    - nats_core: etcd backend, ZMQ + standalone HTTP indexer, TCP request plane
                 (includes NATS interruption/recovery testing)
    - file: file backend, JetStream for KV events, NATS request plane
    - nats_core_multi_thread: etcd backend, ZMQ + standalone HTTP indexer, TCP request plane
    """
    logger.info(
        f"Starting indexers sync test: store_backend={store_backend}, "
        f"durable_kv_events={durable_kv_events}, request_plane={request_plane}"
    )

    # Use the dynamic-port fixture to avoid hardcoded localhost:4222/2379 in parallel runs.
    nats_process, _etcd_process = runtime_services_dynamic_ports

    # Create mocker args dictionary
    # Use 2 DP ranks to test per-dp_rank event ID tracking and recovery
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
        "dp_size": 2,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        store_backend=store_backend,
        request_plane=request_plane,
        zmq_kv_events=zmq_kv_events,
        standalone_indexer=zmq_kv_events,
    ) as mockers:
        # Start mocker instances (2 workers x 2 DP ranks = 4 independent event streams)
        logger.info(f"Starting {NUM_MOCKERS} mocker instances with dp_size=2")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Use the common test implementation (creates its own runtimes for each router)
        # Note: Consumer verification is done inside _test_router_indexers_sync while routers are alive
        # When using durable_kv_events=True, use JetStream mode for the router
        _test_router_indexers_sync(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=NUM_MOCKERS,
            store_backend=store_backend,
            request_plane=request_plane,
            test_nats_interruption=not durable_kv_events and not zmq_kv_events,
            nats_server=nats_process
            if (not durable_kv_events and not zmq_kv_events)
            else None,
            durable_kv_events=durable_kv_events,
            router_event_threads=router_event_threads,
            standalone_indexer_url=mockers.standalone_indexer_url,
        )

        logger.info("Indexers sync test completed successfully")


@pytest.mark.timeout(120)  # bumped for xdist contention (was 42s; ~13.80s serial avg)
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
def test_query_instance_id_returns_worker_and_tokens(
    request, runtime_services_dynamic_ports, predownload_tokenizers, durable_kv_events
):
    """Test query_instance_id annotation with mocker engines."""
    logger.info("Starting KV router query_instance_id annotation test")
    # Use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
    ) as mockers:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

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


@pytest.mark.timeout(90)  # bumped for xdist contention (was 29s; ~9.55s serial avg)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events,use_kv_events,router_event_threads,zmq_kv_events",
    [
        (True, True, 1, False),  # JetStream mode with KV events
        (False, True, 1, False),  # NATS Core mode with local indexer (default)
        (False, False, 1, False),  # Approximate mode (--no-kv-events) - no KV events
        (False, True, 2, False),  # NATS Core mode - multi-threaded indexer
        (False, True, 1, True),  # ZMQ mode: mocker → ZMQ PUB → relay → NATS
    ],
    ids=["jetstream", "nats_core", "no_kv_events", "nats_core_multi_thread", "zmq"],
    indirect=["durable_kv_events"],
)
def test_router_decisions(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    durable_kv_events,
    use_kv_events,
    request_plane,
    router_event_threads,
    zmq_kv_events,
):
    """Validate KV cache prefix reuse and dp_rank routing by sending progressive requests with overlapping prefixes.

    Parameterized to test:
    - JetStream mode: KV events via NATS JetStream (durable)
    - NATS Core mode (default): KV events via NATS Core with local indexer on workers
    - Approximate mode (--no-kv-events): No KV events, router predicts cache state
      based on routing decisions with TTL-based expiration and pruning
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        f"Starting test router decisions: durable_kv_events={durable_kv_events}, use_kv_events={use_kv_events}"
    )

    # Create mocker args dictionary with dp_size=4
    # durable_kv_events=True enables JetStream mode; False (default) uses NATS Core with local indexer
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": 8,
        "dp_size": 4,
        "durable_kv_events": durable_kv_events and use_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=2,
        request_plane=request_plane,
        zmq_kv_events=zmq_kv_events,
        standalone_indexer=zmq_kv_events,
    ) as mockers:
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Initialize mockers
        # Get runtime and create endpoint
        runtime = get_runtime(request_plane=request_plane)
        # Use the namespace from the mockers
        endpoint = runtime.endpoint(f"{mockers.namespace}.mocker.generate")

        _test_router_decisions(
            mockers,
            endpoint,
            MODEL_NAME,
            request,
            test_dp_rank=True,
            use_kv_events=use_kv_events,
            durable_kv_events=durable_kv_events,
            router_event_threads=router_event_threads,
            standalone_indexer_url=mockers.standalone_indexer_url,
        )


@pytest.mark.parametrize("registration_order", ["prefill_first", "decode_first"])
@pytest.mark.parametrize(
    "enable_disagg_bootstrap", [False, True], ids=["no_bootstrap", "with_bootstrap"]
)
@pytest.mark.timeout(180)  # bumped for xdist contention (was 59s; ~19.51s serial avg)
def test_router_decisions_disagg(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    registration_order,
    enable_disagg_bootstrap,
):
    """Validate KV cache prefix reuse in disaggregated prefill-decode setup.

    Tests that progressive requests with overlapping prefixes are routed to the
    same prefill worker due to KV cache reuse.

    Parameterized to test:
    - registration_order: prefill_first vs decode_first
    - enable_disagg_bootstrap: without vs with bootstrap rendezvous
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        f"Starting disaggregated router prefix reuse test "
        f"(registration_order={registration_order}, bootstrap={enable_disagg_bootstrap})"
    )

    # Generate shared namespace for prefill and decode workers
    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"

    # Create mocker args - use NATS Core with local indexer (default mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        # durable_kv_events defaults to False (NATS Core mode)
    }

    if registration_order == "prefill_first":
        # Start prefill workers first
        logger.info("Starting 4 prefill mocker instances (first)")
        with DisaggMockerProcess(
            request,
            namespace=shared_namespace,
            worker_type="prefill",
            mocker_args=mocker_args,
            num_mockers=4,
            request_plane="nats",
            enable_bootstrap=enable_disagg_bootstrap,
        ) as prefill_workers:
            logger.info(f"Prefill workers using endpoint: {prefill_workers.endpoint}")

            # Then start decode workers
            logger.info("Starting 4 decode mocker instances (second)")
            with DisaggMockerProcess(
                request,
                namespace=shared_namespace,
                worker_type="decode",
                mocker_args=mocker_args,
                num_mockers=4,
                request_plane="nats",
            ) as decode_workers:
                logger.info(f"Decode workers using endpoint: {decode_workers.endpoint}")

                # Get unique port for this test
                frontend_port = get_unique_ports(
                    request, num_ports=1, registration_order=registration_order
                )[0]

                # Run disagg routing test
                _test_router_decisions_disagg(
                    prefill_workers=prefill_workers,
                    decode_workers=decode_workers,
                    block_size=BLOCK_SIZE,
                    request=request,
                    frontend_port=frontend_port,
                    test_payload=TEST_PAYLOAD,
                    request_plane="nats",
                )
    else:
        # Start decode workers first
        logger.info("Starting 4 decode mocker instances (first)")
        with DisaggMockerProcess(
            request,
            namespace=shared_namespace,
            worker_type="decode",
            mocker_args=mocker_args,
            num_mockers=4,
            request_plane="nats",
        ) as decode_workers:
            logger.info(f"Decode workers using endpoint: {decode_workers.endpoint}")

            # Then start prefill workers
            logger.info("Starting 4 prefill mocker instances (second)")
            with DisaggMockerProcess(
                request,
                namespace=shared_namespace,
                worker_type="prefill",
                mocker_args=mocker_args,
                num_mockers=4,
                request_plane="nats",
                enable_bootstrap=enable_disagg_bootstrap,
            ) as prefill_workers:
                logger.info(
                    f"Prefill workers using endpoint: {prefill_workers.endpoint}"
                )

                # Get unique port for this test
                frontend_port = get_unique_ports(
                    request, num_ports=1, registration_order=registration_order
                )[0]

                # Run disagg routing test
                _test_router_decisions_disagg(
                    prefill_workers=prefill_workers,
                    decode_workers=decode_workers,
                    block_size=BLOCK_SIZE,
                    request=request,
                    frontend_port=frontend_port,
                    test_payload=TEST_PAYLOAD,
                    request_plane="nats",
                )


@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], indirect=True
)  # Use NATS Core (local indexer)
@pytest.mark.timeout(120)  # bumped for xdist contention (was 39s; ~12.84s serial avg)
def test_busy_threshold_endpoint(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
    durable_kv_events,
):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 503

    For now, this test only verifies the endpoint is accessible and returns valid responses.
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        f"Starting busy_threshold endpoint test with request_plane={request_plane}"
    )

    # Use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        request_plane=request_plane,
    ) as mockers:
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        frontend_port = get_unique_ports(
            request, num_ports=1, request_plane=request_plane
        )[0]

        _test_busy_threshold_endpoint(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            request_plane=request_plane,
        )
