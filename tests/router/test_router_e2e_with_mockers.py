# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: These tests run reliably in serial but have encountered intermittent failures
# under pytest-xdist parallel execution (-n auto). Each test spawns its own
# DistributedRuntime with isolated etcd/NATS and unique namespaces, but the Rust
# runtime may use process-global state (e.g. lazy_static / OnceLock singletons for
# endpoint tables) that races under concurrent xdist workers. Do not add
# @pytest.mark.parallel until DRT endpoint registration is confirmed thread-safe.
#
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (
    _test_busy_threshold_endpoint,
    _test_disagg_background_prefill_sticky_routing,
    _test_disagg_direct_mode,
    _test_disagg_topology_required_prefill_pin_match_and_mismatch,
    _test_python_router_bindings,
    _test_remote_indexer_decisions,
    _test_router_basic,
    _test_router_decisions,
    _test_router_decisions_disagg,
    _test_router_decisions_disagg_round_robin_prefill_dp_rank,
    _test_router_indexers_sync,
    _test_router_overload_503,
    _test_router_override_router_config,
    _test_router_query_instance_id,
    _test_router_threshold_none_disables_rejection,
    _test_router_two_routers,
)
from tests.router.helper import (
    generate_random_suffix,
    get_runtime,
    poll_for_worker_instances,
    topology_env,
)
from tests.router.mocker_process import (
    DisaggMockerProcess,
    MockerProcess,
    _launch_disagg_workers,
)
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
COUNTER_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "counter_worker.py")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_NAME),
]
NUM_MOCKERS = 2
SPEEDUP_RATIO = 10.0
BASE_PORT = 9100  # Base port for general test allocations (frontend, system, etc.)
NUM_REQUESTS = 100
BLOCK_SIZE = 16
ROUTER_OVERLOAD_DEBUG_DYN_LOG = (
    "info,"
    "dynamo_llm::discovery::worker_monitor=debug,"
    "dynamo_llm::kv_router=debug,"
    "dynamo_runtime::pipeline::network::egress::push_router=debug,"
    "dynamo_llm::mocker=debug"
)
PLANNER_PROFILE_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "components/src/dynamo/planner/tests/data/profiling_results/H200_TP1P_TP1D"
)
ROUTER_AIC_CONFIG = {
    "aic_backend": "vllm",
    "aic_system": "h200_sxm",
    "aic_backend_version": "0.14.0",
    "aic_tp_size": 1,
    "aic_model_path": "Qwen/Qwen3-32B",
}
ROUTER_OVERLOAD_503_CASES = (
    pytest.param(
        {
            "blocks_threshold": 0.2,
            "max_tokens": 50,
        },
        id="decode-blocks",
    ),
    pytest.param(
        {
            "blocks_threshold": "None",
            "tokens_threshold": 1,
            "tokens_threshold_frac": "None",
            "router_queue_threshold": "None",
            "max_tokens": 1,
        },
        id="prefill-tokens",
    ),
)
ROUND_ROBIN_MOCKER_SKIP_REASON = (
    "Flaky on CI: tcp nondurable round-robin mocker router path timed out"
)
COUNTER_TEST_PAYLOAD: Dict[str, Any] = {
    "model": "counter",
    "messages": [{"role": "user", "content": "test"}],
    "stream": True,
    "max_tokens": 1,
}


def _require_router_aic() -> dict[str, Any]:
    pytest.importorskip(
        "aiconfigurator", reason="router AIC test requires aiconfigurator"
    )
    return ROUTER_AIC_CONFIG.copy()


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
SOAK_TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "one two three four five six seven eight nine ten",
        }
    ],
    "stream": False,
    "max_tokens": 1,
}


class CounterWorkerProcess:
    """Manages CPU and GPU counter_worker.py subprocesses for device-aware routing tests.

    Launches one worker with CUDA_VISIBLE_DEVICES="" (CPU) and one with "0" (GPU).
    Both register using RouterConfig(RouterMode.DeviceAwareWeighted) so the frontend's
    global router mode is overridden by the per-worker config.
    """

    def __init__(
        self, request, store_backend: str = "etcd", request_plane: str = "nats"
    ):
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "counter"
        self.endpoint_path = f"{self.namespace}.{self.component_name}.generate"
        self.num_workers = 2
        self._request = request
        self._store_backend = store_backend
        self._request_plane = request_plane
        self._cpu_count_file: Optional[str] = None
        self._gpu_count_file: Optional[str] = None
        self._cpu_proc: Optional[ManagedProcess] = None
        self._gpu_proc: Optional[ManagedProcess] = None

    @property
    def cpu_count_file(self) -> str:
        assert self._cpu_count_file is not None
        return self._cpu_count_file

    @property
    def gpu_count_file(self) -> str:
        assert self._gpu_count_file is not None
        return self._gpu_count_file

    def __enter__(self):
        cpu_fd, self._cpu_count_file = tempfile.mkstemp(suffix=".txt")
        os.close(cpu_fd)
        gpu_fd, self._gpu_count_file = tempfile.mkstemp(suffix=".txt")
        os.close(gpu_fd)

        env = os.environ.copy()
        self._cpu_proc = ManagedProcess(
            command=[
                sys.executable,
                COUNTER_WORKER_SCRIPT,
                self._cpu_count_file,
                "cpu",
                self.endpoint_path,
                "--discovery-backend",
                self._store_backend,
                "--request-plane",
                self._request_plane,
                "--router-mode",
                "device-aware-weighted",
            ],
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=self._request.node.name,
            terminate_all_matching_process_names=False,
            display_name="counter-worker-cpu",
        )
        self._gpu_proc = ManagedProcess(
            command=[
                sys.executable,
                COUNTER_WORKER_SCRIPT,
                self._gpu_count_file,
                "gpu",
                self.endpoint_path,
                "--discovery-backend",
                self._store_backend,
                "--request-plane",
                self._request_plane,
                "--router-mode",
                "device-aware-weighted",
            ],
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=self._request.node.name,
            terminate_all_matching_process_names=False,
            display_name="counter-worker-gpu",
        )
        self._cpu_proc.__enter__()
        self._gpu_proc.__enter__()
        logger.info(
            f"Started CPU and GPU counter workers, endpoint: {self.endpoint_path}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for proc, name in [
            (self._cpu_proc, "CPU"),
            (self._gpu_proc, "GPU"),
        ]:
            if proc is not None:
                try:
                    proc.__exit__(exc_type, exc_val, exc_tb)
                except Exception as e:
                    logger.warning(f"Error stopping {name} counter worker: {e}")
        for path in [self._cpu_count_file, self._gpu_count_file]:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


@pytest.mark.timeout(180)  # planner-profile mocker setup can exceed 120s on CI CPUs
@pytest.mark.parametrize(
    "router_mode,durable_kv_events,mocker_args_override",
    [
        pytest.param("kv", False, {}, id="kv-nondurable"),
        pytest.param(
            "kv",
            False,
            {"planner_profile_data": PLANNER_PROFILE_DATA_DIR},
            id="kv-planner",
        ),
        pytest.param(
            "kv",
            False,
            {"aic_perf_model": True, "aic_system": "h200_sxm"},
            id="kv-aic",
        ),
        pytest.param("kv", True, {}, id="kv-durable"),
        pytest.param("round-robin", False, {}, id="roundrobin"),
        pytest.param("random", False, {}, id="random"),
        pytest.param("power-of-two", False, {}, id="power-of-two"),
    ],
    indirect=["durable_kv_events"],
)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.skip(reason=ROUND_ROBIN_MOCKER_SKIP_REASON)
def test_mocker_router(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    router_mode,
    request_plane,
    durable_kv_events,
    mocker_args_override,
):
    """Test router with multiple mocker engine instances across all router modes.

    Covers kv, round-robin, and random routing. Tests both NATS and TCP request planes.
    """
    # runtime_services starts etcd and optionally nats based on request_plane
    logger.info(
        f"Starting mocker router test: router_mode={router_mode}, request_plane={request_plane}"
    )

    # Create mocker args dictionary - use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }
    mocker_args.update(mocker_args_override)

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
            router_mode=router_mode,
            min_initial_workers=mockers.num_workers,
        )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("router_mode", ["kv", "round-robin", "random"])
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
)
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
def test_mocker_router_soak(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    router_mode,
    durable_kv_events,
    request_plane,
):
    mocker_args = {
        "speedup_ratio": 1000.0,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=2,
        request_plane=request_plane,
    ) as mockers:
        frontend_port = get_unique_ports(
            request, num_ports=1, request_plane=request_plane
        )[0]

        _test_router_basic(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=SOAK_TEST_PAYLOAD,
            num_requests=1024,
            request_plane=request_plane,
            router_mode=router_mode,
            min_initial_workers=mockers.num_workers,
        )


@pytest.mark.parametrize("store_backend", ["etcd", "file"])
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
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


@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
)  # Use NATS Core (local indexer)
@pytest.mark.parametrize("overload_config", ROUTER_OVERLOAD_503_CASES)
@pytest.mark.timeout(45)  # ~3x average (~13.10s), rounded up (when enabled)
def test_mocker_kv_router_overload_503(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    durable_kv_events,
    monkeypatch,
    overload_config,
):
    """Test that KV router returns 503 when mocker workers are overloaded."""
    monkeypatch.setenv("DYN_LOG", ROUTER_OVERLOAD_DEBUG_DYN_LOG)
    logger.info("Starting mocker KV router overload test for 503 status")
    # Create mocker args dictionary with limited resources - use local indexer (NATS Core mode)
    mocker_args = {
        "speedup_ratio": 0.01,
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
            **overload_config,
        )


@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
)  # Use NATS Core (local indexer)
@pytest.mark.timeout(45)
def test_mocker_kv_router_threshold_none_disables_rejection(
    request, runtime_services_dynamic_ports, predownload_tokenizers, durable_kv_events
):
    """Test that explicit CLI None thresholds disable KV router overload rejection."""
    logger.info("Starting mocker KV router explicit-None threshold test")
    mocker_args = {
        "speedup_ratio": 0.01,
        "block_size": 4,
        "num_gpu_blocks": 64,
        "durable_kv_events": durable_kv_events,
    }

    with MockerProcess(request, mocker_args=mocker_args, num_mockers=1) as mockers:
        logger.info("Starting single mocker instance with limited resources")
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")

        frontend_port = get_unique_ports(request, num_ports=1)[0]

        _test_router_threshold_none_disables_rejection(
            engine_workers=mockers,
            block_size=4,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=4,
        )


@pytest.mark.timeout(90)  # bumped for xdist contention (was 22s; ~7.10s serial avg)
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
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
    "store_backend,durable_kv_events,request_plane",
    [
        ("etcd", True, "nats"),  # JetStream mode - uses JetStream
        ("etcd", False, "tcp"),  # NATS core mode (with gap detection) - no JetStream
        ("file", True, "nats"),  # File backend - uses JetStream
    ],
    ids=[
        "jetstream",
        "nats_core",
        "file",
    ],
    indirect=["request_plane", "durable_kv_events"],
)
# Known flake (nats_core, file variants): Router and Standalone indexer occasionally
# disagree on event count by 3-4 events (e.g. "Router 1 has 105 events, Standalone A
# has 102 events"). Race in event-sync convergence — needs root-cause investigation,
# not a retry.
@pytest.mark.timeout(300)
def test_indexers_sync(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
    durable_kv_events,
    request_plane,
):
    """
    Test that two KV routers have synchronized indexer states after processing requests.
    This test verifies that both routers converge to the same internal state.

    Tests with three configurations:
    - jetstream: etcd backend, JetStream for KV events, NATS request plane
    - nats_core: etcd backend, NATS Core with gap detection, TCP request plane
    - file: file backend, JetStream for KV events, NATS request plane
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
        zmq_kv_events=True,
        zmq_replay=True,
        standalone_indexer=True,
        model_name=MODEL_NAME,
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
            test_nats_interruption=not durable_kv_events,
            nats_server=nats_process if not durable_kv_events else None,
            durable_kv_events=durable_kv_events,
            standalone_indexer_url=mockers.standalone_indexer_url,
            standalone_indexer_b_url=mockers.standalone_indexer_b_url,
            test_zmq_replay=True,
        )

        logger.info("Indexers sync test completed successfully")


@pytest.mark.timeout(120)  # bumped for xdist contention (was 42s; ~13.80s serial avg)
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
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


@pytest.mark.timeout(300)  # bumped for xdist contention (was 29s; ~9.55s serial avg)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events,use_kv_events,zmq_kv_events,use_remote_indexer,router_predicted_ttl_secs",
    [
        (True, True, False, False, None),  # JetStream mode with KV events
        (
            False,
            True,
            False,
            False,
            None,
        ),  # NATS Core mode with local indexer (default)
        (False, True, False, False, 5.0),  # NATS Core mode with local side indexer
        (False, True, False, True, None),  # NATS Core mode with a served remote indexer
        (False, True, False, True, 5.0),  # Remote indexer plus local side indexer
        (False, False, False, False, None),  # Approximate mode (--no-kv-events)
        (
            False,
            False,
            False,
            True,
            None,
        ),  # Approximate mode with a singleton served remote indexer
        (False, True, True, False, None),  # ZMQ mode: mocker → ZMQ PUB → relay → NATS
    ],
    ids=[
        "jetstream",
        "nats_core",
        "nats_core_predict_on_route",
        "nats_core_remote",
        "nats_core_remote_predict_on_route",
        "no_kv_events",
        "no_kv_events_remote",
        "zmq",
    ],
    indirect=["durable_kv_events"],
)
def test_router_decisions(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    durable_kv_events,
    use_kv_events,
    request_plane,
    zmq_kv_events,
    use_remote_indexer,
    router_predicted_ttl_secs,
):
    """Validate KV cache prefix reuse and dp_rank routing by sending progressive requests with overlapping prefixes.

    Parameterized to test:
    - JetStream mode: KV events via NATS JetStream (durable)
    - NATS Core mode (default): KV events via NATS Core with local indexer on workers
    - NATS Core mode with a served remote indexer
    - Approximate mode (--no-kv-events): No KV events, router predicts cache state
      based on routing decisions with TTL-based expiration and pruning
    - Approximate mode with a singleton served remote indexer
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        "Starting test router decisions: durable_kv_events=%s, use_kv_events=%s, use_remote_indexer=%s, router_predicted_ttl_secs=%s",
        durable_kv_events,
        use_kv_events,
        use_remote_indexer,
        router_predicted_ttl_secs,
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
        standalone_selector=zmq_kv_events,
        model_name=MODEL_NAME,
    ) as mockers:
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        if use_remote_indexer:
            _test_remote_indexer_decisions(
                mockers,
                MODEL_NAME,
                block_size=8,
                use_kv_events=use_kv_events,
                test_dp_rank=True,
                request_plane=request_plane,
                router_predicted_ttl_secs=router_predicted_ttl_secs,
            )
            return

        runtime = get_runtime(request_plane=request_plane)
        endpoint = runtime.endpoint(f"{mockers.namespace}.mocker.generate")

        _test_router_decisions(
            mockers,
            endpoint,
            MODEL_NAME,
            request,
            test_dp_rank=True,
            use_kv_events=use_kv_events,
            durable_kv_events=durable_kv_events,
            standalone_indexer_url=mockers.standalone_indexer_url,
            standalone_selector_url=mockers.standalone_selector_url,
            router_predicted_ttl_secs=router_predicted_ttl_secs,
        )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_router_decisions_router_aic(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
):
    """Validate agg KV-router decisions with router-side AIC enabled on the NATS Core path."""
    logger.info("Starting agg router decisions test with router-side AIC enabled")

    router_aic_config = _require_router_aic()
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": 8,
        "dp_size": 4,
        "durable_kv_events": False,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=2,
        request_plane=request_plane,
        model_name=MODEL_NAME,
    ) as mockers:
        runtime = get_runtime(request_plane=request_plane)
        endpoint = runtime.endpoint(f"{mockers.namespace}.mocker.generate")

        _test_router_decisions(
            mockers,
            endpoint,
            MODEL_NAME,
            request,
            test_dp_rank=True,
            use_kv_events=True,
            durable_kv_events=False,
            router_aic_config=router_aic_config,
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

    with _launch_disagg_workers(
        request,
        shared_namespace,
        registration_order,
        prefill_mocker_args=mocker_args,
        decode_mocker_args=mocker_args,
        num_prefill_mockers=4,
        num_decode_mockers=4,
        enable_disagg_bootstrap=enable_disagg_bootstrap,
    ) as (prefill_workers, decode_workers):
        frontend_port = get_unique_ports(
            request, num_ports=1, registration_order=registration_order
        )[0]
        _test_router_decisions_disagg(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            request_plane="nats",
            enable_bootstrap=enable_disagg_bootstrap,
        )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("discovery_backend", ["etcd"], indirect=True)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
)
def test_disagg_background_prefill_sticky(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    discovery_backend,
    request_plane,
    durable_kv_events,
):
    """Sticky session affinity pins disagg background prefill on TCP/NATS."""
    _ = (runtime_services_dynamic_ports, predownload_tokenizers, durable_kv_events)

    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"
    prefill_mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "dp_size": 2,
    }
    decode_mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    frontend_port = get_unique_ports(
        request,
        num_ports=1,
        store_backend=discovery_backend,
        request_plane=request_plane,
    )[0]
    with FrontendRouterProcess(
        request,
        BLOCK_SIZE,
        frontend_port,
        shared_namespace,
        discovery_backend,
        enforce_disagg=True,
        request_plane=request_plane,
        event_plane="nats",
        durable_kv_events=False,
    ):
        time.sleep(1.0)
        with _launch_disagg_workers(
            request,
            shared_namespace,
            "prefill_first",
            prefill_mocker_args=prefill_mocker_args,
            decode_mocker_args=decode_mocker_args,
            num_prefill_mockers=3,
            num_decode_mockers=2,
            enable_disagg_bootstrap=True,
            store_backend=discovery_backend,
            request_plane=request_plane,
            event_plane="nats",
        ) as (prefill_workers, decode_workers):
            _test_disagg_background_prefill_sticky_routing(
                prefill_workers=prefill_workers,
                decode_workers=decode_workers,
                block_size=BLOCK_SIZE,
                request=request,
                frontend_port=frontend_port,
                model_name=MODEL_NAME,
                store_backend=discovery_backend,
                request_plane=request_plane,
                event_plane="nats",
                frontend_already_running=True,
            )


@pytest.mark.timeout(180)
def test_disagg_topology_required_prefill_pin_match_and_mismatch(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    tmp_path,
):
    """Validate required KV-transfer topology policy from pinned prefill workers."""
    logger.info("Starting disaggregated topology-aware prefill pin test")
    _ = (runtime_services_dynamic_ports, predownload_tokenizers)

    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    prefill_zone_a_env = topology_env(tmp_path, "prefill-zone-a", {"zone": "zone-a"})
    prefill_zone_b_env = topology_env(tmp_path, "prefill-zone-b", {"zone": "zone-b"})
    decode_zone_a_env = topology_env(tmp_path, "decode-zone-a", {"zone": "zone-a"})

    with DisaggMockerProcess(
        request,
        namespace=shared_namespace,
        worker_type="prefill",
        mocker_args=mocker_args,
        num_mockers=1,
        request_plane="tcp",
        env_overrides=prefill_zone_a_env,
    ):
        runtime = get_runtime()
        prefill_endpoint = runtime.endpoint(f"{shared_namespace}.prefill.generate")
        prefill_zone_a_ids = asyncio.run(poll_for_worker_instances(prefill_endpoint, 1))
        assert len(prefill_zone_a_ids) == 1
        prefill_zone_a_id = prefill_zone_a_ids[0]
        logger.info("Prefill zone-a worker id: %s", prefill_zone_a_id)

        with DisaggMockerProcess(
            request,
            namespace=shared_namespace,
            worker_type="prefill",
            mocker_args=mocker_args,
            num_mockers=1,
            request_plane="tcp",
            env_overrides=prefill_zone_b_env,
        ):
            prefill_ids = asyncio.run(poll_for_worker_instances(prefill_endpoint, 2))
            prefill_zone_b_ids = sorted(set(prefill_ids) - {prefill_zone_a_id})
            assert len(prefill_zone_b_ids) == 1, (
                f"Expected one new zone-b prefill worker, got all={prefill_ids}, "
                f"zone_a={prefill_zone_a_id}"
            )
            prefill_zone_b_id = prefill_zone_b_ids[0]
            logger.info("Prefill zone-b worker id: %s", prefill_zone_b_id)

            with DisaggMockerProcess(
                request,
                namespace=shared_namespace,
                worker_type="decode",
                mocker_args=mocker_args,
                num_mockers=2,
                request_plane="tcp",
                env_overrides=decode_zone_a_env,
            ) as decode_workers:
                decode_endpoint = runtime.endpoint(
                    f"{shared_namespace}.backend.generate"
                )
                decode_ids = sorted(
                    asyncio.run(poll_for_worker_instances(decode_endpoint, 2))
                )
                logger.info("Decode zone-a worker ids: %s", decode_ids)

                frontend_port = get_unique_ports(request, num_ports=1)[0]
                _test_disagg_topology_required_prefill_pin_match_and_mismatch(
                    decode_workers=decode_workers,
                    block_size=BLOCK_SIZE,
                    request=request,
                    frontend_port=frontend_port,
                    test_payload=TEST_PAYLOAD,
                    prefill_zone_a_id=prefill_zone_a_id,
                    prefill_zone_b_id=prefill_zone_b_id,
                    shared_namespace=shared_namespace,
                    request_plane="tcp",
                )


@pytest.mark.parametrize("registration_order", ["prefill_first", "decode_first"])
@pytest.mark.parametrize(
    "enable_disagg_bootstrap", [False, True], ids=["no_bootstrap", "with_bootstrap"]
)
@pytest.mark.timeout(180)
def test_router_decisions_disagg_round_robin_prefill_dp_rank(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    registration_order,
    enable_disagg_bootstrap,
):
    """Verify round-robin disagg prefill requests spread KV stores across DP ranks."""
    logger.info(
        "Starting disaggregated round-robin prefill dp-rank test "
        "(registration_order=%s, bootstrap=%s)",
        registration_order,
        enable_disagg_bootstrap,
    )

    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"
    prefill_mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "dp_size": 4,
    }
    decode_mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    def run_case(prefill_workers, decode_workers):
        frontend_port = get_unique_ports(
            request, num_ports=1, registration_order=registration_order
        )[0]
        _test_router_decisions_disagg_round_robin_prefill_dp_rank(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            expected_prefill_dp_ranks=prefill_mocker_args["dp_size"],
            request_plane="nats",
        )

    with _launch_disagg_workers(
        request,
        shared_namespace,
        registration_order,
        prefill_mocker_args=prefill_mocker_args,
        decode_mocker_args=decode_mocker_args,
        num_prefill_mockers=1,
        num_decode_mockers=1,
        enable_disagg_bootstrap=enable_disagg_bootstrap,
    ) as (prefill_workers, decode_workers):
        run_case(prefill_workers, decode_workers)


@pytest.mark.timeout(180)
def test_router_decisions_disagg_router_aic(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
):
    """Validate disagg KV-router decisions with router-side AIC enabled on the default startup path."""
    logger.info("Starting disaggregated router prefix reuse test with router-side AIC")

    router_aic_config = _require_router_aic()
    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with DisaggMockerProcess(
        request,
        namespace=shared_namespace,
        worker_type="prefill",
        mocker_args=mocker_args,
        num_mockers=4,
        request_plane="nats",
        enable_bootstrap=False,
    ) as prefill_workers:
        logger.info(f"Prefill workers using endpoint: {prefill_workers.endpoint}")

        with DisaggMockerProcess(
            request,
            namespace=shared_namespace,
            worker_type="decode",
            mocker_args=mocker_args,
            num_mockers=4,
            request_plane="nats",
        ) as decode_workers:
            logger.info(f"Decode workers using endpoint: {decode_workers.endpoint}")

            frontend_port = get_unique_ports(
                request, num_ports=1, registration_order="prefill_first"
            )[0]

            _test_router_decisions_disagg(
                prefill_workers=prefill_workers,
                decode_workers=decode_workers,
                block_size=BLOCK_SIZE,
                request=request,
                frontend_port=frontend_port,
                test_payload=TEST_PAYLOAD,
                request_plane="nats",
                router_aic_config=router_aic_config,
            )


@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.parametrize(
    "durable_kv_events", [False], ids=["nondurable"], indirect=True
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


@pytest.mark.timeout(180)
def test_disagg_direct_mode_epp_headers(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
):
    """E2E: disaggregated serving with Direct routing mode (simulating GAIE EPP).

    This test verifies the EPP-driven routing path used in the GAIE deploy recipe:
      - Frontend runs with --router-mode direct (no autonomous worker selection)
      - Worker IDs are supplied via x-worker-instance-id / x-prefill-instance-id headers

    Validates:
      1. Requests with explicit headers succeed and report correct worker IDs
      2. Requests without headers are rejected (Direct mode enforces header routing)
    """
    logger.info("Starting disaggregated Direct-mode EPP headers E2E test")

    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with DisaggMockerProcess(
        request,
        namespace=shared_namespace,
        worker_type="prefill",
        mocker_args=mocker_args,
        num_mockers=2,
        request_plane="nats",
    ) as prefill_workers:
        logger.info(f"Prefill workers using endpoint: {prefill_workers.endpoint}")

        with DisaggMockerProcess(
            request,
            namespace=shared_namespace,
            worker_type="decode",
            mocker_args=mocker_args,
            num_mockers=2,
            request_plane="nats",
        ) as decode_workers:
            logger.info(f"Decode workers using endpoint: {decode_workers.endpoint}")

            frontend_port = get_unique_ports(request, num_ports=1)[0]

            _test_disagg_direct_mode(
                prefill_workers=prefill_workers,
                decode_workers=decode_workers,
                request=request,
                frontend_port=frontend_port,
                test_payload=TEST_PAYLOAD,
                request_plane="nats",
            )


def test_router_per_worker_config(
    request,
    runtime_services_dynamic_ports,
    file_storage_backend,
):
    """Test that per-worker RouterConfig(DeviceAwareWeighted) overrides the frontend's
    global round-robin mode. GPU worker receives all requests; CPU worker receives none.

    Workers register with CUDA_VISIBLE_DEVICES="" (CPU) and "0" (GPU) and declare
    RouterConfig(RouterMode.DeviceAwareWeighted) in their MDC. The frontend starts with
    --router-mode round-robin. With the default cuda-to-cpu ratio of 8, all requests go
    to the GPU worker because allowed_cpu_inflight = gpu_inflight / 8 = 0.
    """
    logger.info("Starting per-worker router config override test")

    with CounterWorkerProcess(request) as workers:
        frontend_port = get_unique_ports(request, num_ports=1)[0]
        _test_router_override_router_config(
            endpoint=workers.endpoint_path,
            engine_workers=workers,
            request=request,
            frontend_port=frontend_port,
            test_payload=COUNTER_TEST_PAYLOAD,
            num_requests=5,
            cpu_count_file=workers.cpu_count_file,
            gpu_count_file=workers.gpu_count_file,
        )
