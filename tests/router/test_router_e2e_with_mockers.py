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
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import pytest

from tests.router.common import (
    _test_busy_threshold_endpoint,
    _test_disagg_direct_mode,
    _test_disagg_router_overload_529,
    _test_disagg_topology_required_prefill_pin_match_and_mismatch,
    _test_python_router_bindings,
    _test_remote_indexer_decisions,
    _test_router_decisions_disagg_round_robin_prefill_dp_rank,
    _test_router_overload_529,
    _test_router_override_router_config,
    _test_router_query_instance_id,
    _test_router_threshold_none_disables_rejection,
    _test_router_two_routers,
    _test_session_affinity,
)
from tests.router.e2e_harness import (
    allocate_frontend_ports,
    build_test_payload,
    run_basic_router_test,
    run_disagg_router_decisions_test,
    run_indexers_sync_test,
    run_router_decisions_test,
)
from tests.router.helper import (
    generate_random_suffix,
    get_runtime,
    managed_runtime,
    poll_for_worker_instances,
    topology_env,
)
from tests.router.mocker_process import (
    DisaggMockerProcess,
    MockerProcess,
    launch_disagg_workers,
    wait_for_disagg_workers,
)
from tests.router.router_process import KVRouterProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

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
ROUTER_OVERLOAD_529_CASES = (
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
# Speed isolation: only the *gated* stage is slow (speedup_ratio 0.01); the
# non-gated stage is orders of magnitude faster (100.0) so its latency never
# determines probe cleanup and each case exercises only the intended overload
# signal.
_SLOW_SPEEDUP = 0.01
_FAST_SPEEDUP = 100.0
ROUTER_DISAGG_OVERLOAD_529_CASES = (
    pytest.param(
        {
            # A single prefill worker is sufficient to verify overloaded -> no
            # free prefill worker -> 529. Registered worker types make the model
            # list only after the prefill router activates, so frontend readiness
            # already gates on prefill registration.
            "num_prefill": 1,
            "num_decode": 1,
            "max_tokens": 1,
            # Gate the PREFILL pool only: slow prefill (accumulates tokens), fast
            # decode. Decode/queue thresholds disabled.
            "prefill_speedup": _SLOW_SPEEDUP,
            "decode_speedup": _FAST_SPEEDUP,
            "thresholds": {
                "blocks_threshold": "None",
                "tokens_threshold": 1,
                "tokens_threshold_frac": "None",
                "router_queue_threshold": "None",
            },
        },
        id="prefill-tokens",
    ),
    pytest.param(
        {
            "num_prefill": 1,
            "num_decode": 1,
            "max_tokens": 50,
            # Gate the DECODE pool only: fast prefill, slow decode (fills its
            # limited blocks). Prefill threshold disabled.
            "prefill_speedup": _FAST_SPEEDUP,
            "decode_speedup": _SLOW_SPEEDUP,
            "thresholds": {
                "blocks_threshold": 0.2,
                "tokens_threshold": "None",
                "tokens_threshold_frac": "None",
            },
        },
        id="decode-blocks",
    ),
)
DISAGG_STARTUP_CASES = (
    pytest.param(
        ("frontend", "decode", "prefill"),
        False,
        id="frontend-decode-prefill",
    ),
    pytest.param(
        ("frontend", "prefill", "decode"),
        False,
        id="frontend-prefill-decode",
    ),
    pytest.param(
        ("decode", "frontend", "prefill"),
        False,
        id="decode-frontend-prefill",
    ),
    pytest.param(
        ("prefill", "frontend", "decode"),
        False,
        id="prefill-frontend-decode",
    ),
    pytest.param(
        ("prefill", "decode", "frontend"),
        False,
        id="workers-before-frontend",
    ),
    pytest.param(
        ("frontend", "decode", "prefill"),
        True,
        id="frontend-decode-prefill-bootstrap",
    ),
)
ROUND_ROBIN_MOCKER_SKIP_REASON = (
    "Flaky on CI: TCP round-robin mocker router path timed out"
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
    # Rust AIC callback imports aiconfigurator.sdk.engine.compile_engine
    # (Phase 1.5 API from ai-dynamo/aiconfigurator#1200). PyPI releases
    # predating it don't ship engine.py.
    pytest.importorskip(
        "aiconfigurator.sdk.engine",
        reason="router AIC test requires aiconfigurator.sdk.engine (Phase 1.5)",
    )
    return ROUTER_AIC_CONFIG.copy()


TEST_PAYLOAD = build_test_payload(MODEL_NAME)
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
    "router_mode,mocker_args_override",
    [
        pytest.param("kv", {}, id="kv"),
        pytest.param(
            "kv",
            {"planner_profile_data": PLANNER_PROFILE_DATA_DIR},
            id="kv-planner",
        ),
        pytest.param(
            "kv",
            {"aic_perf_model": True, "aic_system": "h200_sxm"},
            id="kv-aic",
        ),
        pytest.param("round-robin", {}, id="roundrobin"),
        pytest.param("random", {}, id="random"),
        pytest.param("power-of-two", {}, id="power-of-two"),
    ],
)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.skip(reason=ROUND_ROBIN_MOCKER_SKIP_REASON)
def test_mocker_router(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    router_mode,
    request_plane,
    mocker_args_override,
):
    """Test router with multiple mocker engine instances across all router modes.

    Covers kv, round-robin, and random routing. Tests both NATS and TCP request planes.
    """
    # runtime_services starts etcd and optionally nats based on request_plane
    logger.info(
        f"Starting mocker router test: router_mode={router_mode}, request_plane={request_plane}"
    )

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }
    mocker_args.update(mocker_args_override)

    run_basic_router_test(
        engine_process_cls=MockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        num_workers=NUM_MOCKERS,
        single_gpu=False,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
        engine_process_kwargs={"num_mockers": NUM_MOCKERS},
        test_payload=TEST_PAYLOAD,
        num_requests=NUM_REQUESTS,
        router_mode=router_mode,
        min_initial_workers=NUM_MOCKERS,
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("router_mode", ["kv", "round-robin", "random"])
@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
def test_mocker_router_soak(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    router_mode,
    request_plane,
):
    mocker_args = {
        "speedup_ratio": 1000.0,
        "block_size": BLOCK_SIZE,
    }

    run_basic_router_test(
        engine_process_cls=MockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        num_workers=NUM_MOCKERS,
        single_gpu=False,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
        engine_process_kwargs={"num_mockers": NUM_MOCKERS},
        test_payload=SOAK_TEST_PAYLOAD,
        num_requests=1024,
        router_mode=router_mode,
        min_initial_workers=NUM_MOCKERS,
    )


@pytest.mark.parametrize("store_backend", ["etcd", "file"])
@pytest.mark.timeout(180)  # bumped for xdist contention (was 60s; ~19.86s serial avg)
def test_mocker_two_kv_router(
    request,
    runtime_services_dynamic_ports,
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

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
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
        router_ports = allocate_frontend_ports(request, 2)

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


@pytest.mark.parametrize("store_backend", ["etcd", "file"])
@pytest.mark.timeout(180)
def test_mocker_session_affinity(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    file_storage_backend,
    store_backend,
):
    """Replica affinity overrides conflicting per-frontend KV-prefix placement."""
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        store_backend=store_backend,
    ) as mockers:
        _test_session_affinity(
            engine_workers=mockers,
            block_size=BLOCK_SIZE,
            request=request,
            router_ports=allocate_frontend_ports(request, 2),
            test_payload=TEST_PAYLOAD,
            store_backend=store_backend,
        )


@pytest.mark.parametrize("overload_config", ROUTER_OVERLOAD_529_CASES)
@pytest.mark.timeout(45)  # ~3x average (~13.10s), rounded up (when enabled)
def test_mocker_kv_router_overload_529(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch,
    overload_config,
):
    """Test that KV router returns 529 when mocker workers are overloaded."""
    monkeypatch.setenv("DYN_LOG", ROUTER_OVERLOAD_DEBUG_DYN_LOG)
    logger.info("Starting mocker KV router overload test for 529 status")
    mocker_args = {
        "speedup_ratio": 0.01,
        "block_size": 4,  # Smaller block size
        "num_gpu_blocks": 64,  # Limited GPU blocks to exhaust quickly
    }

    with MockerProcess(request, mocker_args=mocker_args, num_mockers=1) as mockers:
        # Start single mocker instance with limited resources
        logger.info("Starting single mocker instance with limited resources")
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")

        # Get unique port for this test
        frontend_port = allocate_frontend_ports(request, 1)[0]

        # Run overload 529 test
        _test_router_overload_529(
            engine_workers=mockers,
            block_size=4,  # Match the mocker's block size
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            **overload_config,
        )


@pytest.mark.timeout(45)
def test_mocker_kv_router_threshold_none_disables_rejection(
    request, runtime_services_dynamic_ports, predownload_tokenizers
):
    """Test that explicit CLI None thresholds disable KV router overload rejection."""
    logger.info("Starting mocker KV router explicit-None threshold test")
    mocker_args = {
        "speedup_ratio": 0.01,
        "block_size": 4,
        "num_gpu_blocks": 64,
    }

    with MockerProcess(request, mocker_args=mocker_args, num_mockers=1) as mockers:
        logger.info("Starting single mocker instance with limited resources")
        logger.info(f"Mocker using endpoint: {mockers.endpoint}")

        frontend_port = allocate_frontend_ports(request, 1)[0]

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
def test_kv_router_bindings(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
):
    """Test KvRouter Python bindings with mocker engines."""
    logger.info("Starting KvRouter bindings test")
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with (
        MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            request_plane=request_plane,
        ) as mockers,
        managed_runtime(request_plane=request_plane) as runtime,
    ):
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Get runtime and create endpoint
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
    "store_backend,request_plane",
    [
        ("etcd", "tcp"),
        ("file", "nats"),
    ],
    ids=[
        "etcd",
        "file",
    ],
    indirect=["request_plane"],
)
@pytest.mark.parametrize("event_plane", ["nats"], indirect=True)
# Known flake: Router and Standalone indexer occasionally
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
    request_plane,
    event_plane,
):
    """
    Test that two KV routers have synchronized indexer states after processing requests.
    This test verifies that both routers converge to the same internal state.

    Tests with etcd and file discovery backends.
    """
    logger.info(
        f"Starting indexers sync test: store_backend={store_backend}, "
        f"request_plane={request_plane}"
    )

    # Create mocker args dictionary
    # Use 2 DP ranks to test per-dp_rank event ID tracking and recovery
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "dp_size": 2,
    }

    run_indexers_sync_test(
        engine_process_cls=MockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        request=request,
        runtime_services_dynamic_ports=runtime_services_dynamic_ports,
        store_backend=store_backend,
        request_plane=request_plane,
        event_plane=event_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
        num_workers=NUM_MOCKERS,
        engine_process_kwargs={
            "num_mockers": NUM_MOCKERS,
            "store_backend": store_backend,
            "raw_kv_events": True,
            "zmq_replay": True,
            "standalone_indexer": True,
            "model_name": MODEL_NAME,
        },
    )


@pytest.mark.timeout(120)  # bumped for xdist contention (was 42s; ~13.80s serial avg)
def test_query_instance_id_returns_worker_and_tokens(
    request, runtime_services_dynamic_ports, predownload_tokenizers
):
    """Test query_instance_id annotation with mocker engines."""
    logger.info("Starting KV router query_instance_id annotation test")
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with MockerProcess(
        request, mocker_args=mocker_args, num_mockers=NUM_MOCKERS
    ) as mockers:
        # Start mocker instances
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        # Get unique port for this test
        frontend_port = allocate_frontend_ports(request, 1)[0]

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
    "use_kv_events,raw_kv_events,use_remote_indexer,router_predicted_ttl_secs,event_plane",
    [
        (True, False, False, None, None),  # Event plane with local indexer
        (True, False, False, 5.0, None),  # Event plane with local side indexer
        (True, False, True, None, None),  # Event plane with remote indexer
        (True, False, True, 5.0, None),  # Remote plus local side indexer
        (False, False, False, None, None),  # Approximate (--no-kv-events)
        (
            False,
            False,
            True,
            None,
            None,
        ),  # Approximate mode with a singleton served remote indexer
        # Raw engine ZMQ → relay → ZMQ event plane, with no NATS service.
        (True, True, False, None, "zmq"),
    ],
    ids=[
        "local_indexer",
        "local_indexer_predict_on_route",
        "remote_indexer",
        "remote_indexer_predict_on_route",
        "no_kv_events",
        "no_kv_events_remote",
        "zmq_nats_free",
    ],
    indirect=["event_plane"],
)
def test_router_decisions(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    use_kv_events,
    request_plane,
    raw_kv_events,
    use_remote_indexer,
    router_predicted_ttl_secs,
    event_plane,
):
    """Validate KV cache prefix reuse and dp_rank routing by sending progressive requests with overlapping prefixes.

    Parameterized to test:
    - Event-plane mode with local indexers on workers
    - Event-plane mode with a served remote indexer
    - Approximate mode (--no-kv-events): No KV events, router predicts cache state
      based on routing decisions with TTL-based expiration and pruning
    - Approximate mode with a singleton served remote indexer
    - NATS-free ZMQ mode: raw engine and Dynamo event-plane hops both use ZMQ
    """
    if event_plane == "zmq":
        nats_process, _ = runtime_services_dynamic_ports
        assert nats_process is None
        assert "NATS_SERVER" not in os.environ

    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        "Starting test router decisions: use_kv_events=%s, use_remote_indexer=%s, router_predicted_ttl_secs=%s, event_plane=%s",
        use_kv_events,
        use_remote_indexer,
        router_predicted_ttl_secs,
        event_plane,
    )

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": 8,
        "dp_size": 4,
    }

    process_kwargs = {
        "num_mockers": NUM_MOCKERS,
        "raw_kv_events": raw_kv_events,
        "standalone_indexer": raw_kv_events,
        "standalone_selector": raw_kv_events,
        "model_name": MODEL_NAME,
    }
    if use_remote_indexer:
        with MockerProcess(
            request,
            mocker_args=mocker_args,
            request_plane=request_plane,
            **process_kwargs,
        ) as mockers:
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

    run_router_decisions_test(
        engine_process_cls=MockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=8,
        component_name="mocker",
        num_workers=NUM_MOCKERS,
        single_gpu=False,
        test_dp_rank=True,
        engine_process_kwargs=process_kwargs,
        test_kwargs={
            "use_kv_events": use_kv_events,
            "router_predicted_ttl_secs": router_predicted_ttl_secs,
        },
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_router_decisions_router_aic(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
):
    """Validate aggregated KV-router decisions with router-side AIC enabled."""
    logger.info("Starting agg router decisions test with router-side AIC enabled")

    router_aic_config = _require_router_aic()
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": 8,
        "dp_size": 4,
    }

    run_router_decisions_test(
        engine_process_cls=MockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=8,
        component_name="mocker",
        num_workers=NUM_MOCKERS,
        single_gpu=False,
        test_dp_rank=True,
        engine_process_kwargs={
            "num_mockers": NUM_MOCKERS,
            "model_name": MODEL_NAME,
        },
        test_kwargs={
            "use_kv_events": True,
            "router_aic_config": router_aic_config,
        },
    )


def _wait_for_frontend_to_observe_single_pd_role(
    frontend: KVRouterProcess,
    worker_type: str,
    timeout: float = 30,
) -> None:
    patterns = {
        "decode": "No prefill endpoint for namespace yet, storing sender for future activation",
        "prefill": "Stored prefill endpoint for future decode WorkerSet registration",
    }
    pattern = patterns[worker_type]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pattern in frontend.read_logs():
            return
        time.sleep(0.1)

    raise AssertionError(
        f"Frontend did not observe the standalone {worker_type} WorkerSet within {timeout}s"
    )


async def _wait_for_disagg_worker_ids(
    frontend_port: int,
    timeout: float = 60,
) -> dict[str, int]:
    payload = {
        **TEST_PAYLOAD,
        "max_tokens": 1,
        "nvext": {"extra_fields": ["worker_id"]},
    }
    deadline = asyncio.get_running_loop().time() + timeout
    last_worker_ids: dict[str, int | None] = {
        "prefill_worker_id": None,
        "decode_worker_id": None,
    }
    client_timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with session.post(
                    f"http://localhost:{frontend_port}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status != 200:
                        await response.read()
                        await asyncio.sleep(0.25)
                        continue

                    async for raw_line in response.content:
                        for line in raw_line.decode(
                            "utf-8", errors="replace"
                        ).splitlines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            worker_ids = chunk.get("nvext", {}).get("worker_id", {})
                            for key in last_worker_ids:
                                if key in worker_ids:
                                    last_worker_ids[key] = worker_ids[key]
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            if all(worker_id is not None for worker_id in last_worker_ids.values()):
                return {
                    key: worker_id
                    for key, worker_id in last_worker_ids.items()
                    if worker_id is not None
                }
            await asyncio.sleep(0.25)

    raise AssertionError(
        f"P/D routing did not become active within {timeout}s; last worker IDs: "
        f"{last_worker_ids}"
    )


@pytest.mark.parametrize(
    ("startup_order", "enable_disagg_bootstrap"), DISAGG_STARTUP_CASES
)
@pytest.mark.timeout(120)
def test_mocker_disagg_startup_lifecycle(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch,
    startup_order,
    enable_disagg_bootstrap,
):
    """A unique P/D topology activates for every meaningful discovery order."""
    monkeypatch.setenv("DYN_LOG", "info,dynamo_llm::discovery=debug")
    namespace = f"test-namespace-{generate_random_suffix()}"
    frontend_port = allocate_frontend_ports(request, 1)[0]
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }
    frontend = None
    workers: dict[str, DisaggMockerProcess] = {}
    expected_worker_ids: dict[str, int] = {}

    with contextlib.ExitStack() as stack:
        for actor in startup_order:
            if actor == "frontend":
                frontend = stack.enter_context(
                    KVRouterProcess(
                        request,
                        BLOCK_SIZE,
                        frontend_port,
                        namespace,
                        "etcd",
                        request_plane="nats",
                        min_initial_workers=1,
                    )
                )
            else:
                worker = stack.enter_context(
                    DisaggMockerProcess(
                        request,
                        namespace=namespace,
                        worker_type=actor,
                        mocker_args=mocker_args,
                        num_mockers=1,
                        enable_bootstrap=(
                            enable_disagg_bootstrap and actor == "prefill"
                        ),
                    )
                )
                workers[actor] = worker
                expected_worker_ids[actor] = wait_for_disagg_workers(
                    worker,
                    store_backend="etcd",
                    request_plane="nats",
                    event_plane=None,
                )[0]

            if frontend is not None and len(workers) == 1:
                _wait_for_frontend_to_observe_single_pd_role(
                    frontend, next(iter(workers))
                )

        assert frontend is not None
        actual_worker_ids = asyncio.run(_wait_for_disagg_worker_ids(frontend_port))

    assert actual_worker_ids["prefill_worker_id"] == expected_worker_ids["prefill"]
    assert actual_worker_ids["decode_worker_id"] == expected_worker_ids["decode"]


@pytest.mark.parametrize(
    "enable_disagg_bootstrap", [False, True], ids=["no_bootstrap", "with_bootstrap"]
)
@pytest.mark.timeout(180)  # bumped for xdist contention (was 59s; ~19.51s serial avg)
def test_router_decisions_disagg(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    enable_disagg_bootstrap,
):
    """Validate KV cache prefix reuse in disaggregated prefill-decode setup.

    Tests that progressive requests with overlapping prefixes are routed to the
    same prefill worker due to KV cache reuse.

    Parameterized with and without bootstrap rendezvous. Startup lifecycle
    ordering is covered separately by ``test_mocker_disagg_startup_lifecycle``.
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        "Starting disaggregated router prefix reuse test "
        f"(bootstrap={enable_disagg_bootstrap})"
    )

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    run_disagg_router_decisions_test(
        engine_process_cls=DisaggMockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        request=request,
        request_plane="nats",
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
        num_prefill_workers=4,
        num_decode_workers=4,
        worker_context_factory=lambda namespace: launch_disagg_workers(
            request,
            namespace,
            "prefill_first",
            prefill_mocker_args=mocker_args,
            decode_mocker_args=mocker_args,
            num_prefill_mockers=4,
            num_decode_mockers=4,
            enable_disagg_bootstrap=enable_disagg_bootstrap,
        ),
        test_payload=TEST_PAYLOAD,
        test_kwargs={"enable_bootstrap": enable_disagg_bootstrap},
    )


@pytest.mark.parametrize("overload_case", ROUTER_DISAGG_OVERLOAD_529_CASES)
@pytest.mark.timeout(120)
def test_mocker_disagg_router_overload_529(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch,
    overload_case,
):
    """Disaggregated load shedding: clients get 529 when the gated pool is busy.

    - prefill-tokens: a low ``--active-prefill-tokens-threshold`` must gate the
      PREFILL pool. This was previously a silent no-op in disagg (the
      overloaded set landed on the decode pool and the prefill router never saw
      it), so this case is the regression guard for that fix.
    - decode-blocks: a low ``--active-decode-blocks-threshold`` must gate the
      DECODE pool (the path that already worked).
    """
    monkeypatch.setenv("DYN_LOG", ROUTER_OVERLOAD_DEBUG_DYN_LOG)
    logger.info("Starting disagg mocker router overload 529 test")

    namespace_suffix = generate_random_suffix()
    shared_namespace = f"test-namespace-{namespace_suffix}"

    # Per-stage args: limited blocks, with only the gated stage slow (speed
    # isolation — see _SLOW_SPEEDUP/_FAST_SPEEDUP).
    def _stage_args(speedup: float) -> Dict[str, Any]:
        return {
            "speedup_ratio": speedup,
            "block_size": 4,
            "num_gpu_blocks": 64,
        }

    with launch_disagg_workers(
        request,
        shared_namespace,
        registration_order="prefill_first",
        prefill_mocker_args=_stage_args(overload_case["prefill_speedup"]),
        decode_mocker_args=_stage_args(overload_case["decode_speedup"]),
        num_prefill_mockers=overload_case["num_prefill"],
        num_decode_mockers=overload_case["num_decode"],
        enable_disagg_bootstrap=False,
    ) as (prefill_workers, decode_workers):
        frontend_port = allocate_frontend_ports(request, 1)[0]
        _test_disagg_router_overload_529(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
            block_size=4,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            max_tokens=overload_case["max_tokens"],
            **overload_case["thresholds"],
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

                frontend_port = allocate_frontend_ports(request, 1)[0]
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


@pytest.mark.parametrize(
    "enable_disagg_bootstrap", [False, True], ids=["no_bootstrap", "with_bootstrap"]
)
@pytest.mark.timeout(180)
def test_router_decisions_disagg_round_robin_prefill_dp_rank(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    enable_disagg_bootstrap,
):
    """Verify round-robin disagg prefill requests spread KV stores across DP ranks."""
    logger.info(
        "Starting disaggregated round-robin prefill dp-rank test (bootstrap=%s)",
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
        frontend_port = allocate_frontend_ports(request, 1)[0]
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

    with launch_disagg_workers(
        request,
        shared_namespace,
        "prefill_first",
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
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    run_disagg_router_decisions_test(
        engine_process_cls=DisaggMockerProcess,
        engine_args_name="mocker_args",
        engine_args=mocker_args,
        request=request,
        request_plane="nats",
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
        num_prefill_workers=4,
        num_decode_workers=4,
        worker_context_factory=lambda namespace: launch_disagg_workers(
            request,
            namespace,
            registration_order="prefill_first",
            prefill_mocker_args=mocker_args,
            decode_mocker_args=mocker_args,
            num_prefill_mockers=4,
            num_decode_mockers=4,
            enable_disagg_bootstrap=False,
        ),
        test_payload=TEST_PAYLOAD,
        test_kwargs={"router_aic_config": router_aic_config},
    )


@pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
@pytest.mark.timeout(120)  # bumped for xdist contention (was 39s; ~12.84s serial avg)
def test_busy_threshold_endpoint(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 529

    For now, this test only verifies the endpoint is accessible and returns valid responses.
    """
    # runtime_services_dynamic_ports handles NATS and etcd startup
    logger.info(
        f"Starting busy_threshold endpoint test with request_plane={request_plane}"
    )

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        request_plane=request_plane,
    ) as mockers:
        logger.info(f"Starting {NUM_MOCKERS} mocker instances")
        logger.info(f"All mockers using endpoint: {mockers.endpoint}")

        frontend_port = allocate_frontend_ports(request, 1)[0]

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
      - Worker IDs are supplied via x-dynamo-worker-instance-id /
        x-dynamo-prefill-instance-id headers

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

    with launch_disagg_workers(
        request,
        shared_namespace,
        registration_order="prefill_first",
        prefill_mocker_args=mocker_args,
        decode_mocker_args=mocker_args,
        num_prefill_mockers=2,
        num_decode_mockers=2,
        enable_disagg_bootstrap=False,
    ) as (prefill_workers, decode_workers):
        frontend_port = allocate_frontend_ports(request, 1)[0]
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
        frontend_port = allocate_frontend_ports(request, 1)[0]
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
