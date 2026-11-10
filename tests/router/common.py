# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import threading
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import aiohttp
import requests

from dynamo.llm import AisPerfConfig, KvRouter, KvRouterConfig
from dynamo.prometheus_names import frontend_service, labels, name_prefix
from tests.router.helper import (
    assert_event_dumps_equal,
    get_runtime,
    managed_runtime,
    poll_for_worker_instances,
    read_routed_worker_from_stream,
    send_inflight_requests,
    send_request_via_python_kv_router,
    unique_request_id,
    verify_response_timing,
    wait_for_frontend_ready,
    wait_for_indexer_workers_active,
    wait_for_workers_ready,
)
from tests.router.router_process import FrontendRouterProcess, KVRouterProcess
from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import reserved_ports
from tests.utils.prometheus import find_metric_samples
from tests.utils.router_logs import (
    parse_kv_event_diagnostics,
    select_kv_event_diagnostics,
    wait_for_kv_event_diagnostics,
)
from tests.utils.router_nvext import parse_sse_json_chunks

if TYPE_CHECKING:
    from tests.conftest import NatsServer

logger = logging.getLogger(__name__)

NUM_REQUESTS = 100
BLOCK_SIZE = 16
MIN_INITIAL_WORKERS_ENV = "DYN_ROUTER_MIN_INITIAL_WORKERS"


@contextlib.contextmanager
def min_initial_workers_env(min_initial_workers: int):
    previous = os.environ.get(MIN_INITIAL_WORKERS_ENV)
    os.environ[MIN_INITIAL_WORKERS_ENV] = str(min_initial_workers)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MIN_INITIAL_WORKERS_ENV, None)
        else:
            os.environ[MIN_INITIAL_WORKERS_ENV] = previous


def _create_kv_router_with_timeout(
    router_factory: Callable[[], KvRouter],
    num_workers: int,
    engine_workers,
    timeout: int = 120,
) -> KvRouter:
    """Create KvRouter in a daemon thread with worker liveness polling.

    KvRouter() is a blocking Rust FFI call that waits for min_initial_workers
    to register.  If a worker crashes (e.g. port conflict, OOM) the call
    blocks forever because pytest signal-based timeout cannot interrupt
    Rust FFI.  This helper runs the call in a daemon thread and polls
    worker liveness every 2 seconds, raising immediately if a worker dies.
    """
    kv_router = None
    kv_router_error = None

    def _create():
        nonlocal kv_router, kv_router_error
        try:
            kv_router = router_factory()
        except Exception as exc:
            kv_router_error = exc

    # NOTE: On timeout or worker death we raise while the daemon thread may
    # still be blocked inside the uninterruptible KvRouter() FFI call.  This is
    # intentional — daemon threads don't block process exit, and the next test
    # reinitializes etcd/NATS state so residual ZMQ handles are harmless.
    with min_initial_workers_env(num_workers):
        router_thread = threading.Thread(target=_create, daemon=True)
        router_thread.start()

        _router_start = time.monotonic()

        while router_thread.is_alive():
            router_thread.join(timeout=2)
            elapsed = time.monotonic() - _router_start
            if elapsed > timeout:
                raise RuntimeError(
                    f"KvRouter initialization timed out after {elapsed:.0f}s "
                    f"waiting for {num_workers} workers to register."
                )
            if hasattr(engine_workers, "worker_processes"):
                for idx, wp in enumerate(engine_workers.worker_processes):
                    if wp.proc and wp.proc.poll() is not None:
                        raise RuntimeError(
                            f"Worker {idx} exited with code {wp.proc.returncode} "
                            f"while waiting for KvRouter to find "
                            f"{num_workers} workers."
                        )

    if kv_router_error is not None:
        raise kv_router_error

    return kv_router


async def _assert_overlap_scores(
    kv_router,
    token_ids: list[int],
    block_size: int,
    expected_device_blocks: dict[tuple[int, int], int],
    label: str,
):
    scores = await kv_router.get_overlap_scores(token_ids, include_shared=False)

    assert scores["block_size"] == block_size
    assert scores["num_blocks"] == len(token_ids) // block_size
    assert scores["shared_cache"]["enabled"] is False

    rows = {(row["worker_id"], row["dp_rank"]): row for row in scores["workers"]}

    for key, expected in expected_device_blocks.items():
        assert key in rows, f"{label}: missing overlap row for {key}"
        row = rows[key]
        assert (
            row["device_blocks"] == expected
        ), f"{label}: expected {key} device_blocks={expected}, got {row}"
        assert row["host_pinned_extension_blocks"] == 0
        assert row["disk_extension_blocks"] == 0
        assert row["shared_beyond_device_blocks"] is None


########################################################
# Test templates
########################################################


def _test_router_basic(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    num_requests: int,
    frontend_timeout: int = 120,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    router_mode: str = "kv",
    min_initial_workers: int | None = None,
):
    """Basic router test: start router, wait for workers and send concurrent requests via HTTP frontend.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This is a shared test implementation for both mocker and vLLM workers.
    Always waits for workers to be properly registered before sending requests to avoid flakiness.

    Supports any router_mode (defaults to "kv" for existing callers).
    block_size is only sent to the frontend CLI when router_mode is "kv".

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port to start the frontend HTTP server on
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        frontend_timeout: Timeout for frontend readiness check (default: 120s)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats", "tcp"). Defaults to "nats".
        router_mode: Router mode ("kv", "round-robin", "random", "direct"). Defaults to "kv".
        min_initial_workers: Optional frontend startup worker gate. Defaults to None.

    Raises:
        AssertionError: If requests fail or frontend doesn't become ready
        TimeoutError: If frontend doesn't become ready within timeout
    """
    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        engine_workers.namespace,
        store_backend,
        request_plane=request_plane,
        router_mode=router_mode,
        min_initial_workers=min_initial_workers,
    ):
        # Start router frontend
        logger.info(
            f"Starting frontend --router-mode {router_mode} on port {frontend_port}"
        )

        frontend_url = f"http://localhost:{frontend_port}"

        # Always wait for workers to register with frontend to avoid flakiness
        logger.info("Waiting for workers to register with frontend...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=frontend_timeout,
                engine_workers=engine_workers,
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        # Send concurrent requests to the frontend
        logger.info(f"Sending {num_requests} concurrent requests to frontend...")
        asyncio.run(
            send_inflight_requests(
                [f"{frontend_url}/v1/chat/completions"],
                test_payload,
                num_requests,
            )
        )

        logger.info(f"Successfully completed {num_requests} requests")


def _test_kv_event_publisher_disabled_diagnostic(
    *,
    frontend,
    engine_workers,
    diagnostic_workers,
    frontend_port: int,
    test_payload: dict,
    model_name: str,
    expected_worker_role: str,
    expected_requirement: str,
    expected_rank_count: int,
    unexpected_worker_roles: tuple[str, ...] = (),
    expected_total_diagnostics: int = 1,
    store_backend: str = "etcd",
    request_plane: str = "tcp",
):
    """Assert an explicit disabled publisher is diagnosed without blocking serving."""

    worker_groups = (
        list(engine_workers)
        if isinstance(engine_workers, (list, tuple))
        else [engine_workers]
    )
    expected_num_workers = sum(group.num_workers for group in worker_groups)

    async def discover_diagnostic_worker_ids() -> set[int]:
        runtime = get_runtime(
            store_backend=store_backend,
            request_plane=request_plane,
        )
        try:
            endpoint = runtime.endpoint(
                f"{diagnostic_workers.namespace}."
                f"{diagnostic_workers.component_name}.generate"
            )
            return set(
                await poll_for_worker_instances(
                    endpoint,
                    diagnostic_workers.num_workers,
                )
            )
        finally:
            runtime.shutdown()

    expected_worker_ids = asyncio.run(discover_diagnostic_worker_ids())
    expected_serving_endpoint = (
        f"{diagnostic_workers.namespace}/{diagnostic_workers.component_name}/generate"
    )
    expected_dp_ranks = ",".join(str(rank) for rank in range(expected_rank_count))

    frontend_url = f"http://localhost:{frontend_port}"
    asyncio.run(
        wait_for_frontend_ready(
            frontend_url=frontend_url,
            expected_num_workers=expected_num_workers,
            timeout=120,
            test_payload=test_payload,
            engine_workers=worker_groups,
            store_backend=store_backend,
            request_plane=request_plane,
        )
    )

    diagnostics = wait_for_kv_event_diagnostics(
        frontend,
        diagnostic_code="kv_event_publisher_disabled",
        expected_count=expected_total_diagnostics,
        worker_role=expected_worker_role,
        timeout_s=10,
    )
    diagnostic = diagnostics[-1]
    assert diagnostic.model == model_name
    assert diagnostic.worker_role == expected_worker_role
    assert diagnostic.requirement == expected_requirement
    assert diagnostic.worker_id in expected_worker_ids
    assert diagnostic.serving_endpoint == expected_serving_endpoint
    assert diagnostic.kv_event_publishing_enabled is False
    assert diagnostic.waited_ms == 0
    assert diagnostic.rank_count == expected_rank_count
    assert diagnostic.dp_ranks == expected_dp_ranks

    async def assert_inference_succeeds() -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{frontend_url}/v1/chat/completions",
                json=test_payload,
            ) as response:
                body = await response.text()
                assert response.status == 200, (
                    "inference must continue when KV event publishing is disabled; "
                    f"status={response.status}, body={body}"
                )

    asyncio.run(assert_inference_succeeds())

    final_diagnostics = parse_kv_event_diagnostics(frontend.read_logs())
    disabled_diagnostics = select_kv_event_diagnostics(
        final_diagnostics,
        diagnostic_code="kv_event_publisher_disabled",
    )
    assert len(disabled_diagnostics) == expected_total_diagnostics, (
        "expected exactly one disabled-publisher diagnostic per worker lifecycle "
        f"after successful inference, got {disabled_diagnostics}"
    )
    for worker_role in unexpected_worker_roles:
        for diagnostic_code in (
            "kv_event_publisher_disabled",
            "kv_event_source_not_observed",
        ):
            unexpected = select_kv_event_diagnostics(
                final_diagnostics,
                diagnostic_code=diagnostic_code,
                worker_role=worker_role,
            )
            assert not unexpected, (
                f"worker role {worker_role!r} does not require KV events, but "
                f"emitted {diagnostic_code!r} diagnostics: {unexpected}"
            )


def _test_router_override_router_config(
    endpoint: str,
    engine_workers,
    request,
    frontend_port: int,
    test_payload: dict,
    num_requests: int,
    cpu_count_file: str,
    gpu_count_file: str,
    block_size: int = BLOCK_SIZE,
    frontend_timeout: int = 120,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Test that router config overrides are honoured by the frontend by starting frontend
    with different router config and expect getting results aligning with the router config
    set by the workers, note that a specific mock worker is used for this test.

    The frontend is started with --router-mode round-robin.

    Workers must already be started with CUDA_VISIBLE_DEVICES="" (CPU) and "0" (GPU)
    and must call register_model(..., router_config=RouterConfig(RouterMode.DeviceAwareWeighted)),
    this configures the router to use device-aware-weighted routing.

    With the default cuda-to-cpu ratio of 8, sequential requests all go to the GPU worker because
    allowed_cpu_inflight = gpu_inflight / 8 = 0, so the CPU worker receives nothing.

    Args:
        endpoint: Dotted endpoint path (e.g. "namespace.component.generate") used to
            poll discovery for worker instance registration before starting the frontend.
        engine_workers: Backend worker instance already initialized with __enter__().
            Must expose .namespace and .num_workers.
        request: Pytest request fixture for managing resources.
        frontend_port: Port to start the HTTP frontend on.
        test_payload: Payload sent to /v1/chat/completions.
        num_requests: Number of sequential requests to send.
        cpu_count_file: Path to the file where the CPU worker writes its request count.
        gpu_count_file: Path to the file where the GPU worker writes its request count.
        block_size: KV-cache block size (ignored for device-aware-weighted, kept for
            interface consistency).
        frontend_timeout: Seconds to wait for the frontend to become ready.
        store_backend: "etcd" or "file".
        request_plane: "nats" or "tcp".

    Raises:
        AssertionError: If routing distribution does not match expectations.
    """

    def _read_count(path: str) -> int:
        try:
            text = open(path).read().strip()
            return int(text) if text else 0
        except (OSError, ValueError):
            return 0

    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        engine_workers.namespace,
        store_backend,
        request_plane=request_plane,
        router_mode="round-robin",
    ):
        frontend_url = f"http://localhost:{frontend_port}"

        # Use endpoint polling to make sure all workers are ready before proceeding,
        # the helper functions will send requests for liveness check which may
        # affect counting if workers are partially ready.
        runtime = get_runtime(store_backend, request_plane)
        endpoint_obj = runtime.endpoint(endpoint)
        asyncio.run(
            poll_for_worker_instances(
                endpoint_obj, engine_workers.num_workers, frontend_timeout
            )
        )

        logger.info("Waiting for workers to register with frontend...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=frontend_timeout,
                engine_workers=engine_workers,
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        logger.info(
            f"Sending {num_requests} requests via device-aware-weighted routing..."
        )
        asyncio.run(
            send_inflight_requests(
                [f"{frontend_url}/v1/chat/completions"],
                test_payload,
                num_requests,
            )
        )

    cpu_count = _read_count(cpu_count_file)
    gpu_count = _read_count(gpu_count_file)

    # There is request sent to indicate liveness, so received request count is
    # larger than the number of requests.
    # This test should actually to confirm that no requests are sent to the CPU worker.
    assert (
        gpu_count >= num_requests
    ), f"GPU worker should receive at least {num_requests} requests, got {gpu_count}"
    assert cpu_count == 0, f"CPU worker should receive 0 requests, got {cpu_count}"
    logger.info(
        f"device-aware-weighted routing verified: GPU={gpu_count}, CPU={cpu_count}"
    )


def _test_router_two_routers(
    engine_workers,
    block_size: int,
    request,
    router_ports: list[int],
    test_payload: dict,
    num_requests: int,
    store_backend: str = "etcd",
):
    """Test two KV routers with alternating requests.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This test:
    1. Starts two KV routers on different ports
    2. Sends requests alternating between the two routers

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        router_ports: List of two port numbers for the routers (e.g., [8091, 8092])
        test_payload: Test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
    """
    kv_routers = []

    try:
        # Start two KV routers on different ports
        for i, port in enumerate(router_ports):
            logger.info(f"Starting KV router frontend on port {port}")
            kv_router = KVRouterProcess(
                request,
                block_size,
                port,
                engine_workers.namespace,
                store_backend,
                min_initial_workers=engine_workers.num_workers,
            )
            kv_router.__enter__()
            kv_routers.append(kv_router)

        # Wait for workers to be ready on both routers
        logger.info("Waiting for workers to register with both routers...")
        for i, port in enumerate(router_ports):
            frontend_url = f"http://localhost:{port}"
            logger.info(f"Waiting for router {i} on port {port} to discover workers...")
            asyncio.run(
                wait_for_frontend_ready(
                    frontend_url=frontend_url,
                    expected_num_workers=engine_workers.num_workers,
                    timeout=120,
                    engine_workers=engine_workers,
                    store_backend=store_backend,
                )
            )
        logger.info("Both routers have discovered workers")

        # Build URLs for both routers
        router_urls = [
            f"http://localhost:{port}/v1/chat/completions" for port in router_ports
        ]

        # Send requests concurrently, alternating between routers
        asyncio.run(
            send_inflight_requests(
                router_urls,
                test_payload,
                num_requests,
            )
        )

        logger.info(
            f"Successfully completed {num_requests} requests across {len(router_ports)} routers"
        )

    finally:
        for kv_router in kv_routers:
            kv_router.__exit__(None, None, None)


def _test_session_affinity(
    engine_workers,
    block_size: int,
    request,
    router_ports: list[int],
    test_payload: dict[str, Any],
    store_backend: str = "etcd",
):
    """Verify replica affinity overrides conflicting per-frontend KV placement."""
    with (
        FrontendRouterProcess(
            request,
            block_size,
            router_ports[0],
            engine_workers.namespace,
            store_backend,
            router_mode="kv",
            min_initial_workers=engine_workers.num_workers,
            event_plane="nats",
            session_affinity_ttl_secs=300,
        ),
        FrontendRouterProcess(
            request,
            block_size,
            router_ports[1],
            engine_workers.namespace,
            store_backend,
            router_mode="kv",
            min_initial_workers=engine_workers.num_workers,
            event_plane="nats",
            session_affinity_ttl_secs=300,
        ),
    ):
        urls = [f"http://localhost:{port}/v1/chat/completions" for port in router_ports]

        async def run_test() -> None:
            runtime = get_runtime(store_backend, "nats")
            endpoint = runtime.endpoint(
                f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
            )
            worker_ids = sorted(
                await poll_for_worker_instances(endpoint, engine_workers.num_workers)
            )
            assert len(worker_ids) >= 2
            worker_a, worker_b = worker_ids[:2]

            for port in router_ports:
                await wait_for_frontend_ready(
                    frontend_url=f"http://localhost:{port}",
                    expected_num_workers=engine_workers.num_workers,
                    timeout=120,
                    engine_workers=engine_workers,
                    store_backend=store_backend,
                    request_plane="nats",
                )

            suffix = uuid.uuid4().hex
            prefix_a = " ".join([f"affinity-alpha-{suffix}"] * (block_size * 2))
            prefix_b = " ".join([f"affinity-beta-{suffix}"] * (block_size * 2))
            session_a_headers = {
                "x-dynamo-session-id": f"local-affinity-a-{uuid.uuid4()}"
            }
            session_b_headers = {
                "x-dynamo-session-id": f"local-affinity-b-{uuid.uuid4()}"
            }

            def payload(content: str, *, query_only: bool = False) -> dict[str, Any]:
                annotations = ["query_instance_id:"] if query_only else []
                return {
                    **test_payload,
                    "messages": [{"role": "user", "content": content}],
                    "stream": True,
                    "max_tokens": 1,
                    "nvext": {
                        "annotations": annotations,
                        "extra_fields": ["worker_id"],
                    },
                }

            async def send(
                client: aiohttp.ClientSession,
                url: str,
                request_payload: dict[str, Any],
                headers: dict[str, str] | None = None,
            ) -> tuple[int, int]:
                async with client.post(
                    url, json=request_payload, headers=headers
                ) as response:
                    body = await response.text()
                    assert response.status == 200, body

                worker_info = None
                for chunk in parse_sse_json_chunks(body):
                    candidate = chunk.get("nvext", {}).get("worker_id")
                    if candidate:
                        worker_info = candidate

                assert worker_info is not None, body
                return (
                    worker_info["decode_worker_id"],
                    worker_info["decode_dp_rank"],
                )

            async def wait_for_prefix_target(
                client: aiohttp.ClientSession,
                url: str,
                content: str,
                expected: tuple[int, int],
            ) -> None:
                for _ in range(50):
                    if (
                        await send(client, url, payload(content, query_only=True))
                        == expected
                    ):
                        return
                    await asyncio.sleep(0.1)
                raise AssertionError(
                    f"KV events did not make prefix target {expected} visible"
                )

            proposal_a = {
                **session_a_headers,
                "x-dynamo-worker-instance-id": str(worker_a),
                "x-dynamo-dp-rank": "0",
            }
            proposal_b = {
                **session_b_headers,
                "x-dynamo-worker-instance-id": str(worker_b),
                "x-dynamo-dp-rank": "0",
            }

            async with aiohttp.ClientSession() as client:
                assert await send(client, urls[0], payload(prefix_a), proposal_a) == (
                    worker_a,
                    0,
                )
                assert await send(client, urls[1], payload(prefix_b), proposal_b) == (
                    worker_b,
                    0,
                )

                await wait_for_prefix_target(client, urls[0], prefix_a, (worker_a, 0))
                await wait_for_prefix_target(client, urls[1], prefix_b, (worker_b, 0))

                deadline = time.monotonic() + 10
                observed_b = None
                observed_a = None
                while time.monotonic() < deadline:
                    observed_b = await send(
                        client,
                        urls[0],
                        payload(prefix_a, query_only=True),
                        session_b_headers,
                    )
                    observed_a = await send(
                        client,
                        urls[1],
                        payload(prefix_b, query_only=True),
                        session_a_headers,
                    )
                    if observed_b == (worker_b, 0) and observed_a == (worker_a, 0):
                        break

                    assert await send(
                        client, urls[0], payload(prefix_a), session_a_headers
                    ) == (worker_a, 0)
                    assert await send(
                        client, urls[1], payload(prefix_b), session_b_headers
                    ) == (worker_b, 0)
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError(
                        "replica affinity did not converge before the deadline: "
                        f"frontend 1 observed {observed_b}, frontend 2 observed {observed_a}"
                    )

                assert await send(
                    client, urls[0], payload(prefix_a), session_b_headers
                ) == (worker_b, 0)
                assert await send(
                    client, urls[1], payload(prefix_b), session_a_headers
                ) == (worker_a, 0)

        asyncio.run(run_test())


def _test_remote_indexer_decisions(
    engine_workers,
    model_name: str,
    block_size: int = 8,
    use_kv_events: bool = True,
    test_dp_rank: bool = True,
    request_plane: str = "nats",
    store_backend: str = "etcd",
    router_predicted_ttl_secs: Optional[float] = None,
):
    """Validate remote-indexer-backed routing decisions using direct KvRouter instances."""

    async def wait_for_served_indexer(
        runtime,
        expected_query_instances: int,
        expected_record_instances: int,
    ) -> None:
        query_endpoint = runtime.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.kv_indexer_query"
        )
        query_client = await query_endpoint.client()
        record_endpoint = runtime.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.kv_indexer_record_routing_decision"
        )
        record_client = await record_endpoint.client()

        for _ in range(120):
            query_ids = set(query_client.instance_ids())
            record_ids = set(record_client.instance_ids())

            if use_kv_events:
                if len(query_ids) >= expected_query_instances and len(record_ids) == 0:
                    return
            elif (
                len(query_ids) == expected_query_instances
                and len(record_ids) == expected_record_instances
                and query_ids == record_ids
            ):
                return

            await asyncio.sleep(0.5)

        raise TimeoutError("Timed out waiting for served indexer endpoints to register")

    async def run_test(runtimes):
        endpoint_path = (
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )
        expected_num_instances = engine_workers.num_workers

        async def make_router(
            *,
            serve_indexer: bool,
            use_remote_indexer: bool,
            router_predicted_ttl_secs: Optional[float] = None,
        ):
            kv_router_config = KvRouterConfig(
                use_kv_events=use_kv_events,
                router_track_prefill_tokens=True,
                serve_indexer=serve_indexer,
                use_remote_indexer=use_remote_indexer,
                router_predicted_ttl_secs=router_predicted_ttl_secs,
            )
            last_error: Exception | None = None
            for _ in range(60):
                runtime = get_runtime(
                    store_backend=store_backend, request_plane=request_plane
                )
                endpoint = runtime.endpoint(endpoint_path)
                try:
                    with min_initial_workers_env(expected_num_instances):
                        kv_router = KvRouter(
                            endpoint=endpoint,
                            block_size=block_size,
                            kv_router_config=kv_router_config,
                        )
                    return runtime, endpoint, kv_router
                except Exception as error:
                    last_error = error
                    runtime.shutdown()
                    if not (serve_indexer or use_remote_indexer):
                        raise
                    del endpoint
                    del runtime
                    await asyncio.sleep(1.0)

            raise AssertionError(
                "Timed out waiting for model discovery before creating remote-indexer router"
            ) from last_error

        serving_runtimes = []
        serving_endpoints = []
        serving_routers = []

        runtime_a, endpoint_a, router_a = await make_router(
            serve_indexer=True, use_remote_indexer=False
        )
        runtimes.append(runtime_a)
        serving_runtimes.append(runtime_a)
        serving_endpoints.append(endpoint_a)
        serving_routers.append(router_a)

        if use_kv_events:
            runtime_b, endpoint_b, router_b = await make_router(
                serve_indexer=True, use_remote_indexer=False
            )
            runtimes.append(runtime_b)
            serving_runtimes.append(runtime_b)
            serving_endpoints.append(endpoint_b)
            serving_routers.append(router_b)

        await wait_for_served_indexer(
            serving_runtimes[0],
            expected_query_instances=len(serving_routers),
            expected_record_instances=0 if use_kv_events else 1,
        )

        consumer_runtime, consumer_endpoint, consumer_router = await make_router(
            serve_indexer=False,
            use_remote_indexer=True,
            router_predicted_ttl_secs=router_predicted_ttl_secs,
        )
        runtimes.append(consumer_runtime)

        worker_ids = sorted(
            await poll_for_worker_instances(
                serving_endpoints[0], expected_num_instances, max_wait_time=120
            )
        )
        if len(worker_ids) >= 2:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[1]
        elif len(worker_ids) == 1 and test_dp_rank:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[0]
        else:
            raise AssertionError(
                f"Need at least 2 routing targets but got {len(worker_ids)} worker(s) "
                f"with test_dp_rank={test_dp_rank}"
            )

        dp_rank_a = 0 if test_dp_rank else None
        dp_rank_b = 1 if test_dp_rank else None
        logger.info(
            "Remote-indexer routing targets: worker_a=%s/%s worker_b=%s/%s",
            worker_a_id,
            dp_rank_a,
            worker_b_id,
            dp_rank_b,
        )

        async def wait_for_serving_indexers_live(
            worker_id: int, dp_rank: Optional[int]
        ) -> None:
            """Send fresh pinned probes until every serving router indexes one.

            Workers start their KV event publishers just after the generate
            endpoint registers, so the first requests race KV source activation
            in the serving routers. A new probe per attempt replaces any event
            lost in that window instead of waiting for it.
            """
            key = (worker_id, dp_rank if dp_rank is not None else 0)
            deadline = time.monotonic() + 45
            attempt = 0
            stale: dict[int, int] = {}
            while time.monotonic() < deadline:
                attempt += 1
                probe = [random.randint(1, 10000) for _ in range(2 * block_size)]
                try:
                    await asyncio.wait_for(
                        send_request_via_python_kv_router(
                            kv_python_router=serving_routers[0],
                            model_name=model_name,
                            token_ids=probe,
                            stop_conditions={"ignore_eos": True, "max_tokens": 2},
                            worker_id=worker_id,
                            dp_rank=dp_rank,
                        ),
                        timeout=max(deadline - time.monotonic(), 0.1),
                    )
                except asyncio.TimeoutError:
                    raise AssertionError(
                        f"Probe request for {key} outlived the readiness deadline "
                        f"on attempt {attempt}"
                    ) from None
                probe_deadline = min(deadline, time.monotonic() + 2.0)
                while True:
                    stale = {}
                    for idx, serving_router in enumerate(serving_routers):
                        try:
                            scores = await asyncio.wait_for(
                                serving_router.get_overlap_scores(
                                    probe, include_shared=False
                                ),
                                timeout=max(probe_deadline - time.monotonic(), 0.1),
                            )
                        except asyncio.TimeoutError:
                            stale[idx] = -1
                            continue
                        blocks_by_rank = {
                            (row["worker_id"], row["dp_rank"]): row["device_blocks"]
                            for row in scores["workers"]
                        }
                        if blocks_by_rank.get(key, 0) < 2:
                            stale[idx] = blocks_by_rank.get(key, 0)
                    if not stale or time.monotonic() >= probe_deadline:
                        break
                    await asyncio.sleep(0.05)
                if not stale:
                    if attempt > 1:
                        logger.warning(
                            "Serving routers indexed %s only after %d probes",
                            key,
                            attempt,
                        )
                    return
            raise AssertionError(
                f"Serving routers {sorted(stale)} never indexed probes for {key} "
                f"after {attempt} attempts: device_blocks={stale}"
            )

        if use_kv_events:
            await wait_for_serving_indexers_live(worker_a_id, dp_rank_a)
            await wait_for_serving_indexers_live(worker_b_id, dp_rank_b)

        blocks = [
            [random.randint(1, 10000) for _ in range(block_size)] for _ in range(7)
        ]
        A, B, C, D, E, F, G = blocks
        request_specs = [
            (serving_routers[0], A + B, worker_a_id, dp_rank_a, 0.1),
            (serving_routers[0], A + C + D, worker_a_id, dp_rank_a, 0.1),
            (serving_routers[-1], A + C + E, worker_b_id, dp_rank_b, 2.0),
            (consumer_router, A + C + D + F, None, None, 2.0),
            (consumer_router, A + C + E + G, None, None, 2.0),
        ]
        dp_a = dp_rank_a if dp_rank_a is not None else 0
        dp_b = dp_rank_b if dp_rank_b is not None else 0
        expected_overlap_by_request = {
            4: {(worker_a_id, dp_a): 3, (worker_b_id, dp_b): 2},
            5: {(worker_a_id, dp_a): 2, (worker_b_id, dp_b): 3},
        }

        responses: list[dict[str, Optional[int]]] = []
        for i, (
            kv_router,
            token_ids,
            forced_worker_id,
            forced_dp_rank,
            sleep_after,
        ) in enumerate(request_specs, start=1):
            logger.info(
                "Sending remote-indexer request %s/5%s%s",
                i,
                (
                    f" forced_worker_id={forced_worker_id}"
                    if forced_worker_id is not None
                    else ""
                ),
                (
                    f" forced_dp_rank={forced_dp_rank}"
                    if forced_dp_rank is not None
                    else ""
                ),
            )
            if i in expected_overlap_by_request:
                await _assert_overlap_scores(
                    kv_router,
                    token_ids,
                    block_size,
                    expected_overlap_by_request[i],
                    f"request {i}",
                )
            result = await send_request_via_python_kv_router(
                kv_python_router=kv_router,
                model_name=model_name,
                token_ids=token_ids,
                stop_conditions={
                    "ignore_eos": True,
                    "max_tokens": 2,
                },
                worker_id=forced_worker_id,
                dp_rank=forced_dp_rank,
                return_worker_ids=True,
            )
            assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
            responses.append(result)
            if sleep_after > 0:
                await asyncio.sleep(sleep_after)

        req4 = responses[3]
        assert req4["prefill_worker_id"] == worker_a_id, (
            f"Request 4: expected prefill_worker_id={worker_a_id} (longest prefix match), "
            f"got {req4['prefill_worker_id']}"
        )
        if test_dp_rank:
            assert req4["prefill_dp_rank"] == dp_rank_a, (
                f"Request 4: expected prefill_dp_rank={dp_rank_a} "
                f"(longest prefix match), got {req4['prefill_dp_rank']}"
            )

        req5 = responses[4]
        assert req5["prefill_worker_id"] == worker_b_id, (
            f"Request 5: expected prefill_worker_id={worker_b_id} (longest prefix match), "
            f"got {req5['prefill_worker_id']}"
        )
        if test_dp_rank:
            assert req5["prefill_dp_rank"] == dp_rank_b, (
                f"Request 5: expected prefill_dp_rank={dp_rank_b} "
                f"(longest prefix match), got {req5['prefill_dp_rank']}"
            )

        await poll_for_worker_instances(
            consumer_endpoint, expected_num_instances, max_wait_time=120
        )

    async def test_sync():
        runtimes = []
        try:
            await run_test(runtimes)
        finally:
            for runtime in runtimes:
                runtime.shutdown()

    asyncio.run(test_sync())


def _test_python_router_bindings(
    engine_workers,
    endpoint,
    block_size: int,
    model_name: str,
    num_workers: int,
):
    """Test KvRouter Python bindings with token streaming and config overrides.

    Assumes engine_workers are already initialized. This test creates a KvRouter
    Python object and sends three test requests to verify:
    1. Token streaming with full router config overrides (overlap_score_credit, router_temperature)
    2. Token streaming without any overrides (uses default config)
    3. Token streaming with partial override (only router_temperature)

    All requests use ignore_eos=True with varying max_tokens to test token generation control.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        endpoint: Dynamo endpoint for the workers
        block_size: Block size for KV cache
        model_name: Model name to use for requests
        num_workers: Expected number of workers

    Raises:
        AssertionError: If requests fail or router doesn't work correctly
    """
    # Create KvRouterConfig with default settings
    kv_router_config = KvRouterConfig()

    kv_router = _create_kv_router_with_timeout(
        router_factory=lambda: KvRouter(
            endpoint=endpoint,
            block_size=block_size,
            kv_router_config=kv_router_config,
        ),
        num_workers=num_workers,
        engine_workers=engine_workers,
    )

    logger.info("Created KvRouter Python object")

    # Wait for workers to be ready
    asyncio.run(wait_for_workers_ready(endpoint, kv_router, num_workers, model_name))

    # Generate random token IDs (100 to 200 tokens)
    num_input_tokens = random.randint(100, 200)
    token_ids = [random.randint(1, 10000) for _ in range(num_input_tokens)]

    # Set up override parameters
    router_config_override = {
        "overlap_score_credit": 0.5,  # Override the default credit
        "router_temperature": 0.5,  # Override the default temperature
    }

    logger.info(f"Generated {num_input_tokens} random token IDs")

    # Test with full overrides
    logger.info(f"Testing with full router config overrides: {router_config_override}")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 20,  # Generate exactly 20 tokens
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            router_config_override=router_config_override,
        )
    )

    # Test without overrides
    logger.info("Testing without router config overrides")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids[:50],  # Use fewer tokens for second test,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 10,  # Generate exactly 10 tokens for the second test
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            # No router_config_override this time
        )
    )

    # Test with partial override (only temperature)
    partial_override = {"router_temperature": 0.1}
    logger.info(f"Testing with partial router config overrides: {partial_override}")
    asyncio.run(
        send_request_via_python_kv_router(
            kv_python_router=kv_router,
            model_name=model_name,
            token_ids=token_ids[:30],  # Use fewer tokens for third test,
            stop_conditions={
                "ignore_eos": True,  # Don't stop on EOS token
                "max_tokens": 5,  # Generate exactly 5 tokens for the third test
            },
            sampling_options={"temperature": 0.7, "top_p": 0.9},
            output_options={
                "include_input_tokens": False,
                "return_full_text": False,
            },
            router_config_override=partial_override,
        )
    )

    logger.info("KvRouter bindings test completed successfully")


def _test_router_query_instance_id(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
):
    """Test query_instance_id annotation returns worker_instance_id and token_data without routing.

    Assumes engine_workers are already initialized. This function manages router lifecycle.

    This tests the early return optimization where a request with 'nvext.annotations': ['query_instance_id']
    receives metadata without waiting for model generation. The router should:
    1. NOT route the request to a worker for generation
    2. Return worker_instance_id as an SSE event (which worker would handle it)
    3. Return token_data as an SSE event (the tokenized input)
    4. Terminate the stream with [DONE]

    This is useful for clients that want to know which worker will handle a request before
    committing to the full generation (e.g., for request routing decisions).

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".

    Raises:
        AssertionError: If annotation response structure is incorrect or contains generation content
    """

    with KVRouterProcess(
        request, block_size, frontend_port, engine_workers.namespace, store_backend
    ):
        # Start KV router (frontend)
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        url = f"http://localhost:{frontend_port}/v1/chat/completions"

        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=f"http://localhost:{frontend_port}",
                expected_num_workers=engine_workers.num_workers,
                timeout=120,
                test_payload=test_payload,
                engine_workers=engine_workers,
                store_backend=store_backend,
            )
        )

        # Test payload with query_instance_id annotation
        # Format: "query_instance_id:" (colon with empty value) for GAIE aggregated mode
        annotated_payload = {
            **test_payload,
            "nvext": {"annotations": ["query_instance_id:"]},
        }

        async def test_annotation_response():
            """Send request with query_instance_id and validate response structure"""
            async with aiohttp.ClientSession() as session:
                logger.info("Sending request with query_instance_id annotation...")

                async with session.post(url, json=annotated_payload) as response:
                    assert (
                        response.status == 200
                    ), f"Expected 200 but got {response.status}"

                    # Collect all response chunks
                    response_chunks = []
                    async for chunk in response.content:
                        if chunk:
                            chunk_str = chunk.decode("utf-8", errors="replace")
                            response_chunks.append(chunk_str)

                    full_response = "".join(response_chunks)
                    logger.info(
                        f"Full SSE response ({len(full_response)} bytes):\n{full_response}"
                    )

                    worker_id_info = None
                    token_list = None

                    for chunk in parse_sse_json_chunks(full_response):
                        logger.info(f"Parsed chunk: {json.dumps(chunk, indent=2)}")

                        nvext = chunk.get("nvext", {})
                        if "worker_id" in nvext:
                            worker_id_info = nvext["worker_id"]
                            logger.info(f"Found worker_id info: {worker_id_info}")
                        if "token_ids" in nvext:
                            token_list = nvext["token_ids"]
                            logger.info(f"Found token_ids: {len(token_list)} tokens")

                    # Validate worker_id info
                    assert (
                        worker_id_info is not None
                    ), f"Missing worker_id in nvext. Response: {full_response}"

                    # For aggregated mode, both prefill and decode should be the same
                    prefill_worker_id = worker_id_info.get("prefill_worker_id")
                    decode_worker_id = worker_id_info.get("decode_worker_id")
                    assert (
                        prefill_worker_id is not None
                    ), f"Missing prefill_worker_id in worker_id: {worker_id_info}"
                    assert (
                        decode_worker_id is not None
                    ), f"Missing decode_worker_id in worker_id: {worker_id_info}"
                    assert (
                        prefill_worker_id == decode_worker_id
                    ), f"For aggregated mode, prefill and decode worker should be same: {worker_id_info}"

                    # Validate token_ids
                    assert (
                        token_list is not None
                    ), f"Missing token_ids in nvext. Response: {full_response}"
                    assert isinstance(
                        token_list, list
                    ), f"token_ids should be a list, got: {type(token_list)}"
                    assert (
                        len(token_list) > 0
                    ), f"token_ids should not be empty: {token_list}"
                    assert all(
                        isinstance(token, int) for token in token_list
                    ), f"All tokens should be integers: {token_list}"

                    logger.info(
                        f"Valid token_ids with {len(token_list)} tokens: {token_list[:10]}{'...' if len(token_list) > 10 else ''}"
                    )

                    return {
                        "prefill_worker_id": prefill_worker_id,
                        "decode_worker_id": decode_worker_id,
                        "token_count": len(token_list),
                        "tokens": token_list,
                    }

        result = asyncio.run(test_annotation_response())

        logger.info("Successfully validated query_instance_id annotation response:")
        logger.info(f"Prefill Worker ID: {result['prefill_worker_id']}")
        logger.info(f"Decode Worker ID: {result['decode_worker_id']}")
        logger.info(f"Token count: {result['token_count']}")


def _parse_frontend_rejection_metric(
    metrics_text: str, model_name: str, endpoint: str
) -> int:
    """Parse frontend model_rejection_total from Prometheus metrics text.

    Args:
        metrics_text: Raw Prometheus metrics text
        model_name: The model name label value
        endpoint: The endpoint label value (e.g. "chat_completions")

    Returns:
        The metric count, or 0 if not found
    """
    metric_name = f"{name_prefix.FRONTEND}_{frontend_service.MODEL_REJECTION_TOTAL}"
    for line in metrics_text.splitlines():
        if not line.startswith(f"{metric_name}{{"):
            continue
        if f'model="{model_name}"' in line and f'endpoint="{endpoint}"' in line:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    return int(float(parts[1]))
                except ValueError:
                    pass
    return 0


def _get_frontend_rejection_metric(
    frontend_port: int,
    model_name: str,
    endpoint: str,
) -> int:
    """Read the frontend rejection counter from the /metrics endpoint.

    Args:
        frontend_port: Port where the frontend /metrics is served
        model_name: The model name label value
        endpoint: The endpoint label value (e.g. "chat_completions")

    Returns:
        The current rejection count
    """
    metrics_url = f"http://localhost:{frontend_port}/metrics"
    try:
        metrics_response = requests.get(metrics_url, timeout=5)
        metrics_response.raise_for_status()
    except requests.RequestException as e:
        raise AssertionError(
            f"Failed to fetch frontend metrics from {metrics_url}: {e}"
        ) from e

    return _parse_frontend_rejection_metric(metrics_response.text, model_name, endpoint)


def _verify_frontend_rejection_metrics(
    frontend_port: int,
    model_name: str,
    endpoint: str,
    expected_count: int,
) -> None:
    """Verify frontend rejection metrics by scraping the /metrics endpoint."""
    metric_count = _get_frontend_rejection_metric(frontend_port, model_name, endpoint)
    logger.info(f"Frontend rejection metric: model_rejection_total={metric_count}")
    assert metric_count == expected_count, (
        f"Frontend model_rejection_total ({metric_count}) does not match "
        f"expected count ({expected_count})"
    )


def _probe_overload_529_and_assert(
    frontend_port: int,
    test_payload: dict,
    max_tokens: int,
):
    """Send staggered streaming requests until the router rejects with 529.

    Shared core for the aggregated and disaggregated overload tests. The caller
    is responsible for starting the frontend with the desired thresholds and
    waiting until it is ready.

    Sends unique (shuffled) prompts 0.1s apart until the router rejects, then
    asserts:
    1. At least one request is rejected with 529 (the threshold gates the pool)
    2. No other status codes appear
    3. The frontend ``model_rejection_total`` metric increases by the 529 count

    Successes are not required: a single overload-shaped request can exceed the
    threshold before dispatch, so an all-529 burst is a valid outcome.
    """
    url = f"http://localhost:{frontend_port}/v1/chat/completions"
    model_name = test_payload.get("model", "")
    # Read after readiness because its retries can contribute unrelated 529s.
    initial_rejection_count = _get_frontend_rejection_metric(
        frontend_port, model_name, "chat_completions"
    )
    test_payload_529 = {
        **test_payload,
        "max_tokens": max_tokens,
    }

    logger.info("Launching streaming requests until the router returns 529...")

    async def exhaust_resources_and_verify_529():
        stop_event = asyncio.Event()
        observed_statuses = []

        async with aiohttp.ClientSession() as session:
            tasks = []

            async def send_request(req_id, payload):
                try:
                    async with session.post(url, json=payload) as response:
                        status = response.status
                        observed_statuses.append(status)

                        if status == 200:
                            logger.info("Request %s accepted", req_id)
                            await stop_event.wait()
                            return status

                        if status == 529:
                            stop_event.set()
                            body = await response.text()
                            logger.info("Request %s got expected 529: %s", req_id, body)
                            return status

                        body = await response.text()
                        logger.info(
                            "Request %s got unexpected status %s: %s",
                            req_id,
                            status,
                            body,
                        )
                        return status
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.info("Request %s failed: %s", req_id, e)
                    raise

            try:
                for i in range(50):
                    if stop_event.is_set():
                        break

                    content_words = test_payload["messages"][0]["content"].split()
                    random.shuffle(content_words)
                    shuffled_content = " ".join(content_words)
                    unique_payload = {
                        **test_payload_529,
                        "messages": [
                            {
                                **test_payload["messages"][0],
                                "content": shuffled_content,
                            }
                        ],
                    }
                    tasks.append(asyncio.create_task(send_request(i, unique_payload)))
                    await asyncio.sleep(0.1)

                if not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        logger.error("Timed out waiting for overload 529")
            finally:
                stop_event.set()
                # Statuses are recorded when headers arrive, so cancelling a task
                # that is still draining its body cannot drop an observed 529.
                done, pending = await asyncio.wait(tasks, timeout=5)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()

            return observed_statuses

    results = asyncio.run(exhaust_resources_and_verify_529())

    # Count outcomes
    num_succeeded = sum(1 for s in results if s == 200)
    num_rejected = sum(1 for s in results if s == 529)
    num_other = sum(1 for s in results if s not in (200, 529))

    logger.info(
        "Results: %s succeeded, %s rejected (529), %s other",
        num_succeeded,
        num_rejected,
        num_other,
    )

    # Assert minimum thresholds
    assert (
        num_other == 0
    ), f"Expected only 200 or 529 responses, but got {num_other} other"
    assert num_rejected > 0, f"Expected at least 1 rejection, but got {num_rejected}"

    # Verify rejection metrics from frontend /metrics endpoint
    _verify_frontend_rejection_metrics(
        frontend_port,
        model_name,
        "chat_completions",
        initial_rejection_count + num_rejected,
    )

    logger.info(
        "Successfully verified overload 529: %s rejected, %s succeeded, metrics match",
        num_rejected,
        num_succeeded,
    )


def _test_router_overload_529(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    blocks_threshold: float | str | None = 0.2,
    tokens_threshold: int | str | None = None,
    tokens_threshold_frac: float | str | None = None,
    router_queue_threshold: float | str | None = None,
    max_tokens: int = 50,
):
    """Test that 529 is returned when all workers are busy, and verify rejection metrics.

    Assumes engine_workers are already initialized. This function manages router lifecycle.
    Uses limited resources to intentionally trigger the overload condition.

    Sends staggered requests (0.1s apart) to exhaust worker resources, then verifies:
    1. Every observed response is either 200 or 529
    2. At least one request is rejected with 529 (worker busy)
    3. The frontend model_rejection_total increase matches the observed 529 count

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache (should be small to exhaust quickly, e.g. 4)
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        blocks_threshold: Active decode blocks threshold for the router (default 0.2)
        tokens_threshold: Active prefill tokens threshold for the router
        tokens_threshold_frac: Fractional active prefill tokens threshold for the router
        router_queue_threshold: Router queue threshold, or "None" to disable queueing
        max_tokens: Output token count for generated overload requests

    Raises:
        AssertionError: If success/rejection counts or metrics don't meet expectations
    """
    logger.info(
        f"Starting KV router frontend on port {frontend_port} with limited resources"
    )

    with KVRouterProcess(
        request=request,
        block_size=block_size,
        frontend_port=frontend_port,
        namespace=engine_workers.namespace,
        blocks_threshold=blocks_threshold,
        tokens_threshold=tokens_threshold,
        tokens_threshold_frac=tokens_threshold_frac,
        router_queue_threshold=router_queue_threshold,
    ):
        frontend_url = f"http://localhost:{frontend_port}"

        logger.info("Waiting for frontend readiness before overload test...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=60,
                engine_workers=engine_workers,
            )
        )

        _probe_overload_529_and_assert(frontend_port, test_payload, max_tokens)


def _test_disagg_router_overload_529(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    blocks_threshold: float | str | None = None,
    tokens_threshold: int | str | None = None,
    tokens_threshold_frac: float | str | None = None,
    router_queue_threshold: float | str | None = None,
    max_tokens: int = 50,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Verify disaggregated load-shedding: clients get 529 when the gated pool is busy.

    Assumes the prefill and decode workers are already running (kept alive by the
    caller); this function owns the frontend (router) lifecycle. Registered
    prefill and decode worker types establish separate pools. The model only
    becomes ready (listed in ``/v1/models``) once both worker types are available,
    so the readiness wait below gates on prefill registration.

    Two configurations exercise the two pools (driven by the thresholds the
    caller passes):
    - Prefill rejection: low ``tokens_threshold`` (active-prefill-tokens),
      decode threshold disabled. Gates the prefill pool — the path that was
      previously a silent no-op.
    - Decode rejection: low ``blocks_threshold`` (active-decode-blocks), prefill
      threshold disabled. Gates the decode pool.

    In both cases the shared probe asserts that some requests succeed, some are
    rejected with 529, and the rejection metric matches.

    Raises:
        AssertionError: If no 529 is observed (the threshold did not gate the
        pool) or if any non-200/529 status appears.
    """
    with KVRouterProcess(
        request=request,
        block_size=block_size,
        frontend_port=frontend_port,
        namespace=decode_workers.namespace,
        store_backend=store_backend,
        blocks_threshold=blocks_threshold,
        tokens_threshold=tokens_threshold,
        tokens_threshold_frac=tokens_threshold_frac,
        router_queue_threshold=router_queue_threshold,
        request_plane=request_plane,
        min_initial_workers=decode_workers.num_workers,
    ):
        logger.info(
            "Starting disagg KV router frontend on port %s for overload 529 test",
            frontend_port,
        )
        frontend_url = f"http://localhost:{frontend_port}"

        logger.info(
            "Waiting for prefill and decode workers to register with frontend..."
        )
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=(
                    prefill_workers.num_workers + decode_workers.num_workers
                ),
                timeout=120,
                engine_workers=[prefill_workers, decode_workers],
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        _probe_overload_529_and_assert(frontend_port, test_payload, max_tokens)


def _test_router_threshold_none_disables_rejection(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    num_requests: int = 4,
):
    """Test that explicit CLI "None" thresholds disable busy-based rejection end to end.

    Assumes engine_workers are already initialized. This function manages router lifecycle.
    Starts the frontend with literal CLI "None" values for all three threshold knobs,
    verifies the /busy_threshold API reports nulls, then sends overload-shaped traffic and
    confirms no request is rejected with 529 and the frontend rejection metric stays at 0.

    Args:
        engine_workers: Backend workers (mocker/vllm) already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        num_requests: Number of concurrent requests to send under load

    Raises:
        AssertionError: If thresholds are not null or any request is rejected/fails
    """
    logger.info(
        "Starting KV router frontend on port %s with explicit CLI None thresholds",
        frontend_port,
    )

    with reserved_ports(1, DynamoPortRange.FRONTEND) as (admin_port,), KVRouterProcess(
        request=request,
        block_size=block_size,
        frontend_port=frontend_port,
        namespace=engine_workers.namespace,
        blocks_threshold="None",
        tokens_threshold="None",
        tokens_threshold_frac="None",
        extra_env={
            "DYN_ENABLE_FRONTEND_ADMIN_API": "1",
            "DYN_FRONTEND_ADMIN_PORT": str(admin_port),
        },
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"
        busy_threshold_url = f"http://127.0.0.1:{admin_port}/busy_threshold"
        model_name = test_payload.get("model", "test-model")
        load_payload = {
            **test_payload,
            "max_tokens": 50,
        }

        logger.info("Waiting for frontend readiness before explicit-None test...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=60,
                engine_workers=engine_workers,
            )
        )

        async def verify_thresholds_and_send_load():
            async with aiohttp.ClientSession() as session:
                logger.info(
                    "Checking /busy_threshold reports nulls for explicit CLI None thresholds"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (get) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_decode_blocks_threshold") is None
                    ), f"Expected active_decode_blocks_threshold=None: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold") is None
                    ), f"Expected active_prefill_tokens_threshold=None: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold_frac") is None
                    ), f"Expected active_prefill_tokens_threshold_frac=None: {data}"
                    logger.info(
                        "POST /busy_threshold returned expected null thresholds: %s",
                        data,
                    )

            logger.info(
                "Launching overload-shaped traffic with explicit None thresholds..."
            )
            stop_event = asyncio.Event()
            response_statuses = asyncio.Queue()

            async with aiohttp.ClientSession() as session:

                async def send_request(req_id: int, payload: dict) -> int:
                    try:
                        async with session.post(chat_url, json=payload) as response:
                            if response.status == 200:
                                logger.info(
                                    "Request %s accepted without rejection", req_id
                                )
                                await response_statuses.put(response.status)
                                await stop_event.wait()
                                return response.status

                            body = await response.text()
                            logger.info(
                                "Request %s got unexpected status %s: %s",
                                req_id,
                                response.status,
                                body,
                            )
                            await response_statuses.put(response.status)
                            return response.status
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.info("Request %s failed with error: %s", req_id, exc)
                        await response_statuses.put(exc)
                        raise

                tasks = []
                try:
                    for i in range(num_requests):
                        content_words = test_payload["messages"][0]["content"].split()
                        random.shuffle(content_words)
                        unique_payload = {
                            **load_payload,
                            "messages": [
                                {
                                    **test_payload["messages"][0],
                                    "content": " ".join(content_words),
                                }
                            ],
                        }
                        tasks.append(
                            asyncio.create_task(send_request(i, unique_payload))
                        )
                        await asyncio.sleep(0.1)
                finally:
                    initial_results = [
                        await response_statuses.get() for _ in range(num_requests)
                    ]
                    stop_event.set()
                    done = await asyncio.gather(*tasks, return_exceptions=True)

                for result in initial_results + done:
                    if isinstance(result, Exception):
                        raise result

                return done

        results = asyncio.run(verify_thresholds_and_send_load())

        num_succeeded = sum(1 for status in results if status == 200)
        num_rejected = sum(1 for status in results if status == 529)
        num_other = sum(1 for status in results if status not in (200, 529))

        logger.info(
            "Results with explicit None thresholds: %s succeeded, %s rejected (529), %s other",
            num_succeeded,
            num_rejected,
            num_other,
        )

        assert num_rejected == 0, f"Expected 0 rejections, but got {num_rejected}"
        assert (
            num_other == 0
        ), f"Expected only 200 or 529 responses, but got {num_other} other"
        assert num_succeeded > 0, "Expected at least one successful request"
        assert (
            num_succeeded == num_requests
        ), f"Expected {num_requests} successful requests, got {num_succeeded}"

        _verify_frontend_rejection_metrics(
            frontend_port,
            model_name,
            "chat_completions",
            0,
        )

        logger.info(
            "Explicit CLI None thresholds disabled busy rejection as expected: %s successes, metrics clean",
            num_succeeded,
        )


async def _zmq_replay_cycle(
    phase: int,
    router,
    router_name: str,
    endpoint,
    indexer_url: str,
    engine_workers,
    send_requests_to_router,
    model_name: str,
):
    """Pause indexer listeners, create gaps, then force each stream to reveal them."""
    await asyncio.sleep(1)
    worker_ids = list(engine_workers.worker_id_to_zmq_ports.keys())
    dp_size = getattr(engine_workers, "dp_size", None) or 1

    logger.info(f"=== ZMQ REPLAY TEST: Phase {phase} ({router_name}) ===")
    async with aiohttp.ClientSession() as session:
        for wid in worker_ids:
            for dp_rank in range(dp_size):
                async with session.post(
                    f"{indexer_url}/test/pause_listener",
                    json={"instance_id": wid, "dp_rank": dp_rank},
                ) as resp:
                    assert (
                        resp.status == 200
                    ), f"Pause {wid}:{dp_rank} failed: {await resp.text()}"

    logger.info("Sending 10 requests while indexer listeners are paused")
    successful_gap = await send_requests_to_router(
        router, 10, f"{router_name} (indexer paused)", endpoint
    )
    assert (
        successful_gap == 10
    ), f"Expected 10 requests while paused, got {successful_gap}"

    async with aiohttp.ClientSession() as session:
        for wid in worker_ids:
            for dp_rank in range(dp_size):
                async with session.post(
                    f"{indexer_url}/test/resume_listener",
                    json={"instance_id": wid, "dp_rank": dp_rank},
                ) as resp:
                    assert (
                        resp.status == 200
                    ), f"Resume {wid}:{dp_rank} failed: {await resp.text()}"

    replay_targets = [
        (wid, dp_rank) for wid in worker_ids for dp_rank in range(dp_size)
    ]
    logger.info(
        "Sending %s targeted requests after resume (triggers gap detection + replay)",
        len(replay_targets),
    )
    post_resume_tasks = []
    for wid, dp_rank in replay_targets:
        request_tokens = [random.randint(1, 10000) for _ in range(30)]
        post_resume_tasks.append(
            asyncio.create_task(
                send_request_via_python_kv_router(
                    kv_python_router=router,
                    model_name=model_name,
                    token_ids=request_tokens,
                    stop_conditions={
                        "ignore_eos": True,
                        "max_tokens": 10,
                    },
                    worker_id=wid,
                    dp_rank=dp_rank,
                )
            )
        )

    post_resume_results = await asyncio.gather(*post_resume_tasks)
    successful_post = sum(1 for result in post_resume_results if result)
    assert successful_post == len(replay_targets), (
        f"Expected {len(replay_targets)} targeted post-resume requests, "
        f"got {successful_post}"
    )
    await asyncio.sleep(2)


def _test_router_indexers_sync(
    engine_workers,
    block_size: int,
    model_name: str,
    num_workers: int,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    event_plane: str | None = None,
    test_nats_interruption: bool = False,
    nats_server: Optional["NatsServer"] = None,
    router_event_threads: int = 4,
    standalone_indexer_url: Optional[str] = None,
    standalone_indexer_b_url: Optional[str] = None,
    test_zmq_replay: bool = False,
):
    """Test that two KV routers have synchronized indexer states after processing requests.

    Assumes engine_workers are already initialized. This test:
    1. Creates first KvRouter (with its own runtime) and sends 25 requests
    2. Creates second KvRouter (with its own runtime, which recovers from workers)
    3. Sends 25 requests to second router
    4. Dumps states from both routers and compares them (should be identical)

    This validates that routers can recover and synchronize state from worker-local indexers.

    When test_nats_interruption=True (requires nats_server and request_plane="tcp"):
    - After first router sends 25 requests, NATS is stopped
    - 10 more requests sent while NATS is down (stored locally by local indexer)
    - NATS restarted (fresh state), recovery mechanism re-syncs
    - Second router starts and sends 25 requests
    - NATS stopped again, 10 more requests sent
    - NATS restarted, 5 more requests sent
    - Verify both routers converge to same state

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        block_size: Block size for KV cache
        model_name: Model name to use for requests
        num_workers: Expected number of workers
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats" or "tcp"). Defaults to "nats".
        event_plane: Event plane to use ("nats" or "zmq"). Defaults to runtime behavior.
        test_nats_interruption: If True, test NATS interruption recovery. Defaults to False.
        nats_server: NatsServer instance for stop/start (required if test_nats_interruption=True).

    Raises:
        AssertionError: If router states don't synchronize correctly or snapshot is missing
    """
    if test_nats_interruption and nats_server is None:
        raise ValueError("nats_server is required when test_nats_interruption=True")

    # Use async to manage the test flow
    async def run_test(runtime_stack):
        kv_router_config = KvRouterConfig(router_event_threads=router_event_threads)

        # If standalone indexer mode, launch workers one-by-one and register.
        # We need to create a temporary endpoint just to discover worker IDs.
        if standalone_indexer_url:
            tmp_runtime = runtime_stack.enter_context(
                managed_runtime(store_backend, request_plane, event_plane=event_plane)
            )
            tmp_endpoint = tmp_runtime.endpoint(
                f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
            )
            await engine_workers.launch_workers_with_indexer(tmp_endpoint)

        def sort_key(event):
            data = event["event"]["data"]["stored"]
            blocks = data["blocks"]
            first_block = blocks[0]
            return (
                event["worker_id"],
                first_block["tokens_hash"],
                data["parent_hash"],
            )

        expected_standalone_key = f"{model_name}:default"

        async def fetch_standalone_events(session, indexer_url, indexer_label):
            async with session.get(f"{indexer_url}/dump") as resp:
                assert resp.status == 200, f"GET /dump failed: {resp.status}"
                dump = await resp.json()

            assert expected_standalone_key in dump, (
                f"{indexer_label} missing dump key '{expected_standalone_key}', "
                f"got keys={list(dump.keys())}"
            )
            for k, v in dump.items():
                assert (
                    isinstance(v, dict) and "events" in v
                ), f"{indexer_label} dump key '{k}' returned unexpected format: {v}"
            return sorted(dump[expected_standalone_key]["events"], key=sort_key)

        async def wait_for_standalone_events(
            indexer_url, expected_events, expected_label, actual_label
        ):
            last_error = None
            async with aiohttp.ClientSession() as session:
                for attempt in range(50):
                    try:
                        actual_events = await fetch_standalone_events(
                            session, indexer_url, actual_label
                        )
                        logger.info(
                            f"{actual_label} has {len(actual_events)} events "
                            f"(attempt {attempt + 1})"
                        )
                        assert_event_dumps_equal(
                            expected_events,
                            actual_events,
                            expected_label,
                            actual_label,
                        )
                        return actual_events
                    except (AssertionError, aiohttp.ClientError) as exc:
                        last_error = exc
                        await asyncio.sleep(0.2)

            assert last_error is not None
            raise last_error

        async def send_requests_to_router(router, num_requests, router_name, endpoint):
            # Now send the actual requests
            tasks = []
            for i in range(num_requests):
                # Generate random token IDs for each request
                logger.debug(f"Sending request {i + 1}/{num_requests} to {router_name}")

                # Generate 30 random tokens
                request_tokens = [random.randint(1, 10000) for _ in range(30)]

                # Send request to mocker via the router
                tasks.append(
                    asyncio.create_task(
                        send_request_via_python_kv_router(
                            kv_python_router=router,
                            model_name=model_name,
                            token_ids=request_tokens,
                            stop_conditions={
                                "ignore_eos": True,  # Don't stop on EOS token
                                "max_tokens": 10,  # Generate exactly 10 tokens
                            },
                        )
                    )
                )

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            successful = sum(1 for r in results if r)
            logger.info(
                f"Completed {successful}/{num_requests} requests for {router_name}"
            )
            return successful

        # Create first runtime and endpoint for router 1
        logger.info("Creating first KV router with its own runtime")
        runtime1 = runtime_stack.enter_context(
            managed_runtime(store_backend, request_plane, event_plane=event_plane)
        )
        endpoint1 = runtime1.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )

        kv_router1 = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint1,
                block_size=block_size,
                kv_router_config=kv_router_config,
            ),
            num_workers=num_workers,
            engine_workers=engine_workers,
        )

        # Wait for workers to be ready
        await wait_for_workers_ready(endpoint1, kv_router1, num_workers, model_name)

        # Send 25 requests to first router
        logger.info("Sending 25 requests to first router")

        # Send requests to first router
        successful1 = await send_requests_to_router(
            kv_router1, 25, "Router 1", endpoint1
        )
        assert (
            successful1 == 25
        ), f"Expected 25 successful requests to router 1, got {successful1}"

        # NATS interruption test: stop NATS, send requests, restart
        if test_nats_interruption:
            await asyncio.sleep(1)

            assert nats_server is not None  # Validated at function entry
            logger.info("=== NATS INTERRUPTION TEST: Phase 1 ===")
            logger.info("Stopping NATS server")
            nats_server.stop()

            logger.info("Sending 10 requests while NATS is down (via TCP)")
            successful_offline1 = await send_requests_to_router(
                kv_router1, 10, "Router 1 (NATS down)", endpoint1
            )
            assert (
                successful_offline1 == 10
            ), f"Expected 10 successful requests while NATS down, got {successful_offline1}"

            logger.info("Restarting NATS server (fresh state)")
            nats_server.start()

            await asyncio.sleep(5)

        if test_zmq_replay and standalone_indexer_url:
            await _zmq_replay_cycle(
                1,
                kv_router1,
                "Router 1",
                endpoint1,
                standalone_indexer_url,
                engine_workers,
                send_requests_to_router,
                model_name,
            )

        await asyncio.sleep(1)

        if standalone_indexer_url and standalone_indexer_b_url:
            logger.info(
                "Waiting for Standalone A to match Router 1 before launching Indexer B"
            )
            state1_before_peer_json = await kv_router1.dump_events()
            sorted_state1_before_peer = sorted(
                json.loads(state1_before_peer_json), key=sort_key
            )
            await wait_for_standalone_events(
                standalone_indexer_url,
                sorted_state1_before_peer,
                "Router 1",
                "Standalone A",
            )
            logger.info("Standalone A matches Router 1 before Indexer B recovery")

        # Create second runtime and endpoint for router 2
        logger.info("Creating second KV router with its own runtime")
        runtime2 = runtime_stack.enter_context(
            managed_runtime(store_backend, request_plane, event_plane=event_plane)
        )
        endpoint2 = runtime2.endpoint(
            f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
        )

        kv_router2 = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint2,
                block_size=block_size,
                kv_router_config=kv_router_config,
            ),
            num_workers=num_workers,
            engine_workers=engine_workers,
        )

        # Launch Indexer B alongside Router 2. Workers are passed via --workers
        # so ZMQ sockets connect before recovery, avoiding the slow-joiner problem.
        if standalone_indexer_b_url:
            engine_workers.launch_indexer()
            await wait_for_indexer_workers_active(
                standalone_indexer_b_url, engine_workers.worker_id_to_zmq_ports
            )
            logger.info(
                f"Launched Indexer B at {standalone_indexer_b_url} "
                f"(P2P recovery from Indexer A)"
            )

        # Send 25 requests to second router with initial retry loop
        logger.info("Sending 25 requests to second router")
        successful2 = await send_requests_to_router(
            kv_router2, 25, "Router 2", endpoint2
        )
        assert (
            successful2 == 25
        ), f"Expected 25 successful requests to router 2, got {successful2}"

        # NATS interruption test: stop NATS again, send requests, restart, send more
        if test_nats_interruption:
            await asyncio.sleep(1)

            assert nats_server is not None  # Validated at function entry
            logger.info("=== NATS INTERRUPTION TEST: Phase 2 ===")
            logger.info("Stopping NATS server")
            nats_server.stop()

            logger.info("Sending 10 requests while NATS is down (via TCP)")
            successful_offline2 = await send_requests_to_router(
                kv_router2, 10, "Router 2 (NATS down)", endpoint2
            )
            assert (
                successful_offline2 == 10
            ), f"Expected 10 successful requests while NATS down, got {successful_offline2}"

            logger.info("Restarting NATS server (fresh state)")
            nats_server.start()
            await asyncio.sleep(5)

            logger.info("Sending 5 more requests after NATS recovery")
            successful_recovery = await send_requests_to_router(
                kv_router1, 5, "Router 1 (post-recovery)", endpoint1
            )
            assert (
                successful_recovery == 5
            ), f"Expected 5 successful requests post-recovery, got {successful_recovery}"

        if test_zmq_replay and standalone_indexer_url:
            await _zmq_replay_cycle(
                2,
                kv_router2,
                "Router 2",
                endpoint2,
                standalone_indexer_url,
                engine_workers,
                send_requests_to_router,
                model_name,
            )

        # Wait for internal synchronization and ZMQ event propagation
        logger.info("Waiting for final synchronization")
        await asyncio.sleep(2)

        # Dump states from all sources
        logger.info("Dumping states from all sources")
        state1_json = await kv_router1.dump_events()
        state2_json = await kv_router2.dump_events()

        state1 = json.loads(state1_json)
        state2 = json.loads(state2_json)

        sorted_state1 = sorted(state1, key=sort_key)
        sorted_state2 = sorted(state2, key=sort_key)

        logger.info(f"Router 1 has {len(sorted_state1)} events")
        logger.info(f"Router 2 has {len(sorted_state2)} events")

        assert_event_dumps_equal(sorted_state1, sorted_state2, "Router 1", "Router 2")
        logger.info("Successfully verified Router 1 and Router 2 states are equal")

        # Verify standalone HTTP indexers build the same tree (via ZMQ)
        if standalone_indexer_url:
            sorted_standalone_a = await wait_for_standalone_events(
                standalone_indexer_url, sorted_state1, "Router 1", "Standalone A"
            )
            logger.info("Standalone A matches Router 1")

            if standalone_indexer_b_url:
                await wait_for_standalone_events(
                    standalone_indexer_b_url,
                    sorted_standalone_a,
                    "Standalone A",
                    "Standalone B",
                )
                logger.info(
                    "All 4 dumps match: Router 1, Router 2, Standalone A, Standalone B"
                )

    async def test_sync():
        with contextlib.ExitStack() as runtime_stack:
            await run_test(runtime_stack)

    asyncio.run(test_sync())

    logger.info("Indexers sync test completed successfully")


def _test_router_decisions_disagg(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    router_ais_config: Optional[dict[str, Any]] = None,
    enable_bootstrap: bool = False,
    require_kv_hit: bool = False,
):
    """Validate KV cache prefix reuse in disaggregated prefill-decode setup via HTTP frontend.

    Assumes prefill_workers and decode_workers are already initialized. This function manages
    router lifecycle and sends progressive requests with overlapping prefixes.

    This test:
    1. Starts the KV router frontend with disagg support
    2. Sends 4 progressive requests where each extends the previous tokens by block_size
    3. Extracts prefill_worker_id and decode_worker_id from response nvext
    4. Verifies all prefill_worker_ids are the same (due to prefix reuse routing)
    5. Verifies prefill_worker_id is NOT in the set of decode_worker_ids (true disagg)

    Args:
        prefill_workers: Prefill workers already initialized with __enter__()
        decode_workers: Decode workers already initialized with __enter__()
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload to send to /v1/chat/completions
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        router_ais_config: Optional AIS router perf-model config for frontend KV routing.
        require_kv_hit: If True, also assert the router predicted a KV cache hit
            (nvext.timing.kv_hit_rate > 0) for requests 2-4, which share a prefix
            with the previous request. This catches worker/router block-hash
            mismatches that routing to the same worker alone would not.

    Raises:
        AssertionError: If prefill_worker_ids differ across requests (prefix reuse failure)
        AssertionError: If prefill_worker_id is in decode_worker_ids (not true disagg)
    """
    with KVRouterProcess(
        request,
        block_size,
        frontend_port,
        decode_workers.namespace,
        store_backend,
        request_plane=request_plane,
        min_initial_workers=decode_workers.num_workers,
        router_ais_config=router_ais_config,
    ):
        # Start KV router frontend - uses decode_workers namespace for discovery
        # The frontend will auto-discover both prefill and decode workers
        logger.info(
            f"Starting KV router frontend on port {frontend_port} for disagg test"
        )

        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        # Wait for workers to register with frontend
        logger.info(
            "Waiting for prefill and decode workers to register with frontend..."
        )
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=(
                    prefill_workers.num_workers + decode_workers.num_workers
                ),
                timeout=120,
                engine_workers=[prefill_workers, decode_workers],
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        async def send_progressive_requests():
            """Send 4 progressive requests with overlapping prefixes and collect worker IDs."""
            prefill_worker_ids = []
            decode_worker_ids = []

            # Generate base tokens for progressive prefix extension
            base_content = test_payload["messages"][0]["content"]

            async with aiohttp.ClientSession() as session:
                for i in range(4):
                    # Build progressive content by repeating base content
                    # Each iteration adds more content to extend the prefix
                    progressive_content = " ".join([base_content] * (i + 1))

                    # Create payload with worker_id and timing in extra_fields
                    payload = {
                        **test_payload,
                        "messages": [
                            {
                                "role": "user",
                                "content": progressive_content,
                            }
                        ],
                        "nvext": {"extra_fields": ["worker_id", "timing"]},
                        "stream": True,
                    }

                    logger.info(
                        f"Sending request {i + 1}/4 with progressive prefix "
                        f"(~{len(progressive_content)} chars)"
                    )

                    async with session.post(chat_url, json=payload) as response:
                        assert (
                            response.status == 200
                        ), f"Request {i + 1} failed with status {response.status}"

                        # Collect all chunks and look for nvext with worker_id and timing
                        prefill_wid = None
                        decode_wid = None
                        timing_info = None

                        body = await response.text()
                        for data in parse_sse_json_chunks(body):
                            nvext = data.get("nvext", {})
                            worker_id_info = nvext.get("worker_id", {})
                            if "prefill_worker_id" in worker_id_info:
                                prefill_wid = worker_id_info["prefill_worker_id"]
                            if "decode_worker_id" in worker_id_info:
                                decode_wid = worker_id_info["decode_worker_id"]
                            if "timing" in nvext:
                                timing_info = nvext["timing"]

                        logger.info(
                            f"Request {i + 1}: prefill_worker_id={prefill_wid}, "
                            f"decode_worker_id={decode_wid}, timing={timing_info}"
                        )

                        if prefill_wid is not None:
                            prefill_worker_ids.append(prefill_wid)
                        if decode_wid is not None:
                            decode_worker_ids.append(decode_wid)

                        # Verify timing info is present and valid.
                        # kv_transfer_estimated_latency_ms is measured on both the original
                        # and bootstrap prefill paths (uses first_token_time as stop).
                        assert (
                            timing_info is not None
                        ), f"Request {i + 1}: Expected timing info in final chunk, got None"
                        verify_response_timing(timing_info, disagg=not enable_bootstrap)
                        if require_kv_hit and i > 0:
                            kv_hit_rate = timing_info.get("kv_hit_rate")
                            assert kv_hit_rate is not None and kv_hit_rate > 0, (
                                f"Request {i + 1}: expected kv_hit_rate > 0 on a shared "
                                f"prefix, got {kv_hit_rate}; worker KV event hashes "
                                f"may not match the router's"
                            )

                    # Small delay between requests
                    await asyncio.sleep(1)

            return prefill_worker_ids, decode_worker_ids

        # Run the progressive requests
        prefill_ids, decode_ids = asyncio.run(send_progressive_requests())

        logger.info(f"Collected prefill_worker_ids: {prefill_ids}")
        logger.info(f"Collected decode_worker_ids: {decode_ids}")

        # Verify we got worker IDs from all requests
        assert len(prefill_ids) == 4, (
            f"Expected 4 prefill_worker_ids, got {len(prefill_ids)}. "
            f"Make sure nvext.extra_fields=['worker_id'] is being processed."
        )

        # Verify prefix reuse behavior.
        #
        # With KV events enabled, the router learns cache state asynchronously.
        # With the TCP request plane, we can observe a transient on the *first* request where
        # the second request is routed before the first request's KV "stored" events have been
        # fully ingested. After ingestion, routing stabilizes.
        #
        # So for TCP we assert that requests 2-4 converge to the same prefill worker; for NATS
        # request plane we keep the stronger assertion that all 4 match.
        if request_plane == "tcp":
            unique_prefill_ids = set(prefill_ids[1:])
            assert len(unique_prefill_ids) == 1, (
                f"Expected prefill requests 2-4 to route to the same worker due to prefix reuse, "
                f"but found {len(unique_prefill_ids)} unique prefill_worker_ids: {unique_prefill_ids}. "
                f"Full list: {prefill_ids}"
            )
        else:
            unique_prefill_ids = set(prefill_ids)
            assert len(unique_prefill_ids) == 1, (
                f"Expected all prefill requests to route to the same worker due to prefix reuse, "
                f"but found {len(unique_prefill_ids)} unique prefill_worker_ids: {unique_prefill_ids}. "
                f"Full list: {prefill_ids}"
            )

        # Verify prefill_worker_id is NOT in decode_worker_ids (true disagg)
        unique_decode_ids = set(decode_ids)
        prefill_id = prefill_ids[0]
        assert prefill_id not in unique_decode_ids, (
            f"Prefill worker {prefill_id} should NOT be in decode workers {unique_decode_ids}. "
            f"This suggests disaggregated mode is not working correctly - "
            f"prefill and decode should use separate worker pools."
        )

        logger.info(
            f"Successfully verified disaggregated routing:\n"
            f"  - All 4 requests routed to same prefill_worker_id={prefill_id} (prefix reuse)\n"
            f"  - Prefill worker is NOT in decode worker set {unique_decode_ids} (true disagg)"
        )


def _test_disagg_topology_required_prefill_pin_match_and_mismatch(
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    prefill_zone_a_id: int,
    prefill_zone_b_id: int,
    shared_namespace: str,
    request_plane: str = "tcp",
):
    """Validate required topology constraints derived from pinned prefill workers."""
    with KVRouterProcess(
        request,
        block_size,
        frontend_port,
        namespace=decode_workers.namespace,
        request_plane=request_plane,
        min_initial_workers=decode_workers.num_workers,
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        async def run_requests() -> None:
            await wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=decode_workers.num_workers,
                timeout=120,
                engine_workers=decode_workers,
                request_plane=request_plane,
            )

            runtime = get_runtime(request_plane=request_plane)
            prefill_endpoint = runtime.endpoint(f"{shared_namespace}.prefill.generate")

            await poll_for_worker_instances(
                prefill_endpoint, decode_workers.num_workers
            )

            async def post_expect_status(
                session: aiohttp.ClientSession,
                payload: dict,
                expected_status: int,
                message: str,
                retry_statuses: set[int] | None = None,
                timeout_s: float = 30.0,
            ) -> str:
                retry_statuses = retry_statuses or set()
                deadline = asyncio.get_running_loop().time() + timeout_s
                attempt = 0
                last_status = None
                last_body = ""

                while True:
                    attempt += 1
                    async with session.post(chat_url, json=payload) as response:
                        response_body = await response.text()
                        if response.status == expected_status:
                            return response_body

                        last_status = response.status
                        last_body = response_body

                    if (
                        last_status not in retry_statuses
                        or asyncio.get_running_loop().time() >= deadline
                    ):
                        raise AssertionError(
                            f"{message}, got status={last_status} body={last_body}"
                        )

                    logger.info(
                        "%s not ready yet: status=%s attempt=%s; retrying...",
                        message,
                        last_status,
                        attempt,
                    )
                    await asyncio.sleep(1.0)

            zone_a_payload = {
                **test_payload,
                "nvext": {
                    "prefill_worker_id": prefill_zone_a_id,
                },
            }
            topology_ready_payload = {
                **zone_a_payload,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False,
            }
            logger.info("Waiting for topology-valid frontend readiness...")
            await wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=decode_workers.num_workers,
                timeout=120,
                test_payload=topology_ready_payload,
                engine_workers=decode_workers,
                request_plane=request_plane,
            )

            async with aiohttp.ClientSession() as session:
                await post_expect_status(
                    session,
                    zone_a_payload,
                    200,
                    "Expected required KV-transfer topology match to succeed",
                    retry_statuses={404},
                )

            zone_b_payload = {
                **test_payload,
                "nvext": {
                    "prefill_worker_id": prefill_zone_b_id,
                },
            }
            async with aiohttp.ClientSession() as session:
                await post_expect_status(
                    session,
                    zone_b_payload,
                    500,
                    "Expected required KV-transfer topology mismatch to fail",
                )

        asyncio.run(run_requests())


def _test_router_decisions_disagg_round_robin_prefill_dp_rank(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    expected_prefill_dp_ranks: int,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Verify disaggregated round-robin requests store prefill KV blocks across DP ranks."""

    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        decode_workers.namespace,
        store_backend,
        request_plane=request_plane,
        router_mode="round-robin",
        min_initial_workers=decode_workers.num_workers,
    ):
        logger.info(
            "Starting round-robin frontend on port %s for disagg prefill dp-rank test",
            frontend_port,
        )

        async def run_test(runtime):
            frontend_url = f"http://localhost:{frontend_port}"
            chat_url = f"{frontend_url}/v1/chat/completions"
            await wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=(
                    prefill_workers.num_workers + decode_workers.num_workers
                ),
                timeout=120,
                engine_workers=[prefill_workers, decode_workers],
                store_backend=store_backend,
                request_plane=request_plane,
            )

            prefill_endpoint = runtime.endpoint(
                f"{prefill_workers.namespace}.prefill.generate"
            )

            observer_router = _create_kv_router_with_timeout(
                router_factory=lambda: KvRouter(
                    endpoint=prefill_endpoint,
                    block_size=block_size,
                    kv_router_config=KvRouterConfig(
                        use_kv_events=True,
                        router_event_threads=4,
                        router_track_prefill_tokens=True,
                        router_prefill_load_model="none",
                    ),
                ),
                num_workers=prefill_workers.num_workers,
                engine_workers=prefill_workers,
            )

            worker_ids = sorted(
                await poll_for_worker_instances(
                    prefill_endpoint,
                    prefill_workers.num_workers,
                )
            )
            prefill_worker_id = worker_ids[0]

            def stored_blocks_by_dp_rank(events_json: str) -> dict[int, int]:
                counts = {dp_rank: 0 for dp_rank in range(expected_prefill_dp_ranks)}
                for event in json.loads(events_json):
                    if event.get("worker_id") != prefill_worker_id:
                        continue
                    stored = event.get("event", {}).get("data", {}).get("stored")
                    if stored is None:
                        continue
                    dp_rank = event.get("event", {}).get("dp_rank", 0)
                    counts[dp_rank] = counts.get(dp_rank, 0) + len(
                        stored.get("blocks", [])
                    )
                return counts

            await asyncio.sleep(2.0)
            baseline_counts = stored_blocks_by_dp_rank(
                await observer_router.dump_events()
            )

            async with aiohttp.ClientSession() as session:
                for request_idx in range(expected_prefill_dp_ranks * 2):
                    prompt_tokens = " ".join(
                        f"prefill-{request_idx}-token-{token_idx}"
                        for token_idx in range(block_size * 3)
                    )
                    payload = {
                        **test_payload,
                        "stream": False,
                        "max_tokens": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt_tokens,
                            }
                        ],
                    }
                    async with session.post(chat_url, json=payload) as response:
                        assert response.status == 200, (
                            f"Request {request_idx + 1} failed with status "
                            f"{response.status}: {await response.text()}"
                        )
                        await response.text()
                    await asyncio.sleep(0.5)

            await asyncio.sleep(2.0)
            final_counts = stored_blocks_by_dp_rank(await observer_router.dump_events())
            return prefill_worker_id, baseline_counts, final_counts

        async def test_sync():
            with managed_runtime(
                store_backend=store_backend, request_plane=request_plane
            ) as runtime:
                return await run_test(runtime)

        prefill_worker_id, baseline_counts, final_counts = asyncio.run(test_sync())

        delta_counts = {
            dp_rank: final_counts.get(dp_rank, 0) - baseline_counts.get(dp_rank, 0)
            for dp_rank in range(expected_prefill_dp_ranks)
        }
        active_dp_ranks = sorted(
            dp_rank for dp_rank, block_count in delta_counts.items() if block_count > 0
        )

        assert active_dp_ranks == list(range(expected_prefill_dp_ranks)), (
            f"Expected round-robin prefill requests for worker {prefill_worker_id} "
            f"to store KV blocks on dp_ranks {list(range(expected_prefill_dp_ranks))}, "
            f"but saw deltas {delta_counts}"
        )


def _test_router_decisions(
    engine_workers,
    endpoint,
    model_name: str,
    request,
    test_dp_rank: bool = False,
    block_size: int = 8,
    use_kv_events: bool = True,
    router_event_threads: int = 4,
    standalone_indexer_url: Optional[str] = None,
    standalone_selector_url: Optional[str] = None,
    router_ais_config: Optional[dict[str, Any]] = None,
    router_predicted_ttl_secs: Optional[float] = None,
    router_approximate_cache_policy: str = "ttl",
    initial_wait: float = 0.25,
):
    """Validate cross-worker routing decisions based on longest prefix match.

    Assumes engine workers are already initialized.
    Seeds two routing targets (worker a and worker b) with different prefix trees,
    then verifies the router picks the correct worker for subsequent requests.

    Test sequence (7 blocks A-G, each block_size tokens, 5 requests):
    1. [A, B]       → force worker a        (seed worker a's tree)
    2. [A, C, D]    → force worker a        (branch under A on worker a)
    3. [A, C, E]    → force worker b        (seed worker b's tree)
    4. [A, C, D, F] → router picks          (worker a wins: prefix [A,C,D]=3 vs worker b [A,C]=2)
    5. [A, C, E, G] → router picks          (worker b wins: prefix [A,C,E]=3 vs worker a [A,C]=2)

    Args:
        engine_workers: Backend worker instance ({MockerProcess, VLLMProcess, TRTLLMProcess}) (already initialized with __enter__())
        endpoint: Endpoint of the engine workers
        model_name: Name of the model
        request: Pytest request fixture
        test_dp_rank: If True, also forces and validates dp_rank routing (for data parallel setups)
        block_size: KV cache block size. Defaults to 8.
        use_kv_events: If True (default), uses KV events from workers. If False, uses
            approximate routing with the configured retention policy (--no-kv-events mode).
        router_ais_config: Optional AIS router perf-model config for direct KvRouter tests.
        router_approximate_cache_policy: Retention policy for the local approximate indexer.

    Raises:
        AssertionError: If routing decisions don't match expected prefix logic
    """

    # Use async to manage the test flow
    async def test_sync():
        # If standalone indexer mode, launch workers one-by-one and register.
        # Must happen before KvRouter creation since KvRouter blocks until workers appear.
        if standalone_indexer_url:
            await engine_workers.launch_workers_with_indexer(endpoint)

        # Workers register one instance per process (not per dp_rank)
        expected_num_instances = engine_workers.num_workers

        kv_router_config = KvRouterConfig(
            use_kv_events=use_kv_events,
            router_event_threads=router_event_threads,
            router_track_prefill_tokens=True,
            router_prefill_load_model=(
                "ais" if router_ais_config is not None else "none"
            ),
            router_predicted_ttl_secs=router_predicted_ttl_secs,
            router_approximate_cache_policy=router_approximate_cache_policy,
        )
        ais_perf_config = (
            AisPerfConfig(config=router_ais_config)
            if router_ais_config is not None
            else None
        )

        kv_router = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint,
                block_size=block_size,
                kv_router_config=kv_router_config,
                ais_perf_config=ais_perf_config,
            ),
            num_workers=expected_num_instances,
            engine_workers=engine_workers,
        )

        # Wait for workers to be ready and get their instance IDs
        worker_ids = await wait_for_workers_ready(
            endpoint,
            kv_router,
            expected_num_workers=expected_num_instances,
            model_name=model_name,
        )
        logger.info(f"Workers ready: {worker_ids}")

        # Determine worker a / worker b routing targets
        if len(worker_ids) >= 2:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[1]
        elif len(worker_ids) == 1 and test_dp_rank:
            worker_a_id = worker_ids[0]
            worker_b_id = worker_ids[0]
        else:
            raise AssertionError(
                f"Need at least 2 routing targets but got {len(worker_ids)} worker(s) "
                f"with test_dp_rank={test_dp_rank}"
            )

        dp_rank_a = 0 if test_dp_rank else None
        dp_rank_b = 1 if test_dp_rank else None

        logger.info(
            f"Routing targets: worker_a=(id={worker_a_id}, dp_rank={dp_rank_a}), "
            f"worker_b=(id={worker_b_id}, dp_rank={dp_rank_b})"
        )

        # Generate 7 random blocks (A-G)
        num_blocks = 7
        blocks = [
            [random.randint(1, 10000) for _ in range(block_size)]
            for _ in range(num_blocks)
        ]
        A, B, C, D, E, F, G = blocks

        # 5 requests with specific prefix structure
        request_specs = [
            # (token_ids, forced_worker_id, forced_dp_rank, sleep_after)
            (A + B, worker_a_id, dp_rank_a, 0.1),  # req1: seed worker a
            (
                A + C + D,
                worker_a_id,
                dp_rank_a,
                0.1,
            ),  # req2: branch under A on worker a
            (A + C + E, worker_b_id, dp_rank_b, 2.0),  # req3: seed worker b
            (
                A + C + D + F,
                None,
                None,
                2.0,
            ),  # req4: router picks (worker a should win)
            (
                A + C + E + G,
                None,
                None,
                2.0,
            ),  # req5: router picks (worker b should win)
        ]
        dp_a = dp_rank_a if dp_rank_a is not None else 0
        dp_b = dp_rank_b if dp_rank_b is not None else 0
        expected_overlap_by_request = {
            4: {(worker_a_id, dp_a): 3, (worker_b_id, dp_b): 2},
            5: {(worker_a_id, dp_a): 2, (worker_b_id, dp_b): 3},
        }

        response_worker_ids: list[dict[str, Optional[int]]] = []

        for i, (token_ids, wid_override, dp_override, sleep_after) in enumerate(
            request_specs
        ):
            log_msg = f"Sending request {i + 1}/5 with {len(token_ids)} tokens"
            if wid_override is not None:
                log_msg += f" - FORCING worker_id={wid_override}"
                if dp_override is not None:
                    log_msg += f", dp_rank={dp_override}"
            logger.info(log_msg)

            request_idx = i + 1
            if request_idx in expected_overlap_by_request:
                await _assert_overlap_scores(
                    kv_router,
                    token_ids,
                    block_size,
                    expected_overlap_by_request[request_idx],
                    f"request {request_idx}",
                )

            result = await send_request_via_python_kv_router(
                kv_python_router=kv_router,
                model_name=model_name,
                token_ids=token_ids,
                # XPU workers have longer startup latency; pass as parameter
                # to avoid affecting CUDA tests.
                initial_wait=initial_wait,
                stop_conditions={
                    "ignore_eos": True,
                    "max_tokens": 2,
                },
                worker_id=wid_override,
                dp_rank=dp_override,
                return_worker_ids=True,
            )
            assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
            response_worker_ids.append(result)
            logger.info(
                f"Request {i + 1} response: prefill_worker_id={result.get('prefill_worker_id')}, "
                f"decode_worker_id={result.get('decode_worker_id')}, "
                f"prefill_dp_rank={result.get('prefill_dp_rank')}, "
                f"decode_dp_rank={result.get('decode_dp_rank')}"
            )

            if sleep_after > 0:
                await asyncio.sleep(sleep_after)

        events_json = await kv_router.dump_events()
        return (
            events_json,
            worker_a_id,
            worker_b_id,
            dp_rank_a,
            dp_rank_b,
            response_worker_ids,
            A + C + D + F,  # req4 tokens for standalone indexer /score verification
        )

    # Run the async test
    (
        events_json,
        worker_a_id,
        worker_b_id,
        dp_rank_a,
        dp_rank_b,
        response_worker_ids,
        req4_tokens,
    ) = asyncio.run(test_sync())

    # Verify request 4 routed to worker a (longest prefix match)
    req4 = response_worker_ids[3]
    assert req4["prefill_worker_id"] == worker_a_id, (
        f"Request 4: expected prefill_worker_id={worker_a_id} (longest prefix match), "
        f"got {req4['prefill_worker_id']}"
    )
    if test_dp_rank:
        assert (
            req4["prefill_dp_rank"] == dp_rank_a
        ), f"Request 4: expected prefill_dp_rank={dp_rank_a}, got {req4['prefill_dp_rank']}"

    # Verify request 5 routed to worker b (longest prefix match)
    req5 = response_worker_ids[4]
    assert req5["prefill_worker_id"] == worker_b_id, (
        f"Request 5: expected prefill_worker_id={worker_b_id} (longest prefix match), "
        f"got {req5['prefill_worker_id']}"
    )
    if test_dp_rank:
        assert (
            req5["prefill_dp_rank"] == dp_rank_b
        ), f"Request 5: expected prefill_dp_rank={dp_rank_b}, got {req5['prefill_dp_rank']}"

    logger.info(
        f"Response routing verified: req4 → worker_a (id={worker_a_id}, dp_rank={dp_rank_a}), "
        f"req5 → worker_b (id={worker_b_id}, dp_rank={dp_rank_b})"
    )

    # Parse events and verify event counts per routing target
    events = json.loads(events_json)

    # Always group by (worker_id, dp_rank)
    events_by_key: dict[tuple[int, int], list[Any]] = {}
    for event in events:
        worker_id = event.get("worker_id")
        dp_rank = event.get("event", {}).get("dp_rank", 0)
        key = (worker_id, dp_rank)
        if key not in events_by_key:
            events_by_key[key] = []
        events_by_key[key].append(event)

    def count_stored_blocks(events: list[Any]) -> int:
        total = 0
        for event in events:
            stored = event.get("event", {}).get("data", {}).get("stored")
            if stored is None:
                continue
            total += len(stored.get("blocks", []))
        return total

    logger.info(
        "Stored blocks by (worker_id, dp_rank): "
        f"{[(key, count_stored_blocks(evts)) for key, evts in events_by_key.items()]}"
    )

    # Worker a key: 5 stored blocks (A, B from req1; C, D from req2; F from req4)
    worker_a_key = (worker_a_id, dp_rank_a if dp_rank_a is not None else 0)
    worker_a_blocks = count_stored_blocks(events_by_key.get(worker_a_key, []))
    assert worker_a_blocks == 5, (
        f"Expected worker_a {worker_a_key} to have 5 stored blocks (A,B + C,D + F), "
        f"but found {worker_a_blocks}"
    )

    # Worker b key: 4 stored blocks (A, C, E from req3; G from req5)
    worker_b_key = (worker_b_id, dp_rank_b if dp_rank_b is not None else 0)
    worker_b_blocks = count_stored_blocks(events_by_key.get(worker_b_key, []))
    assert worker_b_blocks == 4, (
        f"Expected worker_b {worker_b_key} to have 4 stored blocks (A,C,E + G), "
        f"but found {worker_b_blocks}"
    )

    logger.info(
        f"Successfully verified cross-worker routing: "
        f"worker_a {worker_a_key} has {worker_a_blocks} stored blocks, "
        f"worker_b {worker_b_key} has {worker_b_blocks} stored blocks"
    )

    # Verify standalone indexer scores via HTTP POST /query
    if standalone_indexer_url:
        _dp_a = dp_rank_a if dp_rank_a is not None else 0
        _dp_b = dp_rank_b if dp_rank_b is not None else 0

        async def _verify_scores():
            # Wait for ZMQ events to propagate to the indexer
            await asyncio.sleep(3)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{standalone_indexer_url}/query",
                    json={"token_ids": req4_tokens, "model_name": model_name},
                ) as resp:
                    assert resp.status == 200, f"POST /query failed: {resp.status}"
                    scores = (await resp.json())["scores"]

                    id_a = str(worker_a_id)
                    id_b = str(worker_b_id)
                    dp_a = str(_dp_a)
                    dp_b = str(_dp_b)
                    score_a = scores[id_a][dp_a]
                    score_b = scores[id_b][dp_b]

                    logger.info(
                        f"Standalone indexer /query: {id_a}[{dp_a}]={score_a}, "
                        f"{id_b}[{dp_b}]={score_b}"
                    )
                    assert score_a > score_b, (
                        f"Expected instance {id_a} dp_rank {dp_a} score {score_a} > "
                        f"instance {id_b} dp_rank {dp_b} score {score_b} for req4 tokens"
                    )

        asyncio.run(_verify_scores())

    # Verify standalone selection service scores via HTTP POST /overlap_scores.
    if standalone_selector_url:
        _dp_a = dp_rank_a if dp_rank_a is not None else 0
        _dp_b = dp_rank_b if dp_rank_b is not None else 0

        async def _verify_selection_service_scores():
            # The sidecar readiness gate confirms registration; allow KV events to settle.
            await asyncio.sleep(3)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{standalone_selector_url}/overlap_scores",
                    json={"token_ids": req4_tokens, "model_name": model_name},
                ) as resp:
                    assert (
                        resp.status == 200
                    ), f"POST /overlap_scores failed: {resp.status} {await resp.text()}"
                    body = await resp.json()

                    scores = {
                        (worker["worker_id"], worker["dp_rank"]): worker
                        for worker in body["workers"]
                    }
                    score_a = scores[(worker_a_id, _dp_a)]["device_blocks"]
                    score_b = scores[(worker_b_id, _dp_b)]["device_blocks"]

                    logger.info(
                        "Standalone selection /overlap_scores: %s[%s]=%s, %s[%s]=%s",
                        worker_a_id,
                        _dp_a,
                        score_a,
                        worker_b_id,
                        _dp_b,
                        score_b,
                    )
                    assert score_a > score_b, (
                        f"Expected selection service worker {worker_a_id} dp_rank {_dp_a} "
                        f"score {score_a} > worker {worker_b_id} dp_rank {_dp_b} "
                        f"score {score_b} for req4 tokens"
                    )

        asyncio.run(_verify_selection_service_scores())


def _test_router_cache_salt_isolation(
    engine_workers,
    endpoint,
    model_name: str,
    block_size: int,
):
    """Verify cache-salted engine events remain isolated in the router index."""

    async def test_sync():
        expected_num_instances = engine_workers.num_workers
        kv_router = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint,
                block_size=block_size,
                kv_router_config=KvRouterConfig(
                    use_kv_events=True,
                    router_event_threads=4,
                ),
            ),
            num_workers=expected_num_instances,
            engine_workers=engine_workers,
        )

        worker_ids = await wait_for_workers_ready(
            endpoint,
            kv_router,
            expected_num_workers=expected_num_instances,
            model_name=model_name,
        )
        assert len(worker_ids) >= 2, "cache-salt isolation requires two workers"

        worker_a = (worker_ids[0], 0)
        worker_b = (worker_ids[1], 0)
        token_ids = list(range(1_000, 1_000 + block_size * 2))
        expected_blocks = len(token_ids) // block_size

        async def generate(cache_salt: str, worker_id: Optional[int] = None) -> int:
            """Run one request and return the worker it was routed to."""
            routing: dict[str, Any] = {"cache_salt": cache_salt}
            if worker_id is not None:
                routing["backend_instance_id"] = worker_id
            request = {
                "model": model_name,
                "token_ids": token_ids,
                "stop_conditions": {"ignore_eos": True, "max_tokens": 2},
                "sampling_options": {},
                "output_options": {},
                "eos_token_ids": [],
                "extra_args": {"nvext": {"cache_salt": cache_salt}},
                "routing": routing,
            }
            stream = await kv_router.generate_from_request(request)
            return await read_routed_worker_from_stream(stream, f"tenant {cache_salt}")

        async def device_blocks(
            cache_salt: Optional[str],
        ) -> dict[tuple[int, int], int]:
            scores = await kv_router.get_overlap_scores(
                token_ids,
                include_shared=False,
                cache_namespace=cache_salt,
            )
            assert scores["block_size"] == block_size
            assert scores["num_blocks"] == expected_blocks
            assert scores["shared_cache"]["enabled"] is False
            return {
                (row["worker_id"], row["dp_rank"]): row["device_blocks"]
                for row in scores["workers"]
            }

        async def wait_for_scores(
            cache_salt: Optional[str],
            expected: dict[tuple[int, int], int],
        ) -> None:
            deadline = time.monotonic() + 10
            last_scores: dict[tuple[int, int], int] = {}
            expected_nonzero = {key: value for key, value in expected.items() if value}
            while time.monotonic() < deadline:
                last_scores = await device_blocks(cache_salt)
                actual_nonzero = {
                    key: value for key, value in last_scores.items() if value
                }
                if actual_nonzero == expected_nonzero:
                    return
                await asyncio.sleep(0.25)
            raise AssertionError(
                f"cache_salt={cache_salt!r}: expected {expected}, got {last_scores}"
            )

        async def wait_for_worker(cache_salt: str, expected: tuple[int, int]) -> None:
            deadline = time.monotonic() + 10
            last_selection: tuple[int, int, int] | None = None
            while time.monotonic() < deadline:
                last_selection = await kv_router.best_worker(
                    token_ids,
                    cache_namespace=cache_salt,
                )
                if (
                    last_selection[:2] == expected
                    and last_selection[2] == expected_blocks
                ):
                    logger.info(
                        "cache_salt=%r selected worker=%s with %d overlap blocks",
                        cache_salt,
                        expected,
                        expected_blocks,
                    )
                    return
                await asyncio.sleep(0.25)
            raise AssertionError(
                f"cache_salt={cache_salt!r}: expected worker {expected} with "
                f"{expected_blocks} overlap blocks, got {last_selection}"
            )

        await generate("tenant-a", worker_a[0])
        await wait_for_scores("tenant-a", {worker_a: expected_blocks})
        await wait_for_scores("tenant-b", {})
        await wait_for_scores(None, {})

        await generate("tenant-b", worker_b[0])
        await wait_for_scores("tenant-b", {worker_b: expected_blocks})
        await wait_for_scores("tenant-a", {worker_a: expected_blocks})
        await wait_for_scores(None, {})
        await wait_for_worker("tenant-a", worker_a)
        await wait_for_worker("tenant-b", worker_b)

        # Unpinned repeats must reach their tenant's worker. Score checks alone
        # miss a misroute: seeding already met them, and its events may arrive late.
        assert await generate("tenant-a") == worker_a[0]
        assert await generate("tenant-b") == worker_b[0]
        await wait_for_scores("tenant-a", {worker_a: expected_blocks})
        await wait_for_scores("tenant-b", {worker_b: expected_blocks})

    asyncio.run(test_sync())


def _test_router_lora_isolation(
    engine_workers,
    endpoint,
    model_name: str,
    block_size: int,
    lora_name: str,
    lora_uri: str,
):
    """Verify LoRA adapter blocks are reused by adapter requests and stay
    isolated from base-model blocks in the router index."""
    from tests.serve.lora_utils import load_lora_adapter

    async def test_sync():
        expected_num_instances = engine_workers.num_workers
        kv_router = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint,
                block_size=block_size,
                kv_router_config=KvRouterConfig(
                    use_kv_events=True,
                    router_event_threads=4,
                ),
            ),
            num_workers=expected_num_instances,
            engine_workers=engine_workers,
        )

        worker_ids = await wait_for_workers_ready(
            endpoint,
            kv_router,
            expected_num_workers=expected_num_instances,
            model_name=model_name,
        )
        assert len(worker_ids) >= 2, "LoRA isolation requires two workers"

        for system_port in engine_workers.system_ports:
            await asyncio.to_thread(load_lora_adapter, system_port, lora_name, lora_uri)

        worker_a = (worker_ids[0], 0)
        worker_b = (worker_ids[1], 0)
        token_ids = list(range(1_000, 1_000 + block_size * 2))
        expected_blocks = len(token_ids) // block_size

        async def generate(lora: Optional[str], worker_id: Optional[int] = None) -> int:
            """Run one request and return the worker it was routed to."""
            routing: dict[str, Any] = {}
            if lora is not None:
                routing["lora_name"] = lora
            if worker_id is not None:
                routing["backend_instance_id"] = worker_id
            request = {
                "model": lora or model_name,
                "token_ids": token_ids,
                "stop_conditions": {"ignore_eos": True, "max_tokens": 2},
                "sampling_options": {},
                "output_options": {},
                "eos_token_ids": [],
                "routing": routing,
            }
            stream = await kv_router.generate_from_request(request)
            return await read_routed_worker_from_stream(stream, f"lora={lora!r}")

        async def device_blocks(lora: Optional[str]) -> dict[tuple[int, int], int]:
            scores = await kv_router.get_overlap_scores(
                token_ids,
                include_shared=False,
                lora_name=lora,
            )
            assert scores["block_size"] == block_size
            assert scores["num_blocks"] == expected_blocks
            return {
                (row["worker_id"], row["dp_rank"]): row["device_blocks"]
                for row in scores["workers"]
            }

        async def wait_for_scores(
            lora: Optional[str],
            expected: dict[tuple[int, int], int],
        ) -> None:
            deadline = time.monotonic() + 10
            last_scores: dict[tuple[int, int], int] = {}
            while time.monotonic() < deadline:
                last_scores = await device_blocks(lora)
                if {k: v for k, v in last_scores.items() if v} == expected:
                    return
                await asyncio.sleep(0.25)
            raise AssertionError(
                f"lora={lora!r}: expected {expected}, got {last_scores}"
            )

        async def wait_for_worker(lora: Optional[str], expected: tuple[int, int]):
            deadline = time.monotonic() + 10
            last_selection: tuple[int, int, int] | None = None
            while time.monotonic() < deadline:
                last_selection = await kv_router.best_worker(token_ids, lora_name=lora)
                if (
                    last_selection[:2] == expected
                    and last_selection[2] == expected_blocks
                ):
                    return
                await asyncio.sleep(0.25)
            raise AssertionError(
                f"lora={lora!r}: expected worker {expected} with "
                f"{expected_blocks} overlap blocks, got {last_selection}"
            )

        # Adapter blocks are indexed under the adapter, not the base model.
        await generate(lora_name, worker_a[0])
        await wait_for_scores(lora_name, {worker_a: expected_blocks})
        await wait_for_scores(None, {})

        # The same prompt on the base model is cached separately.
        await generate(None, worker_b[0])
        await wait_for_scores(None, {worker_b: expected_blocks})
        await wait_for_scores(lora_name, {worker_a: expected_blocks})
        await wait_for_worker(lora_name, worker_a)
        await wait_for_worker(None, worker_b)

        # Unpinned repeats must go back to the worker that holds their blocks,
        # and must not leave a copy on the other worker.
        assert await generate(lora_name) == worker_a[0]
        assert await generate(None) == worker_b[0]
        await wait_for_scores(lora_name, {worker_a: expected_blocks})
        await wait_for_scores(None, {worker_b: expected_blocks})

    asyncio.run(test_sync())


def _test_busy_threshold_endpoint(
    engine_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Test that the /busy_threshold endpoint can be hit and responds correctly.

    TODO: This doesn't actually test any e2e rejection for now. A proper test would:
    1. Set a very low threshold
    2. Send enough requests to exceed the threshold
    3. Verify that subsequent requests are rejected with 529

    For now, this test only verifies the endpoint is accessible and returns valid responses.

    Args:
        engine_workers: MockerProcess instance (already initialized with __enter__())
        block_size: Block size for KV cache
        request: Pytest request fixture for managing resources
        frontend_port: Port for the frontend HTTP server
        test_payload: Base test payload (used to extract model name)
        store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
        request_plane: Request plane to use ("nats" or "tcp"). Defaults to "nats".

    Raises:
        AssertionError: If endpoint responses are incorrect
    """
    # Initial thresholds - we need to start with these so the monitor is created
    initial_active_decode_blocks_threshold = 0.9
    initial_active_prefill_tokens_threshold = 1000  # Literal token count threshold

    with reserved_ports(1, DynamoPortRange.FRONTEND) as (admin_port,), KVRouterProcess(
        request,
        block_size,
        frontend_port,
        engine_workers.namespace,
        store_backend,
        blocks_threshold=initial_active_decode_blocks_threshold,
        tokens_threshold=initial_active_prefill_tokens_threshold,
        request_plane=request_plane,
        extra_env={
            "DYN_ENABLE_FRONTEND_ADMIN_API": "1",
            "DYN_FRONTEND_ADMIN_PORT": str(admin_port),
        },
    ):
        # Start KV router frontend with initial thresholds to create monitor
        logger.info(f"Starting KV router frontend on port {frontend_port}")

        frontend_url = f"http://localhost:{frontend_port}"
        busy_threshold_url = f"http://127.0.0.1:{admin_port}/busy_threshold"

        # Wait for workers to register with frontend
        logger.info("Waiting for workers to register with frontend...")
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=engine_workers.num_workers,
                timeout=120,
                engine_workers=engine_workers,
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        model_name = test_payload.get("model", "test-model")

        async def test_busy_threshold_api():
            async with aiohttp.ClientSession() as session:
                # Test 1: GET /busy_threshold - list all thresholds
                logger.info("Testing GET /busy_threshold (list all)")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    assert (
                        "thresholds" in data
                    ), f"Expected 'thresholds' key in response: {data}"
                    logger.info(f"GET /busy_threshold response: {data}")

                # Test 2: POST /busy_threshold with model only (get thresholds)
                logger.info(
                    f"Testing POST /busy_threshold to get thresholds for model '{model_name}'"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (get) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == initial_active_decode_blocks_threshold
                    ), f"Expected initial active_decode_blocks_threshold={initial_active_decode_blocks_threshold}: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == initial_active_prefill_tokens_threshold
                    ), f"Expected initial active_prefill_tokens_threshold={initial_active_prefill_tokens_threshold}: {data}"
                    logger.info(
                        f"POST /busy_threshold (get) response: status={response.status}, data={data}"
                    )

                # Test 3: POST /busy_threshold to set active_decode_blocks_threshold only
                test_active_decode_blocks_threshold = 0.75
                logger.info(
                    f"Testing POST /busy_threshold to set active_decode_blocks_threshold={test_active_decode_blocks_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_decode_blocks_threshold": test_active_decode_blocks_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set blocks) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("model") == model_name
                    ), f"Expected model={model_name}: {data}"
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == test_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={test_active_decode_blocks_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set blocks) response: {data}")

                # Test 4: POST /busy_threshold to set active_prefill_tokens_threshold only
                test_active_prefill_tokens_threshold = (
                    2000  # Literal token count threshold
                )
                logger.info(
                    f"Testing POST /busy_threshold to set active_prefill_tokens_threshold={test_active_prefill_tokens_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_prefill_tokens_threshold": test_active_prefill_tokens_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set tokens) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == test_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={test_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set tokens) response: {data}")

                # Test 5: POST /busy_threshold to set both thresholds
                new_active_decode_blocks_threshold = 0.5
                new_active_prefill_tokens_threshold = (
                    1200  # Literal token count threshold
                )
                logger.info(
                    f"Testing POST /busy_threshold to set both thresholds: "
                    f"active_decode_blocks={new_active_decode_blocks_threshold}, active_prefill_tokens={new_active_prefill_tokens_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_decode_blocks_threshold": new_active_decode_blocks_threshold,
                        "active_prefill_tokens_threshold": new_active_prefill_tokens_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set both) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_decode_blocks_threshold")
                        == new_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={new_active_decode_blocks_threshold}: {data}"
                    assert (
                        data.get("active_prefill_tokens_threshold")
                        == new_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={new_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set both) response: {data}")

                # Test 6: GET /busy_threshold - verify thresholds appear in list
                logger.info("Testing GET /busy_threshold to verify thresholds in list")
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    thresholds = data.get("thresholds", [])
                    model_entry = next(
                        (t for t in thresholds if t["model"] == model_name), None
                    )
                    assert (
                        model_entry is not None
                    ), f"Expected model '{model_name}' in thresholds: {data}"
                    assert (
                        model_entry.get("active_decode_blocks_threshold")
                        == new_active_decode_blocks_threshold
                    ), f"Expected active_decode_blocks_threshold={new_active_decode_blocks_threshold}: {data}"
                    assert (
                        model_entry.get("active_prefill_tokens_threshold")
                        == new_active_prefill_tokens_threshold
                    ), f"Expected active_prefill_tokens_threshold={new_active_prefill_tokens_threshold}: {data}"
                    logger.info(f"GET /busy_threshold (after set) response: {data}")

                # Test 7: Invalid active_decode_blocks_threshold value (should fail validation)
                logger.info(
                    "Testing POST /busy_threshold with invalid active_decode_blocks_threshold (>1.0)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_decode_blocks_threshold": 1.5},
                ) as response:
                    assert (
                        response.status == 400
                    ), f"Expected 400 for invalid active_decode_blocks_threshold, got {response.status}"
                    data = await response.json()
                    logger.info(
                        f"POST /busy_threshold (invalid blocks) response: {data}"
                    )

                # Test 8: active_prefill_tokens_threshold accepts large values (should be valid)
                logger.info(
                    "Testing POST /busy_threshold with large active_prefill_tokens_threshold (valid)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_prefill_tokens_threshold": 5000},
                ) as response:
                    assert (
                        response.status == 200
                    ), f"Expected 200 for large active_prefill_tokens_threshold, got {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold") == 5000
                    ), f"Expected active_prefill_tokens_threshold=5000: {data}"
                    logger.info(
                        f"POST /busy_threshold (large tokens threshold) response: {data}"
                    )

                # Test 9: Invalid active_prefill_tokens_threshold value (should fail validation for < 0)
                # Note: Returns 422 because -1.0 can't be deserialized into u64 (type validation)
                # vs Test 7 which returns 400 because 1.5 is a valid f64 but fails range validation
                logger.info(
                    "Testing POST /busy_threshold with invalid active_prefill_tokens_threshold (< 0)"
                )
                async with session.post(
                    busy_threshold_url,
                    json={"model": model_name, "active_prefill_tokens_threshold": -1.0},
                ) as response:
                    assert (
                        response.status == 422
                    ), f"Expected 422 for negative active_prefill_tokens_threshold, got {response.status}"
                    data = await response.json()
                    logger.info(
                        f"POST /busy_threshold (invalid tokens) response: {data}"
                    )

                # Test 10: Set active_prefill_tokens_threshold_frac (fraction of max_num_batched_tokens)
                test_frac_threshold = 0.8
                logger.info(
                    f"Testing POST /busy_threshold to set active_prefill_tokens_threshold_frac={test_frac_threshold}"
                )
                async with session.post(
                    busy_threshold_url,
                    json={
                        "model": model_name,
                        "active_prefill_tokens_threshold_frac": test_frac_threshold,
                    },
                ) as response:
                    assert (
                        response.status == 200
                    ), f"POST /busy_threshold (set frac) failed with status {response.status}"
                    data = await response.json()
                    assert (
                        data.get("active_prefill_tokens_threshold_frac")
                        == test_frac_threshold
                    ), f"Expected active_prefill_tokens_threshold_frac={test_frac_threshold}: {data}"
                    logger.info(f"POST /busy_threshold (set frac) response: {data}")

                # Test 11: Verify frac threshold appears in GET /busy_threshold list
                logger.info(
                    "Testing GET /busy_threshold to verify frac threshold in list"
                )
                async with session.get(busy_threshold_url) as response:
                    assert (
                        response.status == 200
                    ), f"GET /busy_threshold failed with status {response.status}"
                    data = await response.json()
                    thresholds = data.get("thresholds", [])
                    model_entry = next(
                        (t for t in thresholds if t["model"] == model_name), None
                    )
                    assert (
                        model_entry is not None
                    ), f"Expected model '{model_name}' in thresholds: {data}"
                    assert (
                        model_entry.get("active_prefill_tokens_threshold_frac")
                        == test_frac_threshold
                    ), f"Expected active_prefill_tokens_threshold_frac={test_frac_threshold}: {data}"
                    logger.info(
                        f"GET /busy_threshold (after set frac) response: {data}"
                    )

                logger.info("All busy_threshold endpoint tests passed!")

        asyncio.run(test_busy_threshold_api())


def _test_disagg_direct_mode(
    prefill_workers,
    decode_workers,
    request,
    frontend_port: int,
    test_payload: dict,
    request_plane: str = "nats",
):
    """E2E test for disaggregated Direct routing mode (simulating GAIE EPP).

    In Direct mode, the router does not select workers itself.
    Worker IDs must be provided via x-dynamo-worker-instance-id and
    x-dynamo-prefill-instance-id
    HTTP headers. The test verifies:
      1. Requests with explicit worker ID headers succeed and return a valid response.
      2. Requests without headers fail (Direct mode rejects unaddressed requests).

    Args:
        prefill_workers: Prefill mocker workers (already started).
        decode_workers: Decode mocker workers (already started).
        request: Pytest request fixture.
        frontend_port: Port for the Direct-mode frontend HTTP server.
        test_payload: Base test payload for /v1/chat/completions.
        request_plane: Transport for request plane ("nats" or "tcp").
    """
    with FrontendRouterProcess(
        request,
        BLOCK_SIZE,
        frontend_port,
        decode_workers.namespace,
        request_plane=request_plane,
        router_mode="direct",
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"
        runtime = get_runtime(request_plane=request_plane)
        prefill_endpoint = runtime.endpoint(
            f"{decode_workers.namespace}.prefill.generate"
        )
        decode_endpoint = runtime.endpoint(
            f"{decode_workers.namespace}.backend.generate"
        )

        async def wait_for_direct_frontend():
            prefill_ids = await poll_for_worker_instances(
                prefill_endpoint,
                prefill_workers.num_workers,
            )
            decode_ids = await poll_for_worker_instances(
                decode_endpoint,
                decode_workers.num_workers,
            )
            headers = {
                "x-dynamo-worker-instance-id": str(decode_ids[0]),
                "x-dynamo-prefill-instance-id": str(prefill_ids[0]),
                "x-dynamo-dp-rank": "0",
                "x-dynamo-prefill-dp-rank": "0",
            }
            await wait_for_frontend_ready(
                frontend_url=frontend_url,
                timeout=120,
                test_payload={**test_payload, "stream": False},
                request_headers=headers,
            )
            return prefill_ids, decode_ids

        prefill_ids, decode_ids = asyncio.run(wait_for_direct_frontend())
        logger.info(f"Discovered prefill workers: {prefill_ids}")
        logger.info(f"Discovered decode workers: {decode_ids}")

        target_prefill = prefill_ids[0]
        target_decode = decode_ids[0]

        async def run_direct_mode_tests():
            # Test 1: Request WITH correct headers should succeed.
            # In direct mode the router is a passthrough — it does not have a
            # KvRouter and does not record worker IDs on the RequestTracker, so
            # the response's nvext will not contain worker_id info.  We only
            # verify that the request is routed successfully (HTTP 200) and
            # produces a valid chat completion response.
            payload = {
                **test_payload,
                "stream": False,
            }
            headers = {
                "x-dynamo-worker-instance-id": str(target_decode),
                "x-dynamo-prefill-instance-id": str(target_prefill),
                "x-dynamo-dp-rank": "0",
                "x-dynamo-prefill-dp-rank": "0",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    chat_url, json=payload, headers=headers
                ) as response:
                    assert response.status == 200, (
                        "Direct-mode request with headers failed: "
                        f"status={response.status}, body={await response.text()}"
                    )
                    data = await response.json()
                    assert "choices" in data, "Expected 'choices' in response data"
                    assert (
                        len(data["choices"]) > 0
                    ), "Expected at least one choice in response"

                # Test 2: Request WITHOUT headers should fail (Direct mode
                # rejects requests that have no worker ID)
                logger.info(
                    "Sending request without headers (should fail in Direct mode)..."
                )
                no_header_payload = {**test_payload, "stream": False}
                async with session.post(chat_url, json=no_header_payload) as response:
                    assert response.status != 200, (
                        f"Expected non-200 status without routing headers in Direct mode, "
                        f"got {response.status}. Direct mode must reject unaddressed requests."
                    )
                    logger.info(
                        f"Correctly rejected headerless request: status={response.status}"
                    )

        asyncio.run(run_direct_mode_tests())
        logger.info("Direct-mode disagg E2E test passed")


def _test_disagg_per_role_router_modes(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Validate that prefill and decode tiers can run different router modes.

    The prefill mockers advertise ``RouterConfig(RouterMode.KV)`` in their model
    deployment cards while the decode mockers advertise nothing, so decode
    inherits the frontend's ``--router-mode round-robin``. This asserts the two
    hops are genuinely routed on different terms:

    1. Progressive prefix-extending requests converge on ONE prefill worker,
       which only KV routing produces — round-robin would rotate.
    2. Those same requests spread across MORE THAN ONE decode worker, which only
       round-robin produces — KV routing would converge there too.

    Assertion 2 is the one that fails if the per-role override is ignored and
    the whole pipeline silently runs in a single mode.
    """
    num_requests = 4

    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        decode_workers.namespace,
        store_backend,
        request_plane=request_plane,
        router_mode="round-robin",
        min_initial_workers=decode_workers.num_workers,
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        logger.info(
            "Waiting for prefill and decode workers to register with the "
            "round-robin frontend..."
        )
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=(
                    prefill_workers.num_workers + decode_workers.num_workers
                ),
                timeout=120,
                engine_workers=[prefill_workers, decode_workers],
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        async def send_progressive_requests():
            prefill_worker_ids: list[int] = []
            decode_worker_ids: list[int] = []
            base_content = test_payload["messages"][0]["content"]

            async with aiohttp.ClientSession() as session:
                for i in range(num_requests):
                    payload = {
                        **test_payload,
                        "messages": [
                            {
                                "role": "user",
                                "content": " ".join([base_content] * (i + 1)),
                            }
                        ],
                        "nvext": {"extra_fields": ["worker_id"]},
                        "stream": True,
                    }

                    async with session.post(chat_url, json=payload) as response:
                        assert (
                            response.status == 200
                        ), f"Request {i + 1} failed with status {response.status}"

                        prefill_wid = None
                        decode_wid = None
                        body = await response.text()
                        for data in parse_sse_json_chunks(body):
                            worker_id_info = data.get("nvext", {}).get("worker_id", {})
                            if "prefill_worker_id" in worker_id_info:
                                prefill_wid = worker_id_info["prefill_worker_id"]
                            if "decode_worker_id" in worker_id_info:
                                decode_wid = worker_id_info["decode_worker_id"]

                        logger.info(
                            f"Request {i + 1}: prefill_worker_id={prefill_wid}, "
                            f"decode_worker_id={decode_wid}"
                        )
                        if prefill_wid is not None:
                            prefill_worker_ids.append(prefill_wid)
                        if decode_wid is not None:
                            decode_worker_ids.append(decode_wid)

                    await asyncio.sleep(1)

            return prefill_worker_ids, decode_worker_ids

        prefill_ids, decode_ids = asyncio.run(send_progressive_requests())

        logger.info(f"Collected prefill_worker_ids: {prefill_ids}")
        logger.info(f"Collected decode_worker_ids: {decode_ids}")

        assert len(prefill_ids) == num_requests, (
            f"Expected {num_requests} prefill_worker_ids, got {len(prefill_ids)}. "
            f"A prefill hop must have run for every request."
        )
        assert (
            len(decode_ids) == num_requests
        ), f"Expected {num_requests} decode_worker_ids, got {len(decode_ids)}."

        # The prefill tier advertised KV, so prefix reuse must concentrate it.
        # As in the all-KV disagg test, the TCP request plane can show a
        # transient on the first request before the initial "stored" KV events
        # are ingested, so only requests 2..N are required to converge there.
        converged = prefill_ids[1:] if request_plane == "tcp" else prefill_ids
        assert len(set(converged)) == 1, (
            f"Prefill advertised RouterMode.KV, so prefix-extending requests must "
            f"converge on one prefill worker; got {set(converged)}. "
            f"Full list: {prefill_ids}"
        )

        # The decode tier inherited round-robin, so it must NOT converge. If the
        # prefill card's override had leaked into the decode router (or the
        # decode set had been dragged into KV mode), these would collapse to one.
        assert len(set(decode_ids)) > 1, (
            f"Decode inherited round-robin, so requests must spread across "
            f"workers; all {num_requests} landed on {set(decode_ids)}. "
            f"This means the per-role router config did not take effect."
        )

        assert prefill_ids[0] not in set(decode_ids), (
            f"Prefill worker {prefill_ids[0]} should not appear in the decode "
            f"worker set {set(decode_ids)}."
        )

        logger.info(
            "Verified per-role router modes: prefill converged on "
            f"{set(converged)} (KV) while decode spread across "
            f"{set(decode_ids)} (round-robin)"
        )


def _test_disagg_per_role_session_affinity(
    prefill_workers,
    decode_workers,
    block_size: int,
    request,
    frontend_port: int,
    test_payload: dict,
    store_backend: str = "etcd",
    request_plane: str = "nats",
):
    """Validate that session affinity is configured per hop, not per deployment.

    The prefill workers advertise a session-affinity TTL; the decode workers
    advertise the same router mode without one. Both hops then run round-robin,
    so the only thing that can pin a session is its own affinity setting:

    1. Repeated requests carrying one session id stay on ONE prefill worker,
       because that hop has affinity.
    2. The same requests spread across MULTIPLE decode workers, because that hop
       does not -- round-robin rotates.

    Assertion 2 is the one that fails if the two hops share a TTL: decode would
    pin alongside prefill.

    This covers configurability, not expiry. Proving a TTL elapses would mean
    sleeping past it and asserting a rebind, which is timing-dependent and can
    re-select the same worker.
    """
    num_requests = 4

    with FrontendRouterProcess(
        request,
        block_size,
        frontend_port,
        decode_workers.namespace,
        store_backend,
        request_plane=request_plane,
        router_mode="round-robin",
        min_initial_workers=decode_workers.num_workers,
    ):
        frontend_url = f"http://localhost:{frontend_port}"
        chat_url = f"{frontend_url}/v1/chat/completions"

        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=(
                    prefill_workers.num_workers + decode_workers.num_workers
                ),
                timeout=120,
                engine_workers=[prefill_workers, decode_workers],
                store_backend=store_backend,
                request_plane=request_plane,
            )
        )

        session_headers = {"x-dynamo-session-id": f"per-role-ttl-{uuid.uuid4()}"}
        content = test_payload["messages"][0]["content"]

        async def send_session_requests():
            prefill_ids: list[int] = []
            decode_ids: list[int] = []
            async with aiohttp.ClientSession() as session:
                for i in range(num_requests):
                    payload = {
                        **test_payload,
                        "messages": [{"role": "user", "content": content}],
                        "nvext": {"extra_fields": ["worker_id"]},
                        "stream": True,
                        "max_tokens": 1,
                    }
                    async with session.post(
                        chat_url, json=payload, headers=session_headers
                    ) as response:
                        assert (
                            response.status == 200
                        ), f"Request {i + 1} failed with status {response.status}"
                        # worker_id repeats across chunks of one stream; record
                        # it once per request so the counts match the requests.
                        prefill_wid = None
                        decode_wid = None
                        body = await response.text()
                        for data in parse_sse_json_chunks(body):
                            worker_id_info = data.get("nvext", {}).get("worker_id", {})
                            if "prefill_worker_id" in worker_id_info:
                                prefill_wid = worker_id_info["prefill_worker_id"]
                            if "decode_worker_id" in worker_id_info:
                                decode_wid = worker_id_info["decode_worker_id"]
                        if prefill_wid is not None:
                            prefill_ids.append(prefill_wid)
                        if decode_wid is not None:
                            decode_ids.append(decode_wid)
                    await asyncio.sleep(0.5)
            return prefill_ids, decode_ids

        prefill_ids, decode_ids = asyncio.run(send_session_requests())
        logger.info(f"Session-pinned prefill_worker_ids: {prefill_ids}")
        logger.info(f"Session-pinned decode_worker_ids: {decode_ids}")

        assert (
            len(prefill_ids) == num_requests
        ), f"Expected {num_requests} prefill_worker_ids, got {len(prefill_ids)}."
        assert (
            len(decode_ids) == num_requests
        ), f"Expected {num_requests} decode_worker_ids, got {len(decode_ids)}."

        assert len(set(prefill_ids)) == 1, (
            f"Prefill advertised a session-affinity TTL, so one session must stay "
            f"pinned to one prefill worker; got {set(prefill_ids)}. Full: {prefill_ids}"
        )
        assert len(set(decode_ids)) > 1, (
            f"Decode advertised no session-affinity TTL, so the same session must "
            f"not be pinned there; all {num_requests} requests landed on "
            f"{set(decode_ids)}. That means the hops are sharing one TTL."
        )

        logger.info(
            "Verified per-hop session affinity: prefill pinned to "
            f"{set(prefill_ids)} (TTL advertised) while decode spread across "
            f"{set(decode_ids)} (no TTL)"
        )


########################################################
# Frontend / Runtime EPP token parity
########################################################


@dataclass(frozen=True)
class TokenParityCase:
    """One actual Frontend-vs-EPP request pair plus a meaning-changing sibling."""

    name: str
    path: str
    payload: dict[str, Any]
    divergent_payload: dict[str, Any]


def frontend_epp_token_parity_cases(model_name: str) -> list[TokenParityCase]:
    """Reusable actual-token scenarios for native Frontend HTTP vs Runtime EPP.

    Each case carries a real request body and a sibling whose system or content
    meaning changes so the same comparator can reject a live mismatch.
    """
    chat_path = "/v1/chat/completions"
    completion_path = "/v1/completions"
    return [
        TokenParityCase(
            name="plain_chat",
            path=chat_path,
            payload={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Name the first three prime numbers."}
                ],
                "stream": True,
                "max_tokens": 1,
            },
            divergent_payload={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Translate good morning into French."}
                ],
                "stream": True,
                "max_tokens": 1,
            },
        ),
        TokenParityCase(
            name="text_completion",
            path=completion_path,
            payload={
                "model": model_name,
                "prompt": "The first three prime numbers are",
                "stream": True,
                "max_tokens": 1,
            },
            divergent_payload={
                "model": model_name,
                "prompt": "The capital of France is",
                "stream": True,
                "max_tokens": 1,
            },
        ),
        TokenParityCase(
            name="system_chat",
            path=chat_path,
            payload={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Answer with only the numbers."},
                    {"role": "user", "content": "Name the first three prime numbers."},
                ],
                "stream": True,
                "max_tokens": 1,
            },
            divergent_payload={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Answer only in Spanish words."},
                    {"role": "user", "content": "Name the first three prime numbers."},
                ],
                "stream": True,
                "max_tokens": 1,
            },
        ),
        TokenParityCase(
            name="multiturn_chat",
            path=chat_path,
            payload={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "What is two plus two?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "Add three more."},
                ],
                "stream": True,
                "max_tokens": 1,
            },
            divergent_payload={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "What is two plus two?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "Now subtract nine."},
                ],
                "stream": True,
                "max_tokens": 1,
            },
        ),
    ]


def assert_ordered_token_ids_match(
    left: list[int],
    right: list[int],
    *,
    left_label: str,
    right_label: str,
    case_name: str,
) -> None:
    """Compare exact ordered token IDs and report the first differing position."""
    if left == right:
        if not left:
            raise AssertionError(f"{case_name}: both token sequences are empty")
        return
    limit = min(len(left), len(right))
    index = next((i for i in range(limit) if left[i] != right[i]), limit)
    left_token = left[index] if index < len(left) else "<missing>"
    right_token = right[index] if index < len(right) else "<missing>"
    raise AssertionError(
        f"{case_name}: token sequence mismatch at index {index}: "
        f"{left_label}={left_token} {right_label}={right_token} "
        f"lengths {len(left)} vs {len(right)}"
    )


def run_frontend_epp_token_parity(
    stack,
    cases: list[TokenParityCase],
    *,
    include_negative_control: bool = True,
) -> None:
    """Compare live Frontend query_instance_id tokens to live EPP mutations.

    Frontend uses query-only annotation and does not generate. EPP still picks
    and books; the driver completes the ext_proc stream before the next case.
    The optional negative control changes one arm's system or content meaning
    after unchanged-request parity succeeds and requires the same comparator
    to reject the live mismatch.
    """
    if not cases:
        raise AssertionError("token parity requires at least one actual request case")

    for case in cases:
        frontend_tokens = stack.collect_frontend_tokens(
            case.payload,
            path=case.path,
            request_id=unique_request_id(f"fe-{case.name}"),
        )
        epp_tokens = stack.collect_epp_tokens(
            case.payload,
            path=case.path,
            request_id=unique_request_id(f"epp-{case.name}"),
        )
        assert_ordered_token_ids_match(
            frontend_tokens,
            epp_tokens,
            left_label="frontend.nvext.token_ids",
            right_label="epp.nvext.token_data",
            case_name=case.name,
        )
        logger.info(
            "Token parity matched for %s (%s tokens)",
            case.name,
            len(frontend_tokens),
        )

    kube_api = stack.kube_api
    if kube_api is not None and kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(kube_api.failures)
        )

    if not include_negative_control:
        return

    control = cases[0]
    frontend_tokens = stack.collect_frontend_tokens(
        control.payload,
        path=control.path,
        request_id=unique_request_id(f"fe-neg-{control.name}"),
    )
    epp_tokens = stack.collect_epp_tokens(
        control.divergent_payload,
        path=control.path,
        request_id=unique_request_id(f"epp-neg-{control.name}"),
    )
    try:
        assert_ordered_token_ids_match(
            frontend_tokens,
            epp_tokens,
            left_label="frontend.nvext.token_ids",
            right_label="epp.nvext.token_data",
            case_name=f"{control.name}-negative",
        )
    except AssertionError as exc:
        if "mismatch at index" in str(exc):
            logger.info("Negative control rejected live mismatched token sequences")
            if kube_api is not None and kube_api.failures:
                raise AssertionError(
                    "Loopback Kubernetes API recorded unexpected requests: "
                    + "; ".join(kube_api.failures)
                )
            return
        raise
    raise AssertionError(
        f"{control.name}-negative: same comparator accepted diverged live requests"
    )


def run_keyed_tracking_runtime_config(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
    profile_name: str,
) -> None:
    """Consume keyed-xxh3-v1 and require query tokens plus real FE/EPP routes."""
    import warnings

    from tests.router.epp_config_contract import (
        KEYED_TRACKING_PROFILE,
        TRACKING_HASH_ALGORITHM,
        TRACKING_KEY_ID,
        TRACKING_KEY_PUBLIC_BYTES,
        TRACKING_KEY_SIZE,
        TRACKING_RUNTIME_FIELDS,
        _keyed_tracking_consumer_env,
        _plain_chat_token_case,
        _write_tracking_key_file,
        assert_frontend_expected_fields,
        clear_known_manifest_router_envs,
        compare_effective_fields,
        frontend_native_projection,
        native_default_projection,
        observe_epp_init_event,
    )
    from tests.router.epp_parity import collect_frontend_generation_route

    if profile_name != KEYED_TRACKING_PROFILE:
        raise AssertionError(
            f"keyed tracking profile {profile_name!r} != {KEYED_TRACKING_PROFILE!r}"
        )
    clear_known_manifest_router_envs(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("RUST_LOG", "info")
    monkeypatch.setenv("DYN_LOG", "info")
    key_path = tmp_path / "parity-public-tracking-key.bin"
    provenance = _write_tracking_key_file(key_path, TRACKING_KEY_PUBLIC_BYTES)
    if provenance["length"] != TRACKING_KEY_SIZE:
        raise AssertionError(
            f"public tracking key length {provenance['length']} != {TRACKING_KEY_SIZE}"
        )
    consumer_env = _keyed_tracking_consumer_env(key_path)
    for name, value in consumer_env.items():
        monkeypatch.setenv(name, value)
    expected = {
        "router_tracking_hash": TRACKING_HASH_ALGORITHM,
        "router_tracking_key_id": TRACKING_KEY_ID,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frontend = frontend_native_projection(TRACKING_RUNTIME_FIELDS)
    defaults = native_default_projection(TRACKING_RUNTIME_FIELDS)
    assert_frontend_expected_fields(
        profile=profile_name,
        frontend=frontend,
        expected=expected,
        defaults=defaults,
    )
    cases = _plain_chat_token_case(model_name)
    with stack_cls(
        request,
        tmp_path,
        model_name,
        consumer_env=consumer_env,
        log_label=profile_name,
    ) as stack:
        epp = observe_epp_init_event(
            stack.epp_process,
            node_name=request.node.name,
            expected=expected,
        )
        compare_effective_fields(
            profile=profile_name,
            frontend=frontend,
            epp=epp,
            expected=expected,
            defaults=defaults,
        )
        run_frontend_epp_token_parity(stack, cases, include_negative_control=False)
        case = cases[0]
        generation = collect_frontend_generation_route(
            stack.frontend_url,
            case.payload,
            path=case.path,
            request_id=unique_request_id("fe-gen-keyed-plain_chat"),
        )
        if stack.driver is None:
            raise AssertionError("owned ext_proc driver was not saved")
        epp_route = stack.driver.collect_route_observation(
            case.payload,
            path=case.path,
            request_id=unique_request_id("epp-route-keyed-plain_chat"),
        )
        if not epp_route.token_ids:
            raise AssertionError("EPP route token_ids are empty")
        if (generation.worker_id, generation.dp_rank) != (
            epp_route.worker_id,
            epp_route.dp_rank,
        ):
            raise AssertionError(
                "frontend generation and EPP route worker/rank differ: "
                f"fe={generation.worker_id}/{generation.dp_rank} "
                f"epp={epp_route.worker_id}/{epp_route.dp_rank}"
            )
        if generation.dp_rank != 0:
            raise AssertionError(
                f"controlled one-worker rank is {generation.dp_rank}, expected 0"
            )
        frontend_proc = getattr(stack, "frontend_process", None)
        epp_proc = getattr(stack, "epp_process", None)
        if frontend_proc is None or frontend_proc.proc is None:
            raise AssertionError("owned frontend_process was not saved")
        if epp_proc is None or epp_proc.proc is None:
            raise AssertionError("owned epp_process was not saved")
        kube_api = stack.kube_api
        if kube_api is not None and kube_api.failures:
            raise AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(kube_api.failures)
            )
        logger.info(
            "keyed tracking runtime algorithm=%s key_id=%s key_path=%s "
            "key_length=%s key_sha256=%s frontend pid=%s epp pid=%s "
            "fe_worker=%s/%s epp_worker=%s/%s",
            TRACKING_HASH_ALGORITHM,
            TRACKING_KEY_ID,
            provenance["path"],
            provenance["length"],
            provenance["sha256"],
            frontend_proc.proc.pid,
            epp_proc.proc.pid,
            generation.worker_id,
            generation.dp_rank,
            epp_route.worker_id,
            epp_route.dp_rank,
        )


def run_keyed_tracking_key_file_size(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
    arm: str,
) -> None:
    """Reject a 31-byte public tracking key on one owned consumer arm."""
    from tests.router.epp_config_contract import (
        KEYED_TRACKING_SIZE_ARMS,
        TRACKING_KEY_INVALID_BYTES,
        TRACKING_KEY_SIZE_DEADLINE_S,
        TRACKING_KEY_SIZE_DIAGNOSTIC,
        _keyed_tracking_consumer_env,
        _write_tracking_key_file,
        clear_known_manifest_router_envs,
    )
    from tests.router.router_process import observe_owned_initialization_diagnostic

    if arm not in KEYED_TRACKING_SIZE_ARMS:
        raise AssertionError(f"keyed tracking size arm {arm!r} is unknown")
    if TRACKING_KEY_SIZE_DIAGNOSTIC != (
        "router tracking key file must contain exactly 32 raw bytes; found 31"
    ):
        raise AssertionError(
            "keyed tracking size diagnostic is not the source-derived 31-byte message"
        )
    clear_known_manifest_router_envs(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("RUST_LOG", "info")
    monkeypatch.setenv("DYN_LOG", "info")
    key_path = tmp_path / f"parity-public-tracking-key-31-{arm}.bin"
    provenance = _write_tracking_key_file(key_path, TRACKING_KEY_INVALID_BYTES)
    if provenance["length"] != 31:
        raise AssertionError(
            f"invalid tracking key length {provenance['length']} != 31"
        )
    consumer_env = _keyed_tracking_consumer_env(key_path)
    include_frontend = arm == "frontend"
    include_epp = arm == "epp"
    with stack_cls(
        request,
        tmp_path,
        model_name,
        consumer_env=consumer_env,
        include_frontend=include_frontend,
        include_epp=include_epp,
        require_readiness=False,
        log_label=f"key-size-{arm}",
    ) as stack:
        process = (
            getattr(stack, "frontend_process", None)
            if include_frontend
            else getattr(stack, "epp_process", None)
        )
        pid, log = observe_owned_initialization_diagnostic(
            process,
            diagnostic=TRACKING_KEY_SIZE_DIAGNOSTIC,
            deadline_s=TRACKING_KEY_SIZE_DEADLINE_S,
            node_name=request.node.name,
            label=f"{arm}_process",
        )
    logger.info(
        "keyed tracking size diagnostic=%s arm=%s key_path=%s key_length=%s "
        "key_sha256=%s pid=%s log=%s",
        TRACKING_KEY_SIZE_DIAGNOSTIC,
        arm,
        provenance["path"],
        provenance["length"],
        provenance["sha256"],
        pid,
        log,
    )


def run_approximate_cache_policy_metrics(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
    policy_name: str,
) -> None:
    """Read owned FE/EPP configured/effective gauges for LRU or default TTL."""
    import warnings

    from tests.router.epp_config_contract import (
        APPROXIMATE_POLICY_NAMES,
        APPROXIMATE_POLICY_ROW,
        _approximate_consumer_env,
        _assert_core_policy_gauges,
        _frontend_full_kv_json,
        _plain_chat_token_case,
        clear_known_manifest_router_envs,
    )

    if policy_name not in APPROXIMATE_POLICY_NAMES:
        raise AssertionError(f"unknown approximate policy profile {policy_name!r}")
    clear_known_manifest_router_envs(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("RUST_LOG", "info")
    monkeypatch.setenv("DYN_LOG", "info")
    consumer_env = _approximate_consumer_env(policy_name)
    for name, value in consumer_env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        payload = _frontend_full_kv_json()
    if APPROXIMATE_POLICY_ROW in payload:
        raise AssertionError(
            "native to_json now serializes router_approximate_cache_policy; "
            "omission is not a default or support oracle"
        )
    with stack_cls(
        request,
        tmp_path,
        model_name,
        consumer_env=consumer_env,
        core_system_metrics=True,
        log_label=f"approx-{policy_name}",
    ) as stack:
        run_frontend_epp_token_parity(
            stack, _plain_chat_token_case(model_name), include_negative_control=False
        )
        if stack.mockers is None:
            raise AssertionError("owned mocker endpoint was not saved")
        namespace = stack.mockers.namespace
        component = stack.mockers.component_name
        if not namespace or not component:
            raise AssertionError(
                "owned mocker endpoint identity is missing "
                f"namespace={namespace!r} component={component!r}"
            )
        for role in ("frontend", "epp"):
            pid, port, content = stack.core_system_metrics_text(role=role)
            _assert_core_policy_gauges(
                content,
                role=role,
                pid=pid,
                port=port,
                namespace=namespace,
                component=component,
                selected=policy_name,
            )
            logger.info(
                "approximate policy %s %s pid=%s port=%s namespace=%s component=%s",
                policy_name,
                role,
                pid,
                port,
                namespace,
                component,
            )


def run_active_decode_threshold_consumer(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
) -> None:
    """Observe owned FE/EPP remote-load decode thresholds for default and 0.25."""
    from tests.router.epp_config_contract import (
        DECODE_THRESHOLD_CONFIGURED,
        DECODE_THRESHOLD_ENV,
        DECODE_THRESHOLD_PROFILES,
        DECODE_THRESHOLD_ROW,
        ActiveDecodeThresholdConfigMismatch,
        _plain_chat_token_case,
        clear_known_manifest_router_envs,
        decode_threshold_consumer_env,
        mark_owned_log_cursor,
        observe_owned_decode_load_threshold,
    )
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        collect_frontend_generation_route,
    )

    observed: dict[str, tuple[float | None, float | None]] = {}
    for profile_name in DECODE_THRESHOLD_PROFILES:
        clear_known_manifest_router_envs(monkeypatch)
        monkeypatch.setenv("NO_COLOR", "1")
        consumer_env = decode_threshold_consumer_env(profile_name)
        for name, value in consumer_env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        cases = _plain_chat_token_case(model_name)
        log_label = f"decode-{profile_name}"
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            log_label=log_label,
        ) as stack:
            frontend_proc = stack.frontend_process
            epp_proc = stack.epp_process
            node_name = request.node.name
            frontend_pid, frontend_log, frontend_offset = mark_owned_log_cursor(
                frontend_proc,
                label="frontend_process",
                node_name=node_name,
                log_label=log_label,
            )
            epp_pid, epp_log, epp_offset = mark_owned_log_cursor(
                epp_proc,
                label="epp_process",
                node_name=node_name,
                log_label=log_label,
            )
            run_frontend_epp_token_parity(stack, cases, include_negative_control=False)
            case = cases[0]
            generation = collect_frontend_generation_route(
                stack.frontend_url,
                case.payload,
                path=case.path,
                request_id=unique_request_id("fe-gen-decode-plain_chat"),
            )
            if stack.driver is None:
                raise AssertionError("owned ext_proc driver was not saved")
            epp_route = stack.driver.collect_route_observation(
                case.payload,
                path=case.path,
                request_id=unique_request_id("epp-route-decode-plain_chat"),
            )
            if not epp_route.token_ids:
                raise AssertionError("EPP route token_ids are empty")
            if (generation.worker_id, generation.dp_rank) != (
                epp_route.worker_id,
                epp_route.dp_rank,
            ):
                raise AssertionError(
                    "frontend generation and EPP route worker/rank differ: "
                    f"fe={generation.worker_id}/{generation.dp_rank} "
                    f"epp={epp_route.worker_id}/{epp_route.dp_rank}"
                )
            if generation.dp_rank != 0:
                raise AssertionError(
                    f"controlled one-worker rank is {generation.dp_rank}, expected 0"
                )
            if frontend_proc is None or frontend_proc.proc is None:
                raise AssertionError("owned frontend_process was not saved")
            if epp_proc is None or epp_proc.proc is None:
                raise AssertionError("owned epp_process was not saved")
            if (
                frontend_proc.proc.poll() is not None
                or epp_proc.proc.poll() is not None
            ):
                raise AssertionError("owned frontend or EPP process is not live")
            kube_api = stack.kube_api
            if kube_api is not None and kube_api.failures:
                raise AssertionError(
                    "Loopback Kubernetes API recorded unexpected requests: "
                    + "; ".join(kube_api.failures)
                )
            deadline = time.monotonic() + READINESS_DEADLINE_S
            frontend_value = observe_owned_decode_load_threshold(
                frontend_proc,
                label="frontend_process",
                node_name=node_name,
                pid=frontend_pid,
                log=frontend_log,
                offset=frontend_offset,
                worker_id=generation.worker_id,
                dp_rank=generation.dp_rank,
                deadline=deadline,
            )
            epp_value = observe_owned_decode_load_threshold(
                epp_proc,
                label="epp_process",
                node_name=node_name,
                pid=epp_pid,
                log=epp_log,
                offset=epp_offset,
                worker_id=generation.worker_id,
                dp_rank=generation.dp_rank,
                deadline=deadline,
            )
            if profile_name == "default":
                if frontend_value is not None or epp_value is not None:
                    raise AssertionError(
                        "default decode-threshold control is not None: "
                        f"frontend={frontend_value!r} epp={epp_value!r}"
                    )
            else:
                wanted = float(DECODE_THRESHOLD_CONFIGURED)
                if frontend_value != wanted:
                    raise AssertionError(
                        "configured frontend "
                        f"{DECODE_THRESHOLD_ROW}={frontend_value!r} != "
                        f"Some({DECODE_THRESHOLD_CONFIGURED})"
                    )
            if stack.mockers is None:
                raise AssertionError("owned mocker endpoint was not saved")
            namespace = stack.mockers.namespace
            logger.info(
                "decode threshold profile=%s env=%s frontend pid=%s log=%s "
                "value=%s epp pid=%s log=%s value=%s namespace=%s "
                "worker=%s/%s",
                profile_name,
                consumer_env.get(DECODE_THRESHOLD_ENV),
                frontend_pid,
                frontend_log,
                frontend_value,
                epp_pid,
                epp_log,
                epp_value,
                namespace,
                generation.worker_id,
                generation.dp_rank,
            )
            observed[profile_name] = (frontend_value, epp_value)
    if set(observed) != set(DECODE_THRESHOLD_PROFILES):
        raise AssertionError(
            f"decode-threshold evidence {sorted(observed)} != "
            f"{list(DECODE_THRESHOLD_PROFILES)}"
        )
    frontend_value, epp_value = observed["configured"]
    if observed == {
        "default": (None, None),
        "configured": (0.25, None),
    }:
        raise ActiveDecodeThresholdConfigMismatch(
            "configured decode-threshold mismatch after default and route "
            f"controls: frontend={frontend_value!r} epp={epp_value!r}"
        )
    if frontend_value != epp_value:
        raise AssertionError(
            "configured decode-threshold mismatch after default and route "
            f"controls: frontend={frontend_value!r} epp={epp_value!r}"
        )


def run_shared_cache_type_consumer(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
) -> None:
    """Observe omitted none and explicit hicache shared-cache construction."""
    import warnings

    from tests.router.epp_config_contract import (
        SHARED_CACHE_TYPE_FIELDS,
        SHARED_CACHE_TYPE_HICACHE,
        SHARED_CACHE_TYPE_NONE,
        SHARED_CACHE_TYPE_PROFILES,
        SHARED_CACHE_TYPE_ROW,
        _plain_chat_token_case,
        assert_frontend_expected_fields,
        clear_known_manifest_router_envs,
        compare_effective_fields,
        frontend_native_projection,
        native_default_projection,
        observe_epp_init_event,
        observe_owned_hicache_constructor,
        owned_process_log_identity,
        shared_cache_type_consumer_env,
    )
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        collect_frontend_generation_route,
    )

    for profile_name in SHARED_CACHE_TYPE_PROFILES:
        clear_known_manifest_router_envs(monkeypatch)
        monkeypatch.setenv("NO_COLOR", "1")
        consumer_env = shared_cache_type_consumer_env(profile_name)
        for name, value in consumer_env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        expected = {SHARED_CACHE_TYPE_ROW: profile_name}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            frontend = frontend_native_projection(SHARED_CACHE_TYPE_FIELDS)
        defaults = native_default_projection(SHARED_CACHE_TYPE_FIELDS)
        if profile_name == SHARED_CACHE_TYPE_NONE:
            if frontend.get(SHARED_CACHE_TYPE_ROW) != SHARED_CACHE_TYPE_NONE:
                raise AssertionError(
                    f"{profile_name} FE {SHARED_CACHE_TYPE_ROW}="
                    f"{frontend.get(SHARED_CACHE_TYPE_ROW)!r} != "
                    f"{SHARED_CACHE_TYPE_NONE!r}"
                )
            if defaults.get(SHARED_CACHE_TYPE_ROW) != SHARED_CACHE_TYPE_NONE:
                raise AssertionError(
                    f"{profile_name} native default {SHARED_CACHE_TYPE_ROW}="
                    f"{defaults.get(SHARED_CACHE_TYPE_ROW)!r} != "
                    f"{SHARED_CACHE_TYPE_NONE!r}"
                )
        else:
            assert_frontend_expected_fields(
                profile=profile_name,
                frontend=frontend,
                expected=expected,
                defaults=defaults,
            )
        cases = _plain_chat_token_case(model_name)
        log_label = f"shared-cache-{profile_name}"
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            log_label=log_label,
        ) as stack:
            node_name = request.node.name
            epp = observe_epp_init_event(
                stack.epp_process,
                node_name=node_name,
                expected=expected,
            )
            if profile_name == SHARED_CACHE_TYPE_NONE:
                if epp.get(SHARED_CACHE_TYPE_ROW) != SHARED_CACHE_TYPE_NONE:
                    raise AssertionError(
                        f"{profile_name} EPP {SHARED_CACHE_TYPE_ROW}="
                        f"{epp.get(SHARED_CACHE_TYPE_ROW)!r} != "
                        f"{SHARED_CACHE_TYPE_NONE!r}"
                    )
                if frontend[SHARED_CACHE_TYPE_ROW] != epp[SHARED_CACHE_TYPE_ROW]:
                    raise AssertionError(
                        f"{profile_name} {SHARED_CACHE_TYPE_ROW} mismatch "
                        f"frontend={frontend[SHARED_CACHE_TYPE_ROW]!r} "
                        f"epp={epp[SHARED_CACHE_TYPE_ROW]!r}"
                    )
            else:
                compare_effective_fields(
                    profile=profile_name,
                    frontend=frontend,
                    epp=epp,
                    expected=expected,
                    defaults=defaults,
                )
            frontend_proc = stack.frontend_process
            epp_proc = stack.epp_process
            if stack.mockers is None:
                raise AssertionError("owned mocker endpoint was not saved")
            component = stack.mockers.component_name
            if not component:
                raise AssertionError(
                    f"owned mocker component is missing: {component!r}"
                )
            if profile_name == SHARED_CACHE_TYPE_HICACHE:
                frontend_pid, frontend_log = owned_process_log_identity(
                    frontend_proc,
                    label="frontend_process",
                    node_name=node_name,
                    log_label=log_label,
                )
                epp_pid, epp_log = owned_process_log_identity(
                    epp_proc,
                    label="epp_process",
                    node_name=node_name,
                    log_label=log_label,
                )
                deadline = time.monotonic() + READINESS_DEADLINE_S
                observe_owned_hicache_constructor(
                    frontend_proc,
                    label="frontend_process",
                    node_name=node_name,
                    log_label=log_label,
                    pid=frontend_pid,
                    log=frontend_log,
                    component=component,
                    deadline=deadline,
                )
                observe_owned_hicache_constructor(
                    epp_proc,
                    label="epp_process",
                    node_name=node_name,
                    log_label=log_label,
                    pid=epp_pid,
                    log=epp_log,
                    component=component,
                    deadline=deadline,
                )
            run_frontend_epp_token_parity(stack, cases, include_negative_control=False)
            case = cases[0]
            generation = collect_frontend_generation_route(
                stack.frontend_url,
                case.payload,
                path=case.path,
                request_id=unique_request_id("fe-gen-shared-cache-plain_chat"),
            )
            if stack.driver is None:
                raise AssertionError("owned ext_proc driver was not saved")
            epp_route = stack.driver.collect_route_observation(
                case.payload,
                path=case.path,
                request_id=unique_request_id("epp-route-shared-cache-plain_chat"),
            )
            if not epp_route.token_ids:
                raise AssertionError("EPP route token_ids are empty")
            if (generation.worker_id, generation.dp_rank) != (
                epp_route.worker_id,
                epp_route.dp_rank,
            ):
                raise AssertionError(
                    "frontend generation and EPP route worker/rank differ: "
                    f"fe={generation.worker_id}/{generation.dp_rank} "
                    f"epp={epp_route.worker_id}/{epp_route.dp_rank}"
                )
            if generation.dp_rank != 0:
                raise AssertionError(
                    f"controlled one-worker rank is {generation.dp_rank}, expected 0"
                )
            if frontend_proc is None or frontend_proc.proc is None:
                raise AssertionError("owned frontend_process was not saved")
            if epp_proc is None or epp_proc.proc is None:
                raise AssertionError("owned epp_process was not saved")
            if (
                frontend_proc.proc.poll() is not None
                or epp_proc.proc.poll() is not None
            ):
                raise AssertionError("owned frontend or EPP process is not live")
            kube_api = stack.kube_api
            if kube_api is not None and kube_api.failures:
                raise AssertionError(
                    "Loopback Kubernetes API recorded unexpected requests: "
                    + "; ".join(kube_api.failures)
                )
            logger.info(
                "shared-cache-type profile=%s frontend=%s epp=%s "
                "component=%s worker=%s/%s",
                profile_name,
                frontend[SHARED_CACHE_TYPE_ROW],
                epp[SHARED_CACHE_TYPE_ROW],
                component,
                generation.worker_id,
                generation.dp_rank,
            )


def run_session_prefix_index_config_parity(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
) -> None:
    """Owned FE/EPP session-prefix input/config validation parity.

    Controls cover omitted/false success, enabled+kv-events success, YAML
    validation on each arm, and env-true/kv-false terminal outcomes. Raises
    SessionPrefixIndexConfigMismatch only for FE exact diagnostic rejection
    with EPP acceptance and routes.
    """
    # epp_parity pulls gRPC tooling; keep selected-test-only like sibling runners.
    from tests.router.epp_config_contract import (
        ANSI_RE,
        SESSION_PREFIX_INDEX_DEADLINE_S,
        SESSION_PREFIX_INDEX_DIAGNOSTIC,
        SESSION_PREFIX_INDEX_ENV,
        SESSION_PREFIX_INDEX_FIELDS,
        SESSION_PREFIX_INDEX_ROW,
        SessionPrefixIndexConfigMismatch,
        _plain_chat_token_case,
        assert_frontend_session_prefix_index_false,
        clear_known_manifest_router_envs,
        frontend_native_projection,
        session_prefix_index_consumer_env,
    )
    from tests.router.epp_parity import (
        EPP_GRPC_PORT,
        catalog_generate_instances,
        collect_frontend_generation_route,
    )
    from tests.router.router_process import (
        epp_loopback_tls_ready,
        observe_owned_initialization_diagnostic,
    )
    from tests.utils.router_logs import completed_utf8_log_prefix

    if SESSION_PREFIX_INDEX_DIAGNOSTIC != (
        "enable_session_prefix_index requires use_kv_events=true"
    ):
        raise AssertionError(
            "session-prefix diagnostic is not the source-derived validation message"
        )

    def _apply_consumer_env(consumer_env: Mapping[str, str | None]) -> None:
        clear_known_manifest_router_envs(monkeypatch)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("RUST_LOG", "info")
        monkeypatch.setenv("DYN_LOG", "info")
        for name, value in consumer_env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    def _unique_generate_worker(stack) -> tuple[int, Mapping[str, Any]]:
        if stack.kube_api is None:
            raise AssertionError("owned kube_api was not saved")
        if stack.mockers is None:
            raise AssertionError("owned mocker endpoint was not saved")
        catalog = catalog_generate_instances(
            stack.kube_api.listed_metadata(),
            namespace=stack.mockers.namespace,
            component="mocker",
        )
        if len(catalog) != 1:
            raise AssertionError(
                f"expected one generate worker for namespace="
                f"{stack.mockers.namespace!r}, got {sorted(catalog)}"
            )
        worker_id, record = next(iter(catalog.items()))
        owner_uid = record.get("owner_uid")
        if not isinstance(owner_uid, str) or not owner_uid:
            raise AssertionError(
                f"generate worker {worker_id} missing owner_uid: {record!r}"
            )
        expected_uid = stack.identity.pod_uid
        if owner_uid != expected_uid:
            logger.error(
                "generate worker owner_uid mismatch actual=%s expected=%s " "worker=%s",
                owner_uid,
                expected_uid,
                worker_id,
            )
            raise AssertionError(
                f"generate worker {worker_id} owner_uid {owner_uid!r} != "
                f"stack identity pod_uid {expected_uid!r}"
            )
        return worker_id, record

    def _assert_routes_match_catalog(
        *,
        worker_id: int,
        generation,
        epp_route,
    ) -> None:
        expected = (worker_id, 0)
        if (generation.worker_id, generation.dp_rank) != expected:
            raise AssertionError(
                "frontend generation worker/rank "
                f"{generation.worker_id}/{generation.dp_rank} != catalog {expected}"
            )
        if (epp_route.worker_id, epp_route.dp_rank) != expected:
            raise AssertionError(
                "EPP route worker/rank "
                f"{epp_route.worker_id}/{epp_route.dp_rank} != catalog {expected}"
            )

    def _success_plain_chat_routes(
        *,
        consumer_env: Mapping[str, str | None],
        log_label: str,
    ) -> dict[str, Any]:
        cases = _plain_chat_token_case(model_name)
        case = cases[0]
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            log_label=log_label,
        ) as stack:
            run_frontend_epp_token_parity(stack, cases, include_negative_control=False)
            fe_tokens = stack.collect_frontend_tokens(
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"fe-ref-{log_label}-plain_chat"),
            )
            generation = collect_frontend_generation_route(
                stack.frontend_url,
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"fe-gen-{log_label}-plain_chat"),
            )
            if stack.driver is None:
                raise AssertionError("owned ext_proc driver was not saved")
            epp_route = stack.driver.collect_route_observation(
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"epp-route-{log_label}-plain_chat"),
            )
            worker_id, record = _unique_generate_worker(stack)
            _assert_routes_match_catalog(
                worker_id=worker_id,
                generation=generation,
                epp_route=epp_route,
            )
            assert_ordered_token_ids_match(
                fe_tokens,
                epp_route.token_ids,
                left_label="frontend.nvext.token_ids",
                right_label="epp.route.token_ids",
                case_name=f"{log_label}-plain_chat",
            )
            frontend_proc = stack.frontend_process
            epp_proc = stack.epp_process
            if (
                frontend_proc is None
                or frontend_proc.proc is None
                or epp_proc is None
                or epp_proc.proc is None
            ):
                raise AssertionError("owned frontend or EPP process was not saved")
            if (
                frontend_proc.proc.poll() is not None
                or epp_proc.proc.poll() is not None
            ):
                raise AssertionError("owned frontend or EPP process is not live")
            if stack.kube_api.failures:
                raise AssertionError(
                    "Loopback Kubernetes API recorded unexpected requests: "
                    + "; ".join(stack.kube_api.failures)
                )
            evidence = {
                "log_label": log_label,
                "frontend_pid": frontend_proc.proc.pid,
                "epp_pid": epp_proc.proc.pid,
                "worker_id": worker_id,
                "owner_uid": record["owner_uid"],
                "dp_rank": 0,
                "fe_tokens": list(fe_tokens),
            }
            logger.info(
                "session-prefix success label=%s session=%s kv_events=%s "
                "fe_pid=%s epp_pid=%s worker=%s owner=%s tokens=%s",
                log_label,
                consumer_env.get(SESSION_PREFIX_INDEX_ENV),
                consumer_env.get("DYN_ROUTER_USE_KV_EVENTS"),
                evidence["frontend_pid"],
                evidence["epp_pid"],
                worker_id,
                record["owner_uid"],
                len(fe_tokens),
            )
            return evidence

    def _observe_arm_diagnostic(
        *,
        arm: str,
        consumer_env: Mapping[str, str | None],
        log_label: str,
    ) -> tuple[int, Path]:
        include_frontend = arm == "frontend"
        include_epp = arm == "epp"
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            include_frontend=include_frontend,
            include_epp=include_epp,
            require_readiness=False,
            log_label=log_label,
        ) as stack:
            process = stack.frontend_process if include_frontend else stack.epp_process
            pid, log = observe_owned_initialization_diagnostic(
                process,
                diagnostic=SESSION_PREFIX_INDEX_DIAGNOSTIC,
                deadline_s=SESSION_PREFIX_INDEX_DEADLINE_S,
                node_name=request.node.name,
                label=f"{arm}_process",
            )
        logger.info(
            "session-prefix diagnostic arm=%s pid=%s log=%s policy=%s",
            arm,
            pid,
            log,
            consumer_env.get("DYN_ROUTER_POLICY_CONFIG"),
        )
        return pid, log

    def _remaining_budget(deadline: float, *, label: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"{label} exhausted the owned {SESSION_PREFIX_INDEX_DEADLINE_S}s "
                "deadline before completion"
            )
        return remaining

    def _classify_epp_env_conflict_outcome(
        *,
        consumer_env: Mapping[str, str | None],
        log_label: str,
        fe_token_reference: list[int],
    ) -> tuple[str, dict[str, Any]]:
        boundary = re.compile(
            re.escape(SESSION_PREFIX_INDEX_DIAGNOSTIC) + r"(?![0-9A-Za-z.])"
        )
        case = _plain_chat_token_case(model_name)[0]
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            include_frontend=False,
            include_epp=True,
            require_readiness=False,
            log_label=log_label,
        ) as stack:
            process = stack.epp_process
            if process is None or process.proc is None:
                raise AssertionError("owned epp_process was not saved")
            log_path = process.log_path
            if not log_path:
                raise AssertionError("epp_process has no log_path")
            log = Path(log_path)
            node_name = request.node.name
            if node_name not in log.as_posix():
                raise AssertionError(
                    f"epp_process log_path {log} is not this node {node_name}"
                )
            deadline = time.monotonic() + SESSION_PREFIX_INDEX_DEADLINE_S
            while True:
                _remaining_budget(deadline, label="epp conflict observation")
                pid = process.proc.pid
                process.proc.poll()
                raw = log.read_bytes() if log.is_file() else b""
                text = completed_utf8_log_prefix(
                    raw, label=f"epp_process pid={pid} log={log}"
                )
                if boundary.search(ANSI_RE.sub("", text)):
                    _remaining_budget(deadline, label="epp conflict reject")
                    evidence = {
                        "outcome": "reject",
                        "pid": pid,
                        "log": str(log),
                        "diagnostic": SESSION_PREFIX_INDEX_DIAGNOSTIC,
                    }
                    logger.info(
                        "session-prefix env-conflict epp reject pid=%s log=%s",
                        pid,
                        log,
                    )
                    return "reject", evidence
                if epp_loopback_tls_ready(EPP_GRPC_PORT, timeout=0.1):
                    if stack.driver is None:
                        raise AssertionError("owned ext_proc driver was not saved")
                    epp_tokens = stack.driver.collect_token_data(
                        case.payload,
                        path=case.path,
                        request_id=unique_request_id(
                            "epp-token-session-prefix-env-conflict"
                        ),
                        timeout_s=_remaining_budget(
                            deadline, label="epp collect_token_data"
                        ),
                    )
                    epp_route = stack.driver.collect_route_observation(
                        case.payload,
                        path=case.path,
                        request_id=unique_request_id(
                            "epp-route-session-prefix-env-conflict"
                        ),
                        timeout_s=_remaining_budget(
                            deadline, label="epp collect_route_observation"
                        ),
                    )
                    _remaining_budget(deadline, label="epp acceptance completion")
                    worker_id, record = _unique_generate_worker(stack)
                    if (epp_route.worker_id, epp_route.dp_rank) != (worker_id, 0):
                        raise AssertionError(
                            "EPP acceptance worker/rank "
                            f"{epp_route.worker_id}/{epp_route.dp_rank} != "
                            f"catalog {(worker_id, 0)}"
                        )
                    assert_ordered_token_ids_match(
                        fe_token_reference,
                        epp_tokens,
                        left_label="frontend.query_token_ids",
                        right_label="epp.collect_token_data",
                        case_name="session-prefix-env-conflict",
                    )
                    assert_ordered_token_ids_match(
                        fe_token_reference,
                        epp_route.token_ids,
                        left_label="frontend.query_token_ids",
                        right_label="epp.route.token_ids",
                        case_name="session-prefix-env-conflict-route",
                    )
                    if process.proc.poll() is not None:
                        raise AssertionError(
                            "owned epp_process exited after TLS readiness"
                        )
                    if stack.kube_api.failures:
                        raise AssertionError(
                            "Loopback Kubernetes API recorded unexpected requests: "
                            + "; ".join(stack.kube_api.failures)
                        )
                    evidence = {
                        "outcome": "accept",
                        "pid": pid,
                        "log": str(log),
                        "worker_id": worker_id,
                        "owner_uid": record["owner_uid"],
                        "dp_rank": 0,
                        "token_count": len(epp_tokens),
                    }
                    logger.info(
                        "session-prefix env-conflict epp accept pid=%s log=%s "
                        "worker=%s owner=%s tokens=%s",
                        pid,
                        log,
                        worker_id,
                        record["owner_uid"],
                        len(epp_tokens),
                    )
                    return "accept", evidence
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        f"epp_process pid={pid} log={log} neither emitted "
                        f"{SESSION_PREFIX_INDEX_DIAGNOSTIC!r} nor became TLS-ready "
                        "with tokens/routes before the owned deadline; this is not "
                        "an expected readiness timeout or setup failure classification"
                    )
                time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))

    fe_token_reference: list[int] | None = None
    for log_label, session_prefix, use_kv_events, expect_false in (
        ("session-prefix-omitted", None, False, True),
        ("session-prefix-false", "false", False, True),
        ("session-prefix-true-kv-true", "true", True, False),
    ):
        consumer_env = session_prefix_index_consumer_env(
            session_prefix=session_prefix,
            use_kv_events=use_kv_events,
            policy_config=None,
        )
        _apply_consumer_env(consumer_env)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if expect_false:
                assert_frontend_session_prefix_index_false(profile=log_label)
            else:
                frontend_true = frontend_native_projection(SESSION_PREFIX_INDEX_FIELDS)
                if frontend_true.get(SESSION_PREFIX_INDEX_ROW) is not True:
                    raise AssertionError(
                        f"{log_label} FE {SESSION_PREFIX_INDEX_ROW}="
                        f"{frontend_true.get(SESSION_PREFIX_INDEX_ROW)!r} != True"
                    )
        evidence = _success_plain_chat_routes(
            consumer_env=consumer_env,
            log_label=log_label,
        )
        if fe_token_reference is None:
            fe_token_reference = evidence["fe_tokens"]

    if fe_token_reference is None:
        raise AssertionError("no FE plain-chat token reference was captured")

    policy_path = tmp_path / "session-prefix-index-invalid.yaml"
    policy_path.write_text(
        "router:\n" "  enable_session_prefix_index: true\n" "  use_kv_events: false\n",
        encoding="utf-8",
    )
    yaml_env = session_prefix_index_consumer_env(
        session_prefix=None,
        use_kv_events=True,
        policy_config=str(policy_path),
    )
    _apply_consumer_env(yaml_env)
    for arm in ("frontend", "epp"):
        _observe_arm_diagnostic(
            arm=arm,
            consumer_env=yaml_env,
            log_label=f"session-prefix-yaml-{arm}",
        )

    conflict_env = session_prefix_index_consumer_env(
        session_prefix="true",
        use_kv_events=False,
        policy_config=None,
    )
    _apply_consumer_env(conflict_env)
    fe_pid, fe_log = _observe_arm_diagnostic(
        arm="frontend",
        consumer_env=conflict_env,
        log_label="session-prefix-env-true-kv-false-fe",
    )
    epp_outcome, epp_evidence = _classify_epp_env_conflict_outcome(
        consumer_env=conflict_env,
        log_label="session-prefix-env-true-kv-false-epp",
        fe_token_reference=fe_token_reference,
    )
    fe_evidence = {
        "outcome": "reject",
        "pid": fe_pid,
        "log": str(fe_log),
        "diagnostic": SESSION_PREFIX_INDEX_DIAGNOSTIC,
    }
    logger.info(
        "session-prefix env-conflict terminal fe=%s epp=%s",
        fe_evidence,
        epp_evidence,
    )
    if epp_outcome == "reject":
        return
    if epp_outcome == "accept":
        raise SessionPrefixIndexConfigMismatch(
            "FE rejected "
            f"{SESSION_PREFIX_INDEX_DIAGNOSTIC!r} while EPP accepted and routed: "
            f"fe={fe_evidence!r} epp={epp_evidence!r}"
        )
    raise AssertionError(
        f"unrecognized EPP session-prefix outcome {epp_outcome!r}: {epp_evidence!r}"
    )


def run_prefill_admission_input_boundaries(
    request,
    tmp_path,
    monkeypatch,
    model_name: str,
    stack_cls,
    profile_name: str,
) -> None:
    """Observe FE original busy-threshold handles for default/abs/frac profiles.

    Nondefault EPP route success is a fallback control only. This does not claim
    saturation or admission-rejection measurement.
    """
    import aiohttp

    from tests.router.epp_config_contract import (
        PREFILL_ABS_ENV,
        PREFILL_ADMISSION_PROFILES,
        PREFILL_FRAC_ENV,
        _plain_chat_token_case,
        assert_input_startup_capability_source_contracts,
        clear_ais_identity_envs,
        clear_known_manifest_router_envs,
        expected_prefill_admission_handles,
        prefill_admission_consumer_env,
    )
    from tests.router.epp_parity import collect_frontend_generation_route

    if profile_name not in PREFILL_ADMISSION_PROFILES:
        raise AssertionError(
            f"unknown prefill admission profile {profile_name!r}; "
            f"expected one of {PREFILL_ADMISSION_PROFILES}"
        )
    assert_input_startup_capability_source_contracts()
    clear_known_manifest_router_envs(monkeypatch)
    clear_ais_identity_envs(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    wanted_decode, wanted_abs, wanted_frac = expected_prefill_admission_handles(
        profile_name
    )
    cases = _plain_chat_token_case(model_name)
    admin_host = "127.0.0.1"

    with reserved_ports(1, DynamoPortRange.FRONTEND) as (admin_port,):
        consumer_env = prefill_admission_consumer_env(profile_name)
        consumer_env["DYN_ENABLE_FRONTEND_ADMIN_API"] = "1"
        consumer_env["DYN_FRONTEND_ADMIN_HOST"] = admin_host
        consumer_env["DYN_FRONTEND_ADMIN_PORT"] = str(admin_port)
        for name, value in consumer_env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        with stack_cls(
            request,
            tmp_path,
            model_name,
            consumer_env=consumer_env,
            log_label=f"prefill-admission-{profile_name}",
        ) as stack:
            run_frontend_epp_token_parity(stack, cases, include_negative_control=False)
            case = cases[0]
            generation = collect_frontend_generation_route(
                stack.frontend_url,
                case.payload,
                path=case.path,
                request_id=unique_request_id("fe-gen-prefill-admission"),
            )
            if stack.driver is None:
                raise AssertionError("owned ext_proc driver was not saved")
            epp_route = stack.driver.collect_route_observation(
                case.payload,
                path=case.path,
                request_id=unique_request_id("epp-route-prefill-admission"),
            )
            if not epp_route.token_ids:
                raise AssertionError("EPP route token_ids are empty")
            if (generation.worker_id, generation.dp_rank) != (
                epp_route.worker_id,
                epp_route.dp_rank,
            ):
                raise AssertionError(
                    "frontend generation and EPP route worker/rank differ: "
                    f"fe={generation.worker_id}/{generation.dp_rank} "
                    f"epp={epp_route.worker_id}/{epp_route.dp_rank}"
                )
            if generation.dp_rank != 0:
                raise AssertionError(
                    f"controlled one-worker rank is {generation.dp_rank}, expected 0"
                )
            frontend_proc = stack.frontend_process
            epp_proc = stack.epp_process
            if frontend_proc is None or frontend_proc.proc is None:
                raise AssertionError("owned frontend_process was not saved")
            if epp_proc is None or epp_proc.proc is None:
                raise AssertionError("owned epp_process was not saved")
            if (
                frontend_proc.proc.poll() is not None
                or epp_proc.proc.poll() is not None
            ):
                raise AssertionError("owned frontend or EPP process is not live")
            kube_api = stack.kube_api
            if kube_api is not None and kube_api.failures:
                raise AssertionError(
                    "Loopback Kubernetes API recorded unexpected requests: "
                    + "; ".join(kube_api.failures)
                )
            busy_url = f"http://{admin_host}:{admin_port}/busy_threshold"
            timeout = aiohttp.ClientTimeout(total=10)

            async def _get_thresholds() -> dict:
                async with aiohttp.ClientSession(
                    timeout=timeout, trust_env=False
                ) as session:
                    async with session.get(busy_url) as response:
                        if response.status != 200:
                            raise AssertionError(
                                f"GET /busy_threshold failed with status {response.status}"
                            )
                        payload = await response.json()
                        if not isinstance(payload, dict):
                            raise AssertionError(
                                f"GET /busy_threshold top-level is not a dict: {payload!r}"
                            )
                        return payload

            data = asyncio.run(_get_thresholds())
            if "thresholds" not in data:
                raise AssertionError(
                    f"GET /busy_threshold missing thresholds key: {data!r}"
                )
            thresholds = data["thresholds"]
            if not isinstance(thresholds, list) or len(thresholds) != 1:
                raise AssertionError(
                    f"GET /busy_threshold expected exactly one model entry, got {data!r}"
                )
            entry = thresholds[0]
            if not isinstance(entry, dict):
                raise AssertionError(f"busy_threshold entry is not a dict: {entry!r}")
            if "model" not in entry:
                raise AssertionError(f"busy_threshold entry missing model: {entry!r}")
            if entry["model"] != model_name:
                raise AssertionError(
                    f"busy_threshold model {entry['model']!r} != {model_name!r}"
                )
            required_keys = (
                "active_decode_blocks_threshold",
                "active_prefill_tokens_threshold",
                "active_prefill_tokens_threshold_frac",
            )
            missing = [key for key in required_keys if key not in entry]
            if missing:
                raise AssertionError(
                    f"busy_threshold entry missing keys {missing}: {entry!r}"
                )
            got_decode = entry["active_decode_blocks_threshold"]
            got_abs = entry["active_prefill_tokens_threshold"]
            got_frac = entry["active_prefill_tokens_threshold_frac"]
            if (
                got_decode != wanted_decode
                or got_abs != wanted_abs
                or got_frac != wanted_frac
            ):
                raise AssertionError(
                    f"profile {profile_name} FE handles "
                    f"decode/abs/frac={got_decode!r}/{got_abs!r}/{got_frac!r} != "
                    f"{wanted_decode!r}/{wanted_abs!r}/{wanted_frac!r}"
                )
            logger.info(
                "prefill admission profile=%s abs_env=%s frac_env=%s "
                "admin=%s handles=%s/%s/%s fe_worker=%s/%s epp_ok=True",
                profile_name,
                consumer_env.get(PREFILL_ABS_ENV),
                consumer_env.get(PREFILL_FRAC_ENV),
                admin_port,
                got_decode,
                got_abs,
                got_frac,
                generation.worker_id,
                generation.dp_rank,
            )


def frontend_epp_tools_token_parity_cases(model_name: str) -> list[TokenParityCase]:
    """Reusable actual-token scenarios for tool schema and tool-history rendering.

    Each case carries a real chat body and a sibling that changes only the tool
    schema or only the tool reply so the same comparator can reject a live
    mismatch. Payloads are built from local fresh copies.
    """
    chat_path = "/v1/chat/completions"
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    user_message = {"role": "user", "content": "What is the weather in Paris?"}

    schema_payload = {
        "model": model_name,
        "messages": [deepcopy(user_message)],
        "tools": [deepcopy(weather_tool)],
        "stream": True,
        "max_tokens": 1,
    }
    schema_divergent = deepcopy(schema_payload)
    schema_divergent["tools"][0]["function"]["parameters"]["properties"]["unit"] = {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
    }

    history_payload = {
        "model": model_name,
        "messages": [
            deepcopy(user_message),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": json.dumps({"temperature": 18}),
            },
            {"role": "user", "content": "Should I bring a jacket?"},
        ],
        "tools": [deepcopy(weather_tool)],
        "stream": True,
        "max_tokens": 1,
    }
    history_divergent = deepcopy(history_payload)
    history_divergent["messages"][2]["content"] = json.dumps({"temperature": -12})

    return [
        TokenParityCase(
            name="tool_schema",
            path=chat_path,
            payload=schema_payload,
            divergent_payload=schema_divergent,
        ),
        TokenParityCase(
            name="tool_history",
            path=chat_path,
            payload=history_payload,
            divergent_payload=history_divergent,
        ),
    ]


def run_frontend_epp_tools_token_parity(
    stack,
    cases: list[TokenParityCase],
) -> None:
    """Compare live Frontend/EPP tokens for tool schema and tool-history cases.

    Each case is handed to the existing runner with the live negative control
    enabled so a schema-only mutation and a reply-only mutation must both be
    rejected. The declaration case also collects original, tools-omitted, and
    changed-schema bodies on each arm so a silent ignore of tools cannot pass.
    """
    if not cases:
        raise AssertionError(
            "tools token parity requires at least one actual request case"
        )

    for case in cases:
        run_frontend_epp_token_parity(stack, [case], include_negative_control=True)

    declaration = next((case for case in cases if case.name == "tool_schema"), None)
    if declaration is None:
        raise AssertionError(
            "tools token parity requires a tool_schema declaration case"
        )

    omitted_payload = deepcopy(declaration.payload)
    del omitted_payload["tools"]

    bodies = (
        ("baseline", declaration.payload),
        ("tools_omitted", omitted_payload),
        ("changed_schema", declaration.divergent_payload),
    )
    observed: dict[str, dict[str, list[int]]] = {"frontend": {}, "epp": {}}
    for body_name, payload in bodies:
        observed["frontend"][body_name] = stack.collect_frontend_tokens(
            payload,
            path=declaration.path,
            request_id=unique_request_id(f"fe-{declaration.name}-{body_name}"),
        )
        observed["epp"][body_name] = stack.collect_epp_tokens(
            payload,
            path=declaration.path,
            request_id=unique_request_id(f"epp-{declaration.name}-{body_name}"),
        )

    arm_labels = {
        "frontend": "frontend.nvext.token_ids",
        "epp": "epp.nvext.token_data",
    }
    for arm_name, label in arm_labels.items():
        arm_tokens = observed[arm_name]
        for other_name in ("tools_omitted", "changed_schema"):
            case_name = f"{declaration.name}-{arm_name}-{other_name}"
            baseline_tokens = arm_tokens["baseline"]
            variant_tokens = arm_tokens[other_name]
            assert (
                baseline_tokens != variant_tokens
            ), f"{case_name}: {label} baseline matched {other_name}"
            logger.info(
                "Tools token sensitivity confirmed for %s (%s vs %s)",
                case_name,
                f"{label}.baseline",
                f"{label}.{other_name}",
            )

    if stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )


########################################################
# Frontend / Runtime EPP first-selection conformance
########################################################


@dataclass(frozen=True)
class FirstSelectionCase:
    """One actual HTTP body used for a quiescent first-selection comparison."""

    name: str
    path: str
    payload: dict[str, Any]
    label: str


def frontend_epp_first_selection_cases(model_name: str) -> list[FirstSelectionCase]:
    """Opposite long-prefix completions for aggregated first-selection parity."""
    shared = "prefix cache routing stem " * 8
    path = "/v1/completions"
    return [
        FirstSelectionCase(
            name="route_a",
            path=path,
            payload={
                "model": model_name,
                "prompt": shared + ("north unique routing branch " * 12),
                "stream": True,
                "max_tokens": 1,
            },
            label="A",
        ),
        FirstSelectionCase(
            name="route_b",
            path=path,
            payload={
                "model": model_name,
                "prompt": shared + ("south unique routing branch " * 12),
                "stream": True,
                "max_tokens": 1,
            },
            label="B",
        ),
    ]


def _route_mismatch_message(
    left,
    right,
    *,
    left_label: str,
    right_label: str,
    case_name: str,
) -> str:
    return (
        f"{case_name}: route observation mismatch: "
        f"{left_label}=(worker_id={left.worker_id}, dp_rank={left.dp_rank}) "
        f"{right_label}=(worker_id={right.worker_id}, dp_rank={right.dp_rank})"
    )


class SessionAffinityParityMismatch(AssertionError):
    """Observed FE versus EPP session-affinity first-selection gap."""


class PrefillQueueRejectionParityMismatch(AssertionError):
    """Observed FE typed queue rejection versus EPP aggregated fallback."""


class PriorityHeaderParityMismatch(AssertionError):
    """Observed FE high-then-low versus EPP header-precedence low-then-high."""


def assert_route_observations_match(
    left,
    right,
    *,
    left_label: str,
    right_label: str,
    case_name: str,
) -> None:
    """Compare typed first-selection routes only: (worker_id, dp_rank)."""
    if (left.worker_id, left.dp_rank) != (right.worker_id, right.dp_rank):
        raise AssertionError(
            _route_mismatch_message(
                left,
                right,
                left_label=left_label,
                right_label=right_label,
                case_name=case_name,
            )
        )


def assert_route_observations_differ(
    left,
    right,
    *,
    left_label: str,
    right_label: str,
    case_name: str,
) -> None:
    """Require the same route comparator to reject distinct live observations."""
    left_route = (left.worker_id, left.dp_rank)
    right_route = (right.worker_id, right.dp_rank)
    if left_route == right_route:
        raise AssertionError(
            f"{case_name}: live observations are not distinct routes: "
            f"{left_label}={left_route} {right_label}={right_route}"
        )
    try:
        assert_route_observations_match(
            left,
            right,
            left_label=left_label,
            right_label=right_label,
            case_name=case_name,
        )
    except AssertionError as exc:
        expected = _route_mismatch_message(
            left,
            right,
            left_label=left_label,
            right_label=right_label,
            case_name=case_name,
        )
        if str(exc) == expected:
            return
        raise
    raise AssertionError(
        f"{case_name}: same route comparator accepted distinct live observations "
        f"{left_label}=(worker_id={left.worker_id}, dp_rank={left.dp_rank}) "
        f"{right_label}=(worker_id={right.worker_id}, dp_rank={right.dp_rank})"
    )


def _assert_catalog_valid_route(
    observation, worker_ids: dict[str, int], case_name: str
) -> None:
    if observation.dp_rank != 0:
        raise AssertionError(
            f"{case_name}: expected aggregated dp_rank=0, got {observation.dp_rank}"
        )
    if observation.worker_id not in worker_ids.values():
        raise AssertionError(
            f"{case_name}: worker_id {observation.worker_id} is not a catalog "
            f"generate target {worker_ids}"
        )


def _wait_for_provider_query_increase(stack, before: float, *, arm: str) -> float:
    from tests.router.epp_parity import PROVIDER_QUERY_SETTLE_S

    deadline = time.monotonic() + 5.0
    last = before
    while time.monotonic() < deadline:
        time.sleep(PROVIDER_QUERY_SETTLE_S)
        last = stack.provider_query_count()
        if last > before:
            return last
    raise AssertionError(
        f"{arm}: provider {stack.component_name}.kv_indexer_query request "
        f"counter did not increase (before={before}, after={last})"
    )


def _stable_provider_query_baseline(stack, *, arm: str) -> float:
    from tests.router.epp_parity import PROVIDER_QUERY_SETTLE_S

    first = stack.provider_query_count()
    time.sleep(PROVIDER_QUERY_SETTLE_S)
    second = stack.provider_query_count()
    if first != second:
        raise AssertionError(
            f"{arm}: provider query counter is not stable before the arm "
            f"({first} -> {second})"
        )
    return second


def _require_cache_sensitive_policy(stack) -> dict[str, float]:
    parameters = stack.policy_parameters
    required = (
        "overlap_score_credit",
        "overlap_score_credit_decay",
        "prefill_load_scale",
        "router_temperature",
    )
    missing = [name for name in required if name not in parameters]
    if missing:
        raise AssertionError(
            f"aggregated policy_parameters missing {missing}: {parameters}"
        )
    if parameters["overlap_score_credit"] != 1.0:
        raise AssertionError(
            "aggregated overlap_score_credit must be 1.0, "
            f"got {parameters['overlap_score_credit']}"
        )
    if parameters["overlap_score_credit_decay"] != 0.0:
        raise AssertionError(
            "aggregated overlap_score_credit_decay must be 0.0, "
            f"got {parameters['overlap_score_credit_decay']}"
        )
    if parameters["prefill_load_scale"] != 1.0:
        raise AssertionError(
            "aggregated prefill_load_scale must be 1.0, "
            f"got {parameters['prefill_load_scale']}"
        )
    if parameters["router_temperature"] != 0.0:
        raise AssertionError(
            "aggregated router_temperature must be 0 for deterministic first "
            f"selection, got {parameters['router_temperature']}"
        )
    encoded = stack.provider_config_json
    if not isinstance(encoded, str) or not encoded:
        raise AssertionError(
            "served KvRouter config JSON was not returned by the native child"
        )
    config = json.loads(encoded)
    if config["router_track_prefill_tokens"] is not True:
        raise AssertionError(
            "served KvRouter must retain router_track_prefill_tokens for "
            "cache-sensitive prefill accounting"
        )
    if config["router_track_output_blocks"] is not False:
        raise AssertionError("served KvRouter must keep output-block tracking off")
    if config["router_replica_sync"] is not False:
        raise AssertionError("served KvRouter must keep replica sync off")
    return parameters


def _router_event_sort_key(event: dict[str, Any]) -> tuple[Any, ...]:
    try:
        data = event["event"]["data"]["stored"]
        first_block = data["blocks"][0]
        return (
            event["worker_id"],
            first_block["tokens_hash"],
            data["parent_hash"],
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise AssertionError(
            "provider dump_events entry is not a stored RouterEvent"
        ) from exc


def _provider_cache_snapshot(
    stack, *, timeout_s: float | None = None
) -> list[dict[str, Any]]:
    if timeout_s is None:
        raw = stack.dump_provider_events()
    else:
        raw = stack.dump_provider_events(timeout_s=timeout_s)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise AssertionError("provider dump_events must be a JSON list of RouterEvent")
    events: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise AssertionError("provider dump_events list contains a non-object")
        events.append(item)
    return sorted(events, key=_router_event_sort_key)


def _assert_provider_snapshot_unchanged(
    frozen: list[dict[str, Any]], current: list[dict[str, Any]], case_name: str
) -> None:
    assert_event_dumps_equal(
        frozen,
        current,
        f"{case_name}.frozen",
        f"{case_name}.current",
    )


def _instance_worker_block_size(stack) -> int:
    size = stack.worker_block_size
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise AssertionError(
            f"stack worker_block_size {size!r} is not a positive instance size"
        )
    return size


def _require_overlap_block_size(
    scores: Mapping[str, Any], worker_block_size: int, *, label: str
) -> None:
    got = scores.get("block_size")
    if got != worker_block_size:
        raise AssertionError(
            f"{label}: provider overlap block_size={got!r} != instance "
            f"worker_block_size={worker_block_size}"
        )


def observe_owned_frontend_model_block_size(
    stack,
    *,
    expected: int,
    node_name: str,
    deadline: float,
    tokens: list[int],
) -> None:
    """Read the owned Frontend model block-size gauge and provider query under one deadline."""
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise AssertionError(
            f"expected model block size {expected!r} is not a positive worker-card value"
        )
    if not tokens:
        raise AssertionError("provider overlap query requires nonempty tokens")
    process = stack.frontend_process
    if process is None or process.proc is None:
        raise AssertionError("owned frontend_process was not saved")
    if process.proc.poll() is not None:
        raise AssertionError("owned frontend_process is not live")
    pid = process.proc.pid
    port = stack.frontend_port
    if port is None:
        raise AssertionError("owned frontend HTTP port was not saved")
    if not process.log_path:
        raise AssertionError("owned frontend_process has no log_path")
    log_path = Path(process.log_path)
    pair_label = stack.consumer_pair_label
    if not pair_label:
        raise AssertionError("owned consumer pair label was not saved")
    posix = log_path.as_posix()
    if node_name not in posix:
        raise AssertionError(
            f"frontend_process log_path {log_path} is not this node {node_name}"
        )
    if pair_label not in posix:
        raise AssertionError(
            f"frontend_process log_path {log_path} is not this pair {pair_label}"
        )
    metric_name = f"{name_prefix.FRONTEND}_{frontend_service.MODEL_KV_CACHE_BLOCK_SIZE}"
    model = stack.model_name
    if not model:
        raise AssertionError("owned model label is missing")
    worker_card = _instance_worker_block_size(stack)
    cli_size = stack.frontend_cli_block_size
    while True:
        if stack.frontend_process is not process:
            raise AssertionError("owned frontend_process is not the captured process")
        if process.proc is None or process.proc.poll() is not None:
            raise AssertionError("owned frontend_process is not live")
        if process.proc.pid != pid:
            raise AssertionError(
                f"owned frontend_process moved from pid={pid} to {process.proc.pid}"
            )
        if stack.frontend_port != port:
            raise AssertionError("owned frontend HTTP port changed during observation")
        if Path(process.log_path) != log_path:
            raise AssertionError(
                f"owned frontend_process log moved from {log_path} to {process.log_path}"
            )
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"http://127.0.0.1:{port}/metrics", timeout=2)
        if response.status_code != 200:
            raise AssertionError(
                f"frontend_process pid={pid} HTTP /metrics on port {port} "
                f"returned HTTP {response.status_code}"
            )
        family = find_metric_samples(response.text, metric_name)
        samples = find_metric_samples(response.text, metric_name, {labels.MODEL: model})
        if family and samples:
            if len(family) != 1 or len(samples) != 1:
                raise AssertionError(
                    f"frontend_process pid={pid} port={port} {metric_name} "
                    f"model={model!r} is ambiguous: family={len(family)} "
                    f"model_samples={len(samples)}"
                )
            value = samples[0]
            if value != float(expected):
                raise AssertionError(
                    f"frontend_process pid={pid} port={port} {metric_name} "
                    f"model={model!r} value={value!r} != worker-card {expected}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"frontend_process pid={pid} port={port} provider overlap "
                    "query exhausted the readiness budget"
                )

            async def _provider_block_size() -> Any:
                return await asyncio.wait_for(
                    stack.provider_overlap_async(tokens, timeout_s=remaining),
                    timeout=remaining,
                )

            try:
                scores = stack.run_until_complete(_provider_block_size())
            except asyncio.TimeoutError as exc:
                raise AssertionError(
                    f"frontend_process pid={pid} port={port} provider overlap "
                    "query outlived the remaining readiness budget"
                ) from exc
            _require_overlap_block_size(
                scores, expected, label=f"{pair_label}-provider-query"
            )
            logger.info(
                "owned frontend model block-size pid=%s port=%s model=%s "
                "metric=%s worker_card=%s cli=%s provider_block_size=%s "
                "pair=%s",
                pid,
                port,
                model,
                value,
                worker_card,
                cli_size,
                scores.get("block_size"),
                pair_label,
            )
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"frontend_process pid={pid} port={port} missing "
                f"{metric_name} model={model!r} worker-card={expected}"
            )
        remaining = deadline - time.monotonic()
        time.sleep(min(0.05, max(remaining, 0.0)))


def _prepare_first_selection_snapshot(
    stack,
    cases: list[FirstSelectionCase],
    *,
    expected_model_block_size: int | None = None,
):
    # Unrelated router tests import common; keep epp_parity/grpc_tools
    # selected-test-only so ordinary collection does not require them.
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        collect_frontend_query_token_ids,
    )

    if stack.provider_router is None:
        raise AssertionError("first-selection readiness requires a served KvRouter")

    worker_block_size = _instance_worker_block_size(stack)
    policy = _require_cache_sensitive_policy(stack)
    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    logger.info(
        "first-selection readiness workers=%s provider=%s policy=%s configs=%s",
        stack.worker_ids,
        stack.provider_instance_id,
        policy,
        stack.worker_configs,
    )

    async def activate_publishers() -> None:
        deadline = time.monotonic() + READINESS_DEADLINE_S
        for worker_id, label in ((worker_a, "A"), (worker_b, "B")):
            key = (worker_id, 0)
            attempt = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"publisher activation for {label} exhausted the 45s "
                        f"budget after {attempt} probes key={key}"
                    )
                attempt += 1
                probe = [random.randint(1, 10000) for _ in range(2 * worker_block_size)]
                try:
                    await asyncio.wait_for(
                        stack.provider_generate(
                            model_name=stack.model_name,
                            token_ids=probe,
                            stop_conditions={"ignore_eos": True, "max_tokens": 2},
                            worker_id=worker_id,
                            dp_rank=0,
                            timeout_s=remaining,
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise AssertionError(
                        f"publisher probe request/stream for {label} outlived "
                        f"the 45s readiness budget on attempt {attempt}"
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"publisher activation for {label} exhausted the 45s "
                        f"budget after probe {attempt} before overlap"
                    )
                probe_deadline = min(deadline, time.monotonic() + 2.0)
                indexed = False
                while time.monotonic() < probe_deadline:
                    query_budget = probe_deadline - time.monotonic()
                    if query_budget <= 0:
                        break
                    try:
                        scores = await asyncio.wait_for(
                            stack.provider_overlap_async(probe, timeout_s=query_budget),
                            timeout=query_budget,
                        )
                    except asyncio.TimeoutError:
                        if time.monotonic() >= deadline:
                            raise AssertionError(
                                f"publisher overlap query for {label} outlived "
                                "the 45s readiness budget"
                            )
                        break
                    _require_overlap_block_size(
                        scores,
                        worker_block_size,
                        label=f"publisher-{label}",
                    )
                    blocks = {
                        (row["worker_id"], row["dp_rank"]): row["device_blocks"]
                        for row in scores["workers"]
                    }
                    if blocks.get(key, 0) >= 2:
                        indexed = True
                        break
                    await asyncio.sleep(
                        min(0.05, max(probe_deadline - time.monotonic(), 0.0))
                    )
                if indexed:
                    break

    stack.run_until_complete(activate_publishers())

    token_ids_by_label: dict[str, list[int]] = {}
    stack.open_consumer_pair(include_epp=False)
    try:
        for case in cases:
            token_ids_by_label[case.label] = collect_frontend_query_token_ids(
                stack.frontend_url,
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"fe-ready-{case.name}"),
            )
        if expected_model_block_size is not None:
            observe_owned_frontend_model_block_size(
                stack,
                expected=expected_model_block_size,
                node_name=stack.request.node.name,
                deadline=time.monotonic() + READINESS_DEADLINE_S,
                tokens=token_ids_by_label[cases[0].label],
            )
    finally:
        stack.close_consumer_pair()

    async def warm_and_gate() -> None:
        deadline = time.monotonic() + READINESS_DEADLINE_S
        last: dict[str, dict[tuple[int, int], int]] = {}
        last_loads: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"{case.name}: pinned warmup exhausted the 45s budget before send"
                )
            tokens = token_ids_by_label[case.label]
            target = stack.worker_ids[case.label]
            try:
                await asyncio.wait_for(
                    stack.provider_generate(
                        model_name=stack.model_name,
                        token_ids=tokens,
                        stop_conditions={"ignore_eos": True, "max_tokens": 2},
                        worker_id=target,
                        dp_rank=0,
                        timeout_s=remaining,
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                raise AssertionError(
                    f"{case.name}: pinned warmup request/stream outlived the "
                    "45s readiness budget"
                ) from exc
            complete_blocks = len(tokens) // worker_block_size
            if complete_blocks < 2:
                raise AssertionError(
                    f"{case.name}: readiness tokens must cover at least two full "
                    f"blocks, got {len(tokens)} tokens for "
                    f"worker_block_size={worker_block_size}"
                )
        while time.monotonic() < deadline:
            ready = True
            for case in cases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    ready = False
                    break
                tokens = token_ids_by_label[case.label]
                try:
                    scores = await asyncio.wait_for(
                        stack.provider_overlap_async(tokens, timeout_s=remaining),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise AssertionError(
                        f"{case.name}: get_overlap_scores outlived the remaining "
                        "45s readiness budget"
                    ) from exc
                _require_overlap_block_size(scores, worker_block_size, label=case.name)
                blocks = {
                    (row["worker_id"], row["dp_rank"]): row["device_blocks"]
                    for row in scores["workers"]
                }
                last[case.label] = blocks
                preferred = stack.worker_ids[case.label]
                other = worker_b if case.label == "A" else worker_a
                preferred_blocks = blocks.get((preferred, 0), 0)
                other_blocks = blocks.get((other, 0), 0)
                complete_blocks = len(tokens) // worker_block_size
                if (
                    preferred_blocks < complete_blocks
                    or preferred_blocks <= other_blocks
                ):
                    ready = False
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    ready = False
                    break
                try:
                    loads = await asyncio.wait_for(
                        stack.provider_potential_loads_async(
                            tokens, timeout_s=remaining
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise AssertionError(
                        f"{case.name}: get_potential_loads outlived the remaining "
                        "45s readiness budget"
                    ) from exc
                last_loads[case.label] = loads
                by_worker = {(row["worker_id"], row["dp_rank"]): row for row in loads}
                preferred_load = by_worker.get((preferred, 0))
                other_load = by_worker.get((other, 0))
                if preferred_load is None or other_load is None:
                    ready = False
                    continue
                if (
                    preferred_load["active_requests"] != 0
                    or other_load["active_requests"] != 0
                ):
                    ready = False
                    continue
                if (
                    preferred_load["potential_prefill_tokens"]
                    >= other_load["potential_prefill_tokens"]
                ):
                    ready = False
            if ready:
                logger.info(
                    "first-selection snapshot margins=%s loads=%s policy=%s",
                    last,
                    last_loads,
                    policy,
                )
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))
        raise AssertionError(
            "provider overlap/load snapshot did not reach a full-block non-tie "
            f"quiescent margin before the 45s readiness budget: {last} "
            f"loads={last_loads}"
        )

    stack.run_until_complete(warm_and_gate())
    stack.token_ids_by_label = token_ids_by_label


def run_frontend_epp_first_selection_parity(
    stack,
    cases: list[FirstSelectionCase],
    *,
    expected_model_block_size: int | None = None,
) -> None:
    """Compare live FE query-only and first EPP selections on a frozen snapshot."""
    from tests.router.epp_parity import READINESS_DEADLINE_S

    if len(cases) != 2 or {case.label for case in cases} != {"A", "B"}:
        raise AssertionError("first-selection parity requires opposite A/B cases")

    _prepare_first_selection_snapshot(
        stack, cases, expected_model_block_size=expected_model_block_size
    )
    frozen_events = _provider_cache_snapshot(stack)
    logger.info(
        "first-selection frozen snapshot events=%s workers=%s provider=%s",
        len(frozen_events),
        stack.worker_ids,
        stack.provider_instance_id,
    )

    observed: dict[str, tuple[Any, Any]] = {}
    for case in cases:
        stack.open_consumer_pair()
        try:
            if expected_model_block_size is not None:
                observe_owned_frontend_model_block_size(
                    stack,
                    expected=expected_model_block_size,
                    node_name=stack.request.node.name,
                    deadline=time.monotonic() + READINESS_DEADLINE_S,
                    tokens=stack.token_ids_by_label[case.label],
                )
            before = _stable_provider_query_baseline(stack, arm=f"{case.name}-frontend")
            frontend_route = stack.collect_frontend_route(
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"fe-sel-{case.name}"),
            )
            after_fe = _wait_for_provider_query_increase(
                stack, before, arm=f"{case.name}-frontend"
            )
            epp_before = _stable_provider_query_baseline(stack, arm=f"{case.name}-epp")
            if epp_before < after_fe:
                raise AssertionError(
                    f"{case.name}-epp: provider counter decreased between arms "
                    f"({after_fe} -> {epp_before})"
                )
            epp_route = stack.collect_epp_route(
                case.payload,
                path=case.path,
                request_id=unique_request_id(f"epp-sel-{case.name}"),
            )
            after_epp = _wait_for_provider_query_increase(
                stack, epp_before, arm=f"{case.name}-epp"
            )
            _assert_catalog_valid_route(
                frontend_route, stack.worker_ids, f"{case.name}-frontend"
            )
            _assert_catalog_valid_route(epp_route, stack.worker_ids, f"{case.name}-epp")
            expected_worker = stack.worker_ids[case.label]
            if frontend_route.worker_id != expected_worker:
                raise AssertionError(
                    f"{case.name}: frontend first selection {frontend_route.worker_id} "
                    f"does not follow the verified snapshot worker {expected_worker}"
                )
            assert_ordered_token_ids_match(
                frontend_route.token_ids,
                epp_route.token_ids,
                left_label="frontend.token_ids",
                right_label="epp.token_ids",
                case_name=case.name,
            )
            assert_route_observations_match(
                frontend_route,
                epp_route,
                left_label="frontend",
                right_label="epp",
                case_name=case.name,
            )
            current_events = _provider_cache_snapshot(stack)
            _assert_provider_snapshot_unchanged(
                frozen_events, current_events, case.name
            )
            logger.info(
                "%s pair result frontend=%s epp=%s counters=%s->%s->%s",
                case.name,
                (frontend_route.worker_id, frontend_route.dp_rank),
                (epp_route.worker_id, epp_route.dp_rank),
                before,
                after_fe,
                after_epp,
            )
            observed[case.label] = (frontend_route, epp_route)
        finally:
            stack.close_consumer_pair()

    if observed["A"][0].worker_id == observed["B"][0].worker_id:
        raise AssertionError(
            "R_A and R_B must flip the expected generate target; "
            f"both selected worker_id={observed['A'][0].worker_id}"
        )
    logger.info(
        "first-selection target flip A=%s B=%s",
        observed["A"][0].worker_id,
        observed["B"][0].worker_id,
    )

    control_a = cases[0] if cases[0].label == "A" else cases[1]
    control_b = cases[0] if cases[0].label == "B" else cases[1]
    stack.open_consumer_pair()
    try:
        if expected_model_block_size is not None:
            observe_owned_frontend_model_block_size(
                stack,
                expected=expected_model_block_size,
                node_name=stack.request.node.name,
                deadline=time.monotonic() + READINESS_DEADLINE_S,
                tokens=stack.token_ids_by_label[control_a.label],
            )
        before = _stable_provider_query_baseline(stack, arm="negative-frontend")
        frontend_route = stack.collect_frontend_route(
            control_a.payload,
            path=control_a.path,
            request_id=unique_request_id("fe-neg-route-a"),
        )
        after_fe = _wait_for_provider_query_increase(
            stack, before, arm="negative-frontend"
        )
        epp_before = _stable_provider_query_baseline(stack, arm="negative-epp")
        if epp_before < after_fe:
            raise AssertionError(
                f"negative-epp: provider counter decreased between arms "
                f"({after_fe} -> {epp_before})"
            )
        epp_route = stack.collect_epp_route(
            control_b.payload,
            path=control_b.path,
            request_id=unique_request_id("epp-neg-route-b"),
        )
        _wait_for_provider_query_increase(stack, epp_before, arm="negative-epp")
        _assert_catalog_valid_route(
            frontend_route, stack.worker_ids, "negative-frontend"
        )
        _assert_catalog_valid_route(epp_route, stack.worker_ids, "negative-epp")
        assert_route_observations_differ(
            frontend_route,
            epp_route,
            left_label="frontend.R_A",
            right_label="epp.R_B",
            case_name="live-negative-route",
        )
        current_events = _provider_cache_snapshot(stack)
        _assert_provider_snapshot_unchanged(
            frozen_events, current_events, "live-negative-route"
        )
        logger.info(
            "live-negative-route rejected frontend.R_A=%s epp.R_B=%s",
            (frontend_route.worker_id, frontend_route.dp_rank),
            (epp_route.worker_id, epp_route.dp_rank),
        )
    finally:
        stack.close_consumer_pair()

    if stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )


def run_dynamic_block_size_consumer(
    request,
    tmp_path,
    model_name: str,
    stack_cls,
    *,
    worker_block_size: int,
    frontend_cli_block_size: int,
) -> None:
    """Attribute Dynamic FE/EPP block size to the worker card, not the FE CLI."""
    cases = frontend_epp_first_selection_cases(model_name)
    with stack_cls(
        request,
        tmp_path,
        model_name,
        worker_block_size=worker_block_size,
        frontend_cli_block_size=frontend_cli_block_size,
    ) as stack:
        run_frontend_epp_first_selection_parity(
            stack,
            cases,
            expected_model_block_size=worker_block_size,
        )


########################################################
# Frontend / Runtime EPP input-family conformance
########################################################


def _with_nvext(payload: dict[str, Any], **fields: Any) -> dict[str, Any]:
    body = deepcopy(payload)
    nvext = dict(body.get("nvext") or {})
    nvext.update(fields)
    body["nvext"] = nvext
    return body


def _overlap_blocks(scores: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    return {
        (row["worker_id"], row["dp_rank"]): row["device_blocks"]
        for row in scores["workers"]
    }


def _active_requests(loads: list[Mapping[str, Any]]) -> dict[tuple[int, int], int]:
    return {(row["worker_id"], row["dp_rank"]): row["active_requests"] for row in loads}


def _dp0_worker_load_rows(
    loads: list[Mapping[str, Any]],
    worker_a: int,
    worker_b: int,
    case_name: str,
) -> dict[tuple[int, int], Mapping[str, Any]] | None:
    """Return exact DP0 rows for both catalog workers, or None if incomplete.

    Duplicates fail immediately. Empty or missing rows are not evidence.
    """
    if not loads:
        return None
    seen: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in loads:
        key = (row["worker_id"], row["dp_rank"])
        if key in seen:
            raise AssertionError(f"{case_name}: duplicate potential_load row {key}")
        seen[key] = row
    expected = {(worker_a, 0), (worker_b, 0)}
    if set(seen) != expected:
        return None
    return seen


def _exact_dp0_zero_active(
    loads: list[Mapping[str, Any]],
    worker_a: int,
    worker_b: int,
    case_name: str,
) -> bool:
    rows = _dp0_worker_load_rows(loads, worker_a, worker_b, case_name)
    if rows is None:
        return False
    return all(rows[key]["active_requests"] == 0 for key in rows)


def _namespaced_overlap_snapshot(
    stack,
    token_ids: list[int],
    *,
    worker_a: int,
    worker_b: int,
    deadline: float | None = None,
) -> dict[str | None, Any]:
    from tests.router.epp_parity import READINESS_DEADLINE_S

    tables: dict[str | None, Any] = {}
    for namespace in ("tenant-a", "tenant-b", None):
        if deadline is None:
            timeout_s = READINESS_DEADLINE_S
        else:
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                raise AssertionError(
                    "cache-salt readiness exhausted the 45s budget before "
                    f"overlap snapshot cache_namespace={namespace!r}"
                )
        scores = stack.provider_overlap_scores(
            token_ids, cache_namespace=namespace, timeout_s=timeout_s
        )
        if deadline is None:
            timeout_s = READINESS_DEADLINE_S
        else:
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                raise AssertionError(
                    "cache-salt readiness exhausted the 45s budget before "
                    f"load snapshot cache_namespace={namespace!r}"
                )
        loads = stack.provider_potential_loads(
            token_ids, cache_namespace=namespace, timeout_s=timeout_s
        )
        rows = _dp0_worker_load_rows(
            loads, worker_a, worker_b, f"cache-salt:{namespace}"
        )
        tables[namespace] = {
            "blocks": _overlap_blocks(scores),
            "active": _active_requests(loads),
            "rows": rows,
        }
    return tables


def _assert_overlap_snapshots_match(
    frozen: Mapping[str | None, Any],
    current: Mapping[str | None, Any],
    case_name: str,
) -> None:
    if frozen != current:
        raise AssertionError(
            f"{case_name}: namespaced overlap/load tables changed: "
            f"frozen={frozen} current={current}"
        )


def _assert_zero_active_load(
    tables: Mapping[str | None, Any],
    worker_a: int,
    worker_b: int,
    case_name: str,
) -> None:
    for namespace, table in tables.items():
        rows = table["rows"]
        if rows is None or any(rows[key]["active_requests"] != 0 for key in rows):
            raise AssertionError(
                f"{case_name}: active load is not exactly both DP0 workers "
                f"at zero for cache_namespace={namespace!r}: {rows}"
            )


def _compare_input_routes(
    stack,
    payload: dict[str, Any],
    *,
    path: str,
    case_name: str,
    headers: Mapping[str, str] | None = None,
    expected_worker: int | None = None,
    compare_tokens: bool = True,
):
    before = _stable_provider_query_baseline(stack, arm=f"{case_name}-frontend")
    frontend_route = stack.collect_frontend_route(
        payload,
        path=path,
        request_id=unique_request_id(f"fe-{case_name}"),
        headers=headers,
    )
    after_fe = _wait_for_provider_query_increase(
        stack, before, arm=f"{case_name}-frontend"
    )
    epp_before = _stable_provider_query_baseline(stack, arm=f"{case_name}-epp")
    if epp_before < after_fe:
        raise AssertionError(
            f"{case_name}-epp: provider counter decreased between arms "
            f"({after_fe} -> {epp_before})"
        )
    epp_route = stack.collect_epp_route(
        payload,
        path=path,
        request_id=unique_request_id(f"epp-{case_name}"),
        headers=headers,
    )
    _wait_for_provider_query_increase(stack, epp_before, arm=f"{case_name}-epp")
    _assert_catalog_valid_route(
        frontend_route, stack.worker_ids, f"{case_name}-frontend"
    )
    _assert_catalog_valid_route(epp_route, stack.worker_ids, f"{case_name}-epp")
    if compare_tokens:
        assert_ordered_token_ids_match(
            frontend_route.token_ids,
            epp_route.token_ids,
            left_label="frontend.token_ids",
            right_label="epp.token_ids",
            case_name=case_name,
        )
    assert_route_observations_match(
        frontend_route,
        epp_route,
        left_label="frontend",
        right_label="epp",
        case_name=case_name,
    )
    if expected_worker is not None and frontend_route.worker_id != expected_worker:
        raise AssertionError(
            f"{case_name}: frontend selected {frontend_route.worker_id}, "
            f"expected {expected_worker}"
        )
    return frontend_route, epp_route


def frontend_epp_cache_salt_case(model_name: str) -> FirstSelectionCase:
    """One fixed completion body reused across two cache namespaces."""
    return FirstSelectionCase(
        name="cache_salt",
        path="/v1/completions",
        payload={
            "model": model_name,
            "prompt": "cache salt routing stem " * 8 + ("shared salt body " * 12),
            "stream": True,
            "max_tokens": 1,
        },
        label="S",
    )


def run_frontend_epp_cache_salt_parity(stack, case: FirstSelectionCase) -> None:
    """Compare FE/EPP first selections under controlled namespaced indexer state.

    Native Mocker does not publish salted engine KV events. This profile seeds
    the approximate indexer through public best_worker and is not proof of
    engine cache-salt isolation.
    """
    from dynamo.llm import RoutingConstraints
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        ZONE_A_TAINT,
        ZONE_B_TAINT,
        collect_frontend_query_token_ids,
    )

    _require_cache_sensitive_policy(stack)
    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    if stack.worker_configs[worker_a]["taints"] & {ZONE_A_TAINT} != {ZONE_A_TAINT}:
        raise AssertionError(
            f"worker A {worker_a} missing published {ZONE_A_TAINT}: "
            f"{stack.worker_configs[worker_a]}"
        )
    if stack.worker_configs[worker_b]["taints"] & {ZONE_B_TAINT} != {ZONE_B_TAINT}:
        raise AssertionError(
            f"worker B {worker_b} missing published {ZONE_B_TAINT}: "
            f"{stack.worker_configs[worker_b]}"
        )

    stack.open_consumer_pair(include_epp=False)
    deadline = time.monotonic() + READINESS_DEADLINE_S

    def remaining(step: str) -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise AssertionError(
                f"cache-salt readiness exhausted the 45s budget at {step}"
            )
        return left

    try:
        token_ids = collect_frontend_query_token_ids(
            stack.frontend_url,
            case.payload,
            path=case.path,
            request_id=unique_request_id("fe-ready-cache-salt"),
            timeout_s=remaining("tokenize"),
        )
    finally:
        stack.close_consumer_pair()
    if len(token_ids) // 16 < 2:
        raise AssertionError(
            f"{case.name}: readiness tokens must cover at least two full "
            f"blocks, got {len(token_ids)} tokens"
        )

    selected_a = stack.provider_best_worker(
        token_ids,
        timeout_s=remaining("seed-tenant-a"),
        update_indexer=True,
        cache_namespace="tenant-a",
        routing_constraints=RoutingConstraints(required_taints={ZONE_A_TAINT}),
        request_id=None,
    )
    selected_b = stack.provider_best_worker(
        token_ids,
        timeout_s=remaining("seed-tenant-b"),
        update_indexer=True,
        cache_namespace="tenant-b",
        routing_constraints=RoutingConstraints(required_taints={ZONE_B_TAINT}),
        request_id=None,
    )
    if (selected_a[0], selected_a[1]) != (worker_a, 0):
        raise AssertionError(
            f"best_worker tenant-a/zone-a returned {selected_a}, expected "
            f"{(worker_a, 0)}"
        )
    if (selected_b[0], selected_b[1]) != (worker_b, 0):
        raise AssertionError(
            f"best_worker tenant-b/zone-b returned {selected_b}, expected "
            f"{(worker_b, 0)}"
        )

    last_tables: dict[str | None, Any] = {}
    while time.monotonic() < deadline:
        last_tables = _namespaced_overlap_snapshot(
            stack,
            token_ids,
            worker_a=worker_a,
            worker_b=worker_b,
            deadline=deadline,
        )
        blocks_a = last_tables["tenant-a"]["blocks"]
        blocks_b = last_tables["tenant-b"]["blocks"]
        default_blocks = {
            key: value for key, value in last_tables[None]["blocks"].items() if value
        }
        winner_a = blocks_a.get((worker_a, 0), 0)
        winner_b = blocks_b.get((worker_b, 0), 0)
        other_a = blocks_a.get((worker_b, 0), 0)
        other_b = blocks_b.get((worker_a, 0), 0)
        if (
            winner_a > 0
            and winner_b > 0
            and winner_a > other_a
            and winner_b > other_b
            and not default_blocks
        ):
            if all(
                table["rows"] is not None
                and all(
                    table["rows"][key]["active_requests"] == 0 for key in table["rows"]
                )
                for namespace, table in last_tables.items()
            ):
                _assert_zero_active_load(
                    last_tables, worker_a, worker_b, "cache-salt-setup"
                )
                break
        time.sleep(0.05)
    else:
        raise AssertionError(
            "approximate namespaced overlap did not reach opposite unique "
            f"winners with empty default namespace: {last_tables}"
        )
    frozen = last_tables
    logger.info("cache-salt controlled overlap tables=%s", frozen)

    comparisons = (
        ("salt-a", _with_nvext(case.payload, cache_salt="tenant-a"), None, worker_a),
        ("salt-b", _with_nvext(case.payload, cache_salt="tenant-b"), None, worker_b),
        (
            "header-over-body",
            _with_nvext(case.payload, cache_salt="tenant-a"),
            {"x-tenant-id": "tenant-b"},
            worker_b,
        ),
    )
    seen_tokens: list[int] | None = None
    for name, payload, headers, expected in comparisons:
        stack.open_consumer_pair()
        try:
            frontend_route, epp_route = _compare_input_routes(
                stack,
                payload,
                path=case.path,
                case_name=name,
                headers=headers,
                expected_worker=expected,
            )
            if seen_tokens is None:
                seen_tokens = frontend_route.token_ids
            elif frontend_route.token_ids != seen_tokens:
                raise AssertionError(
                    f"{name}: token ids changed across cache-salt variants"
                )
            if epp_route.token_ids != seen_tokens:
                raise AssertionError(
                    f"{name}: EPP token ids diverged from the shared salt body"
                )
            _assert_overlap_snapshots_match(
                frozen,
                _namespaced_overlap_snapshot(
                    stack, token_ids, worker_a=worker_a, worker_b=worker_b
                ),
                name,
            )
        finally:
            stack.close_consumer_pair()

    if stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )


def frontend_epp_constraints_cold_case(model_name: str) -> FirstSelectionCase:
    return FirstSelectionCase(
        name="constraints_cold",
        path="/v1/completions",
        payload={
            "model": model_name,
            "prompt": "constraints cold routing stem " * 8
            + ("unique cold branch " * 12),
            "stream": True,
            "max_tokens": 1,
        },
        label="C",
    )


def run_frontend_epp_routing_constraints_parity(
    stack, cached_cases: list[FirstSelectionCase], cold_case: FirstSelectionCase
) -> None:
    """Compare FE/EPP routes under published worker taints and request constraints."""
    from dynamo.llm import RoutingConstraints
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        ZONE_A_TAINT,
        ZONE_B_TAINT,
        collect_frontend_query_token_ids,
    )

    preferred_weight = 0.85
    if len(cached_cases) != 2 or {case.label for case in cached_cases} != {"A", "B"}:
        raise AssertionError("constraints parity requires opposite A/B cache cases")

    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    if stack.worker_configs[worker_a]["taints"] & {ZONE_A_TAINT} != {ZONE_A_TAINT}:
        raise AssertionError(
            f"worker A {worker_a} missing published {ZONE_A_TAINT}: "
            f"{stack.worker_configs[worker_a]}"
        )
    if stack.worker_configs[worker_b]["taints"] & {ZONE_B_TAINT} != {ZONE_B_TAINT}:
        raise AssertionError(
            f"worker B {worker_b} missing published {ZONE_B_TAINT}: "
            f"{stack.worker_configs[worker_b]}"
        )

    _prepare_first_selection_snapshot(stack, cached_cases)
    frozen_events = _provider_cache_snapshot(stack)
    cached_a = next(case for case in cached_cases if case.label == "A")
    stack.open_consumer_pair()
    try:
        _compare_input_routes(
            stack,
            cached_a.payload,
            path=cached_a.path,
            case_name="unconstrained-cache-a",
            expected_worker=worker_a,
        )
        required_b = RoutingConstraints(required_taints={ZONE_B_TAINT})
        if required_b.required_taints != {ZONE_B_TAINT}:
            raise AssertionError(
                "native required_taints contract mismatch: "
                f"{required_b.required_taints}"
            )
        _compare_input_routes(
            stack,
            _with_nvext(
                cached_a.payload,
                routing_constraints={"required_taints": [ZONE_B_TAINT]},
            ),
            path=cached_a.path,
            case_name="required-taint-b",
            expected_worker=worker_b,
        )
        _assert_provider_snapshot_unchanged(
            frozen_events, _provider_cache_snapshot(stack), "required-taint-b"
        )
    finally:
        stack.close_consumer_pair()

    stack.open_consumer_pair(include_epp=False)
    try:
        cold_tokens = collect_frontend_query_token_ids(
            stack.frontend_url,
            cold_case.payload,
            path=cold_case.path,
            request_id=unique_request_id("fe-ready-constraints-cold"),
        )
    finally:
        stack.close_consumer_pair()

    deadline = time.monotonic() + READINESS_DEADLINE_S
    last_blocks: dict[tuple[int, int], int] = {}
    last_loads: list[Any] = []
    cold_ready = False
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        scores = stack.provider_overlap_scores(cold_tokens, timeout_s=remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        loads = stack.provider_potential_loads(cold_tokens, timeout_s=remaining)
        last_blocks = _overlap_blocks(scores)
        last_loads = loads
        rows = _dp0_worker_load_rows(loads, worker_a, worker_b, "preferred-taint-cold")
        if rows is None:
            time.sleep(0.05)
            continue
        load_a = rows[(worker_a, 0)]
        load_b = rows[(worker_b, 0)]
        prefill_a = load_a["potential_prefill_tokens"]
        prefill_b = load_b["potential_prefill_tokens"]
        if (
            last_blocks.get((worker_a, 0), 0) == 0
            and last_blocks.get((worker_b, 0), 0) == 0
            and load_a["active_requests"] == 0
            and load_b["active_requests"] == 0
            and prefill_a == prefill_b
            and prefill_a > 0
        ):
            cold_ready = True
            break
        time.sleep(0.05)
    if not cold_ready:
        raise AssertionError(
            "preferred-taint cold overlap/load did not reach zero-cache "
            "zero-active equal positive potential_prefill: "
            f"blocks={last_blocks} loads={last_loads}"
        )
    preferred_frozen = _provider_cache_snapshot(stack)

    prefer_a = RoutingConstraints(preferred_taints={ZONE_A_TAINT: preferred_weight})
    prefer_b = RoutingConstraints(preferred_taints={ZONE_B_TAINT: preferred_weight})
    weight_a = prefer_a.preferred_taints.get(ZONE_A_TAINT)
    weight_b = prefer_b.preferred_taints.get(ZONE_B_TAINT)
    if weight_a is None or abs(float(weight_a) - preferred_weight) > 1e-5:
        raise AssertionError(
            "native preferred_taints weight for zone-a is not 0.85: "
            f"{prefer_a.preferred_taints}"
        )
    if weight_b is None or abs(float(weight_b) - preferred_weight) > 1e-5:
        raise AssertionError(
            "native preferred_taints weight for zone-b is not 0.85: "
            f"{prefer_b.preferred_taints}"
        )

    stack.open_consumer_pair()
    try:
        prefer_a_routes = _compare_input_routes(
            stack,
            _with_nvext(
                cold_case.payload,
                routing_constraints={
                    "preferred_taints": {ZONE_A_TAINT: preferred_weight}
                },
            ),
            path=cold_case.path,
            case_name="preferred-taint-a",
            expected_worker=worker_a,
        )
        prefer_b_routes = _compare_input_routes(
            stack,
            _with_nvext(
                cold_case.payload,
                routing_constraints={
                    "preferred_taints": {ZONE_B_TAINT: preferred_weight}
                },
            ),
            path=cold_case.path,
            case_name="preferred-taint-b",
            expected_worker=worker_b,
        )
        if prefer_a_routes[0].token_ids != prefer_b_routes[0].token_ids:
            raise AssertionError("preferred-taint cases used different token ids")
        if prefer_a_routes[0].token_ids != cold_tokens:
            raise AssertionError(
                "preferred-taint tokens diverged from the cold tokenization"
            )
        assert_route_observations_differ(
            prefer_a_routes[0],
            prefer_b_routes[1],
            left_label="frontend.preferred.A",
            right_label="epp.preferred.B",
            case_name="preferred-taint-live-negative",
        )
        _assert_provider_snapshot_unchanged(
            preferred_frozen,
            _provider_cache_snapshot(stack),
            "preferred-taint",
        )
    finally:
        stack.close_consumer_pair()

    if stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )


def run_frontend_epp_session_id_parity(stack, cases: list[FirstSelectionCase]) -> None:
    """Establish real FE session affinity, then compare the mismatch observation."""
    from tests.router.epp_parity import READINESS_DEADLINE_S

    if len(cases) != 2 or {case.label for case in cases} != {"A", "B"}:
        raise AssertionError("session parity requires opposite A/B cases")

    _prepare_first_selection_snapshot(stack, cases)
    frozen_events = _provider_cache_snapshot(stack)
    case_a = next(case for case in cases if case.label == "A")
    case_b = next(case for case in cases if case.label == "B")
    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    session_headers = {"x-dynamo-session-id": f"inputs-session-{uuid.uuid4()}"}
    baseline_a = stack.token_ids_by_label["A"]

    stack.open_consumer_pair()
    try:
        _compare_input_routes(
            stack,
            case_a.payload,
            path=case_a.path,
            case_name="no-session-a",
            expected_worker=worker_a,
        )
        _compare_input_routes(
            stack,
            case_b.payload,
            path=case_b.path,
            case_name="no-session-b",
            expected_worker=worker_b,
        )
        _assert_provider_snapshot_unchanged(
            frozen_events, _provider_cache_snapshot(stack), "no-session-setup"
        )

        generation = stack.collect_frontend_generation(
            case_a.payload,
            path=case_a.path,
            request_id=unique_request_id("fe-session-establish-a"),
            headers=session_headers,
        )
        _assert_catalog_valid_route(
            generation, stack.worker_ids, "fe-session-establish-a"
        )
        if generation.worker_id != worker_a:
            raise AssertionError(
                "FE session establishment did not bind worker A: "
                f"{generation.worker_id} != {worker_a}"
            )
        epp_seed = stack.collect_epp_route(
            case_a.payload,
            path=case_a.path,
            request_id=unique_request_id("epp-session-seed-a"),
            headers=session_headers,
        )
        _assert_catalog_valid_route(epp_seed, stack.worker_ids, "epp-session-seed-a")
        if epp_seed.worker_id != worker_a:
            raise AssertionError(
                "EPP session seed did not select catalog worker A: "
                f"{epp_seed.worker_id} != {worker_a}"
            )
        assert_ordered_token_ids_match(
            baseline_a,
            epp_seed.token_ids,
            left_label="baseline.A.token_ids",
            right_label="epp.seed.token_ids",
            case_name="epp-session-seed-a",
        )
        deadline = time.monotonic() + READINESS_DEADLINE_S
        last_snapshot = None
        after_establish = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            loads = stack.provider_potential_loads(baseline_a, timeout_s=remaining)
            if not _exact_dp0_zero_active(
                loads, worker_a, worker_b, "session-establish"
            ):
                last_snapshot = None
                time.sleep(0.05)
                continue
            current = _provider_cache_snapshot(stack)
            if last_snapshot is not None and current == last_snapshot:
                after_establish = current
                break
            last_snapshot = current
            time.sleep(0.05)
        if after_establish is None:
            raise AssertionError(
                "session establishment events did not settle after worker A bind"
            )

        bound_before = _stable_provider_query_baseline(stack, arm="fe-session-prefix-b")
        bound = stack.collect_frontend_route(
            case_b.payload,
            path=case_b.path,
            request_id=unique_request_id("fe-session-prefix-b"),
            headers=session_headers,
        )
        after_bound = _wait_for_provider_query_increase(
            stack, bound_before, arm="fe-session-prefix-b"
        )
        _assert_catalog_valid_route(bound, stack.worker_ids, "fe-session-prefix-b")
        if bound.worker_id != worker_a:
            raise AssertionError(
                "FE session S did not stay bound to worker A on prefix B: "
                f"{bound.worker_id} != {worker_a}"
            )
        control_before = _stable_provider_query_baseline(
            stack, arm="fe-nosession-prefix-b"
        )
        if control_before < after_bound:
            raise AssertionError(
                "fe-nosession-prefix-b: provider counter decreased between arms "
                f"({after_bound} -> {control_before})"
            )
        control = stack.collect_frontend_route(
            case_b.payload,
            path=case_b.path,
            request_id=unique_request_id("fe-nosession-prefix-b"),
        )
        after_control = _wait_for_provider_query_increase(
            stack, control_before, arm="fe-nosession-prefix-b"
        )
        _assert_catalog_valid_route(control, stack.worker_ids, "fe-nosession-prefix-b")
        if control.worker_id != worker_b:
            raise AssertionError(
                "FE no-session control did not prefer worker B on prefix B: "
                f"{control.worker_id} != {worker_b}"
            )
        epp_before = _stable_provider_query_baseline(stack, arm="epp-session-prefix-b")
        if epp_before < after_control:
            raise AssertionError(
                "epp-session-prefix-b: provider counter decreased between arms "
                f"({after_control} -> {epp_before})"
            )
        epp_bound = stack.collect_epp_route(
            case_b.payload,
            path=case_b.path,
            request_id=unique_request_id("epp-session-prefix-b"),
            headers=session_headers,
        )
        _wait_for_provider_query_increase(stack, epp_before, arm="epp-session-prefix-b")
        _assert_catalog_valid_route(epp_bound, stack.worker_ids, "epp-session-prefix-b")
        assert_ordered_token_ids_match(
            bound.token_ids,
            control.token_ids,
            left_label="frontend.session.token_ids",
            right_label="frontend.nosession.token_ids",
            case_name="session-prefix-b",
        )
        assert_ordered_token_ids_match(
            bound.token_ids,
            epp_bound.token_ids,
            left_label="frontend.session.token_ids",
            right_label="epp.session.token_ids",
            case_name="session-prefix-b",
        )
        _assert_provider_snapshot_unchanged(
            after_establish,
            _provider_cache_snapshot(stack),
            "session-prefix-b",
        )
        if stack.kube_api.failures:
            raise AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(stack.kube_api.failures)
            )
        if (
            bound.worker_id == worker_a
            and epp_bound.worker_id == worker_b
            and bound.dp_rank == 0
            and epp_bound.dp_rank == 0
        ):
            raise SessionAffinityParityMismatch(
                _route_mismatch_message(
                    bound,
                    epp_bound,
                    left_label="frontend.session",
                    right_label="epp.session",
                    case_name="session-prefix-b",
                )
            )
        assert_route_observations_match(
            bound,
            epp_bound,
            left_label="frontend.session",
            right_label="epp.session",
            case_name="session-prefix-b",
        )
    finally:
        stack.close_consumer_pair()

    if stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )


########################################################
# Frontend / Runtime EPP priority queue conformance
########################################################


def frontend_epp_priority_case(model_name: str) -> dict[str, Any]:
    return {
        "path": "/v1/completions",
        "payload": {
            "model": model_name,
            "prompt": "priority queue stem " * 8 + ("held prefill body " * 8),
            "stream": True,
            "max_tokens": 1,
        },
    }


def _priority_request_id() -> str:
    return str(uuid.uuid4())


def _canonical_request_uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{label}: request id {value!r} is empty")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise AssertionError(f"{label}: request id {value!r} is not a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise AssertionError(
            f"{label}: request id {value!r} is not canonical {canonical}"
        )
    return canonical


def _cmpl_internal_request_id(raw: Any, *, label: str) -> str:
    prefix = "cmpl-"
    if not isinstance(raw, str) or not raw.startswith(prefix):
        raise AssertionError(
            f"{label}: SSE id {raw!r} does not have the exact {prefix} prefix"
        )
    return _canonical_request_uuid(raw[len(prefix) :], label=label)


def _prefill_records_after(stack, boundary: int) -> list[dict[str, Any]]:
    records = stack.selection_records()
    if boundary < 0 or boundary > len(records):
        raise AssertionError(
            f"owned selection boundary {boundary} is outside {len(records)} records"
        )
    return [
        record
        for record in records[boundary:]
        if record.get("worker_type") == "prefill"
    ]


def _require_eligible_prefill_booking(
    record: Mapping[str, Any], *, expected_prefill: int, label: str
) -> str:
    request_id = _canonical_request_uuid(record.get("request_id"), label=label)
    if record.get("worker_type") != "prefill":
        raise AssertionError(f"{label}: booking is not prefill: {record}")
    candidates = record.get("candidates") or []
    if record.get("candidate_count") != 1 or len(candidates) != 1:
        raise AssertionError(
            f"{label}: booking is not one candidate_count/triple: {record}"
        )
    worker_id, dp_rank, cost = candidates[0]
    if worker_id != expected_prefill or dp_rank != 0:
        raise AssertionError(
            f"{label}: booking worker/rank is {(worker_id, dp_rank)}, "
            f"expected ({expected_prefill}, 0): {record}"
        )
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        raise AssertionError(f"{label}: booking cost is not numeric: {record}")
    if cost != cost or cost in (float("inf"), float("-inf")):
        raise AssertionError(f"{label}: booking cost is not finite: {record}")
    return request_id


def _require_no_prior_selection(stack, *, case_name: str) -> int:
    prior = stack.selection_records()
    if prior:
        raise AssertionError(
            f"{case_name}: owned selection records are not empty before blocker: "
            f"{prior}"
        )
    return 0


def _require_fe_generation_internal_ids(
    outcomes: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    booked_blocker_id: str,
    case_name: str,
) -> dict[str, str]:
    ids: dict[str, str] = {}
    for label, outcome in outcomes:
        if "error" in outcome:
            raise AssertionError(f"{case_name} FE {label} failed: {outcome['error']}")
        if outcome.get("status") != 200:
            raise AssertionError(
                f"{case_name} FE {label} HTTP {outcome.get('status')}: "
                f"{str(outcome.get('text', ''))[:300]}"
            )
        internal = outcome.get("internal_id")
        if not isinstance(internal, str) or not internal:
            raise AssertionError(
                f"{case_name} FE {label} missing response-owned internal id"
            )
        ids[label] = internal
    if ids["blocker"] != booked_blocker_id:
        raise AssertionError(
            f"{case_name}: blocker SSE internal id {ids['blocker']!r} != "
            f"post-book {booked_blocker_id!r}"
        )
    if len(set(ids.values())) != 3:
        raise AssertionError(f"{case_name}: FE internal ids are not distinct: {ids}")
    return ids


def _start_frontend_generation(
    stack,
    payload,
    *,
    path: str,
    request_id: str,
    headers,
    deadline: float,
    label: str,
):
    from tests.router.epp_parity import (
        PRIORITY_HOLD_DEADLINE_S,
        _frontend_sse_data_events,
        merge_request_headers,
    )

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssertionError(f"{label}: FE hold budget exhausted before HTTP start")
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            merged = dict(merge_request_headers(request_id, headers))
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    f"{stack.frontend_url}{path}",
                    json=payload,
                    headers=merged,
                    timeout=remaining + PRIORITY_HOLD_DEADLINE_S,
                    stream=True,
                )
                outcome["status"] = response.status_code
                outcome["text"] = response.text
                if response.status_code != 200:
                    return
                events = _frontend_sse_data_events(response.text)
                for event in events[:-1]:
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError as exc:
                        raise AssertionError(
                            f"{label}: FE generation stream has a "
                            "non-JSON data field"
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise AssertionError(
                            f"{label}: FE generation stream data field "
                            "is not an object"
                        )
                    if "error" in chunk:
                        raise AssertionError(
                            f"{label}: FE generation stream contained " "an error chunk"
                        )
                    derived = _cmpl_internal_request_id(chunk.get("id"), label=label)
                    recorded = outcome.get("internal_id")
                    if recorded is None:
                        outcome["internal_id"] = derived
                    elif recorded != derived:
                        raise AssertionError(
                            f"{label}: FE generation SSE ids are mixed "
                            f"{recorded!r} vs {derived!r}"
                        )
                if "internal_id" not in outcome:
                    raise AssertionError(
                        f"{label}: FE generation stream has no cmpl- internal id"
                    )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    stack.track_http_thread(thread, label)
    return thread, outcome


def _assert_held_generation_open(
    outcome: Mapping[str, Any],
    thread,
    *,
    label: str,
    when: str,
) -> None:
    if "error" in outcome:
        raise AssertionError(f"{label} finished with error {when}: {outcome['error']}")
    status = outcome.get("status")
    if status is not None and status != 200:
        raise AssertionError(
            f"{label} HTTP {status} {when}: " f"{str(outcome.get('text', ''))[:300]}"
        )
    if "text" in outcome:
        raise AssertionError(f"{label} completed {when}: {outcome!r}")
    if not thread.is_alive():
        raise AssertionError(f"{label} thread exited {when}")


def _require_priority_threads_finished(
    threads: Sequence[tuple[str, Any]],
    *,
    case_name: str,
) -> None:
    alive = [label for label, thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError(f"{case_name}: FE threads still alive after join: {alive}")


def _wait_prefill_booking(
    stack,
    *,
    boundary: int,
    deadline: float,
    expected_prefill: int,
    blocker_thread,
    blocker_out,
    case_name: str,
):
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        _assert_held_generation_open(
            blocker_out,
            blocker_thread,
            label=f"{case_name}: FE blocker",
            when="before booking",
        )
        fresh = _prefill_records_after(stack, boundary)
        if len(fresh) > 1:
            raise AssertionError(
                f"{case_name}: FE blocker booking is ambiguous: {fresh}"
            )
        if len(fresh) == 1:
            request_id = _require_eligible_prefill_booking(
                fresh[0],
                expected_prefill=expected_prefill,
                label=f"{case_name}-blocker",
            )
            return request_id, fresh[0]
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: FE blocker post-book prefill missing after boundary "
        f"{boundary}: {stack.selection_records()}"
    )


def _wait_fe_priority_prefill_order(
    stack,
    *,
    low_id: str,
    high_id: str,
    blocker_id: str,
    boundary: int,
    expected_prefill: int,
    deadline: float,
    case_name: str,
) -> list[str]:
    last: list[str] = []
    blocker_count = 0
    while time.monotonic() < deadline:
        last = []
        seen: set[str] = set()
        blocker_count = 0
        for record in _prefill_records_after(stack, boundary):
            rid = record.get("request_id")
            if rid == blocker_id:
                if blocker_count:
                    raise AssertionError(
                        f"{case_name}: duplicate FE blocker prefill {record}"
                    )
                _require_eligible_prefill_booking(
                    record,
                    expected_prefill=expected_prefill,
                    label=f"{case_name}-blocker",
                )
                blocker_count += 1
                continue
            if rid not in {low_id, high_id}:
                raise AssertionError(
                    f"{case_name}: unexpected FE prefill booking {record}"
                )
            if rid in seen:
                raise AssertionError(
                    f"{case_name}: duplicate FE prefill booking {record}"
                )
            _require_eligible_prefill_booking(
                record,
                expected_prefill=expected_prefill,
                label=f"{case_name}-{rid}",
            )
            seen.add(rid)
            last.append(rid)
        if blocker_count != 1:
            if last:
                raise AssertionError(
                    f"{case_name}: FE post-boundary cohort is not exactly "
                    f"one blocker plus low/high: blocker={blocker_count} "
                    f"low/high={last}"
                )
        elif len(last) == 2:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: FE post-boundary prefill cohort is blocker="
        f"{blocker_count} low/high={last}, expected one blocker plus both"
    )


def _wait_pending_prefill(stack, expected: float, deadline: float, case_name: str):
    last = None
    observed: list[str] = []
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        last, observed = stack.frontend_pending_prefill(timeout_s=min(2.0, remaining))
        if last == expected:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: frontend prefill pending is {last}, expected {expected}; "
        f"observed {observed}"
    )


def _priority_inputs(kind: str):
    low_body = {"agent_hints": {"priority": 1, "strict_priority": 0}}
    high_body = {"agent_hints": {"priority": 8, "strict_priority": 1}}
    if kind == "body_strict":
        return low_body, high_body, {}, {}
    if kind == "header_strict":
        return (
            {"agent_hints": {"priority": 1, "strict_priority": 1}},
            {"agent_hints": {"priority": 1, "strict_priority": 0}},
            {"x-dynamo-request-strict-priority": "0"},
            {"x-dynamo-request-strict-priority": "1"},
        )
    if kind == "header_normal":
        return (
            {"agent_hints": {"priority": 100, "strict_priority": 0}},
            {"agent_hints": {"priority": 1, "strict_priority": 0}},
            {"x-dynamo-request-priority": "1"},
            {"x-dynamo-request-priority": "100"},
        )
    raise AssertionError(f"unknown priority kind {kind}")


def _close_priority_held_streams(*handles) -> None:
    errors: list[BaseException] = []
    for handle in handles:
        if handle is None:
            continue
        try:
            handle.close()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def _raise_priority_cleanup(body_error, close_error) -> None:
    if close_error is not None:
        cleanup = RuntimeError(f"{type(close_error).__name__}: {close_error}")
        if body_error is not None:
            raise cleanup from body_error
        raise cleanup
    if body_error is not None:
        raise body_error


def _stop_frontend_and_join_http(stack) -> None:
    errors: list[BaseException] = []
    try:
        stack.stop_owned_frontend()
    except BaseException as exc:
        errors.append(exc)
    try:
        stack.join_owned_http_threads()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise errors[0]


def _finish_fe_priority_hold(stack, body_error) -> None:
    cleanup = None
    try:
        stack.resume_owned_prefill()
    except BaseException as exc:
        cleanup = exc
    try:
        _stop_frontend_and_join_http(stack)
    except BaseException as exc:
        if cleanup is None:
            cleanup = exc
    if cleanup is not None:
        raise RuntimeError(f"{type(cleanup).__name__}: {cleanup}") from (
            body_error if body_error is not None else cleanup
        )
    if body_error is not None:
        raise body_error


def _assert_priority_catalog(stack, catalog: dict[str, int], when: str) -> None:
    current = stack.catalog_snapshot()
    if current != catalog or current != stack.worker_ids:
        raise AssertionError(
            f"catalog/lease changed {when}: {current} vs {catalog} "
            f"/ {stack.worker_ids}"
        )


def _post_fe_priority_sentinel(
    stack,
    case: dict[str, Any],
    payload: dict[str, Any],
    sentinel_id: str,
    deadline: float,
    case_name: str,
):
    from tests.router.epp_parity import _typed_fe_queue_rejection, merge_request_headers

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssertionError(f"{case_name}: FE hold budget exhausted before sentinel")
    headers = dict(merge_request_headers(sentinel_id, None))
    with requests.Session() as session:
        session.trust_env = False
        sentinel = session.post(
            f"{stack.frontend_url}{case['path']}",
            json=payload,
            headers=headers,
            timeout=remaining,
        )
    return _typed_fe_queue_rejection(sentinel)


def _assert_priority_hold_pending(
    stack,
    *,
    low_thread,
    high_thread,
    low_out,
    high_out,
    blocker_id: str,
    boundary: int,
    expected_prefill: int,
    case_name: str,
) -> None:
    _assert_held_generation_open(
        low_out,
        low_thread,
        label=f"{case_name}: FE low",
        when="during hold",
    )
    _assert_held_generation_open(
        high_out,
        high_thread,
        label=f"{case_name}: FE high",
        when="during hold",
    )
    fresh = _prefill_records_after(stack, boundary)
    if len(fresh) != 1:
        raise AssertionError(
            f"{case_name}: FE hold has {len(fresh)} post-boundary prefill "
            f"bookings, expected only the blocker: {fresh}"
        )
    booked = _require_eligible_prefill_booking(
        fresh[0],
        expected_prefill=expected_prefill,
        label=f"{case_name}-blocker",
    )
    if booked != blocker_id:
        raise AssertionError(
            f"{case_name}: FE hold prefill {booked!r} != blocker {blocker_id!r}"
        )


def _wait_blocker_queue_ordering(stack, deadline: float, case_name: str) -> None:
    last: list[str] = []
    while time.monotonic() < deadline:
        last = stack.queue_ordering_records(after_bytes=0)
        if len(last) > 1:
            raise AssertionError(
                f"{case_name}: FE blocker queue ordering records are {last}, "
                "not exactly 1"
            )
        if len(last) == 1:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: FE blocker queue ordering records are {last}, "
        "expected exactly 1"
    )


def _wait_priority_queue_ordering(
    stack,
    *,
    after_bytes: int,
    deadline: float,
    case_name: str,
    low_thread,
    high_thread,
    low_out,
    high_out,
    blocker_id: str,
    boundary: int,
    expected_prefill: int,
) -> None:
    from tests.router.epp_parity import PRIORITY_QUEUE_LIMIT

    # Closed traffic, one scheduler actor: "ordering request" is not enqueue
    # proof. The later single FE 529 is the full-capacity proof.
    last: list[str] = []
    while time.monotonic() < deadline:
        _assert_priority_hold_pending(
            stack,
            low_thread=low_thread,
            high_thread=high_thread,
            low_out=low_out,
            high_out=high_out,
            blocker_id=blocker_id,
            boundary=boundary,
            expected_prefill=expected_prefill,
            case_name=case_name,
        )
        last = stack.queue_ordering_records(after_bytes=after_bytes)
        if len(last) > PRIORITY_QUEUE_LIMIT:
            raise AssertionError(
                f"{case_name}: FE queue ordering records are {last}, "
                f"not exactly {PRIORITY_QUEUE_LIMIT}"
            )
        if len(last) == PRIORITY_QUEUE_LIMIT:
            _assert_priority_hold_pending(
                stack,
                low_thread=low_thread,
                high_thread=high_thread,
                low_out=low_out,
                high_out=high_out,
                blocker_id=blocker_id,
                boundary=boundary,
                expected_prefill=expected_prefill,
                case_name=case_name,
            )
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: FE queue ordering records are {last}, "
        f"expected exactly {PRIORITY_QUEUE_LIMIT}"
    )


def _prove_fe_priority_capacity(
    stack,
    case: dict[str, Any],
    *,
    catalog: dict[str, int],
    deadline: float,
    case_name: str,
    blocker_id: str,
    boundary: int,
    expected_prefill: int,
    low_id: str,
    high_id: str,
    sentinel_id: str,
    low_payload: dict[str, Any],
    high_payload: dict[str, Any],
    low_headers,
    high_headers,
    low_label: str,
    high_label: str,
):
    from tests.router.epp_parity import PRIORITY_QUEUE_LIMIT

    _wait_blocker_queue_ordering(stack, deadline, case_name)
    log_cursor = stack.frontend_log_bytes()
    low_thread, low_out = _start_frontend_generation(
        stack,
        low_payload,
        path=case["path"],
        request_id=low_id,
        headers=low_headers or None,
        deadline=deadline,
        label=low_label,
    )
    high_thread, high_out = _start_frontend_generation(
        stack,
        high_payload,
        path=case["path"],
        request_id=high_id,
        headers=high_headers or None,
        deadline=deadline,
        label=high_label,
    )
    _wait_priority_queue_ordering(
        stack,
        after_bytes=log_cursor,
        deadline=deadline,
        case_name=case_name,
        low_thread=low_thread,
        high_thread=high_thread,
        low_out=low_out,
        high_out=high_out,
        blocker_id=blocker_id,
        boundary=boundary,
        expected_prefill=expected_prefill,
    )
    _assert_priority_catalog(stack, catalog, "before FE sentinel")
    sentinel = _post_fe_priority_sentinel(
        stack, case, low_payload, sentinel_id, deadline, case_name
    )
    _assert_priority_hold_pending(
        stack,
        low_thread=low_thread,
        high_thread=high_thread,
        low_out=low_out,
        high_out=high_out,
        blocker_id=blocker_id,
        boundary=boundary,
        expected_prefill=expected_prefill,
        case_name=case_name,
    )
    _wait_pending_prefill(stack, float(PRIORITY_QUEUE_LIMIT), deadline, case_name)
    _assert_priority_hold_pending(
        stack,
        low_thread=low_thread,
        high_thread=high_thread,
        low_out=low_out,
        high_out=high_out,
        blocker_id=blocker_id,
        boundary=boundary,
        expected_prefill=expected_prefill,
        case_name=case_name,
    )
    _assert_priority_catalog(stack, catalog, "during FE hold")
    return low_thread, low_out, high_thread, high_out, sentinel


def _wait_epp_blocker_queue_ordering(stack, deadline: float, case_name: str) -> None:
    # The held blocker schedules prefill and decode before mutation returns.
    last: list[str] = []
    while time.monotonic() < deadline:
        last = stack.epp_queue_ordering_records(after_bytes=0)
        if len(last) > 2:
            raise AssertionError(
                f"{case_name}: EPP blocker queue ordering records are {last}, "
                "not exactly 2"
            )
        if len(last) == 2:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: EPP blocker queue ordering records are {last}, "
        "expected exactly 2"
    )


def _wait_epp_low_high_queue_ordering(
    stack,
    *,
    after_bytes: int,
    deadline: float,
    case_name: str,
    epp_low,
    epp_high,
) -> None:
    from tests.router.epp_parity import PRIORITY_QUEUE_LIMIT

    last: list[str] = []
    while time.monotonic() < deadline:
        if (
            _poll_held_admission(epp_low, 0.0) is not None
            or _poll_held_admission(epp_high, 0.0) is not None
        ):
            raise AssertionError(f"{case_name}: EPP low/high admitted before sentinel")
        last = stack.epp_queue_ordering_records(after_bytes=after_bytes)
        if len(last) > PRIORITY_QUEUE_LIMIT:
            raise AssertionError(
                f"{case_name}: EPP queue ordering records are {last}, "
                f"not exactly {PRIORITY_QUEUE_LIMIT}"
            )
        if len(last) == PRIORITY_QUEUE_LIMIT:
            if (
                _poll_held_admission(epp_low, 0.0) is not None
                or _poll_held_admission(epp_high, 0.0) is not None
            ):
                raise AssertionError(
                    f"{case_name}: EPP low/high admitted before sentinel"
                )
            return
        time.sleep(0.05)
    raise AssertionError(
        f"{case_name}: EPP queue ordering records are {last}, "
        f"expected exactly {PRIORITY_QUEUE_LIMIT}"
    )


def _wait_epp_prefill_fallback(
    stack, *, after_bytes: int, deadline: float, case_name: str
) -> None:
    last: list[str] = []
    while time.monotonic() < deadline:
        last = stack.epp_prefill_fallback_rejections(after_bytes=after_bytes)
        if len(last) > 1:
            raise AssertionError(
                f"{case_name}: EPP prefill fallback records are extra: " f"{len(last)}"
            )
        if len(last) == 1:
            return
        time.sleep(0.05)
    raise AssertionError(f"{case_name}: EPP prefill fallback queue rejection is absent")


def _prove_epp_priority_capacity(
    stack,
    case: dict[str, Any],
    *,
    catalog: dict[str, int],
    case_name: str,
    blocker_obs,
    low_id: str,
    high_id: str,
    sentinel_id: str,
    low_payload: dict[str, Any],
    high_payload: dict[str, Any],
    low_headers,
    high_headers,
):
    from tests.router.epp_parity import PRIORITY_HOLD_DEADLINE_S

    deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
    _wait_epp_blocker_queue_ordering(stack, deadline, case_name)
    log_cursor = stack.epp_log_bytes()
    with contextlib.ExitStack() as acquired:
        epp_low = stack.driver.open_held_stream(
            low_payload,
            path=case["path"],
            request_id=low_id,
            headers=low_headers or None,
        )
        acquired.callback(epp_low.close)
        epp_high = stack.driver.open_held_stream(
            high_payload,
            path=case["path"],
            request_id=high_id,
            headers=high_headers or None,
        )
        acquired.callback(epp_high.close)
        _wait_epp_low_high_queue_ordering(
            stack,
            after_bytes=log_cursor,
            deadline=deadline,
            case_name=case_name,
            epp_low=epp_low,
            epp_high=epp_high,
        )
        _assert_priority_catalog(stack, catalog, "before EPP sentinel")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"{case_name}: EPP hold budget exhausted before sentinel"
            )
        epp_sentinel = stack.driver.open_held_stream(
            low_payload,
            path=case["path"],
            request_id=sentinel_id,
            accept_queue_rejection=True,
        )
        acquired.callback(epp_sentinel.close)
        sentinel_outcome = epp_sentinel.wait_routing_mutation(timeout_s=remaining)
        sentinel_kind = _classify_epp_sentinel_outcome(
            sentinel_outcome, case_name, catalog, blocker_obs
        )
        if sentinel_kind == "aggregated":
            _wait_epp_prefill_fallback(
                stack,
                after_bytes=log_cursor,
                deadline=deadline,
                case_name=case_name,
            )
        if (
            _poll_held_admission(epp_low, 0.0) is not None
            or _poll_held_admission(epp_high, 0.0) is not None
        ):
            raise AssertionError(
                f"{case_name}: EPP low/high admitted before blocker release"
            )
        _assert_priority_catalog(stack, catalog, "during EPP hold")
        acquired.pop_all()
        return epp_low, epp_high, epp_sentinel, sentinel_kind, sentinel_outcome


def _poll_held_admission(handle, timeout_s: float):
    try:
        return handle.wait_routing_mutation(timeout_s=timeout_s)
    except TimeoutError:
        return None


def _classify_epp_sentinel_outcome(
    outcome, case_name: str, catalog, blocker_obs
) -> str:
    from tests.router.epp_parity import QueueRejectionOutcome, RouteObservation

    if isinstance(outcome, QueueRejectionOutcome):
        return "typed"
    if not (
        isinstance(outcome, RouteObservation) and outcome.routing_mode == "aggregated"
    ):
        raise AssertionError(
            f"{case_name}: EPP sentinel is neither typed 503 nor "
            f"aggregated fallback: {outcome!r}"
        )
    if outcome.worker_id != catalog["decode"]:
        raise AssertionError(
            f"{case_name}: EPP aggregated sentinel worker_id is "
            f"{outcome.worker_id}, not catalog decode {catalog['decode']}"
        )
    if outcome.dp_rank != 0:
        raise AssertionError(
            f"{case_name}: EPP aggregated sentinel dp_rank is "
            f"{outcome.dp_rank}, not DP0"
        )
    if outcome.token_ids != blocker_obs.token_ids:
        raise AssertionError(
            f"{case_name}: EPP aggregated sentinel token_ids differ from " "blocker"
        )
    if outcome.prefill_worker_id is not None:
        raise AssertionError(
            f"{case_name}: EPP aggregated sentinel reserved prefill "
            f"{outcome.prefill_worker_id}"
        )
    return "aggregated"


def run_frontend_epp_priority_parity(stack, model_name: str, kind: str) -> None:
    """Compare FE/EPP admission under a held 1P+1D queue of two."""
    from tests.router.epp_parity import PRIORITY_HOLD_DEADLINE_S

    case = frontend_epp_priority_case(model_name)
    low_nvext, high_nvext, low_headers, high_headers = _priority_inputs(kind)
    catalog = stack.catalog_snapshot()
    if catalog != stack.worker_ids:
        raise AssertionError(
            f"catalog changed before hold: {catalog} vs {stack.worker_ids}"
        )
    blocker_id = _priority_request_id()
    low_id = _priority_request_id()
    high_id = _priority_request_id()
    sentinel_id = _priority_request_id()
    low_payload = _with_nvext(case["payload"], **low_nvext)
    high_payload = _with_nvext(case["payload"], **high_nvext)
    deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
    stack.pause_owned_prefill(deadline)
    fe_error = None
    try:
        _assert_priority_catalog(stack, catalog, "after SIGSTOP")
        boundary = _require_no_prior_selection(stack, case_name=kind)
        blocker_thread, blocker = _start_frontend_generation(
            stack,
            case["payload"],
            path=case["path"],
            request_id=blocker_id,
            headers=None,
            deadline=deadline,
            label=f"{kind}-blocker",
        )
        blocker_internal, _booking = _wait_prefill_booking(
            stack,
            boundary=boundary,
            deadline=deadline,
            expected_prefill=catalog["prefill"],
            blocker_thread=blocker_thread,
            blocker_out=blocker,
            case_name=kind,
        )
        (
            low_thread,
            low_out,
            high_thread,
            high_out,
            _fe_sentinel,
        ) = _prove_fe_priority_capacity(
            stack,
            case,
            catalog=catalog,
            deadline=deadline,
            case_name=kind,
            blocker_id=blocker_internal,
            boundary=boundary,
            expected_prefill=catalog["prefill"],
            low_id=low_id,
            high_id=high_id,
            sentinel_id=sentinel_id,
            low_payload=low_payload,
            high_payload=high_payload,
            low_headers=low_headers,
            high_headers=high_headers,
            low_label=f"{kind}-low",
            high_label=f"{kind}-high",
        )
        stack.resume_owned_prefill()
        join_deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
        leftover = join_deadline - time.monotonic()
        for thread in (blocker_thread, low_thread, high_thread):
            thread.join(timeout=max(leftover, 0.0))
            leftover = join_deadline - time.monotonic()
        _require_priority_threads_finished(
            (
                ("blocker", blocker_thread),
                ("low", low_thread),
                ("high", high_thread),
            ),
            case_name=kind,
        )
        internals = _require_fe_generation_internal_ids(
            (
                ("blocker", blocker),
                ("low", low_out),
                ("high", high_out),
            ),
            booked_blocker_id=blocker_internal,
            case_name=kind,
        )
        fe_order = _wait_fe_priority_prefill_order(
            stack,
            low_id=internals["low"],
            high_id=internals["high"],
            blocker_id=blocker_internal,
            boundary=boundary,
            expected_prefill=catalog["prefill"],
            deadline=join_deadline,
            case_name=kind,
        )
        if fe_order != [internals["high"], internals["low"]]:
            raise AssertionError(
                f"{kind}: FE admission order {fe_order} is not high then low"
            )
    except BaseException as exc:
        fe_error = exc
    _finish_fe_priority_hold(stack, fe_error)

    epp_low = None
    epp_high = None
    epp_blocker = None
    epp_sentinel = None
    body_error = None
    close_error = None
    try:
        epp_blocker = stack.driver.open_held_stream(
            case["payload"],
            path=case["path"],
            request_id=_priority_request_id(),
        )
        blocker_obs = epp_blocker.wait_routing_mutation(
            timeout_s=PRIORITY_HOLD_DEADLINE_S
        )
        if blocker_obs.routing_mode != "disaggregated":
            raise AssertionError(
                f"{kind}: EPP blocker is {blocker_obs.routing_mode}, "
                "not a held prefill reservation"
            )
        if blocker_obs.prefill_worker_id != catalog["prefill"]:
            raise AssertionError(
                f"{kind}: EPP blocker did not reserve prefill "
                f"{catalog['prefill']}: {blocker_obs.prefill_worker_id}"
            )
        _assert_priority_catalog(stack, catalog, "after EPP prefill reservation")
        (
            epp_low,
            epp_high,
            epp_sentinel,
            sentinel_kind,
            _,
        ) = _prove_epp_priority_capacity(
            stack,
            case,
            catalog=catalog,
            case_name=kind,
            blocker_obs=blocker_obs,
            low_id=low_id,
            high_id=high_id,
            sentinel_id=sentinel_id,
            low_payload=low_payload,
            high_payload=high_payload,
            low_headers=low_headers,
            high_headers=high_headers,
        )
        if sentinel_kind == "aggregated":
            epp_sentinel.release()
        if (
            _poll_held_admission(epp_low, 0.0) is not None
            or _poll_held_admission(epp_high, 0.0) is not None
        ):
            raise AssertionError(
                f"{kind}: EPP low/high admitted before blocker release"
            )
        _assert_priority_catalog(stack, catalog, "while EPP prefill blocker held")
        if stack.kube_api.failures:
            raise AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(stack.kube_api.failures)
            )
        epp_blocker.release()
        admitted = None
        pending = None
        admit_deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
        while time.monotonic() < admit_deadline:
            remaining = min(0.2, admit_deadline - time.monotonic())
            _poll_held_admission(epp_high, remaining)
            _poll_held_admission(epp_low, remaining)
            if epp_high.observation is not None and epp_low.observation is None:
                admitted, pending = "high", "low"
                break
            if epp_low.observation is not None and epp_high.observation is None:
                admitted, pending = "low", "high"
                break
            if epp_low.observation is not None and epp_high.observation is not None:
                raise AssertionError(f"{kind}: both EPP streams admitted at once")
        if (admitted, pending) not in {("high", "low"), ("low", "high")}:
            raise AssertionError(
                f"{kind}: EPP first admission is {admitted} then {pending}, "
                "not exactly one of high-then-low or low-then-high"
            )
        first = epp_high if admitted == "high" else epp_low
        leftover = epp_low if pending == "low" else epp_high
        first.release()
        leftover_deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
        leftover_obs = None
        while leftover_obs is None and time.monotonic() < leftover_deadline:
            leftover_obs = _poll_held_admission(
                leftover, leftover_deadline - time.monotonic()
            )
        if leftover_obs is None:
            raise AssertionError(f"{kind}: EPP leftover {pending} never admitted")
        for label, obs in (
            (admitted, first.observation),
            (pending, leftover_obs),
        ):
            if obs is None:
                raise AssertionError(f"{kind}: EPP {label} has no observation")
            if obs.routing_mode != "disaggregated":
                raise AssertionError(
                    f"{kind}: EPP {label} is {obs.routing_mode}, "
                    "not a held prefill reservation"
                )
            if obs.prefill_worker_id != catalog["prefill"]:
                raise AssertionError(
                    f"{kind}: EPP {label} did not reserve prefill "
                    f"{catalog['prefill']}: {obs.prefill_worker_id}"
                )
            if obs.worker_id != catalog["decode"]:
                raise AssertionError(
                    f"{kind}: EPP {label} worker_id is {obs.worker_id}, "
                    f"not catalog decode {catalog['decode']}"
                )
            if obs.dp_rank != 0:
                raise AssertionError(
                    f"{kind}: EPP {label} dp_rank is {obs.dp_rank}, not DP0"
                )
            if obs.token_ids != blocker_obs.token_ids:
                raise AssertionError(
                    f"{kind}: EPP {label} token_ids differ from blocker"
                )
        leftover.release()
        if epp_sentinel.queue_rejection is None:
            epp_sentinel.release()
        _assert_priority_catalog(stack, catalog, "after EPP admission")
        if stack.kube_api.failures:
            raise AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(stack.kube_api.failures)
            )
        if admitted != "high" or pending != "low":
            if (
                kind in {"header_strict", "header_normal"}
                and fe_order == [internals["high"], internals["low"]]
                and admitted == "low"
                and pending == "high"
            ):
                raise PriorityHeaderParityMismatch(
                    f"{kind}: EPP admission order is {admitted} then {pending}, "
                    "not high then low"
                )
            raise AssertionError(
                f"{kind}: EPP admission order is {admitted} then {pending}, "
                "not high then low"
            )
    except BaseException as exc:
        body_error = exc
    finally:
        try:
            _close_priority_held_streams(epp_blocker, epp_low, epp_high, epp_sentinel)
        except BaseException as exc:
            close_error = exc
    _raise_priority_cleanup(body_error, close_error)


def run_frontend_epp_priority_sentinel_fallback(stack, model_name: str) -> None:
    """Compare FE typed 529 to the EPP third-waiter outcome."""
    from tests.router.epp_parity import PRIORITY_HOLD_DEADLINE_S

    case = frontend_epp_priority_case(model_name)
    low_nvext, high_nvext, low_headers, high_headers = _priority_inputs("body_strict")
    catalog = stack.catalog_snapshot()
    if catalog != stack.worker_ids:
        raise AssertionError(
            f"catalog changed before hold: {catalog} vs {stack.worker_ids}"
        )
    blocker_id = _priority_request_id()
    low_id = _priority_request_id()
    high_id = _priority_request_id()
    sentinel_id = _priority_request_id()
    low_payload = _with_nvext(case["payload"], **low_nvext)
    high_payload = _with_nvext(case["payload"], **high_nvext)
    deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
    stack.pause_owned_prefill(deadline)
    fe_error = None
    try:
        _assert_priority_catalog(stack, catalog, "after SIGSTOP")
        boundary = _require_no_prior_selection(stack, case_name="sentinel")
        blocker_thread, blocker = _start_frontend_generation(
            stack,
            case["payload"],
            path=case["path"],
            request_id=blocker_id,
            headers=None,
            deadline=deadline,
            label="sentinel-blocker",
        )
        blocker_internal, _booking = _wait_prefill_booking(
            stack,
            boundary=boundary,
            deadline=deadline,
            expected_prefill=catalog["prefill"],
            blocker_thread=blocker_thread,
            blocker_out=blocker,
            case_name="sentinel",
        )
        (
            low_thread,
            low_out,
            high_thread,
            high_out,
            fe_sentinel,
        ) = _prove_fe_priority_capacity(
            stack,
            case,
            catalog=catalog,
            deadline=deadline,
            case_name="sentinel",
            blocker_id=blocker_internal,
            boundary=boundary,
            expected_prefill=catalog["prefill"],
            low_id=low_id,
            high_id=high_id,
            sentinel_id=sentinel_id,
            low_payload=low_payload,
            high_payload=high_payload,
            low_headers=low_headers,
            high_headers=high_headers,
            low_label="sentinel-low",
            high_label="sentinel-high",
        )
        stack.resume_owned_prefill()
        join_deadline = time.monotonic() + PRIORITY_HOLD_DEADLINE_S
        leftover = join_deadline - time.monotonic()
        for thread in (blocker_thread, low_thread, high_thread):
            thread.join(timeout=max(leftover, 0.0))
            leftover = join_deadline - time.monotonic()
        _require_priority_threads_finished(
            (
                ("blocker", blocker_thread),
                ("low", low_thread),
                ("high", high_thread),
            ),
            case_name="sentinel",
        )
        _require_fe_generation_internal_ids(
            (
                ("blocker", blocker),
                ("low", low_out),
                ("high", high_out),
            ),
            booked_blocker_id=blocker_internal,
            case_name="sentinel",
        )
    except BaseException as exc:
        fe_error = exc
    _finish_fe_priority_hold(stack, fe_error)

    epp_low = None
    epp_high = None
    epp_blocker = None
    epp_sentinel = None
    body_error = None
    close_error = None
    try:
        epp_blocker = stack.driver.open_held_stream(
            case["payload"],
            path=case["path"],
            request_id=_priority_request_id(),
        )
        blocker_obs = epp_blocker.wait_routing_mutation(
            timeout_s=PRIORITY_HOLD_DEADLINE_S
        )
        if (
            blocker_obs.routing_mode != "disaggregated"
            or blocker_obs.prefill_worker_id != catalog["prefill"]
        ):
            raise AssertionError(
                "sentinel EPP blocker is not a held prefill reservation: "
                f"{blocker_obs}"
            )
        _assert_priority_catalog(stack, catalog, "after EPP prefill reservation")
        (
            epp_low,
            epp_high,
            epp_sentinel,
            sentinel_kind,
            sentinel_outcome,
        ) = _prove_epp_priority_capacity(
            stack,
            case,
            catalog=catalog,
            case_name="sentinel",
            blocker_obs=blocker_obs,
            low_id=low_id,
            high_id=high_id,
            sentinel_id=sentinel_id,
            low_payload=low_payload,
            high_payload=high_payload,
            low_headers=low_headers,
            high_headers=high_headers,
        )
        _assert_priority_catalog(stack, catalog, "after EPP admission")
        if stack.kube_api.failures:
            raise AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(stack.kube_api.failures)
            )
        if sentinel_kind == "aggregated":
            from tests.router.epp_parity import _typed_fe_queue_rejection

            _typed_fe_queue_rejection(fe_sentinel)
            raise PrefillQueueRejectionParityMismatch(
                "FE typed 529 versus EPP aggregated fallback: FE HTTP "
                f"{fe_sentinel.status_code} with {fe_sentinel.text[:200]!r}; "
                f"EPP {sentinel_outcome.routing_mode!r}"
            )
        if sentinel_kind != "typed":
            raise AssertionError(
                "FE typed 529 versus unexpected EPP sentinel: "
                f"{sentinel_kind!r} {sentinel_outcome!r}"
            )
    except BaseException as exc:
        body_error = exc
    finally:
        try:
            _close_priority_held_streams(epp_blocker, epp_low, epp_high, epp_sentinel)
        except BaseException as exc:
            close_error = exc
    _raise_priority_cleanup(body_error, close_error)


def run_native_consumer_profile(request, tmp_path, model_name, profile_name):
    """Thin bridge: topology routing stack + native provider overlays + helper.

    Scenario logic stays in ``epp_native_consumers``. Selected-test imports keep
    ordinary ``common`` collection free of EPP/grpc_tools.
    """
    from tests.router.epp_native_consumers import (
        native_provider_config_overrides,
        native_provider_env_overrides,
    )
    from tests.router.epp_native_consumers import (
        run_native_consumer_profile as _run_native_consumer_profile,
    )
    from tests.router.epp_parity import EppFrontendRoutingParityStack

    with EppFrontendRoutingParityStack(
        request,
        tmp_path,
        model_name,
        enable_worker_topology=True,
        provider_env_overrides=native_provider_env_overrides(),
        provider_config_overrides=native_provider_config_overrides(),
    ) as stack:
        return _run_native_consumer_profile(
            stack, profile_name=profile_name, tmp_path=tmp_path
        )


def run_named_policy_catalog(request, tmp_path, model_name):
    """Thin bridge: default provider, topology stack, named-policy catalog helper.

    Scenario logic stays in ``epp_selector_catalog``. Selected-test imports keep
    ordinary ``common`` collection free of EPP/grpc_tools.
    """
    from tests.router.epp_parity import EppFrontendRoutingParityStack
    from tests.router.epp_selector_catalog import assert_named_policy_source_contract
    from tests.router.epp_selector_catalog import (
        run_named_policy_catalog as _run_named_policy_catalog,
    )

    assert_named_policy_source_contract()
    with EppFrontendRoutingParityStack(
        request,
        tmp_path,
        model_name,
        enable_worker_topology=True,
    ) as stack:
        return _run_named_policy_catalog(stack, tmp_path=tmp_path)
