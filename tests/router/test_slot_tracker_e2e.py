# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for the current router-owned slot-tracker lifecycle.

IMPORTANT TEST CONTRACT:
- These tests validate the current router-owned Add, MarkPrefillCompleted, and
  Free lifecycle.
- Zero-buffer polling controls when the router consumes worker responses; it
  does not pause mocker execution.
- These tests make no claim about mocker-owned load or cleanup timing.
- If https://github.com/ai-dynamo/dynamo/issues/10511 moves prefill-complete or
  Free ownership to engine-published acknowledgements/events, these tests will
  likely require redesign.
- Under engine-owned lifecycle events, response_buffer_size=0 cannot delay the
  mocker from publishing completion or Free.
- Do not preserve these assumptions later with sleeps or artificial mocker
  slowdown.
"""

import asyncio
import gc
import itertools
from collections.abc import AsyncIterator
from typing import Any

import pytest

from dynamo.llm import KvRouter, KvRouterConfig
from tests.router.common import _create_kv_router_with_timeout
from tests.router.helper import generate_random_suffix, get_runtime
from tests.router.mocker_process import MockerProcess, launch_disagg_workers
from tests.utils.constants import ROUTER_MODEL_NAME

MODEL_NAME = ROUTER_MODEL_NAME
BLOCK_SIZE = 16
PROBE_TOKENS = list(range(100, 100 + BLOCK_SIZE * 4))
LOAD_TIMEOUT_S = 15.0
STREAM_TIMEOUT_S = 30.0
POLL_INTERVAL_S = 0.05

LoadKey = tuple[int, int]
LoadValue = tuple[int, int]
LoadSnapshot = dict[LoadKey, LoadValue]

_request_counter = itertools.count(1)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_NAME),
    pytest.mark.timeout(180),
]


def _request_tokens() -> list[int]:
    base = 1_000 + next(_request_counter) * 100
    return list(range(base, base + BLOCK_SIZE * 4))


def _router_config(**overrides: Any) -> KvRouterConfig:
    values = {
        "use_kv_events": False,
        "router_assume_kv_reuse": False,
        "router_queue_threshold": None,
    }
    values.update(overrides)
    return KvRouterConfig(**values)


def _create_router(
    engine_workers,
    router_config: KvRouterConfig,
    discovery_backend: str,
    request_plane: str,
):
    runtime = get_runtime(discovery_backend, request_plane)
    endpoint = runtime.endpoint(
        f"{engine_workers.namespace}.{engine_workers.component_name}.generate"
    )
    router = _create_kv_router_with_timeout(
        router_factory=lambda: KvRouter(
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            kv_router_config=router_config,
        ),
        num_workers=engine_workers.num_workers,
        engine_workers=engine_workers,
    )
    return runtime, endpoint, router


async def _snapshot(router: KvRouter) -> LoadSnapshot:
    rows = await router.get_potential_loads(PROBE_TOKENS)
    snapshot = {
        (row["worker_id"], row["dp_rank"]): (
            row["potential_prefill_tokens"],
            row["potential_decode_blocks"],
        )
        for row in rows
    }
    assert len(snapshot) == len(rows), f"Duplicate worker load rows: {rows}"
    assert snapshot, "Expected at least one worker load row"
    return snapshot


def _single_delta(current: LoadSnapshot, baseline: LoadSnapshot) -> LoadValue:
    assert (
        current.keys() == baseline.keys()
    ), f"Worker load rows changed: baseline={baseline}, current={current}"
    assert len(current) == 1, f"Expected one worker/DP rank, got {current}"
    key = next(iter(current))
    current_prefill, current_decode = current[key]
    baseline_prefill, baseline_decode = baseline[key]
    return current_prefill - baseline_prefill, current_decode - baseline_decode


def _matches_phase(delta: LoadValue, prefill_active: bool, decode_active: bool) -> bool:
    prefill, decode = delta
    prefill_matches = prefill > 0 if prefill_active else prefill == 0
    decode_matches = decode > 0 if decode_active else decode == 0
    return prefill_matches and decode_matches


async def _wait_for_phase(
    router: KvRouter,
    baseline: LoadSnapshot,
    *,
    prefill_active: bool,
    decode_active: bool,
    description: str,
    timeout_s: float = LOAD_TIMEOUT_S,
) -> LoadValue:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    last_delta: LoadValue | None = None

    while loop.time() < deadline:
        last_delta = _single_delta(await _snapshot(router), baseline)
        if _matches_phase(last_delta, prefill_active, decode_active):
            return last_delta
        await asyncio.sleep(POLL_INTERVAL_S)

    raise AssertionError(
        f"Timed out waiting for {description}; last load delta={last_delta}"
    )


async def _open_request(
    router: KvRouter, *, max_tokens: int = 8
) -> AsyncIterator[dict[str, Any]]:
    return await router.generate(
        token_ids=_request_tokens(),
        model=MODEL_NAME,
        stop_conditions={"ignore_eos": True, "max_tokens": max_tokens},
        response_buffer_size=0,
    )


async def _next_nonempty_token(stream: AsyncIterator[dict[str, Any]]) -> dict[str, Any]:
    async def poll() -> dict[str, Any]:
        async for response in stream:
            if isinstance(response, dict) and response.get("token_ids"):
                return response
        raise AssertionError("Stream ended before returning a non-empty token")

    return await asyncio.wait_for(poll(), timeout=STREAM_TIMEOUT_S)


async def _drain(stream: AsyncIterator[dict[str, Any]]) -> None:
    async def drain() -> None:
        async for _ in stream:
            pass

    await asyncio.wait_for(drain(), timeout=STREAM_TIMEOUT_S)


async def _poll_through_terminal(stream: AsyncIterator[dict[str, Any]]) -> None:
    async def poll() -> None:
        async for response in stream:
            if isinstance(response, dict) and response.get("finish_reason") is not None:
                return
        raise AssertionError("Stream ended before returning a terminal response")

    await asyncio.wait_for(poll(), timeout=STREAM_TIMEOUT_S)


async def _establish_replica_readiness(
    source: KvRouter,
    observer: KvRouter,
    source_baseline: LoadSnapshot,
    observer_baseline: LoadSnapshot,
) -> None:
    for _ in range(5):
        stream = await _open_request(source, max_tokens=2)
        observed_add = False
        try:
            await _wait_for_phase(
                observer,
                observer_baseline,
                prefill_active=True,
                decode_active=True,
                description="replicated sacrificial Add",
                timeout_s=2.0,
            )
            observed_add = True
        except AssertionError:
            pass
        finally:
            del stream
            gc.collect()
            await _wait_for_phase(
                source,
                source_baseline,
                prefill_active=False,
                decode_active=False,
                description="sacrificial source Free",
            )
            await _wait_for_phase(
                observer,
                observer_baseline,
                prefill_active=False,
                decode_active=False,
                description="replicated sacrificial Free",
            )

        if observed_add:
            return

    raise AssertionError("Replica subscription did not observe a sacrificial Add")


async def _assert_replicated_lifecycle(
    source: KvRouter,
    observer: KvRouter,
    source_baseline: LoadSnapshot,
    observer_baseline: LoadSnapshot,
) -> None:
    stream = await _open_request(source)
    try:
        for router, baseline, role in (
            (source, source_baseline, "source"),
            (observer, observer_baseline, "observer"),
        ):
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=True,
                decode_active=True,
                description=f"{role} Add",
            )

        await _next_nonempty_token(stream)
        for router, baseline, role in (
            (source, source_baseline, "source"),
            (observer, observer_baseline, "observer"),
        ):
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=False,
                decode_active=True,
                description=f"{role} MarkPrefillCompleted",
            )
    finally:
        del stream
        gc.collect()

    for router, baseline, role in (
        (source, source_baseline, "source"),
        (observer, observer_baseline, "observer"),
    ):
        await _wait_for_phase(
            router,
            baseline,
            prefill_active=False,
            decode_active=False,
            description=f"{role} Free",
        )


@pytest.fixture
def aggregated_mocker(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    discovery_backend,
    request_plane,
    durable_kv_events,
):
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    with MockerProcess(
        request,
        mocker_args={
            "speedup_ratio": 10.0,
            "block_size": BLOCK_SIZE,
            "durable_kv_events": durable_kv_events,
        },
        num_mockers=1,
        store_backend=discovery_backend,
        request_plane=request_plane,
    ) as mocker:
        yield mocker


@pytest.fixture
def disagg_mockers(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    discovery_backend,
    request_plane,
    durable_kv_events,
):
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    namespace = f"slot-tracker-{generate_random_suffix()}"
    mocker_args = {
        "speedup_ratio": 10.0,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": durable_kv_events,
    }
    with launch_disagg_workers(
        request,
        namespace=namespace,
        registration_order="prefill_first",
        prefill_mocker_args=mocker_args,
        decode_mocker_args=mocker_args,
        num_prefill_mockers=1,
        num_decode_mockers=1,
        enable_disagg_bootstrap=False,
        store_backend=discovery_backend,
        request_plane=request_plane,
    ) as workers:
        yield workers


def test_unexpected_drop_and_normal_completion(
    aggregated_mocker,
    discovery_backend,
    request_plane,
) -> None:
    async def run() -> None:
        runtime, endpoint, router = _create_router(
            aggregated_mocker,
            _router_config(),
            discovery_backend,
            request_plane,
        )
        _ = runtime, endpoint
        baseline = await _snapshot(router)
        dropped_stream = None
        completed_stream = None

        try:
            dropped_stream = await _open_request(router)
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=True,
                decode_active=True,
                description="retained unpolled request",
            )
            await asyncio.sleep(0)
            retained_delta = _single_delta(await _snapshot(router), baseline)
            assert _matches_phase(retained_delta, True, True), retained_delta

            dropped_stream = None
            gc.collect()
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=False,
                decode_active=False,
                description="drop cleanup",
            )

            completed_stream = await _open_request(router, max_tokens=2)
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=True,
                decode_active=True,
                description="normally completed request Add",
            )
            await _poll_through_terminal(completed_stream)
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=False,
                decode_active=False,
                description="normal completion cleanup",
            )
            completed_stream = None
        finally:
            del dropped_stream, completed_stream
            gc.collect()

    asyncio.run(run())


def test_router_observed_first_token_marks_prefill_complete(
    aggregated_mocker,
    discovery_backend,
    request_plane,
) -> None:
    async def run() -> None:
        runtime, endpoint, router = _create_router(
            aggregated_mocker,
            _router_config(),
            discovery_backend,
            request_plane,
        )
        _ = runtime, endpoint
        baseline = await _snapshot(router)
        stream = await _open_request(router)

        try:
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=True,
                decode_active=True,
                description="request Add before router polling",
            )
            await _next_nonempty_token(stream)
            await _wait_for_phase(
                router,
                baseline,
                prefill_active=False,
                decode_active=True,
                description="router-observed first-token prefill completion",
            )
        finally:
            del stream
            gc.collect()

        await _wait_for_phase(
            router,
            baseline,
            prefill_active=False,
            decode_active=False,
            description="decode cleanup after first-token drop",
        )

    asyncio.run(run())


def test_bidirectional_replica_lifecycle(
    aggregated_mocker,
    discovery_backend,
    request_plane,
) -> None:
    async def run() -> None:
        config = _router_config(router_replica_sync=True)
        runtime_a, endpoint_a, router_a = _create_router(
            aggregated_mocker,
            config,
            discovery_backend,
            request_plane,
        )
        runtime_b, endpoint_b, router_b = _create_router(
            aggregated_mocker,
            config,
            discovery_backend,
            request_plane,
        )
        _ = runtime_a, endpoint_a, runtime_b, endpoint_b
        baseline_a = await _snapshot(router_a)
        baseline_b = await _snapshot(router_b)

        await _establish_replica_readiness(router_a, router_b, baseline_a, baseline_b)
        await _establish_replica_readiness(router_b, router_a, baseline_b, baseline_a)
        await _assert_replicated_lifecycle(router_a, router_b, baseline_a, baseline_b)
        await _assert_replicated_lifecycle(router_b, router_a, baseline_b, baseline_a)

    asyncio.run(run())


def test_disaggregated_role_attribution(
    disagg_mockers,
    discovery_backend,
    request_plane,
) -> None:
    async def run() -> None:
        prefill_workers, decode_workers = disagg_mockers
        prefill_runtime, prefill_endpoint, prefill_router = _create_router(
            prefill_workers,
            _router_config(router_track_active_blocks=False),
            discovery_backend,
            request_plane,
        )
        decode_runtime, decode_endpoint, decode_router = _create_router(
            decode_workers,
            _router_config(router_track_prefill_tokens=False),
            discovery_backend,
            request_plane,
        )
        _ = (
            prefill_runtime,
            prefill_endpoint,
            decode_runtime,
            decode_endpoint,
        )
        prefill_baseline = await _snapshot(prefill_router)
        decode_baseline = await _snapshot(decode_router)
        prefill_stream = None
        decode_stream = None

        try:
            prefill_stream = await _open_request(prefill_router)
            await _wait_for_phase(
                prefill_router,
                prefill_baseline,
                prefill_active=True,
                decode_active=False,
                description="prefill-only role accounting",
            )
            await _next_nonempty_token(prefill_stream)
            await _wait_for_phase(
                prefill_router,
                prefill_baseline,
                prefill_active=False,
                decode_active=False,
                description="prefill-only completion",
            )
            await _drain(prefill_stream)
            prefill_stream = None

            decode_stream = await _open_request(decode_router)
            await _wait_for_phase(
                decode_router,
                decode_baseline,
                prefill_active=False,
                decode_active=True,
                description="decode-only role accounting",
            )
            decode_stream = None
            gc.collect()
            await _wait_for_phase(
                decode_router,
                decode_baseline,
                prefill_active=False,
                decode_active=False,
                description="decode-only drop cleanup",
            )
        finally:
            del prefill_stream, decode_stream
            gc.collect()

    asyncio.run(run())
