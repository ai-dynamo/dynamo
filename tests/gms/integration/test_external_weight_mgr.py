# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Callable, Protocol

import pytest
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.server.fsm import ServerState

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

from ..harness.gms import GMSServerProcess
from ..harness.runtime import send_completion
from ..harness.sglang import SGLangWithGMSProcess
from ..harness.vllm import VLLMWithGMSProcess

# guard: external_weight_writer imports torch at module level
pytest.importorskip("torch", reason="torch is required")
from ..harness.external_weight_writer import run_external_weight_writer  # noqa: E402

pytestmark = [pytest.mark.nightly]

# Event flow under test:
# 1. The engine starts in read-only mode and waits for a committed weights layout.
# 2. An external writer acquires RW on the weights GMS, loads dummy weights, commits, and exits.
# 3. The engine comes online with those committed weights while owning its own KV-cache RW layout.
# 4. The engine sleeps, preserving its weight VAs but dropping the KV-cache layout.
# 5. A second external writer acquires RW on the weights GMS, creates a fresh committed layout with
#    different allocation IDs but the same structural layout, and exits.
# 6. The engine wakes, remaps the preserved weight VAs into the new committed layout, recreates its
#    KV cache in a new RW layout, and serves inference without a stale-layout error.


class _SleepWakeEngine(Protocol):
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    def sleep(self) -> dict:
        ...

    def wake(self) -> dict:
        ...


def _list_committed_weight_allocations(
    socket_path: str,
) -> list[tuple[int, str, int, int, str]]:
    with _GMSClientSession(
        socket_path, RequestedLockType.RO, timeout_ms=None
    ) as reader:
        return [
            (
                int(info.layout_slot),
                str(info.allocation_id),
                int(info.size),
                int(info.aligned_size),
                str(info.tag),
            )
            for info in reader.list_allocations()
        ]


def _run_external_weight_mgr_test(
    request,
    ports: dict,
    backend: str,
    make_engine: Callable[[], _SleepWakeEngine],
) -> None:
    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, tag="weights")
        )
        kv_cache_gms = stack.enter_context(
            GMSServerProcess(request, device=0, tag="kv_cache")
        )
        stack.enter_context(
            DynamoFrontendProcess(request, frontend_port=ports["frontend"])
        )

        engine = make_engine()
        stack.callback(engine.__exit__, None, None, None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            start_future = executor.submit(engine.__enter__)
            try:
                # The read-only engine must stall until some external writer
                # publishes the first committed weights layout.
                time.sleep(2.0)
                assert (
                    not start_future.done()
                ), "read-only engine should still be waiting for committed weights"
                assert weights_gms.get_runtime_state().state == ServerState.EMPTY
                assert kv_cache_gms.get_runtime_state().state == ServerState.EMPTY

                # Publish the first weights layout out-of-process, then let the
                # engine finish importing those committed weights.
                run_external_weight_writer(backend)
                start_future.result(timeout=300)

                first_weights_state = weights_gms.get_runtime_state()
                first_kv_state = kv_cache_gms.get_runtime_state()
                first_allocations = _list_committed_weight_allocations(
                    weights_gms.socket_path
                )

                assert first_weights_state.state == ServerState.RO
                assert first_weights_state.allocation_count > 0
                assert first_weights_state.memory_layout_hash
                assert first_kv_state.state == ServerState.RW
                assert first_allocations

                result = send_completion(ports["frontend"])
                assert result["choices"]

                # Sleep preserves the engine's weight VAs but tears down the KV
                # cache layout so the process can wake against a later weights layout.
                assert engine.sleep()["status"] == "ok"

                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_sleep = weights_gms.get_runtime_state()
                    kv_after_sleep = kv_cache_gms.get_runtime_state()
                    if (
                        weights_after_sleep.state == ServerState.COMMITTED
                        and weights_after_sleep.memory_layout_hash
                        == first_weights_state.memory_layout_hash
                        and weights_after_sleep.ro_session_count == 0
                        and kv_after_sleep.state == ServerState.EMPTY
                        and kv_after_sleep.allocation_count == 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "engine sleep did not settle the expected GMS state"
                        )
                    time.sleep(0.1)

                # Publish a second committed weights layout with the same logical
                # layout but fresh allocation IDs.
                run_external_weight_writer(backend)

                deadline = time.monotonic() + 30.0
                while True:
                    second_weights_state = weights_gms.get_runtime_state()
                    second_kv_state = kv_cache_gms.get_runtime_state()
                    if (
                        second_weights_state.state == ServerState.COMMITTED
                        and second_weights_state.memory_layout_hash
                        == first_weights_state.memory_layout_hash
                        and second_weights_state.allocation_count
                        == first_weights_state.allocation_count
                        and second_kv_state.state == ServerState.EMPTY
                        and second_kv_state.allocation_count == 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "external writer did not publish a new committed weights layout"
                        )
                    time.sleep(0.1)

                # The second publish must reuse the same layout slots and sizes so
                # RO remap can bind the preserved VAs to new allocations.
                second_allocations = _list_committed_weight_allocations(
                    weights_gms.socket_path
                )
                assert len(second_allocations) == len(first_allocations)
                assert [item[0] for item in second_allocations] == [
                    item[0] for item in first_allocations
                ]
                assert [item[2:] for item in second_allocations] == [
                    item[2:] for item in first_allocations
                ]
                assert [item[1] for item in second_allocations] != [
                    item[1] for item in first_allocations
                ]

                # The weights GMS should show the expected publish progression:
                # first publish, old layout cleanup, second publish.
                weights_events = weights_gms.get_event_history().events
                assert [event.kind for event in weights_events] == [
                    "rw_connected",
                    "committed",
                    "allocations_cleared",
                    "rw_connected",
                    "committed",
                ]
                assert (
                    weights_events[2].allocation_count
                    == first_weights_state.allocation_count
                )

                # Wake should remap the preserved RO weight VAs into the new
                # committed layout and recreate KV cache in a new RW layout.
                assert engine.wake()["status"] == "ok"

                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_wake = weights_gms.get_runtime_state()
                    kv_after_wake = kv_cache_gms.get_runtime_state()
                    if (
                        weights_after_wake.state == ServerState.RO
                        and weights_after_wake.memory_layout_hash
                        == first_weights_state.memory_layout_hash
                        and kv_after_wake.state == ServerState.RW
                        and kv_after_wake.allocation_count > 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "engine wake did not restore the expected GMS state"
                        )
                    time.sleep(0.1)

                # A normal inference after wake proves the remapped weights and
                # recreated KV cache are usable end to end.
                result = send_completion(ports["frontend"], "updated weights")
                assert result["choices"]
            finally:
                if start_future.done():
                    try:
                        start_future.result()
                    except Exception:
                        pass


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_external_weight_mgr_vllm(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    ports = gms_ports
    _run_external_weight_mgr_test(
        request,
        ports,
        "vllm",
        make_engine=lambda: VLLMWithGMSProcess(
            request,
            "engine",
            ports["shadow_system"],
            ports["shadow_kv_event"],
            ports["shadow_nixl"],
            ports["frontend"],
            read_only_weights=True,
        ),
    )


@pytest.mark.skip(reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2567")
@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_external_weight_mgr_sglang(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    ports = gms_ports
    _run_external_weight_mgr_test(
        request,
        ports,
        "sglang",
        make_engine=lambda: SGLangWithGMSProcess(
            request,
            "engine",
            ports["shadow_system"],
            ports["shadow_sglang"],
            ports["frontend"],
            read_only_weights=True,
        ),
    )
