# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Callable

import pytest
from gpu_memory_service.server.fsm import ServerState

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

from ..harness.gms import ThreadedGMSServer
from ..harness.runtime import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    get_gpu_memory_used,
    send_completion,
)
from ..harness.sglang import SGLangWithGMSProcess
from ..harness.vllm import VLLMWithGMSProcess

pytestmark = [pytest.mark.nightly]

# Event flow under test:
# 1. Shadow A starts with committed weights and a live RW KV layout, then sleeps.
# 2. Shadow B starts from the same committed weights layout, then sleeps as well.
# 3. Primary wakes and owns the next RW KV layout.
# 4. Shadow A wakes after a forced primary disconnect and enters a new RW layout.
# 5. Shadow A blocks on allocation_oom until the still-alive primary is killed.
# 6. After primary death, the old KV layout clears and Shadow A finishes wake.

logger = logging.getLogger(__name__)


def _kill_process_group(process: ManagedProcess) -> None:
    pid = process.get_pid()
    if pid is None:
        logger.warning("kill process group: no PID available")
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        logger.warning("kill process group: process %d already dead", pid)
        return

    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _is_process_alive(process: ManagedProcess) -> bool:
    pid = process.get_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_weights_published_once(events) -> None:
    assert [event.kind for event in events] == ["rw_connected", "committed"]


def _assert_cleared_rw_layout_prefix(events, cleared_layouts: int) -> None:
    expected_prefix = [
        "rw_connected",
        "rw_aborted",
        "allocations_cleared",
    ] * cleared_layouts
    assert [event.kind for event in events[: len(expected_prefix)]] == expected_prefix
    clear_counts = [
        event.allocation_count
        for event in events
        if event.kind == "allocations_cleared"
    ]
    assert len(clear_counts) >= cleared_layouts
    assert all(count > 0 for count in clear_counts[:cleared_layouts])


def _sleep_shadow(
    frontend_port: int,
    weights_gms: ThreadedGMSServer,
    kv_cache_gms: ThreadedGMSServer,
    shadow: ManagedProcess,
    expected_weights_hash: str | None = None,
) -> tuple[str, int, int]:
    result = send_completion(frontend_port)
    assert result["choices"], "Shadow inference failed"
    logger.info("Shadow inference OK: %s", result)

    deadline = time.monotonic() + 30.0
    while True:
        weights_state = weights_gms.get_runtime_state()
        kv_state = kv_cache_gms.get_runtime_state()
        if (
            weights_state.state == ServerState.RO
            and weights_state.allocation_count > 0
            and weights_state.memory_layout_hash
            and kv_state.state == ServerState.RW
            and kv_state.allocation_count > 0
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("shadow startup did not stabilize GMS state")
        time.sleep(0.1)

    if expected_weights_hash is not None:
        assert weights_state.memory_layout_hash == expected_weights_hash

    shadow_memory_before_sleep = get_gpu_memory_used()
    assert shadow.sleep()["status"] == "ok"
    shadow_memory_after_sleep = get_gpu_memory_used()
    shadow_released_bytes = shadow_memory_before_sleep - shadow_memory_after_sleep
    logger.info(
        "Shadow sleep: %.2f -> %.2f GiB (freed %.0f MB)",
        shadow_memory_before_sleep / (1 << 30),
        shadow_memory_after_sleep / (1 << 30),
        shadow_released_bytes / (1 << 20),
    )
    assert shadow_memory_after_sleep < shadow_memory_before_sleep
    assert shadow_released_bytes > 0

    deadline = time.monotonic() + 30.0
    while True:
        weights_after_sleep = weights_gms.get_runtime_state()
        kv_after_sleep = kv_cache_gms.get_runtime_state()
        if (
            weights_after_sleep.state == ServerState.COMMITTED
            and weights_after_sleep.allocation_count == weights_state.allocation_count
            and weights_after_sleep.memory_layout_hash
            == weights_state.memory_layout_hash
            and kv_after_sleep.state == ServerState.EMPTY
            and kv_after_sleep.allocation_count == 0
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("shadow sleep did not clear GMS state")
        time.sleep(0.1)

    return (
        weights_state.memory_layout_hash,
        shadow_released_bytes,
        shadow_memory_after_sleep,
    )


def _run_shadow_failover_test(
    request,
    ports: dict,
    make_shadow_a: Callable[[], ManagedProcess],
    make_shadow_b: Callable[[], ManagedProcess],
    make_primary: Callable[[], ManagedProcess],
) -> None:
    frontend_port = ports["frontend"]

    with ExitStack() as stack:
        weights_gms = stack.enter_context(ThreadedGMSServer(device=0, tag="weights"))
        kv_cache_gms = stack.enter_context(ThreadedGMSServer(device=0, tag="kv_cache"))
        stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=frontend_port,
                display_name="frontend",
            )
        )
        with make_shadow_a() as shadow_a:
            (
                weights_hash,
                shadow_a_released_bytes,
                _shadow_a_memory_after_sleep,
            ) = _sleep_shadow(frontend_port, weights_gms, kv_cache_gms, shadow_a)
            with make_shadow_b() as shadow_b:
                (
                    sleeping_weights_hash,
                    _shadow_b_released_bytes,
                    sleeping_memory_after_sleep,
                ) = _sleep_shadow(
                    frontend_port,
                    weights_gms,
                    kv_cache_gms,
                    shadow_b,
                    expected_weights_hash=weights_hash,
                )
                assert sleeping_weights_hash == weights_hash

                weights_events_after_shadow_sleep = (
                    weights_gms.get_event_history().events
                )
                _assert_weights_published_once(weights_events_after_shadow_sleep)

                kv_events_after_shadow_sleep = kv_cache_gms.get_event_history().events
                _assert_cleared_rw_layout_prefix(kv_events_after_shadow_sleep, 2)

                with make_primary() as primary:
                    result = send_completion(frontend_port, "Primary test")
                    assert result["choices"], "Primary inference failed"
                    logger.info("Primary inference OK: %s", result)

                    primary_memory_in_use = get_gpu_memory_used()
                    logger.info(
                        "Primary active memory: %.2f GiB",
                        primary_memory_in_use / (1 << 30),
                    )
                    assert primary_memory_in_use > sleeping_memory_after_sleep
                    assert (
                        (primary_memory_in_use - sleeping_memory_after_sleep)
                        >= shadow_a_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION
                    )

                    deadline = time.monotonic() + 30.0
                    while True:
                        weights_with_primary = weights_gms.get_runtime_state()
                        kv_with_primary = kv_cache_gms.get_runtime_state()
                        if (
                            weights_with_primary.state == ServerState.RO
                            and weights_with_primary.ro_session_count >= 1
                            and weights_with_primary.allocation_count > 0
                            and weights_with_primary.memory_layout_hash == weights_hash
                            and kv_with_primary.state == ServerState.RW
                            and kv_with_primary.allocation_count > 0
                        ):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "primary did not acquire KV cache GMS state"
                            )
                        time.sleep(0.1)
                    expected_kv_kinds_before_disconnect = [
                        "rw_connected",
                        "rw_aborted",
                        "allocations_cleared",
                        "rw_connected",
                        "rw_aborted",
                        "allocations_cleared",
                        "rw_connected",
                    ]
                    assert [
                        event.kind for event in kv_cache_gms.get_event_history().events
                    ] == expected_kv_kinds_before_disconnect

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        # Shadow A wakes while Shadow B remains asleep. After we
                        # force-disconnect the primary from GMS, Shadow A should enter
                        # a new RW layout but block on real CUDA OOM until the primary dies.
                        wake_future = executor.submit(shadow_a.wake, 180)
                        deadline = time.monotonic() + 10.0
                        while time.monotonic() < deadline:
                            if wake_future.done():
                                break
                            time.sleep(0.2)
                        assert not wake_future.done(), (
                            "Shadow wake completed before the primary died; "
                            "KV cache RW handoff did not block as expected"
                        )
                        kv_while_blocked = kv_cache_gms.get_runtime_state()
                        assert kv_while_blocked.state == ServerState.RW
                        assert kv_while_blocked.allocation_count > 0

                        kv_cache_gms.disconnect_rw_session()

                        expected_kv_kinds_while_blocked = (
                            expected_kv_kinds_before_disconnect
                            + [
                                "rw_aborted",
                                "allocations_cleared",
                                "rw_connected",
                                "allocation_oom",
                            ]
                        )
                        blocked_allocation_count: int | None = None
                        deadline = time.monotonic() + 30.0
                        while time.monotonic() < deadline:
                            kv_after_forced_disconnect = (
                                kv_cache_gms.get_runtime_state()
                            )
                            kv_events_after_forced_disconnect = (
                                kv_cache_gms.get_event_history().events
                            )
                            if (
                                kv_after_forced_disconnect.state == ServerState.RW
                                and [
                                    event.kind
                                    for event in kv_events_after_forced_disconnect
                                ]
                                == expected_kv_kinds_while_blocked
                                and not wake_future.done()
                            ):
                                blocked_allocation_count = (
                                    kv_after_forced_disconnect.allocation_count
                                )
                                if (
                                    blocked_allocation_count
                                    < kv_while_blocked.allocation_count
                                    and blocked_allocation_count
                                    == kv_events_after_forced_disconnect[
                                        -1
                                    ].allocation_count
                                ):
                                    break
                            time.sleep(0.2)
                        else:
                            raise TimeoutError(
                                "shadow never entered a new KV-cache layout blocked on allocation"
                            )

                        assert blocked_allocation_count is not None
                        linger_deadline = time.monotonic() + 3.0
                        while time.monotonic() < linger_deadline:
                            kv_while_lingering = kv_cache_gms.get_runtime_state()
                            kv_events_while_lingering = (
                                kv_cache_gms.get_event_history().events
                            )
                            assert kv_while_lingering.state == ServerState.RW
                            assert (
                                kv_while_lingering.allocation_count
                                == blocked_allocation_count
                            )
                            assert [
                                event.kind for event in kv_events_while_lingering
                            ] == expected_kv_kinds_while_blocked
                            assert _is_process_alive(
                                primary
                            ), "primary died before the linger window completed"
                            assert (
                                not wake_future.done()
                            ), "shadow wake completed while the primary was still alive"
                            time.sleep(0.2)

                        primary_memory_before_kill = get_gpu_memory_used()
                        _kill_process_group(primary)
                        primary_memory_after_kill = get_gpu_memory_used()
                        logger.info(
                            "Primary kill snapshot: %.2f -> %.2f GiB",
                            primary_memory_before_kill / (1 << 30),
                            primary_memory_after_kill / (1 << 30),
                        )

                        deadline = time.monotonic() + 30.0
                        while time.monotonic() < deadline:
                            kv_after_primary_kill = kv_cache_gms.get_runtime_state()
                            if (
                                kv_after_primary_kill.state == ServerState.RW
                                and kv_after_primary_kill.allocation_count > 0
                            ):
                                break
                            time.sleep(0.2)
                        else:
                            raise TimeoutError(
                                "shadow did not reacquire KV cache after failover"
                            )

                        wake_result = wake_future.result(timeout=180)

                assert wake_result["status"] == "ok"
                shadow_memory_after_wake = get_gpu_memory_used()
                shadow_reacquired_bytes = (
                    shadow_memory_after_wake - sleeping_memory_after_sleep
                )
                logger.info(
                    "Shadow wake memory: %.2f GiB (reacquired %.0f MB)",
                    shadow_memory_after_wake / (1 << 30),
                    shadow_reacquired_bytes / (1 << 20),
                )
                assert shadow_memory_after_wake > sleeping_memory_after_sleep
                assert (
                    shadow_reacquired_bytes
                ) >= shadow_a_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

                # Once the primary is gone, the failover shadow should finish wake
                # with the same committed weights layout and a new live RW KV-cache layout.
                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_wake = weights_gms.get_runtime_state()
                    kv_after_wake = kv_cache_gms.get_runtime_state()
                    if (
                        weights_after_wake.state == ServerState.RO
                        and weights_after_wake.ro_session_count >= 1
                        and weights_after_wake.allocation_count > 0
                        and weights_after_wake.memory_layout_hash
                        == weights_with_primary.memory_layout_hash
                        and kv_after_wake.state == ServerState.RW
                        and kv_after_wake.allocation_count > 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "shadow wake did not restore the expected GMS state"
                        )
                    time.sleep(0.1)

                # The final KV history should show the full handoff:
                # shadow A slept -> shadow B slept -> primary layout ->
                # primary abort/clear -> shadow A reconnects -> shadow A sees OOM.
                weights_events_after_wake = weights_gms.get_event_history().events
                _assert_weights_published_once(weights_events_after_wake)

                kv_events_after_wake = kv_cache_gms.get_event_history().events
                _assert_cleared_rw_layout_prefix(kv_events_after_wake, 3)
                assert [event.kind for event in kv_events_after_wake] == [
                    "rw_connected",
                    "rw_aborted",
                    "allocations_cleared",
                    "rw_connected",
                    "rw_aborted",
                    "allocations_cleared",
                    "rw_connected",
                    "rw_aborted",
                    "allocations_cleared",
                    "rw_connected",
                    "allocation_oom",
                ]

                result = send_completion(frontend_port, "Post failover")
                assert result["choices"], "Shadow inference after failover failed"
                logger.info("Shadow inference after failover OK: %s", result)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover_vllm(
    request, runtime_services_dynamic_ports, gms_ports, predownload_models
):
    ports = gms_ports
    _run_shadow_failover_test(
        request,
        ports,
        make_shadow_a=lambda: VLLMWithGMSProcess(
            request,
            "shadow-a",
            ports["shadow_system"],
            ports["shadow_kv_event"],
            ports["shadow_nixl"],
            ports["frontend"],
        ),
        make_shadow_b=lambda: VLLMWithGMSProcess(
            request,
            "shadow-b",
            ports["shadow2_system"],
            ports["shadow2_kv_event"],
            ports["shadow2_nixl"],
            ports["frontend"],
        ),
        make_primary=lambda: VLLMWithGMSProcess(
            request,
            "primary",
            ports["primary_system"],
            ports["primary_kv_event"],
            ports["primary_nixl"],
            ports["frontend"],
        ),
    )


@pytest.mark.skip(reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2567")
@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover_sglang(
    request, runtime_services_dynamic_ports, gms_ports, predownload_models
):
    ports = gms_ports
    _run_shadow_failover_test(
        request,
        ports,
        make_shadow_a=lambda: SGLangWithGMSProcess(
            request,
            "shadow-a",
            ports["shadow_system"],
            ports["shadow_sglang"],
            ports["frontend"],
        ),
        make_shadow_b=lambda: SGLangWithGMSProcess(
            request,
            "shadow-b",
            ports["shadow2_system"],
            ports["shadow2_sglang"],
            ports["frontend"],
        ),
        make_primary=lambda: SGLangWithGMSProcess(
            request,
            "primary",
            ports["primary_system"],
            ports["primary_sglang"],
            ports["frontend"],
        ),
    )
