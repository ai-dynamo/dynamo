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
from gpu_memory_service.common.types import ServerState
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

# Event flow under test:
# 1. Shadow A starts with committed weights and a live RW KV epoch, then sleeps.
# 2. Shadow B starts from the same committed weights epoch, then sleeps as well.
# 3. Primary wakes and owns the next RW KV epoch.
# 4. Shadow A wakes after a forced primary disconnect and enters a new RW epoch.
# 5. Shadow A blocks on allocation_oom until the still-alive primary is killed.
# 6. After primary death, the old KV epoch clears and Shadow A finishes wake.

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


def _assert_weights_published_once(events, epoch_id: int) -> None:
    pairs = [(event.kind, event.epoch_id) for event in events]
    connect = ("rw_connected", epoch_id)
    commit = ("committed", epoch_id)
    assert connect in pairs
    assert commit in pairs
    assert pairs.count(connect) == 1
    assert pairs.count(commit) == 1
    assert pairs.index(connect) < pairs.index(commit)


def _assert_cleared_rw_epoch(events, epoch_id: int) -> None:
    pairs = [(event.kind, event.epoch_id) for event in events]
    connect = ("rw_connected", epoch_id)
    abort = ("rw_aborted", epoch_id)
    clear = ("allocations_cleared", epoch_id)
    assert connect in pairs
    assert abort in pairs
    assert clear in pairs
    assert pairs.count(connect) == 1
    assert pairs.count(abort) == 1
    assert pairs.count(clear) == 1
    assert pairs.index(connect) < pairs.index(abort)
    assert pairs.index(abort) < pairs.index(clear)
    assert (
        next(
            event
            for event in events
            if event.kind == "allocations_cleared" and event.epoch_id == epoch_id
        ).allocation_count
        > 0
    )


def _sleep_shadow(
    frontend_port: int,
    weights_gms: ThreadedGMSServer,
    kv_cache_gms: ThreadedGMSServer,
    shadow: ManagedProcess,
    expected_weights_epoch_id: int | None = None,
) -> tuple[int, int, int, int]:
    result = send_completion(frontend_port)
    assert result["choices"], "Shadow inference failed"
    logger.info("Shadow inference OK: %s", result)

    deadline = time.monotonic() + 30.0
    while True:
        weights_state = weights_gms.get_runtime_state()
        kv_state = kv_cache_gms.get_runtime_state()
        if (
            weights_state.committed_epoch_id is not None
            and weights_state.active_rw_epoch_id is None
            and weights_state.allocation_count > 0
            and kv_state.state == ServerState.RW
            and kv_state.active_rw_epoch_id is not None
            and kv_state.committed_epoch_id is None
            and kv_state.allocation_count > 0
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("shadow startup did not stabilize GMS state")
        time.sleep(0.1)

    if expected_weights_epoch_id is not None:
        assert weights_state.committed_epoch_id == expected_weights_epoch_id

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
            and weights_after_sleep.committed_epoch_id
            == weights_state.committed_epoch_id
            and weights_after_sleep.active_rw_epoch_id is None
            and weights_after_sleep.allocation_count == weights_state.allocation_count
            and weights_after_sleep.memory_layout_hash
            == weights_state.memory_layout_hash
            and kv_after_sleep.state == ServerState.EMPTY
            and kv_after_sleep.committed_epoch_id is None
            and kv_after_sleep.active_rw_epoch_id is None
            and kv_after_sleep.allocation_count == 0
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("shadow sleep did not clear GMS state")
        time.sleep(0.1)

    return (
        weights_state.committed_epoch_id,
        kv_state.active_rw_epoch_id,
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
                weights_epoch_id,
                shadow_a_kv_epoch_id,
                shadow_a_released_bytes,
                _shadow_a_memory_after_sleep,
            ) = _sleep_shadow(frontend_port, weights_gms, kv_cache_gms, shadow_a)
            with make_shadow_b() as shadow_b:
                (
                    sleeping_weights_epoch_id,
                    shadow_b_kv_epoch_id,
                    _shadow_b_released_bytes,
                    sleeping_memory_after_sleep,
                ) = _sleep_shadow(
                    frontend_port,
                    weights_gms,
                    kv_cache_gms,
                    shadow_b,
                    expected_weights_epoch_id=weights_epoch_id,
                )
                assert sleeping_weights_epoch_id == weights_epoch_id
                assert shadow_b_kv_epoch_id != shadow_a_kv_epoch_id

                weights_events_after_shadow_sleep = (
                    weights_gms.get_event_history().events
                )
                _assert_weights_published_once(
                    weights_events_after_shadow_sleep, weights_epoch_id
                )

                kv_events_after_shadow_sleep = kv_cache_gms.get_event_history().events
                _assert_cleared_rw_epoch(
                    kv_events_after_shadow_sleep, shadow_a_kv_epoch_id
                )
                _assert_cleared_rw_epoch(
                    kv_events_after_shadow_sleep, shadow_b_kv_epoch_id
                )

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
                            and weights_with_primary.committed_epoch_id
                            == weights_epoch_id
                            and weights_with_primary.active_rw_epoch_id is None
                            and weights_with_primary.allocation_count > 0
                            and kv_with_primary.state == ServerState.RW
                            and kv_with_primary.active_rw_epoch_id is not None
                            and kv_with_primary.committed_epoch_id is None
                            and kv_with_primary.allocation_count > 0
                        ):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "primary did not acquire KV cache GMS state"
                            )
                        time.sleep(0.1)
                    primary_kv_epoch_id = kv_with_primary.active_rw_epoch_id

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        # Shadow A wakes while Shadow B remains asleep. After we
                        # force-disconnect the primary from GMS, Shadow A should enter
                        # a new RW epoch but block on real CUDA OOM until the primary dies.
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
                        assert (
                            kv_while_blocked.active_rw_epoch_id == primary_kv_epoch_id
                        )

                        kv_cache_gms.disconnect_rw_session()

                        deadline = time.monotonic() + 30.0
                        while time.monotonic() < deadline:
                            kv_after_forced_disconnect = (
                                kv_cache_gms.get_runtime_state()
                            )
                            kv_events_after_forced_disconnect = (
                                kv_cache_gms.get_event_history().events
                            )
                            saw_shadow_oom = any(
                                event.kind == "allocation_oom"
                                and event.epoch_id
                                == kv_after_forced_disconnect.active_rw_epoch_id
                                for event in kv_events_after_forced_disconnect
                            )
                            if (
                                kv_after_forced_disconnect.state == ServerState.RW
                                and kv_after_forced_disconnect.active_rw_epoch_id
                                is not None
                                and kv_after_forced_disconnect.active_rw_epoch_id
                                != primary_kv_epoch_id
                                and saw_shadow_oom
                                and not wake_future.done()
                            ):
                                break
                            time.sleep(0.2)
                        else:
                            raise TimeoutError(
                                "shadow never entered a new KV-cache epoch blocked on allocation"
                            )

                        linger_deadline = time.monotonic() + 3.0
                        while time.monotonic() < linger_deadline:
                            kv_while_lingering = kv_cache_gms.get_runtime_state()
                            kv_events_while_lingering = (
                                kv_cache_gms.get_event_history().events
                            )
                            assert kv_while_lingering.state == ServerState.RW
                            assert (
                                kv_while_lingering.active_rw_epoch_id
                                == kv_after_forced_disconnect.active_rw_epoch_id
                            )
                            assert any(
                                event.kind == "allocation_oom"
                                and event.epoch_id
                                == kv_while_lingering.active_rw_epoch_id
                                for event in kv_events_while_lingering
                            )
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
                                and kv_after_primary_kill.active_rw_epoch_id is not None
                                and kv_after_primary_kill.active_rw_epoch_id
                                != primary_kv_epoch_id
                                and kv_after_primary_kill.allocation_count > 0
                            ):
                                break
                            time.sleep(0.2)
                        else:
                            raise TimeoutError(
                                "shadow did not reacquire KV cache after failover"
                            )
                        shadow_kv_epoch_id = kv_after_primary_kill.active_rw_epoch_id

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
                # with the same committed weights epoch and a new live RW KV-cache epoch.
                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_wake = weights_gms.get_runtime_state()
                    kv_after_wake = kv_cache_gms.get_runtime_state()
                    if (
                        weights_after_wake.state == ServerState.RO
                        and weights_after_wake.ro_session_count >= 1
                        and weights_after_wake.committed_epoch_id == weights_epoch_id
                        and weights_after_wake.active_rw_epoch_id is None
                        and weights_after_wake.allocation_count > 0
                        and weights_after_wake.memory_layout_hash
                        == weights_with_primary.memory_layout_hash
                        and kv_after_wake.state == ServerState.RW
                        and kv_after_wake.active_rw_epoch_id == shadow_kv_epoch_id
                        and kv_after_wake.committed_epoch_id is None
                        and kv_after_wake.allocation_count > 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "shadow wake did not restore the expected GMS state"
                        )
                    time.sleep(0.1)

                # The final KV history should show the full handoff:
                # shadow A slept -> shadow B slept -> primary epoch ->
                # primary abort/clear -> shadow A reconnects -> shadow A sees OOM.
                weights_events_after_wake = weights_gms.get_event_history().events
                _assert_weights_published_once(
                    weights_events_after_wake, weights_epoch_id
                )

                kv_events_after_wake = kv_cache_gms.get_event_history().events
                _assert_cleared_rw_epoch(kv_events_after_wake, shadow_a_kv_epoch_id)
                _assert_cleared_rw_epoch(kv_events_after_wake, shadow_b_kv_epoch_id)
                _assert_cleared_rw_epoch(kv_events_after_wake, primary_kv_epoch_id)

                kv_pairs_after_wake = [
                    (event.kind, event.epoch_id) for event in kv_events_after_wake
                ]
                shadow_connect = ("rw_connected", shadow_kv_epoch_id)
                shadow_oom = ("allocation_oom", shadow_kv_epoch_id)
                assert shadow_connect in kv_pairs_after_wake
                assert shadow_oom in kv_pairs_after_wake
                assert kv_pairs_after_wake.count(shadow_connect) == 1
                assert kv_pairs_after_wake.count(shadow_oom) == 1
                assert kv_pairs_after_wake.index(
                    ("allocations_cleared", primary_kv_epoch_id)
                ) < kv_pairs_after_wake.index(shadow_connect)
                assert kv_pairs_after_wake.index(
                    shadow_connect
                ) < kv_pairs_after_wake.index(shadow_oom)

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
