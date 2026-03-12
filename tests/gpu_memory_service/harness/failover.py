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

from gpu_memory_service.common.types import ServerState
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

from .gms import GMSServerProcess
from .runtime import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    get_gpu_memory_used,
    send_completion,
)

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


def run_shadow_failover_test(
    request,
    ports: dict,
    make_shadow: Callable[[], ManagedProcess],
    make_primary: Callable[[], ManagedProcess],
) -> None:
    frontend_port = ports["frontend"]

    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="weights")
        )
        kv_cache_gms = stack.enter_context(
            GMSServerProcess(request, device=0, scope="kv_cache")
        )
        stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=frontend_port,
                display_name="frontend",
            )
        )
        with make_shadow() as shadow:
            result = send_completion(frontend_port)
            assert result["choices"], "Shadow inference failed"
            logger.info("Shadow inference OK: %s", result)

            deadline = time.monotonic() + 30.0
            while True:
                weights_initial = weights_gms.get_runtime_state()
                kv_initial = kv_cache_gms.get_runtime_state()
                if (
                    weights_initial.committed_epoch_id is not None
                    and weights_initial.active_rw_epoch_id is None
                    and weights_initial.allocation_count > 0
                    and kv_initial.state == ServerState.RW
                    and kv_initial.active_rw_epoch_id is not None
                    and kv_initial.committed_epoch_id is None
                    and kv_initial.allocation_count > 0
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("shadow startup did not stabilize GMS state")
                time.sleep(0.1)

            shadow_memory_before_sleep = get_gpu_memory_used()
            assert shadow.sleep()["status"] == "ok"
            shadow_memory_after_sleep = get_gpu_memory_used()
            shadow_released_bytes = (
                shadow_memory_before_sleep - shadow_memory_after_sleep
            )
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
                weights_after_shadow_sleep = weights_gms.get_runtime_state()
                kv_after_shadow_sleep = kv_cache_gms.get_runtime_state()
                if (
                    weights_after_shadow_sleep.state == ServerState.COMMITTED
                    and weights_after_shadow_sleep.committed_epoch_id
                    == weights_initial.committed_epoch_id
                    and weights_after_shadow_sleep.active_rw_epoch_id is None
                    and weights_after_shadow_sleep.allocation_count
                    == weights_initial.allocation_count
                    and weights_after_shadow_sleep.memory_layout_hash
                    == weights_initial.memory_layout_hash
                    and kv_after_shadow_sleep.state == ServerState.EMPTY
                    and kv_after_shadow_sleep.committed_epoch_id is None
                    and kv_after_shadow_sleep.active_rw_epoch_id is None
                    and kv_after_shadow_sleep.allocation_count == 0
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("shadow sleep did not clear GMS state")
                time.sleep(0.1)

            weights_events_after_shadow_sleep = weights_gms.get_event_history().events
            assert [
                (event.kind, event.epoch_id)
                for event in weights_events_after_shadow_sleep
            ] == [
                ("rw_connected", weights_initial.committed_epoch_id),
                ("committed", weights_initial.committed_epoch_id),
            ]

            kv_events_after_shadow_sleep = kv_cache_gms.get_event_history().events
            assert [
                (event.kind, event.epoch_id) for event in kv_events_after_shadow_sleep
            ] == [
                ("rw_connected", kv_initial.active_rw_epoch_id),
                ("rw_aborted", kv_initial.active_rw_epoch_id),
                ("allocations_cleared", kv_initial.active_rw_epoch_id),
            ]
            assert kv_events_after_shadow_sleep[-1].allocation_count > 0

            with make_primary() as primary:
                result = send_completion(frontend_port, "Primary test")
                assert result["choices"], "Primary inference failed"
                logger.info("Primary inference OK: %s", result)

                primary_memory_in_use = get_gpu_memory_used()
                logger.info(
                    "Primary active memory: %.2f GiB",
                    primary_memory_in_use / (1 << 30),
                )
                assert primary_memory_in_use > shadow_memory_after_sleep
                assert (
                    primary_memory_in_use - shadow_memory_after_sleep
                ) >= shadow_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

                deadline = time.monotonic() + 30.0
                while True:
                    weights_with_primary = weights_gms.get_runtime_state()
                    kv_with_primary = kv_cache_gms.get_runtime_state()
                    if (
                        weights_with_primary.state == ServerState.RO
                        and weights_with_primary.ro_session_count >= 1
                        and weights_with_primary.committed_epoch_id
                        == weights_initial.committed_epoch_id
                        and weights_with_primary.active_rw_epoch_id is None
                        and weights_with_primary.allocation_count
                        == weights_initial.allocation_count
                        and kv_with_primary.state == ServerState.RW
                        and kv_with_primary.active_rw_epoch_id is not None
                        and kv_with_primary.committed_epoch_id is None
                        and kv_with_primary.allocation_count > 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("primary did not acquire KV cache GMS state")
                    time.sleep(0.1)
                primary_kv_epoch_id = kv_with_primary.active_rw_epoch_id

                with ThreadPoolExecutor(max_workers=1) as executor:
                    wake_future = executor.submit(shadow.wake, 180)
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
                    assert kv_while_blocked.active_rw_epoch_id == primary_kv_epoch_id

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
                shadow_memory_after_wake - shadow_memory_after_sleep
            )
            logger.info(
                "Shadow wake memory: %.2f GiB (reacquired %.0f MB)",
                shadow_memory_after_wake / (1 << 30),
                shadow_reacquired_bytes / (1 << 20),
            )
            assert shadow_memory_after_wake > shadow_memory_after_sleep
            assert (
                shadow_reacquired_bytes
            ) >= shadow_released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

            deadline = time.monotonic() + 30.0
            while True:
                weights_after_wake = weights_gms.get_runtime_state()
                kv_after_wake = kv_cache_gms.get_runtime_state()
                if (
                    weights_after_wake.state == ServerState.RO
                    and weights_after_wake.ro_session_count >= 1
                    and weights_after_wake.committed_epoch_id
                    == weights_initial.committed_epoch_id
                    and weights_after_wake.active_rw_epoch_id is None
                    and weights_after_wake.allocation_count
                    == weights_initial.allocation_count
                    and weights_after_wake.memory_layout_hash
                    == weights_initial.memory_layout_hash
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

            weights_events_after_wake = weights_gms.get_event_history().events
            assert [
                (event.kind, event.epoch_id) for event in weights_events_after_wake
            ] == [
                ("rw_connected", weights_initial.committed_epoch_id),
                ("committed", weights_initial.committed_epoch_id),
            ]

            kv_events_after_wake = kv_cache_gms.get_event_history().events
            assert [(event.kind, event.epoch_id) for event in kv_events_after_wake] == [
                ("rw_connected", kv_initial.active_rw_epoch_id),
                ("rw_aborted", kv_initial.active_rw_epoch_id),
                ("allocations_cleared", kv_initial.active_rw_epoch_id),
                ("rw_connected", primary_kv_epoch_id),
                ("rw_aborted", primary_kv_epoch_id),
                ("allocations_cleared", primary_kv_epoch_id),
                ("rw_connected", shadow_kv_epoch_id),
            ]
            assert kv_events_after_wake[2].allocation_count > 0
            assert kv_events_after_wake[5].allocation_count > 0

            for i in range(3):
                result = send_completion(frontend_port, f"Verify {i}")
                assert result["choices"], f"Verification {i} failed"
            logger.info("All verification passed")
