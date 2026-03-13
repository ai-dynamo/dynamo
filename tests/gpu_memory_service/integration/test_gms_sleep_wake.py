# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from typing import Callable

import pytest
from gpu_memory_service.common.types import ServerState
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

from ..harness.gms import GMSServerProcess
from ..harness.runtime import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    get_gpu_memory_used,
    send_completion,
)
from ..harness.sglang import SGLangWithGMSProcess
from ..harness.vllm import VLLMWithGMSProcess

# Event flow under test:
# 1. Weights are published once as a committed epoch.
# 2. KV cache starts as a live RW epoch.
# 3. Sleep keeps weights committed but aborts and clears the KV epoch.
# 4. Wake reconnects weights as RO to the same committed epoch.
# 5. Wake recreates KV cache in a fresh RW epoch after the old one was cleared.

logger = logging.getLogger(__name__)


def _run_sleep_wake_test(
    request,
    ports: dict,
    make_engine: Callable[[], ManagedProcess],
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
        with make_engine() as engine:
            result = send_completion(ports["frontend"])
            logger.info("Initial inference result: %s", result)
            assert result["choices"]

            # Before sleep, weights must already be published as a committed epoch
            # while KV cache remains a live RW epoch owned by the engine.
            deadline = time.monotonic() + 30.0
            while True:
                weights_before_sleep = weights_gms.get_runtime_state()
                kv_before_sleep = kv_cache_gms.get_runtime_state()
                if (
                    weights_before_sleep.committed_epoch_id is not None
                    and weights_before_sleep.active_rw_epoch_id is None
                    and weights_before_sleep.allocation_count > 0
                    and kv_before_sleep.state == ServerState.RW
                    and kv_before_sleep.active_rw_epoch_id is not None
                    and kv_before_sleep.committed_epoch_id is None
                    and kv_before_sleep.allocation_count > 0
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("initial GMS state did not stabilize")
                time.sleep(0.1)

            mem_before = get_gpu_memory_used()
            logger.info("Memory before sleep: %.0f MB", mem_before / (1 << 20))

            sleep_result = engine.sleep()
            assert sleep_result["status"] == "ok"

            mem_after_sleep = get_gpu_memory_used()
            released_bytes = mem_before - mem_after_sleep
            logger.info("Memory after sleep: %.0f MB", mem_after_sleep / (1 << 20))
            assert mem_after_sleep < mem_before, "Sleep should reduce memory"
            assert released_bytes > 0

            # Sleep preserves the committed weights epoch but aborts and clears the
            # mutable KV-cache epoch, which is what should release GPU memory.
            deadline = time.monotonic() + 30.0
            while True:
                weights_after_sleep = weights_gms.get_runtime_state()
                kv_after_sleep = kv_cache_gms.get_runtime_state()
                if (
                    weights_after_sleep.state == ServerState.COMMITTED
                    and weights_after_sleep.committed_epoch_id
                    == weights_before_sleep.committed_epoch_id
                    and weights_after_sleep.active_rw_epoch_id is None
                    and weights_after_sleep.allocation_count
                    == weights_before_sleep.allocation_count
                    and weights_after_sleep.memory_layout_hash
                    == weights_before_sleep.memory_layout_hash
                    and kv_after_sleep.state == ServerState.EMPTY
                    and kv_after_sleep.committed_epoch_id is None
                    and kv_after_sleep.active_rw_epoch_id is None
                    and kv_after_sleep.allocation_count == 0
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "sleep did not drive GMS into the expected state"
                    )
                time.sleep(0.1)

            # Weights are immutable across sleep/wake, so their event history should
            # still be the original publish: connect once, commit once.
            weights_events = weights_gms.get_event_history().events
            weights_pairs = [(event.kind, event.epoch_id) for event in weights_events]
            weights_connect = ("rw_connected", weights_before_sleep.committed_epoch_id)
            weights_commit = ("committed", weights_before_sleep.committed_epoch_id)
            assert weights_connect in weights_pairs
            assert weights_commit in weights_pairs
            assert weights_pairs.count(weights_connect) == 1
            assert weights_pairs.count(weights_commit) == 1
            assert weights_pairs.index(weights_connect) < weights_pairs.index(
                weights_commit
            )

            # KV cache is different: sleep must abort the old RW epoch and clear its
            # server-owned allocations before wake can start a new RW epoch.
            kv_events = kv_cache_gms.get_event_history().events
            kv_pairs = [(event.kind, event.epoch_id) for event in kv_events]
            kv_connect = ("rw_connected", kv_before_sleep.active_rw_epoch_id)
            kv_abort = ("rw_aborted", kv_before_sleep.active_rw_epoch_id)
            kv_clear = ("allocations_cleared", kv_before_sleep.active_rw_epoch_id)
            assert kv_connect in kv_pairs
            assert kv_abort in kv_pairs
            assert kv_clear in kv_pairs
            assert kv_pairs.count(kv_connect) == 1
            assert kv_pairs.count(kv_abort) == 1
            assert kv_pairs.count(kv_clear) == 1
            assert kv_pairs.index(kv_connect) < kv_pairs.index(kv_abort)
            assert kv_pairs.index(kv_abort) < kv_pairs.index(kv_clear)
            assert (
                next(
                    event
                    for event in kv_events
                    if event.kind == "allocations_cleared"
                    and event.epoch_id == kv_before_sleep.active_rw_epoch_id
                ).allocation_count
                > 0
            )

            wake_result = engine.wake()
            assert wake_result["status"] == "ok"

            mem_after_wake = get_gpu_memory_used()
            reacquired_bytes = mem_after_wake - mem_after_sleep
            logger.info("Memory after wake: %.0f MB", mem_after_wake / (1 << 20))
            assert mem_after_wake > mem_after_sleep, "Wake should reacquire memory"
            assert (
                reacquired_bytes
            ) >= released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION

            # Wake reconnects weights as RO to the same committed epoch, but KV cache
            # must come back as a fresh RW epoch with new allocations.
            deadline = time.monotonic() + 30.0
            while True:
                weights_after_wake = weights_gms.get_runtime_state()
                kv_after_wake = kv_cache_gms.get_runtime_state()
                if (
                    weights_after_wake.state == ServerState.RO
                    and weights_after_wake.committed_epoch_id
                    == weights_before_sleep.committed_epoch_id
                    and weights_after_wake.active_rw_epoch_id is None
                    and weights_after_wake.allocation_count
                    == weights_before_sleep.allocation_count
                    and weights_after_wake.memory_layout_hash
                    == weights_before_sleep.memory_layout_hash
                    and kv_after_wake.state == ServerState.RW
                    and kv_after_wake.active_rw_epoch_id is not None
                    and kv_after_wake.active_rw_epoch_id
                    != kv_before_sleep.active_rw_epoch_id
                    and kv_after_wake.committed_epoch_id is None
                    and kv_after_wake.allocation_count > 0
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("wake did not restore the expected GMS state")
                time.sleep(0.1)

            weights_events_after_wake = weights_gms.get_event_history().events
            weights_pairs_after_wake = [
                (event.kind, event.epoch_id) for event in weights_events_after_wake
            ]
            assert weights_connect in weights_pairs_after_wake
            assert weights_commit in weights_pairs_after_wake
            assert weights_pairs_after_wake.count(weights_connect) == 1
            assert weights_pairs_after_wake.count(weights_commit) == 1
            assert weights_pairs_after_wake.index(
                weights_connect
            ) < weights_pairs_after_wake.index(weights_commit)

            # The wake history should therefore extend the old KV sequence with one
            # new RW connect after the previous epoch was fully cleared.
            kv_events_after_wake = kv_cache_gms.get_event_history().events
            kv_pairs_after_wake = [
                (event.kind, event.epoch_id) for event in kv_events_after_wake
            ]
            kv_reconnect = ("rw_connected", kv_after_wake.active_rw_epoch_id)
            assert kv_connect in kv_pairs_after_wake
            assert kv_abort in kv_pairs_after_wake
            assert kv_clear in kv_pairs_after_wake
            assert kv_reconnect in kv_pairs_after_wake
            assert kv_pairs_after_wake.count(kv_connect) == 1
            assert kv_pairs_after_wake.count(kv_abort) == 1
            assert kv_pairs_after_wake.count(kv_clear) == 1
            assert kv_pairs_after_wake.count(kv_reconnect) == 1
            assert kv_pairs_after_wake.index(kv_connect) < kv_pairs_after_wake.index(
                kv_abort
            )
            assert kv_pairs_after_wake.index(kv_abort) < kv_pairs_after_wake.index(
                kv_clear
            )
            assert kv_pairs_after_wake.index(kv_clear) < kv_pairs_after_wake.index(
                kv_reconnect
            )
            assert (
                next(
                    event
                    for event in kv_events_after_wake
                    if event.kind == "allocations_cleared"
                    and event.epoch_id == kv_before_sleep.active_rw_epoch_id
                ).allocation_count
                > 0
            )

            result = send_completion(ports["frontend"], "Goodbye")
            logger.info("Post-wake inference result: %s", result)
            assert result["choices"]

            logger.info(
                "Memory freed: %.0f MB", (mem_before - mem_after_sleep) / (1 << 20)
            )


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake_vllm(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    ports = gms_ports
    _run_sleep_wake_test(
        request,
        ports,
        make_engine=lambda: VLLMWithGMSProcess(
            request,
            "engine",
            ports["shadow_system"],
            ports["shadow_kv_event"],
            ports["shadow_nixl"],
            ports["frontend"],
        ),
    )


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake_sglang(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    ports = gms_ports
    _run_sleep_wake_test(
        request,
        ports,
        make_engine=lambda: SGLangWithGMSProcess(
            request,
            "engine",
            ports["shadow_system"],
            ports["shadow_sglang"],
            ports["frontend"],
        ),
    )
