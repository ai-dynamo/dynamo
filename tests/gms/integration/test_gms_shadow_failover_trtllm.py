# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import signal
import time
from contextlib import ExitStack

import pytest
from gpu_memory_service.common.types import ServerState

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

from ..harness.gms import GMSServerProcess
from ..harness.runtime import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    get_gpu_memory_used,
    send_completion,
    wait_for_memory_drop,
)
from ..harness.trtllm import TRTLLM_GMS_MODEL_NAME, TRTLLMWithGMSProcess

# TRTLLM shadow failover semantics:
# 1. Shadow A starts, publishes weights as committed GMS epoch, sleeps (KV freed).
# 2. Shadow B starts, imports weights as RO from committed epoch, sleeps (KV freed).
# 3. Primary starts, imports weights as RO from committed epoch, runs inference.
# 4. Primary is killed; GPU memory is released.
# 5. Shadow A wakes: reconnects weights as RO, recreates KV cache via TRTLLM VMM.
# 6. Inference succeeds on Shadow A.
#
# Unlike vLLM/SGLang, TRTLLM manages KV cache locally (no kv_cache GMS server).
# There is no GMS-mediated blocking during wake — once the primary frees GPU memory,
# Shadow A can allocate its KV cache immediately.

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


def _sleep_engine(
    engine: ManagedProcess,
    weights_gms: GMSServerProcess,
    frontend_port: int,
    *,
    expected_weights_epoch_id: int | None = None,
) -> tuple[int, int, int]:
    """Run inference, verify GMS state, call sleep, return (epoch_id, released_bytes, mem_after)."""
    result = send_completion(frontend_port, model=TRTLLM_GMS_MODEL_NAME)
    assert result["choices"], "Inference failed before sleep"

    deadline = time.monotonic() + 60.0
    while True:
        state = weights_gms.get_runtime_state()
        if (
            state.committed_epoch_id is not None
            and state.active_rw_epoch_id is None
            and state.allocation_count > 0
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("weights GMS did not reach committed state before sleep")
        time.sleep(0.1)

    if expected_weights_epoch_id is not None:
        assert state.committed_epoch_id == expected_weights_epoch_id, (
            f"Expected weights epoch {expected_weights_epoch_id}, "
            f"got {state.committed_epoch_id}"
        )

    mem_before = get_gpu_memory_used()
    sleep_result = engine.sleep()
    assert sleep_result["status"] == "ok"

    mem_after = wait_for_memory_drop(mem_before, timeout_s=30.0)
    released_bytes = mem_before - mem_after
    logger.info(
        "%s sleep: %.2f → %.2f GiB (freed %.0f MB)",
        getattr(engine, "engine_id", "engine"),
        mem_before / (1 << 30),
        mem_after / (1 << 30),
        released_bytes / (1 << 20),
    )
    assert mem_after < mem_before, "Sleep should reduce GPU memory"
    assert released_bytes > 0

    # After sleep, weights must still be committed and unchanged.
    deadline = time.monotonic() + 30.0
    while True:
        state_after = weights_gms.get_runtime_state()
        if state_after.state == ServerState.COMMITTED:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("weights GMS did not reach COMMITTED state after sleep")
        time.sleep(0.1)

    assert state_after.committed_epoch_id == state.committed_epoch_id
    assert state_after.active_rw_epoch_id is None
    assert state_after.allocation_count == state.allocation_count
    assert state_after.memory_layout_hash == state.memory_layout_hash

    return state.committed_epoch_id, released_bytes, mem_after


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover_trtllm(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    """Two sleeping shadows and one primary: kill the primary, wake shadow A, verify inference."""
    ports = gms_ports

    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, tag="weights")
        )
        stack.enter_context(
            DynamoFrontendProcess(
                request, frontend_port=ports["frontend"], display_name="frontend"
            )
        )

        with TRTLLMWithGMSProcess(
            request, "shadow-a", ports["shadow_system"], ports["frontend"]
        ) as shadow_a:
            weights_epoch_id, shadow_a_released, sleeping_mem = _sleep_engine(
                shadow_a, weights_gms, ports["frontend"]
            )

            with TRTLLMWithGMSProcess(
                request, "shadow-b", ports["shadow2_system"], ports["frontend"]
            ) as shadow_b:
                _, _, sleeping_mem = _sleep_engine(
                    shadow_b,
                    weights_gms,
                    ports["frontend"],
                    expected_weights_epoch_id=weights_epoch_id,
                )

                # Weights event history: a single RW connect + commit for epoch.
                weights_events_sleeping = weights_gms.get_event_history().events
                weights_pairs_sleeping = [
                    (e.kind, e.epoch_id) for e in weights_events_sleeping
                ]
                assert ("rw_connected", weights_epoch_id) in weights_pairs_sleeping
                assert ("committed", weights_epoch_id) in weights_pairs_sleeping
                assert (
                    weights_pairs_sleeping.count(("rw_connected", weights_epoch_id))
                    == 1
                )
                assert (
                    weights_pairs_sleeping.count(("committed", weights_epoch_id)) == 1
                )

                with TRTLLMWithGMSProcess(
                    request, "primary", ports["primary_system"], ports["frontend"]
                ) as primary:
                    result = send_completion(
                        ports["frontend"], "Primary test", model=TRTLLM_GMS_MODEL_NAME
                    )
                    assert result["choices"], "Primary inference failed"
                    logger.info("Primary inference OK: %s", result)

                    # Primary uses the same committed weights epoch (as RO).
                    deadline = time.monotonic() + 30.0
                    while True:
                        state_with_primary = weights_gms.get_runtime_state()
                        if (
                            state_with_primary.state == ServerState.RO
                            and state_with_primary.committed_epoch_id
                            == weights_epoch_id
                            and state_with_primary.ro_session_count >= 1
                        ):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "primary did not connect to committed weights epoch"
                            )
                        time.sleep(0.1)

                    primary_mem = get_gpu_memory_used()
                    logger.info(
                        "Primary active memory: %.2f GiB", primary_mem / (1 << 30)
                    )
                    assert primary_mem > sleeping_mem
                    assert (
                        primary_mem - sleeping_mem
                        >= shadow_a_released * MIN_EXPECTED_MEMORY_RETURN_FRACTION
                    )

                    # Kill the primary to free GPU memory so Shadow A can allocate KV cache.
                    logger.info("Killing primary to trigger failover")
                    _kill_process_group(primary)

                # Shadow A wakes: reconnects weights RO, recreates KV cache locally.
                wake_result = shadow_a.wake(timeout=180)
                assert wake_result["status"] == "ok"

                mem_after_wake = get_gpu_memory_used()
                reacquired = mem_after_wake - sleeping_mem
                logger.info(
                    "Shadow A wake: %.2f GiB (reacquired %.0f MB)",
                    mem_after_wake / (1 << 30),
                    reacquired / (1 << 20),
                )
                assert mem_after_wake > sleeping_mem
                assert (
                    reacquired
                    >= shadow_a_released * MIN_EXPECTED_MEMORY_RETURN_FRACTION
                )

                # Weights server must be back in RO state with the same committed epoch.
                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_wake = weights_gms.get_runtime_state()
                    if (
                        weights_after_wake.state == ServerState.RO
                        and weights_after_wake.committed_epoch_id == weights_epoch_id
                        and weights_after_wake.active_rw_epoch_id is None
                        and weights_after_wake.allocation_count > 0
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "shadow A did not reconnect to committed weights epoch"
                        )
                    time.sleep(0.1)

                result = send_completion(
                    ports["frontend"], "Post failover", model=TRTLLM_GMS_MODEL_NAME
                )
                assert result["choices"], "Shadow A inference after failover failed"
                logger.info("Shadow A post-failover inference OK: %s", result)
