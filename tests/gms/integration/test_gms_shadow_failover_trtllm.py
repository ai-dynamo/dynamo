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
)
from ..harness.trtllm import TRTLLM_GMS_MODEL_NAME, TRTLLMWithGMSProcess, _sleep_engine

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
            weights_hash, shadow_a_released, sleeping_mem = _sleep_engine(
                shadow_a, weights_gms, ports["frontend"]
            )

            with TRTLLMWithGMSProcess(
                request, "shadow-b", ports["shadow2_system"], ports["frontend"]
            ) as shadow_b:
                _, _, sleeping_mem = _sleep_engine(
                    shadow_b,
                    weights_gms,
                    ports["frontend"],
                    expected_weights_hash=weights_hash,
                )

                # Weights event history: a single RW connect + commit for the published layout.
                weights_events_sleeping = weights_gms.get_event_history().events
                assert [event.kind for event in weights_events_sleeping] == [
                    "rw_connected",
                    "committed",
                ]

                with TRTLLMWithGMSProcess(
                    request, "primary", ports["primary_system"], ports["frontend"]
                ) as primary:
                    result = send_completion(
                        ports["frontend"], "Primary test", model=TRTLLM_GMS_MODEL_NAME
                    )
                    assert result["choices"], "Primary inference failed"
                    logger.info("Primary inference OK: %s", result)

                    # Primary uses the same committed weights layout (as RO).
                    deadline = time.monotonic() + 30.0
                    while True:
                        state_with_primary = weights_gms.get_runtime_state()
                        if (
                            state_with_primary.state == ServerState.RO
                            and state_with_primary.ro_session_count >= 1
                            and state_with_primary.allocation_count > 0
                            and state_with_primary.memory_layout_hash == weights_hash
                        ):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "primary did not connect to committed weights layout"
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

                # Weights server must be back in RO state with the same committed layout.
                deadline = time.monotonic() + 30.0
                while True:
                    weights_after_wake = weights_gms.get_runtime_state()
                    if (
                        weights_after_wake.state == ServerState.RO
                        and weights_after_wake.allocation_count > 0
                        and weights_after_wake.memory_layout_hash == weights_hash
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "shadow A did not reconnect to committed weights layout"
                        )
                    time.sleep(0.1)

                result = send_completion(
                    ports["frontend"], "Post failover", model=TRTLLM_GMS_MODEL_NAME
                )
                assert result["choices"], "Shadow A inference after failover failed"
                logger.info("Shadow A post-failover inference OK: %s", result)
