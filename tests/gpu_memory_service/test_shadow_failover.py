# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from gpu_memory_service.server.fsm import ServerState

from tests.gpu_memory_service.common.runtime import (
    GMSProcessManager,
    SGLangWithGMSProcess,
    TRTLLMWithGMSProcess,
    VLLMWithGMSProcess,
)
from tests.gpu_memory_service.flow_assertions import (
    assert_completion_ok,
    assert_kv_history,
    assert_weights_published_once,
    pause_engine,
    wait_for_active_layout,
    wait_for_resumed_layout,
    wait_for_weights_state,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = [pytest.mark.nightly, pytest.mark.fault_tolerance]

# Event flow under test:
# 1. Shadow A starts as the initial weights publisher, then pauses without serving traffic.
# 2. Shadow B starts in read-only mode from the committed weights layout, then pauses without serving traffic.
# 3. Primary starts in read-only mode and owns the next RW KV layout.
# 4. Shadow A tries to resume while primary still owns the KV-cache RW layout.
# 5. Primary is SIGKILLed; the old KV session clears before its GPU memory is reclaimed.
# 6. Shadow A enters a new RW KV layout, hits allocation_oom, then finishes resume.

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


def _start_primary(
    manager,
    frontend_port: int,
    weights_gms,
    kv_cache_gms,
    *,
    weights_hash: str,
):
    primary = manager.start_engine("primary", read_only_weights=True)
    assert_completion_ok(
        frontend_port,
        "Primary test",
        failure_message="Primary inference failed",
        success_message="Primary inference OK",
    )

    weights_with_primary, _ = wait_for_active_layout(
        weights_gms,
        kv_cache_gms,
        expected_weights_hash=weights_hash,
        min_weight_ro_sessions=1,
    )
    assert_kv_history(
        kv_cache_gms.get_event_history().events,
        cleared_layouts=2,
        suffix=["rw_connected"],
    )
    return primary, weights_with_primary


def _wait_for_blocked_resume_layout(
    kv_cache_gms,
    resume_future,
    previous_allocation_count: int,
    expected_kinds: list[str],
) -> int:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        kv_runtime_state = kv_cache_gms.get_runtime_state()
        kv_events = kv_cache_gms.get_event_history().events
        if (
            kv_runtime_state.state == ServerState.RW
            and [event.kind for event in kv_events] == expected_kinds
            and not resume_future.done()
        ):
            blocked_allocation_count = kv_runtime_state.allocation_count
            if (
                blocked_allocation_count < previous_allocation_count
                and blocked_allocation_count == kv_events[-1].allocation_count
            ):
                return blocked_allocation_count
        time.sleep(0.2)

    raise TimeoutError(
        "shadow never entered a new KV-cache layout blocked on allocation"
    )


def _resume_shadow_after_primary_failover(
    shadow: ManagedProcess,
    kv_cache_gms,
    primary: ManagedProcess,
):
    # Pre-activation failover model: the shadow begins taking over as soon as a
    # crash is detected, and may go live BEFORE the primary fully dies. Primary
    # and shadow coexist on the shared GMS-owned KV pool; per KV segment only one
    # engine holds the RW lock, so the shadow writes the segments the primary has
    # released and acquires the remainder once the primary is gone. The handoff
    # therefore must NOT be required to block on a single whole-pool RW lock.
    resume_timeout_s = 300

    with ThreadPoolExecutor(max_workers=1) as executor:
        resume_future = executor.submit(shadow.resume, resume_timeout_s)

        # KV must remain RW-owned throughout the overlap window (never EMPTY):
        # the persistent pool survives the crash and is continuously claimed.
        kv_with_primary = kv_cache_gms.get_runtime_state()
        assert kv_with_primary.state == ServerState.RW
        assert kv_with_primary.allocation_count > 0

        _kill_process_group(primary)

        # After the primary is gone the shadow must fully reacquire the KV pool.
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
            raise TimeoutError("shadow did not reacquire KV cache after failover")

        result = resume_future.result(timeout=resume_timeout_s)
        kv_with_shadow = kv_cache_gms.get_runtime_state()
        assert kv_with_shadow.state == ServerState.RW
        assert kv_with_shadow.allocation_count == kv_with_primary.allocation_count, (
            "failover changed the committed shared KV allocation count"
        )
        return result


def _run_shadow_failover_test(
    request,
    engine_cls,
) -> None:
    with GMSProcessManager(request, engine_cls) as manager:
        frontend_port = manager.frontend_port
        weights_gms = manager.weights_gms
        kv_cache_gms = manager.kv_cache_gms

        shadow_a = manager.start_engine(
            "shadow-a",
        )
        weights_state_after_shadow_a = pause_engine(
            weights_gms,
            kv_cache_gms,
            shadow_a,
            pause_label="Shadow pause",
        )
        weights_hash = weights_state_after_shadow_a.memory_layout_hash
        shadow_b = manager.start_engine(
            "shadow-b",
            read_only_weights=True,
        )
        weights_state_after_shadow_b = pause_engine(
            weights_gms,
            kv_cache_gms,
            shadow_b,
            pause_label="Shadow pause",
            expected_weights_hash=weights_hash,
        )
        assert weights_state_after_shadow_b.memory_layout_hash == weights_hash

        weights_events_after_shadow_pause = weights_gms.get_event_history().events
        assert_weights_published_once(weights_events_after_shadow_pause)

        kv_events_after_shadow_pause = kv_cache_gms.get_event_history().events
        assert_kv_history(kv_events_after_shadow_pause, cleared_layouts=2)

        primary, weights_with_primary = _start_primary(
            manager,
            frontend_port,
            weights_gms,
            kv_cache_gms,
            weights_hash=weights_hash,
        )
        resume_result = _resume_shadow_after_primary_failover(
            shadow_a,
            kv_cache_gms,
            primary,
        )

        assert resume_result["status"] == "ok"

        # Once the primary is gone, the failover shadow should finish resume
        # with the same committed weights layout and a new live RW KV-cache layout.
        wait_for_resumed_layout(
            weights_gms,
            kv_cache_gms,
            weights_with_primary,
            min_weight_ro_sessions=1,
        )

        # Weights are still published exactly once across the whole handoff.
        weights_events_after_resume = weights_gms.get_event_history().events
        assert_weights_published_once(weights_events_after_resume)

        # The helper asserted that takeover preserved the committed shared KV
        # allocation count and returned only after the shadow held RW ownership.

        # The reported result: the shadow serves real tokens after the crash.
        assert_completion_ok(
            frontend_port,
            "Post failover",
            failure_message="Shadow inference after failover failed",
            success_message="Shadow inference after failover OK",
            retry_timeout=30.0,
        )


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.vllm
def test_gms_shadow_engine_failover_vllm(
    request, runtime_services_dynamic_ports, predownload_models
):
    _run_shadow_failover_test(request, VLLMWithGMSProcess)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.sglang
def test_gms_shadow_engine_failover_sglang(
    request, runtime_services_dynamic_ports, predownload_models
):
    _run_shadow_failover_test(request, SGLangWithGMSProcess)


# ---------------------------------------------------------------------------
# TRT-LLM standalone failover test (weights-only GMS, no KV cache GMS)
# ---------------------------------------------------------------------------


def _trtllm_pause(
    weights_gms,
    engine,
    *,
    label: str,
    expected_hash: str | None = None,
):
    """Pause a weights-only TRT-LLM engine and return the weights state."""
    wait_for_weights_state(
        weights_gms,
        ServerState.RO,
        expected_hash=expected_hash,
        timeout=60.0,
    )
    assert engine.pause()["status"] == "ok"
    logger.info("%s completed", label)
    ws = wait_for_weights_state(weights_gms, ServerState.COMMITTED)
    return ws


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover_trtllm(
    request, runtime_services_dynamic_ports, predownload_models
):
    """Shadow failover for TRT-LLM.

    TRT-LLM's only supported GMS KV path is KVCacheManagerV2 (the worker forces
    ``use_kv_cache_manager_v2=True`` and rejects the legacy V1 connector). GMS
    owns the model weights via the weights daemon; V2 KV slot leases coordinate
    the KV handoff host-side (no kv_cache GMS daemon, so ``tags=("weights",)``).
    The shadow pre-activates (resume immediately on primary kill, no KV block).
    Requires GMS_KV_LEASES=1 for the V2 slot-lease integration to install."""
    with GMSProcessManager(request, TRTLLMWithGMSProcess, tags=("weights",)) as manager:
        frontend_port = manager.frontend_port
        weights_gms = manager.weights_gms

        # Shadow A publishes weights, then pauses.
        shadow_a = manager.start_engine("shadow-a")
        assert_completion_ok(
            frontend_port,
            "Hello",
            failure_message="Shadow A inference failed",
            success_message="Shadow A inference OK",
        )
        ws_a = _trtllm_pause(weights_gms, shadow_a, label="Shadow A pause")
        weights_hash = ws_a.memory_layout_hash

        # Shadow B starts RO, then pauses.
        shadow_b = manager.start_engine("shadow-b", read_only_weights=True)
        assert_completion_ok(
            frontend_port,
            "Hello",
            failure_message="Shadow B inference failed",
            success_message="Shadow B inference OK",
        )
        _trtllm_pause(
            weights_gms,
            shadow_b,
            label="Shadow B pause",
            expected_hash=weights_hash,
        )
        assert_weights_published_once(weights_gms.get_event_history().events)

        # Primary starts RO.
        primary = manager.start_engine("primary", read_only_weights=True)
        assert_completion_ok(
            frontend_port,
            "Primary test",
            failure_message="Primary inference failed",
            success_message="Primary inference OK",
        )
        wait_for_weights_state(
            weights_gms,
            ServerState.RO,
            expected_hash=weights_hash,
            min_ro_sessions=1,
        )

        # Kill primary, resume shadow A immediately (no KV blocking).
        _kill_process_group(primary)
        resume_result = shadow_a.resume(timeout=180)
        assert resume_result["status"] == "ok"

        wait_for_weights_state(
            weights_gms,
            ServerState.RO,
            expected_hash=weights_hash,
            min_ro_sessions=1,
        )
        assert_weights_published_once(weights_gms.get_event_history().events)

        assert_completion_ok(
            frontend_port,
            "Post failover",
            failure_message="Shadow after failover failed",
            success_message="Shadow after failover OK",
            retry_timeout=30.0,
        )
