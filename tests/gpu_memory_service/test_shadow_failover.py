# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re
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


def _directory_diagnostics(*processes: ManagedProcess) -> str:
    lines = []
    for process in processes:
        for line in process.read_logs().splitlines():
            if any(
                marker in line.lower()
                for marker in (
                    "gms-kvdiag",
                    "gms-kvdirectory",
                    "hbm directory adoption",
                    "bulk hbm hydration",
                    "traceback",
                    "runtimeerror",
                )
            ):
                lines.append(line)
    return "\n".join(lines[-80:])


def _wait_for_directory_writer(directory, manifest, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        result = directory.directory_lookup(manifest, [])
        if result[2] == expected or time.monotonic() >= deadline:
            return result
        time.sleep(0.01)


def _wait_for_hbm_inventory(directory, writer, epoch, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        protected, rejected = directory.directory_hbm_inventory(
            writer, epoch, scope="vllm"
        )
        if rejected or protected or time.monotonic() >= deadline:
            return protected, rejected
        time.sleep(0.01)


def _wait_for_log(process: ManagedProcess, marker: str, timeout=30.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        logs = process.read_logs()
        if marker in logs:
            return logs
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out waiting for {marker!r}\n" + _directory_diagnostics(process)
            )
        time.sleep(0.05)


_HBM_RECOVERY_PROMPT = (
    "GMS persistent HBM recovery probe. The promoted shadow must reuse this exact "
    "deterministic prefix without recomputing its key value cache. "
) * 16


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
    resume_timeout_s = 300
    expected_kv_kinds_while_blocked = [
        "rw_connected",
        "rw_aborted",
        "allocations_cleared",
    ] * 3 + ["rw_connected", "allocation_oom"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        resume_future = executor.submit(shadow.resume, resume_timeout_s)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if resume_future.done():
                break
            time.sleep(0.2)
        assert not resume_future.done(), (
            "Shadow resume completed before the primary died; "
            "KV cache RW handoff did not block as expected"
        )

        kv_with_primary = kv_cache_gms.get_runtime_state()
        assert kv_with_primary.state == ServerState.RW
        assert kv_with_primary.allocation_count > 0

        _kill_process_group(primary)

        _wait_for_blocked_resume_layout(
            kv_cache_gms,
            resume_future,
            kv_with_primary.allocation_count,
            expected_kv_kinds_while_blocked,
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
            raise TimeoutError("shadow did not reacquire KV cache after failover")

        return resume_future.result(timeout=resume_timeout_s)


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

        # The final KV history should show the full handoff:
        # shadow A paused -> shadow B paused -> primary layout ->
        # primary abort/clear -> shadow A reconnects -> shadow A sees OOM.
        weights_events_after_resume = weights_gms.get_event_history().events
        assert_weights_published_once(weights_events_after_resume)

        kv_events_after_resume = kv_cache_gms.get_event_history().events
        assert_kv_history(
            kv_events_after_resume,
            cleared_layouts=3,
            suffix=["rw_connected", "allocation_oom"],
        )

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
@pytest.mark.profiled_vram_gib(8.0)
@pytest.mark.requested_vllm_kv_cache_bytes(5_000_000_000)
@pytest.mark.timeout(600)
@pytest.mark.vllm
def test_gms_authoritative_hbm_failover_vllm(
    request, runtime_services_dynamic_ports, predownload_models, monkeypatch
):
    """Exercise production automatic takeover without test-side promotion."""
    from gms_kv_ring.daemon.client import DaemonClient

    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_LOCK_BEFORE_INIT", "0")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("VLLM_GMS_GPU_MEM_UTIL", "0.20")

    with GMSProcessManager(request, VLLMWithGMSProcess, kv_directory=True) as manager:
        assert manager.frontend_port is not None
        assert manager.kv_cache_gms is not None
        assert manager.kv_directory_socket is not None
        assert manager.kv_directory_manifest is not None

        primary = manager.start_engine("0")
        primary_output = assert_completion_ok(
            manager.frontend_port,
            _HBM_RECOVERY_PROMPT,
            failure_message="Primary HBM warmup failed",
            success_message="Primary HBM warmup OK",
            body_overrides={"temperature": 0},
        )
        assert (
            assert_completion_ok(
                manager.frontend_port,
                _HBM_RECOVERY_PROMPT,
                failure_message="Primary deterministic repeat failed",
                success_message="Primary deterministic repeat OK",
                body_overrides={"temperature": 0},
            )
            == primary_output
        )

        with DaemonClient(manager.kv_directory_socket) as directory:
            _entries, epoch, writer = _wait_for_directory_writer(
                directory, manager.kv_directory_manifest, "engine-0"
            )
            assert writer == "engine-0"
            protected, rejected = _wait_for_hbm_inventory(directory, writer, epoch)
            assert not rejected
            assert sum(map(len, protected.values())) > 0

        allocation_count = manager.kv_cache_gms.get_runtime_state().allocation_count
        shadow = manager.start_engine("1", read_only_weights=True)
        _wait_for_log(shadow, "Engine sleeping, waiting for failover lock")

        # Starting the standby must not withdraw or pause the active primary.
        assert (
            assert_completion_ok(
                manager.frontend_port,
                _HBM_RECOVERY_PROMPT,
                failure_message="Primary stopped serving while shadow waited",
                success_message="Primary remained active while shadow waited",
                body_overrides={"temperature": 0},
            )
            == primary_output
        )

        _kill_process_group(primary)
        with DaemonClient(manager.kv_directory_socket) as directory:
            _entries, _epoch, writer = _wait_for_directory_writer(
                directory, manager.kv_directory_manifest, "engine-1", timeout=30.0
            )
            assert writer == "engine-1", _directory_diagnostics(primary, shadow)

        _wait_for_log(shadow, "Engine awake, registering with discovery")
        shadow_output = assert_completion_ok(
            manager.frontend_port,
            _HBM_RECOVERY_PROMPT,
            failure_message="Automatic shadow HBM recovery failed",
            success_message="Automatic shadow HBM recovery OK",
            retry_timeout=30.0,
            body_overrides={"temperature": 0},
            min_cached_tokens=1,
        )
        assert shadow_output == primary_output
        assert (
            manager.kv_cache_gms.get_runtime_state().allocation_count
            == allocation_count
        ), "automatic takeover changed the persistent KV allocation count"
        diagnostics = _directory_diagnostics(primary, shadow)
        for counter in ("adopted_hbm_blocks", "bulk_hydrated_hbm_blocks"):
            logs = _wait_for_log(shadow, f"{counter}=")
            values = [int(value) for value in re.findall(rf"{counter}=(\d+)", logs)]
            assert values and max(values) > 0, diagnostics


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


@pytest.mark.skip(reason="Nightly CI failure: https://linear.app/nvidia/issue/OPS-4450")
@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover_trtllm(
    request, runtime_services_dynamic_ports, predownload_models
):
    """Weights-only shadow failover for TRT-LLM (no KV cache GMS)."""
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
