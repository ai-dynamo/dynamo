# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

import pytest
from gpu_memory_service.server.fsm import ServerState

from tests.gpu_memory_service.common.runtime import (
    GMSProcessManager,
    SGLangWithGMSProcess,
    TRTLLMWithGMSProcess,
    VLLMWithGMSProcess,
    get_gpu_memory_used,
)
from tests.gpu_memory_service.flow_assertions import (
    assert_completion_ok,
    assert_kv_history,
    assert_memory_restored_after_quiesce,
    assert_weights_published_once,
    quiesce_engine,
    wait_for_resumed_layout,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

pytestmark = [pytest.mark.nightly, pytest.mark.fault_tolerance]

# Event flow under test:
# 1. Weights are published once as a committed layout.
# 2. KV cache starts as a live RW layout build.
# 3. Quiesce keeps weights committed but aborts and clears the KV layout.
# 4. Resume reconnects weights as RO to the same committed layout.
# 5. Resume recreates KV cache in a fresh RW layout after the old one was cleared.

logger = logging.getLogger(__name__)


def _run_quiesce_resume_test(
    request,
    engine_cls,
) -> None:
    with GMSProcessManager(request, engine_cls) as manager:
        frontend_port = manager.frontend_port
        weights_gms = manager.weights_gms
        kv_cache_gms = manager.kv_cache_gms
        engine = manager.start_engine("engine")
        assert_completion_ok(
            frontend_port,
            "Hello",
            failure_message="Initial inference failed",
            success_message="Initial inference result",
        )

        # Before quiesce, weights must already be published and visible to RO
        # readers while KV cache remains a live RW layout owned by the engine.
        weights_before_quiesce, released_bytes, mem_after_quiesce = quiesce_engine(
            weights_gms,
            kv_cache_gms,
            engine,
            quiesce_label="Engine quiesce",
        )

        # Weights are immutable across quiesce/resume, so their event history should
        # still be the original publish: connect once, commit once.
        weights_events = weights_gms.get_event_history().events
        assert_weights_published_once(weights_events)

        # KV cache is different: quiesce must abort the old RW layout and clear
        # its server-owned allocations before resume can start a new RW layout.
        kv_events = kv_cache_gms.get_event_history().events
        assert_kv_history(kv_events, cleared_layouts=1)
        assert kv_events[-1].allocation_count > 0

        resume_result = engine.resume()
        assert resume_result["status"] == "ok"

        mem_after_resume = get_gpu_memory_used()
        assert_memory_restored_after_quiesce(
            "Memory after resume",
            mem_after_quiesce,
            mem_after_resume,
            released_bytes,
        )

        # Resume reconnects weights as RO to the same committed layout, but KV cache
        # must come back as a fresh RW layout with new allocations.
        wait_for_resumed_layout(
            weights_gms,
            kv_cache_gms,
            weights_before_quiesce,
        )

        weights_events_after_resume = weights_gms.get_event_history().events
        assert_weights_published_once(weights_events_after_resume)

        # The resume history should therefore extend the old KV sequence with one
        # new RW connect after the previous layout was fully cleared.
        kv_events_after_resume = kv_cache_gms.get_event_history().events
        assert_kv_history(
            kv_events_after_resume,
            cleared_layouts=1,
            suffix=["rw_connected"],
        )
        assert kv_events_after_resume[2].allocation_count > 0

        assert_completion_ok(
            frontend_port,
            "Goodbye",
            failure_message="Post-resume inference failed",
            success_message="Post-resume inference result",
        )

        logger.info("Memory freed: %.0f MB", released_bytes / (1 << 20))


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
@pytest.mark.vllm
def test_gms_basic_quiesce_resume_vllm(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    _run_quiesce_resume_test(request, VLLMWithGMSProcess)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(300)
@pytest.mark.sglang
def test_gms_basic_quiesce_resume_sglang(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    _run_quiesce_resume_test(request, SGLangWithGMSProcess)


# ---------------------------------------------------------------------------
# TRT-LLM standalone tests (weights-only GMS topology, no KV cache GMS)
# ---------------------------------------------------------------------------


def _wait_for_weights_state(
    weights_gms,
    expected_state,
    *,
    min_ro_sessions: int = 0,
    expected_hash: str | None = None,
    timeout: float = 30.0,
):
    """Poll until the weights GMS daemon reaches *expected_state*."""
    deadline = time.monotonic() + timeout
    while True:
        ws = weights_gms.get_runtime_state()
        if (
            ws.state == expected_state
            and ws.allocation_count > 0
            and ws.memory_layout_hash
            and ws.ro_session_count >= min_ro_sessions
            and (expected_hash is None or ws.memory_layout_hash == expected_hash)
        ):
            return ws
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Weights: state={ws.state} (want {expected_state}), "
                f"allocs={ws.allocation_count}, hash={ws.memory_layout_hash}"
            )
        time.sleep(0.1)


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_basic_quiesce_resume_trtllm(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    """Weights-only quiesce/resume for TRT-LLM (no KV cache GMS)."""
    with GMSProcessManager(request, TRTLLMWithGMSProcess, tags=("weights",)) as manager:
        frontend_port = manager.frontend_port
        weights_gms = manager.weights_gms
        engine = manager.start_engine("engine")

        assert_completion_ok(
            frontend_port,
            "Hello",
            failure_message="Initial inference failed",
            success_message="Initial inference OK",
        )

        ws = _wait_for_weights_state(weights_gms, ServerState.RO, timeout=60.0)
        weights_hash = ws.memory_layout_hash

        mem_before = get_gpu_memory_used()
        assert engine.quiesce()["status"] == "ok"
        mem_after = get_gpu_memory_used()
        released = mem_before - mem_after
        logger.info(
            "TRT-LLM quiesce: %.2f -> %.2f GiB (freed %.0f MB)",
            mem_before / (1 << 30),
            mem_after / (1 << 30),
            released / (1 << 20),
        )
        assert released > 0

        _wait_for_weights_state(
            weights_gms, ServerState.COMMITTED, expected_hash=weights_hash
        )
        assert_weights_published_once(weights_gms.get_event_history().events)

        assert engine.resume()["status"] == "ok"
        mem_resumed = get_gpu_memory_used()
        assert_memory_restored_after_quiesce(
            "TRT-LLM resume",
            mem_after,
            mem_resumed,
            released,
            min_fraction=0.6,
        )

        _wait_for_weights_state(weights_gms, ServerState.RO, expected_hash=weights_hash)
        assert_weights_published_once(weights_gms.get_event_history().events)

        assert_completion_ok(
            frontend_port,
            "Goodbye",
            failure_message="Post-resume inference failed",
            success_message="Post-resume inference OK",
        )
        logger.info("Memory freed: %.0f MB", released / (1 << 20))


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_read_only_import_trtllm(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    """A second TRT-LLM process with read_only_weights=True imports weights
    from the committed layout published by the first, sharing GPU memory."""
    with GMSProcessManager(request, TRTLLMWithGMSProcess, tags=("weights",)) as manager:
        frontend_port = manager.frontend_port
        weights_gms = manager.weights_gms

        manager.start_engine("rw-engine")
        ws = _wait_for_weights_state(weights_gms, ServerState.RO, timeout=60.0)
        weights_hash = ws.memory_layout_hash

        manager.start_engine("ro-engine", read_only_weights=True)
        _wait_for_weights_state(
            weights_gms,
            ServerState.RO,
            min_ro_sessions=1,
            expected_hash=weights_hash,
            timeout=60.0,
        )

        assert_completion_ok(
            frontend_port,
            "Hello",
            failure_message="RW+RO inference failed",
            success_message="RW+RO inference OK",
        )
