# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

import requests
from gpu_memory_service.server.fsm import ServerState

from tests.utils.client import send_request
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.payloads import CompletionPayload

logger = logging.getLogger(__name__)


def assert_completion_ok(
    frontend_port: int,
    prompt: str,
    *,
    failure_message: str,
    success_message: str,
    retry_timeout: float = 0.0,
    retry_interval: float = 1.0,
):
    completion = CompletionPayload(
        body={
            "model": FAULT_TOLERANCE_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 20,
        },
        expected_response=[],
        expected_log=[],
        timeout=120,
        port=frontend_port,
    )
    deadline = time.monotonic() + retry_timeout
    while True:
        response = send_request(
            url=completion.url(),
            payload=completion.body,
            timeout=completion.timeout,
            method=completion.method,
        )
        try:
            completion.process_response(response)
            result = response.json()
            if not isinstance(result, dict) or not result.get("choices"):
                raise AssertionError(failure_message)
            logger.info("%s: %s", success_message, result)
            return
        except (AssertionError, KeyError, requests.RequestException, ValueError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(retry_interval)


def quiesce_engine(
    weights_gms,
    kv_cache_gms,
    engine,
    *,
    quiesce_label: str,
    expected_weights_hash: str | None = None,
):
    weights_state, _ = wait_for_active_layout(
        weights_gms,
        kv_cache_gms,
        expected_weights_hash=expected_weights_hash,
    )

    assert engine.quiesce()["status"] == "ok"
    logger.info("%s completed", quiesce_label)

    wait_for_quiesced_layout(weights_gms, kv_cache_gms, weights_state)
    return weights_state


def wait_for_active_layout(
    weights_gms,
    kv_cache_gms,
    *,
    expected_weights_hash: str | None = None,
    min_weight_ro_sessions: int = 0,
    timeout: float = 30.0,
):
    deadline = time.monotonic() + timeout
    while True:
        weights_state = weights_gms.get_runtime_state()
        kv_state = kv_cache_gms.get_runtime_state()
        if (
            weights_state.state == ServerState.RO
            and weights_state.ro_session_count >= min_weight_ro_sessions
            and weights_state.allocation_count > 0
            and weights_state.memory_layout_hash
            and kv_state.state == ServerState.RW
            and kv_state.allocation_count > 0
        ):
            if (
                expected_weights_hash is None
                or weights_state.memory_layout_hash == expected_weights_hash
            ):
                return weights_state, kv_state
        if time.monotonic() > deadline:
            raise TimeoutError("GMS state did not reach the active layout")
        time.sleep(0.1)


def wait_for_quiesced_layout(
    weights_gms,
    kv_cache_gms,
    weights_state_before_quiesce,
    *,
    require_no_ro_sessions: bool = False,
    timeout: float = 30.0,
):
    deadline = time.monotonic() + timeout
    while True:
        weights_after_quiesce = weights_gms.get_runtime_state()
        kv_after_quiesce = kv_cache_gms.get_runtime_state()
        if (
            weights_after_quiesce.state == ServerState.COMMITTED
            and weights_after_quiesce.allocation_count
            == weights_state_before_quiesce.allocation_count
            and weights_after_quiesce.memory_layout_hash
            == weights_state_before_quiesce.memory_layout_hash
            and kv_after_quiesce.state == ServerState.EMPTY
            and kv_after_quiesce.allocation_count == 0
        ):
            if (
                not require_no_ro_sessions
                or weights_after_quiesce.ro_session_count == 0
            ):
                return weights_after_quiesce, kv_after_quiesce
        if time.monotonic() > deadline:
            raise TimeoutError("GMS state did not reach the quiesced layout")
        time.sleep(0.1)


def wait_for_resumed_layout(
    weights_gms,
    kv_cache_gms,
    weights_state_before_quiesce,
    *,
    min_weight_ro_sessions: int = 0,
    timeout: float = 30.0,
):
    deadline = time.monotonic() + timeout
    while True:
        weights_after_resume = weights_gms.get_runtime_state()
        kv_after_resume = kv_cache_gms.get_runtime_state()
        if (
            weights_after_resume.state == ServerState.RO
            and weights_after_resume.ro_session_count >= min_weight_ro_sessions
            and weights_after_resume.allocation_count
            == weights_state_before_quiesce.allocation_count
            and weights_after_resume.memory_layout_hash
            == weights_state_before_quiesce.memory_layout_hash
            and kv_after_resume.state == ServerState.RW
            and kv_after_resume.allocation_count > 0
        ):
            return weights_after_resume, kv_after_resume
        if time.monotonic() > deadline:
            raise TimeoutError("GMS state did not reach the resumed layout")
        time.sleep(0.1)


def assert_weights_published_once(events) -> None:
    assert [event.kind for event in events] == ["rw_connected", "committed"]


def assert_kv_history(
    events,
    *,
    cleared_layouts: int,
    suffix: list[str] | None = None,
) -> None:
    expected_kinds = [
        "rw_connected",
        "rw_aborted",
        "allocations_cleared",
    ] * cleared_layouts
    if suffix is not None:
        expected_kinds.extend(suffix)

    assert [event.kind for event in events] == expected_kinds
    clear_counts = [
        event.allocation_count
        for event in events
        if event.kind == "allocations_cleared"
    ]
    assert len(clear_counts) >= cleared_layouts
    assert all(count > 0 for count in clear_counts[:cleared_layouts])


def wait_for_weights_state(
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
