# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from contextlib import ExitStack

import pytest
from gpu_memory_service.common.types import ServerState

from tests.utils.managed_process import DynamoFrontendProcess

from ..harness.gms import GMSServerProcess
from ..harness.runtime import (
    MIN_EXPECTED_MEMORY_RETURN_FRACTION,
    get_gpu_memory_used,
    send_completion,
    wait_for_memory_drop,
)
from ..harness.trtllm import (
    TRTLLM_GMS_MODEL_NAME,
    TRTLLM_GMS_READ_ONLY_CONFIG,
    TRTLLMWithGMSProcess,
)

# TRTLLM sleep/wake semantics (differs from vLLM/SGLang):
# - Weights are published once to GMS as a committed layout (shared via weights server).
# - KV cache is managed entirely by TRTLLM's own VMM — no kv_cache GMS server needed.
# - Sleep: KV cache is freed via collective RPC or local VMM tagged ops (GPU memory drops),
#   while weights remain committed in GMS (unmap VAs + abort from weights server).
# - Wake: weights reconnect as RO to the same committed layout, then KV cache is
#   recreated in a fresh local VMM region.

logger = logging.getLogger(__name__)


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_basic_sleep_wake_trtllm(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    """Single TRTLLM engine: sleep releases KV cache + unmaps weights; wake restores both."""
    ports = gms_ports
    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, tag="weights")
        )
        stack.enter_context(
            DynamoFrontendProcess(request, frontend_port=ports["frontend"])
        )
        with TRTLLMWithGMSProcess(
            request, "engine", ports["shadow_system"], ports["frontend"]
        ) as engine:
            result = send_completion(ports["frontend"], model=TRTLLM_GMS_MODEL_NAME)
            assert result["choices"]
            logger.info("Initial inference: %s", result)

            # Wait for weights to reach committed state (no active RW epoch, data present).
            deadline = time.monotonic() + 60.0
            while True:
                weights_before = weights_gms.get_runtime_state()
                if (
                    weights_before.state == ServerState.RO
                    and weights_before.allocation_count > 0
                    and weights_before.memory_layout_hash
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "weights GMS did not reach committed state before sleep"
                    )
                time.sleep(0.1)

            weights_hash = weights_before.memory_layout_hash
            mem_before = get_gpu_memory_used()
            logger.info("Memory before sleep: %.2f GiB", mem_before / (1 << 30))

            sleep_result = engine.sleep()
            assert sleep_result["status"] == "ok"

            # Poll until GPU memory drops (KV cache freed via TRTLLM VMM).
            mem_after_sleep = wait_for_memory_drop(mem_before, timeout_s=30.0)
            released_bytes = mem_before - mem_after_sleep
            logger.info(
                "Memory after sleep: %.2f GiB (freed %.0f MB)",
                mem_after_sleep / (1 << 30),
                released_bytes / (1 << 20),
            )
            assert mem_after_sleep < mem_before, "Sleep should reduce GPU memory"
            assert released_bytes > 0

            # Weights layout must be unchanged: committed and unmodified, no active clients.
            deadline = time.monotonic() + 30.0
            while True:
                weights_after_sleep = weights_gms.get_runtime_state()
                if weights_after_sleep.state == ServerState.COMMITTED:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "weights GMS did not reach COMMITTED state after sleep"
                    )
                time.sleep(0.1)

            assert (
                weights_after_sleep.allocation_count == weights_before.allocation_count
            )
            assert weights_after_sleep.memory_layout_hash == weights_hash

            # Weights event history: single RW connect + commit, no subsequent events.
            weights_events = weights_gms.get_event_history().events
            assert [event.kind for event in weights_events] == [
                "rw_connected",
                "committed",
            ]

            wake_result = engine.wake()
            assert wake_result["status"] == "ok"

            mem_after_wake = get_gpu_memory_used()
            reacquired_bytes = mem_after_wake - mem_after_sleep
            logger.info(
                "Memory after wake: %.2f GiB (reacquired %.0f MB)",
                mem_after_wake / (1 << 30),
                reacquired_bytes / (1 << 20),
            )
            assert mem_after_wake > mem_after_sleep, "Wake should recover GPU memory"
            assert (
                reacquired_bytes >= released_bytes * MIN_EXPECTED_MEMORY_RETURN_FRACTION
            )

            # After wake, TRTLLM reconnects to the same committed weights layout as RO.
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
                    raise TimeoutError("weights GMS did not reach RO state after wake")
                time.sleep(0.1)

            result = send_completion(
                ports["frontend"], "Goodbye", model=TRTLLM_GMS_MODEL_NAME
            )
            assert result["choices"]
            logger.info("Post-wake inference: %s", result)


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_read_only_import_trtllm(
    request,
    runtime_services_dynamic_ports,
    gms_ports,
    predownload_models,
):
    """A second TRTLLM process with gms_read_only=True imports weights from the
    committed epoch published by the first, sharing GPU memory via GMS."""
    ports = gms_ports
    with ExitStack() as stack:
        weights_gms = stack.enter_context(
            GMSServerProcess(request, device=0, tag="weights")
        )
        stack.enter_context(
            DynamoFrontendProcess(request, frontend_port=ports["frontend"])
        )
        with TRTLLMWithGMSProcess(
            request, "rw-engine", ports["shadow_system"], ports["frontend"]
        ):
            # Wait for the RW engine to publish its committed weights layout.
            deadline = time.monotonic() + 60.0
            while True:
                state = weights_gms.get_runtime_state()
                if (
                    state.state == ServerState.RO
                    and state.allocation_count > 0
                    and state.memory_layout_hash
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "RW engine did not commit weights to GMS in time"
                    )
                time.sleep(0.1)

            weights_hash = state.memory_layout_hash

            with TRTLLMWithGMSProcess(
                request,
                "ro-engine",
                ports["shadow2_system"],
                ports["frontend"],
                model_loader_extra_config=TRTLLM_GMS_READ_ONLY_CONFIG,
            ):
                # The RO engine should import from the committed layout and expose
                # itself as another RO session on the same weights server.
                deadline = time.monotonic() + 60.0
                while True:
                    state_with_ro = weights_gms.get_runtime_state()
                    if (
                        state_with_ro.state == ServerState.RO
                        and state_with_ro.ro_session_count >= 1
                        and state_with_ro.memory_layout_hash == weights_hash
                    ):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "RO engine did not connect to committed weights layout"
                        )
                    time.sleep(0.1)

                result = send_completion(ports["frontend"], model=TRTLLM_GMS_MODEL_NAME)
                assert result["choices"]
                logger.info("Inference with RW+RO engines: %s", result)
