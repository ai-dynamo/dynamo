# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI smoke tests for approximate MM KV routing.

Approximate routing uses ``--router-mode kv --no-router-kv-events`` on the
frontend. The router derives its stickiness signal from its own routing
decisions plus frontend-synthesized mm_hashes; it does NOT subscribe to
backend KV events. This means it works on any backend regardless of whether
the backend emits KV events (and is the only KV-routing option for backends
that don't emit MM-aware KV events).

This file is the regression guard for the approx-MM-routing mechanism:
  - does the frontend start in approx mode,
  - does a repeat request with the same image(s) produce a routing overlap,
  - does a distinct-image request NOT produce spurious overlap.

Backends covered:
  - vLLM
  - TRT-LLM
  - SGLang
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from tests.conftest import EtcdServer, NatsServer
from tests.mm_router.test_vllm_mm_router_e2e import (  # reuse fixtures/helpers
    _COMMON_PROCESS_KWARGS,
    BLOCK_SIZE,
    VLLM_MM_MODEL,
    _make_data_uri,
    _make_process_env,
    _send_request_get_overlap,
)
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

try:
    from tests.mm_router.test_mm_router_e2e import TRTLLM_MM_MODEL
except ImportError:
    TRTLLM_MM_MODEL = None  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Shared payload + assertion helpers
# --------------------------------------------------------------------------

# Three distinct colors → three distinct mm_hashes, used by the multi-image
# tests to exercise the per-image-hash accumulation path.
_COLORS = [(200, 50, 100), (10, 200, 40), (50, 50, 200)]


def _payload_with_images(
    model: str,
    image_uris: list[str],
    prompt: str = "Describe what you see.",
) -> dict[str, Any]:
    """Build an OpenAI-style chat payload with an arbitrary number of images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
    }


def _assert_repeat_overlap_grows(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> None:
    """Send the same payload twice; 2nd request must have strictly more overlap."""
    overlap_1, _total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, f"{label} #1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, f"{label} #2"
    )
    assert overlap_2 > overlap_1, (
        f"{label}: expected overlap to grow on repeat "
        f"(got {overlap_1} → {overlap_2} of {total_2})"
    )
    assert total_2 > 0, f"{label}: total blocks should be positive"


def _assert_distinct_payloads_no_false_overlap(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    label: str,
) -> None:
    """Send A then B with a different image. B's overlap must be bounded by the
    shared text prefix only — the image block must NOT be claimed as overlap."""
    _send_request_get_overlap(frontend_port, router_proc, payload_a, f"{label} A")
    overlap_b, total_b, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload_b, f"{label} B"
    )
    assert total_b > 0
    assert overlap_b < total_b, (
        f"{label}: different images should not fully-overlap "
        f"(got {overlap_b}/{total_b})"
    )


def _prepare_log_dir(request, suffix: str) -> str:
    """Per-call fresh tempdir under the system tempfile root, isolated from the repo tree."""
    return tempfile.mkdtemp(prefix=f"{request.node.name}_{suffix}_")


def _maybe_model_mark(model):
    """Apply ``@pytest.mark.model(model)`` only when ``model`` is not None.

    Lets us skip the predownload marker cleanly when an optional backend
    (e.g. TRT-LLM) isn't installed, instead of forcing a fake model name into
    the predownload fixture and breaking collection.
    """
    if model is None:
        return lambda fn: fn
    return pytest.mark.model(model)


# --------------------------------------------------------------------------
# Shared: Approx-mode Frontend (adds --router-mode kv and --no-router-kv-events)
# --------------------------------------------------------------------------


class ApproxFrontendProcess(ManagedProcess):
    """Frontend launched with approx MM KV routing (no kv events)."""

    def __init__(
        self,
        request,
        *,
        frontend_port: int,
    ):
        # Approx mode mirrors launch_approx_routing.sh: KV router with
        # synthetic events (--no-router-kv-events). We deliberately do NOT
        # pass --dyn-chat-processor or --model-name — the backend advertises
        # its own served_model_name via etcd, and adding --model-name on the
        # frontend overrides it to a normalized (lower-cased) id, which
        # causes /v1/chat/completions to 404 on the original-cased request.
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(frontend_port),
                "--router-mode",
                "kv",
                "--kv-cache-block-size",
                str(BLOCK_SIZE),
                "--no-router-kv-events",  # ← approx mode
            ],
            env=_make_process_env(log_level="debug"),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "approx-frontend"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def approx_runtime_services(request):
    # Use MonkeyPatch so env vars are restored even if teardown is skipped.
    mp = pytest.MonkeyPatch()
    with NatsServer(request, port=0) as nats, EtcdServer(request, port=0) as etcd:
        mp.setenv("NATS_SERVER", f"nats://localhost:{nats.port}")
        mp.setenv("ETCD_ENDPOINTS", f"http://localhost:{etcd.port}")
        try:
            yield
        finally:
            mp.undo()


pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
]


# --------------------------------------------------------------------------
# vLLM approx MM routing
# --------------------------------------------------------------------------


class ApproxVLLMWorkerProcess(ManagedProcess):
    """vLLM backend worker for approx-mode tests (no __internal suffix)."""

    def __init__(self, request, *, system_port: int, kv_event_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                VLLM_MM_MODEL,
                "--enable-multimodal",
                "--gpu-memory-utilization",
                "0.85",
                "--max-model-len",
                "8192",
                "--served-model-name",
                VLLM_MM_MODEL,
                "--kv-events-config",
                (
                    f'{{"publisher":"zmq","topic":"kv-events",'
                    f'"endpoint":"tcp://*:{kv_event_port}",'
                    f'"enable_kv_cache_events":true}}'
                ),
            ],
            env=_make_process_env(log_level="info", DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.status_code == 200,
                )
            ],
            timeout=600,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_prepare_log_dir(request, "approx-vllm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def approx_vllm_services(request, approx_runtime_services):
    frontend_port, vllm_port, kv_event_port = allocate_ports(count=3, start_port=10000)
    with ApproxVLLMWorkerProcess(
        request, system_port=vllm_port, kv_event_port=kv_event_port
    ):
        with ApproxFrontendProcess(
            request, frontend_port=frontend_port
        ) as frontend_proc:
            yield frontend_port, frontend_proc


@pytest.mark.timeout(240)
@pytest.mark.vllm
@pytest.mark.model(VLLM_MM_MODEL)
def test_approx_vllm_mm_single_image_overlap(approx_vllm_services):
    """Same image repeated should produce routing overlap > 0 on the 2nd request."""
    frontend_port, router_proc = approx_vllm_services
    payload = _payload_with_images(VLLM_MM_MODEL, [_make_data_uri(_COLORS[0])])
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-vllm-single"
    )


@pytest.mark.timeout(240)
@pytest.mark.vllm
@pytest.mark.model(VLLM_MM_MODEL)
def test_approx_vllm_mm_different_images_no_false_overlap(approx_vllm_services):
    """Different images should NOT create spurious overlap at the MM block."""
    frontend_port, router_proc = approx_vllm_services
    payload_a = _payload_with_images(VLLM_MM_MODEL, [_make_data_uri(_COLORS[0])])
    payload_b = _payload_with_images(VLLM_MM_MODEL, [_make_data_uri(_COLORS[1])])
    _assert_distinct_payloads_no_false_overlap(
        frontend_port, router_proc, payload_a, payload_b, "approx-vllm-diff"
    )


@pytest.mark.timeout(240)
@pytest.mark.vllm
@pytest.mark.model(VLLM_MM_MODEL)
def test_approx_vllm_mm_multi_image_overlap(approx_vllm_services):
    """Repeated multi-image request should produce overlap on the 2nd request."""
    frontend_port, router_proc = approx_vllm_services
    payload = _payload_with_images(VLLM_MM_MODEL, [_make_data_uri(c) for c in _COLORS])
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-vllm-multi"
    )


# --------------------------------------------------------------------------
# TRT-LLM approx MM routing
# --------------------------------------------------------------------------


class ApproxTRTLLMWorkerProcess(ManagedProcess):
    """TRT-LLM backend worker for approx-mode tests (no __internal suffix)."""

    def __init__(self, request, *, system_port: int):
        from tests.mm_router.test_mm_router_e2e import NAMESPACE, _check_ready

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.trtllm",
                "--model-path",
                TRTLLM_MM_MODEL,
                "--served-model-name",
                TRTLLM_MM_MODEL,
                "--endpoint",
                f"dyn://{NAMESPACE}.trtllm.generate",
                "--modality",
                "multimodal",
                "--publish-events-and-metrics",
                "--kv-block-size",
                str(BLOCK_SIZE),
            ],
            env=_make_process_env(log_level="info", DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.trtllm"],
            log_dir=_prepare_log_dir(request, "approx-trtllm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def approx_trtllm_services(request, approx_runtime_services):
    if TRTLLM_MM_MODEL is None:
        pytest.skip("TRTLLM test fixtures not available")
    frontend_port, trtllm_port = allocate_ports(count=2, start_port=10100)
    with ApproxTRTLLMWorkerProcess(request, system_port=trtllm_port):
        with ApproxFrontendProcess(
            request, frontend_port=frontend_port
        ) as frontend_proc:
            yield frontend_port, frontend_proc


@pytest.mark.timeout(240)
@pytest.mark.trtllm
@_maybe_model_mark(TRTLLM_MM_MODEL)
def test_approx_trtllm_mm_single_image_overlap(approx_trtllm_services):
    """Same image repeated should produce routing overlap > 0 on the 2nd request."""
    frontend_port, router_proc = approx_trtllm_services
    payload = _payload_with_images(TRTLLM_MM_MODEL, [_make_data_uri(_COLORS[0])])
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-trtllm-single"
    )


@pytest.mark.timeout(240)
@pytest.mark.trtllm
@_maybe_model_mark(TRTLLM_MM_MODEL)
def test_approx_trtllm_mm_different_images_no_false_overlap(approx_trtllm_services):
    """Different images should NOT create spurious overlap at the MM block."""
    frontend_port, router_proc = approx_trtllm_services
    payload_a = _payload_with_images(TRTLLM_MM_MODEL, [_make_data_uri(_COLORS[0])])
    payload_b = _payload_with_images(TRTLLM_MM_MODEL, [_make_data_uri(_COLORS[1])])
    _assert_distinct_payloads_no_false_overlap(
        frontend_port, router_proc, payload_a, payload_b, "approx-trtllm-diff"
    )


@pytest.mark.timeout(240)
@pytest.mark.trtllm
@_maybe_model_mark(TRTLLM_MM_MODEL)
def test_approx_trtllm_mm_multi_image_overlap(approx_trtllm_services):
    """Repeated multi-image request should produce overlap on the 2nd request."""
    frontend_port, router_proc = approx_trtllm_services
    payload = _payload_with_images(
        TRTLLM_MM_MODEL, [_make_data_uri(c) for c in _COLORS]
    )
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-trtllm-multi"
    )


# --------------------------------------------------------------------------
# SGLang approx MM routing
# --------------------------------------------------------------------------
#
# SGLang serves multimodal from a single aggregated worker (same shape as
# vLLM and TRT-LLM in this file). The launch mirrors
# ``examples/backends/sglang/launch/agg_vision.sh``.


SGLANG_MM_MODEL = os.getenv("DYN_TEST_SGLANG_MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
SGLANG_CHAT_TEMPLATE = os.getenv("DYN_TEST_SGLANG_CHAT_TEMPLATE", "qwen2-vl")


class ApproxSGLangMMWorkerProcess(ManagedProcess):
    """SGLang aggregated multimodal worker for approx-mode tests."""

    def __init__(self, request, *, system_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                SGLANG_MM_MODEL,
                "--served-model-name",
                SGLANG_MM_MODEL,
                "--chat-template",
                SGLANG_CHAT_TEMPLATE,
                "--page-size",
                "16",
                "--tp",
                "1",
                "--trust-remote-code",
                "--skip-tokenizer-init",
            ],
            env=_make_process_env(log_level="info", DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.status_code == 200,
                )
            ],
            timeout=900,
            straggler_commands=["-m dynamo.sglang"],
            log_dir=_prepare_log_dir(request, "approx-sglang-mm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def approx_sglang_services(request, approx_runtime_services):
    try:
        import sglang  # noqa: F401
    except ImportError:
        pytest.skip("sglang not installed in this env")

    frontend_port, sglang_port = allocate_ports(count=2, start_port=10200)
    with ApproxSGLangMMWorkerProcess(request, system_port=sglang_port):
        with ApproxFrontendProcess(
            request, frontend_port=frontend_port
        ) as frontend_proc:
            yield frontend_port, frontend_proc


@pytest.mark.timeout(600)
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MM_MODEL)
def test_approx_sglang_mm_single_image_overlap(approx_sglang_services):
    """Same image repeated should produce routing overlap > 0 on the 2nd request."""
    frontend_port, router_proc = approx_sglang_services
    payload = _payload_with_images(SGLANG_MM_MODEL, [_make_data_uri(_COLORS[0])])
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-sglang-single"
    )


@pytest.mark.timeout(600)
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MM_MODEL)
def test_approx_sglang_mm_different_images_no_false_overlap(approx_sglang_services):
    """Different images should NOT create spurious overlap at the MM block."""
    frontend_port, router_proc = approx_sglang_services
    payload_a = _payload_with_images(SGLANG_MM_MODEL, [_make_data_uri(_COLORS[0])])
    payload_b = _payload_with_images(SGLANG_MM_MODEL, [_make_data_uri(_COLORS[1])])
    _assert_distinct_payloads_no_false_overlap(
        frontend_port, router_proc, payload_a, payload_b, "approx-sglang-diff"
    )


@pytest.mark.timeout(600)
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MM_MODEL)
def test_approx_sglang_mm_multi_image_overlap(approx_sglang_services):
    """Repeated multi-image request should produce overlap on the 2nd request."""
    frontend_port, router_proc = approx_sglang_services
    payload = _payload_with_images(
        SGLANG_MM_MODEL, [_make_data_uri(c) for c in _COLORS]
    )
    _assert_repeat_overlap_grows(
        frontend_port, router_proc, payload, "approx-sglang-multi"
    )
