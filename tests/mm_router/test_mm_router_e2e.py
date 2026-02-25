# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for multimodal KV routing with TRT-LLM.

Architecture:
  Frontend -> MM Router Worker -> TRT-LLM Worker
                (computes mm_hash)   (publishes KV events)

This test validates MM-aware routing by sending repeated multimodal requests and
asserting that router overlap is greater than 1 block (regression guard against
text-only/partially-matched hash paths that typically show 1/N overlap).
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import time
from io import BytesIO
from typing import Any, Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

TRTLLM_MM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
TRTLLM_MM_MODEL_TYPE = "qwen2_vl"
BLOCK_SIZE = 32
NAMESPACE = "test-mm"
# Broad guardrails for TRT-LLM + Qwen2-VL-2B under block size 32.
THREE_IMAGE_TOTAL_BLOCKS_RANGE = (80, 520)
SINGLE_IMAGE_TOTAL_BLOCKS_RANGE = (20, 260)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.model(TRTLLM_MM_MODEL),
]

_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]
_ALT_COLORS = [
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]
_SINGLE_IMAGE_FRESH_COLOR = (123, 45, 67)
_DOUBLE_IMAGE_FRESH_COLOR = (89, 210, 34)
_STAIRCASE_IMAGE_FRESH_COLOR = (17, 99, 201)
_SWAP_ORDER_FRESH_COLORS = [(14, 141, 77), (211, 66, 101), (44, 91, 233)]
# Contract with lib/llm/src/kv_router/push_router.rs "[ROUTING]" debug log.
# Keep this parser in sync with the router log format.
_ROUTING_RECORD_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(\d+)/(\d+)\s*blocks overlap"
)


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _make_process_env(log_level: str = "debug", **extra) -> dict[str, str]:
    env = os.environ.copy()
    env["DYN_LOG"] = log_level
    env["DYN_NAMESPACE"] = NAMESPACE
    env["DYN_REQUEST_PLANE"] = "nats"
    env.update(extra)
    return env


def _prepare_log_dir(request, suffix: str) -> str:
    log_dir = f"{request.node.name}_{suffix}"
    shutil.rmtree(log_dir, ignore_errors=True)
    return log_dir


_COMMON_PROCESS_KWARGS: dict[str, Any] = {
    "display_output": True,
    "terminate_all_matching_process_names": False,
}


class TRTLLMWorkerProcess(ManagedProcess):
    """TRT-LLM backend worker that emits KV events."""

    def __init__(self, request, *, system_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.trtllm",
                "--model-path",
                TRTLLM_MM_MODEL,
                "--served-model-name",
                f"{TRTLLM_MM_MODEL}__internal",
                "--endpoint",
                f"dyn://{NAMESPACE}.trtllm.generate",
                "--modality",
                "multimodal",
                "--publish-events-and-metrics",
                "--kv-block-size",
                str(BLOCK_SIZE),
            ],
            env=_make_process_env(DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.trtllm"],
            log_dir=_prepare_log_dir(request, "trtllm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


class TRTLLMMMRouterWorkerProcess(ManagedProcess):
    """TRT-LLM MM router worker."""

    def __init__(self, request, *, system_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "examples.backends.trtllm.mm_router_worker",
                "--model",
                TRTLLM_MM_MODEL,
                "--model-type",
                TRTLLM_MM_MODEL_TYPE,
                "--namespace",
                NAMESPACE,
                "--component",
                "mm_router",
                "--endpoint",
                "generate",
                "--downstream-component",
                "trtllm",
                "--downstream-endpoint",
                "generate",
                "--block-size",
                str(BLOCK_SIZE),
            ],
            env=_make_process_env(
                DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]',
                DYN_SYSTEM_PORT=str(system_port),
            ),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=240,
            straggler_commands=["mm_router_worker"],
            log_dir=_prepare_log_dir(request, "trtllm-mm-router"),
            **_COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    """Frontend HTTP ingress."""

    def __init__(self, request, *, frontend_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(frontend_port),
                "--router-mode",
                "round-robin",
            ],
            env=_make_process_env(log_level="info"),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "trtllm-mm-frontend"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def mm_runtime_services(request):
    with NatsServer(request, port=0) as nats, EtcdServer(request, port=0) as etcd:
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"
        yield
        os.environ.pop("NATS_SERVER", None)
        os.environ.pop("ETCD_ENDPOINTS", None)


@pytest.fixture(scope="module")
def start_trtllm_mm_services(
    request, mm_runtime_services
) -> Generator[tuple[int, ManagedProcess], None, None]:
    frontend_port, trtllm_port, router_port = allocate_ports(count=3, start_port=10000)

    with TRTLLMWorkerProcess(request, system_port=trtllm_port):
        time.sleep(15)
        with TRTLLMMMRouterWorkerProcess(
            request, system_port=router_port
        ) as router_proc:
            time.sleep(5)
            with FrontendProcess(request, frontend_port=frontend_port):
                yield frontend_port, router_proc


def _make_data_uri(color: tuple[int, int, int], size: int = 1024) -> str:
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_payload(
    image_uris: list[str], prompt: str = "Describe what you see."
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})

    return {
        "model": TRTLLM_MM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
    }


def _extract_routing_records(log_text: str) -> list[tuple[int, int]]:
    return [
        (int(overlap), int(total))
        for overlap, total in _ROUTING_RECORD_PATTERN.findall(log_text)
    ]


def _wait_for_new_routing_score(
    router_proc: ManagedProcess,
    start_offset: int,
    pre_request_routing_count: int,
    timeout_s: float = 120.0,
) -> tuple[int, int, str]:
    deadline = time.time() + timeout_s
    last_segment = ""

    while time.time() < deadline:
        full_logs = router_proc.read_logs()
        segment = full_logs[start_offset:]
        last_segment = segment
        records = _extract_routing_records(full_logs)
        if len(records) >= pre_request_routing_count + 1:
            overlap, total = records[-1]
            return overlap, total, segment
        time.sleep(1)

    fallback_records = _extract_routing_records(last_segment)
    if fallback_records:
        overlap, total = fallback_records[-1]
        return overlap, total, last_segment
    return 0, 0, last_segment


def _send_request_get_overlap(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> tuple[int, int, str]:
    """Send one request and read the new routing overlap score."""
    pre_request_logs = router_proc.read_logs()
    start_offset = len(pre_request_logs)
    pre_request_routing_count = len(_extract_routing_records(pre_request_logs))
    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=240,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "choices" in data, f"Missing choices in response: {data}"

    overlap, total, segment = _wait_for_new_routing_score(
        router_proc=router_proc,
        start_offset=start_offset,
        pre_request_routing_count=pre_request_routing_count,
        timeout_s=120,
    )
    print(f"[MM_ROUTER_E2E] {label}: current={overlap}/{total}")
    return overlap, total, segment


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_text_only_overlap_repeated_prompt(
    start_trtllm_mm_services, predownload_models
):
    """Text-only routing should increase overlap on repeat and then stabilize."""
    frontend_port, router_proc = start_trtllm_mm_services

    prompt = (
        "TEXT routing e2e unique case zeta-7f31. "
        "Repeat this sentence to force multiple KV blocks. "
    ) * 80
    payload = _build_payload([], prompt=prompt)

    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "text_only_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "text_only_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, router_proc, payload, "text_only_req3"
    )

    assert total_1 > 0 and total_2 > 0 and total_3 > 0, (
        f"Expected non-zero total blocks for text-only request, got "
        f"{total_1}, {total_2}, {total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert abs(total_1 - total_2) <= 2 and abs(total_2 - total_3) <= 2, (
        f"Expected text-only total blocks to remain stable across repeats, got "
        f"req1={total_1}, req2={total_2}, req3={total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second text-only overlap > first, got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third text-only overlap == second, got "
        f"req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_repeated_three_images(
    start_trtllm_mm_services, predownload_models
):
    """For repeated same 3-image request: low first overlap, then increase, then stable."""
    frontend_port, router_proc = start_trtllm_mm_services

    image_uris = [_make_data_uri(c) for c in _COLORS]
    payload = _build_payload(
        image_uris, prompt="MM routing e2e: repeated same 3-image request."
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_3_images_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_3_images_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_3_images_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got "
        f"req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for same 3 images (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_repeated_single_image(
    start_trtllm_mm_services, predownload_models
):
    """For repeated same single-image request: low first overlap, then increase, then stable."""
    frontend_port, router_proc = start_trtllm_mm_services

    payload = _build_payload(
        [_make_data_uri(_SINGLE_IMAGE_FRESH_COLOR)],
        prompt="MM routing e2e: repeated same single-image request.",
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_single_image_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_single_image_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_single_image_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got "
        f"req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    low, high = SINGLE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for same 1 image (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_repeated_two_identical_images(
    start_trtllm_mm_services, predownload_models
):
    """For repeated same two-identical-image request: low first overlap, then increase, then stable."""
    frontend_port, router_proc = start_trtllm_mm_services

    image_uri = _make_data_uri(_DOUBLE_IMAGE_FRESH_COLOR)
    payload = _build_payload(
        [image_uri, image_uri],
        prompt="MM routing e2e: repeated same two-identical-image request.",
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_two_identical_images_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_two_identical_images_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, router_proc, payload, "same_two_identical_images_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got "
        f"req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_staircase_single_to_double_to_triple_identical_image(
    start_trtllm_mm_services, predownload_models
):
    """Single->double->triple identical image requests follow prefix-overlap semantics."""
    frontend_port, router_proc = start_trtllm_mm_services

    image_uri = _make_data_uri(_STAIRCASE_IMAGE_FRESH_COLOR)
    staircase_prompt = "MM routing e2e: staircase."
    payload_single = _build_payload([image_uri], prompt=staircase_prompt)
    payload_double = _build_payload([image_uri, image_uri], prompt=staircase_prompt)
    payload_triple = _build_payload(
        [image_uri, image_uri, image_uri], prompt=staircase_prompt
    )

    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, payload_single, "staircase_1x_image"
    )
    overlap_2, total_2, segment_2 = _send_request_get_overlap(
        frontend_port, router_proc, payload_double, "staircase_2x_image"
    )
    time.sleep(1)
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, router_proc, payload_triple, "staircase_3x_image"
    )

    assert overlap_2 > overlap_1, (
        f"Expected overlap to increase from 1 image to 2 images, got "
        f"1x={overlap_1}/{total_1}, 2x={overlap_2}/{total_2}.\n"
        f"Recent router logs:\n{segment_2[-4000:]}"
    )
    assert abs(overlap_3 - overlap_2) <= 1, (
        "Expected first 3-image request overlap to stay near 2-image overlap "
        "(third-image suffix is cold on first 3-image request), got "
        f"2x={overlap_2}/{total_2}, 3x={overlap_3}/{total_3}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )

    total_step_12 = total_2 - total_1
    total_step_23 = total_3 - total_2
    assert abs(total_step_12 - total_step_23) <= 4, (
        "Expected similar total-block increment per additional identical image, got "
        f"step(1->2)={total_step_12}, step(2->3)={total_step_23}.\n"
        f"Recent router logs:\n{segment_3[-4000:]}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_diff_images_less_than_same(
    start_trtllm_mm_services, predownload_models
):
    """Different images should produce lower overlap than repeated identical images."""
    frontend_port, router_proc = start_trtllm_mm_services
    baseline_payload = _build_payload(
        [_make_data_uri(c) for c in _COLORS],
        prompt="MM routing e2e: baseline same-images overlap.",
    )
    overlap_baseline_1, total_baseline_1, _ = _send_request_get_overlap(
        frontend_port, router_proc, baseline_payload, "baseline_same_images_req1"
    )
    overlap_baseline_2, total_baseline_2, segment_baseline = _send_request_get_overlap(
        frontend_port, router_proc, baseline_payload, "baseline_same_images_req2"
    )
    overlap_baseline = max(overlap_baseline_1, overlap_baseline_2)
    total_baseline = total_baseline_2
    assert abs(total_baseline_1 - total_baseline_2) <= 4, (
        "Expected total blocks to stay nearly identical for repeated same request, "
        f"got req1={total_baseline_1}, req2={total_baseline_2}"
    )
    assert overlap_baseline >= 2, (
        f"Baseline overlap did not reach 2 blocks. "
        f"got {overlap_baseline}/{total_baseline}.\n"
        f"Recent router logs:\n{segment_baseline[-4000:]}"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_baseline <= high, (
        f"Unexpected total blocks for baseline same-images request: "
        f"got {total_baseline}, expected in [{low}, {high}]"
    )

    probe_payload = _build_payload(
        [_make_data_uri(c) for c in _ALT_COLORS],
        prompt="MM routing e2e: baseline same-images overlap.",
    )
    overlap_probe, total_probe, segment_probe = _send_request_get_overlap(
        frontend_port, router_proc, probe_payload, "probe_different_images_req1"
    )
    assert (
        total_probe > 0
    ), f"No routing score found.\nRecent logs:\n{segment_probe[-4000:]}"
    assert abs(total_probe - total_baseline) <= 4, (
        f"Expected different-images total blocks to stay near baseline, "
        f"got different={total_probe}, baseline={total_baseline}"
    )
    assert overlap_probe < overlap_baseline, (
        f"Expected different-images overlap < baseline overlap, "
        f"got different={overlap_probe}/{total_probe}, "
        f"baseline={overlap_baseline}/{total_baseline}.\n"
        f"Recent router logs:\n{segment_probe[-4000:]}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_same_images_different_prompt_less_than_same_prompt(
    start_trtllm_mm_services, predownload_models
):
    """Same images but different prompt should produce lower overlap than repeated same prompt."""
    frontend_port, router_proc = start_trtllm_mm_services
    baseline_payload = _build_payload(
        [_make_data_uri(c) for c in _COLORS],
        prompt="MM routing e2e: prompt-sensitive baseline alpha.",
    )
    overlap_baseline_1, total_baseline_1, _ = _send_request_get_overlap(
        frontend_port,
        router_proc,
        baseline_payload,
        "baseline_same_images_prompt_a_req1",
    )
    overlap_baseline_2, total_baseline_2, segment_baseline = _send_request_get_overlap(
        frontend_port,
        router_proc,
        baseline_payload,
        "baseline_same_images_prompt_a_req2",
    )
    overlap_baseline = max(overlap_baseline_1, overlap_baseline_2)
    total_baseline = total_baseline_2
    assert abs(total_baseline_1 - total_baseline_2) <= 4, (
        "Expected total blocks to stay nearly identical for repeated same request, "
        f"got req1={total_baseline_1}, req2={total_baseline_2}"
    )
    assert overlap_baseline >= 2, (
        f"Baseline overlap did not reach 2 blocks. "
        f"got {overlap_baseline}/{total_baseline}.\n"
        f"Recent router logs:\n{segment_baseline[-4000:]}"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_baseline <= high, (
        f"Unexpected total blocks for baseline same-images request: "
        f"got {total_baseline}, expected in [{low}, {high}]"
    )

    probe_payload = _build_payload(
        [_make_data_uri(c) for c in _COLORS],
        prompt="MM routing e2e: prompt-sensitive baseline omega.",
    )
    overlap_probe, total_probe, segment_probe = _send_request_get_overlap(
        frontend_port, router_proc, probe_payload, "probe_same_images_prompt_b_req1"
    )
    assert (
        total_probe > 0
    ), f"No routing score found.\nRecent logs:\n{segment_probe[-4000:]}"
    assert abs(total_probe - total_baseline) <= 4, (
        f"Expected different-prompt total blocks to stay near baseline, "
        f"got different_prompt={total_probe}, baseline={total_baseline}"
    )
    assert overlap_probe < overlap_baseline, (
        f"Expected different-prompt overlap < baseline overlap, "
        f"got different_prompt={overlap_probe}/{total_probe}, "
        f"baseline={overlap_baseline}/{total_baseline}.\n"
        f"Recent router logs:\n{segment_probe[-4000:]}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.nightly
def test_trtllm_mm_overlap_swapped_order_less_than_same_order(
    start_trtllm_mm_services, predownload_models
):
    """Swapping order of three distinct images should result in near-zero overlap."""
    frontend_port, router_proc = start_trtllm_mm_services
    ordered_uris = [_make_data_uri(c) for c in _SWAP_ORDER_FRESH_COLORS]
    ordered_payload = _build_payload(
        ordered_uris, prompt="MM routing e2e: order sensitivity ordered baseline."
    )
    swapped_payload = _build_payload(
        list(reversed(ordered_uris)),
        prompt="MM routing e2e: order sensitivity ordered baseline.",
    )

    overlap_ordered_1, total_ordered_1, _ = _send_request_get_overlap(
        frontend_port,
        router_proc,
        ordered_payload,
        "ordered_distinct_images_req1",
    )
    overlap_ordered_2, total_ordered_2, segment_ordered_2 = _send_request_get_overlap(
        frontend_port,
        router_proc,
        ordered_payload,
        "ordered_distinct_images_req2",
    )
    overlap_swapped, total_swapped, segment_swapped = _send_request_get_overlap(
        frontend_port,
        router_proc,
        swapped_payload,
        "swapped_distinct_images_req1",
    )

    assert overlap_ordered_2 > overlap_ordered_1, (
        "Expected repeated identical order to increase overlap before swapped-order "
        "probe, "
        f"got req1={overlap_ordered_1}/{total_ordered_1}, "
        f"req2={overlap_ordered_2}/{total_ordered_2}.\n"
        f"Recent router logs:\n{segment_ordered_2[-4000:]}"
    )
    assert abs(total_swapped - total_ordered_2) <= 4, (
        f"Expected swapped-order total blocks to stay near ordered baseline, "
        f"got swapped={total_swapped}, ordered={total_ordered_2}"
    )
    assert overlap_swapped <= 1, (
        "Expected near-zero overlap for swapped order of three distinct images "
        f"(allowing 1 shared text block), got {overlap_swapped}/{total_swapped}.\n"
        f"Recent router logs:\n{segment_swapped[-4000:]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
