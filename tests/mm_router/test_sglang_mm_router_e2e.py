# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression test for SGLang multimodal KV routing.

Architecture:
  Frontend (KV router + SGLang chat processor) -> SGLang worker

Unlike the vLLM MM router tests, this path does not launch a standalone
MM router worker. The frontend performs multimodal routing inline and logs
the overlap score directly.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Any, Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports, deallocate_ports

SGLANG_MM_MODEL = os.getenv("DYN_TEST_SGLANG_MM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
BLOCK_SIZE = 16
REQUEST_PLANE = "tcp"
THREE_IMAGE_TOTAL_BLOCKS_RANGE = (180, 340)
SINGLE_IMAGE_TOTAL_BLOCKS_RANGE = (60, 160)
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
_HTTP_IMAGE_COLORS = [(180, 30, 90), (30, 180, 90), (90, 30, 180)]
_ROUTING_RECORD_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(\d+)/(\d+)\s*blocks overlap"
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.model(SGLANG_MM_MODEL),
]


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _make_process_env(
    *,
    namespace: str,
    request_plane: str,
    log_level: str = "debug",
    **extra: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["DYN_LOG"] = log_level
    env["DYN_NAMESPACE"] = namespace
    env["DYN_REQUEST_PLANE"] = request_plane
    env["PYTHONUNBUFFERED"] = "1"
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


class SGLangMMWorkerProcess(ManagedProcess):
    """Aggregated SGLang worker that emits KV events."""

    def __init__(
        self,
        request,
        *,
        namespace: str,
        request_plane: str,
        system_port: int,
        kv_event_port: int,
    ):
        kv_events_config = (
            '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%d"}'
            % kv_event_port
        )
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                SGLANG_MM_MODEL,
                "--served-model-name",
                SGLANG_MM_MODEL,
                "--page-size",
                str(BLOCK_SIZE),
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.7",
                "--disable-cuda-graph",
                "--trust-remote-code",
                "--kv-events-config",
                kv_events_config,
            ],
            env=_make_process_env(
                namespace=namespace,
                request_plane=request_plane,
                log_level="info",
                DYN_SYSTEM_PORT=str(system_port),
                PYTHONHASHSEED="0",
            ),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.sglang"],
            log_dir=_prepare_log_dir(request, "sglang-mm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    """Frontend with embedded KV routing and SGLang chat processor."""

    def __init__(
        self,
        request,
        *,
        namespace: str,
        request_plane: str,
        frontend_port: int,
    ):
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
                "--discovery-backend",
                "etcd",
                "--namespace",
                namespace,
                "--dyn-chat-processor",
                "sglang",
            ],
            env=_make_process_env(
                namespace=namespace,
                request_plane=request_plane,
                log_level="debug",
            ),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "sglang-mm-frontend"),
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
def start_sglang_mm_services(
    request,
    mm_runtime_services,
) -> Generator[tuple[int, ManagedProcess], None, None]:
    _ = mm_runtime_services

    namespace = f"test-sglang-mm-{uuid.uuid4().hex[:10]}"
    frontend_port, system_port, kv_event_port = allocate_ports(count=3, start_port=10000)
    request.addfinalizer(
        lambda: deallocate_ports([frontend_port, system_port, kv_event_port])
    )

    with SGLangMMWorkerProcess(
        request,
        namespace=namespace,
        request_plane=REQUEST_PLANE,
        system_port=system_port,
        kv_event_port=kv_event_port,
    ):
        with FrontendProcess(
            request,
            namespace=namespace,
            request_plane=REQUEST_PLANE,
            frontend_port=frontend_port,
        ) as frontend_proc:
            yield frontend_port, frontend_proc


def _make_png_bytes(color: tuple[int, int, int], size: int = 1024) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_uri(color: tuple[int, int, int], size: int = 1024) -> str:
    b64 = base64.b64encode(_make_png_bytes(color, size)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_payload(
    image_uris: list[str], prompt: str = "Describe what you see."
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})

    return {
        "model": SGLANG_MM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
    }


def _make_image_handler(image_map: dict[str, bytes]) -> type:
    """Create an HTTP handler class that serves images from the given map."""

    class _ImageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = image_map.get(self.path)
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    return _ImageHandler


@pytest.fixture(scope="module")
def http_image_server() -> Generator[list[str], None, None]:
    """Serve pre-generated PNG images over HTTP for the duration of the module."""
    (port,) = allocate_ports(count=1, start_port=18000)

    image_map: dict[str, bytes] = {}
    for i, color in enumerate(_HTTP_IMAGE_COLORS):
        image_map[f"/image_{i}.png"] = _make_png_bytes(color)

    server = HTTPServer(("127.0.0.1", port), _make_image_handler(image_map))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    urls = [
        f"http://127.0.0.1:{port}/image_{i}.png" for i in range(len(_HTTP_IMAGE_COLORS))
    ]
    yield urls

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _extract_routing_records(log_text: str) -> list[tuple[int, int]]:
    return [
        (int(overlap), int(total))
        for overlap, total in _ROUTING_RECORD_PATTERN.findall(log_text)
    ]


def _wait_for_new_routing_score(
    frontend_proc: ManagedProcess,
    start_offset: int,
    pre_request_routing_count: int,
    timeout_s: float = 120.0,
) -> tuple[int, int, str]:
    deadline = time.time() + timeout_s
    last_segment = ""

    while time.time() < deadline:
        full_logs = frontend_proc.read_logs()
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
    frontend_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> tuple[int, int, str]:
    pre_request_logs = frontend_proc.read_logs()
    start_offset = len(pre_request_logs)
    pre_request_routing_count = len(_extract_routing_records(pre_request_logs))

    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "choices" in data, f"Missing choices in response: {data}"

    overlap, total, segment = _wait_for_new_routing_score(
        frontend_proc=frontend_proc,
        start_offset=start_offset,
        pre_request_routing_count=pre_request_routing_count,
        timeout_s=120,
    )
    print(f"[SGLANG_MM_ROUTER_E2E] {label}: current={overlap}/{total}")
    time.sleep(1)
    return overlap, total, segment


@pytest.mark.pre_merge
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_repeated_single_image(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Repeated identical image requests should reuse more than one KV block."""
    frontend_port, frontend_proc = start_sglang_mm_services

    payload = _build_payload(
        [_make_data_uri(_SINGLE_IMAGE_FRESH_COLOR)],
        prompt="SGLang MM routing e2e: repeated same single-image request.",
    )

    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_single_image_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_single_image_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_single_image_req3"
    )

    assert total_1 > 0 and total_2 > 0 and total_3 > 0, (
        f"Expected non-zero total blocks, got {total_1}, {total_2}, {total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert abs(total_1 - total_2) <= 2 and abs(total_2 - total_3) <= 2, (
        "Expected total blocks to remain stable across repeated requests, got "
        f"req1={total_1}, req2={total_2}, req3={total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_1 <= 1, (
        f"Expected first overlap <= 1 on cold cache, got req1={overlap_1}/{total_1}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > 1, (
        f"Expected repeated multimodal request to reuse more than one block, got req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    low, high = SINGLE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for same 1 image (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_text_only_overlap_repeated_prompt(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Text-only routing should increase overlap on repeat and then stabilize."""
    frontend_port, frontend_proc = start_sglang_mm_services

    prompt = (
        "TEXT routing e2e unique case zeta-7f31. "
        "Repeat this sentence to force multiple KV blocks. "
    ) * 80
    payload = _build_payload([], prompt=prompt)

    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "text_only_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "text_only_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "text_only_req3"
    )

    assert total_1 > 0 and total_2 > 0 and total_3 > 0, (
        f"Expected non-zero total blocks for text-only request, got "
        f"{total_1}, {total_2}, {total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert abs(total_1 - total_2) <= 2 and abs(total_2 - total_3) <= 2, (
        f"Expected text-only total blocks to remain stable across repeats, got "
        f"req1={total_1}, req2={total_2}, req3={total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second text-only overlap > first, got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third text-only overlap == second, got "
        f"req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_repeated_three_images(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """For repeated same 3-image request: low first overlap, then increase, then stable."""
    frontend_port, frontend_proc = start_sglang_mm_services

    image_uris = [_make_data_uri(c) for c in _COLORS]
    payload = _build_payload(
        image_uris, prompt="MM routing e2e: repeated same 3-image request."
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_3_images_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_3_images_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_3_images_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for same 3 images (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_repeated_two_identical_images(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """For repeated same two-identical-image request: low first overlap, then increase, then stable."""
    frontend_port, frontend_proc = start_sglang_mm_services

    image_uri = _make_data_uri(_DOUBLE_IMAGE_FRESH_COLOR)
    payload = _build_payload(
        [image_uri, image_uri],
        prompt="MM routing e2e: repeated same two-identical-image request.",
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_two_identical_images_req1"
    )
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_two_identical_images_req2"
    )
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "same_two_identical_images_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_staircase_single_to_double_to_triple_identical_image(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Single->double->triple identical image requests grow linearly in total blocks.

    SGLang pads multimodal routing tokens with request-level MM hash values, so a
    1x-image request does not share the routed image prefix of the corresponding
    2x- or 3x-image request. Unlike the vLLM MM router, these staircase probes
    should remain near-cold from an overlap perspective while total blocks grow
    by a stable amount for each additional identical image.
    """
    frontend_port, frontend_proc = start_sglang_mm_services

    image_uri = _make_data_uri(_STAIRCASE_IMAGE_FRESH_COLOR)
    staircase_prompt = "MM routing e2e: staircase."
    payload_single = _build_payload([image_uri], prompt=staircase_prompt)
    payload_double = _build_payload([image_uri, image_uri], prompt=staircase_prompt)
    payload_triple = _build_payload(
        [image_uri, image_uri, image_uri], prompt=staircase_prompt
    )

    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload_single, "staircase_1x_image"
    )
    time.sleep(1)
    overlap_2, total_2, segment_2 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload_double, "staircase_2x_image"
    )
    time.sleep(1)
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload_triple, "staircase_3x_image"
    )

    assert overlap_1 <= 1 and overlap_2 <= 1 and overlap_3 <= 1, (
        "Expected near-cold overlap across 1x/2x/3x identical-image staircase "
        "requests because SGLang routing pads image spans with request-level MM "
        f"hashes, got 1x={overlap_1}/{total_1}, 2x={overlap_2}/{total_2}, "
        f"3x={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )

    assert total_2 > total_1 and total_3 > total_2, (
        "Expected total blocks to increase with each additional identical image, "
        f"got 1x={total_1}, 2x={total_2}, 3x={total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )

    total_step_12 = total_2 - total_1
    total_step_23 = total_3 - total_2
    assert abs(total_step_12 - total_step_23) <= 4, (
        "Expected similar total-block increment per additional identical image, got "
        f"step(1->2)={total_step_12}, step(2->3)={total_step_23}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    low, high = SINGLE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_1 <= high, (
        f"Unexpected total blocks for staircase 1 image (1024): "
        f"got {total_1}, expected in [{low}, {high}]"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for staircase 3 images (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_diff_images_less_than_same(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Different images should produce lower overlap than repeated identical images."""
    frontend_port, frontend_proc = start_sglang_mm_services
    baseline_payload = _build_payload(
        [_make_data_uri(c) for c in _COLORS],
        prompt="MM routing e2e: baseline same-images overlap.",
    )
    overlap_baseline_1, total_baseline_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, baseline_payload, "baseline_same_images_req1"
    )
    overlap_baseline_2, total_baseline_2, segment_baseline = _send_request_get_overlap(
        frontend_port, frontend_proc, baseline_payload, "baseline_same_images_req2"
    )
    overlap_baseline = max(overlap_baseline_1, overlap_baseline_2)
    total_baseline = total_baseline_2
    assert abs(total_baseline_1 - total_baseline_2) <= 4, (
        "Expected total blocks to stay nearly identical for repeated same request, "
        f"got req1={total_baseline_1}, req2={total_baseline_2}"
    )
    assert overlap_baseline >= 2, (
        f"Baseline overlap did not reach 2 blocks. got {overlap_baseline}/{total_baseline}.\n"
        f"Recent frontend logs:\n{segment_baseline[-4000:]}"
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
        frontend_port, frontend_proc, probe_payload, "probe_different_images_req1"
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
        f"Recent frontend logs:\n{segment_probe[-4000:]}"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_same_images_different_prompt_less_than_same_prompt(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Same images but different prompt should produce lower overlap than repeated same prompt."""
    frontend_port, frontend_proc = start_sglang_mm_services
    baseline_payload = _build_payload(
        [_make_data_uri(c) for c in _COLORS],
        prompt="MM routing e2e: prompt-sensitive baseline alpha.",
    )
    overlap_baseline_1, total_baseline_1, _ = _send_request_get_overlap(
        frontend_port,
        frontend_proc,
        baseline_payload,
        "baseline_same_images_prompt_a_req1",
    )
    overlap_baseline_2, total_baseline_2, segment_baseline = _send_request_get_overlap(
        frontend_port,
        frontend_proc,
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
        f"Baseline overlap did not reach 2 blocks. got {overlap_baseline}/{total_baseline}.\n"
        f"Recent frontend logs:\n{segment_baseline[-4000:]}"
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
        frontend_port, frontend_proc, probe_payload, "probe_same_images_prompt_b_req1"
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
        f"Recent frontend logs:\n{segment_probe[-4000:]}"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_swapped_order_less_than_same_order(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
):
    """Swapping order of three distinct images should result in near-zero overlap."""
    frontend_port, frontend_proc = start_sglang_mm_services
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
        frontend_proc,
        ordered_payload,
        "ordered_distinct_images_req1",
    )
    overlap_ordered_2, total_ordered_2, segment_ordered_2 = _send_request_get_overlap(
        frontend_port,
        frontend_proc,
        ordered_payload,
        "ordered_distinct_images_req2",
    )
    overlap_swapped, total_swapped, segment_swapped = _send_request_get_overlap(
        frontend_port,
        frontend_proc,
        swapped_payload,
        "swapped_distinct_images_req1",
    )

    assert overlap_ordered_2 > overlap_ordered_1, (
        "Expected repeated identical order to increase overlap before swapped-order probe, "
        f"got req1={overlap_ordered_1}/{total_ordered_1}, req2={overlap_ordered_2}/{total_ordered_2}.\n"
        f"Recent frontend logs:\n{segment_ordered_2[-4000:]}"
    )
    assert abs(total_swapped - total_ordered_2) <= 4, (
        f"Expected swapped-order total blocks to stay near ordered baseline, "
        f"got swapped={total_swapped}, ordered={total_ordered_2}"
    )
    assert overlap_swapped <= 1, (
        "Expected near-zero overlap for swapped order of three distinct images "
        f"(allowing 1 shared text block), got {overlap_swapped}/{total_swapped}.\n"
        f"Recent frontend logs:\n{segment_swapped[-4000:]}"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_repeated_http_images(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
    http_image_server,
):
    """For repeated same 3-HTTP-image request: low first overlap, then increase, then stable."""
    frontend_port, frontend_proc = start_sglang_mm_services

    payload = _build_payload(
        http_image_server, prompt="MM routing e2e: repeated same 3 HTTP images."
    )
    overlap_1, total_1, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "http_3_images_req1"
    )
    time.sleep(1)
    overlap_2, total_2, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "http_3_images_req2"
    )
    time.sleep(1)
    overlap_3, total_3, segment_3 = _send_request_get_overlap(
        frontend_port, frontend_proc, payload, "http_3_images_req3"
    )

    assert overlap_1 <= 1, (
        f"Expected first overlap <=1, got req1={overlap_1}/{total_1}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_2 > overlap_1, (
        f"Expected second overlap > first, got req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    assert overlap_3 == overlap_2, (
        f"Expected third overlap == second, got req2={overlap_2}/{total_2}, req3={overlap_3}/{total_3}.\n"
        f"Recent frontend logs:\n{segment_3[-4000:]}"
    )
    low, high = THREE_IMAGE_TOTAL_BLOCKS_RANGE
    assert low <= total_3 <= high, (
        f"Unexpected total blocks for same 3 HTTP images (1024): "
        f"got {total_3}, expected in [{low}, {high}]"
    )


@pytest.mark.nightly
@pytest.mark.timeout(1800)
def test_sglang_mm_overlap_http_vs_data_uri_same_image(
    start_sglang_mm_services,
    predownload_models,
    set_ucx_tls_no_mm,
    http_image_server,
):
    """HTTP URL and data URI for the same image should produce identical KV cache hashes."""
    frontend_port, frontend_proc = start_sglang_mm_services

    color = _HTTP_IMAGE_COLORS[0]
    data_uri = _make_data_uri(color)
    http_url = http_image_server[0]

    data_uri_payload = _build_payload(
        [data_uri], prompt="MM routing e2e: HTTP vs data URI same image."
    )
    overlap_data, total_data, _ = _send_request_get_overlap(
        frontend_port, frontend_proc, data_uri_payload, "data_uri_seed"
    )

    time.sleep(1)

    http_payload = _build_payload(
        [http_url], prompt="MM routing e2e: HTTP vs data URI same image."
    )
    overlap_http, total_http, segment_http = _send_request_get_overlap(
        frontend_port, frontend_proc, http_payload, "http_probe"
    )

    assert total_http > 0, (
        f"No routing score for HTTP request.\n"
        f"Recent frontend logs:\n{segment_http[-4000:]}"
    )
    assert abs(total_http - total_data) <= 2, (
        f"Expected HTTP and data URI total blocks to match, "
        f"got http={total_http}, data_uri={total_data}.\n"
        f"Recent frontend logs:\n{segment_http[-4000:]}"
    )
    assert overlap_http > overlap_data, (
        f"Expected HTTP probe overlap > data URI seed overlap "
        f"(proving image cache hit, not just text overlap), "
        f"got http={overlap_http}/{total_http}, data_uri={overlap_data}/{total_data}.\n"
        f"Recent frontend logs:\n{segment_http[-4000:]}"
    )
