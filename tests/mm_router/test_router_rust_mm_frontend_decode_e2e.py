# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for MM-aware KV routing with frontend media decoding.

Architecture:
  Frontend (Rust preprocessor + KV router + MediaLoader)
       ├─ DOWNLOADS the image bytes (because the worker registered a
       │  `media_decoder` via `--frontend-decoding`)
       ├─ DECODES into RGB bytes via the in-process MediaDecoder
       ├─ HASHES the decoded bytes (xxh3 over RGB) → mm_hash
       ├─ Reads (W, H) from the decoded shape (no Range fetch)
       └─ Forwards mm_hashes via extra_args["mm_hashes"]
   → vLLM worker (publishes KV events with the forwarded UUID)

The headline behavior we exercise here, distinct from the URL-passthrough
path tested in test_router_rust_mm_router_e2e.py, is **content-addressed**
mm_hash: two distinct URLs serving the same image bytes must produce the
same `mm_hash` and therefore route the second request to the warm worker.
Without content hashing (e.g. if the code regressed to URL hashing in the
decoded branch), the second request would miss and overlap would be
text-prefix only.

The video case applies the same contract to sampled RGB frames plus the
model-visible temporal metadata and verifies grouped video hashes reach vLLM.

Kept in a separate module from test_router_rust_mm_router_e2e.py so its
module-scoped fixture (different worker, different DYN_NAMESPACE) does
not collide with the URL-passthrough fixture's registry entries.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports
from tests.utils.router_logs import (
    extract_router_kv_overlap_records,
    wait_for_router_kv_overlap,
)

VLLM_MM_MODEL = os.getenv("DYN_TEST_VLLM_MM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
VIDEO_NUM_FRAMES = int(os.getenv("DYN_TEST_MM_VIDEO_NUM_FRAMES", "4"))
BLOCK_SIZE = 16
# Distinct namespace from test_router_rust_mm_router_e2e.py so the two
# modules' workers/frontends don't register against each other.
NAMESPACE = "router-rust-mm-fed"

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.model(VLLM_MM_MODEL),
    pytest.mark.requested_vllm_kv_cache_bytes(1_719_075_000),
    pytest.mark.profiled_vram_gib(18.7),
]


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _make_process_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DYN_LOG"] = (
        "info,mm_routing=debug,"
        "dynamo_kv_router::scheduling=debug,"
        "dynamo_llm::kv_router=debug"
    )
    env["DYN_NAMESPACE"] = NAMESPACE
    env["DYN_REQUEST_PLANE"] = "tcp"
    env["DYN_MM_ALLOW_INTERNAL"] = "1"
    env.update(extra)
    return env


def _prepare_log_dir(request, suffix: str) -> str:
    return tempfile.mkdtemp(prefix=f"{request.node.name}_{suffix}_")


_COMMON_PROCESS_KWARGS: dict[str, Any] = {
    # Keep logs file-only; live tee can lag under GPU-parallel CI while tests poll files.
    "display_output": False,
    "terminate_all_matching_process_names": False,
}


def _vllm_gpu_mem_args(default_utilization: str) -> list[str]:
    return build_gpu_mem_args("build_vllm_gpu_mem_args") or [
        "--gpu-memory-utilization",
        default_utilization,
    ]


class VLLMWorkerFrontendDecodeProcess(ManagedProcess):
    """vLLM backend with `--frontend-decoding` so the model card carries a
    `media_decoder` and the frontend's MediaLoader runs in-process."""

    def __init__(self, request, *, system_port: int, kv_event_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                VLLM_MM_MODEL,
                "--enable-multimodal",
                "--frontend-decoding",
                "--block-size",
                str(BLOCK_SIZE),
                "--enforce-eager",
                *_vllm_gpu_mem_args("0.40"),
                "--max-model-len",
                "4096",
                "--kv-events-config",
                (
                    f'{{"publisher":"zmq","topic":"kv-events",'
                    f'"endpoint":"tcp://*:{kv_event_port}",'
                    f'"enable_kv_cache_events": true}}'
                ),
            ],
            env=_make_process_env(
                DYN_SYSTEM_PORT=str(system_port),
                DYN_MM_VIDEO_NUM_FRAMES=str(VIDEO_NUM_FRAMES),
            ),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_prepare_log_dir(request, "vllm-worker-fed"),
            **_COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int):
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
                "--model-name",
                VLLM_MM_MODEL,
            ],
            env=_make_process_env(),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "frontend-fed"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def mm_runtime_services(request):
    with (
        NatsServer(request, port=0) as nats,
        EtcdServer(request, port=0) as etcd,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setenv("NATS_SERVER", f"nats://localhost:{nats.port}")
        mp.setenv("ETCD_ENDPOINTS", f"http://localhost:{etcd.port}")
        yield


@pytest.fixture(scope="module")
def start_frontend_decode_services(
    request, mm_runtime_services
) -> Generator[tuple[int, ManagedProcess], None, None]:
    # start_port intentionally distinct from the URL-passthrough module's
    # range (11000s) so the two suites don't collide on local ports.
    frontend_port, vllm_port, kv_event_port = allocate_ports(count=3, start_port=11500)
    with VLLMWorkerFrontendDecodeProcess(
        request, system_port=vllm_port, kv_event_port=kv_event_port
    ):
        time.sleep(2)  # allow ZMQ publisher to bind
        with FrontendProcess(request, frontend_port=frontend_port) as fe:
            yield frontend_port, fe


def _make_png_bytes(color: tuple[int, int, int], size: int = 256) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_payload(image_uris: list[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": "Describe this image."}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": VLLM_MM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    }


def _build_video_payload(video_uri: str) -> dict[str, Any]:
    return {
        "model": VLLM_MM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": video_uri}},
                ],
            }
        ],
        "max_tokens": 4,
    }


def _build_mixed_payload(
    first_image_uri: str, video_uri: str, second_image_uri: str
) -> dict[str, Any]:
    return {
        "model": VLLM_MM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare the images with the video between them.",
                    },
                    {"type": "image_url", "image_url": {"url": first_image_uri}},
                    {"type": "video_url", "video_url": {"url": video_uri}},
                    {"type": "image_url", "image_url": {"url": second_image_uri}},
                ],
            }
        ],
        "max_tokens": 4,
    }


def _send(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> tuple[int, int, dict[str, Any]]:
    pre_request_logs = router_proc.read_logs()
    start_offset = len(pre_request_logs)
    pre_request_record_count = len(extract_router_kv_overlap_records(pre_request_logs))
    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=240,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    overlap, total, _recent_logs = wait_for_router_kv_overlap(
        router_proc.read_logs,
        start_offset=start_offset,
        pre_request_record_count=pre_request_record_count,
        context=label,
        log_label="frontend",
    )
    print(f"[FED] {label}: overlap={overlap}/{total} usage={data.get('usage')}")
    time.sleep(1)
    return overlap, total, data


def _make_image_handler(image_map: dict[str, bytes]) -> type:
    class _ImageHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            data = image_map.get(path)
            if data is None:
                self.send_error(404)
                return
            content_type = "video/mp4" if path.endswith(".mp4") else "image/png"
            range_hdr = self.headers.get("Range", "")
            if range_hdr.startswith("bytes="):
                spec = range_hdr[len("bytes=") :]
                lo_s, _, hi_s = spec.partition("-")
                try:
                    lo = int(lo_s) if lo_s else 0
                    hi = int(hi_s) if hi_s else len(data) - 1
                except ValueError:
                    self.send_error(416)
                    return
                hi = min(hi, len(data) - 1)
                lo = max(lo, 0)
                if lo > hi:
                    self.send_error(416)
                    return
                chunk = data[lo : hi + 1]
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    return _ImageHandler


@pytest.fixture(scope="module")
def http_image_server_with_alias() -> Generator[dict[str, str], None, None]:
    """Serve PNGs over HTTP. Two distinct paths (`/image_A.png` and
    `/image_A_alias.png`) return *byte-identical* content; the
    content-hash assertion below relies on this. Distinct paths defeat
    URL-string hashing — only content hashing collides them. `/image_B.png`
    has distinct content for mixed-media ordering coverage."""
    (port,) = allocate_ports(count=1, start_port=18600)
    primary_bytes = _make_png_bytes((180, 30, 90))
    image_map: dict[str, bytes] = {
        "/image_A.png": primary_bytes,
        "/image_A_alias.png": primary_bytes,  # byte-identical, different URL
        "/image_B.png": _make_png_bytes((20, 140, 210)),
    }
    server = HTTPServer(("127.0.0.1", port), _make_image_handler(image_map))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "primary": f"http://127.0.0.1:{port}/image_A.png",
            "alias": f"http://127.0.0.1:{port}/image_A_alias.png",
            "secondary": f"http://127.0.0.1:{port}/image_B.png",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def http_video_server_with_alias() -> Generator[dict[str, str], None, None]:
    (port,) = allocate_ports(count=1, start_port=18700)
    video_bytes = (
        Path(__file__).resolve().parents[2] / "lib/llm/tests/data/media/240p_100.mp4"
    ).read_bytes()
    media_map = {
        "/video_A.mp4": video_bytes,
        "/video_A_alias.mp4": video_bytes,
    }
    server = HTTPServer(("127.0.0.1", port), _make_image_handler(media_map))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "primary": f"http://127.0.0.1:{port}/video_A.mp4",
            "alias": f"http://127.0.0.1:{port}/video_A_alias.mp4",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.timeout(300)
def test_frontend_decode_content_hash_collides_across_urls(
    start_frontend_decode_services,
    predownload_models,
    http_image_server_with_alias,
):
    """The headline frontend-decode contract: two distinct URLs serving
    the same image bytes must produce the same `mm_hash` (content-addressed
    via xxh3 over decoded RGB), so the second request hits the warm worker.

    If the decoded branch ever regresses to URL-string hashing, this test
    will see only text-prefix overlap (overlap_2 ≈ 1 like overlap_1) and
    fail. That is exactly the regression we want this gate to catch.
    """
    frontend_port, router_proc = start_frontend_decode_services
    primary_url = http_image_server_with_alias["primary"]
    alias_url = http_image_server_with_alias["alias"]

    overlap_1, total_1, _ = _send(
        frontend_port, router_proc, _build_payload([primary_url]), "fed_primary"
    )
    overlap_2, total_2, data_2 = _send(
        frontend_port, router_proc, _build_payload([alias_url]), "fed_alias"
    )

    assert total_1 > 1 and total_2 > 1, (
        f"expected non-trivial total blocks for MM request, got "
        f"{total_1}, {total_2}"
    )
    assert overlap_2 > overlap_1 + 1, (
        f"frontend-decode mm_hash must be content-addressed: distinct URLs "
        f"serving the same image bytes should collide on the routing key. "
        f"Got primary={overlap_1}/{total_1}, alias={overlap_2}/{total_2}. "
        f"If overlap_2 ≈ overlap_1, the decoded branch is hashing the URL "
        f"instead of the bytes."
    )

    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    prompt_tokens_2 = data_2.get("usage", {}).get("prompt_tokens", 0)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2, (
        f"expected high vLLM cached_tokens on warm alias request "
        f"(content hash should match the worker's KV cache key), got "
        f"cached={cached_2}, prompt_tokens={prompt_tokens_2}"
    )


@pytest.mark.timeout(300)
def test_frontend_decode_logs_decoded_bytes_source(
    start_frontend_decode_services,
    predownload_models,
    http_image_server_with_alias,
):
    """Smoke check: the frontend-decode path must emit the
    `source=decoded_bytes` MM-routing log line, proving we took the
    content-hash branch and not the url_fallback branch."""
    frontend_port, router_proc = start_frontend_decode_services
    _send(
        frontend_port,
        router_proc,
        _build_payload([http_image_server_with_alias["primary"]]),
        "fed_smoke",
    )
    # tracing's pretty/compact format wraps each field in ANSI escapes
    # (`[3msource[0m[2m=[0m"decoded_bytes"`) so a literal `source="decoded_bytes"`
    # substring search misses. The bareword "decoded_bytes" is unique to our
    # `hash_source = "decoded_bytes"` literal, so checking for it is enough.
    log_text = router_proc.read_logs()
    assert "decoded_bytes" in log_text, (
        "frontend-decode path should hash decoded bytes; missing the "
        "`decoded_bytes` source tag means the code took either the "
        "`url_fallback` branch (descriptor lost source_storage) or the "
        "URL-passthrough branch (media_loader was None)."
    )


@pytest.mark.timeout(300)
def test_frontend_decoded_video_routes_by_sampled_content(
    start_frontend_decode_services,
    predownload_models,
    http_video_server_with_alias,
):
    frontend_port, router_proc = start_frontend_decode_services
    overlap_1, total_1, _ = _send(
        frontend_port,
        router_proc,
        _build_video_payload(http_video_server_with_alias["primary"]),
        "fed_video_primary",
    )
    overlap_2, total_2, data_2 = _send(
        frontend_port,
        router_proc,
        _build_video_payload(http_video_server_with_alias["alias"]),
        "fed_video_alias",
    )

    assert total_1 > 1 and total_2 > 1
    assert overlap_2 > overlap_1, (
        "frontend-decoded aliases of the same video should share the sampled "
        f"video routing key, got {overlap_1}/{total_1} then {overlap_2}/{total_2}"
    )
    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    prompt_tokens_2 = data_2.get("usage", {}).get("prompt_tokens", 0)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2
    assert "video routing metadata resolved" in router_proc.read_logs()


@pytest.mark.timeout(300)
def test_frontend_decoded_mixed_media_preserves_i_v_i_order(
    start_frontend_decode_services,
    predownload_models,
    http_image_server_with_alias,
    http_video_server_with_alias,
):
    frontend_port, router_proc = start_frontend_decode_services
    image_1 = http_image_server_with_alias["primary"]
    image_2 = http_image_server_with_alias["secondary"]
    video = http_video_server_with_alias["primary"]

    payload = _build_mixed_payload(image_1, video, image_2)
    overlap_1, total_1, _ = _send(
        frontend_port, router_proc, payload, "fed_mixed_i1_v_i2"
    )
    overlap_2, total_2, data_2 = _send(
        frontend_port, router_proc, payload, "fed_mixed_i1_v_i2_repeat"
    )
    swapped_overlap, swapped_total, _ = _send(
        frontend_port,
        router_proc,
        _build_mixed_payload(image_2, video, image_1),
        "fed_mixed_i2_v_i1",
    )

    assert total_1 > 1 and total_2 > 1 and swapped_total > 1
    assert overlap_2 > overlap_1 and overlap_2 >= total_2 - 1, (
        "an exact I-V-I repeat should reuse every complete mixed-media block, got "
        f"{overlap_1}/{total_1} then {overlap_2}/{total_2}"
    )
    assert swapped_overlap < overlap_2, (
        "swapping two distinct images around the same video must change the "
        f"routing prefix, got repeat={overlap_2}/{total_2}, "
        f"swapped={swapped_overlap}/{swapped_total}"
    )

    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    prompt_tokens_2 = data_2.get("usage", {}).get("prompt_tokens", 0)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2
