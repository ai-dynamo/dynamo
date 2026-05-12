# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for MM-aware KV routing with the Rust frontend (lightseek-mm).

Architecture:
  Frontend (Rust preprocessor + lightseek + KV router)
       └─ resolves <|image_pad|>, computes per-image N via lightseek,
          expands placeholders, hashes URL→u64→64-char hex,
          forwards mm_hashes via extra_args["mm_hashes"]
       → vLLM worker (publishes KV events with the forwarded UUID)

These tests assert that:
  1. Same-image repeats see >1 block of router-side overlap and high
     vLLM cached_tokens (the bit-exact MM cache hit).
  2. The data: URI path works equivalently to http URLs.
  3. Different images on the same prompt produce strictly less overlap
     than identical-image repeats.

The fallback path (model unsupported by lightseek or image-token
unresolvable) is unit-tested via `image_token::tests` and the
graceful-degrade branches in `gather_mm_exact_routing_info`; reproducing
it e2e would require a model whose loading is heavy and whose worker
support is fragile (e.g. Phi-4-multimodal-instruct + trust_remote_code),
so we cover that path at the Rust unit-test boundary.
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Any, Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

VLLM_MM_MODEL = os.getenv("DYN_TEST_VLLM_MM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
BLOCK_SIZE = 16
NAMESPACE = "router-rust-mm"

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

# Format produced by lib/llm/src/kv_router/push_router.rs's [ROUTING] log.
_ROUTING_RECORD_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(\d+)/(\d+)\s*blocks overlap"
)


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _make_process_env(
    log_level: str = (
        "info,mm_routing=debug,"
        "dynamo_kv_router::scheduling=debug,"
        # The [ROUTING] log line we parse below lives here at debug level.
        "dynamo_llm::kv_router=debug"
    ),
    **extra,
) -> dict[str, str]:
    env = os.environ.copy()
    env["DYN_LOG"] = log_level
    env["DYN_NAMESPACE"] = NAMESPACE
    env["DYN_REQUEST_PLANE"] = "tcp"
    env["DYN_MM_ALLOW_INTERNAL"] = "1"
    env.update(extra)
    return env


def _prepare_log_dir(request, suffix: str) -> str:
    return tempfile.mkdtemp(prefix=f"{request.node.name}_{suffix}_")


_COMMON_PROCESS_KWARGS: dict[str, Any] = {
    "display_output": True,
    "terminate_all_matching_process_names": False,
}


def _vllm_gpu_mem_args(default_utilization: str) -> list[str]:
    return build_gpu_mem_args("build_vllm_gpu_mem_args") or [
        "--gpu-memory-utilization",
        default_utilization,
    ]


class VLLMWorkerProcess(ManagedProcess):
    """vLLM backend that publishes KV events the router can consume."""

    def __init__(self, request, *, system_port: int, kv_event_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                VLLM_MM_MODEL,
                "--enable-multimodal",
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
            env=_make_process_env(DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_prepare_log_dir(request, "vllm-worker"),
            **_COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    """Rust frontend with lightseek-mm. No --dyn-chat-processor flag (default 'dynamo')."""

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
            log_dir=_prepare_log_dir(request, "router-rust-frontend"),
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
def start_router_rust_mm_services(
    request, mm_runtime_services
) -> Generator[tuple[int, ManagedProcess], None, None]:
    frontend_port, vllm_port, kv_event_port = allocate_ports(count=3, start_port=11000)
    with VLLMWorkerProcess(request, system_port=vllm_port, kv_event_port=kv_event_port):
        time.sleep(2)  # allow ZMQ publisher to bind
        with FrontendProcess(request, frontend_port=frontend_port) as frontend_proc:
            yield frontend_port, frontend_proc


def _make_png_bytes(color: tuple[int, int, int], size: int = 256) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_data_uri(color: tuple[int, int, int], size: int = 256) -> str:
    b64 = base64.b64encode(_make_png_bytes(color, size)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_payload(
    image_uris: list[str], prompt: str = "Describe this image."
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": VLLM_MM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    }


def _extract_routing_records(log_text: str) -> list[tuple[int, int]]:
    return [
        (int(overlap), int(total))
        for overlap, total in _ROUTING_RECORD_PATTERN.findall(log_text)
    ]


def _wait_for_new_routing_score(
    router_proc: ManagedProcess,
    pre_request_records: int,
    timeout_s: float = 25.0,
) -> tuple[int, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        records = _extract_routing_records(router_proc.read_logs())
        if len(records) >= pre_request_records + 1:
            return records[-1]
        time.sleep(1)
    return 0, 0


def _send(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> tuple[int, int, dict[str, Any]]:
    pre_count = len(_extract_routing_records(router_proc.read_logs()))
    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=240,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "choices" in data, f"missing choices in response: {data}"
    overlap, total = _wait_for_new_routing_score(router_proc, pre_count)
    print(
        f"[ROUTER_RUST_MM] {label}: overlap={overlap}/{total} usage={data.get('usage')}"
    )
    time.sleep(1)
    return overlap, total, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_router_rust_mm_repeated_single_image_overlap(
    start_router_rust_mm_services, predownload_models
):
    """Same image twice → second request shows full image-block overlap and
    high vLLM cached_tokens (proves bit-exact mm_hash forwarding works)."""
    frontend_port, router_proc = start_router_rust_mm_services
    image = _make_data_uri((201, 17, 99))
    payload = _build_payload([image])

    overlap_1, total_1, data_1 = _send(
        frontend_port, router_proc, payload, "single_req1"
    )
    overlap_2, total_2, data_2 = _send(
        frontend_port, router_proc, payload, "single_req2"
    )

    assert (
        total_1 > 1 and total_2 > 1
    ), f"expected non-trivial total blocks for MM request, got {total_1}, {total_2}"
    assert overlap_2 > overlap_1 + 1, (
        f"expected second-request overlap to dominate first by more than the "
        f"text-prefix block alone, got req1={overlap_1}/{total_1}, "
        f"req2={overlap_2}/{total_2}"
    )

    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    prompt_tokens_2 = data_2.get("usage", {}).get("prompt_tokens", 0)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2, (
        f"expected high vLLM cached_tokens on warm request, got "
        f"cached={cached_2}, prompt_tokens={prompt_tokens_2}"
    )


@pytest.mark.timeout(300)
def test_router_rust_mm_data_uri_works(
    start_router_rust_mm_services, predownload_models
):
    """data: URI path: mm_hash is computed from the URI bytes, dim parsed
    in-memory (no HTTP fetch). Same data URI repeated → high overlap."""
    frontend_port, router_proc = start_router_rust_mm_services
    uri = _make_data_uri((34, 210, 89))
    payload = _build_payload([uri])

    _, _, _ = _send(frontend_port, router_proc, payload, "data_uri_req1")
    overlap_2, total_2, data_2 = _send(
        frontend_port, router_proc, payload, "data_uri_req2"
    )

    assert (
        overlap_2 > 1 and total_2 > 1
    ), f"expected MM-aware overlap on data URI repeat, got {overlap_2}/{total_2}"
    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    assert (
        cached_2 is not None and cached_2 > 0
    ), f"expected cached_tokens > 0 on warm data URI request, got {cached_2}"


@pytest.mark.timeout(300)
def test_router_rust_mm_different_images_lower_overlap(
    start_router_rust_mm_services, predownload_models
):
    """Same prompt, different image: a warm-A repeat must show higher
    overlap than a cold first-time B request. With a single worker doing
    serial requests, this proves that a different mm_hash produces a
    different block-hash sequence (i.e. the image content actually
    contributes to the routing key, not just the text-prefix block)."""
    frontend_port, router_proc = start_router_rust_mm_services
    # Use a freshly-randomized image so we don't collide with the cache
    # populated by other tests in this module-scoped fixture.
    img_a = _make_data_uri((11, 22, 33))
    img_b = _make_data_uri((222, 111, 200))
    pa = _build_payload([img_a])
    pb = _build_payload([img_b])

    # Warm A on the worker.
    _send(frontend_port, router_proc, pa, "warm_A")
    # Repeat A: image-A blocks are cached → high overlap.
    overlap_a_repeat, total_a_repeat, _ = _send(
        frontend_port, router_proc, pa, "A_repeat"
    )
    # First-ever B: the worker has never seen img_b's mm_hash, so its
    # image blocks must NOT match anything cached. Only the text-prefix
    # block (no mm_hash mixed in) can possibly overlap.
    overlap_b_cold, total_b_cold, _ = _send(frontend_port, router_proc, pb, "B_cold")

    assert overlap_a_repeat > overlap_b_cold, (
        f"expected warm-A repeat overlap ({overlap_a_repeat}/{total_a_repeat}) "
        f"to exceed cold-B overlap ({overlap_b_cold}/{total_b_cold}). "
        f"If equal, the mm_hash isn't differentiating images — image-block "
        f"hashes are colliding across distinct content."
    )
    assert overlap_b_cold <= 1, (
        f"cold first-time image B should overlap at most the text-prefix block; "
        f"got {overlap_b_cold}/{total_b_cold}. Higher overlap means image "
        f"blocks aren't actually content-addressed."
    )


def _make_image_handler(image_map: dict[str, bytes]) -> type:
    """Stdlib HTTP handler that serves the given image bytes by path. Honors
    `Range: bytes=A-B` requests so we exercise the same code path our
    `fetch_image_dims` uses (header-only 64KB Range fetch)."""

    class _ImageHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            # Strip query string so the server resolves to the same bytes
            # regardless of any signed-URL params the caller appended.
            # The frontend's routing hash uses the full URL (different
            # query strings → different `mm_hash`), but the backend's
            # image fetch still serves the same bytes either way — which
            # is what we want for the distinct-query-strings test below.
            path = self.path.split("?", 1)[0]
            data = image_map.get(path)
            if data is None:
                self.send_error(404)
                return
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
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, format, *args):  # silence access logs
            pass

    return _ImageHandler


@pytest.fixture(scope="module")
def http_image_server() -> Generator[dict[str, str], None, None]:
    """Serve a small set of PNGs over HTTP for the duration of the module.

    Yields a dict mapping role names (e.g. "A", "B") to URLs. The Rust
    frontend's url-passthrough path will Range-fetch each one to read its
    `(W, H)` header. The server lives on 127.0.0.1 with `DYN_MM_ALLOW_INTERNAL=1`
    set in the frontend env so the loopback fetch is permitted.
    """
    (port,) = allocate_ports(count=1, start_port=18500)
    palette = {
        "A": (180, 30, 90),
        "B": (30, 180, 90),
    }
    image_map: dict[str, bytes] = {
        f"/image_{role}.png": _make_png_bytes(color) for role, color in palette.items()
    }
    server = HTTPServer(("127.0.0.1", port), _make_image_handler(image_map))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {role: f"http://127.0.0.1:{port}/image_{role}.png" for role in palette}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.timeout(300)
def test_router_rust_mm_repeated_http_image_overlap(
    start_router_rust_mm_services, predownload_models, http_image_server
):
    """HTTP URL path: same URL twice → MM-aware overlap on the warm
    request. Exercises our `fetch_image_dims` Range-fetch flow that
    only runs when `media_decoder` is null (the production vLLM-backed
    VLM config)."""
    frontend_port, router_proc = start_router_rust_mm_services
    payload = _build_payload([http_image_server["A"]])

    overlap_1, total_1, _ = _send(frontend_port, router_proc, payload, "http_req1")
    overlap_2, total_2, data_2 = _send(frontend_port, router_proc, payload, "http_req2")

    assert total_1 > 1 and total_2 > 1, (
        f"expected non-trivial total blocks for HTTP MM request, got "
        f"{total_1}, {total_2}"
    )
    assert overlap_2 > overlap_1 + 1, (
        f"expected warm-request overlap to dominate cold by more than just the "
        f"text-prefix block, got req1={overlap_1}/{total_1}, "
        f"req2={overlap_2}/{total_2} — if not, header-fetch + lightseek expansion "
        f"is misaligned with the worker's KV events"
    )
    cached_2 = (data_2.get("usage", {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    prompt_tokens_2 = data_2.get("usage", {}).get("prompt_tokens", 0)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2, (
        f"expected high vLLM cached_tokens on warm HTTP request, got "
        f"cached={cached_2}, prompt_tokens={prompt_tokens_2}"
    )


@pytest.mark.timeout(300)
def test_router_rust_mm_http_url_distinct_query_strings_dont_collide(
    start_router_rust_mm_services, predownload_models, http_image_server
):
    """URL-passthrough routing hashes the full URL — distinct query strings
    on the same image path produce distinct `mm_hash` values. We do NOT
    normalize cache-buster / signed-URL params because the URL alone can't
    tell us whether `?v=2` means "cache-busted same image" or "version 2 of
    a different image". Workloads with rotating signed URLs over a stable
    object should use `--frontend-decoding`, which hashes the decoded RGB
    bytes (see `test_router_rust_mm_frontend_decode_e2e.py`).

    Test: warm with `?v=1`, repeat with `?v=2` — second request must NOT
    show MM-block overlap beyond the text-prefix baseline (i.e. the system
    treats them as different images at the routing layer)."""
    frontend_port, router_proc = start_router_rust_mm_services
    base_url = http_image_server["B"]
    payload_v1 = _build_payload([f"{base_url}?v=1&sig=abc"])
    payload_v2 = _build_payload([f"{base_url}?v=2&sig=def"])

    overlap_1, total_1, _ = _send(
        frontend_port, router_proc, payload_v1, "distinct_url_req1"
    )
    overlap_2, total_2, _ = _send(
        frontend_port, router_proc, payload_v2, "distinct_url_req2"
    )

    # Both requests share the prompt's text prefix tokens but the image
    # blocks must differ — the second request's overlap should not exceed
    # the first by more than 1 block (the typical text-prefix baseline).
    assert overlap_2 <= overlap_1 + 1, (
        f"expected `?v=1` and `?v=2` to route as different images, but "
        f"req2 overlap jumped well above req1's text-prefix baseline. Got "
        f"req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}. If req2 is "
        f"much larger, the frontend is normalizing cache-busters (regression "
        f"to the strip-list behavior)."
    )


@pytest.mark.timeout(300)
def test_router_rust_mm_logs_lightseek_initialization(
    start_router_rust_mm_services, predownload_models
):
    """Smoke test: frontend must emit the lightseek init + image-token-resolved
    log lines for the served model. Catches regressions where the model dir
    isn't reachable from the MDC or the resolver silently misses a tier."""
    frontend_port, router_proc = start_router_rust_mm_services
    # Send any request to ensure the frontend has fully initialized.
    _send(
        frontend_port,
        router_proc,
        _build_payload([_make_data_uri((1, 2, 3))]),
        "smoke",
    )
    log_text = router_proc.read_logs()
    assert (
        "MM-aware KV routing enabled (lightseek)" in log_text
    ), "frontend should emit lightseek init log line on model registration"
    assert (
        "resolved image-placeholder token id" in log_text
    ), "image_token resolver should log which tier produced the hit"
