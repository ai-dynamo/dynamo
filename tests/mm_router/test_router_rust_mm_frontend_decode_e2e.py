# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for MM-aware KV routing with frontend image decoding.

Architecture:
  Frontend (Rust preprocessor + lightseek + KV router + MediaLoader)
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

Kept in a separate module from test_router_rust_mm_router_e2e.py so its
module-scoped fixture (different worker, different DYN_NAMESPACE) does
not collide with the URL-passthrough fixture's registry entries.
"""

from __future__ import annotations

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
# Distinct namespace from test_router_rust_mm_router_e2e.py so the two
# modules' workers/frontends don't register against each other.
NAMESPACE = "router-rust-mm-fed"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.model(VLLM_MM_MODEL),
    pytest.mark.requested_vllm_kv_cache_bytes(1_719_075_000),
    pytest.mark.profiled_vram_gib(18.7),
]

_ROUTING_RECORD_PATTERN = re.compile(
    r"\[ROUTING\].*with\s*(\d+)/(\d+)\s*blocks overlap"
)


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
    "display_output": True,
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
            env=_make_process_env(DYN_SYSTEM_PORT=str(system_port)),
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


def _extract_routing_records(log_text: str) -> list[tuple[int, int]]:
    return [(int(o), int(t)) for o, t in _ROUTING_RECORD_PATTERN.findall(log_text)]


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
    overlap, total = _wait_for_new_routing_score(router_proc, pre_count)
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

        def log_message(self, format, *args):
            pass

    return _ImageHandler


@pytest.fixture(scope="module")
def http_image_server_with_alias() -> Generator[dict[str, str], None, None]:
    """Serve PNGs over HTTP. Two distinct paths (`/image_A.png` and
    `/image_A_alias.png`) return *byte-identical* content; the
    content-hash assertion below relies on this. Distinct paths defeat
    URL-string hashing — only content hashing collides them."""
    (port,) = allocate_ports(count=1, start_port=18600)
    primary_bytes = _make_png_bytes((180, 30, 90))
    image_map: dict[str, bytes] = {
        "/image_A.png": primary_bytes,
        "/image_A_alias.png": primary_bytes,  # byte-identical, different URL
    }
    server = HTTPServer(("127.0.0.1", port), _make_image_handler(image_map))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "primary": f"http://127.0.0.1:{port}/image_A.png",
            "alias": f"http://127.0.0.1:{port}/image_A_alias.png",
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
    `source=decoded_bytes` lightseek log line, proving we took the
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
