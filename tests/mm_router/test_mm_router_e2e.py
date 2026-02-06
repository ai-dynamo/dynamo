# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for multimodal KV routing with TRT-LLM.

Architecture:
  Frontend -> MM Router Worker -> TRT-LLM Worker
                (computes mm_hash)   (inference)

Tests verify overlap programmatically via KvIndexer.find_matches():
the test creates its own KvIndexer subscribed to TRT-LLM's NATS KV events,
computes mm-aware block hashes locally, and queries for matching blocks.

Requires GPU + Qwen2-VL-2B-Instruct. NATS/etcd started via fixtures.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import time
from io import BytesIO
from typing import Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

logger = logging.getLogger(__name__)

MM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MM_MODEL_TYPE = "qwen2_vl"
BLOCK_SIZE = 32
NAMESPACE = "test-mm"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.model(MM_MODEL_NAME),
]


# -- Test images (unique colors per test to avoid cross-test cache hits) ------

_TEST_IMAGES: dict[str, str] = {}
_COLOR_RGB: dict[str, tuple] = {
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "teal": (0, 128, 128),
    "maroon": (128, 0, 0),
}


def get_test_image(color: str) -> str:
    """Get or create a cached test image as base64 data URL."""
    if color not in _TEST_IMAGES:
        from PIL import Image

        img = Image.new("RGB", (1024, 1024), _COLOR_RGB[color])
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        _TEST_IMAGES[color] = f"data:image/png;base64,{b64}"
    return _TEST_IMAGES[color]


# -- Worker processes ---------------------------------------------------------


def _check_ready(response) -> bool:
    """Health check callback shared by TRT-LLM and MM Router workers."""
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


class TRTLLMWorkerProcess(ManagedProcess):
    """TRT-LLM Worker (backend inference)."""

    def __init__(self, request, *, system_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_REQUEST_PLANE"] = "nats"
        env["DYN_SYSTEM_PORT"] = str(system_port)

        log_dir = f"{request.node.name}_trtllm-worker"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.trtllm",
                "--model-path",
                MM_MODEL_NAME,
                "--served-model-name",
                f"{MM_MODEL_NAME}__internal",
                "--endpoint",
                f"dyn://{NAMESPACE}.trtllm.generate",
                "--modality",
                "multimodal",
                "--publish-events-and-metrics",
                "--kv-block-size",
                str(BLOCK_SIZE),
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=600,
            display_output=True,
            terminate_existing=False,
            straggler_commands=["-m dynamo.trtllm"],
            log_dir=log_dir,
        )


class MMRouterWorkerProcess(ManagedProcess):
    """MM Router Worker (computes mm_hash, routes requests)."""

    def __init__(self, request, *, system_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_REQUEST_PLANE"] = "nats"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)

        log_dir = f"{request.node.name}_mm-router-worker"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python3",
                "-m",
                "examples.backends.trtllm.mm_router_worker",
                "--model",
                MM_MODEL_NAME,
                "--model-type",
                MM_MODEL_TYPE,
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
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=120,
            display_output=True,
            terminate_existing=False,
            straggler_commands=["mm_router_worker"],
            log_dir=log_dir,
        )


class MMFrontendProcess(ManagedProcess):
    """Frontend HTTP ingress."""

    def __init__(self, request, *, frontend_port: int):
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_REQUEST_PLANE"] = "nats"

        log_dir = f"{request.node.name}_mm-frontend"
        shutil.rmtree(log_dir, ignore_errors=True)

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
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=120,
            display_output=True,
            terminate_existing=False,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=log_dir,
        )


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def mm_runtime_services(request):
    """Module-scoped NATS + etcd."""
    with NatsServer(request, port=0) as nats, EtcdServer(request, port=0) as etcd:
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats.port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd.port}"
        logger.info(f"NATS:{nats.port}, etcd:{etcd.port}")
        yield
        os.environ.pop("NATS_SERVER", None)
        os.environ.pop("ETCD_ENDPOINTS", None)


@pytest.fixture(scope="module")
def start_mm_services(request, mm_runtime_services) -> Generator[int, None, None]:
    """Start TRT-LLM Worker -> MM Router Worker -> Frontend. Yields frontend port."""
    ports = allocate_ports(count=3, start_port=10000)
    frontend_port, trtllm_port, mm_router_port = ports

    with TRTLLMWorkerProcess(request, system_port=trtllm_port):
        time.sleep(15)  # model loading
        with MMRouterWorkerProcess(request, system_port=mm_router_port):
            time.sleep(5)
            with MMFrontendProcess(request, frontend_port=frontend_port):
                logger.info(
                    f"All MM services ready at http://localhost:{frontend_port}"
                )
                yield frontend_port


@pytest.fixture(scope="module")
def mm_test_tools(mm_runtime_services, start_mm_services):
    """Module-scoped (indexer, tokenizer, processor) for overlap verification."""
    from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
    from transformers import AutoProcessor

    from dynamo._core import KvIndexer
    from dynamo.runtime import DistributedRuntime

    loop = asyncio.new_event_loop()
    runtime = DistributedRuntime(loop, "etcd", "nats")

    component = runtime.namespace(NAMESPACE).component("trtllm")
    indexer = KvIndexer(component, BLOCK_SIZE)
    time.sleep(2)  # wait for indexer to connect

    tokenizer = tokenizer_factory(MM_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MM_MODEL_NAME, trust_remote_code=True)

    yield indexer, tokenizer, processor

    runtime.shutdown()
    loop.close()


# -- Helpers ------------------------------------------------------------------


def _build_mm_messages(text: str, image_urls: list[str]) -> list[dict]:
    """Build OpenAI-format messages with images."""
    content = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _send_and_wait(messages: list[dict], frontend_port: int) -> str:
    """Send a chat request, validate response, wait for KV event propagation.

    Returns the response content text.
    """
    payload = {"model": MM_MODEL_NAME, "messages": messages, "max_tokens": 50}
    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "choices" in data, f"Response missing 'choices': {data}"
    content = data["choices"][0]["message"]["content"]
    assert content, "Empty response content"

    time.sleep(3)  # wait for KV events to propagate via NATS
    return content


def _compute_overlap(messages: list[dict], mm_test_tools) -> int:
    """Compute total block overlap for given messages against the test KvIndexer."""
    from dynamo._core import compute_block_hash_for_seq_py
    from examples.backends.trtllm.mm_router_worker.mm_processor import (
        build_block_mm_infos,
        extract_image_urls,
        process_multimodal,
    )

    indexer, tokenizer, processor = mm_test_tools
    image_urls = extract_image_urls(messages)

    processed = process_multimodal(
        messages=messages,
        image_urls=image_urls,
        tokenizer=tokenizer,
        processor=processor,
        model=MM_MODEL_NAME,
        model_type=MM_MODEL_TYPE,
    )

    block_mm_infos = build_block_mm_infos(
        num_tokens=len(processed.tokens),
        block_size=BLOCK_SIZE,
        mm_hashes=processed.mm_hashes,
        image_ranges=processed.image_ranges,
    )

    block_hashes = compute_block_hash_for_seq_py(
        processed.tokens, BLOCK_SIZE, block_mm_infos
    )

    logger.info(
        f"{len(block_hashes)} blocks, {len(image_urls)} images, "
        f"{len(processed.tokens)} tokens"
    )

    async def _query():
        return await indexer.find_matches(block_hashes)

    loop = asyncio.new_event_loop()
    try:
        scores = loop.run_until_complete(_query())
    finally:
        loop.close()

    total = sum(scores.scores.values())
    logger.info(f"Overlap: {scores.scores}, total={total}")
    return total


# -- Tests --------------------------------------------------------------------


@pytest.mark.timeout(600)
@pytest.mark.nightly
def test_same_single_image_overlap(
    start_mm_services, mm_test_tools, predownload_models
):
    """Same single image -> overlap > 0."""
    cyan = get_test_image("cyan")
    messages = _build_mm_messages("What color is this image?", [cyan])

    _send_and_wait(messages, start_mm_services)

    overlap = _compute_overlap(messages, mm_test_tools)
    assert overlap > 0, f"Expected overlap > 0, got {overlap}"


@pytest.mark.timeout(600)
@pytest.mark.nightly
def test_same_multiple_images_overlap(
    start_mm_services, mm_test_tools, predownload_models
):
    """Same two images -> overlap > 0."""
    magenta = get_test_image("magenta")
    yellow = get_test_image("yellow")
    messages = _build_mm_messages(
        "What colors are these two images?", [magenta, yellow]
    )

    _send_and_wait(messages, start_mm_services)

    overlap = _compute_overlap(messages, mm_test_tools)
    assert overlap > 0, f"Expected overlap > 0, got {overlap}"


@pytest.mark.timeout(600)
@pytest.mark.nightly
def test_different_images_no_overlap(
    start_mm_services, mm_test_tools, predownload_models
):
    """Different image -> overlap == 0."""
    orange = get_test_image("orange")
    pink = get_test_image("pink")

    warm_messages = _build_mm_messages("What color is this image?", [orange])
    _send_and_wait(warm_messages, start_mm_services)

    probe_messages = _build_mm_messages("What color is this image?", [pink])
    overlap = _compute_overlap(probe_messages, mm_test_tools)
    assert overlap == 0, f"Expected overlap == 0, got {overlap}"


@pytest.mark.timeout(600)
@pytest.mark.nightly
def test_swapped_image_order_less_overlap(
    start_mm_services, mm_test_tools, predownload_models
):
    """Swapped image order -> less overlap than identical order."""
    teal = get_test_image("teal")
    maroon = get_test_image("maroon")

    original = _build_mm_messages("What colors are these two images?", [teal, maroon])
    _send_and_wait(original, start_mm_services)

    overlap_same = _compute_overlap(original, mm_test_tools)
    assert overlap_same > 0, f"Expected overlap > 0, got {overlap_same}"

    swapped = _build_mm_messages("What colors are these two images?", [maroon, teal])
    overlap_swapped = _compute_overlap(swapped, mm_test_tools)
    assert (
        overlap_swapped < overlap_same
    ), f"Expected swapped ({overlap_swapped}) < same ({overlap_same})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
